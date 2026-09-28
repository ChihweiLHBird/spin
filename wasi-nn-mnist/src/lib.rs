//! End-to-end `wasi:nn` demo for Spin: classify a handwritten digit.
//!
//! `POST /classify` accepts a PNG or JPEG image of any size, 784 raw grayscale
//! bytes, or a 3136-byte little-endian f32 tensor, runs the ONNX model-zoo
//! MNIST-12 network through `wasi:nn`, and answers with JSON.

mod nn {
    wit_bindgen::generate!({
        world: "mnist",
        path: "wit",
        generate_all,
    });
}

use std::time::Instant;

use image::imageops::FilterType;
use nn::wasi::nn::errors::Error as NnError;
use nn::wasi::nn::graph::{self, ExecutionTarget, GraphEncoding};
use nn::wasi::nn::tensor::{Tensor, TensorType};
use spin_sdk::http::body::IncomingBodyExt;
use spin_sdk::http::{HeaderMap, HeaderValue, IntoResponse, Method, Request, StatusCode};
use spin_sdk::http_service;

/// MNIST-12 from the ONNX model zoo (MIT). Input `Input3` is f32 [1,1,28,28]
/// with white digits on black scaled to 0..1; output `Plus214_Output_0` is
/// f32 [1,10] of pre-softmax scores.
const MODEL: &[u8] = include_bytes!("../model/mnist-12.onnx");
const INPUT_NAME: &str = "Input3";
const SIDE: u32 = 28;
const PIXELS: usize = (SIDE * SIDE) as usize;
const CLASSES: usize = 10;

const USAGE: &str = "wasi-nn MNIST demo

POST /classify            body: PNG or JPEG image (any size), or 784 raw grayscale bytes (28x28),
                          or 3136 raw little-endian f32 bytes (the [1,1,28,28] tensor, used as-is)
      ?invert=auto|true|false   MNIST expects a white digit on black; auto (default) inverts
                                bright images such as black-on-white drawings
GET  /                    this text

  curl -X POST --data-binary @samples/digit-7-idx0.png http://127.0.0.1:3000/classify
  curl -X POST --data-binary @samples/mnist_5.jpg      http://127.0.0.1:3000/classify

The model is loaded through wasi:nn on every request (Phase 1 has no host cache);
the JSON reports graph_load and compute times separately so that cost is visible.
";

enum AppError {
    BadInput(String),
    Nn(String),
}

type Reply = (StatusCode, HeaderMap, String);

#[http_service]
async fn handle(req: Request) -> impl IntoResponse {
    let (parts, body) = req.into_parts();
    let path = parts.uri.path();

    if parts.method == Method::GET && (path == "/" || path.is_empty()) {
        return text(StatusCode::OK, USAGE.to_string());
    }
    if parts.method == Method::POST && path == "/classify" {
        let invert = query_param(parts.uri.query(), "invert").map(str::to_owned);
        let bytes = match body.bytes().await {
            Ok(bytes) => bytes,
            Err(e) => {
                return text(
                    StatusCode::BAD_REQUEST,
                    format!("could not read request body: {e:?}\n"),
                );
            }
        };
        return match classify(&bytes, invert.as_deref()) {
            Ok(json) => respond(StatusCode::OK, "application/json", json),
            Err(AppError::BadInput(msg)) => text(StatusCode::BAD_REQUEST, format!("{msg}\n")),
            Err(AppError::Nn(msg)) => text(StatusCode::INTERNAL_SERVER_ERROR, format!("{msg}\n")),
        };
    }
    text(
        StatusCode::NOT_FOUND,
        "not found; GET / for usage\n".to_string(),
    )
}

fn respond(status: StatusCode, content_type: &'static str, body: String) -> Reply {
    let mut headers = HeaderMap::new();
    headers.insert("content-type", HeaderValue::from_static(content_type));
    (status, headers, body)
}

fn text(status: StatusCode, body: String) -> Reply {
    respond(status, "text/plain; charset=utf-8", body)
}

fn query_param<'a>(query: Option<&'a str>, key: &str) -> Option<&'a str> {
    query?
        .split('&')
        .filter_map(|pair| pair.split_once('=').or(Some((pair, ""))))
        .find(|(k, _)| *k == key)
        .map(|(_, v)| v)
}

fn classify(bytes: &[u8], invert: Option<&str>) -> Result<String, AppError> {
    let (input, kind, inverted) = decode_input(bytes, invert)?;
    let run = infer(&input)?;
    let probabilities = softmax(&run.logits);
    let (digit, confidence) =
        probabilities
            .iter()
            .enumerate()
            .fold(
                (0, f32::MIN),
                |best, (i, &p)| if p > best.1 { (i, p) } else { best },
            );
    let list = |values: &[f32], places: usize| {
        values
            .iter()
            .map(|v| format!("{v:.places$}"))
            .collect::<Vec<_>>()
            .join(",")
    };
    Ok(format!(
        "{{\"digit\":{digit},\"confidence\":{confidence:.4},\"probabilities\":[{}],\"logits\":[{}],\"input\":\"{kind}\",\"inverted\":{inverted},\"timings_ms\":{{\"graph_load\":{:.2},\"compute\":{:.2}}}}}\n",
        list(&probabilities, 4),
        list(&run.logits, 3),
        run.load_ms,
        run.compute_ms,
    ))
}

/// Turn the request body into 784 f32 pixels in MNIST's convention.
fn decode_input(bytes: &[u8], invert: Option<&str>) -> Result<(Vec<f32>, String, bool), AppError> {
    // A ready-made tensor is passed through untouched.
    if bytes.len() == PIXELS * 4 {
        let tensor = bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        return Ok((tensor, "f32-tensor".to_string(), false));
    }

    let (gray, kind) = if bytes.len() == PIXELS {
        (bytes.to_vec(), "u8-gray-28x28".to_string())
    } else {
        let img = image::load_from_memory(bytes).map_err(|e| {
            AppError::BadInput(format!(
                "body is not a PNG/JPEG image, 784 grayscale bytes, or 3136 f32 bytes ({} bytes received): {e}",
                bytes.len()
            ))
        })?;
        let (w, h) = (img.width(), img.height());
        let small = img
            .resize_exact(SIDE, SIDE, FilterType::Triangle)
            .to_luma8();
        (small.into_raw(), format!("image-{w}x{h}"))
    };

    let mut pixels: Vec<f32> = gray.iter().map(|&p| p as f32 / 255.0).collect();
    let mean = pixels.iter().sum::<f32>() / pixels.len() as f32;
    let inverted = match invert.map(str::to_ascii_lowercase).as_deref() {
        None | Some("auto") => mean > 0.5,
        Some("true" | "1" | "yes") => true,
        Some("false" | "0" | "no") => false,
        Some(other) => {
            return Err(AppError::BadInput(format!(
                "invert={other}: expected auto, true, or false"
            )));
        }
    };
    if inverted {
        for p in &mut pixels {
            *p = 1.0 - *p;
        }
    }
    Ok((pixels, kind, inverted))
}

struct Inference {
    logits: [f32; CLASSES],
    load_ms: f64,
    compute_ms: f64,
}

/// The whole wasi:nn conversation: load the graph, create an execution
/// context, run one named input tensor through it, read the output tensor.
fn infer(input: &[f32]) -> Result<Inference, AppError> {
    let started = Instant::now();
    let graph = graph::load(&[MODEL.to_vec()], GraphEncoding::Onnx, ExecutionTarget::Cpu)
        .map_err(nn_error("graph::load"))?;
    let context = graph
        .init_execution_context()
        .map_err(nn_error("init-execution-context"))?;
    let load_ms = millis(started);

    let data: Vec<u8> = input.iter().flat_map(|v| v.to_le_bytes()).collect();
    let tensor = Tensor::new(&[1, 1, SIDE, SIDE], TensorType::Fp32, &data);

    let started = Instant::now();
    let outputs = context
        .compute(vec![(INPUT_NAME.to_string(), tensor)])
        .map_err(nn_error("compute"))?;
    let compute_ms = millis(started);

    let (name, output) = outputs
        .into_iter()
        .next()
        .ok_or_else(|| AppError::Nn("compute returned no output tensors".to_string()))?;
    let bytes = output.data();
    if bytes.len() != CLASSES * 4 {
        return Err(AppError::Nn(format!(
            "unexpected output tensor {name}: {} bytes, dimensions {:?}",
            bytes.len(),
            output.dimensions()
        )));
    }
    let mut logits = [0f32; CLASSES];
    for (slot, chunk) in logits.iter_mut().zip(bytes.chunks_exact(4)) {
        *slot = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
    }
    Ok(Inference {
        logits,
        load_ms,
        compute_ms,
    })
}

fn nn_error(stage: &'static str) -> impl Fn(NnError) -> AppError {
    move |e| {
        AppError::Nn(format!(
            "wasi:nn {stage} failed: {:?}: {}",
            e.code(),
            e.data()
        ))
    }
}

fn millis(since: Instant) -> f64 {
    since.elapsed().as_secs_f64() * 1000.0
}

fn softmax(logits: &[f32; CLASSES]) -> [f32; CLASSES] {
    let max = logits.iter().cloned().fold(f32::MIN, f32::max);
    let mut out = [0f32; CLASSES];
    let mut sum = 0f32;
    for (o, &l) in out.iter_mut().zip(logits) {
        *o = (l - max).exp();
        sum += *o;
    }
    for o in &mut out {
        *o /= sum;
    }
    out
}
