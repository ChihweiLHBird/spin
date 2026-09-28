mod nn {
    wit_bindgen::generate!({
        world: "smoke",
        path: "wit",
        generate_all,
    });
}

use nn::wasi::nn::errors::ErrorCode;
use nn::wasi::nn::graph::{self, ExecutionTarget, GraphEncoding};
use spin_sdk::http::{IntoResponse, Request, StatusCode};
use spin_sdk::http_service;

/// Smoke guest from the Phase 1 tutorial: `graph::load` of junk bytes should
/// fail inside the ONNX backend, which shows the import was linked.
#[http_service]
async fn infer(_req: Request) -> impl IntoResponse {
    let junk = b"not-an-onnx-model".to_vec();
    match graph::load(&[junk], GraphEncoding::Onnx, ExecutionTarget::Cpu) {
        Ok(_graph) => (
            StatusCode::OK,
            "graph::load succeeded; expected the backend to reject junk bytes".to_string(),
        ),
        Err(error) => {
            let code = error.code();
            let data = error.data();
            let kind = match code {
                ErrorCode::RuntimeError | ErrorCode::InvalidEncoding => "runtime",
                _ => "unexpected",
            };
            (
                StatusCode::OK,
                format!("wasi:nn graph::load returned a {kind} error ({code:?}): {data}"),
            )
        }
    }
}
