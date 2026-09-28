# wasi-nn MNIST demo

A Spin HTTP component that classifies handwritten digits with the ONNX model-zoo [MNIST-12](https://github.com/onnx/models/tree/main/validated/vision/classification/mnist) network, running on the host's ONNX Runtime through Spin's `wasi:nn` support (Phase 1 factor).

Build the guest with `cargo build --target wasm32-wasip2 --release` in this directory, then run it with a Spin built with the ONNX backend: `cargo run --features wasi-nn-onnx -- up -f wasi-nn-mnist/spin.toml` from the repository root.

`POST /classify` takes a PNG or JPEG of any size, 784 raw grayscale bytes, or a 3136-byte f32 tensor, and returns JSON with the predicted digit, softmax confidence, all ten scores, and the time spent in `graph::load` versus `compute`. `GET /` prints usage. `./test.sh` posts every file in `samples/` and checks the answers.

`samples/` holds one real MNIST test image per digit (`digit-<label>-idx<n>.png`, plus 224×224 upscales and raw `.u8` copies), `mnist_5.jpg` from the `ort` crate's test suite, and `zoo-test0.f32le`, the model zoo's own test tensor with its expected output in `zoo-test0.expected.json`. The model file is MIT licensed; MNIST images are from the public test set.
