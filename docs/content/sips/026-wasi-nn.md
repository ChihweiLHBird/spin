title = "SIP 026 - wasi-nn Support"
template = "main"
date = "2026-09-05T00:00:00Z"
---

Summary: Add support for the standard `wasi:nn` machine-learning inference interface to Spin, backed by the Bytecode Alliance's `wasmtime-wasi-nn` crate, with host-managed named models declared per component in the manifest, an opt-in per-component capability for guest-supplied model bytes, operator configuration in runtime config, build-time validation against target environments, and the ONNX Runtime backend included in the default build.

Owner(s): [zhiwei.liang@zliang.me](mailto:zhiwei.liang@zliang.me)

Created: September 5, 2026

## Background

[wasi-nn](https://github.com/WebAssembly/wasi-nn) is the WASI proposal for machine-learning inference. A guest loads a model as a *graph*, creates an *execution context*, feeds *tensors* in, and reads tensors back; the host runs the model on a native backend such as ONNX Runtime, OpenVINO, or PyTorch, with whatever hardware acceleration the host has. The proposal is at WASI Phase 2 and its current interface is the `wasi:nn@0.2.0-rc-2024-10-28` snapshot. The specification repository has been quiet since that snapshot, but implementations are active: `wasmtime-wasi-nn` ships with every Wasmtime release, and WasmEdge's WASI-NN plugin underpins LlamaEdge.

Spin has no `wasi:nn` support today. Its only inference interface is the proprietary `fermyon:spin/llm@2.0.0` ([SIP 015](015-llm.md)), which exposes text generation and embeddings at a high level and is backed by Candle locally or by a remote HTTP service. That interface cannot run an arbitrary vision or tabular model, cannot use ONNX Runtime or OpenVINO, and is itself being evaluated for a rewrite. A component written against `wasi:nn` runs unchanged on Wasmtime, WasmEdge, and WAMR; Spin has no `wasi:nn` implementation, so such a component fails to instantiate.

`wasi:nn` offers two ways to obtain a graph. `graph::load(builder: list<list<u8>>, encoding, target)` takes the model bytes from the guest. `graph::load-by-name(name: string)` asks the host for a model it already has; the proposal leaves how names are registered "implementation-specific". Every wasi-nn host we surveyed (Wasmtime's CLI, WasmEdge, WAMR) exposes both paths unconditionally, because they are single-tenant runtimes. Spin runs components that do not trust each other, so this SIP treats the two paths differently.

## Proposal

### Interface and versioning

Vendor `wasi:nn@0.2.0-rc-2024-10-28` under `wit/deps/nn@0.2.0-rc-2024-10-28/` and import it into the `platform` world of `wit/world.wit` behind `@unstable(feature = wasi-nn)`, as `wasi:otel` is today. The gate reflects the WIT's release-candidate status and can be lifted when the proposal publishes a stable version. The host provides the interface unconditionally; there is no CLI flag, because the manifest keys below already make every capability opt-in per component.

### Host implementation

A new factor crate, `spin-factor-wasi-nn`, wraps `wasmtime-wasi-nn`: it links the crate's host implementation into the linker and builds a per-instance `WasiNnCtx` from Spin state. It is registered in `TriggerFactors` like every other factor. `wasmtime-wasi-nn` is released in lockstep with Wasmtime and pins the exact matching `wasmtime` version, so it moves with Spin's regular Wasmtime updates.

### Manifest: two new component keys

```toml
[component.classifier]
source = "classifier.wasm"
wasi_nn_models = ["resnet50"]      # host-managed models this component may load by name
wasi_nn_encodings = ["onnx"]       # optional: this component may also load its own ONNX bytes
```

`wasi_nn_models` lists the host-managed models the component may obtain through `load-by-name`. `wasi_nn_encodings` lists the graph encodings the component may pass to `load` together with its own bytes. Both default to empty, so a component that declares neither can still instantiate with the `wasi:nn` import, but every load fails. This is the same deny-unless-declared rule that `allowed_outbound_hosts`, `key_value_stores`, and `sqlite_databases` follow.

The keys are wasi-nn's own and deliberately do not reuse the LLM factor's `ai_models`: that factor is under evaluation for a rewrite, and coupling the two would constrain both. The `wasi_nn_` prefix names the interface rather than the concept, unlike its concept-named neighbours, because the concept name is already taken.

### Runtime configuration: the operator maps names to models

```toml
[wasi_nn_model.resnet50]
type = "onnx"                     # onnx | openvino | pytorch
path = "/models/resnet50.onnx"
target = "cpu"                    # cpu | gpu | tpu
```

Each `[wasi_nn_model.<name>]` table binds a name to a backend, a model file, and an execution target, following the `[key_value_store.<label>]` and `[sqlite_database.<label>]` shape. The guest only ever asks for `"resnet50"`; the operator decides ONNX on CPU on one host and OpenVINO on GPU on another without rebuilding the component. Optional per-model fields `preload` and `pool_size` are described under performance. An optional `[wasi_nn] allow_guest_models = false` table gives operators a hard override of `wasi_nn_encodings` on hosts that do not use a target-environment catalogue.

### Enforcement

Enforcement follows the three layers Spin already uses for other capabilities.

1. **At runtime, in the factor.** `configure_app` loads each configured model once per application and records each component's `wasi_nn_models` and `wasi_nn_encodings`. `prepare` gives each instance a registry scoped to its allowed names and installs backends only for its allowed encodings. `load-by-name` of an unknown or undeclared name returns `not-found`; `load` with an undeclared encoding returns `invalid-encoding`. Returning `not-found` for both unknown and undeclared names means a component cannot probe the host's catalogue.
2. **At build time, in target-environment validation.** `ConfigurationConstraints` in `crates/environments` gains `wasi_nn_models` and `wasi_nn_encodings`, checked the way `key_value_stores` is today (each declared value must appear in the environment's permitted list), so a hosted environment can refuse components that need models it does not offer, refuse guest-supplied bytes altogether, or allow `onnx` while refusing `pytorch`.
3. **At composition time, in the capabilities crate.** A new `wasi_nn_models` capability set covering the four `wasi:nn` interfaces lets a parent component decide whether composed dependencies inherit `wasi:nn`, and the deny adapter is rebuilt accordingly. The adapter works per interface and cannot distinguish `load` from `load-by-name`, so the encoding check remains a runtime check.

### Why guest-supplied model bytes are opt-in

Raw `load` is wasi-nn's original and primary API, and the proposal permits a host both to offer it and to refuse it: `load` returns `result<graph, error>`, and the error codes include `security` and `unsupported-operation`. Spin gates it per component for four reasons. The bytes are parsed by native backends outside the Wasm sandbox, and a parser bug there is a sandbox escape; according to maintainer comments on the wasi-nn and Wasmtime trackers ([wasi-nn#88](https://github.com/WebAssembly/wasi-nn/issues/88), [wasmtime#3977](https://github.com/bytecodealliance/wasmtime/pull/3977)), this is why TensorFlow support was never merged into `wasmtime-wasi-nn`. A native inference cannot be interrupted by Spin's epoch mechanism. A loaded session consumes host memory and possibly a GPU outside the instance's limits. Some formats reference external files by path, so each backend's handling of in-memory models needs review before untrusted guests get the path.

This departs from the wasi-nn hosts we surveyed on purpose. Wasmtime's CLI, WasmEdge, and WAMR all allow raw `load` whenever wasi-nn is enabled and offer no named-only mode; none of them is a multi-tenant host, and none records a policy decision. The shared inference platforms surveyed, including Spin's own LLM factor, expose host-named base models only; the closest exception, Cloudflare Workers AI, accepts only size-capped LoRA adapters on top of catalogue models.

### Performance and model lifecycle

Models are loaded once per application in `configure_app`, not per request. A loaded `wasmtime-wasi-nn` graph is reference-counted and shared across instances, so the per-request cost is a clone of a handle, not a parse of a model file. One property of the ONNX backend matters for capacity planning: it wraps the ONNX Runtime session in a mutex held for the whole `compute`, so a shared graph admits one inference at a time; OpenVINO does not have this limit. The per-model `pool_size` field, default 1, loads that many independent sessions to trade memory for concurrency. A later `preload = false` may defer loading to first use.

`wasmtime-wasi-nn` exposes synchronous host functions, so each in-flight `load` or `compute` occupies a Tokio worker thread for its duration. This is accepted for the initial implementation; an alternative in which Spin generates its own asynchronous bindings and runs backend work on the blocking pool is recorded under future work.

### Build and distribution

The ONNX Runtime backend is part of the default feature set of the `spin` binary, as the `llm` feature is today; OpenVINO, PyTorch, and CUDA stay behind opt-in features. `ort`, the crate behind the backend, downloads a hash-verified prebuilt ONNX Runtime archive once per machine at build time, or uses a system copy found through `pkg-config` or `ORT_LIB_LOCATION`, and links it statically, so no shared library ships alongside `spin`. Prebuilt archives exist for all five main release targets but not for musl, so the `build-spin-static` jobs must build with `--no-default-features --features llm,cpu-time-metrics` or supply a from-source ONNX Runtime. The binary-size increase will be measured and reported with the first implementation PR.

### Rollout

1. Factor, WIT vendoring, `TriggerFactors` registration, and both manifest keys with their runtime and build-time checks. Nothing ships without the declared-capability gate.
2. Application-scoped model registry and `[wasi_nn_model.<name>]` runtime configuration, including `pool_size`.
3. Capabilities-crate set and deny-adapter rebuild.

## Future work

- Spin-owned asynchronous bindings for `wasi:nn`, running backend work on the blocking pool, if worker blocking or error clarity becomes a problem in practice.
- Lazy loading and eviction of models, for deployments that declare more models than fit in memory.
- Treating `execution-target = gpu` as a declared capability, if accelerator time on shared hosts needs governance.
- Following the wasi-nn proposal if it is rebased onto WASI 0.3 and gains an asynchronous `compute`, which would also open the door to host-side batching.
- Additional backends in the default build once their distribution story matches ONNX Runtime's.
