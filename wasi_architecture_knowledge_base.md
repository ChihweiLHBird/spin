# Architectural Knowledge Base: WASI, Wasmtime, Spin, and Edge AI Systems

This document is an engineering handover and reference summary covering the intersection of Wasmtime (the runtime engine), Fermyon/Spin (the application platform), and WASI (WebAssembly System Interface) specifications, with a focus on edge AI inference (`wasi-nn`).

Written against Spin `main` at `c28c4f92` (Wasmtime **49.0.0**, `wasmtime-wasi-nn` **49.0.0**, minimum Rust **1.96**, workspace version `4.2.0-pre0`, `spin:up@4.1.0`). Where a statement is an architectural opinion rather than a verifiable fact, it is labelled **(opinion)**.

---

## 1. The Core Runtime Paradigm: Spin vs. Wasmtime

Cloud-native WebAssembly execution relies on a separation between **design time** (abstract guest requests) and **runtime** (concrete host provision).

```text
+-------------------------------------------------------+
| GUEST COMPONENT                                       |
| - Compiled against standard WASI/WIT interfaces       |
| - Declares abstract capabilities (e.g., KV "default") |
+-------------------------------------------------------+
                          │
                          ▼  (WASI import call, Canonical ABI)
+-------------------------------------------------------+
| WASMTIME                                              |
| - Manages linear-memory sandbox and CPU execution    |
| - Implements system-level WASI itself (wasmtime-wasi) |
| - Dispatches component import calls to host functions |
|   registered in the `Linker`                          |
+-------------------------------------------------------+
                          │
                          ▼  (host function call)
+-------------------------------------------------------+
| SPIN HOST (Factors)                                  |
| - Resolves logical targets to concrete drivers       |
| - Manages pooling, state, sockets, auth, access rules |
| - Examples: SQLite, Redis, model engines, KV stores   |
+-------------------------------------------------------+
```

### Execution flow

1. **Design-time declaration.** The developer declares logical dependencies in `spin.toml` (e.g., `key_value_stores = ["default"]`) without compiling a specific database driver into the `.wasm` guest.

2. **Runtime configuration.** The operator maps that logical dependency to physical infrastructure in `runtime-config.toml` (e.g., `"default"` → Redis or local SQLite).

3. **Linking.** When the guest invokes an imported WASI function, Wasmtime dispatches it — through the component model's **Canonical ABI** — to the host function that Spin registered in the `wasmtime::component::Linker`. This is an ordinary cross-boundary **function call**. In WebAssembly a "trap" means an abnormal abort — for example an out-of-bounds access or `unreachable`. Spin's host code (Rust) then executes the operation.

---

## 2. WASI Categorization: System vs. Application Level

WASI specifications carry **no** explicit metadata flag separating "system" from "application" layers. The standard is built around **virtualizability**: any interface may be implemented or intercepted by another component. The split below is an informal, editorial convention, not something encoded in WASI.

| Layer category | Provided by | Characteristics | Examples in Spin today |
| :--- | :--- | :--- | :--- |
| **System-level primitives** | **`wasmtime-wasi`** (the implementation) | Map onto OS kernel abstractions. Spin supplies *policy/capabilities* (preopened dirs, socket allow-lists) but does not re-implement the I/O. | `wasi:random`, `wasi:clocks`, `wasi:filesystem`, `wasi:sockets` (all wired in `crates/factor-wasi`) |
| **Application-level services** | **Spin Factors** (host adapters) | Distributed/cloud primitives needing host state or protocol orchestration. | `wasi:keyvalue`, `wasi:config/store`, plus Spin's proprietary `spin:*` interfaces |

Spin implements neither standard `wasi:sql` nor standard `wasi:messaging`. It ships proprietary equivalents (`spin:postgres`, `spin:mysql`, `spin:redis`, `spin:mqtt`). See §3.

Spin does more than pass parameters for the system layer: `factor-wasi` configures filesystem preopens (`SpinFilesMounter`, using `FsPerms`) and per-component outbound socket allow-lists. Socket creation is allowed explicitly (`allow_tcp` / `allow_udp`) because `wasmtime-wasi` 49 denies it by default. It does not re-implement the underlying syscalls — those are `wasmtime-wasi`'s.

---

## 3. Spin Interface Support Registry

Verified against `wit/world.wit` (package `spin:up@4.1.0`) and the `crates/` tree on `main` at `c28c4f92`. The `platform` world imports the **WASI 0.2.6**, **0.3.0-rc-2026-03-15**, and **0.3.0 (final)** CLI worlds — WASI 0.3.0 was ratified 2026-06-11 — and Spin runs on **Wasmtime 49.0.0** (`wasmtime-wasi` with the `p3` feature). `TriggerFactors` has twelve fields, from `otel` through `llm`. The root default features are `["llm", "cpu-time-metrics"]`.

### Supported

- **`wasi:otel` (observability).** Imported as `wasi:otel/imports@0.2.0-rc.2`, gated behind `@unstable(feature = wasi-otel)` and the `--experimental-wasi-otel` flag; implemented by `crates/factor-otel`. Lets guests emit OpenTelemetry spans to the host.
- **`wasi:config/store`.** Imported as `wasi:config/store@0.2.0-draft-2024-09-27`; backed by Spin's variables/config subsystem (`crates/factor-variables`). The standard `wasi-config` interface, *separate from* the proprietary `spin:variables`.
- **`wasi:keyvalue`.** Imported as `wasi:keyvalue/imports@0.2.0-draft2` (alongside the proprietary `spin:key-value`).

### Not supported — open engineering opportunities

- **`wasi:logging`.** **Not implemented.** There is no `wasi:logging` host binding anywhere in the tree. Spin captures guest **stdout/stderr** through `wasi:cli` stdio and forwards it to host aggregators. That capture is separate from the `wasi:logging` interface (which provides a level-aware `log(level, context, message)` import). Implementing `wasi:logging` would be a small, self-contained factor.
- **`wasi:blobstore`.** Absent. Guests fall back to outbound HTTP to reach object stores (e.g., S3). A `BlobstoreFactor` would host this.
- **`wasi:messaging`.** The standard interface is absent. Spin supports outbound publish today via the proprietary `spin:mqtt` interface (`factor-outbound-mqtt`) and outbound Redis, and it supports inbound triggers (Redis pub/sub, MQTT). What is missing is the *standard* `wasi:messaging` client interface and brokers like Kafka / RabbitMQ / NATS.
- **`wasi:sql`.** Spin uses proprietary namespaces `spin:postgres@{3.0.0, 4.2.0}` and `spin:mysql@3.0.0`. Adopting standard `wasi:sql` would improve cross-runtime portability.
- **`wasi:nn`.** **Not supported** (the subject of §4). `wasmtime-wasi-nn` is not a dependency (no references in `Cargo.lock`).
- **Lower-priority / less-standardized:** `wasi:crypto`, distributed locking, data-parallel offload, timezone-aware clocks, and embedded buses (`wasi:i2c`, `wasi:gpio`, `wasi:spi`, etc.). All absent.

`spin build` validates an application against its target environments. `crates/build` calls `spin_environments::validate_application_against_environment_ids`, which checks each component's imports against the environment's published worlds with `wac_types::validate_target`. Environment definitions also accept `[configuration]` constraints (`key_value_stores`, `sqlite_databases`, `ai_models`) checked against the component manifest. SIP 025 documents that mechanism. A wasi-nn factor extends the same `[configuration]` list with `wasi_nn_models` and `wasi_nn_encodings`.

`crates/world/src/lib.rs`'s `bindgen!` binds an inline world that includes `spin:up/platform@4.0.0` (from `wit/deps/spin@4.0.0/world.wit`) plus explicit extra includes; it does not include the root `wit/world.wit` (`spin:up@4.1.0`). An import added only to `wit/world.wit` is published to guests and environments and produces no `spin_world` host bindings; those need an include in the inline world. The `@unstable(feature = wasi-otel)` include in `platform@4.0.0` does yield `spin_world::wasi::otel::*` bindings with no `features:` option, so gating is not expected to block generation.

---

## 4. Deep Dive: `wasi-nn` Implementation Architecture

> **Status:** Spin has **no** `wasi-nn` support today. `wasmtime-wasi-nn` is not a dependency. Everything in this section is a **design proposal**. The operational patterns it describes already exist in Spin's proprietary **`fermyon:spin/llm@2.0.0`** factor (`crates/factor-llm`); a `wasi-nn` factor would replicate them.

### What the runtime gives you for free

The Bytecode Alliance ships **`wasmtime-wasi-nn`** (versioned in lockstep with Wasmtime; `49.0.0` is the version in Spin's lock). It implements the host side of the **`wasi:nn@0.2.0-rc-2024-10-28`** WIT interface and embeds native bindings to hardware-accelerated backends. The published manifest pins `wasmtime` and `wiggle` to 49.0.0 and the optional `openvino` dependency to 0.11. Facts about the public API:

- **Backends** (each behind a Cargo feature): `openvino`, `onnx` (via the `ort` crate 2.0.0-rc.10, ONNX Runtime 1.22.0; `onnx-download` auto-fetches prebuilt ONNX Runtime binaries), `pytorch` (via `tch`/libtorch), and `winml` (Windows only). **Default features are `["openvino", "winml"]`**, which are awkward on Linux — integrators should set `default-features = false` and opt into a backend explicitly. Optional `ort-tracing` exists too.
- **Key types:** `WasiNnCtx::new(backends, registry)` holds the per-context state; `WasiNnView<'a> { ctx, table }` bundles the context with the component `ResourceTable`; `wasmtime_wasi_nn::wit::add_to_linker(linker, |t| -> WasiNnView)` registers the host implementation; `wasmtime_wasi_nn::preload(&[(backend, dir)])` returns `(Vec<Backend>, Registry)` and preloads named graphs from directories using `ExecutionTarget::Cpu`.
- **Shareability:** a loaded `Graph` is `Arc<dyn BackendGraph>` (cheaply cloneable across instances). A `Backend` is `Box<dyn BackendInner>` (not cloneable). All three of `BackendInner`, `BackendGraph`, `GraphRegistry` are `Send + Sync`.
- **Two load paths:** `graph::load(bytes, encoding, target)` (guest supplies the model bytes; dispatched to the matching backend) and `graph::load-by-name(name)` (looked up in the host `registry`). The latter is the natural hook for host-managed model caching.
- **Host functions are synchronous.** Bindgen uses `imports: { default: trappable }`. Spin's engine is async, so each `load` / `compute` pins a Tokio worker. `OnnxGraph` is `Arc<Mutex<ort::Session>>` and holds the lock across `session.run`, so a shared ONNX graph admits one inference at a time. OpenVINO creates an `InferRequest` per execution context and calls `infer()` without holding the compiled-model lock for the duration of inference. PyTorch is `Arc<Mutex<tch::CModule>>`, the same serialising shape as ONNX.

What `wasmtime-wasi-nn` does **not** provide is application-layer awareness: manifests, tenancy, model lifecycle, batching. Those are Spin's job. The proposed `wasi-nn` factor should address:

### 1. Cold-start weight-loading

- **Problem:** calling `graph::load` per request rebuilds the backend session from a large model file (parsing, optimization), spiking latency from microseconds to seconds.
- **Proposed solution (precedent exists):** load/cache graphs once in host state and hand cheap `Arc` clones to each ephemeral instance via `load-by-name`. Spin's LLM factor already does this: `crates/llm-local` caches models in a `HashMap<ModelName, Arc<…>>` and shares them across instances. A shared ONNX session still serialises `compute` on one mutex (§ above); a per-model pool of sessions is the scaling knob.

### 2. Access governance / tenancy

- **Problem:** Wasmtime has no notion of a manifest or per-component authorization.
- **Proposed solution (precedent exists):** enforce per-component model allow-lists from the manifest. Spin's LLM factor already implements this with the `ai_models` metadata key (`ALLOWED_MODELS_KEY = MetadataKey::new("ai_models")` in `factor-llm`, enforced per component in `host.rs`). wasi-nn uses its own keys, because the LLM factor is being evaluated for a rewrite:

  ```toml
  [component.classifier]
  source = "vision.wasm"
  wasi_nn_models = ["resnet50"]      # host-registered models
  wasi_nn_encodings = ["onnx"]       # guest-supplied bytes; denied when omitted
  ```

  Environment definitions can constrain manifest keys at build time (`[configuration]` in `crates/environments`, today for `key_value_stores`, `sqlite_databases`, `ai_models`). Phase 2 adds `wasi_nn_models` and `wasi_nn_encodings` there, so each wasi-nn key feeds three layers: runtime enforcement by the factor, build-time validation against target environments, and composition inheritance via the capabilities deny adapter.

  Filtering `load-by-name` is not sufficient while raw `graph::load(bytes, ...)` remains available. Raw loading is denied unless the component declares the encoding in `wasi_nn_encodings`. With `wasmtime-wasi-nn`'s host code, `prepare` installs only the backends for those encodings, so an undeclared encoding surfaces as `invalid-encoding` ("unable to find a backend for this encoding"). Named-model misses and disallowed names both surface as `not-found`.

### 3. Hardware/backend decoupling

- **Problem:** raw `graph::load` makes the guest choose both `GraphEncoding` and `ExecutionTarget`, so it does not fully decouple deployment hardware.
- **Proposed solution (aspirational):** the guest calls `load-by-name("resnet50")`; the host loads that name through the backend and target selected in `runtime-config.toml` under `[wasi_nn_model.<name>]`. `wasmtime-wasi-nn` already exposes the backend and target primitives needed for this host-managed path.

### 4. Concurrency throttling / batching

- **Problem:** large bursts of concurrent instances all calling native backends can saturate host CPU/accelerators.
- **Proposed solution (aspirational, not implemented in Spin or `wasmtime-wasi-nn`):** buffer inbound tensor requests at the host, assemble hardware-efficient batches, run inference, and route results back to the originating async instances.

### 5. `wasi-nn` vs `wasi-webgpu` prioritization **(opinion)**

For request-driven serverless edge AI, `wasi-nn` is a higher-value target than `wasi-webgpu`. This is an architectural judgement, not a fact:

| Aspect | `wasi-nn` | `wasi-webgpu` |
| :--- | :--- | :--- |
| Abstraction | High-level graph/tensor inference | Low-level GPU pipelines, buffers, WGSL shaders |
| Init overhead | Lower — host can cache graphs and reuse sessions | Higher — guest manages pipelines/buffers per use |
| Guest footprint | Lean — the execution engine is entirely host-side | Heavier — the guest carries shader/orchestration logic (GPU execution is still host-side) |
| Best fit | Ephemeral, scale-to-zero functions | Long-running, stateful GPU workloads |

**Proposal status:** the official [WASI proposal registry](https://github.com/WebAssembly/WASI/blob/main/docs/Proposals.md) lists both proposals at **WASI proposal Phase 2** (a standardization stage, unrelated to this repo's Phase 1 / Phase 2 document names). Their health profiles differ:

- **`wasi-nn`** (champions: Andrew Brown, Mingqiu Sun) — the published WIT package is still `wasi:nn@0.2.0-rc-2024-10-28`, and a stage-3 support survey ([wasi-nn#83](https://github.com/WebAssembly/wasi-nn/issues/83)) has sat open since Nov 2024, so standardization is stalled. The *implementations* keep shipping: `wasmtime-wasi-nn` is released with each Wasmtime train (including an ONNX CUDA execution provider), and WasmEdge's WASI-NN plugin (GGML/llama.cpp) is the foundation of LlamaEdge.
- **`wasi-webgpu` / "GFX"** (champions: Mendy Berger, Sean Isom) — the current WIT is **`wasi:webgpu@0.3.0-rc.2`**, not a final 0.3.0 release. There is no implementation in the Bytecode Alliance Wasmtime tree. The external wasi-gfx project publishes an embeddable [`wasi-webgpu-wasmtime` 0.2.0](https://crates.io/crates/wasi-webgpu-wasmtime) host crate that depends on Wasmtime 47. Spin is on Wasmtime 49, so that crate does not drop into this tree as-is.

WASI 0.3.0 itself (ratified 2026-06-11) covers only the Phase-3 core set (cli / clocks / random / filesystem / sockets / http); neither wasi-nn nor wasi-webgpu is part of it, and wasi-nn's WIT has not been rebased onto 0.3. The prioritization above remains an opinion. `wasi-nn` is still the more direct Spin inference fit because it is a high-level graph/tensor API shipped in the Wasmtime monorepo at the same version Spin already uses, while `wasi:webgpu` is lower-level, externally maintained, still on an RC WIT, and a Wasmtime train behind.

---

## 5. Vectors for Continued Exploration

1. **Async (WASI 0.3).** WASI 0.3.0 went final on 2026-06-11; Spin is on Wasmtime 49 with `wasmtime-wasi`'s `p3` feature, imports the final 0.3.0 worlds, and has a `crates/wasi-async` (with `stream.rs`). Investigate whether `stream<T>` / `future<T>` in the Canonical ABI can improve `wasi-nn` batching once the `wasi-nn` WIT itself gains async `compute`. That WIT is still synchronous and has not been rebased onto 0.3.

2. **Standardizing Spin's AI surface.** Spin's inference today is the proprietary **`fermyon:spin/llm@2.0.0`** interface (the package is `fermyon:spin/llm`, not `fermyon:llm`), backed by **Candle** (`candle-*` 0.8) locally and a remote-HTTP engine. It is separate from `wasmtime-wasi-nn`. Converging onto standard `wasi:nn` is therefore both an *interface* change and a *backend* change, and the abstraction levels differ (`llm` exposes `infer`/`embeddings`; `wasi:nn` exposes graphs/tensors).

3. **Host-glue prototyping.** Map a Spin `Factor` onto `wasmtime_wasi_nn::wit::{WasiNnCtx, WasiNnView, add_to_linker}` — structurally identical to how `crates/factor-wasi` wires `WasiCtx` / `WasiCtxView` through `InitContextExt` and `InitContext::get_data_with_table`. `Factor::init` is generic (`fn init<T: InitContext<Self>>`). See the companion docs `wasi-nn-phase-1-tutorial.md` (thin pass-through) and `wasi-nn-phase-2-design.md` (host model registry + governance).

The `Factor` linking pattern `factor-wasi` uses is the one Phase 1 copies. `factor-wasi` also links `wasmtime_wasi::p2` and `::p3` (and older snapshots) because the platform world imports several WASI generations; `wasi:nn` needs a single `add_to_linker`.
