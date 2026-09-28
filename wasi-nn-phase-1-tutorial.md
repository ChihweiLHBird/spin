# Tutorial: Phase 1 — a thin `wasi-nn` pass-through factor for Spin

This is a step-by-step guide for adding **`wasi:nn`** support to Spin by wrapping the Bytecode Alliance's `wasmtime-wasi-nn` crate in a new Spin **Factor**. It is the minimal, feature-gated integration: it links the standard `wasi:nn` host implementation into Spin and exposes the real native backends (ONNX Runtime, OpenVINO, PyTorch). It deliberately does **not** add host-side model caching, per-component access control, or backend selection via `runtime-config` — those are Phase 2 (see `wasi-nn-phase-2-design.md`).

Written for Spin `main` at `c28c4f92`: Wasmtime **49.0.0**, minimum Rust **1.96**, workspace version `4.2.0-pre0`, published world `spin:up@4.1.0`, and `wasmtime-wasi-nn` **49.0.0**. Upstream Spin does not yet depend on `wasmtime-wasi-nn`.

---

## 0. What you are building

```
guest .wasm                       Spin host process
  imports wasi:nn@0.2.0-rc-2024-10-28
        │
        │  graph::load / load-by-name / compute   (Canonical ABI call)
        ▼
  wasmtime Linker  ──►  WasiNnFactor (NEW)
                          ├─ init():    wit::add_to_linker(...)        ← registers host impl
                          └─ prepare(): WasiNnCtx::new(preload(&[])?)  ← per-instance context
                                              │
                                              ▼
                                   wasmtime-wasi-nn backends
                                   (ONNX Runtime / OpenVINO / PyTorch)
```

The new factor is structurally identical to `crates/factor-wasi` (which wraps `WasiCtx`/`WasiCtxView`). We reuse the same pattern to hand `wasmtime-wasi-nn` its `WasiNnView` (which needs both our `WasiNnCtx` and the shared `ResourceTable`).

**Scope of Phase 1**

- ✅ Guests can call `graph::load(bytes, encoding, target)`, `init-execution-context`, and `compute`.
- ✅ Real backends via Cargo features (`onnx` recommended on Linux).
- ✅ ONNX Runtime backend compiled into the default `spin` build (the same way `llm` ships Candle by default); OpenVINO, PyTorch and CUDA stay opt-in; `--no-default-features` drops it.
- ❌ No host model cache (each instance builds its own `WasiNnCtx`).
- ❌ No `load-by-name` registry populated from the manifest (the registry is empty).
- ❌ No `wasi_nn_models` allow-list and no `wasi_nn_encodings` check (any component with the import can load any model it has bytes for).
- ⚠️ `graph::load` and `compute` run as **synchronous** host calls (that is how `wasmtime-wasi-nn` generates its bindings), so each in-flight call pins a Tokio worker thread in Spin's async engine. Accepted for Phase 1; `wasi-nn-phase-2-design.md` §9.1 records the deferred alternative.

---

## 1. Prerequisites & backend choice

Use Rust 1.96 or newer, matching the workspace's current minimum. Native backend requirements apply only to the backend feature you select.

`wasmtime-wasi-nn`'s **default features are `["openvino", "winml"]`**, which are inconvenient on Linux (OpenVINO needs the Intel toolkit installed; WinML is Windows-only). So we will set `default-features = false` and pick a backend explicitly.

Recommended for development on Linux: the **ONNX Runtime** backend via the `ort` crate. Use the `onnx-download` feature so `ort` fetches a prebuilt ONNX Runtime shared library automatically — no system package required.

| Backend | Cargo features to enable | System requirement |
| :-- | :-- | :-- |
| ONNX Runtime (recommended) | `onnx`, `onnx-download` | none (auto-downloaded) |
| ONNX Runtime + CUDA GPU | `onnx`, `onnx-cuda` | CUDA toolkit + driver |
| OpenVINO | `openvino` | OpenVINO runtime installed |
| PyTorch | `pytorch` | libtorch |

---

## 2. Vendor the `wasi-nn` WIT

Spin vendors WIT packages as plain directories under `wit/deps/`. Add the `wasi:nn` package at the exact version `wasmtime-wasi-nn` 49.0.0 implements: **`wasi:nn@0.2.0-rc-2024-10-28`**. (The version must match, or the guest's imports won't resolve against the host.) The directory name drops the `wasi-` prefix — the repo convention for vendored snapshot packages is `cli@0.2.6`, `otel@0.2.0-rc.2`, `keyvalue-2024-10-17`, etc.; WIT resolution keys off the `package` declaration inside the file, but stick to the convention. (`wit/deps/` also holds a handful of flat single-file packages — `cli.wit`, `http.wit`, … — which are the *released* WASI 0.3.0 packages. A snapshot/RC package like `wasi:nn@0.2.0-rc-2024-10-28` belongs in a versioned directory, as above.)

```bash
# From the repo root. Grab the WIT from the crate you will depend on.
mkdir -p wit/deps/nn@0.2.0-rc-2024-10-28
curl -sL https://static.crates.io/crates/wasmtime-wasi-nn/wasmtime-wasi-nn-49.0.0.crate \
  | tar -xz -C /tmp
cp /tmp/wasmtime-wasi-nn-49.0.0/wit/wasi-nn.wit \
   wit/deps/nn@0.2.0-rc-2024-10-28/wasi-nn.wit
```

Then add the import to the **`platform`** world in `wit/world.wit`, gated behind an unstable feature exactly like `wasi-otel` already is:

```wit
// wit/world.wit  → inside `world platform { ... }`
  @unstable(feature = wasi-nn)
  include wasi:nn/ml@0.2.0-rc-2024-10-28;
```

`wasi:nn`'s `ml` world simply imports its four interfaces (`tensor`, `graph`, `inference`, `errors`), so `include` pulls them all in. (Equivalent explicit form, if you prefer: four `import wasi:nn/<iface>@0.2.0-rc-2024-10-28;` lines, each preceded by its own `@unstable(feature = wasi-nn)`.)

> The `@unstable` gate means existing guests are unaffected; only guests that opt into the `wasi-nn` feature see the import. The host always provides it (step 4), which is fine — a host may offer more than a guest uses.
>
> Three clarifications on what this step is for:
>
> 1. It is **not** what makes runtime linking work. A guest's `wasi:nn` imports are satisfied at instantiation by the factor's `add_to_linker` (step 4) whether or not `wit/world.wit` mentions `wasi:nn`. This step declares the import in Spin's published world so guest tooling and deployment-target validation know about it. `spin build` validates components against `[application] targets`: `crates/build` calls `spin_environments::validate_application_against_environment_ids`, which checks each component's imports against the environment's published worlds with `wac_types::validate_target`. Components can override targets. SIP 025 (`docs/content/sips/025-validate-target-environment.md`) documents that mechanism. Environment definitions can also carry `[configuration]` constraints (`key_value_stores`, `sqlite_databases`, `ai_models`) that are checked against the component's manifest keys at build time; Phase 2 adds `wasi_nn_models` and `wasi_nn_encodings` to that list.
> 2. The vendoring and the world edit must travel together: everything that parses `wit/` — notably `crates/world`'s `bindgen!`, which points at `path: "../../wit"` — must be able to resolve every package referenced from `world.wit`, or it stops compiling. With the package vendored, `cargo check -p spin-world` passes with the gated include in place.
> 3. It does **not** make `spin-world` generate host bindings for `wasi:nn` either. The `bindgen!` in `crates/world/src/lib.rs` binds an inline world that includes `spin:up/platform@4.0.0` (resolved from `wit/deps/spin@4.0.0/world.wit`) plus a few explicit extra includes such as `wasi:keyvalue/imports@0.2.0-draft2`; it never includes the root `wit/world.wit` (`spin:up@4.1.0`). Phase 1 does not need Spin-generated bindings, because it links `wasmtime-wasi-nn`'s own. If you later want `spin_world::wasi::nn::*` host traits (the deferred idea in `wasi-nn-phase-2-design.md` §9.1), add `include wasi:nn/ml@0.2.0-rc-2024-10-28;` to that inline world. For reference, the `@unstable(feature = wasi-otel)` include in the same `platform@4.0.0` world does yield `spin_world::wasi::otel::*` bindings today with no `features:` option in the macro call, so the gate is not expected to block generation.

---

## 3. Create the `spin-factor-wasi-nn` crate

The workspace `members` list uses the `crates/*` glob, so a new directory under `crates/` is picked up automatically — no `Cargo.toml` workspace edit needed for membership.

### 3a. `crates/factor-wasi-nn/Cargo.toml`

```toml
[package]
name = "spin-factor-wasi-nn"
version.workspace = true
authors.workspace = true
edition.workspace = true
license.workspace = true
homepage.workspace = true
repository.workspace = true
rust-version.workspace = true

[features]
# The crate itself enables no backend; the root `spin` crate turns `onnx` on in
# its default feature set (see §4). `onnx-download` makes ort fetch a prebuilt
# ONNX Runtime archive at build time.
onnx = ["wasmtime-wasi-nn/onnx", "wasmtime-wasi-nn/onnx-download"]
onnx-cuda = ["onnx", "wasmtime-wasi-nn/onnx-cuda"]
openvino = ["wasmtime-wasi-nn/openvino"]
pytorch = ["wasmtime-wasi-nn/pytorch"]

[dependencies]
anyhow = { workspace = true }
spin-factors = { path = "../factors" }
wasmtime = { workspace = true }
wasmtime-wasi-nn = { workspace = true }

[dev-dependencies]
spin-factors-test = { path = "../factors-test" }
tokio = { workspace = true, features = ["macros", "rt"] }

[lints]
workspace = true
```

### 3b. Add `wasmtime-wasi-nn` to the workspace dependency table

In the **root** `Cargo.toml`, under `[workspace.dependencies]` (next to the existing `wasmtime = "49.0.0"`), add:

```toml
wasmtime-wasi-nn = { version = "49.0.0", default-features = false }
```

Keep this version in lockstep with the workspace `wasmtime` — the crate is released with every Wasmtime train and depends on the exact matching versions (`wasmtime-wasi-nn` 49.0.0 requires `wasmtime = "49.0.0"` and `wiggle = "=49.0.0"`, both already in Spin's lock). So the two versions move together. Spin's `Cargo.lock` pins 49.0.0, so cargo resolves `wasmtime-wasi-nn` to 49.0.0 as well; a `cargo update` that lifts one lifts both.

`default-features = false` is the important part — it drops the `openvino` + `winml` defaults so a plain `cargo build` stays lightweight.

### 3c. `crates/factor-wasi-nn/src/lib.rs`

This is the whole factor. It mirrors `factor-wasi`'s `InitContextExt` trick to pass `wasmtime-wasi-nn` a plain `fn` accessor that yields a `WasiNnView`.

```rust
//! A Spin factor that provides the standard `wasi:nn` interface by wrapping
//! `wasmtime-wasi-nn`. See `wasi-nn-phase-1-tutorial.md`.

use spin_factors::{
    ConfigureAppContext, Factor, InitContext, PrepareContext, RuntimeFactors,
    SelfInstanceBuilder,
};
use wasmtime::component::Linker;
use wasmtime_wasi_nn::preload;
use wasmtime_wasi_nn::wit::{WasiNnCtx, WasiNnView};

/// The factor that exposes `wasi:nn` to guest components.
#[derive(Default)]
pub struct WasiNnFactor {
    _priv: (),
}

impl WasiNnFactor {
    pub fn new() -> Self {
        Self::default()
    }
}

impl Factor for WasiNnFactor {
    type RuntimeConfig = ();
    type AppState = ();
    type InstanceBuilder = InstanceState;

    fn init<T: InitContext<Self>>(&mut self, ctx: &mut T) -> anyhow::Result<()> {
        // Register the wasi:nn host implementation in the shared Linker.
        ctx.link_nn_bindings(wasmtime_wasi_nn::wit::add_to_linker)?;
        Ok(())
    }

    fn configure_app<T: RuntimeFactors>(
        &self,
        _ctx: ConfigureAppContext<T, Self>,
    ) -> anyhow::Result<Self::AppState> {
        Ok(())
    }

    fn prepare<T: RuntimeFactors>(
        &self,
        _ctx: PrepareContext<T, Self>,
    ) -> anyhow::Result<Self::InstanceBuilder> {
        // Build a fresh wasi-nn context for this component instance.
        // `preload(&[])` returns every backend that was compiled in (cheap
        // `default()` values) plus an empty named-graph registry. The expensive
        // work (building an inference session) happens lazily inside the
        // backend when the guest calls `graph::load`.
        let (backends, registry) = preload(&[])?;
        Ok(InstanceState {
            ctx: WasiNnCtx::new(backends, registry),
        })
    }
}

/// Per-instance state: owns this component instance's wasi-nn context.
pub struct InstanceState {
    ctx: WasiNnCtx,
}

// The builder *is* the instance state (no extra build step needed).
impl SelfInstanceBuilder for InstanceState {}

/// Extension trait (mirrors `spin-factor-wasi`) that lets us pass
/// `wasmtime-wasi-nn`'s `add_to_linker` a plain `fn` accessor producing a
/// `WasiNnView`. A `WasiNnView` borrows BOTH our `WasiNnCtx` and the shared
/// component `ResourceTable`, which we obtain together via `get_data_with_table`.
trait InitContextExt: InitContext<WasiNnFactor> {
    fn get_nn(data: &mut Self::StoreData) -> WasiNnView<'_> {
        let (state, table) = Self::get_data_with_table(data);
        WasiNnView::new(table, &mut state.ctx)
    }

    fn link_nn_bindings(
        &mut self,
        add_to_linker: fn(
            &mut Linker<Self::StoreData>,
            fn(&mut Self::StoreData) -> WasiNnView<'_>,
        ) -> wasmtime::Result<()>,
    ) -> wasmtime::Result<()> {
        add_to_linker(self.linker(), Self::get_nn)
    }
}

impl<T: InitContext<WasiNnFactor>> InitContextExt for T {}
```

Why this compiles (the load-bearing facts):

- `wasmtime_wasi_nn::wit::add_to_linker::<T>(l, f: fn(&mut T) -> WasiNnView)` is exactly the shape `link_nn_bindings` expects, with `T = Self::StoreData`.
- `InitContext::get_data_with_table(store)` returns `(&mut FactorInstanceState<Self>, &mut ResourceTable)`. Because `InstanceState: SelfInstanceBuilder`, `FactorInstanceState<WasiNnFactor>` *is* `InstanceState`, so `state.ctx` is our `WasiNnCtx`.
- `WasiNnView::new(table, ctx)` takes the two disjoint mutable borrows returned in that tuple.
- `Factor::init` is declared with a named generic (`fn init<T: InitContext<Self>>(&mut self, ctx: &mut T)`), so the impl must use the same form — writing `ctx: &mut impl InitContext<Self>` fails with E0643. `factor-wasi` uses the identical form.
- `factor-wasi` links several WASI generations because Spin's `platform` world imports `wasi:cli` at 0.2.6, `0.3.0-rc-2026-03-15`, and 0.3.0 final (plus older snapshots). `wasi:nn` ships a single synchronous 0.2-style interface — its one `add_to_linker` is the whole job.

---

## 4. Register the factor in `TriggerFactors`

Edit `crates/runtime-factors/src/lib.rs`:

```rust
use spin_factor_wasi_nn::WasiNnFactor;          // add with the other `use`s

#[derive(RuntimeFactors)]
pub struct TriggerFactors {
    pub otel: OtelFactor,
    pub wasi: WasiFactor,
    pub nn: WasiNnFactor,                        // add (position is not significant)
    // ... existing fields ...
    pub llm: LlmFactor,
}
```

And in `TriggerFactors::new(...)`, add the field initializer:

```rust
        Ok(Self {
            // ... existing ...
            nn: WasiNnFactor::new(),
            llm: LlmFactor::new( /* ... */ ),
        })
```

`TriggerFactors` currently has twelve fields, from `otel` through `llm`.

### 4a. Required: a runtime-config impl in `crates/runtime-config`

`#[derive(RuntimeFactors)]` generates a `TriggerFactorsRuntimeConfig` whose `from_source` requires `TomlRuntimeConfigSource: FactorRuntimeConfigSource<F>` for **every** factor field — including ours, even though our `RuntimeConfig` is `()`. Without this step, `cargo check -p spin-runtime-factors` fails with:

```text
error[E0277]: the trait bound `TomlRuntimeConfigSource<'_, '_>:
  FactorRuntimeConfigSource<WasiNnFactor>` is not satisfied
```

Mirror the existing `WasiFactor` impl — the same `()`-config shape — in `crates/runtime-config/src/lib.rs`:

`crates/runtime-config/Cargo.toml`, under `[dependencies]`:

```toml
spin-factor-wasi-nn = { path = "../factor-wasi-nn" }
```

`crates/runtime-config/src/lib.rs`:

```rust
use spin_factor_wasi_nn::WasiNnFactor;   // with the other factor imports

impl FactorRuntimeConfigSource<WasiNnFactor> for TomlRuntimeConfigSource<'_, '_> {
    fn get_runtime_config(&mut self) -> anyhow::Result<Option<()>> {
        Ok(None)
    }
}
```

This is also exactly where Phase 2 later plugs in: its `[wasi_nn_model.*]` TOML parsing replaces this `Ok(None)`.

Like the LLM factor, `WasiNnFactor` is always present. When no backend feature is enabled, `preload(&[])` yields zero backends and `graph::load` returns an `invalid-encoding` error — a graceful "not configured" state. When a backend feature is on, it works. (A clearer `unsupported-operation` error for this case would need Spin to own the host bindings, a deferred idea in Phase 2 §9.1.)

> Alternative (more surgical, more boilerplate): make the dependency `optional` and `#[cfg(feature = "wasi-nn")]`-gate the struct field + initializer. The always-present approach above matches how `factor-llm` is wired and avoids conditional-field handling in the derive.

### Feature plumbing

Mirror the existing `llm` feature chain. The ONNX backend goes **into** `default`, exactly as `llm` does; OpenVINO, PyTorch and CUDA stay opt-in.

`crates/runtime-factors/Cargo.toml`:

```toml
[features]
# ... existing llm features ...
wasi-nn-onnx = ["spin-factor-wasi-nn/onnx"]
wasi-nn-onnx-cuda = ["spin-factor-wasi-nn/onnx-cuda"]
wasi-nn-openvino = ["spin-factor-wasi-nn/openvino"]
wasi-nn-pytorch = ["spin-factor-wasi-nn/pytorch"]

[dependencies]
# ... existing ...
spin-factor-wasi-nn = { path = "../factor-wasi-nn" }
```

Root `Cargo.toml` (top-level `spin` binary), with `wasi-nn-onnx` added to `default`:

```toml
[features]
default = ["llm", "cpu-time-metrics", "wasi-nn-onnx"]   # ONNX backend on by default
# ...
wasi-nn-onnx = ["spin-runtime-factors/wasi-nn-onnx"]
wasi-nn-onnx-cuda = ["spin-runtime-factors/wasi-nn-onnx-cuda"]
wasi-nn-openvino = ["spin-runtime-factors/wasi-nn-openvino"]
wasi-nn-pytorch = ["spin-runtime-factors/wasi-nn-pytorch"]
```

What putting `onnx` + `onnx-download` into `default` means, from `ort-sys` 2.0.0-rc.10's `build.rs` (the `ort` version `wasmtime-wasi-nn` 49.0.0 pins; it bundles ONNX Runtime 1.22.0): unless a system ONNX Runtime is found first via `pkg-config` (`libonnxruntime` whose minor version is at least 22) or `ORT_LIB_LOCATION`, every build downloads a prebuilt archive from `cdn.pyke.io`, verifies its SHA-256, and caches it under the user cache directory, so the download happens once per machine per version. The CPU archive contains a static `libonnxruntime.a`; `ort` links it statically together with the platform C++ runtime, so no shared library ships next to `spin`. `wasmtime-wasi-nn` enables `ort`'s `copy-dylibs` feature, which copies shared libraries beside the binary when the archive contains them (CUDA); the CPU static archive does not need that. CPU prebuilt archives exist for `x86_64`/`aarch64-unknown-linux-gnu`, `x86_64`/`aarch64-apple-darwin`, `x86_64`/`aarch64-pc-windows-msvc`, and `wasm32-unknown-emscripten`. That covers every `build-and-sign` job in `.github/workflows/release.yml` (Linux gnu amd64 and aarch64, macOS Intel and Apple Silicon, Windows amd64). There is **no musl** archive. The `build-spin-static` jobs in `release.yml` and `build.yml` build `x86_64-unknown-linux-musl` and `aarch64-unknown-linux-musl`, where `ort-sys` panics with "downloaded binaries not available for target"; those two jobs need `--no-default-features --features llm,cpu-time-metrics` (or a from-source musl ONNX Runtime via `ORT_LIB_LOCATION`). With `CARGO_NET_OFFLINE=true` or `ORT_SKIP_DOWNLOAD=1` the download is skipped and linking fails unless one of the local sources above is provided.

---

## 5. Build

```bash
# Validate the published world after adding the vendored WIT.
cargo check -p spin-world

# The factor crate alone: no backend, no native ML deps.
cargo build -p spin-factor-wasi-nn

# Validate TriggerFactors and the runtime-config source implementation.
cargo check -p spin-runtime-factors

# Default build: includes the ONNX backend (downloads ONNX Runtime once per machine).
cargo build

# Without any wasi-nn backend (what the musl release jobs must do, see §4):
cargo build --no-default-features --features llm,cpu-time-metrics

# Lint as Spin requires:
make lint
```

If you enabled `openvino` instead, install the OpenVINO runtime first (the crate uses `runtime-linking`, so it loads the shared library at runtime).

---

## 6. Verify it works

### 6a. Smoke test (no model needed) — proves the host wiring

Write a tiny guest that imports `wasi:nn` and calls `load` with junk bytes. With the `onnx` backend compiled, the call reaches the backend and fails with a *runtime* error (not a *linking* error) — which proves the import resolved and dispatched into `wasmtime-wasi-nn`.

A `factors-test` unit test in `crates/factor-wasi-nn/tests/` is the fastest loop. Model it on `crates/factor-llm/tests/factor_test.rs`: build a `TestEnvironment` with `WasiNnFactor`, instantiate a guest that imports `wasi:nn`, and assert instantiation succeeds.

### 6b. End-to-end with a real ONNX model

1. Author a guest component that generates bindings from the vendored WIT (`wit-bindgen`), then runs the standard flow. The call shapes are:

   ```rust
   // pseudocode against wasi:nn@0.2.0-rc-2024-10-28
   use wasi::nn::graph::{load, ExecutionTarget, GraphEncoding};
   use wasi::nn::tensor::{Tensor, TensorType};

   let model: Vec<u8> = include_bytes!("mobilenet.onnx").to_vec();
   let graph = load(&[model], GraphEncoding::Onnx, ExecutionTarget::Cpu)?;
   let ctx = graph.init_execution_context()?;

   // named-tensor = tuple<string, tensor>
   let input = Tensor::new(&[1, 3, 224, 224], TensorType::Fp32, &input_bytes);
   let outputs = ctx.compute(&[("input".to_string(), input)])?;
   // outputs: list<(string, tensor)> — read dimensions()/ty()/data() off each.
   ```

   Note the WIT `compute` takes and returns **named tensors** (`list<tuple<string, tensor>>`); there are no separate `set-input`/`get-output` calls in the WIT (component-model) ABI.

2. Minimal `spin.toml`:

   ```toml
   spin_manifest_version = 2
   [application]
   name = "wasi-nn-demo"

   [[trigger.http]]
   route = "/infer"
   component = "classifier"

   [component.classifier]
   source = "target/wasm32-wasip2/release/classifier.wasm"
   [component.classifier.build]
   command = "cargo build --target wasm32-wasip2 --release"   # same target as Spin's http-rust template
   ```

3. Run with the backend feature enabled:

   ```bash
   cargo run -- up -f spin.toml
   curl localhost:3000/infer
   ```

> Reference guests: the `wasmtime-wasi-nn` repo ships example components and `tests/test-programs.rs` exercising exactly these calls — useful to copy from.

---

## 7. Troubleshooting

| Symptom | Cause | Fix |
| :-- | :-- | :-- |
| `component imports ... wasi:nn ... not provided` at instantiation | Factor not registered, or version mismatch | Confirm `WasiNnFactor` is in `TriggerFactors`; ensure guest imports `@0.2.0-rc-2024-10-28` and the vendored WIT matches |
| `E0277: ... FactorRuntimeConfigSource<WasiNnFactor> ... not satisfied` building `spin-runtime-factors` | Step 4a skipped | Add the `FactorRuntimeConfigSource<WasiNnFactor>` impl in `crates/runtime-config` |
| `load` returns `invalid-encoding` | No backend compiled for that encoding, or (Phase 2) the component did not declare it | The default build has ONNX; enable `wasi-nn-openvino` / `wasi-nn-pytorch` for other encodings; check you did not build with `--no-default-features` |
| Build panics: `downloaded binaries not available for target …-linux-musl` | `ort` ships no prebuilt ONNX Runtime for musl | Build musl targets with `--no-default-features --features llm,cpu-time-metrics`, or point `ORT_LIB_LOCATION` at a from-source ONNX Runtime |
| Offline build fails to link `onnxruntime` | Download skipped (`CARGO_NET_OFFLINE=true` or `ORT_SKIP_DOWNLOAD=1`) | Provide a pkg-config `libonnxruntime` on the 1.22 line (minor version ≥ 22) or set `ORT_LIB_LOCATION` |
| Build pulls OpenVINO/WinML unexpectedly | `default-features` left on | Set `default-features = false` on the workspace `wasmtime-wasi-nn` dep |
| ONNX link/load errors at runtime | ONNX Runtime lib missing | Use the `onnx-download` feature (already in `onnx` above) |
| Borrow/lifetime error in `get_nn` | Tried to split `ctx`/`table` manually | Get both from one `get_data_with_table` call, as shown |

---

## 8. What Phase 1 leaves for Phase 2

- **Cold starts:** each instance rebuilds its session on `graph::load`. Phase 2 preloads graphs once into a shared registry (`Graph` is `Arc`-cloneable) and serves them via `load-by-name`.
- **Governance:** Phase 2 enforces a per-component `wasi_nn_models` allow-list for `load-by-name`, and makes raw guest-supplied `load(bytes)` a declared per-component capability, `wasi_nn_encodings = ["onnx"]` in `spin.toml`, denied by default (Phase 2 §9.6). Phase 1 allows it for every component, which is fine for a spike but not for merging.
- **Backend/target selection:** Phase 2 drives backend + `execution-target` from `runtime-config.toml` instead of compile-time features only.
- **Blocking host calls:** Phase 1's `load`/`compute` are synchronous host functions on Spin's async engine. Phase 2 §9.1 keeps the deferred alternative, Spin-owned async bindings running the backend work in `spawn_blocking`.

See `wasi-nn-phase-2-design.md`.
