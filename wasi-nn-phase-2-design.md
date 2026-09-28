# Phase 2 — Host-managed `wasi-nn`: model registry, caching & governance

This is a **design/architecture sketch** for the follow-up to Phase 1. It describes concepts and component boundaries, not full implementation. It assumes Phase 1 (the thin `spin-factor-wasi-nn` wrapping `wasmtime-wasi-nn`) is already in place — Phase 2 evolves that same factor rather than replacing it.

Written against Spin `main` at `c28c4f92` (Wasmtime and `wasmtime-wasi-nn` **49.0.0**, Rust **1.96**, `spin:up@4.1.0`). The shapes this design depends on: `factor-llm`'s `ALLOWED_MODELS_KEY = MetadataKey::new("ai_models")`, `crates/capabilities`' `AI_MODELS = &["fermyon:spin/llm", "fermyon:spin/llm@2.0.0"]` plus its `make -C crates/capabilities adapter` rebuild step, the `FactorRuntimeConfigSource` hook point in `crates/runtime-config`, and `wasmtime-wasi-nn`'s `pub trait GraphRegistry { get, get_mut }` with its blanket `impl<T: GraphRegistry + 'static> From<T> for Registry`, `WasiNnCtx::new(impl IntoIterator<Item = Backend>, Registry)`, and `WasiNnView`'s dual `&mut ctx` / `&mut ResourceTable` borrow. wasi-nn owns its own manifest keys (§9.4); the LLM items are precedent for the pattern, not keys to reuse.

Goal: turn the pass-through into a Spin-native subsystem that follows Spin's design-time/runtime split and matches the UX of the existing `fermyon:spin/llm` factor. Concretely, close Phase 1's three gaps:

1. **Cold starts** — stop rebuilding inference sessions per request.
2. **Governance** — enforce which host-managed models each component may use.
3. **Backend decoupling** — choose backend/hardware at deploy time, not compile time.

---

## 1. Key architectural insight (what makes this fit Spin)

`wasmtime-wasi-nn`'s `WasiNnView` borrows `&mut WasiNnCtx` **and** `&mut ResourceTable`, so each component instance must own its `WasiNnCtx`; we cannot share one `WasiNnCtx` behind a lock. But the expensive thing — a **loaded graph** — is internally `Arc<dyn BackendGraph>`, so `Graph` is cheaply cloneable and `Send + Sync`. It is self-contained: it can create execution contexts without the originating `Backend` staying alive.

So the design is:

- **Load each model at app scope**, storing cloneable `Graph` values in a shared registry.
- **Give every instance a cheap, per-instance `WasiNnCtx`** whose registry is a thin view over those shared graphs, scoped to what the component is allowed to see.
- **Install raw-load backends only for encodings the component declares.** A guest can otherwise bypass `wasi_nn_models` by calling `graph::load(bytes, ...)` directly. Raw loading is a declared per-component capability (`wasi_nn_encodings`), denied by default (§9.6).

This is the same shape as `factor-llm` today: an app-scoped, shared, cached engine (`Arc`) plus a per-instance `allowed_models` set.

---

## 2. Where state lives (Factor lifecycle mapping)

| Factor stage | Responsibility in Phase 2 |
| :-- | :-- |
| `configure_app` (per loaded app) | Read model declarations (manifest + `runtime-config`). For each declared model, load it via the configured backend and store the resulting `Graph` in a shared `AppState` registry. Build the per-component `wasi_nn_models` allow-list map (the same shape `factor-llm` uses for its `ai_models`). |
| `prepare` (per instance) | Construct a cheap per-instance `WasiNnCtx`: backends only for encodings this component declared in `wasi_nn_encodings`, and a **scoped registry** exposing only the graphs this component's `wasi_nn_models` permits. |
| `init` (once) | Same as Phase 1 — `wit::add_to_linker`. Unchanged. |

Spin's current `FactorsExecutor` calls `configure_app` once while constructing a loaded app and reuses its `AppState` for instances. The `Factor` contract itself allows other runtimes to reconfigure an app, so this is app-scoped reuse rather than a global exactly-once guarantee.

```
AppState (shared, built in configure_app)
  ├─ models: Arc<HashMap<String /*name*/, Graph /*internally Arc-backed*/>>
  │                                                          ← loaded once, cached
  └─ component_allowed_models: HashMap<ComponentId, Arc<HashSet<String>>>

InstanceState (per instance, built in prepare)
  └─ ctx: WasiNnCtx {
        backends: [only declared encodings],  // undeclared encodings have no backend
        registry: ScopedRegistry {             // implements wasmtime_wasi_nn GraphRegistry
            models: Arc<HashMap<..>>,          // cloned Arc handle (cheap)
            allowed: Arc<HashSet<String>>,     // this component's allow-list
        }
     }
```

`ScopedRegistry` is a small custom `impl GraphRegistry`. Its `get` returns a graph only if its name is in `allowed`; its `get_mut` returns `None` because the shared map is deliberately read-only. The WIT/component `load-by-name` path used here calls `get`, clones the `Graph`, and puts that clone in the instance's resource table. Thus `load-by-name("resnet50")` succeeds only for authorized components without copying or rebuilding the model.

`GraphRegistry` is a public trait and `Registry` carries a blanket `impl<T: GraphRegistry + 'static> From<T> for Registry`, so the whole hookup is `WasiNnCtx::new(backends, ScopedRegistry { /* … */ }.into())` — `WasiNnCtx::new` accepts any `impl IntoIterator<Item = Backend>` plus a `Registry`.

---

## 3. Configuration model (design-time vs runtime)

Follow Spin's existing split, mirroring `key_value_stores` / `llm_compute`.

**Design time — `spin.toml`** declares *which abstract models a component may use* (the governance list, in wasi-nn's own `wasi_nn_models` key; wasi-nn does not reuse the LLM factor's `ai_models`, see §9.4):

```toml
[component.classifier]
source = "vision.wasm"
wasi_nn_models = ["resnet50"]      # wasi-nn's own key, not the LLM factor's ai_models
wasi_nn_encodings = ["onnx"]       # optional: may also load its own ONNX bytes (§9.6)
```

Environment definitions carry `[configuration]` constraints (`key_value_stores`, `sqlite_databases`, `ai_models`) that `spin build`'s target-environment validation (`crates/environments`) checks against the manifest. wasi-nn's keys are not in that list yet; Phase 2 adds `wasi_nn_models` and `wasi_nn_encodings` to `ConfigurationConstraints` with one `validate_only_permitted` call each (§9.6). The keys stay separate from `ai_models` because the LLM factor is being evaluated for a rewrite: wasi-nn must not depend on its keys or types.

**Runtime — `runtime-config.toml`** maps each abstract model name to a concrete backend, location, and execution target (the decoupling point):

```toml
[wasi_nn_model.resnet50]
type = "onnx"                 # onnx | openvino | pytorch  (see §9.5)
path = "/models/resnet50.onnx"
target = "gpu"                # cpu | gpu | tpu
```

The guest stays portable: it only ever calls `load-by-name("resnet50")`. The operator decides ONNX-on-CPU here, OpenVINO-on-GPU there.

> Reuse `spin_factors::runtime_config::toml` (as `factor-llm`'s `spin.rs` does) for parsing. The runtime-config wiring point already exists after Phase 1: tutorial step 4a adds a `FactorRuntimeConfigSource<WasiNnFactor>` impl in `crates/runtime-config` returning `Ok(None)`; Phase 2 upgrades it to parse `[wasi_nn_model.*]`.
>
> `crates/capabilities` is a separate concern: it controls what composed component dependencies may inherit, not the direct factor authorization above. Supporting `wasi:nn` there is more than a one-line constant edit. It requires a new `wasi_nn_models` entry in `CAPABILITY_SETS` listing all four versioned `wasi:nn` interfaces (the LLM factor's `AI_MODELS` stays untouched), exporting and implementing them in `deny-adapter`, then rebuilding the checked-in adapter with `make -C crates/capabilities adapter`. The deny adapter matches imports by the interface they implement, including named imports such as `primary (implements …)`, and it will also plug a semver-compatible adapter export. It still cannot express per-function or per-encoding policy, because `wasi:nn/graph` holds both `load` and `load-by-name`.

---

## 4. Solved gaps

### 4.1 Model-load cold-start elimination (caching)
Models load into `AppState` during app configuration; instances receive cheap `Graph` clones. In Spin's current executor, a 2 GB ONNX graph is parsed/optimized once per loaded app, not once per request. This is the primary win and the main reason Phase 2 exists.

### 4.2 Access governance (tenancy)
`load-by-name` is gated by the per-component `wasi_nn_models` allow-list via `ScopedRegistry`. Unauthorized lookups return `not-found`. Raw `graph::load(bytes, ...)` is available only for encodings listed in that component's `wasi_nn_encodings` (§9.6). This parallels `factor-llm`'s host-side model checks. The `capabilities` deny adapter is an additional integration for composed dependencies, not the primary enforcement point.

### 4.3 Backend / hardware decoupling
`runtime-config` selects backend + `execution-target` per model. The same guest binary runs on an Intel CPU node (OpenVINO) or an NVIDIA node (ONNX+CUDA) with no recompilation.

---

## 5. Optional / future concepts (YAGNI until needed)

- **Raw `load(bytes)` compatibility.** Raw loading is a declared per-component capability, deny by default (§9.6).
- **Concurrency throttling / batching.** Buffer inbound tensors at the host, assemble hardware-efficient batches, scatter results back. Meaningful only under heavy concurrent load and best revisited once `wasi-nn`'s `compute` gains an async form (the wasi-nn WIT is still synchronous 0.2-style; the spec has not been rebased onto WASI 0.3). Spin runs on Wasmtime 49 with `wasmtime-wasi`'s `p3` feature (WASI 0.3.0 went final 2026-06-11) and has a `wasi-async` crate, so the host-side async groundwork exists.
- **Lazy / LRU model loading.** If preloading every declared model at boot is too much memory, load on first `load-by-name` and cache with an eviction policy.

---

## 6. Component boundaries & what changes from Phase 1

| Unit | New / changed | Purpose |
| :-- | :-- | :-- |
| `WasiNnFactor::configure_app` | new logic | Parse config, load+cache graphs, build allow-list map |
| `AppState` | new type (was `()`) | Hold shared `Arc` graph registry + per-component allow-lists |
| `ScopedRegistry` | new, small type | Read-only `impl GraphRegistry` over shared graphs, filtered by allow-list |
| `RuntimeConfig` + TOML types | new | `[wasi_nn_model.<name>]` → type/path/target |
| `prepare` | changed | Build cheap per-instance ctx pointing at the scoped registry, with only declared encodings |
| `init` | unchanged | Still just `wit::add_to_linker` |
| `capabilities` + deny adapter | follow-up | Govern `wasi:nn` inheritance for composed dependencies and rebuild the adapter Wasm |

Everything else (the linker wiring, the WIT vendoring, the `TriggerFactors` registration, the feature gating) is inherited from Phase 1 unchanged.

---

## 7. Risks & open questions

- **`WasiNnCtx` ownership vs sharing.** Per-instance ctx is required (the `WasiNnView` borrow model), so sharing happens at the `Arc<Graph>` layer, not the ctx layer. `ScopedRegistry` is the seam that makes this clean.
- **`Registry` from `preload` is not shareable.** We do **not** reuse the `Box<dyn GraphRegistry>` that `preload` returns; we implement our own `Arc`-backed registry instead. Also, `preload` always loads directory models with `ExecutionTarget::Cpu`, so target-aware Phase 2 should use `backend::list()` plus `BackendFromDir::load_from_dir(path, target)` during app configuration. The blanket `From<T: GraphRegistry>` impl on `Registry` is what makes the custom registry drop-in.
- **Backend lifetime.** A loaded `Graph` is self-contained (holds its own session), so app-scoped backends used only for loading need not be kept alive for inference. For components that declare `wasi_nn_encodings`, re-creating cheap `backend::list()` defaults per instance is fine for the raw-`load` path.
- **Error surface.** The codes a guest sees are fixed by `wasmtime-wasi-nn`. §9.3 records them: `not-found` covers both unknown and disallowed names.
- **Synchronous app loading.** `configure_app` is synchronous, so eager model loading adds startup latency. The current executor amortizes that work across all instances of the loaded app; if startup blocking becomes unacceptable, move to lazy app-state loading rather than doing the work per instance.
- **Memory pressure.** Eager preload of all declared models can be large; see the lazy/LRU option in §5.
- **`backend` module stability.** Keep depending on `wasmtime-wasi-nn` for both layers for now — the host code (`wit::add_to_linker`, `WasiNnCtx`, `WasiNnView`) and the `backend` module (ONNX Runtime via `ort` including the CUDA execution provider, OpenVINO, PyTorch, WinML) — with the forced version lockstep with `wasmtime` and `wiggle`, `default-features = false`, and the ONNX backend in the default feature set (the two musl release jobs opt out because `ort` ships no musl binaries; tutorial §4). The `backend` module is `pub` and used by the Wasmtime CLI, but it is not documented as a stability boundary, so a future Wasmtime train could reshape it.

---

## 8. Suggested increments

1. **Declared capability first.** Add the `wasi_nn_models` and `wasi_nn_encodings` manifest keys (`crates/manifest`, JSON schema), the matching `ConfigurationConstraints` fields in `crates/environments`, and make the factor's `prepare` install only the backends the component declared (§9.6). Until this lands, Phase 1 grants raw `load` to every component and must not be merged.
2. `AppState` + `RuntimeConfig` + TOML parsing; load app-scoped graphs; keep raw `load` for declared encodings. (Caching.)
3. `ScopedRegistry` + `wasi_nn_models` enforcement on `load-by-name`. (Governance; denials surface as the crate's `not-found`, see §9.3.)
4. Runtime-config backend/target selection per model, including the §9.2 `pool_size` knob. (Decoupling; table name `[wasi_nn_model.<name>]`, §9.5.)
5. Deny-adapter integration for composed dependencies.
6. **Deferred:** Spin-owned async bindings (§9.1) if blocking or error clarity becomes a problem in practice.

---

## 9. Design decisions

Phase 2 keeps `wasmtime-wasi-nn`'s host code (§9.1 records the alternative) and makes raw `load` a declared per-component capability (§9.6). §§1–8 are that plan.

### 9.1 Alternative (deferred): Spin owns the host code

`wasmtime-wasi-nn` generates synchronous host functions (`imports: { default: trappable }`), so on Spin's async engine each in-flight `load` or `compute` occupies a Tokio worker thread for its whole duration, and the error codes a guest sees are fixed by the crate. The same integration could instead be built the way every other Spin factor is: generate Spin's own async bindings from the vendored WIT (an `include wasi:nn/ml@0.2.0-rc-2024-10-28;` in the inline world of `crates/world/src/lib.rs`), implement the four `Host` traits in the factor with the backend work in `tokio::task::spawn_blocking`, and keep `wasmtime-wasi-nn` only for its `backend` module. That shape would also let policy denials return a clear error, allow byte caps on tensors and model builders, and remove the need for `WasiNnCtx`, `WasiNnView`, and `ScopedRegistry`, since `load-by-name` becomes an allow-list check plus an `Arc` clone. Keep the crate's host code for now; revisit if worker blocking or error clarity becomes a problem in practice.

Spin's engine already enables the async component model (`wasm_component_model_async(true)` in `crates/core`). Wasmtime 49's `Config::async_support` is a deprecated no-op marked "no longer has any effect", so async support is unconditional.

### 9.2 A shared ONNX `Graph` serialises inference

`OnnxGraph` is `Arc<Mutex<ort::Session>>`; every execution context created from it clones that `Arc`, and `compute` holds the lock across `session.run` (`src/backend/onnx.rs`). PyTorch has the same shape (`Arc<Mutex<tch::CModule>>`). OpenVINO differs: it creates an `InferRequest` per execution context and calls `infer()` without holding the compiled-model lock. So the "load once, share the `Arc`" design in §1/§4.1 removes reload cost but caps concurrency per ONNX model at one inference at a time (ONNX Runtime's intra-op threads still parallelise inside that one inference). Combined with 9.1 this is worse than it looks: workers block *waiting for a mutex*, not just computing. Under `spawn_blocking` the waiting moves to the blocking pool and the async workers stay free. The scaling knob is a per-model pool of N independently loaded sessions (N× memory); make it a per-model runtime-config field and default it to 1.

### 9.3 Error surface (resolves §7's open question)

With the crate's host code the codes are fixed by `wasmtime-wasi-nn`: a name missing from the `ScopedRegistry` — unknown or simply not allowed for this component — yields `not-found`; a raw `load` for an encoding whose backend is not installed in this instance (§9.6) yields `invalid-encoding` with the message "unable to find a backend for this encoding"; backend failures yield `runtime-error`; the crate never emits `security`, `unknown`, or `too-large`. Returning `not-found` for both unknown and disallowed names means tenants cannot probe the catalogue, at the cost of a less helpful developer message, so document both cases in the troubleshooting table. (Under the §9.1 alternative, `security` would be the natural code for policy denials; the WIT defines it as "insufficient privilege" but illustrates it with hardware access, so that is a reasonable reading, not a mandate.)

### 9.4 wasi-nn owns its manifest keys

wasi-nn does not reuse the LLM factor's `ai_models` key. The LLM factor, together with the local llama engine, is being evaluated for a rewrite, so wasi-nn must not couple to its keys or types. wasi-nn declares its own keys — `wasi_nn_models` for host-registered models and `wasi_nn_encodings` for guest-supplied bytes — read through its own `MetadataKey`s, not `ALLOWED_MODELS_KEY`. The three enforcement layers still apply, each as an addition rather than an edit to LLM code: the factor enforces at runtime; `crates/environments` gains `wasi_nn_models` and `wasi_nn_encodings` constraint fields; `crates/capabilities` gains a `wasi_nn_models` set mapping to the four `wasi:nn/*@0.2.0-rc-2024-10-28` interfaces, plus the `make -C crates/capabilities adapter` rebuild. The names follow the interface (`wasi_nn_*`) rather than the concept because the concept name is taken; the neighbours (`key_value_stores`, `sqlite_databases`) are concept-named, so this is a deliberate exception. None of these names come from the wasi-nn proposal or any upstream host; they are Spin-internal choices.

### 9.5 Runtime-config key: `[wasi_nn_model.<name>]`

Existing keys are singular labelled typed tables: `[key_value_store.<label>] type = "redis"`, `[sqlite_database.<label>] type = "libsql"`, and `[llm_compute] type = "spin"`. `[wasi_nn_model.<name>] type = "onnx"` with `path`, `target`, `preload`, and `pool_size` follows that shape, pairs with the manifest key `wasi_nn_models` the way `key_value_store` pairs with `key_value_stores`, and slots straight into `ResolvedRuntimeConfig::summarize`'s `summarize_labeled_typed_tables` in `crates/runtime-config`. Neither the wasi-nn proposal nor any upstream host defines a config format, so this is a Spin-consistency choice. §3's example uses this name.

### 9.6 Guest-supplied model bytes are a declared per-component capability

**Decision.** Raw `graph::load(bytes, encoding, target)` is denied unless the component declares the capability in `spin.toml`. This follows Spin's convention for every host resource (outbound hosts, files, key-value stores, SQLite, `ai_models`, environment, variables): the developer declares, the factor enforces at runtime, an environment definition can constrain at build time, and the capabilities crate uses the same vocabulary for composition inheritance. Raw loading is wasi-nn's primary, original API and the spec permits both allowing and refusing it. Spin applies the same deny-unless-declared rule it uses for other host resources.

**Why it is a resource and not a plain function call.** The bytes are parsed by native backends outside the Wasm sandbox (Wasmtime never merged TensorFlow because its operators can reach files and the network; spec issue wasi-nn#88 raises the same point for other backends); a native inference cannot be interrupted by Spin's epoch mechanism; a loaded session consumes host memory and possibly a GPU outside the instance's linear-memory limits; and some formats reference external files by path (ONNX external data; WasmEdge's Piper backend takes host paths from a guest config blob), so each backend's handling of in-memory models needs checking before untrusted guests get the path.

**Precedent.** Every existing wasi-nn host allows raw `load` by default and none has a named-only mode: the Wasmtime CLI once `-S nn` is on (`-S nn-graph` only adds named graphs), the WasmEdge plugin (registers `load`, `load_by_name`, `load_by_name_with_config` unconditionally and implements named preloads as `preload:<path>` bytes through the same `load`), WAMR (whose `load_by_name` is a raw host path), and the unmaintained third-party `Iceber/wasmcloud-wasi-nn` prototype. wasmCloud itself never shipped wasi-nn (Q1-2025 roadmap unfulfilled; a July-2026 community call pivoted to a chat-style API with operator-resolved model artifacts). None of those hosts is multi-tenant and none records a policy decision; raw `load` is the 2020 API surface, and `load-by-name` was added in 2023 (wasi-nn#36, PR #38) for model size and cross-instance reuse, explicitly not for trust. Spin departs from implementer precedent here, on the strength of its own capability convention. `wasmtime-wasi-nn`'s clean separation of backends from the named registry (`WasiNnCtx::new(backends, registry)`) is the property to preserve; WasmEdge's conflation of the two paths is what to avoid.

**Design.**

1. **Manifest (developer declares).** A new component key, list-shaped like its siblings and scoped by encoding because that is what the capability really grants ("this component may feed bytes to the ONNX parser"): `wasi_nn_encodings = ["onnx"]`, default empty (alongside `wasi_nn_models`, §9.4). Per-encoding granularity lets an environment permit ONNX but not PyTorch. Touches `crates/manifest/src/schema/v2/component.rs` and the JSON schema.
2. **Factor enforces at runtime.** `configure_app` reads the key per component (through wasi-nn's own `MetadataKey`, alongside the one for `wasi_nn_models`) and stores `component_allowed_encodings`; `prepare` copies the component's set into `InstanceState`; `load` checks `encoding ∈ allowed_encodings`. With the crate's host code `prepare` installs only the backends whose encodings the component declared (`backend::list()` filtered by `encoding()`), so an undeclared encoding surfaces as the crate's `invalid-encoding` error, message "unable to find a backend for this encoding"; the troubleshooting table must list that message with the manifest fix. Under the §9.1 alternative the violation would instead return `security` with a message naming the manifest key, mirroring `factor-llm`'s `access_denied_error`.
3. **Environment validates at build time.** Add `wasi_nn_models: Option<Vec<String>>` and `wasi_nn_encodings: Option<Vec<String>>` to `ConfigurationConstraints` in `crates/environments/src/environment/definition.rs` and one `validate_only_permitted` call each; a hosted environment sets it to `[]` to refuse guest-supplied models before deploy, a dev environment omits it.
4. **Composition inheritance.** The deny adapter works per interface (matching the interface a named import implements, and semver-compatible export names) and `wasi:nn/graph` holds both `load` and `load-by-name`, so it cannot express per-function or per-encoding policy. A new `wasi_nn_models` capability set governs inheritance of all four `wasi:nn` interfaces; the encoding check stays a runtime check in the factor.
5. **Operator veto (optional).** A runtime-config hard switch in a separate policy table, such as `[wasi_nn] allow_guest_models = false` (kept out of the `[wasi_nn_model.*]` namespace so that table stays purely per-model), for self-hosters without an environment catalogue. Secondary; the manifest declaration is the primary mechanism.
6. **Error mapping** as in §9.3. Byte caps on tensors and model builders are not available with the crate's host code (it applies none); they come with the §9.1 alternative.

**Consequence for Phase 1.** Phase 1 as written allows raw `load` for every component because it is the only path that works there. That is acceptable for the spike but means Phase 1 must not be merged upstream as-is; increment 1 in §8 introduces the declared capability before anything ships.

---

## 10. Path to upstream

The goal is to contribute this to upstream Spin. `GOVERNANCE.md` requires a SIP for substantial changes, so the vocabulary above is written up as **SIP 026** at `docs/content/sips/026-wasi-nn.md`. The SIP index lists only accepted SIPs (through 025), so 026 is not listed there until it is accepted. Consequences for the plan: Phase 1 is never a PR on its own, because it grants raw `load` to every component; the first code PR is §8 increment 1. The SIP covers the whole vocabulary — `wasi_nn_models`, `wasi_nn_encodings`, `[wasi_nn_model.<name>]`, the environment constraints, the capabilities set, and ONNX Runtime in the default build — even though code lands incrementally. Expected review flashpoints, each answered in the SIP from this document: denying raw `load` by default (§9.6), ONNX Runtime in the default build with the musl opt-out (tutorial §4), and synchronous host calls (§9.1). One number the SIP still needs is the binary-size delta from statically linking ONNX Runtime; measure it with the first build.
