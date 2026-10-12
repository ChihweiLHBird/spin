use spin_factors::{
    ConfigureAppContext, Factor, InitContext, PrepareContext, RuntimeFactors,
    SelfInstanceBuilder,
};
use wasmtime::component::Linker;
use wasmtime_wasi_nn::preload;
use wasmtime_wasi_nn::wit::{WasiNnCtx, WasiNnView};

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
        ctx.link_nn_bindings(wasmtime_wasi_nn::wit::add_to_linker)?;
        Ok(())
    }

    fn configure_app<T: RuntimeFactors>(
        &self,
        _ctx: ConfigureAppContext<T, Self>,
    ) -> anyhow::Result<Self::AppState>
    {
        Ok(())
    }

    fn prepare<T: RuntimeFactors>(
        &self,
        _ctx: PrepareContext<T, Self>,
    ) -> anyhow::Result<Self::InstanceBuilder>
    {
        let (backends, registry) = preload(&[])?;
        Ok(InstanceState {
            ctx: WasiNnCtx::new(backends, registry),
        })
    }
}

pub struct InstanceState {
    ctx: WasiNnCtx,
}

impl SelfInstanceBuilder for InstanceState {}

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
