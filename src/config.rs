use burn::prelude::Config as BurnConfig;

#[derive(Debug, BurnConfig)]
pub struct TransformerConfig {
    #[config(default = 768)]
    d_model: usize,
    #[config(default = 12)]
    n_layers: usize,

    #[config(default = 12)]
    n_head: usize,
    #[config(default = 64)]
    d_head: usize,

    #[config(default = 50257)]
    d_vocab: usize,
    #[config(default = 1024)]
    n_ctx: usize,

    #[config(default = 3072)]
    d_mlp: usize,

    #[config(default = 1e-5)]
    layernorm_eps: f32,
    #[config(default = 0.02)]
    init_range: f64,
}

impl TransformerConfig {
    pub fn d_model(&self) -> usize {
        self.d_model
    }

    pub fn n_layers(&self) -> usize {
        self.n_layers
    }

    pub fn n_head(&self) -> usize {
        self.n_head
    }

    pub fn d_head(&self) -> usize {
        self.d_head
    }

    pub fn d_vocab(&self) -> usize {
        self.d_vocab
    }

    pub fn n_ctx(&self) -> usize {
        self.n_ctx
    }

    pub fn d_mlp(&self) -> usize {
        self.d_mlp
    }

    pub fn layernorm_eps(&self) -> f32 {
        self.layernorm_eps
    }

    pub fn init_range(&self) -> f64 {
        self.init_range
    }
}
