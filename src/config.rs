use burn::prelude::{Config as BurnConfig, *};

use crate::layernorm::LayerNormConfig;

#[derive(Debug, BurnConfig)]
pub struct Config {
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
