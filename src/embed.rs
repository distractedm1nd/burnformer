use burn::{module::Param, nn::Initializer, prelude::*};

use crate::config::TransformerConfig;

pub struct EmbedConfig {
    d_vocab: usize,
    d_model: usize,
    n_ctx: usize,
    init_range: f64,
}

impl From<TransformerConfig> for EmbedConfig {
    fn from(value: TransformerConfig) -> Self {
        Self {
            d_vocab: value.d_vocab(),
            d_model: value.d_model(),
            n_ctx: value.n_ctx(),
            init_range: value.init_range(),
        }
    }
}

impl EmbedConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> Embed<B> {
        Embed::init(self, device)
    }
}

#[derive(Debug, Module)]
struct Embed<B: Backend> {
    w_e: Param<Tensor<B, 2>>,
}

impl<B: Backend> Embed<B> {
    pub fn init(cfg: &EmbedConfig, device: &B::Device) -> Self {
        let w_e = Initializer::Normal {
            mean: 0.0,
            std: cfg.init_range,
        }
        .init([cfg.d_vocab, cfg.d_model], device);
        Self { w_e }
    }

    /// [`tokens`] has shape (batch pos)
    /// return value has shape (batch pos d_model)
    pub fn forward(&self, tokens: Tensor<B, 2, Int>) -> Tensor<B, 3> {
        let shape = tokens.shape();
        let (batch, pos) = (shape[0], shape[1]);
        let d_model = self.w_e.val().shape()[1];

        self.w_e
            .val()
            .select(0, tokens.reshape([batch * pos]))
            .reshape([batch, pos, d_model])
    }
}

#[derive(Debug, Module)]
pub struct PosEmbed<B: Backend> {
    w_pos: Param<Tensor<B, 2>>,
}

impl<B: Backend> PosEmbed<B> {
    pub fn init(&self, cfg: EmbedConfig, device: &B::Device) -> Self {
        let w_pos = Initializer::Normal {
            mean: 0.0,
            std: cfg.init_range,
        }
        .init([cfg.n_ctx, cfg.d_model], device);
        Self { w_pos }
    }

    /// [`tokens`] has shape (batch pos)
    /// return value has shape (batch pos d_model)
    pub fn forward(&self, tokens: Tensor<B, 2>) -> Tensor<B, 3> {
        let shape = tokens.shape();
        let (batch, seq_length) = (shape[0], shape[1]);

        self.w_pos
            .val()
            .slice([0..seq_length])
            .unsqueeze::<3>()
            .repeat_dim(0, batch)
    }
}
