use burn::prelude::*;

use crate::{
    attention::{Attention, AttentionConfig},
    config::TransformerConfig,
    layernorm::{LayerNorm, LayerNormConfig},
    mlp::{MLP, MLPConfig},
};

pub struct TransformerBlockConfig {
    attn_cfg: AttentionConfig,
    ln_cfg: LayerNormConfig,
    mlp_cfg: MLPConfig,
}

impl From<TransformerConfig> for TransformerBlockConfig {
    fn from(cfg: TransformerConfig) -> Self {
        Self {
            attn_cfg: cfg.clone().into(),
            ln_cfg: cfg.clone().into(),
            mlp_cfg: cfg.into(),
        }
    }
}

pub struct TransformerBlock<B: Backend> {
    ln1: LayerNorm<B>,
    attn: Attention<B>,
    ln2: LayerNorm<B>,
    mlp: MLP<B>,
}

impl<B: Backend> TransformerBlock<B> {
    pub fn init(cfg: &TransformerBlockConfig, device: &B::Device) -> Self {
        Self {
            ln1: cfg.ln_cfg.init(device),
            attn: cfg.attn_cfg.init(device),
            ln2: cfg.ln_cfg.init(device),
            mlp: cfg.mlp_cfg.init(device),
        }
    }

    /// (batch pos d_model) -> (batch pos d_model)
    pub fn forward(&self, resid_pre: Tensor<B, 3>) -> Tensor<B, 3> {
        let post_attention =
            self.attn.forward(self.ln1.forward(resid_pre.clone())) + resid_pre.clone();
        self.mlp.forward(self.ln2.forward(post_attention)) + resid_pre
    }
}
