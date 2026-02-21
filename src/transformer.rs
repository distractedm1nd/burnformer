use burn::prelude::*;

use crate::{attention::Attention, layernorm::LayerNorm, mlp::MLP};

pub struct TransformerBlock<B: Backend> {
    ln1: LayerNorm<B>,
    attn: Attention<B>,
    ln2: LayerNorm<B>,
    mlp: MLP<B>,
}

impl<B: Backend> TransformerBlock<B> {
    /// (batch pos d_model) -> (batch pos d_model)
    pub fn forward(&self, resid_pre: Tensor<B, 3>) -> Tensor<B, 3> {
        let post_attention =
            self.attn.forward(self.ln1.forward(resid_pre.clone())) + resid_pre.clone();
        self.mlp.forward(self.ln2.forward(post_attention)) + resid_pre
    }
}
