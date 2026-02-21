#![allow(dead_code)]

use burn::module::Module;
use burn::prelude::*;
use burn::tensor::Tensor;

use crate::{
    config::TransformerConfig,
    embed::{Embed, EmbedConfig, PosEmbed, Unembed},
    layernorm::{LayerNorm, LayerNormConfig},
    transformer::{TransformerBlock, TransformerBlockConfig},
};

mod attention;
mod config;
mod embed;
mod layernorm;
mod mlp;
mod transformer;

#[derive(Debug, Module)]
struct DemoTransformer<B: Backend> {
    embed: Embed<B>,
    pos_embed: PosEmbed<B>,
    blocks: Vec<TransformerBlock<B>>,
    ln_final: LayerNorm<B>,
    unembed: Unembed<B>,
}

impl<B: Backend> DemoTransformer<B> {
    pub fn init(cfg: &TransformerConfig, device: &B::Device) -> Self {
        let embed_cfg: EmbedConfig = cfg.clone().into();
        let tb_cfg: TransformerBlockConfig = cfg.clone().into();
        let mut blocks = Vec::with_capacity(cfg.n_layers());
        for _ in 0..cfg.n_layers() {
            blocks.push(tb_cfg.init(device))
        }
        Self {
            embed: embed_cfg.init(device),
            pos_embed: embed_cfg.init_pos(device),
            blocks,
            ln_final: LayerNormConfig::from(cfg.clone()).init(device),
            unembed: embed_cfg.init_unembed(device),
        }
    }

    /// (batch pos) -> (batch pos d_vocab)
    pub fn forward(&self, tokens: Tensor<B, 2, Int>) -> Tensor<B, 3> {
        let embed = self.embed.forward(tokens.clone());
        let pos = self.pos_embed.forward(tokens);
        let mut residual = embed + pos;

        for block in &self.blocks {
            residual = block.forward(residual)
        }

        residual = self.ln_final.forward(residual);
        self.unembed.forward(residual)
    }
}

fn main() {}
