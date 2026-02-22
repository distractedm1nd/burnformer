#![allow(dead_code)]

use burn::backend::libtorch::LibTorchDevice;
use burn::backend::{Autodiff, LibTorch};
use burn::module::Module;
use burn::nn::loss::CrossEntropyLossConfig;
use burn::optim::AdamWConfig;
use burn::prelude::*;
use burn::tensor::Tensor;
use burn::tensor::activation::log_softmax;
use burn::train::ClassificationOutput;
use serde::{Deserialize, Serialize};
use tokenizers::tokenizer::Result;

use crate::train::TrainingConfig;
use crate::{
    config::TransformerConfig,
    embed::{Embed, EmbedConfig, PosEmbed, Unembed},
    layernorm::{LayerNorm, LayerNormConfig},
    transformer::{TransformerBlock, TransformerBlockConfig},
};

mod attention;
mod config;
mod embed;
mod inference;
mod layernorm;
mod mlp;
mod train;
mod transformer;

#[derive(Debug, Module)]
pub struct DemoTransformer<B: Backend> {
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

    pub fn forward_classification(&self, tokens: Tensor<B, 2, Int>) -> ClassificationOutput<B> {
        let [batch, pos] = tokens.dims();
        let logits = self.forward(tokens.clone()); // [batch, posn, d_vocab]
        let [_, _, d_vocab] = logits.dims();

        let logits_sliced = logits.slice([0..batch, 0..pos - 1]);

        // ~ext tokens (batch, pos-1), then flatten to (batch * pos-1)
        let targets_shifted = tokens
            .slice([0..batch, 1..pos])
            .reshape([batch * (pos - 1)]);

        // flatten logits to (batch * pos-1, d_vocab) for CE
        let logits_flat = logits_sliced.reshape([batch * (pos - 1), d_vocab]);

        let loss = CrossEntropyLossConfig::new()
            .init(&logits_flat.device())
            .forward(logits_flat.clone(), targets_shifted.clone());

        ClassificationOutput {
            loss,
            output: logits_flat,
            targets: targets_shifted,
        }
    }
}

#[derive(Debug, Clone)]
pub struct TinyStoryBatch<B: Backend> {
    // (batch pos)
    pub tokens: Tensor<B, 2, Int>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TinyStory {
    pub text: String,
}

// (batch pos d_vocab) -> (batch pos) -> (batch pos-1)
pub fn get_log_probs<B: Backend>(logits: Tensor<B, 3>, tokens: Tensor<B, 2, Int>) -> Tensor<B, 2> {
    let log_probs = log_softmax(logits, 2); // (batch, posn, d_vocab)
    let [batch, posn, _d_vocab] = log_probs.dims();
    let log_probs_sliced = log_probs.slice([0..batch, 0..posn - 1]);
    let next_tokens = tokens.slice([0..batch, 1..posn]);
    let index = next_tokens.unsqueeze_dim(2); // (batch, posn-1, 1)
    let gathered = log_probs_sliced.gather(2, index.float().int()); // (batch, posn-1, 1)
    gathered.squeeze()
}

fn main() -> Result<()> {
    type MyBackend = LibTorch;
    type MyAutodiff = Autodiff<LibTorch>;

    let device = LibTorchDevice::Cuda(0);
    let cfg = TransformerConfig::new()
        .with_d_model(32)
        .with_n_head(16)
        .with_d_head(2)
        .with_d_mlp(32 * 4)
        .with_n_layers(4)
        .with_n_ctx(128);

    let artifact_dir = "./artifacts/";
    train::train::<MyAutodiff>(
        artifact_dir,
        TrainingConfig {
            model_cfg: cfg,
            num_epochs: 10,
            batch_size: 4,
            max_steps_per_epoch: 500,
            eval_examples: 1000,
            optim: AdamWConfig::new(),
            learning_rate: 1e-3,
            seed: 42,
        },
        device,
    );

    Ok(())
}
