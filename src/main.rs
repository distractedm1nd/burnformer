#![allow(dead_code)]
use burn::backend::LibTorch;
use burn::backend::libtorch::LibTorchDevice;
use burn::data::dataloader::batcher::Batcher;
use burn::data::dataset::SqliteDataset;
use burn::tensor::Tensor;
use burn::tensor::backend::AutodiffBackend;
use burn::train::{InferenceStep, TrainStep};
use burn::{backend, optim, prelude::*};
use burn::{
    data::dataset::{Dataset, HuggingfaceDatasetLoader},
    module::Module,
};
use serde::{Deserialize, Serialize};
use tokenizers::Token;
use tokenizers::tokenizer::{Result, Tokenizer};

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

#[derive(Clone)]
pub struct TinyStoriesBatcher {
    tokenizer: Tokenizer,
    n_ctx: usize,
}

pub struct TinyStoryBatch<B: Backend> {
    // (batch pos)
    pub tokens: Tensor<B, 2, Int>,
}

impl<B: Backend> Batcher<B, TinyStory, TinyStoryBatch<B>> for TinyStoriesBatcher {
    fn batch(&self, items: Vec<TinyStory>, device: &<B as Backend>::Device) -> TinyStoryBatch<B> {
        let tokens: Vec<u32> = items
            .iter()
            .flat_map(|story| {
                self.tokenizer
                    .encode(story.text.clone(), true)
                    .unwrap()
                    .get_ids()
                    .to_vec()
            })
            .collect();

        let mut chunks = tokens.chunks_exact(self.n_ctx);
        let batch: Vec<_> = chunks
            .by_ref()
            .map(TensorData::from)
            .map(|data| Tensor::<B, 1, Int>::from_data(data, device))
            .collect();

        assert!(
            !batch.is_empty(),
            "Batch contained fewer tokens than n_ctx. Reduce n_ctx or increase batch_size."
        );

        let token_batch = Tensor::stack::<2>(batch, 0);

        TinyStoryBatch {
            tokens: token_batch,
        }
    }
}

// impl<B: AutodiffBackend> TrainStep for DemoTransformer<B> {
//     type Input: ;
//     type Output;

//     fn step(&self, item: Self::Input) -> burn::train::TrainOutput<Self::Output> {
//         todo!()
//     }
// }
// impl<B: AutodiffBackend> InferenceStep for DemoTransformer<B> {}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TinyStory {
    pub text: String,
}

#[derive(Clone)]
pub struct TrainingConfig {
    pub model_cfg: TransformerConfig,
    pub optim: optim::AdamWConfig,
    // 10
    pub num_epochs: usize,
    // 32
    pub batch_size: usize,
    // 500
    pub max_steps_per_epoch: usize,
    pub seed: u64,
}

// pub fn train<B: AutodiffBackend>(config: TrainingConfig, device: B::Device) {}

type MyBackend = LibTorch;

fn main() -> Result<()> {
    let device = LibTorchDevice::Cuda(0);
    let cfg = TransformerConfig::new()
        .with_d_model(32)
        .with_n_head(16)
        .with_d_head(2)
        .with_d_mlp(32 * 4)
        .with_n_layers(4)
        .with_n_ctx(128);

    let tokenizer = Tokenizer::from_pretrained("bert-base-cased", None)?;

    let train: SqliteDataset<TinyStory> = HuggingfaceDatasetLoader::new("roneneldan/TinyStories")
        .dataset("train")
        .unwrap();

    // let test: SqliteDataset<TinyStory> = HuggingfaceDatasetLoader::new("roneneldan/TinyStories")
    //     .dataset("test")
    //     .unwrap();

    let batcher = TinyStoriesBatcher {
        tokenizer,
        n_ctx: cfg.n_ctx(),
    };

    let b1 = train.iter().take(32).collect();
    let b: TinyStoryBatch<MyBackend> = batcher.batch(b1, &device);

    println!("{}", train.get(0).unwrap().text);
    println!("{:?}", b.tokens);
    Ok(())
}
