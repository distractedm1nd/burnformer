#![allow(dead_code)]
use std::sync::Arc;

use burn::backend::libtorch::LibTorchDevice;
use burn::backend::{Autodiff, LibTorch};
use burn::data::dataloader::batcher::Batcher;
use burn::data::dataloader::{DataLoader, DataLoaderBuilder};
use burn::data::dataset::SqliteDataset;
use burn::nn::loss::CrossEntropyLossConfig;
use burn::optim::AdamWConfig;
use burn::record::CompactRecorder;
use burn::tensor::Tensor;
use burn::tensor::activation::log_softmax;
use burn::tensor::backend::AutodiffBackend;
use burn::train::metric::{AccuracyMetric, LossMetric};
use burn::train::{
    ClassificationOutput, InferenceStep, Learner, RegressionOutput, SupervisedTraining,
    TrainOutput, TrainStep,
};
use burn::{backend, optim, prelude::*};
use burn::{
    data::dataset::{Dataset, HuggingfaceDatasetLoader, transform::SamplerDataset},
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
pub struct TinyStoriesBatcher {
    tokenizer: Tokenizer,
    n_ctx: usize,
}

#[derive(Debug, Clone)]
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

fn create_artifact_dir(artifact_dir: &str) {
    // Remove existing artifacts before to get an accurate learner summary
    std::fs::remove_dir_all(artifact_dir).ok();
    std::fs::create_dir_all(artifact_dir).ok();
}

impl<B: AutodiffBackend> TrainStep for DemoTransformer<B> {
    type Input = TinyStoryBatch<B>;
    type Output = ClassificationOutput<B>;

    fn step(&self, batch: Self::Input) -> TrainOutput<Self::Output> {
        let out = self.forward_classification(batch.tokens);
        TrainOutput::new(self, out.loss.backward(), out)
    }
}

impl<B: Backend> InferenceStep for DemoTransformer<B> {
    type Input = TinyStoryBatch<B>;
    type Output = ClassificationOutput<B>;

    fn step(&self, item: Self::Input) -> Self::Output {
        self.forward_classification(item.tokens)
    }
}

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
    // 1000
    pub eval_examples: usize,
    // 1e-3
    pub learning_rate: f64,
    pub seed: u64,
}

pub fn train<B: AutodiffBackend>(artifact_dir: &str, config: TrainingConfig, device: B::Device) {
    create_artifact_dir(artifact_dir);
    // config
    //     .save(format!("{artifact_dir}/config.json"))
    //     .expect("Config should be saved successfully");

    B::seed(&device, config.seed);

    let tokenizer = Tokenizer::from_pretrained("bert-base-cased", None).unwrap();

    let batcher = TinyStoriesBatcher {
        tokenizer,
        n_ctx: config.model_cfg.n_ctx(),
    };

    let train: SqliteDataset<TinyStory> = HuggingfaceDatasetLoader::new("roneneldan/TinyStories")
        .dataset("train")
        .unwrap();
    let test: SqliteDataset<TinyStory> = HuggingfaceDatasetLoader::new("roneneldan/TinyStories")
        .dataset("validation")
        .unwrap();
    let train_examples_per_epoch = config
        .batch_size
        .saturating_mul(config.max_steps_per_epoch)
        .min(train.len());
    let eval_examples = config.eval_examples.min(test.len());
    let train = SamplerDataset::without_replacement(train, train_examples_per_epoch);
    let test = SamplerDataset::without_replacement(test, eval_examples);

    let train_loader: Arc<dyn DataLoader<B, TinyStoryBatch<B>>> =
        DataLoaderBuilder::new(batcher.clone())
            .batch_size(config.batch_size)
            .num_workers(16)
            .shuffle(config.seed)
            .build(train);
    let test_loader = DataLoaderBuilder::new(batcher)
        .batch_size(config.batch_size)
        .num_workers(16)
        .shuffle(config.seed)
        .build(test);

    let training = SupervisedTraining::new(artifact_dir, train_loader, test_loader)
        .metrics((AccuracyMetric::new(), LossMetric::new()))
        .with_file_checkpointer(CompactRecorder::new())
        .num_epochs(config.num_epochs)
        .summary();

    let model = DemoTransformer::init(&config.model_cfg, &device);
    let result = training.launch(Learner::new(
        model,
        config.optim.init(),
        config.learning_rate,
    ));

    result
        .model
        .save_file(format!("{artifact_dir}/model"), &CompactRecorder::new())
        .expect("Trained model should be saved successfully");
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
    train::<MyAutodiff>(
        artifact_dir,
        TrainingConfig {
            model_cfg: cfg,
            num_epochs: 10,
            batch_size: 32,
            max_steps_per_epoch: 500,
            eval_examples: 1000,
            optim: AdamWConfig::new(),
            learning_rate: 1e-3,
            seed: 42,
        },
        device.clone(),
    );

    Ok(())
}
