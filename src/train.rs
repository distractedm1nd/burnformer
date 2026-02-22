#![allow(dead_code)]
use std::sync::Arc;

use burn::data::dataloader::batcher::Batcher;
use burn::data::dataloader::{DataLoader, DataLoaderBuilder};
use burn::data::dataset::SqliteDataset;
use burn::record::CompactRecorder;
use burn::tensor::Tensor;
use burn::tensor::backend::AutodiffBackend;
use burn::train::metric::{AccuracyMetric, LossMetric};
use burn::train::{ClassificationOutput, Learner, SupervisedTraining, TrainOutput, TrainStep};
use burn::{
    data::dataset::{Dataset, HuggingfaceDatasetLoader, transform::SamplerDataset},
    module::Module,
};
use burn::{optim, prelude::*};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use tokenizers::tokenizer::Tokenizer;

use crate::config::TransformerConfig;
use crate::{DemoTransformer, TinyStory, TinyStoryBatch};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TokenWindow {
    pub tokens: Vec<u32>,
}

#[derive(Clone, Debug)]
pub struct TokenWindowDataset {
    tokens: Arc<Vec<u32>>,
    n_ctx: usize,
}

impl TokenWindowDataset {
    fn new(tokens: Vec<u32>, n_ctx: usize) -> Self {
        Self {
            tokens: Arc::new(tokens),
            n_ctx,
        }
    }

    fn token_count(&self) -> usize {
        self.tokens.len()
    }
}

impl Dataset<TokenWindow> for TokenWindowDataset {
    fn get(&self, index: usize) -> Option<TokenWindow> {
        let start = index.checked_mul(self.n_ctx)?;
        let end = start.checked_add(self.n_ctx)?;
        if end > self.tokens.len() {
            return None;
        }

        Some(TokenWindow {
            tokens: self.tokens[start..end].to_vec(),
        })
    }

    fn len(&self) -> usize {
        self.tokens.len() / self.n_ctx
    }
}

#[derive(Debug, Clone)]
pub struct TinyStoriesBatcher {
    n_ctx: usize,
}

impl<B: Backend> Batcher<B, TokenWindow, TinyStoryBatch<B>> for TinyStoriesBatcher {
    fn batch(&self, items: Vec<TokenWindow>, device: &<B as Backend>::Device) -> TinyStoryBatch<B> {
        let batch: Vec<_> = items
            .into_iter()
            .map(|window| {
                assert_eq!(
                    window.tokens.len(),
                    self.n_ctx,
                    "Found pre-tokenized window with wrong sequence length."
                );
                Tensor::<B, 1, Int>::from_data(TensorData::from(window.tokens.as_slice()), device)
            })
            .collect();

        let token_batch = Tensor::stack::<2>(batch, 0);
        TinyStoryBatch {
            tokens: token_batch,
        }
    }
}

pub fn pretokenize_dataset<D: Dataset<TinyStory> + Sync>(
    dataset: &D,
    tokenizer: &Tokenizer,
    n_ctx: usize,
    split: &str,
) -> TokenWindowDataset {
    const STORIES_PER_CHUNK: usize = 512;

    let chunk_ranges: Vec<(usize, usize)> = (0..dataset.len())
        .step_by(STORIES_PER_CHUNK)
        .map(|start| {
            let end = (start + STORIES_PER_CHUNK).min(dataset.len());
            (start, end)
        })
        .collect();

    let mut token_chunks: Vec<(usize, Vec<u32>)> = chunk_ranges
        .into_par_iter()
        .map(|(start, end)| {
            let mut texts = Vec::with_capacity(end - start);
            for index in start..end {
                if let Some(story) = dataset.get(index) {
                    texts.push(story.text);
                }
            }

            let encodings = tokenizer.clone().encode_batch(texts, true).unwrap();
            let total_tokens: usize = encodings
                .iter()
                .map(|encoding| encoding.get_ids().len())
                .sum();
            let mut tokens = Vec::with_capacity(total_tokens);
            for encoding in encodings {
                tokens.extend_from_slice(encoding.get_ids());
            }

            (start, tokens)
        })
        .collect();

    token_chunks.sort_by_key(|(start, _)| *start);
    let total_tokens: usize = token_chunks.iter().map(|(_, tokens)| tokens.len()).sum();
    let usable_tokens = total_tokens - (total_tokens % n_ctx);
    let mut flat_tokens = Vec::with_capacity(usable_tokens);

    for (_, tokens) in token_chunks {
        if flat_tokens.len() >= usable_tokens {
            break;
        }
        let remaining = usable_tokens - flat_tokens.len();
        if tokens.len() <= remaining {
            flat_tokens.extend_from_slice(&tokens);
        } else {
            flat_tokens.extend_from_slice(&tokens[..remaining]);
        }
    }

    let dataset = TokenWindowDataset::new(flat_tokens, n_ctx);
    let token_mib =
        dataset.token_count() as f64 * std::mem::size_of::<u32>() as f64 / 1024.0 / 1024.0;
    println!(
        "Pretokenized {split}: {} windows ({} tokens, {:.1} MiB)",
        dataset.len(),
        dataset.token_count(),
        token_mib
    );

    dataset
}

pub fn create_artifact_dir(artifact_dir: &str) {
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

pub fn train<B: AutodiffBackend>(artifact_dir: &str, config: TrainingConfig, device: B::Device) {
    create_artifact_dir(artifact_dir);
    // config
    //     .save(format!("{artifact_dir}/config.json"))
    //     .expect("Config should be saved successfully");

    B::seed(&device, config.seed);

    let tokenizer = Tokenizer::from_pretrained("openai-community/gpt2", None).unwrap();

    let n_ctx = config.model_cfg.n_ctx();
    let batcher = TinyStoriesBatcher { n_ctx };

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
    let train = pretokenize_dataset(&train, &tokenizer, n_ctx, "train");
    let test = pretokenize_dataset(&test, &tokenizer, n_ctx, "validation");
    let num_workers = 16;

    let train_loader: Arc<dyn DataLoader<B, TinyStoryBatch<B>>> =
        DataLoaderBuilder::new(batcher.clone())
            .batch_size(config.batch_size)
            .num_workers(num_workers)
            .set_device(device.clone())
            .shuffle(config.seed)
            .build(train);
    let test_loader = DataLoaderBuilder::new(batcher)
        .batch_size(config.batch_size)
        .num_workers(num_workers)
        .set_device(device.clone())
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

#[derive(Clone)]
pub struct TrainingConfig {
    pub model_cfg: TransformerConfig,
    pub optim: optim::AdamWConfig,
    pub num_epochs: usize,
    pub batch_size: usize,
    pub max_steps_per_epoch: usize,
    pub eval_examples: usize,
    pub learning_rate: f64,
    pub seed: u64,
}
