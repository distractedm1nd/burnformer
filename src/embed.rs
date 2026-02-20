use burn::{module::Param, nn::Initializer, prelude::*};

pub struct Config {
    d_vocab: usize,
    d_model: usize,
    n_ctx: usize,
    init_range: f64,
}

#[derive(Debug, Module)]
struct Embed<B: Backend> {
    w_e: Param<Tensor<B, 2>>,
}

impl<B: Backend> Embed<B> {
    pub fn init(&self, cfg: Config, device: &B::Device) -> Self {
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
    pub fn init(&self, cfg: Config, device: &B::Device) -> Self {
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

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::Wgpu;
    use burn::prelude::Backend;

    type TestBackend = Wgpu;

    #[test]
    fn embed_init_builds_vocab_by_model_matrix() {
        let device = <TestBackend as Backend>::Device::default();
        let (d_vocab, d_model, n_ctx) = (11, 7, 16);
        let cfg = Config {
            d_vocab,
            d_model,
            n_ctx,
            init_range: 0.02,
        };

        let seed: Embed<TestBackend> = Embed {
            w_e: Initializer::Zeros.init([1, 1], &device),
        };
        let embed = seed.init(cfg, &device);
        let shape = embed.w_e.val().shape();

        assert_eq!(shape[0], d_vocab);
        assert_eq!(shape[1], d_model);
    }

    #[test]
    fn embed_forward_returns_batch_pos_model_tensor() {
        let device = <TestBackend as Backend>::Device::default();
        let (batch, seq_length, d_vocab, d_model, n_ctx) = (2, 3, 13, 5, 16);
        let cfg = Config {
            d_vocab,
            d_model,
            n_ctx,
            init_range: 0.02,
        };
        let seed: Embed<TestBackend> = Embed {
            w_e: Initializer::Zeros.init([1, 1], &device),
        };
        let embed = seed.init(cfg, &device);
        let tokens = Tensor::<TestBackend, 2, Int>::from_data([[0, 1, 2], [3, 4, 5]], &device);

        let output: Tensor<TestBackend, 3> = embed.forward(tokens);
        let shape = output.shape();

        assert_eq!(shape[0], batch);
        assert_eq!(shape[1], seq_length);
        assert_eq!(shape[2], d_model);
    }

    #[test]
    fn embed_forward_matches_token_row_lookup() {
        let device = <TestBackend as Backend>::Device::default();
        let cfg = Config {
            d_vocab: 4,
            d_model: 3,
            n_ctx: 8,
            init_range: 0.02,
        };
        let seed: Embed<TestBackend> = Embed {
            w_e: Initializer::Zeros.init([1, 1], &device),
        };
        let mut embed = seed.init(cfg, &device);
        let known_w_e = Tensor::<TestBackend, 2>::from_data(
            [
                [10.0, 11.0, 12.0],
                [20.0, 21.0, 22.0],
                [30.0, 31.0, 32.0],
                [40.0, 41.0, 42.0],
            ],
            &device,
        );
        embed.w_e = Param::from_tensor(known_w_e);

        let tokens = Tensor::<TestBackend, 2, Int>::from_data([[2, 0], [3, 1]], &device);
        let output = embed.forward(tokens);
        let values = output.into_data().to_vec::<f32>().unwrap();

        assert_eq!(
            values,
            vec![
                30.0, 31.0, 32.0, 10.0, 11.0, 12.0, 40.0, 41.0, 42.0, 20.0, 21.0, 22.0
            ]
        );
    }

    #[test]
    fn pos_embed_init_builds_ctx_by_model_matrix() {
        let device = <TestBackend as Backend>::Device::default();
        let (n_ctx, d_model, d_vocab) = (19, 6, 32);
        let cfg = Config {
            d_vocab,
            d_model,
            n_ctx,
            init_range: 0.02,
        };

        let seed: PosEmbed<TestBackend> = PosEmbed {
            w_pos: Initializer::Zeros.init([1, 1], &device),
        };
        let pos_embed = seed.init(cfg, &device);
        let shape = pos_embed.w_pos.val().shape();

        assert_eq!(shape[0], n_ctx);
        assert_eq!(shape[1], d_model);
    }

    #[test]
    fn pos_embed_forward_returns_batch_pos_model_tensor() {
        let device = <TestBackend as Backend>::Device::default();
        let (batch, seq_length, n_ctx, d_model, d_vocab) = (3, 4, 17, 8, 32);
        let cfg = Config {
            d_vocab,
            d_model,
            n_ctx,
            init_range: 0.02,
        };
        let seed: PosEmbed<TestBackend> = PosEmbed {
            w_pos: Initializer::Zeros.init([1, 1], &device),
        };
        let pos_embed = seed.init(cfg, &device);
        let tokens = Tensor::<TestBackend, 2>::zeros([batch, seq_length], &device);

        let output: Tensor<TestBackend, 3> = pos_embed.forward(tokens);
        let shape = output.shape();

        assert_eq!(shape[0], batch);
        assert_eq!(shape[1], seq_length);
        assert_eq!(shape[2], d_model);
    }
}
