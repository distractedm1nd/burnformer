use burn::{module::Param, nn::Initializer, prelude::*, tensor::activation::gelu};

use crate::config::TransformerConfig;

pub struct MLPConfig {
    pub d_mlp: usize,
    pub d_model: usize,
    pub init_range: f64,
}

impl MLPConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> MultiLayerPerceptron<B> {
        MultiLayerPerceptron::init(self, device)
    }
}

impl From<TransformerConfig> for MLPConfig {
    fn from(cfg: TransformerConfig) -> Self {
        Self {
            d_mlp: cfg.d_mlp(),
            d_model: cfg.d_model(),
            init_range: cfg.init_range(),
        }
    }
}

#[derive(Debug, Module)]
pub struct MultiLayerPerceptron<B: Backend> {
    /// (d_model d_mlp)
    w_in: Param<Tensor<B, 2>>,
    /// (d_mlp d_model)
    w_out: Param<Tensor<B, 2>>,
    /// (d_mlp)
    b_in: Param<Tensor<B, 1>>,
    /// (d_model)
    b_out: Param<Tensor<B, 1>>,
}

impl<B: Backend> MultiLayerPerceptron<B> {
    pub fn init(cfg: &MLPConfig, device: &B::Device) -> Self {
        let MLPConfig {
            d_mlp,
            d_model,
            init_range,
        } = cfg;
        let w_in = Initializer::Normal {
            mean: 0.0,
            std: *init_range,
        }
        .init([d_model, d_mlp], device);
        let b_in = Initializer::Zeros.init([d_mlp], device);

        let w_out = Initializer::Normal {
            mean: 0.0,
            std: *init_range,
        }
        .init([d_mlp, d_model], device);
        let b_out = Initializer::Zeros.init([d_model], device);

        Self {
            w_in,
            w_out,
            b_in,
            b_out,
        }
    }

    /// (batch pos d_model) -> (batch pos d_model)
    pub fn forward(&self, normalized_resid_mid: Tensor<B, 3>) -> Tensor<B, 3> {
        let l1 =
            normalized_resid_mid.matmul(self.w_in.val().unsqueeze()) + self.b_in.val().unsqueeze();
        let activated = gelu(l1);
        activated.matmul(self.w_out.val().unsqueeze()) + self.b_out.val().unsqueeze()
    }
}
