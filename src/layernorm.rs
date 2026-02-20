use burn::{module::Param, nn::Initializer, prelude::*};

pub struct Config {
    d_model: usize,
    eps: f32,
}

#[derive(Module, Debug)]
pub struct LayerNorm<B: Backend> {
    /// γ
    /// Shape: (d_model)
    scale: Param<Tensor<B, 1>>,
    /// β
    /// Shape: (d_model)
    bias: Param<Tensor<B, 1>>,
    eps: f32,
}

impl<B: Backend> LayerNorm<B> {
    pub fn init(&self, cfg: Config, device: &B::Device) -> Self {
        let Config { d_model, eps } = cfg;
        let scale = Initializer::Ones.init([d_model], device);
        let bias = Initializer::Zeros.init([d_model], device);

        Self { scale, bias, eps }
    }

    /// [`residual`] has shape (batch pos d_model)
    pub fn forward<const D: usize>(&self, residual: Tensor<B, D>) -> Tensor<B, D> {
        let residual_mean = residual.clone().mean_dim(D - 1);

        let residual_std = residual.clone().var(D - 1).add_scalar(self.eps).sqrt();
        let normalized = (residual - residual_mean) / residual_std;
        // normalized * self.w + self.b
        normalized
            .mul(self.scale.val().unsqueeze())
            .add(self.bias.val().unsqueeze())
    }
}
