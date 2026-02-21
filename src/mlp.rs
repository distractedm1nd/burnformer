use burn::{module::Param, prelude::*, tensor::activation::gelu};

#[derive(Debug, Module)]
pub struct MLP<B: Backend> {
    /// (d_model d_mlp)
    w_in: Param<Tensor<B, 2>>,
    /// (d_mlp d_model)
    w_out: Param<Tensor<B, 2>>,
    /// (d_mlp)
    b_in: Param<Tensor<B, 1>>,
    /// (d_model)
    b_out: Param<Tensor<B, 1>>,
}

impl<B: Backend> MLP<B> {
    /// (batch pos d_model) -> (batch pos d_model)
    pub fn forward(&self, normalized_resid_mid: Tensor<B, 3>) -> Tensor<B, 3> {
        let l1 =
            normalized_resid_mid.matmul(self.w_in.val().unsqueeze()) + self.b_in.val().unsqueeze();
        let activated = gelu(l1);
        activated.matmul(self.w_out.val().unsqueeze()) + self.b_out.val().unsqueeze()
    }
}
