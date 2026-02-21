use burn::{module::Param, prelude::*, tensor::bf16};

pub struct Attention<B: Backend> {
    /// (w: (n_heads, d_model, d_head), b: (n_heads, d_head))
    q: (Param<Tensor<B, 3>>, Param<Tensor<B, 2>>),
    /// (w: (n_heads, d_model, d_head), b: (n_heads, d_head))
    k: (Param<Tensor<B, 3>>, Param<Tensor<B, 2>>),
    /// (w: (n_heads, d_model, d_head), b: (n_heads, d_head))
    v: (Param<Tensor<B, 3>>, Param<Tensor<B, 2>>),
    /// (w: (n_heads, d_head, d_model), b: (d_model))
    o: (Param<Tensor<B, 3>>, Param<Tensor<B, 1>>),
}

impl<B: Backend> Attention<B> {
    /// Applies a causal mask to attention scores, and returns masked scores.
    /// [`attn_scores`] has shape (batch n_heads query_pos key_pos)
    /// returned [`Tensor`] has (batch n_heads query_pos key_pos)
    fn apply_causal_mask(&self, attn_scores: Tensor<B, 3>) -> Tensor<B, 3> {
        // let dims = attn_scores.shape().dims;
        // can i even use ones_like here? even if I can, probably not as efficient?
        let ones = Tensor::ones_like(&attn_scores);
        let mask = ones.tril(-1).bool();

        attn_scores.mask_fill(mask, bf16::NEG_INFINITY)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::Wgpu;
    use burn::prelude::Backend;

    type TestBackend = Wgpu;

    #[test]
    // making sure it works as i expect
    fn test_tril() {
        let device = Default::default();
        // let shape = (2, 2, 128, 128);
        let ones = Tensor::<TestBackend, 2>::from_data(
            [[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]],
            &device,
        );
        let mask = ones.clone().tril(-1).bool();
        println!("{}", mask);
        println!("{}", ones.mask_fill(mask, 3.))
    }
}
