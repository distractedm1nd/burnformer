use burn::{module::Param, prelude::*, tensor::bf16};

pub struct Attention<B: Backend> {
    /// (w: (n_heads d_model d_head), b: (n_heads d_head))
    q: (Param<Tensor<B, 3>>, Param<Tensor<B, 2>>),
    /// (w: (n_heads d_model d_head), b: (n_heads d_head))
    k: (Param<Tensor<B, 3>>, Param<Tensor<B, 2>>),
    /// (w: (n_heads d_model d_head), b: (n_heads d_head))
    v: (Param<Tensor<B, 3>>, Param<Tensor<B, 2>>),
    /// (w: (n_heads d_head d_model), b: (d_model))
    o: (Param<Tensor<B, 3>>, Param<Tensor<B, 1>>),
}

impl<B: Backend> Attention<B> {
    /// (batch pos d_model) -> (batch pos d_model)
    pub fn forward(&self, normalized_resid_pre: Tensor<B, 3>) -> Tensor<B, 3> {
        let (w_q, b_q) = self.q.clone();
        // (batch pos d_model) -> (batch, n_head, pos, d_head)
        let q = normalized_resid_pre
            .clone() // (b pos d_m)
            .unsqueeze_dim::<4>(1) // (b, 1, pos, d_m)
            .matmul(w_q.val().unsqueeze()) // (b, 1, pos, d_m) x (1 n_h d_m d_h) -> (b n_h p d_h)
            .add(b_q.val().unsqueeze::<4>().reshape([1, -1, 1, 0])); // (b n_h p d_h) + (1 n_h 1 d_h)

        let (w_k, b_k) = self.k.clone();
        // (batch pos d_model) -> (batch, n_head, pos, d_head)
        let k = normalized_resid_pre
            .clone() // (b pos d_m)
            .unsqueeze_dim::<4>(1) // (b, 1, pos, d_m)
            .matmul(w_k.val().unsqueeze()) // (b, 1, pos, d_m) x (1 n_h d_m d_h) -> (b n_h p d_h)
            .add(b_k.val().unsqueeze::<4>().reshape([1, -1, 1, 0])); // (b n_h p d_h) + (1 n_h 1 d_h)

        let (w_v, b_v) = self.v.clone();
        // (batch pos d_model) -> (batch, n_head, pos, d_head)
        let v = normalized_resid_pre
            .clone() // (b pos d_m)
            .unsqueeze_dim::<4>(1) // (b, 1, pos, d_m)
            .matmul(w_v.val().unsqueeze()) // (b, 1, pos, d_m) x (1 n_h d_m d_h) -> (b n_h p d_h)
            .add(b_v.val().unsqueeze::<4>().reshape([1, -1, 1, 0])); // (b n_h p d_h) + (1 n_h 1 d_h)

        todo!()
    }

    /// Applies a causal mask to attention scores, and returns masked scores.
    /// (batch n_heads query_pos key_pos) -> (batch n_heads query_pos key_pos)
    fn apply_causal_mask(&self, attn_scores: Tensor<B, 4>) -> Tensor<B, 4> {
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

    #[test]
    // still just trying to play with burn tensor ops
    fn test_matmul() {
        let device = Default::default();
        // shape: (3, 3, 4) batch pos d_head
        let x = Tensor::<TestBackend, 3>::from_data(
            [
                [[1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3]],
                [[4, 4, 4, 4], [5, 5, 5, 5], [6, 6, 6, 6]],
                [[7, 7, 7, 7], [8, 8, 8, 8], [9, 9, 9, 9]],
            ],
            &device,
        );
        // shape (2, 4, 2) n_head d_model d_head
        let w_q = Tensor::<TestBackend, 3>::from_data(
            [
                [[-10, -10], [0, 0], [10, 10], [20, 20]],
                [[-1, -1], [-2, -2], [-3, -3], [-4, -4]],
            ],
            &device,
        );

        println!("{}", x.unsqueeze_dim::<4>(1).matmul(w_q.unsqueeze()))
    }
}
