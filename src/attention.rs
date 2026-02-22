use burn::{
    module::Param,
    nn::Initializer,
    prelude::*,
    tensor::{activation::softmax, bf16},
};

use crate::config::TransformerConfig;

pub struct AttentionConfig {
    d_model: usize,

    n_head: usize,
    d_head: usize,

    init_range: f64,
}

impl From<TransformerConfig> for AttentionConfig {
    fn from(value: TransformerConfig) -> Self {
        Self {
            d_model: value.d_model(),
            n_head: value.n_head(),
            d_head: value.d_head(),
            init_range: value.init_range(),
        }
    }
}

impl AttentionConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> Attention<B> {
        Attention::init(self, device)
    }
}

#[derive(Debug, Module)]
pub struct Attention<B: Backend> {
    /// (w: (n_heads d_model d_head), b: (n_heads d_head))
    q: (Param<Tensor<B, 3>>, Param<Tensor<B, 2>>),
    /// (w: (n_heads d_model d_head), b: (n_heads d_head))
    k: (Param<Tensor<B, 3>>, Param<Tensor<B, 2>>),
    /// (w: (n_heads d_model d_head), b: (n_heads d_head))
    v: (Param<Tensor<B, 3>>, Param<Tensor<B, 2>>),
    /// (w: (n_heads d_head d_model), b: (d_model))
    o: (Param<Tensor<B, 3>>, Param<Tensor<B, 1>>),

    // sqrt(d_head)
    attn_scale: f64,
}

impl<B: Backend> Attention<B> {
    pub fn init(cfg: &AttentionConfig, device: &B::Device) -> Self {
        let AttentionConfig {
            d_model,
            n_head,
            d_head,
            init_range,
        } = cfg;

        // Q
        let w = Initializer::Normal {
            mean: 0.0,
            std: *init_range,
        }
        .init([n_head, d_model, d_head], device);
        let b = Initializer::Zeros.init([n_head, d_head], device);
        let q = (w, b);

        // K
        let w = Initializer::Normal {
            mean: 0.0,
            std: *init_range,
        }
        .init([n_head, d_model, d_head], device);
        let b = Initializer::Zeros.init([n_head, d_head], device);
        let k = (w, b);

        // V
        let w = Initializer::Normal {
            mean: 0.0,
            std: *init_range,
        }
        .init([n_head, d_model, d_head], device);
        let b = Initializer::Zeros.init([n_head, d_head], device);
        let v = (w, b);

        // O
        let w = Initializer::Normal {
            mean: 0.0,
            std: *init_range,
        }
        .init([n_head, d_head, d_model], device);
        let b = Initializer::Zeros.init([d_model], device);
        let o = (w, b);

        let attn_scale = (*d_head as f64).sqrt();

        Self {
            q,
            k,
            v,
            o,
            attn_scale,
        }
    }

    /// (batch pos d_model) -> (batch pos d_model)
    pub fn forward(&self, normalized_resid_pre: Tensor<B, 3>) -> Tensor<B, 3> {
        // note: arena makes qkv tensors shape (batch pos n_head d_head) but I find this way more intuitive,
        // thinking about batch and n_head as the outer dimensions makes more sense to me, and they have to make this
        // transformation anyways downstream

        let (w_q, b_q) = self.q.clone();
        // (batch pos d_model) -> (batch, n_head, pos, d_head)
        let q = normalized_resid_pre
            .clone() // (b pos d_m)
            .unsqueeze_dim::<4>(1) // (b, 1, pos, d_m)
            .matmul(w_q.val().unsqueeze()) // (b, 1, pos, d_m) @ (1 n_h d_m d_h) -> (b n_h p d_h)
            .add(b_q.val().unsqueeze::<4>().reshape([1, -1, 1, 0])); // (b n_h p d_h) + (1 n_h 1 d_h)

        let (w_k, b_k) = self.k.clone();
        // (batch pos d_model) -> (batch, n_head, pos, d_head)
        let k = normalized_resid_pre
            .clone() // (b pos d_m)
            .unsqueeze_dim::<4>(1) // (b, 1, pos, d_m)
            .matmul(w_k.val().unsqueeze()) // (b, 1, pos, d_m) @ (1 n_h d_m d_h) -> (b n_h p d_h)
            .add(b_k.val().unsqueeze::<4>().reshape([1, -1, 1, 0])); // (b n_h p d_h) + (1 n_h 1 d_h)

        let (w_v, b_v) = self.v.clone();
        // (batch pos d_model) -> (batch, n_head, pos, d_head)
        let v = normalized_resid_pre // (b pos d_m)
            .unsqueeze_dim::<4>(1) // (b, 1, pos, d_m)
            .matmul(w_v.val().unsqueeze()) // (b, 1, pos, d_m) @ (1 n_h d_m d_h) -> (b n_h p d_h)
            .add(b_v.val().unsqueeze::<4>().reshape([1, -1, 1, 0])); // (b n_h p d_h) + (1 n_h 1 d_h)

        let attn_scores = q.matmul(k.transpose()); // (b n_h p_q d_h) @ (b n_h d_h p_k)^T -> (b n_h p_q p_k)
        let masked = self.apply_causal_mask(attn_scores.div_scalar(self.attn_scale));
        let a = softmax(masked, 3);

        // (batch n_h pos_q pos_k) @ (batch n_h pos_k, d_head) -> (batch n_h pos_q d_head)
        let z = a.matmul(v);
        let (w_o, b_o) = self.o.clone();
        // (batch n_h pos_q d_head) -> (batch pos d_model)
        z.matmul(w_o.val().unsqueeze()) // (b n_h p_q d_m)
            .sum_dim(1) // (b 1 p_q d_m)
            .squeeze_dim(1) // (b p_q d_m)
            .add(b_o.val().unsqueeze()) // (b p_q d_m) + (1 1 d_m)
    }

    /// Applies a causal mask to attention scores, and returns masked scores.
    /// (batch n_heads query_pos key_pos) -> (batch n_heads query_pos key_pos)
    fn apply_causal_mask(&self, attn_scores: Tensor<B, 4>) -> Tensor<B, 4> {
        const D: usize = 4;
        let dims = attn_scores.shape().dims;
        let mask =
            Tensor::<B, 2, Bool>::tril_mask([dims[D - 2], dims[D - 1]], 0, &attn_scores.device());
        attn_scores.mask_fill(mask.unsqueeze::<D>(), bf16::NEG_INFINITY)
    }
}
