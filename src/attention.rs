//! The attention mechanism with RoPE (Rotary Position Embedding).

use crate::{embed::RotaryEmbedding, optim::Adam};
use crate::transformer::Layer;
use ndarray::Array2;
use rand_distr::{Distribution, Normal};

use crate::util::constants as consts;
use std::f32;

pub struct SelfAttention {
    pub embed_dim: usize,
    #[allow(dead_code)]
    head_dim: usize,
    w_q: Array2<f32>,
    w_k: Array2<f32>,
    w_v: Array2<f32>,
    rope: RotaryEmbedding,

    cached_input: Option<Array2<f32>>,

    optimizer_w_q: Adam,
    optimizer_w_k: Adam,
    optimizer_w_v: Adam,
}

impl Default for SelfAttention {
    fn default() -> Self {
        SelfAttention::new(consts::EMBEDDING_DIM)
    }
}

impl SelfAttention {
    pub fn new(embedding_dim: usize) -> Self {
        let mut rng = rand::rng();
        let head_dim = embedding_dim; // For single-head attention, head_dim = embedding_dim

        // Xavier/He initialization: std = sqrt(2 / fan_in)
        let std = (2.0 / embedding_dim as f32).sqrt();
        let normal = Normal::new(0.0, std).unwrap();

        SelfAttention {
            embed_dim: embedding_dim,
            head_dim,
            w_q: Array2::from_shape_fn((embedding_dim, embedding_dim), |_| normal.sample(&mut rng)),
            w_k: Array2::from_shape_fn((embedding_dim, embedding_dim), |_| normal.sample(&mut rng)),
            w_v: Array2::from_shape_fn((embedding_dim, embedding_dim), |_| normal.sample(&mut rng)),
            rope: RotaryEmbedding::new(head_dim),
            cached_input: None,
            optimizer_w_q: Adam::new((embedding_dim, embedding_dim)),
            optimizer_w_k: Adam::new((embedding_dim, embedding_dim)),
            optimizer_w_v: Adam::new((embedding_dim, embedding_dim)),
        }
    }

    fn compute_qkv(&self, input: &Array2<f32>) -> (Array2<f32>, Array2<f32>, Array2<f32>) {
        let q = input.dot(&self.w_q);
        let k = input.dot(&self.w_k);
        let v = input.dot(&self.w_v);
        
        // Apply RoPE to Q and K
        let (q_rope, k_rope) = self.rope.apply_qk(&q, &k);
        
        (q_rope, k_rope, v)
    }

    fn attention(&self, q: &Array2<f32>, k: &Array2<f32>, v: &Array2<f32>) -> Array2<f32> {
        let dk = (self.embed_dim as f32).sqrt();

        let k_t = k.t();
        let mut scores = q.dot(&k_t) / dk;

        // Apply causal masking - prevent attention to future tokens
        let seq_len = scores.shape()[0];
        for i in 0..seq_len {
            for j in (i + 1)..seq_len {
                scores[[i, j]] = f32::NEG_INFINITY;
            }
        }

        let weights = self.softmax(&scores);
        weights.dot(v)
    }

    fn softmax(&self, scores: &Array2<f32>) -> Array2<f32> {
        let mut result = scores.clone();

        // Apply softmax row-wise
        for mut row in result.rows_mut() {
            let max_val = row.iter().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();
            // Calculate exp for each element
            let exp_values: Vec<f32> = row.iter().map(|&x| (x - max_val).exp()).collect();
            let sum_exp: f32 = exp_values.iter().sum();

            // Normalize by sum
            for (i, &exp_val) in exp_values.iter().enumerate() {
                row[i] = exp_val / sum_exp;
            }
        }

        result
    }

    fn softmax_backward(softmax_output: &Array2<f32>, grad_output: &Array2<f32>) -> Array2<f32> {
        let mut grad_input = softmax_output.clone();

        for ((mut grad_row, softmax_row), grad_out_row) in grad_input
            .outer_iter_mut()
            .zip(softmax_output.outer_iter())
            .zip(grad_output.outer_iter())
        {
            let dot = softmax_row
                .iter()
                .zip(grad_out_row.iter())
                .map(|(&y_i, &dy_i)| y_i * dy_i)
                .sum::<f32>();

            for ((g, &y_i), &dy_i) in grad_row
                .iter_mut()
                .zip(softmax_row.iter())
                .zip(grad_out_row.iter())
            {
                *g = y_i * (dy_i - dot);
            }
        }

        grad_input
    }
}

impl Layer for SelfAttention {
    fn layer_type(&self) -> crate::transformer::LayerType {
        crate::transformer::LayerType::SelfAttention
    }

    fn forward(&mut self, input: &Array2<f32>) -> Array2<f32> {
        self.cached_input = Some(input.clone());
        let qkv = self.compute_qkv(input);
        let attention = self.attention(&qkv.0, &qkv.1, &qkv.2);
        attention + input // residual connection (no LayerNorm here)
    }

    fn backward(&mut self, grads: &Array2<f32>, lr: f32) -> Array2<f32> {
        let input = self.cached_input.as_ref().unwrap();
        
        // Forward pass through linear layers
        let q_pre_rope = input.dot(&self.w_q);
        let k_pre_rope = input.dot(&self.w_k);
        let v = input.dot(&self.w_v);
        
        // Apply RoPE to get q and k
        let (q, k) = self.rope.apply_qk(&q_pre_rope, &k_pre_rope);
        
        let dk = self.w_q.shape()[1] as f32;
        let scale = dk.sqrt();

        let mut scores = q.dot(&k.t()) / scale;

        // Apply causal masking - prevent attention to future tokens
        let seq_len = scores.shape()[0];
        for i in 0..seq_len {
            for j in (i + 1)..seq_len {
                scores[[i, j]] = f32::NEG_INFINITY;
            }
        }

        let attn_weights = self.softmax(&scores);

        // Backward pass through attention
        let grad_attn_weights = grads.dot(&v.t());
        let grad_v = attn_weights.t().dot(grads);

        // Softmax backward
        let grad_scores = SelfAttention::softmax_backward(&attn_weights, &grad_attn_weights);

        // Gradients w.r.t Q and K (after RoPE)
        let grad_q = grad_scores.dot(&k);
        let grad_k = grad_scores.t().dot(&q);

        // Backward through RoPE
        let grad_q_pre_rope = self.rope.backward_rotation(&grad_q, seq_len);
        let grad_k_pre_rope = self.rope.backward_rotation(&grad_k, seq_len);

        // Gradients w.r.t weight matrices
        let grad_w_q = input.t().dot(&grad_q_pre_rope);
        let grad_w_k = input.t().dot(&grad_k_pre_rope);
        let grad_w_v = input.t().dot(&grad_v);

        // Gradient w.r.t input
        let grad_input_attention = grad_q_pre_rope.dot(&self.w_q.t()) 
            + grad_k_pre_rope.dot(&self.w_k.t()) 
            + grad_v.dot(&self.w_v.t());

        // Add gradient from residual connection
        let grad_input = grad_input_attention + grads;

        // Update weights
        self.optimizer_w_q.step(&mut self.w_q, &grad_w_q, lr);
        self.optimizer_w_k.step(&mut self.w_k, &grad_w_k, lr);
        self.optimizer_w_v.step(&mut self.w_v, &grad_w_v, lr);

        grad_input
    }
}
