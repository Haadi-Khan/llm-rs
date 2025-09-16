//! Transformer Embedding Layer
//!
//! Embedding is the first step in the transformer architecture. It converts the
//! input into a dense vector representation. It uses token embeddings and
//! positional encoding.
//!
//! References:
//! - ["RoFormer: Enhanced Transformer with Rotary Position Embedding" (Su et al., 2021)](https://arxiv.org/abs/2104.09864)
//!

use ndarray::{Array1, Array2, s};
use rand::Rng;

use crate::{optim::Adam, token::Vocab, util::constants as consts};

#[derive(Debug, Clone)]
/// Embedding Layer with Token and Positional Embeddings
///
/// Stores token embeddings, positional embeddings, cached input for backpropagation,
/// and optimizers for both embeddings.
pub struct Embed {
    pub token: Array2<f32>,
    pub position: Array2<f32>,
    pub cached_input: Option<Array2<f32>>,
    pub token_optimizer: Adam,
    pub positional_optimizer: Adam,
}

impl Default for Embed {
    fn default() -> Self {
        Self {
            token: Self::init_embeddings(consts::VOCAB_SIZE, consts::EMBEDDING_DIM),
            position: Self::init_positional_embeddings(consts::MAX_SEQ_LEN, consts::EMBEDDING_DIM),
            cached_input: None,
            token_optimizer: Adam::new((Vocab::default_words().len(), consts::EMBEDDING_DIM)),
            positional_optimizer: Adam::new((consts::MAX_SEQ_LEN, consts::EMBEDDING_DIM)),
        }
    }
}

impl Embed {
    /// Initialize token embeddings with random values
    fn init_embeddings(vocab_size: usize, embedding_dim: usize) -> Array2<f32> {
        let mut rng = rand::rng();
        Array2::from_shape_fn((vocab_size, embedding_dim), |_| rng.random_range(-1.0..1.0))
    }

    /// Initialize positional embeddings with random values
    fn init_positional_embeddings(max_seq_len: usize, embedding_dim: usize) -> Array2<f32> {
        let mut rng = rand::rng();
        Array2::from_shape_fn((max_seq_len, embedding_dim), |_| {
            rng.random_range(-1.0..1.0)
        })
    }

    /// Get token embeddings for given token IDs
    fn get_token_embeddings(embeddings: &Array2<f32>, token_ids: &[usize]) -> Array2<f32> {
        let mut token_embeds = Array2::<f32>::zeros((token_ids.len(), embeddings.ncols()));
        for (i, &token_id) in token_ids.iter().enumerate() {
            token_embeds.row_mut(i).assign(&embeddings.row(token_id));
        }
        token_embeds
    }

    /// Get positional embeddings for given sequence length
    fn get_positional_embeddings(
        positional_encodings: &Array2<f32>,
        seq_len: usize,
    ) -> Array2<f32> {
        positional_encodings.slice(s![0..seq_len, ..]).to_owned()
    }

    /// Embed tokens by summing token and positional embeddings
    pub fn embed_tokens(&self, token_ids: &[usize]) -> Array2<f32> {
        println!("token_ids: {:?}", token_ids);
        let token_embeds = Self::get_token_embeddings(&self.token, token_ids);
        let position_embeds = Self::get_positional_embeddings(&self.position, token_ids.len());
        token_embeds + position_embeds
    }
}

impl super::Layer for Embed {
    fn forward(&mut self, input: &Array2<f32>) -> Array2<f32> {
        // input shape is [1, sequence_length]
        self.cached_input = Some(input.clone());
        let token_ids: Vec<usize> = input.iter().map(|&x| x as usize).collect();
        self.embed_tokens(&token_ids) // shape is [sequence_length, embedding_dim]
    }

    fn backward(&mut self, grads: &Array2<f32>, lr: f32) -> Array2<f32> {
        let input = self.cached_input.as_ref().unwrap();
        let token_ids: Vec<usize> = input.iter().map(|&x| x as usize).collect();
        let grads = grads.view(); // (sequence_length, embedding_dim)

        // Initialize gradients for embeddings
        let mut token_grads = Array2::zeros(self.token.dim());
        let mut positional_grads = Array2::zeros(self.position.dim());

        for (i, &token_id) in token_ids.iter().enumerate() {
            if token_id >= self.token.nrows() {
                panic!(
                    "Token ID {} out of bounds for vocab size {}",
                    token_id,
                    self.token.nrows()
                );
            }
            let grad_row = grads.row(i);

            // Accumulate token embedding gradients efficiently (no temp variable)
            {
                let mut token_row = token_grads.row_mut(token_id);
                token_row += &grad_row;
            }

            // Accumulate positional embedding gradients efficiently (no temp variable)
            {
                let mut pos_row = positional_grads.row_mut(i);
                pos_row += &grad_row;
            }
        }

        self.token_optimizer.step(&mut self.token, &token_grads, lr);
        self.positional_optimizer
            .step(&mut self.position, &positional_grads, lr);

        // Return gradient to propagate further back
        grads.to_owned()
    }
}

#[derive(Debug, Clone)]
/// Rotary Position Embedding (RoPE) implementation
pub struct RotaryEmbedding {
    head_dim: usize,
    inv_freq: Array1<f32>,
}

impl RotaryEmbedding {
    /// Create a new RotaryEmbedding instance
    pub fn new(head_dim: usize) -> Self {
        assert!(head_dim % 2 == 0, "head_dim must be even for RoPE");

        let mut inv_freq = Array1::<f32>::zeros(head_dim / 2);
        for i in 0..(head_dim / 2) {
            let freq = 10000.0_f32.powf(-((2 * i) as f32) / head_dim as f32);
            inv_freq[i] = freq;
        }

        RotaryEmbedding { head_dim, inv_freq }
    }

    /// Compute cosine and sine matrices for given sequence length
    fn compute_cos_sin(&self, seq_len: usize) -> (Array2<f32>, Array2<f32>) {
        let mut cos = Array2::<f32>::zeros((seq_len, self.head_dim / 2));
        let mut sin = Array2::<f32>::zeros((seq_len, self.head_dim / 2));

        for m in 0..seq_len {
            for (i, &freq) in self.inv_freq.iter().enumerate() {
                let angle = m as f32 * freq;
                cos[[m, i]] = angle.cos();
                sin[[m, i]] = angle.sin();
            }
        }

        (cos, sin)
    }

    /// Apply RoPE rotation to input vectors
    /// x shape: (seq_len, head_dim)
    /// cos, sin shape: (seq_len, head_dim/2)
    ///
    /// Returns rotated vectors of same shape as x
    /// Uses the formula:
    /// [ x_even * cos - x_odd * sin, x_even * sin + x_odd * cos ]
    /// for each pair of even and odd dimensions
    ///
    /// This is equivalent to a 2D rotation in the plane of each pair of dimensions
    /// where x_even = x[2i], x_odd = x[2i+1]
    /// for i in 0..(head_dim/2)
    fn apply_rotation(&self, x: &Array2<f32>, cos: &Array2<f32>, sin: &Array2<f32>) -> Array2<f32> {
        let (seq_len, head_dim) = x.dim();
        assert_eq!(head_dim, self.head_dim);

        let mut output = Array2::<f32>::zeros((seq_len, head_dim));

        for t in 0..seq_len {
            for i in 0..(head_dim / 2) {
                let x_even = x[[t, 2 * i]];
                let x_odd = x[[t, 2 * i + 1]];
                let cos_val = cos[[t, i]];
                let sin_val = sin[[t, i]];

                output[[t, 2 * i]] = x_even * cos_val - x_odd * sin_val;
                output[[t, 2 * i + 1]] = x_even * sin_val + x_odd * cos_val;
            }
        }

        output
    }

    /// Apply RoPE to query and key matrices
    pub fn apply_qk(&self, q: &Array2<f32>, k: &Array2<f32>) -> (Array2<f32>, Array2<f32>) {
        let seq_len = q.shape()[0];
        let (cos, sin) = self.compute_cos_sin(seq_len);

        let q_rot = self.apply_rotation(q, &cos, &sin);
        let k_rot = self.apply_rotation(k, &cos, &sin);

        (q_rot, k_rot)
    }

    /// Backward pass through RoPE rotation
    /// Given gradients w.r.t rotated vectors, compute gradients w.r.t original vectors
    pub fn backward_rotation(&self, grad_rotated: &Array2<f32>, seq_len: usize) -> Array2<f32> {
        let (cos, sin) = self.compute_cos_sin(seq_len);
        let mut grad_original = Array2::<f32>::zeros(grad_rotated.dim());

        for t in 0..seq_len {
            for i in 0..(self.head_dim / 2) {
                let grad_even = grad_rotated[[t, 2 * i]];
                let grad_odd = grad_rotated[[t, 2 * i + 1]];
                let cos_val = cos[[t, i]];
                let sin_val = sin[[t, i]];

                // Inverse rotation: R^T = R(-θ)
                grad_original[[t, 2 * i]] = grad_even * cos_val + grad_odd * sin_val;
                grad_original[[t, 2 * i + 1]] = -grad_even * sin_val + grad_odd * cos_val;
            }
        }

        grad_original
    }
}
