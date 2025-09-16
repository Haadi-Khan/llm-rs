//! The first step in the transformer pipeline. Transform all the input tokens
//! into our embedding space. This is our word2vec and RoPE step.

use ndarray::{s, Array1, Array2};
use rand::Rng;

use crate::util::constants as consts;

pub struct Embed {
    pub token: Array2<f32>,
    pub position: Array2<f32>,
}

impl Default for Embed { 
    fn default() -> Self {
        Self { 
            token: Self::init_embeddings(consts::VOCAB_SIZE, consts::EMBEDDING_DIM),
            position: Self::init_positional_embeddings(consts::MAX_SEQ_LEN, consts::EMBEDDING_DIM),
         }
    }
}

impl Embed {
    fn init_embeddings(vocab_size: usize, embedding_dim: usize) -> Array2<f32> {
        let mut rng = rand::rng();
        Array2::from_shape_fn((vocab_size, embedding_dim), |_| rng.random_range(-1.0..1.0))
    }

    fn init_positional_embeddings(max_seq_len: usize, embedding_dim: usize) -> Array2<f32> {
        let mut rng = rand::rng();
        Array2::from_shape_fn((max_seq_len, embedding_dim), |_| rng.random_range(-1.0..1.0))
    }

    fn get_token_embeddings(embeddings: &Array2<f32>, token_ids: &[usize]) -> Array2<f32> {
        let mut token_embeds = Array2::<f32>::zeros((token_ids.len(), embeddings.ncols()));
        for (i, &token_id) in token_ids.iter().enumerate() {
            token_embeds.row_mut(i).assign(&embeddings.row(token_id));
        }
        token_embeds
    }

    fn get_positional_embeddings(positional_encodings: &Array2<f32>, seq_len: usize) -> Array2<f32> {
        positional_encodings.slice(s![0..seq_len, ..]).to_owned()
    }

    pub fn embed_tokens(
        &self,
        token_ids: &[usize]
    ) -> Array2<f32> {
        println!("token_ids: {:?}", token_ids);
        let token_embeds = Self::get_token_embeddings(&self.token, token_ids);
        let position_embeds = Self::get_positional_embeddings(&self.position, token_ids.len());
        token_embeds + position_embeds
    }
}

/// Rotary Position Embedding (RoPE) implementation
pub struct RotaryEmbedding {
    head_dim: usize,
    inv_freq: Array1<f32>,
}

impl RotaryEmbedding {
    pub fn new(head_dim: usize) -> Self {
        assert!(head_dim % 2 == 0, "head_dim must be even for RoPE");
        
        let mut inv_freq = Array1::<f32>::zeros(head_dim / 2);
        for i in 0..(head_dim / 2) {
            let freq = 10000.0_f32.powf(-((2 * i) as f32) / head_dim as f32);
            inv_freq[i] = freq;
        }
        
        RotaryEmbedding { head_dim, inv_freq }
    }
    
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