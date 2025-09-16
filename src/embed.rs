//! The first step in the transformer pipeline. Transform all the input tokens
//! into our embedding space. This is our word2vec and RoPE step.

use ndarray::{s, Array2};
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