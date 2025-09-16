use super::{attention::SelfAttention, ffn::FeedForward, layernorm::LayerNorm};
use ndarray::Array2;

#[derive(Debug, Clone)]
pub struct TransformerBlock {
    attention: SelfAttention,
    feed_forward: FeedForward,
    norm1: LayerNorm,
    norm2: LayerNorm,
}

impl TransformerBlock {
    pub fn new(embedding_dim: usize, hidden_dim: usize) -> Self {
        TransformerBlock {
            attention: SelfAttention::new(embedding_dim),
            feed_forward: FeedForward::new(embedding_dim, hidden_dim),
            norm1: LayerNorm::new(embedding_dim),
            norm2: LayerNorm::new(embedding_dim),
        }
    }
}

impl super::Layer for TransformerBlock {
    fn forward(&mut self, input: &Array2<f32>) -> Array2<f32> {
        let attention_out = self.attention.forward(input);
        let norm1_out = self.norm1.normalize(&attention_out);

        let ff_out = self.feed_forward.forward(&norm1_out);

        self.norm2.normalize(&ff_out)
    }

    fn backward(&mut self, grads: &Array2<f32>, lr: f32) -> Array2<f32> {
        let grad_norm2 = self.norm2.backward(grads, lr);
        let grad_ffn = self.feed_forward.backward(&grad_norm2, lr);
        let grad_norm1 = self.norm1.backward(&grad_ffn, lr);

        self.attention.backward(&grad_norm1, lr)
    }
}
