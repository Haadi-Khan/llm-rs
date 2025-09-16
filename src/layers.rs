pub mod attention;
pub mod embed;
pub mod ffn;
pub mod layernorm;
pub mod transformer;
pub mod project;

use ndarray::Array2;

use crate::layers::{
    attention::SelfAttention, embed::Embed, ffn::FeedForward, layernorm::LayerNorm, project::OutputProjection, transformer::TransformerBlock
};

pub trait Layer {
    fn forward(&mut self, input: &Array2<f32>) -> Array2<f32>;
    fn backward(&mut self, grads: &Array2<f32>, lr: f32) -> Array2<f32>;
}

#[derive(Debug, Clone)]
pub enum LayerType {
    SelfAttention(SelfAttention),
    FeedForward(FeedForward),
    Embed(Embed),
    LayerNorm(LayerNorm),
    TransformerBlock(TransformerBlock),
    OutputProjection(OutputProjection),
}

impl Layer for LayerType {
    fn forward(&mut self, input: &Array2<f32>) -> Array2<f32> {
        match self {
            LayerType::SelfAttention(layer) => layer.forward(input),
            LayerType::FeedForward(layer) => layer.forward(input),
            LayerType::Embed(layer) => layer.forward(input),
            LayerType::LayerNorm(layer) => layer.forward(input),
            LayerType::TransformerBlock(layer) => layer.forward(input),
            LayerType::OutputProjection(layer) => layer.forward(input),
        }
    }

    fn backward(&mut self, grads: &Array2<f32>, lr: f32) -> Array2<f32> {
        match self {
            LayerType::SelfAttention(layer) => layer.backward(grads, lr),
            LayerType::FeedForward(layer) => layer.backward(grads, lr),
            LayerType::Embed(layer) => layer.backward(grads, lr),
            LayerType::LayerNorm(layer) => layer.backward(grads, lr),
            LayerType::TransformerBlock(layer) => layer.backward(grads, lr),
            LayerType::OutputProjection(layer) => layer.backward(grads, lr),
        }
    }
}