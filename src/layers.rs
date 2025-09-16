//! Layers for the transformer architecture.
//!
//! Provides a modular implementation of various layers used in transformer models,
//! including self-attention, feed-forward networks, embedding layers, layer normalization,
//! and output projection. Each layer supports forward and backward passes for training.
//!

pub mod attention;
pub mod embed;
pub mod ffn;
pub mod layernorm;
pub mod project;
pub mod transformer;

use ndarray::Array2;

use crate::layers::{
    attention::SelfAttention, embed::Embed, ffn::FeedForward, layernorm::LayerNorm,
    project::OutputProjection, transformer::TransformerBlock,
};

pub trait Layer {
    fn forward(&mut self, input: &Array2<f32>) -> Array2<f32>;
    fn backward(&mut self, grads: &Array2<f32>, lr: f32) -> Array2<f32>;
}

#[derive(Debug, Clone)]
pub enum LayerType {
    SelfAttention(Box<SelfAttention>),
    FeedForward(Box<FeedForward>),
    Embed(Box<Embed>),
    LayerNorm(Box<LayerNorm>),
    TransformerBlock(Box<TransformerBlock>),
    OutputProjection(Box<OutputProjection>),
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
