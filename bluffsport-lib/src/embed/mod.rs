//! Text embedding using local models
//!
//! Uses BAAI/bge-large-en-v1.5 via the fastembed crate (ONNX runtime).
//!
//! # Model Details
//!
//! - Dimensions: 1024
//! - Max tokens: 512
//! - MTEB retrieval score: ~54%
//!
//! # Usage
//!
//! ```ignore
//! use bluffsport_lib::embed::Embedder;
//!
//! let mut embedder = BgeEmbedder::new()?;
//!
//! // Embed documents (for indexing)
//! let doc_embeddings = embedder.embed_documents(&["Match report...", "News article..."])?;
//!
//! // Embed query (for searching)
//! let query_embedding = embedder.embed_query("Who scored the winning goal?")?;
//! ```

use crate::{Result};

/// A vector embedding - fixed size array of floats
pub type Embedding = Vec<f32>;

/// Trait for text embedding models
pub trait Embedder: Send + Sync {
    /// Embed multiple documents for indexing
    ///
    /// Documents may be batched for efficiency.
    fn embed_documents(&mut self, texts: &[&str]) -> Result<Vec<Embedding>>;

    /// Embed a single query for searching
    ///
    /// Note: Some models (like BGE) use different prompts for queries vs documents.
    /// This method handles that distinction.
    fn embed_query(&mut self, text: &str) -> Result<Embedding>;

    /// Returns the embedding dimension
    fn dimension(&self) -> usize;

    /// Returns the model name/identifier
    fn model_name(&self) -> &str;
}

mod bge;
pub use bge::*;