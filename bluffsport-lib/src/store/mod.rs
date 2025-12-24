//! Vector storage backends
//!
//! Start with in-memory storage for development, then migrate to pgvector.
//!
//! # Storage Model
//!
//! Each stored item consists of:
//! - Chunk: the original text and metadata
//! - Embedding: the vector representation
//!
//! # Usage
//!
//! ```ignore
//! use bluffsport_lib::store::{VectorStore, MemoryStore};
//!
//! let store = MemoryStore::new();
//!
//! // Insert chunks with their embeddings
//! store.insert(&chunks, &embeddings)?;
//!
//! // Search by vector similarity
//! let results = store.search(&query_embedding, 5)?;
//! ```

use crate::chunk::Chunk;
use crate::embed::Embedding;
use crate::Result;

/// A search result with similarity score
#[derive(Debug, Clone)]
pub struct SearchResult {
    /// The matched chunk
    pub chunk: Chunk,
    /// Similarity score (higher is more similar)
    /// For cosine similarity: -1.0 to 1.0
    pub score: f32,
}

impl PartialOrd for SearchResult {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.score.partial_cmp(&other.score)
    }
}

impl Ord for SearchResult {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        if self.score > other.score {
            std::cmp::Ordering::Greater
        } else if self.score < other.score {
            std::cmp::Ordering::Less
        } else {
            std::cmp::Ordering::Equal
        }
    }
}

impl PartialEq for SearchResult {
    fn eq(&self, other: &Self) -> bool {
        (self.score - other.score) < f32::EPSILON
    }
}

impl Eq for SearchResult {}

/// Trait for vector storage backends
pub trait VectorStore: Send + Sync {
    /// Insert chunks with their embeddings
    ///
    /// # Arguments
    /// * `chunks` - The text chunks to store
    /// * `embeddings` - Corresponding embeddings (must be same length)
    fn insert(&mut self, chunks: &[Chunk], embeddings: &[Embedding]) -> Result<()>;

    /// Search for similar chunks
    ///
    /// # Arguments
    /// * `query_embedding` - The query vector
    /// * `k` - Number of results to return
    ///
    /// # Returns
    /// Top-k results sorted by similarity (highest first)
    fn search(&self, query_embedding: &Embedding, k: usize) -> Result<Vec<SearchResult>>;

    /// Get total number of stored chunks
    fn len(&self) -> usize;

    /// Check if store is empty
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Clear all stored data
    fn clear(&mut self);
}

mod memory;

pub use memory::*;
