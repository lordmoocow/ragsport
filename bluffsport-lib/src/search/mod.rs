//! High-level search interface
//!
//! Combines embedder and store into a unified search API.
//!
//! # Usage
//!
//! ```ignore
//! use bluffsport_lib::search::SearchEngine;
//!
//! let engine = SearchEngine::new(embedder, store);
//!
//! // Index documents
//! engine.index(&chunks)?;
//!
//! // Search
//! let results = engine.search("Who won the 2023 Ashes?", 5)?;
//! for result in results {
//!     println!("{}: {:.3}", result.chunk.content, result.score);
//! }
//! ```

use crate::chunk::Chunk;
use crate::embed::Embedder;
use crate::store::{SearchResult, VectorStore};
use crate::Result;

/// High-level search engine combining embedding and storage
pub struct SearchEngine<E: Embedder, S: VectorStore> {
    embedder: E,
    store: S,
}

impl<E: Embedder, S: VectorStore> SearchEngine<E, S> {
    /// Create a new search engine
    pub fn new(embedder: E, store: S) -> Self {
        Self { embedder, store }
    }

    /// Index chunks by computing embeddings and storing them
    pub fn index(&mut self, chunks: &[Chunk]) -> Result<()> {
        if chunks.is_empty() {
            return Ok(());
        }

        // Extract text content for embedding
        let texts: Vec<&str> = chunks.iter().map(|c| c.content.as_str()).collect();

        // Compute embeddings
        let embeddings = self.embedder.embed_documents(&texts)?;

        // Store chunks with embeddings
        self.store.insert(chunks, &embeddings)?;

        Ok(())
    }

    /// Search for chunks similar to the query
    pub fn search(&mut self, query: &str, k: usize) -> Result<Vec<SearchResult>> {
        let query_embedding = self.embedder.embed_query(query)?;
        self.store.search(&query_embedding, k)
    }

    /// Get the number of indexed chunks
    pub fn len(&self) -> usize {
        self.store.len()
    }

    /// Check if the index is empty
    pub fn is_empty(&self) -> bool {
        self.store.is_empty()
    }

    /// Get reference to the embedder
    pub fn embedder(&self) -> &E {
        &self.embedder
    }

    /// Get reference to the store
    pub fn store(&self) -> &S {
        &self.store
    }

    /// Get mutable reference to the store
    pub fn store_mut(&mut self) -> &mut S {
        &mut self.store
    }
}
