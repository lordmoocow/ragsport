//! High-level search interface
//!
//! Combines embedder and store into a unified search API.
//!
//! # Usage
//!
//! ```ignore
//! use bluffsport_lib::search::SearchEngine;
//!
//! // Basic search (bi-encoder only)
//! let mut engine = SearchEngine::new(embedder, store);
//! engine.index(&chunks)?;
//! let results = engine.search("Who won the 2023 Ashes?", 5)?;
//!
//! // Two-stage search with reranking
//! let mut engine = SearchEngine::with_rerank(embedder, store, reranker);
//! engine.index(&chunks)?;
//! let results = engine.search_reranked("Who won?", 5, 20)?; // top 5 from 20 candidates
//! ```

use crate::chunk::Chunk;
use crate::embed::Embedder;
use crate::rerank::{NoReranker, Reranker};
use crate::store::{SearchResult, VectorStore};
use crate::Result;

/// High-level search engine combining embedding, storage, and optional reranking.
pub struct SearchEngine<E: Embedder, S: VectorStore, R: Reranker = NoReranker> {
    embedder: E,
    store: S,
    reranker: Option<R>,
}

// Constructor for engines without reranking
impl<E: Embedder, S: VectorStore> SearchEngine<E, S, NoReranker> {
    /// Create a new search engine without reranking.
    #[must_use]
    pub fn new(embedder: E, store: S) -> Self {
        Self {
            embedder,
            store,
            reranker: None,
        }
    }
}

// Constructor for engines with reranking
impl<E: Embedder, S: VectorStore, R: Reranker> SearchEngine<E, S, R> {
    /// Create a new search engine with reranking enabled.
    #[must_use]
    pub fn with_rerank(embedder: E, store: S, reranker: R) -> Self {
        Self {
            embedder,
            store,
            reranker: Some(reranker),
        }
    }
}

// Common methods available on all engines
impl<E: Embedder, S: VectorStore, R: Reranker> SearchEngine<E, S, R> {
    /// Index chunks by computing embeddings and storing them.
    pub fn index(&mut self, chunks: &[Chunk]) -> Result<()> {
        if chunks.is_empty() {
            return Ok(());
        }

        let texts: Vec<&str> = chunks.iter().map(|c| c.content.as_str()).collect();
        let embeddings = self.embedder.embed_documents(&texts)?;
        self.store.insert(chunks, &embeddings)?;

        Ok(())
    }

    /// Search for chunks similar to the query using bi-encoder similarity.
    ///
    /// This performs a single-stage search using vector similarity only.
    /// For better precision, use [`search_reranked`](Self::search_reranked) if a reranker is configured.
    pub fn search(&mut self, query: &str, k: usize) -> Result<Vec<SearchResult>> {
        let query_embedding = self.embedder.embed_query(query)?;
        self.store.search(&query_embedding, k)
    }

    /// Two-stage search: retrieve candidates with bi-encoder, rerank with cross-encoder.
    ///
    /// # Arguments
    /// * `query` - The search query
    /// * `k` - Number of final results to return
    /// * `n` - Number of candidates to retrieve for reranking (should be >= k)
    ///
    /// Falls back to basic search if no reranker is configured.
    pub fn search_reranked(
        &mut self,
        query: &str,
        k: usize,
        n: usize,
    ) -> Result<Vec<SearchResult>> {
        let query_embedding = self.embedder.embed_query(query)?;

        let Some(reranker) = &mut self.reranker else {
            // No reranker configured, fall back to basic search
            return self.store.search(&query_embedding, k);
        };

        let candidates = self.store.search(&query_embedding, n)?;
        let chunks: Vec<Chunk> = candidates.into_iter().map(|r| r.chunk).collect();
        let ranked = reranker.rerank(query, chunks, k)?;

        Ok(ranked
            .into_iter()
            .map(|(chunk, score)| SearchResult { chunk, score })
            .collect())
    }

    /// Returns the number of indexed chunks.
    #[must_use]
    pub fn len(&self) -> usize {
        self.store.len()
    }

    /// Returns `true` if no chunks are indexed.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.store.is_empty()
    }

    /// Returns a reference to the embedder.
    #[must_use]
    pub fn embedder(&self) -> &E {
        &self.embedder
    }

    /// Returns a reference to the store.
    #[must_use]
    pub fn store(&self) -> &S {
        &self.store
    }

    /// Returns a mutable reference to the store.
    pub fn store_mut(&mut self) -> &mut S {
        &mut self.store
    }
}
