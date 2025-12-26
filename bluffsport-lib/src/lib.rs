//! BluffSport - RAG library for sports knowledge retrieval
//!
//! # Architecture
//!
//! ```text
//! Document -> Chunker -> Embedder -> Store
//!                                      |
//! Query -> Embedder -> Search <--------+
//!                         |
//!                      Results
//! ```
//!
//! # Example
//!
//! ```ignore
//! use bluffsport_lib::{chunk::ParagraphChunker, embed::BgeEmbedder, store::MemoryStore};
//!
//! let chunker = ParagraphChunker::new();
//! let embedder = BgeEmbedder::new()?;
//! let store = MemoryStore::new();
//!
//! // Index a document
//! let chunks = chunker.chunk(&document);
//! let embeddings = embedder.embed(&chunks)?;
//! store.insert(&chunks, &embeddings)?;
//!
//! // Search
//! let query_embedding = embedder.embed_query("Who won the match?")?;
//! let results = store.search(&query_embedding, 5)?;
//! ```

pub mod chunk;
pub mod embed;
pub mod error;
pub mod search;
pub mod store;
pub mod rerank;

pub use error::{Error, Result};
