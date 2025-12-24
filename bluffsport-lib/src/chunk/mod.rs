//! Document chunking strategies
//!
//! Different content types benefit from different chunking approaches:
//! - News articles: paragraph boundaries
//! - Match reports: event boundaries (goals, cards, etc.)
//! - Rule books: section/clause boundaries
//!
//! # Implementing a Chunker
//!
//! ```ignore
//! use bluffsport_lib::chunk::{Chunker, Chunk, ChunkMetadata};
//!
//! struct MyChunker { /* ... */ }
//!
//! impl Chunker for MyChunker {
//!     fn chunk(&self, content: &str, metadata: ChunkMetadata) -> Vec<Chunk> {
//!         // Your chunking logic here
//!         todo!()
//!     }
//! }
//! ```

use serde::{Deserialize, Serialize};

/// A chunk of text with its metadata
#[derive(Debug, Clone, Serialize, Deserialize, Eq, PartialEq)]
pub struct Chunk {
    /// Unique identifier for this chunk
    pub id: String,
    /// The text content of this chunk
    pub content: String,
    /// Metadata about the source and position
    pub metadata: ChunkMetadata,
}

/// Metadata associated with a chunk
#[derive(Debug, Clone, Default, Serialize, Deserialize, Eq, PartialEq)]
pub struct ChunkMetadata {
    /// Source document identifier
    pub source_id: Option<String>,
    /// Source type (e.g., "news", "match_report", "rulebook")
    pub source_type: Option<String>,
    /// Position within the source document (0-indexed)
    pub position: usize,
    /// Total number of chunks from this source
    pub total_chunks: Option<usize>,
    /// Optional timestamp for time-sensitive content
    pub timestamp: Option<String>,
    /// Arbitrary additional metadata
    #[serde(flatten)]
    pub extra: std::collections::HashMap<String, String>,
}

/// Trait for document chunking strategies
///
/// Implement this trait to create custom chunking logic for different
/// content types.
pub trait Chunker: Send + Sync {
    /// Split content into chunks
    ///
    /// # Arguments
    /// * `content` - The text content to chunk
    /// * `metadata` - Base metadata to attach to each chunk
    ///
    /// # Returns
    /// A vector of chunks, each with unique IDs and position metadata
    fn chunk(&self, content: &str, metadata: ChunkMetadata) -> Vec<Chunk>;

    /// Returns the name of this chunking strategy
    fn name(&self) -> &str;
}

mod fixed;
mod paragraph;

pub use fixed::*;
pub use paragraph::*;
