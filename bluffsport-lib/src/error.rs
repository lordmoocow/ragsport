//! Error types for BluffSport

use thiserror::Error;

/// Result type alias for BluffSport operations
pub type Result<T> = std::result::Result<T, Error>;

/// Errors that can occur in BluffSport operations
#[derive(Error, Debug)]
pub enum Error {
    /// Failed to load or run the embedding model
    #[error("embedding error: {0}")]
    Embedding(String),

    /// Failed to chunk a document
    #[error("chunking error: {0}")]
    Chunking(String),

    /// Failed to store or retrieve from vector store
    #[error("store error: {0}")]
    Store(String),

    /// Document or chunk not found
    #[error("not found: {0}")]
    NotFound(String),

    /// Invalid input provided
    #[error("invalid input: {0}")]
    InvalidInput(String),
}
