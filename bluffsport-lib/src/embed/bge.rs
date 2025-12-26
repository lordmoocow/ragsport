use fastembed::{EmbeddingModel, InitOptions, TextEmbedding};

use crate::embed::{Embedder, Embedding};
use crate::{Error, Result};

/// BGE embedder using BAAI/bge-large-en-v1.5.
///
/// Uses fastembed for ONNX-based inference. This model produces 1024-dimensional
/// embeddings and supports up to 512 tokens per input.
pub struct BgeEmbedder {
    model: TextEmbedding,
}

impl BgeEmbedder {
    /// Create a new BGE embedder.
    ///
    /// Downloads the model on first use (~1.2GB).
    pub fn new() -> Result<Self> {
        let opts = InitOptions::new(EmbeddingModel::BGELargeENV15)
            .with_show_download_progress(true);

        TextEmbedding::try_new(opts)
            .map(|model| Self { model })
            .map_err(|e| Error::Embedding(e.to_string()))
    }
}

impl Embedder for BgeEmbedder {
    fn model_name(&self) -> &str {
        "BAAI/bge-large-en-v1.5"
    }

    fn dimension(&self) -> usize {
        1024
    }

    fn embed_documents(&mut self, texts: &[&str]) -> Result<Vec<Embedding>> {
        self.model
            .embed(texts, None)
            .map_err(|e| Error::Embedding(e.to_string()))
    }

    fn embed_query(&mut self, text: &str) -> Result<Embedding> {
        // BGE uses a special prompt prefix for queries to improve retrieval
        let query_text = format!("Represent this sentence for searching relevant passages: {text}");

        self.model
            .embed(vec![query_text], None)
            .map_err(|e| Error::Embedding(e.to_string()))?
            .into_iter()
            .next()
            .ok_or_else(|| Error::Embedding("model returned no embeddings".to_string()))
    }
}
