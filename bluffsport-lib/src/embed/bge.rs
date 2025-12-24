use fastembed::{InitOptions, TextEmbedding};

use crate::{Error, Result, embed::Embedder};

/// BGE embedder using BAAI/bge-large-en-v1.5
///
/// Uses fastembed for ONNX-based inference.
pub struct BgeEmbedder {
    model: TextEmbedding,
}

impl BgeEmbedder {
    /// Create a new BGE embedder
    ///
    /// This will download the model on first use (~1.2GB).
    pub fn new() -> Result<Self> {
        match TextEmbedding::try_new(
            InitOptions::new(fastembed::EmbeddingModel::BGELargeENV15)
                .with_show_download_progress(true),
        ) {
            Err(e) => Err(Error::Embedding(e.to_string())),
            Ok(model) => Ok(Self { model }),
        }
    }
}

impl Embedder for BgeEmbedder {
    fn model_name(&self) -> &str {
        "BAAI/bge-large-en-v1.5"
    }

    fn embed_documents(&mut self, texts: &[&str]) -> Result<Vec<super::Embedding>> {
        match self.model.embed(texts, None) {
            Err(e) => Err(Error::Embedding(e.to_string())),
            Ok(v) => Ok(v),
        }
    }

    fn embed_query(&mut self, text: &str) -> Result<super::Embedding> {
        let embeds = match self.model.embed(
            vec![format!(
                "Represent this sentence for searching relevant passages: {text}"
            )],
            None,
        ) {
            Err(e) => return Err(Error::Embedding(e.to_string())),
            Ok(embeds) => embeds
        };

        if let Some(e) = embeds.first() {
            Ok(e.to_vec())
        } else {
            Err(Error::Embedding("failed".to_string()))
        }
    }

    fn dimension(&self) -> usize {
        1024
    }
}
