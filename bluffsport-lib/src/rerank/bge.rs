use fastembed::{RerankInitOptions, RerankerModel, TextRerank};

use crate::chunk::Chunk;
use crate::rerank::Reranker;
use crate::{Error, Result};

/// BGE reranker using BAAI/bge-reranker-base.
///
/// Cross-encoder model for two-stage retrieval. Scores query-document pairs
/// together for more accurate relevance ranking than bi-encoder similarity.
pub struct BgeReranker {
    model: TextRerank,
}

impl BgeReranker {
    /// Create a new BGE reranker.
    ///
    /// Downloads the model on first use (~300MB).
    pub fn new() -> Result<Self> {
        let opts = RerankInitOptions::new(RerankerModel::BGERerankerBase)
            .with_show_download_progress(true);

        TextRerank::try_new(opts)
            .map(|model| Self { model })
            .map_err(|e| Error::Reranking(e.to_string()))
    }
}

impl Reranker for BgeReranker {
    fn score(&mut self, query: &str, chunk: &Chunk) -> Result<f32> {
        self.rerank(query, vec![chunk.clone()], 1)
            .map(|results| results.first().map_or(0.0, |(_, score)| *score))
    }

    fn rerank(
        &mut self,
        query: &str,
        chunks: Vec<Chunk>,
        top_k: usize,
    ) -> Result<Vec<(Chunk, f32)>> {
        // Wrap chunks in Option for ownership transfer after reranking
        let mut chunks: Vec<Option<Chunk>> = chunks.into_iter().map(Some).collect();

        // Extract document texts for the model
        let docs: Vec<&str> = chunks
            .iter()
            .map(|c| c.as_ref().expect("chunks are Some").content.as_str())
            .collect();

        // Rerank and get sorted results with indices
        let results = self
            .model
            .rerank(query, &docs, false, None)
            .map_err(|e| Error::Reranking(e.to_string()))?;

        // Take top_k results, moving chunks out by index
        Ok(results
            .into_iter()
            .take(top_k)
            .map(|rr| {
                let chunk = chunks[rr.index]
                    .take()
                    .expect("each index should be used once");
                (chunk, rr.score)
            })
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chunk::ChunkMetadata;

    fn make_chunk(id: &str, content: &str) -> Chunk {
        Chunk {
            id: id.to_string(),
            content: content.to_string(),
            metadata: ChunkMetadata::default(),
        }
    }

    #[test]
    #[ignore] // Requires model download, run with: cargo test -- --ignored
    fn test_relevant_scores_higher_than_irrelevant() {
        let mut reranker = BgeReranker::new().unwrap();

        let query = "Who won the Ashes cricket series?";
        let relevant = make_chunk("1", "England won the Ashes series 4-1 against Australia in 2023.");
        let irrelevant = make_chunk("2", "The weather in London was cloudy with occasional rain.");

        let score_relevant = reranker.score(query, &relevant).unwrap();
        let score_irrelevant = reranker.score(query, &irrelevant).unwrap();

        assert!(
            score_relevant > score_irrelevant,
            "Relevant chunk should score higher: {score_relevant:.4} vs {score_irrelevant:.4}",
        );
    }

    #[test]
    #[ignore] // Requires model download
    fn test_rerank_returns_sorted_by_relevance() {
        let mut reranker = BgeReranker::new().unwrap();

        let query = "What was the final score?";
        let chunks = vec![
            make_chunk("1", "The stadium was packed with fans."),
            make_chunk("2", "The final score was 3-2 after extra time."),
            make_chunk("3", "Weather conditions were perfect for the match."),
        ];

        let results = reranker.rerank(query, chunks, 3).unwrap();

        // The "final score" chunk should be ranked first
        assert_eq!(results[0].0.id, "2", "Score-related chunk should rank first");

        // Scores should be in descending order
        for window in results.windows(2) {
            assert!(
                window[0].1 >= window[1].1,
                "Results should be sorted by score descending"
            );
        }
    }

    #[test]
    #[ignore] // Requires model download
    fn test_rerank_respects_top_k() {
        let mut reranker = BgeReranker::new().unwrap();

        let query = "test query";
        let chunks = vec![
            make_chunk("1", "First document"),
            make_chunk("2", "Second document"),
            make_chunk("3", "Third document"),
            make_chunk("4", "Fourth document"),
        ];

        let results = reranker.rerank(query, chunks, 2).unwrap();

        assert_eq!(results.len(), 2, "Should return exactly top_k results");
    }
}
