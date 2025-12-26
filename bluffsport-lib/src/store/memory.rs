use std::collections::{BinaryHeap, HashMap};

use crate::chunk::Chunk;
use crate::embed::Embedding;
use crate::store::{SearchResult, VectorStore};
use crate::Result;

/// In-memory vector store for development and testing.
///
/// Uses brute-force cosine similarity search. Suitable for small datasets
/// (< 10k chunks). For production, use a proper vector database.
pub struct MemoryStore {
    chunks: HashMap<String, Chunk>,
    embeddings: HashMap<String, Embedding>,
}

impl MemoryStore {
    /// Create a new empty in-memory store.
    #[must_use]
    pub fn new() -> Self {
        Self {
            chunks: HashMap::new(),
            embeddings: HashMap::new(),
        }
    }
}

impl Default for MemoryStore {
    fn default() -> Self {
        Self::new()
    }
}

impl VectorStore for MemoryStore {
    fn insert(&mut self, chunks: &[Chunk], embeddings: &[Embedding]) -> Result<()> {
        for (chunk, embedding) in chunks.iter().zip(embeddings) {
            self.chunks.insert(chunk.id.clone(), chunk.clone());
            self.embeddings.insert(chunk.id.clone(), embedding.clone());
        }
        Ok(())
    }

    fn search(&self, query: &Embedding, k: usize) -> Result<Vec<SearchResult>> {
        let mut results = BinaryHeap::with_capacity(self.chunks.len());

        for (id, chunk) in &self.chunks {
            let embedding = self
                .embeddings
                .get(id)
                .expect("chunk and embedding should have matching ids");

            results.push(SearchResult {
                chunk: chunk.clone(),
                score: cosine_similarity(query, embedding),
            });
        }

        Ok(results
            .into_sorted_vec()
            .into_iter()
            .rev()
            .take(k)
            .collect())
    }

    fn len(&self) -> usize {
        self.chunks.len()
    }

    fn clear(&mut self) {
        self.chunks.clear();
        self.embeddings.clear();
    }
}

/// Compute cosine similarity between two vectors.
///
/// Returns a value in [-1, 1] where 1 means identical direction.
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len(), "vectors must have same length");

    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }

    dot / (norm_a * norm_b)
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
    fn test_cosine_similarity_identical() {
        let a = vec![1.0, 0.0, 0.0];
        let sim = cosine_similarity(&a, &a);
        assert!((sim - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_similarity_orthogonal() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![0.0, 1.0, 0.0];
        let sim = cosine_similarity(&a, &b);
        assert!(sim.abs() < 1e-6);
    }

    #[test]
    fn test_cosine_similarity_opposite() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![-1.0, 0.0, 0.0];
        let sim = cosine_similarity(&a, &b);
        assert!((sim + 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_insert_and_len() {
        let mut store = MemoryStore::new();
        assert_eq!(store.len(), 0);
        assert!(store.is_empty());

        let chunks = vec![make_chunk("1", "hello"), make_chunk("2", "world")];
        let embeddings = vec![vec![1.0, 0.0], vec![0.0, 1.0]];

        store.insert(&chunks, &embeddings).unwrap();
        assert_eq!(store.len(), 2);
        assert!(!store.is_empty());
    }

    #[test]
    fn test_search_returns_sorted() {
        let mut store = MemoryStore::new();

        let chunks = vec![
            make_chunk("1", "far away"),
            make_chunk("2", "very close"),
            make_chunk("3", "medium"),
        ];
        // Query will be [1, 0, 0]
        let embeddings = vec![
            vec![0.0, 1.0, 0.0], // orthogonal to query
            vec![1.0, 0.0, 0.0], // identical to query
            vec![0.5, 0.5, 0.0], // somewhat similar
        ];

        store.insert(&chunks, &embeddings).unwrap();

        let query = vec![1.0, 0.0, 0.0];
        let results = store.search(&query, 3).unwrap();

        assert_eq!(results.len(), 3);
        assert_eq!(results[0].chunk.id, "2"); // highest similarity
        assert_eq!(results[1].chunk.id, "3"); // medium
        assert_eq!(results[2].chunk.id, "1"); // lowest
    }

    #[test]
    fn test_search_respects_k() {
        let mut store = MemoryStore::new();

        let chunks = vec![
            make_chunk("1", "a"),
            make_chunk("2", "b"),
            make_chunk("3", "c"),
        ];
        let embeddings = vec![vec![1.0, 0.0], vec![0.9, 0.1], vec![0.8, 0.2]];

        store.insert(&chunks, &embeddings).unwrap();

        let query = vec![1.0, 0.0];
        let results = store.search(&query, 2).unwrap();

        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_search_k_larger_than_store() {
        let mut store = MemoryStore::new();

        let chunks = vec![make_chunk("1", "only one")];
        let embeddings = vec![vec![1.0, 0.0]];

        store.insert(&chunks, &embeddings).unwrap();

        let query = vec![1.0, 0.0];
        let results = store.search(&query, 100).unwrap();

        assert_eq!(results.len(), 1);
    }

    #[test]
    fn test_clear() {
        let mut store = MemoryStore::new();

        let chunks = vec![make_chunk("1", "hello")];
        let embeddings = vec![vec![1.0]];

        store.insert(&chunks, &embeddings).unwrap();
        assert_eq!(store.len(), 1);

        store.clear();
        assert_eq!(store.len(), 0);
        assert!(store.is_empty());
    }

    #[test]
    fn test_deduplication_by_id() {
        let mut store = MemoryStore::new();

        // Insert same ID twice with different content
        let chunks1 = vec![make_chunk("same-id", "first content")];
        let chunks2 = vec![make_chunk("same-id", "second content")];
        let embeddings = vec![vec![1.0]];

        store.insert(&chunks1, &embeddings).unwrap();
        store.insert(&chunks2, &embeddings).unwrap();

        assert_eq!(store.len(), 1);
    }

    #[test]
    fn test_empty_search() {
        let store = MemoryStore::new();
        let query = vec![1.0, 0.0];
        let results = store.search(&query, 5).unwrap();
        assert!(results.is_empty());
    }
}
