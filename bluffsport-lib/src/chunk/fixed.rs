use std::hash::{DefaultHasher, Hash, Hasher};

use crate::chunk::{Chunk, ChunkMetadata, Chunker};

/// Fixed-size chunker - splits by character/token count
///
/// Good for: baseline experiments, consistent chunk sizes
///
/// Parameters to consider:
/// - chunk_size: target size in characters or tokens
/// - overlap: how much overlap between adjacent chunks
pub struct FixedSizeChunker {
    pub chunk_size: usize,
    pub overlap: usize,
}

impl Chunker for FixedSizeChunker {
    fn name(&self) -> &str {
        return "fixed";
    }

    fn chunk(&self, content: &str, mut metadata: ChunkMetadata) -> Vec<Chunk> {
        // set common chunking related metadata
        let stride = self.chunk_size.saturating_sub(self.overlap);
        let total = content.len().div_ceil(stride);
        metadata.total_chunks = Some(total);

        // divide content by chunk length and construct chunks
        let mut chunks = Vec::with_capacity(total);
        for i in 0..total {
            // determine start/end of slice on utf8 char boundaries
            let start = content.floor_char_boundary(i * stride);
            let end = content.ceil_char_boundary(start + self.chunk_size);
            
            // slice the content for the current chunk
            let c = &content[start..end];

            // clone metadata and add chunk specific info
            let mut m = metadata.clone();
            m.position = start;

            // add chunk to collection
            chunks.push(Chunk {
                id: generate_id(c),
                content: c.to_string(),
                metadata: m,
            });
        }
        chunks
    }
}

fn generate_id(string: &str) -> String {
    let mut hasher = DefaultHasher::new();
    string.hash(&mut hasher);
    format!("{:x}", hasher.finish())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn meta() -> ChunkMetadata {
        ChunkMetadata::default()
    }

    #[test]
    fn test_basic_chunking() {
        let chunker = FixedSizeChunker { chunk_size: 10, overlap: 0 };
        let content = "0123456789abcdefghij"; // 20 chars
        let chunks = chunker.chunk(content, meta());

        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].content, "0123456789");
        assert_eq!(chunks[1].content, "abcdefghij");
    }

    #[test]
    fn test_overlap() {
        let chunker = FixedSizeChunker { chunk_size: 10, overlap: 5 };
        let content = "0123456789abcdefghij"; // 20 chars, stride=5
        let chunks = chunker.chunk(content, meta());

        // stride=5, so positions: 0, 5, 10, 15
        assert_eq!(chunks.len(), 4);
        assert_eq!(chunks[0].content, "0123456789");  // 0..10
        assert_eq!(chunks[1].content, "56789abcde");  // 5..15
        assert_eq!(chunks[2].content, "abcdefghij");  // 10..20
        assert_eq!(chunks[3].content, "fghij");       // 15..20 (truncated)
    }

    #[test]
    fn test_unicode_safety() {
        let chunker = FixedSizeChunker { chunk_size: 5, overlap: 0 };
        let content = "Hello 👋 World"; // emoji is 4 bytes

        // Should not panic on unicode boundaries
        let chunks = chunker.chunk(content, meta());
        assert!(!chunks.is_empty());

        // All chunks should be valid UTF-8 (implicit - would panic otherwise)
        for chunk in &chunks {
            assert!(!chunk.content.is_empty());
        }
    }

    #[test]
    fn test_unique_ids() {
        let chunker = FixedSizeChunker { chunk_size: 5, overlap: 0 };
        let content = "aaaaabbbbb"; // different content = different IDs
        let chunks = chunker.chunk(content, meta());

        assert_eq!(chunks.len(), 2);
        assert_ne!(chunks[0].id, chunks[1].id);
    }

    #[test]
    fn test_same_content_same_id() {
        let chunker = FixedSizeChunker { chunk_size: 5, overlap: 0 };
        let content = "aaaaaaaaaa"; // same content chunks
        let chunks = chunker.chunk(content, meta());

        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].id, chunks[1].id); // same content = same hash
    }

    #[test]
    fn test_metadata_position() {
        let chunker = FixedSizeChunker { chunk_size: 10, overlap: 0 };
        let content = "0123456789abcdefghij";
        let chunks = chunker.chunk(content, meta());

        assert_eq!(chunks[0].metadata.position, 0);
        assert_eq!(chunks[1].metadata.position, 10);
    }

    #[test]
    fn test_empty_content() {
        let chunker = FixedSizeChunker { chunk_size: 10, overlap: 0 };
        let chunks = chunker.chunk("", meta());
        assert!(chunks.is_empty());
    }
}