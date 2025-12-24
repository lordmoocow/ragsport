use crate::chunk::{Chunk, ChunkMetadata, Chunker};
use std::hash::{DefaultHasher, Hash, Hasher};

/// Paragraph chunker - splits on paragraph boundaries
///
/// Good for: news articles, prose content
///
/// Parameters to consider:
/// - min_chunk_size: merge small paragraphs
/// - max_chunk_size: split large paragraphs
pub struct ParagraphChunker {
    pub min_size: usize,
    pub max_size: usize,
}

impl Chunker for ParagraphChunker {
    fn name(&self) -> &str {
        return "paragraph"
    }

    fn chunk(&self, content: &str, mut metadata: ChunkMetadata) -> Vec<Chunk> {
        let paragraphs: Vec<_> = Paragraphs::from(content).collect();

        // set common chunking related metadata
        let total = paragraphs.len();
        metadata.total_chunks = Some(total);

        // divide content by chunk length and construct chunks
        let mut chunks = Vec::with_capacity(total);
        let mut buffer = "".to_owned();
        let mut start = 0;
        for (i, p) in paragraphs {
            // build up buffer until at least min size
            buffer = match buffer.len() {
                0 => {
                    start = i;
                    p.to_string()
                },
                _ => format!("{buffer}\n\n{p}"),
            };

            if buffer.len() < self.min_size {
                continue;
            } else {
                // clone metadata and add chunk specific info
                let mut m = metadata.clone();
                m.position = start;

                // add chunk to collection
                chunks.push(Chunk {
                    id: generate_id(&buffer),
                    content: buffer.clone(),
                    metadata: m,
                });

                // clear buffer for next chunk
                buffer.clear();
            }
        }

        // flush buffer chunk
        if buffer.len() > 0 {
            let mut m = metadata.clone();
            m.position = start;

            // add chunk to collection
            chunks.push(Chunk {
                id: generate_id(&buffer),
                content: buffer.clone(),
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

struct Paragraphs<'a> {
    s: &'a str,
    pos: usize,
}

impl<'a> Paragraphs<'a> {
    fn from(s: &'a str) -> Self {
        Self { s, pos: 0 }
    }
}

impl<'a> Iterator for Paragraphs<'a> {
    type Item = (usize, &'a str);
    
    fn next(&mut self) -> Option<Self::Item> {
        // Find the start of the paragraph.
        let mut pos = loop {
            if self.s.is_empty() {
                return None
            }
            let (line, rest) = split_first_line(self.s);
            if line.chars().all(char::is_whitespace) {
                // Discard blank line.
                self.s = rest;
            } else {
                // Found a non-blank line.
                break line.len();
            }
        };

        // Find the end of the paragraph.
        loop {
            let (line, rest) = split_first_line(&self.s[pos..]);
            self.pos += line.len();
            if line.chars().all(char::is_whitespace) {
                // Blank line or empty line: end of paragraph.
                let result = &self.s[..pos];
                self.s = rest;
                return Some((self.pos, result));
            }
            // Non-blank line: continue looping.
            pos += line.len();
        }
    }
}

fn split_first_line(s: &str) -> (&str, &str) {
    let len = match s.find('\n') {
        Some(i) => i + 1,
        None => s.len(),
    };
    s.split_at(len)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn meta() -> ChunkMetadata {
        ChunkMetadata::default()
    }

    #[test]
    fn test_single_paragraph() {
        let chunker = ParagraphChunker { min_size: 0, max_size: 1000 };
        let content = "This is a single paragraph.";
        let chunks = chunker.chunk(content, meta());

        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].content, "This is a single paragraph.");
    }

    #[test]
    fn test_multiple_paragraphs() {
        let chunker = ParagraphChunker { min_size: 0, max_size: 1000 };
        let content = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph.";
        let chunks = chunker.chunk(content, meta());

        assert_eq!(chunks.len(), 3);
        assert!(chunks[0].content.contains("First"));
        assert!(chunks[1].content.contains("Second"));
        assert!(chunks[2].content.contains("Third"));
    }

    #[test]
    fn test_min_size_merging() {
        let chunker = ParagraphChunker { min_size: 50, max_size: 1000 };
        let content = "Short.\n\nAlso short.\n\nThis one is a bit longer paragraph.";
        let chunks = chunker.chunk(content, meta());

        // Small paragraphs should be merged until min_size is reached
        assert!(chunks.len() < 3);
    }

    #[test]
    fn test_trailing_content_flushed() {
        let chunker = ParagraphChunker { min_size: 100, max_size: 1000 };
        let content = "Short para one.\n\nShort para two.";
        let chunks = chunker.chunk(content, meta());

        // Even though combined < min_size, trailing content should be flushed
        assert!(!chunks.is_empty());
        // The content should contain both paragraphs
        let all_content: String = chunks.iter().map(|c| c.content.clone()).collect();
        assert!(all_content.contains("one"));
        assert!(all_content.contains("two"));
    }

    #[test]
    fn test_empty_content() {
        let chunker = ParagraphChunker { min_size: 0, max_size: 1000 };
        let chunks = chunker.chunk("", meta());
        assert!(chunks.is_empty());
    }

    #[test]
    fn test_whitespace_only() {
        let chunker = ParagraphChunker { min_size: 0, max_size: 1000 };
        let chunks = chunker.chunk("\n\n\n   \n\n", meta());
        assert!(chunks.is_empty());
    }

    #[test]
    fn test_paragraphs_iterator() {
        let content = "Para one.\n\nPara two.\n\nPara three.";
        let paras: Vec<_> = Paragraphs::from(content).collect();

        assert_eq!(paras.len(), 3);
    }

    #[test]
    fn test_unique_ids() {
        let chunker = ParagraphChunker { min_size: 0, max_size: 1000 };
        let content = "First unique paragraph.\n\nSecond unique paragraph.";
        let chunks = chunker.chunk(content, meta());

        assert_eq!(chunks.len(), 2);
        assert_ne!(chunks[0].id, chunks[1].id);
    }
}
