use crate::chunk::Chunk;
use crate::Result;

pub trait Reranker {
    fn score(&mut self, query: &str, chunk: &Chunk) -> Result<f32>;
    fn rerank(&mut self, query: &str, chunks: Vec<Chunk>, top_k: usize) -> Result<Vec<(Chunk, f32)>>;
}

pub struct NoReranker;

impl Reranker for NoReranker {
    fn score(&mut self, _query: &str, _chunk: &Chunk) -> Result<f32> {
        unreachable!()
    }

    fn rerank(&mut self, _query: &str, _chunks: Vec<Chunk>, _top_k: usize) -> Result<Vec<(Chunk, f32)>> {
        unreachable!()
    }
}

mod bge;

pub use bge::*;
