//! BluffSport CLI - for testing the RAG library
//!
//! # Commands
//!
//! ```bash
//! # Chunk a document and show results
//! bluffsport chunk --strategy paragraph input.txt
//!
//! # Embed text and show vector stats
//! bluffsport embed "Who won the 2023 Ashes?"
//!
//! # Demo: index a file and search it
//! bluffsport demo input.txt "who scored"
//! ```

use std::fs;

use anyhow::Result;
use bluffsport_lib::{
    chunk::{Chunk, ChunkMetadata, Chunker, FixedSizeChunker, ParagraphChunker},
    embed::{BgeEmbedder, Embedder},
    rerank::BgeReranker,
    search::SearchEngine,
    store::MemoryStore,
};
use clap::{Parser, Subcommand};
use tracing_subscriber::EnvFilter;

#[derive(Parser)]
#[command(name = "bluffsport")]
#[command(about = "RAG library for sports knowledge retrieval")]
#[command(version)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Chunk a document using specified strategy
    Chunk {
        /// Input file to chunk
        input: String,

        /// Chunking strategy: "fixed" or "paragraph"
        #[arg(short, long, default_value = "paragraph")]
        strategy: String,

        /// Chunk size (for fixed strategy) / max size (for paragraph)
        #[arg(long, default_value = "512")]
        size: usize,

        /// Overlap (for fixed) / min size (for paragraph)
        #[arg(long, default_value = "50")]
        overlap: usize,
    },

    /// Embed text and show vector info
    Embed {
        /// Text to embed
        text: String,

        /// Treat as query (uses query prompt prefix)
        #[arg(short, long)]
        query: bool,
    },

    /// Demo: index a file and search it (all in one command)
    Demo {
        /// Input file to index
        input: String,

        /// Query to search for
        query: String,

        /// Number of results to return
        #[arg(short, long, default_value = "3")]
        k: usize,

        /// Number of candidates for reranking (only used with --rerank)
        #[arg(short, long, default_value = "20")]
        n: usize,

        /// Chunking strategy
        #[arg(long, default_value = "paragraph")]
        strategy: String,

        /// Enable reranking (two-stage retrieval with cross-encoder)
        #[arg(short, long)]
        rerank: bool,
    },
}

fn chunk_text(text: &str, strategy: &str, size: usize, overlap: usize) -> Vec<Chunk> {
    let meta = ChunkMetadata::default();
    match strategy {
        "fixed" => {
            let chunker = FixedSizeChunker { chunk_size: size, overlap };
            chunker.chunk(text, meta)
        }
        _ => {
            let chunker = ParagraphChunker { max_size: size, min_size: overlap };
            chunker.chunk(text, meta)
        }
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env())
        .init();

    let cli = Cli::parse();

    match cli.command {
        Commands::Chunk {
            input,
            strategy,
            size,
            overlap,
        } => {
            let text = fs::read_to_string(&input)?;
            let chunks = chunk_text(&text, &strategy, size, overlap);

            println!("Chunked '{}' into {} chunks using {} strategy:\n", input, chunks.len(), strategy);
            for (i, chunk) in chunks.iter().enumerate() {
                println!("--- Chunk {} ({}B, id: {}) ---", i + 1, chunk.content.len(), &chunk.id[..8]);
                // Show preview (first 200 chars)
                let preview: String = chunk.content.chars().take(200).collect();
                println!("{}{}\n", preview, if chunk.content.len() > 200 { "..." } else { "" });
            }
        }

        Commands::Embed { text, query } => {
            println!("Loading BGE model (first run downloads ~1.2GB)...");
            let mut embedder = BgeEmbedder::new()?;

            let embedding = if query {
                println!("Embedding as query: {}", text);
                embedder.embed_query(&text)?
            } else {
                println!("Embedding as document: {}", text);
                embedder.embed_documents(&[text.as_str()])?
                    .into_iter()
                    .next()
                    .expect("should have one embedding")
            };

            println!("\nEmbedding stats:");
            println!("  Dimensions: {}", embedding.len());
            println!("  First 5 values: {:?}", &embedding[..5]);
            println!("  Min: {:.4}", embedding.iter().cloned().fold(f32::INFINITY, f32::min));
            println!("  Max: {:.4}", embedding.iter().cloned().fold(f32::NEG_INFINITY, f32::max));
        }

        Commands::Demo {
            input,
            query,
            k,
            n,
            strategy,
            rerank,
        } => {
            // Load and chunk document
            println!("Loading '{input}'...");
            let text = fs::read_to_string(&input)?;
            let chunks = chunk_text(&text, &strategy, 512, 100);
            println!("Created {} chunks using {strategy} strategy", chunks.len());

            // Initialize embedder and store
            println!("\nLoading BGE model (first run downloads ~1.2GB)...");
            let embedder = BgeEmbedder::new()?;
            let store = MemoryStore::new();

            // Search with or without reranking
            let results = if rerank {
                println!("Loading reranker model...");
                let reranker = BgeReranker::new()?;
                let mut engine = SearchEngine::with_rerank(embedder, store, reranker);
                println!("Indexing {} chunks...", chunks.len());
                engine.index(&chunks)?;
                println!("Done! Index contains {} chunks", engine.len());
                println!("\nSearching: '{query}' (k={k}, n={n}, reranking enabled)");
                engine.search_reranked(&query, k, n)?
            } else {
                let mut engine = SearchEngine::new(embedder, store);
                println!("Indexing {} chunks...", chunks.len());
                engine.index(&chunks)?;
                println!("Done! Index contains {} chunks", engine.len());
                println!("\nSearching: '{query}' (k={k})");
                engine.search(&query, k)?
            };

            println!("\n=== Results ===\n");
            for (i, result) in results.iter().enumerate() {
                println!("#{} (score: {:.4})", i + 1, result.score);
                println!("---");
                let preview: String = result.chunk.content.chars().take(300).collect();
                let ellipsis = if result.chunk.content.len() > 300 { "..." } else { "" };
                println!("{preview}{ellipsis}\n");
            }
        }
    }

    Ok(())
}
