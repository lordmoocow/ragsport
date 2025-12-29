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
//!
//! # Evaluate: run test queries and measure metrics
//! bluffsport eval --queries eval/queries.json article1.txt article2.txt
//! ```

use std::collections::HashMap;
use std::fs;
use std::path::Path;
use std::time::Instant;

use anyhow::Result;
use bluffsport_lib::{
    chunk::{Chunk, ChunkMetadata, Chunker, FixedSizeChunker, ParagraphChunker},
    embed::{BgeEmbedder, Embedder},
    rerank::BgeReranker,
    search::SearchEngine,
    store::MemoryStore,
};
use clap::{Parser, Subcommand};
use serde::Deserialize;
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

    /// Evaluate: run test queries and measure retrieval metrics
    Eval {
        /// Input files to index
        #[arg(required = true)]
        inputs: Vec<String>,

        /// JSON file with test queries
        #[arg(short, long)]
        queries: String,

        /// Number of results to return per query
        #[arg(short, long, default_value = "5")]
        k: usize,

        /// Number of candidates for reranking (only used with --rerank)
        #[arg(short, long, default_value = "20")]
        n: usize,

        /// Chunking strategy: "fixed" or "paragraph"
        #[arg(long, default_value = "paragraph")]
        strategy: String,

        /// Chunk size (for fixed) / max size (for paragraph)
        #[arg(long, default_value = "512")]
        size: usize,

        /// Overlap (for fixed) / min size (for paragraph)
        #[arg(long, default_value = "50")]
        overlap: usize,

        /// Enable reranking (two-stage retrieval with cross-encoder)
        #[arg(short, long)]
        rerank: bool,

        /// Show detailed results for each query
        #[arg(long)]
        verbose: bool,

        /// Output results as JSON (for LLM-as-judge evaluation)
        #[arg(long)]
        json: bool,
    },
}

// --- Evaluation types ---

#[derive(Debug, Deserialize)]
struct QueryFile {
    queries: Vec<TestQuery>,
}

#[derive(Debug, Deserialize)]
struct TestQuery {
    id: String,
    query: String,
    #[serde(default)]
    relevant_keywords: Vec<String>,
    #[serde(default)]
    expected_answer: String,
    #[serde(default)]
    source: String,
}

// JSON output types for LLM-as-judge
#[derive(Debug, serde::Serialize)]
struct JsonOutput {
    config: JsonConfig,
    results: Vec<JsonQueryResult>,
}

#[derive(Debug, serde::Serialize)]
struct JsonConfig {
    strategy: String,
    chunk_size: usize,
    reranking: bool,
    k: usize,
}

#[derive(Debug, serde::Serialize)]
struct JsonQueryResult {
    query_id: String,
    query: String,
    expected_answer: String,
    retrieved_chunks: Vec<JsonChunk>,
    latency_ms: u128,
}

#[derive(Debug, serde::Serialize)]
struct JsonChunk {
    rank: usize,
    score: f32,
    content: String,
}

#[derive(Debug)]
struct QueryResult {
    query_id: String,
    relevant_in_top_k: usize,
    first_relevant_rank: Option<usize>,
    latency_ms: u128,
}

#[derive(Debug)]
struct EvalMetrics {
    precision_at_k: f64,
    recall_at_k: f64,
    mrr: f64,
    avg_latency_ms: f64,
    queries_with_relevant: usize,
    total_queries: usize,
}

// --- Helper functions ---

fn chunk_text(text: &str, strategy: &str, size: usize, overlap: usize) -> Vec<Chunk> {
    let meta = ChunkMetadata::default();
    match strategy {
        "fixed" => {
            let chunker = FixedSizeChunker {
                chunk_size: size,
                overlap,
            };
            chunker.chunk(text, meta)
        }
        _ => {
            let chunker = ParagraphChunker {
                max_size: size,
                min_size: overlap,
            };
            chunker.chunk(text, meta)
        }
    }
}

fn chunk_with_source(
    text: &str,
    source: &str,
    strategy: &str,
    size: usize,
    overlap: usize,
) -> Vec<Chunk> {
    let mut meta = ChunkMetadata::default();
    meta.source_id = Some(source.to_string());
    match strategy {
        "fixed" => {
            let chunker = FixedSizeChunker {
                chunk_size: size,
                overlap,
            };
            chunker.chunk(text, meta)
        }
        _ => {
            let chunker = ParagraphChunker {
                max_size: size,
                min_size: overlap,
            };
            chunker.chunk(text, meta)
        }
    }
}

/// Check if a chunk is relevant by counting keyword matches
fn is_relevant(content: &str, keywords: &[String], min_matches: usize) -> bool {
    let content_lower = content.to_lowercase();
    let matches = keywords
        .iter()
        .filter(|kw| content_lower.contains(&kw.to_lowercase()))
        .count();
    matches >= min_matches
}

fn calculate_metrics(results: &[QueryResult], k: usize) -> EvalMetrics {
    let total = results.len();
    if total == 0 {
        return EvalMetrics {
            precision_at_k: 0.0,
            recall_at_k: 0.0,
            mrr: 0.0,
            avg_latency_ms: 0.0,
            queries_with_relevant: 0,
            total_queries: 0,
        };
    }

    // Precision@k: average fraction of relevant results in top k
    let total_relevant: usize = results.iter().map(|r| r.relevant_in_top_k).sum();
    let precision = total_relevant as f64 / (total * k) as f64;

    // For recall, we assume each query has exactly 1 relevant doc
    // So recall@k = fraction of queries where we found at least 1 relevant
    let queries_with_relevant = results.iter().filter(|r| r.relevant_in_top_k > 0).count();
    let recall = queries_with_relevant as f64 / total as f64;

    // MRR: mean reciprocal rank
    let mrr: f64 = results
        .iter()
        .map(|r| match r.first_relevant_rank {
            Some(rank) => 1.0 / rank as f64,
            None => 0.0,
        })
        .sum::<f64>()
        / total as f64;

    // Average latency
    let avg_latency = results.iter().map(|r| r.latency_ms).sum::<u128>() as f64 / total as f64;

    EvalMetrics {
        precision_at_k: precision,
        recall_at_k: recall,
        mrr,
        avg_latency_ms: avg_latency,
        queries_with_relevant,
        total_queries: total,
    }
}

#[tokio::main]
async fn main() -> Result<()> {
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

            println!(
                "Chunked '{}' into {} chunks using {} strategy:\n",
                input,
                chunks.len(),
                strategy
            );
            for (i, chunk) in chunks.iter().enumerate() {
                println!(
                    "--- Chunk {} ({}B, id: {}) ---",
                    i + 1,
                    chunk.content.len(),
                    &chunk.id[..8]
                );
                let preview: String = chunk.content.chars().take(200).collect();
                let ellipsis = if chunk.content.len() > 200 { "..." } else { "" };
                println!("{preview}{ellipsis}\n");
            }
        }

        Commands::Embed { text, query } => {
            println!("Loading BGE model (first run downloads ~1.2GB)...");
            let mut embedder = BgeEmbedder::new()?;

            let embedding = if query {
                println!("Embedding as query: {text}");
                embedder.embed_query(&text)?
            } else {
                println!("Embedding as document: {text}");
                embedder
                    .embed_documents(&[text.as_str()])?
                    .into_iter()
                    .next()
                    .expect("should have one embedding")
            };

            println!("\nEmbedding stats:");
            println!("  Dimensions: {}", embedding.len());
            println!("  First 5 values: {:?}", &embedding[..5]);
            println!(
                "  Min: {:.4}",
                embedding.iter().cloned().fold(f32::INFINITY, f32::min)
            );
            println!(
                "  Max: {:.4}",
                embedding.iter().cloned().fold(f32::NEG_INFINITY, f32::max)
            );
        }

        Commands::Demo {
            input,
            query,
            k,
            n,
            strategy,
            rerank,
        } => {
            println!("Loading '{input}'...");
            let text = fs::read_to_string(&input)?;
            let chunks = chunk_text(&text, &strategy, 512, 100);
            println!("Created {} chunks using {strategy} strategy", chunks.len());

            println!("\nLoading BGE model (first run downloads ~1.2GB)...");
            let embedder = BgeEmbedder::new()?;
            let store = MemoryStore::new();

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
                let ellipsis = if result.chunk.content.len() > 300 {
                    "..."
                } else {
                    ""
                };
                println!("{preview}{ellipsis}\n");
            }
        }

        Commands::Eval {
            inputs,
            queries,
            k,
            n,
            strategy,
            size,
            overlap,
            rerank,
            verbose,
            json,
        } => {
            // Load test queries
            let query_json_str = fs::read_to_string(&queries)?;
            let query_file: QueryFile = serde_json::from_str(&query_json_str)?;

            if !json {
                println!("Loaded {} test queries from {queries}", query_file.queries.len());
            }

            // Load and chunk all input documents
            let mut all_chunks = Vec::new();
            let mut source_map: HashMap<String, String> = HashMap::new();

            for input in &inputs {
                let path = Path::new(input);
                let filename = path.file_name().unwrap().to_string_lossy().to_string();
                let text = fs::read_to_string(input)?;
                let chunks = chunk_with_source(&text, &filename, &strategy, size, overlap);
                if !json {
                    println!("  {filename}: {} chunks", chunks.len());
                }
                source_map.insert(filename, input.clone());
                all_chunks.extend(chunks);
            }
            if !json {
                println!("Total: {} chunks indexed", all_chunks.len());
                println!("\nLoading models...");
            }

            // Initialize engine
            let embedder = BgeEmbedder::new()?;
            let store = MemoryStore::new();

            if json {
                // JSON output mode for LLM-as-judge
                let json_results = if rerank {
                    let reranker = BgeReranker::new()?;
                    let mut engine = SearchEngine::with_rerank(embedder, store, reranker);
                    engine.index(&all_chunks)?;
                    run_eval_json(&mut engine, &query_file.queries, k, Some(n))?
                } else {
                    let mut engine = SearchEngine::new(embedder, store);
                    engine.index(&all_chunks)?;
                    run_eval_json(&mut engine, &query_file.queries, k, None)?
                };

                let output = JsonOutput {
                    config: JsonConfig {
                        strategy: strategy.clone(),
                        chunk_size: size,
                        reranking: rerank,
                        k,
                    },
                    results: json_results,
                };
                println!("{}", serde_json::to_string_pretty(&output)?);
            } else {
                // Standard output mode
                println!("\n=== Evaluation Config ===");
                println!("Strategy: {strategy} (size={size}, overlap={overlap})");
                println!("Reranking: {}", if rerank { format!("yes (n={n})") } else { "no".into() });
                println!("k: {k}");
                println!();

                let query_results = if rerank {
                    let reranker = BgeReranker::new()?;
                    let mut engine = SearchEngine::with_rerank(embedder, store, reranker);
                    engine.index(&all_chunks)?;
                    run_eval(&mut engine, &query_file.queries, k, Some(n), verbose)?
                } else {
                    let mut engine = SearchEngine::new(embedder, store);
                    engine.index(&all_chunks)?;
                    run_eval(&mut engine, &query_file.queries, k, None, verbose)?
                };

                let metrics = calculate_metrics(&query_results, k);

                println!("\n=== Results ===");
                println!("Precision@{k}: {:.3}", metrics.precision_at_k);
                println!("Recall@{k}:    {:.3}", metrics.recall_at_k);
                println!("MRR:           {:.3}", metrics.mrr);
                println!("Avg latency:   {:.1}ms", metrics.avg_latency_ms);
                println!(
                    "Queries with relevant results: {}/{}",
                    metrics.queries_with_relevant, metrics.total_queries
                );
            }
        }
    }

    Ok(())
}

/// Run evaluation with either reranking or basic engine
fn run_eval<E, S, R>(
    engine: &mut SearchEngine<E, S, R>,
    queries: &[TestQuery],
    k: usize,
    n: Option<usize>,
    verbose: bool,
) -> Result<Vec<QueryResult>>
where
    E: bluffsport_lib::embed::Embedder,
    S: bluffsport_lib::store::VectorStore,
    R: bluffsport_lib::rerank::Reranker,
{
    let mut results = Vec::with_capacity(queries.len());

    for test_query in queries {
        let start = Instant::now();

        let search_results = match n {
            Some(n_val) => engine.search_reranked(&test_query.query, k, n_val)?,
            None => engine.search(&test_query.query, k)?,
        };

        let latency = start.elapsed().as_millis();

        // Evaluate results
        let mut relevant_count = 0;
        let mut first_relevant = None;

        for (rank, result) in search_results.iter().enumerate() {
            // Require at least 2 keyword matches for relevance
            if is_relevant(&result.chunk.content, &test_query.relevant_keywords, 2) {
                relevant_count += 1;
                if first_relevant.is_none() {
                    first_relevant = Some(rank + 1); // 1-indexed
                }
            }
        }

        if verbose {
            let status = if relevant_count > 0 { "✓" } else { "✗" };
            println!(
                "{status} {} | relevant: {}/{} | first@{} | {}ms",
                test_query.id,
                relevant_count,
                k,
                first_relevant.map_or("-".into(), |r| r.to_string()),
                latency
            );

            if relevant_count == 0 {
                println!("  Query: {}", test_query.query);
                println!("  Keywords: {:?}", test_query.relevant_keywords);
                if let Some(top) = search_results.first() {
                    let preview: String = top.chunk.content.chars().take(100).collect();
                    println!("  Top result: {preview}...");
                }
            }
        }

        results.push(QueryResult {
            query_id: test_query.id.clone(),
            relevant_in_top_k: relevant_count,
            first_relevant_rank: first_relevant,
            latency_ms: latency,
        });
    }

    Ok(results)
}

/// Run evaluation and return JSON-formatted results for LLM-as-judge
fn run_eval_json<E, S, R>(
    engine: &mut SearchEngine<E, S, R>,
    queries: &[TestQuery],
    k: usize,
    n: Option<usize>,
) -> Result<Vec<JsonQueryResult>>
where
    E: bluffsport_lib::embed::Embedder,
    S: bluffsport_lib::store::VectorStore,
    R: bluffsport_lib::rerank::Reranker,
{
    let mut results = Vec::with_capacity(queries.len());

    for test_query in queries {
        let start = Instant::now();

        let search_results = match n {
            Some(n_val) => engine.search_reranked(&test_query.query, k, n_val)?,
            None => engine.search(&test_query.query, k)?,
        };

        let latency = start.elapsed().as_millis();

        let retrieved_chunks: Vec<JsonChunk> = search_results
            .iter()
            .enumerate()
            .map(|(i, r)| JsonChunk {
                rank: i + 1,
                score: r.score,
                content: r.chunk.content.clone(),
            })
            .collect();

        results.push(JsonQueryResult {
            query_id: test_query.id.clone(),
            query: test_query.query.clone(),
            expected_answer: test_query.expected_answer.clone(),
            retrieved_chunks,
            latency_ms: latency,
        });
    }

    Ok(results)
}
