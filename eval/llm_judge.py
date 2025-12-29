#!/usr/bin/env python3
"""
LLM-as-Judge evaluation for BluffSport RAG pipeline using Claude CLI.

Usage:
    ./target/release/bluffsport eval --queries eval/queries-llm-judge.json \
        eval/article1.txt eval/article2.txt eval/article3.txt eval/sports_sample.txt \
        --json | python eval/llm_judge.py

Requires: claude CLI installed and authenticated
"""

import json
import sys
import subprocess
from dataclasses import dataclass


@dataclass
class JudgmentResult:
    query_id: str
    query: str
    scores: list[int]  # 0-2 for each chunk
    first_relevant_rank: int | None
    latency_ms: int


def judge_relevance(query: str, expected: str, chunk: str) -> int:
    """Use Claude CLI to judge relevance of a chunk to a query. Returns 0, 1, or 2."""

    prompt = f"""You are evaluating search results for a RAG (Retrieval Augmented Generation) system.

Query: {query}
Expected answer: {expected}

Retrieved chunk:
---
{chunk[:1500]}
---

Rate the relevance of this chunk to answering the query:
- 0 = Irrelevant: The chunk does not contain information useful for answering the query
- 1 = Partially relevant: The chunk contains some related information but not a direct answer
- 2 = Highly relevant: The chunk contains information that directly answers or strongly supports answering the query

Respond with ONLY a single digit: 0, 1, or 2. Nothing else."""

    try:
        result = subprocess.run(
            ["claude", "-p", prompt, "--tools", "", "--no-session-persistence"],
            capture_output=True,
            text=True,
            timeout=60
        )
        text = result.stdout.strip()
        # Extract first digit from response
        for char in text:
            if char.isdigit():
                score = int(char)
                return min(max(score, 0), 2)  # Clamp to 0-2
        return 0  # Default to irrelevant if no digit found
    except (subprocess.TimeoutExpired, subprocess.SubprocessError) as e:
        print(f"  Warning: Claude CLI error: {e}", file=sys.stderr)
        return 0


def calculate_metrics(results: list[JudgmentResult], k: int) -> dict:
    """Calculate evaluation metrics from judgment results."""

    total_queries = len(results)
    if total_queries == 0:
        return {}

    # Precision@k: average proportion of relevant (score > 0) in top k
    precision_sum = sum(
        sum(1 for s in r.scores if s > 0) / k
        for r in results
    )
    precision = precision_sum / total_queries

    # Recall@k: fraction of queries with at least one relevant result
    queries_with_relevant = sum(1 for r in results if any(s > 0 for s in r.scores))
    recall = queries_with_relevant / total_queries

    # MRR: mean reciprocal rank of first relevant result
    mrr_sum = 0.0
    for r in results:
        if r.first_relevant_rank:
            mrr_sum += 1.0 / r.first_relevant_rank
    mrr = mrr_sum / total_queries

    # Average relevance score (0-2 scale)
    all_scores = [s for r in results for s in r.scores]
    avg_score = sum(all_scores) / len(all_scores) if all_scores else 0

    # Highly relevant rate (score = 2)
    highly_relevant = sum(1 for s in all_scores if s == 2)
    highly_relevant_rate = highly_relevant / len(all_scores) if all_scores else 0

    # Average latency
    avg_latency = sum(r.latency_ms for r in results) / total_queries

    return {
        "precision_at_k": precision,
        "recall_at_k": recall,
        "mrr": mrr,
        "avg_relevance_score": avg_score,
        "highly_relevant_rate": highly_relevant_rate,
        "queries_with_relevant": queries_with_relevant,
        "total_queries": total_queries,
        "avg_latency_ms": avg_latency,
    }


def main():
    # Read JSON from stdin
    try:
        data = json.load(sys.stdin)
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON input: {e}", file=sys.stderr)
        sys.exit(1)

    config = data.get("config", {})
    query_results = data.get("results", [])

    print(f"LLM-as-Judge Evaluation (using Claude CLI)")
    print(f"=" * 50)
    print(f"Config: {config.get('strategy')} chunking, k={config.get('k')}, reranking={config.get('reranking')}")
    print(f"Queries: {len(query_results)}")
    print()

    judgment_results = []

    for i, qr in enumerate(query_results):
        query_id = qr["query_id"]
        query = qr["query"]
        expected = qr["expected_answer"]
        chunks = qr["retrieved_chunks"]
        latency = qr["latency_ms"]

        print(f"[{i+1}/{len(query_results)}] Judging: {query_id}...", end=" ", flush=True)

        scores = []
        first_relevant = None

        for j, chunk in enumerate(chunks):
            score = judge_relevance(query, expected, chunk["content"])
            scores.append(score)
            if score > 0 and first_relevant is None:
                first_relevant = j + 1  # 1-indexed

        status = "✓" if any(s > 0 for s in scores) else "✗"
        print(f"{status} scores={scores}")

        judgment_results.append(JudgmentResult(
            query_id=query_id,
            query=query,
            scores=scores,
            first_relevant_rank=first_relevant,
            latency_ms=latency,
        ))

    # Calculate and print metrics
    k = config.get("k", 5)
    metrics = calculate_metrics(judgment_results, k)

    print()
    print("=" * 50)
    print("Results (LLM-as-Judge)")
    print("=" * 50)
    print(f"Precision@{k}:        {metrics['precision_at_k']:.3f}")
    print(f"Recall@{k}:           {metrics['recall_at_k']:.3f}")
    print(f"MRR:                  {metrics['mrr']:.3f}")
    print(f"Avg relevance (0-2):  {metrics['avg_relevance_score']:.2f}")
    print(f"Highly relevant rate: {metrics['highly_relevant_rate']:.1%}")
    print(f"Queries w/ relevant:  {metrics['queries_with_relevant']}/{metrics['total_queries']}")
    print(f"Avg latency:          {metrics['avg_latency_ms']:.1f}ms")

    # Output JSON summary
    print()
    print("JSON Summary:")
    print(json.dumps({
        "config": config,
        "metrics": metrics,
        "per_query": [
            {"id": r.query_id, "scores": r.scores, "first_relevant": r.first_relevant_rank}
            for r in judgment_results
        ]
    }, indent=2))


if __name__ == "__main__":
    main()
