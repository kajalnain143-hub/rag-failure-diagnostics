"""Batch benchmark execution for labeled RAG datasets."""
from __future__ import annotations

from time import perf_counter
from .evaluation_metrics import aggregate_metrics, evaluate_case

ALLOWED_STRATEGIES = {"tfidf", "hybrid", "hybrid_mmr"}


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return round(ordered[lower] * (1 - fraction) + ordered[upper] * fraction, 2)


def run_dataset(dataset_name: str, cases: list[dict], strategies: list[str],
                chunks: list[dict], embed_query, retrieve, save_experiment,
                mmr_diversity: float = 0.35) -> dict:
    """Run every case against each selected strategy and persist aggregate runs."""
    output = {}
    for strategy in strategies:
        case_rows, latencies, details = [], [], []
        for case in cases:
            started = perf_counter()
            query_vector = embed_query(case["question"]) if strategy != "tfidf" else None
            embedded = perf_counter()
            retrieved = retrieve(
                case["question"], chunks, case["k"], query_embedding=query_vector,
                strategy=strategy, mmr_diversity=mmr_diversity,
            )
            finished = perf_counter()
            metrics = evaluate_case(case, retrieved, case["k"])
            case_rows.append(metrics)
            latency_ms = round((finished - started) * 1000, 2)
            latencies.append(latency_ms)
            details.append({
                "query_id": case["query_id"], "metrics": metrics,
                "retrieved_chunk_ids": [item["chunk_id"] for item in retrieved],
                "latency_ms": latency_ms,
                "embedding_ms": round((embedded - started) * 1000, 2),
                "retrieval_ms": round((finished - embedded) * 1000, 2),
            })
        aggregate = aggregate_metrics(case_rows)
        performance = {
            "latency_avg_ms": round(sum(latencies) / len(latencies), 2),
            "latency_p50_ms": _percentile(latencies, 0.50),
            "latency_p95_ms": _percentile(latencies, 0.95),
            "total_latency_ms": round(sum(latencies), 2),
        }
        metrics = {**aggregate, **performance}
        run_id = save_experiment(
            f"Batch benchmark · {dataset_name} · {strategy}",
            {"dataset_name": dataset_name, "strategy": strategy,
             "case_count": len(cases), "mmr_diversity": mmr_diversity},
            metrics,
            "Automated batch run; latency includes query embedding where used and retrieval.",
        )
        output[strategy] = {
            "metrics": metrics, "cases": details, "experiment_id": run_id,
        }
    return {
        "dataset_name": dataset_name,
        "case_count": len(cases),
        "strategies": output,
        "metric_note": "Ranking scores depend on manual relevant_chunk_ids labels; evidence coverage is lexical overlap. Latency is measured on this API process and corpus.",
    }
