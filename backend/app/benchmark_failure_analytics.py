"""Rule-based failure signals derived from saved benchmark metrics."""
from __future__ import annotations

from .failure_clinic import PATTERNS

_PATTERN_LOOKUP = {item["id"]: item for item in PATTERNS}


def analyze_benchmark_failures(runs: list[dict], dataset_name: str) -> dict:
    selected = [
        run for run in runs
        if run.get("config", {}).get("dataset_name") == dataset_name
        and run.get("config", {}).get("strategy") in {"tfidf", "hybrid", "hybrid_mmr"}
        and all(key in run.get("metrics", {}) for key in (
            "recall_at_k", "mrr", "ndcg_at_k", "evidence_term_coverage",
        ))
    ]
    selected.sort(key=lambda run: (str(run.get("created_at", "")), int(run["id"])))
    latest = {}
    for run in selected:
        latest[run["config"]["strategy"]] = run
    signals = []
    for strategy, run in latest.items():
        metrics = run["metrics"]
        recall, mrr = float(metrics["recall_at_k"]), float(metrics["mrr"])
        ndcg, coverage = float(metrics["ndcg_at_k"]), float(metrics["evidence_term_coverage"])
        common = {"strategy": strategy, "run_id": run["id"], "snapshot_id": run["config"].get("snapshot_id")}
        if recall < 0.5 or mrr < 0.5 or ndcg < 0.5:
            signals.append({**common, "pattern_id": "P03",
                "severity": "high" if recall < 0.25 else "medium",
                "evidence": f"Retrieval ranking is weak: Recall@K={recall:.3f}, MRR={mrr:.3f}, nDCG@K={ndcg:.3f}.",
                "recommendation": "Inspect query embeddings, retrieval scores, top-K ranking, and relevant-chunk labels; compare strategies on the same snapshot."})
        if coverage < 0.4:
            signals.append({**common, "pattern_id": "P02", "severity": "medium",
                "evidence": f"Lexical evidence coverage is low ({coverage:.3f}); relevant facts may be split across chunks or use different terminology.",
                "recommendation": "Inspect source text extraction, chunk size/overlap, and whether reference-answer terms appear in retrieved context."})
        if run["config"].get("case_count", 0) < 10:
            signals.append({**common, "pattern_id": "P09", "severity": "low",
                "evidence": f"Only {run['config'].get('case_count', 0)} labeled cases were used; this is a limited evaluation sample.",
                "recommendation": "Expand the labeled dataset with edge cases, paraphrases, and negative examples before treating the benchmark as representative."})
    counts = {}
    for signal in signals:
        pid = signal["pattern_id"]
        counts[pid] = counts.get(pid, 0) + 1
    distribution = []
    for pid, count in sorted(counts.items(), key=lambda item: (-item[1], item[0])):
        pattern = _PATTERN_LOOKUP[pid]
        distribution.append({"pattern_id": pid, "name": pattern["name"],
                             "symptom": pattern["symptom"], "signal_count": count})
    return {
        "dataset_name": dataset_name, "analyzed_strategy_count": len(latest),
        "signal_count": len(signals), "distribution": distribution, "signals": signals,
        "note": "Counts are rule-triggered benchmark signals across the latest run per strategy, not confirmed incidents or per-query failure rates. P02/P03/P09 are hypotheses. P01 hallucination and generation failures require answer-level or incident evidence not contained in aggregate retrieval metrics.",
    }
