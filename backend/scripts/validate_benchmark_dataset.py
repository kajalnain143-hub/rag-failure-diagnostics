"""Validate a labeled RAG benchmark against a fresh /api/benchmark/corpus export.

Usage:
  python scripts/validate_benchmark_dataset.py --corpus corpus.json --dataset dataset.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_json(path: str):
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def validate_dataset(dataset, corpus):
    errors = []
    cases = dataset.get("cases") if isinstance(dataset, dict) else dataset
    chunks = corpus.get("chunks") if isinstance(corpus, dict) else None
    if not isinstance(cases, list) or not cases:
        return ["Dataset must be a non-empty JSON array or an object containing a non-empty 'cases' array."]
    if not isinstance(chunks, list):
        return ["Corpus export must be the JSON response from GET /api/benchmark/corpus."]
    available_ids = {
        item.get("chunk_id") for item in chunks
        if isinstance(item, dict) and isinstance(item.get("chunk_id"), int)
        and not isinstance(item.get("chunk_id"), bool)
    }
    seen_query_ids = set()
    for index, case in enumerate(cases, start=1):
        prefix = f"Case {index}"
        if not isinstance(case, dict):
            errors.append(f"{prefix}: must be a JSON object.")
            continue
        query_id = case.get("query_id")
        if not isinstance(query_id, str) or not query_id.strip():
            errors.append(f"{prefix}: query_id must be a non-empty string.")
        elif query_id in seen_query_ids:
            errors.append(f"{prefix}: duplicate query_id '{query_id}'.")
        else:
            seen_query_ids.add(query_id)
        for field in ("question", "reference_answer"):
            value = case.get(field)
            if not isinstance(value, str) or not value.strip():
                errors.append(f"{prefix}: {field} must be a non-empty string.")
        relevant = case.get("relevant_chunk_ids")
        if not isinstance(relevant, list) or not relevant:
            errors.append(f"{prefix}: relevant_chunk_ids must be a non-empty array.")
        else:
            if any(isinstance(value, bool) or not isinstance(value, int) for value in relevant):
                errors.append(f"{prefix}: relevant_chunk_ids must contain integer chunk IDs only.")
            elif len(set(relevant)) != len(relevant):
                errors.append(f"{prefix}: relevant_chunk_ids contains duplicates.")
            else:
                missing = sorted(set(relevant) - available_ids)
                if missing:
                    errors.append(f"{prefix}: chunk IDs not present in current corpus export: {missing}.")
        cutoff = case.get("k", 5)
        if isinstance(cutoff, bool) or not isinstance(cutoff, int) or not 1 <= cutoff <= 50:
            errors.append(f"{prefix}: k must be an integer between 1 and 50.")
    return errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", required=True, help="JSON saved from GET /api/benchmark/corpus")
    parser.add_argument("--dataset", required=True, help="Labeled benchmark JSON file")
    args = parser.parse_args()
    try:
        errors = validate_dataset(load_json(args.dataset), load_json(args.corpus))
    except (OSError, json.JSONDecodeError) as exc:
        parser.error(f"Could not read valid JSON input: {exc}")
    if errors:
        print(f"Benchmark validation failed ({len(errors)} issue(s)):")
        for error in errors:
            print(f"- {error}")
        raise SystemExit(1)
    print("Benchmark validation passed.")
    print("All query IDs are unique and every gold chunk ID exists in the supplied corpus export.")
    print("Reminder: relevance labels and reference answers still require human review.")


if __name__ == "__main__":
    main()
