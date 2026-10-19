"""Evaluate document-level Recall@K for the PRISM knowledge index."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from prism.rag.index import DEFAULT_EMBEDDING_MODEL, E5Embedder
from prism.rag.retriever import HybridRetriever


def evaluate_retriever(
    retriever: HybridRetriever, evaluation_path: str | Path, top_k: int = 5
) -> dict[str, object]:
    evaluation = json.loads(Path(evaluation_path).read_text(encoding="utf-8"))
    results = []
    for item in evaluation["queries"]:
        retrieved = retriever.retrieve(item["query"], top_k=top_k)
        retrieved_ids = {result["document_id"] for result in retrieved}
        expected_ids = set(item["expected_document_ids"])
        hit = bool(retrieved_ids.intersection(expected_ids))
        results.append(
            {
                "query": item["query"],
                "hit": hit,
                "expected_document_ids": sorted(expected_ids),
                "retrieved_document_ids": sorted(retrieved_ids),
            }
        )
    recall = sum(result["hit"] for result in results) / len(results)
    return {"top_k": top_k, "recall_at_k": recall, "queries": results}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", default="artifacts/knowledge")
    parser.add_argument("--evaluation", default="documents/evaluation_queries.json")
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--output", default="artifacts/knowledge/evaluation_report.json")
    args = parser.parse_args()
    retriever = HybridRetriever(args.artifact_dir, E5Embedder(args.embedding_model))
    report = evaluate_retriever(retriever, args.evaluation, args.top_k)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if report["recall_at_k"] < 0.85:
        raise SystemExit("Retrieval gate failed: Recall@K is below 0.85")


if __name__ == "__main__":
    main()
