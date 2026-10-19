"""Hybrid dense and BM25 retrieval with reciprocal-rank fusion."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import faiss
import numpy as np
from rank_bm25 import BM25Okapi

from prism.rag.index import DEFAULT_EMBEDDING_MODEL, E5Embedder, Embedder


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def reciprocal_rank_fusion(
    ranked_lists: list[list[int]], rank_constant: int = 60
) -> dict[int, float]:
    scores: dict[int, float] = {}
    for ranked in ranked_lists:
        for rank, item_index in enumerate(ranked, start=1):
            scores[item_index] = scores.get(item_index, 0.0) + 1.0 / (rank_constant + rank)
    return scores


def has_sufficient_evidence(
    results: list[dict[str, Any]],
    minimum_dense_score: float = 0.78,
    minimum_bm25_score: float = 1.0,
) -> bool:
    if not results:
        return False
    best_dense = max((result.get("dense_score") or 0.0) for result in results)
    best_sparse = max(result.get("bm25_score", 0.0) for result in results)
    return best_dense >= minimum_dense_score or best_sparse >= minimum_bm25_score


class HybridRetriever:
    def __init__(self, artifact_dir: str | Path, embedder: Embedder) -> None:
        artifact_path = Path(artifact_dir)
        self.embedder = embedder
        self.index = faiss.read_index(str(artifact_path / "knowledge.faiss"))
        self.chunks = [
            json.loads(line)
            for line in (artifact_path / "chunks.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if self.index.ntotal != len(self.chunks):
            raise ValueError("FAISS index and chunk store contain different item counts")
        self.bm25 = BM25Okapi([tokenize(chunk["text"]) for chunk in self.chunks])

    def retrieve(
        self, query: str, top_k: int = 5, candidate_k: int = 10
    ) -> list[dict[str, Any]]:
        if not query.strip():
            raise ValueError("query must not be empty")
        candidate_k = min(candidate_k, len(self.chunks))
        query_embedding = np.asarray(self.embedder.encode_query(query), dtype=np.float32)
        dense_scores, dense_indices = self.index.search(query_embedding, candidate_k)
        dense_ranked = [int(index) for index in dense_indices[0] if index >= 0]

        sparse_scores = self.bm25.get_scores(tokenize(query))
        sparse_ranked = np.argsort(sparse_scores)[::-1][:candidate_k].astype(int).tolist()
        fused_scores = reciprocal_rank_fusion([dense_ranked, sparse_ranked])
        final_indices = sorted(fused_scores, key=fused_scores.get, reverse=True)[:top_k]
        dense_score_by_index = {
            int(index): float(score)
            for index, score in zip(dense_indices[0], dense_scores[0], strict=True)
            if index >= 0
        }
        return [
            {
                **self.chunks[index],
                "retrieval_score": fused_scores[index],
                "dense_score": dense_score_by_index.get(index),
                "bm25_score": float(sparse_scores[index]),
            }
            for index in final_indices
        ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query")
    parser.add_argument("--artifact-dir", default="artifacts/knowledge")
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()
    retriever = HybridRetriever(args.artifact_dir, E5Embedder(args.embedding_model))
    results = retriever.retrieve(args.query, args.top_k)
    response = {
        "status": "EVIDENCE_FOUND" if has_sufficient_evidence(results) else "INSUFFICIENT_EVIDENCE",
        "results": results if has_sufficient_evidence(results) else [],
    }
    print(json.dumps(response, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
