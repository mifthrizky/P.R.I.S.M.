"""Build the offline FAISS index and chunk store."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from prism.ml.artifacts import sha256_file

if TYPE_CHECKING:
    from prism.rag.ingest import DocumentChunk

DEFAULT_EMBEDDING_MODEL = "intfloat/multilingual-e5-small"


class Embedder(Protocol):
    dimension: int

    def encode_passages(self, texts: list[str]) -> np.ndarray: ...

    def encode_query(self, text: str) -> np.ndarray: ...


class E5Embedder:
    def __init__(self, model_name: str = DEFAULT_EMBEDDING_MODEL) -> None:
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.dimension = int(self.model.get_embedding_dimension())

    def _encode(self, texts: list[str]) -> np.ndarray:
        embeddings = self.model.encode(
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return np.asarray(embeddings, dtype=np.float32)

    def encode_passages(self, texts: list[str]) -> np.ndarray:
        return self._encode([f"passage: {text}" for text in texts])

    def encode_query(self, text: str) -> np.ndarray:
        return self._encode([f"query: {text}"])


def write_chunks(chunks: list[DocumentChunk], path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as chunk_file:
        for chunk in chunks:
            chunk_file.write(json.dumps(chunk.to_dict(), ensure_ascii=False) + "\n")


def build_index(
    manifest_path: str | Path,
    source_dir: str | Path,
    output_dir: str | Path,
    embedder: Embedder,
) -> dict[str, object]:
    from prism.rag.ingest import ingest_documents

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    chunks = ingest_documents(manifest_path, source_dir)
    embeddings = embedder.encode_passages([chunk.text for chunk in chunks])
    if embeddings.shape != (len(chunks), embedder.dimension):
        raise ValueError("Embedding output shape does not match the chunk collection")

    index = faiss.IndexFlatIP(embedder.dimension)
    index.add(embeddings)
    index_path = output_path / "knowledge.faiss"
    chunks_path = output_path / "chunks.jsonl"
    metadata_path = output_path / "index_metadata.json"
    faiss.write_index(index, str(index_path))
    write_chunks(chunks, chunks_path)
    metadata = {
        "built_at_utc": datetime.now(UTC).isoformat(),
        "backend": "IndexFlatIP",
        "embedding_model": getattr(embedder, "model_name", "test-embedder"),
        "embedding_dimension": embedder.dimension,
        "chunk_count": len(chunks),
        "document_count": len({chunk.document_id for chunk in chunks}),
        "manifest_sha256": sha256_file(manifest_path),
        "chunks_sha256": sha256_file(chunks_path),
    }
    with metadata_path.open("w", encoding="utf-8", newline="\n") as metadata_file:
        json.dump(metadata, metadata_file, indent=2, sort_keys=True)
        metadata_file.write("\n")
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="documents/manifest.json")
    parser.add_argument("--source-dir", default="documents/source")
    parser.add_argument("--output-dir", default="artifacts/knowledge")
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    args = parser.parse_args()
    metadata = build_index(
        args.manifest,
        args.source_dir,
        args.output_dir,
        E5Embedder(args.embedding_model),
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
