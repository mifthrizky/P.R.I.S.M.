"""Verified PDF ingestion with page-preserving chunks."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pymupdf

from prism.ml.artifacts import sha256_file


@dataclass(frozen=True)
class DocumentChunk:
    chunk_id: str
    document_id: str
    title: str
    revision: str
    page: int
    categories: list[str]
    source_url: str
    text: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def chunk_page(
    text: str, chunk_words: int = 350, overlap_words: int = 50
) -> list[str]:
    if chunk_words <= overlap_words:
        raise ValueError("chunk_words must be greater than overlap_words")
    words = normalize_text(text).split()
    if not words:
        return []
    step = chunk_words - overlap_words
    return [" ".join(words[start : start + chunk_words]) for start in range(0, len(words), step)]


def load_manifest(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as manifest_file:
        manifest = json.load(manifest_file)
    documents = manifest.get("documents", [])
    if not documents:
        raise ValueError("Document manifest is empty")
    return documents


def ingest_documents(
    manifest_path: str | Path,
    source_dir: str | Path,
    chunk_words: int = 350,
    overlap_words: int = 50,
) -> list[DocumentChunk]:
    chunks: list[DocumentChunk] = []
    for document in load_manifest(manifest_path):
        pdf_path = Path(source_dir) / document["filename"]
        if not pdf_path.exists():
            raise FileNotFoundError(f"Missing source document: {pdf_path}")
        actual_hash = sha256_file(pdf_path)
        if actual_hash != document["sha256"]:
            raise ValueError(f"Checksum mismatch for {pdf_path.name}")

        with pymupdf.open(pdf_path) as pdf:
            for page_index, page in enumerate(pdf):
                for chunk_index, text in enumerate(
                    chunk_page(page.get_text(), chunk_words, overlap_words)
                ):
                    chunks.append(
                        DocumentChunk(
                            chunk_id=(
                                f"{document['document_id']}-p{page_index + 1}-c{chunk_index + 1}"
                            ),
                            document_id=document["document_id"],
                            title=document["title"],
                            revision=document["revision"],
                            page=page_index + 1,
                            categories=list(document["categories"]),
                            source_url=document["source_url"],
                            text=text,
                        )
                    )
    if not chunks:
        raise ValueError("No text chunks were extracted from the source documents")
    return chunks
