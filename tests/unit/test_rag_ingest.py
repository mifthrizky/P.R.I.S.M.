import json

import pymupdf
import pytest

from prism.ml.artifacts import sha256_file
from prism.rag.ingest import chunk_page, ingest_documents


def test_chunk_page_preserves_overlap():
    text = " ".join(f"word-{index}" for index in range(12))

    chunks = chunk_page(text, chunk_words=5, overlap_words=2)

    assert chunks[0].split()[-2:] == chunks[1].split()[:2]


def test_ingest_documents_preserves_page_and_provenance(tmp_path):
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    pdf_path = source_dir / "safety.pdf"
    pdf = pymupdf.open()
    page = pdf.new_page()
    page.insert_text((72, 72), "Lock out electrical energy before maintenance.")
    pdf.save(pdf_path)
    pdf.close()
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "documents": [
                    {
                        "document_id": "TEST-1",
                        "title": "Safety Test",
                        "filename": pdf_path.name,
                        "revision": "1",
                        "source_url": "https://example.invalid/safety.pdf",
                        "sha256": sha256_file(pdf_path),
                        "categories": ["test"],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    chunks = ingest_documents(manifest_path, source_dir)

    assert len(chunks) == 1
    assert chunks[0].document_id == "TEST-1"
    assert chunks[0].page == 1
    assert "electrical energy" in chunks[0].text


def test_ingest_rejects_checksum_mismatch(tmp_path):
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    (source_dir / "bad.pdf").write_bytes(b"not-a-pdf")
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "documents": [
                    {
                        "document_id": "BAD",
                        "title": "Bad",
                        "filename": "bad.pdf",
                        "revision": "1",
                        "source_url": "https://example.invalid/bad.pdf",
                        "sha256": "0" * 64,
                        "categories": [],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Checksum mismatch"):
        ingest_documents(manifest_path, source_dir)
