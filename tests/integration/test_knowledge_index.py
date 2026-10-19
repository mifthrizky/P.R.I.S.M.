import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
KNOWLEDGE_DIR = PROJECT_ROOT / "artifacts" / "knowledge"


def test_knowledge_index_has_page_level_provenance():
    metadata = json.loads((KNOWLEDGE_DIR / "index_metadata.json").read_text(encoding="utf-8"))
    chunks = [
        json.loads(line)
        for line in (KNOWLEDGE_DIR / "chunks.jsonl").read_text(encoding="utf-8").splitlines()
    ]

    assert metadata["document_count"] == 4
    assert metadata["chunk_count"] == len(chunks)
    assert all(chunk["document_id"] and chunk["page"] >= 1 for chunk in chunks)
    assert all(chunk["source_url"].startswith("https://") for chunk in chunks)


def test_retrieval_evaluation_meets_recall_gate():
    report = json.loads(
        (KNOWLEDGE_DIR / "evaluation_report.json").read_text(encoding="utf-8")
    )

    assert report["recall_at_k"] >= 0.85
    assert all(item["hit"] for item in report["queries"])
