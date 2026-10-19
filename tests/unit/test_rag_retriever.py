from prism.rag.retriever import has_sufficient_evidence, reciprocal_rank_fusion, tokenize


def test_tokenize_is_case_insensitive_and_removes_punctuation():
    assert tokenize("Lockout/Tagout: MACHINE #1") == ["lockout", "tagout", "machine", "1"]


def test_reciprocal_rank_fusion_rewards_items_in_both_lists():
    scores = reciprocal_rank_fusion([[1, 2, 3], [3, 1, 4]])

    assert scores[1] > scores[2]
    assert scores[3] > scores[4]


def test_evidence_gate_rejects_weak_retrieval_results():
    weak = [{"dense_score": 0.5, "bm25_score": 0.0}]
    strong = [{"dense_score": 0.5, "bm25_score": 3.0}]

    assert has_sufficient_evidence(weak) is False
    assert has_sufficient_evidence(strong) is True
