"""Unit tests for Reciprocal Rank Fusion math."""
from src.retrieval.hybrid_retriever import reciprocal_rank_fusion


def test_rrf_single_list_preserves_order() -> None:
    list1 = [{"chunk_id": "a"}, {"chunk_id": "b"}, {"chunk_id": "c"}]
    result = reciprocal_rank_fusion([list1], top_k=3)
    assert [r["chunk_id"] for r in result] == ["a", "b", "c"]


def test_rrf_overlap_boosts_score() -> None:
    list1 = [{"chunk_id": "a"}, {"chunk_id": "b"}]
    list2 = [{"chunk_id": "b"}, {"chunk_id": "c"}]
    result = reciprocal_rank_fusion([list1, list2], top_k=3)
    assert result[0]["chunk_id"] == "b"


def test_rrf_respects_top_k() -> None:
    list1 = [{"chunk_id": f"a{i}"} for i in range(10)]
    result = reciprocal_rank_fusion([list1], top_k=3)
    assert len(result) == 3


def test_rrf_attaches_score_field() -> None:
    list1 = [{"chunk_id": "a"}]
    result = reciprocal_rank_fusion([list1], top_k=1)
    assert "rrf_score" in result[0]
    assert result[0]["rrf_score"] > 0


def test_rrf_empty_lists() -> None:
    assert reciprocal_rank_fusion([], top_k=5) == []
    assert reciprocal_rank_fusion([[]], top_k=5) == []
