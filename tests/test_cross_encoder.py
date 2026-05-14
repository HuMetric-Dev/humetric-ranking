from __future__ import annotations

from humetric_ranking import CrossEncoderReranker


def test_no_op_rerank_preserves_order_when_unloaded() -> None:
    rer = CrossEncoderReranker()
    assert not rer.is_loaded
    out = rer.rerank("rust dev", [("a", "rust engineer"), ("b", "react dev")]).unwrap()
    # Same order, zero scores.
    assert [pid for pid, _ in out] == ["a", "b"]
    assert all(score == 0.0 for _, score in out)


def test_empty_input_when_loaded_with_fake() -> None:
    class _Fake:
        def predict(self, pairs):  # type: ignore[no-untyped-def]
            import numpy as np

            return np.array([0.0] * len(pairs), dtype="float32")

    rer = CrossEncoderReranker(_model=_Fake())
    assert rer.is_loaded
    assert rer.rerank("q", []).unwrap() == []


def test_fake_loaded_sorts_by_score() -> None:
    class _Fake:
        def predict(self, pairs):  # type: ignore[no-untyped-def]
            # Score = length of candidate text (the second element).
            import numpy as np

            return np.array([len(p[1]) for p in pairs], dtype="float32")

    rer = CrossEncoderReranker(_model=_Fake())
    out = rer.rerank("q", [("a", "short"), ("b", "much longer text"), ("c", "tiny")]).unwrap()
    assert [pid for pid, _ in out] == ["b", "a", "c"]
