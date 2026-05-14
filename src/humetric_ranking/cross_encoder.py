from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from humetric_core import Err, Ok, Result

from humetric_ranking.errors import CrossEncoderUnavailable, ModelLoadFailed, RankingError


@dataclass(slots=True)
class CrossEncoderReranker:
    """Optional second-stage rerank via a cross-encoder.

    On the Mac (no CUDA, no checkpoint), `rerank()` is a no-op — it returns
    the candidates in the order it received them. On the GPU box with a
    checkpoint loaded, it scores each (query, candidate_text) pair and
    sorts.
    """

    _model: Any | None = None

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def rerank(
        self, query: str, candidates: list[tuple[str, str]]
    ) -> Result[list[tuple[str, float]], RankingError]:
        """`candidates` is a list of (person_id, candidate_text).

        Returns a list of (person_id, score) sorted desc by score. If the
        cross-encoder isn't loaded, returns Ok with a zero score for each
        item in input order — callers can treat this as a no-op rerank.
        """
        if self._model is None:
            return Ok([(pid, 0.0) for pid, _ in candidates])
        if not candidates:
            return Ok([])

        pairs = [[query, text] for _, text in candidates]
        try:
            scores = self._model.predict(pairs)
        except (RuntimeError, ValueError) as e:
            return Err(CrossEncoderUnavailable(reason=str(e)))

        arr = np.asarray(scores, dtype=np.float32)
        order = np.argsort(-arr)
        return Ok([(candidates[int(i)][0], float(arr[int(i)])) for i in order])


def load_cross_encoder(
    path: str | Path = "BAAI/bge-reranker-base",
) -> Result[CrossEncoderReranker, RankingError]:
    """Try to load `bge-reranker-base`. Failure is *not* fatal at the caller
    level — most callers should match `Err(CrossEncoderUnavailable)` and
    continue with the no-op CrossEncoderReranker()."""
    try:
        from sentence_transformers import CrossEncoder
    except ImportError as e:
        return Err(CrossEncoderUnavailable(reason=f"sentence-transformers not installed: {e}"))

    try:
        model = CrossEncoder(str(path))
    except (OSError, ValueError, RuntimeError) as e:
        return Err(ModelLoadFailed(path=str(path), reason=str(e)))

    return Ok(CrossEncoderReranker(_model=model))
