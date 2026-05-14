from __future__ import annotations

import numpy as np

from humetric_ranking import (
    ModelMissing,
    TrainData,
    TrainFailed,
    load_reranker,
    train_reranker,
)


def _make_training_data() -> TrainData:
    """3 query groups of 4 candidates each. Each group has 1 positive
    correlated with feature[0] being high; the rest negatives."""
    rng = np.random.default_rng(seed=0)
    rows = []
    labels = []
    groups = []
    for _ in range(8):
        for is_pos in (1, 0, 0, 0):
            features = rng.uniform(0.0, 0.5, size=8).astype(np.float32)
            if is_pos:
                features[0] = 1.0
            rows.append(features)
            labels.append(is_pos)
        groups.append(4)
    return TrainData(
        features=np.stack(rows),
        labels=np.array(labels, dtype=np.float32),
        groups=np.array(groups, dtype=np.int32),
    )


def test_train_and_predict_ranks_positives_higher() -> None:
    data = _make_training_data()
    reranker = train_reranker(data, num_iters=30).unwrap()

    # Inference: a clear positive vs a clear negative.
    feats = np.array(
        [
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    scores = reranker.predict(feats).unwrap()
    assert scores[0] > scores[1]


def test_train_rejects_empty_data() -> None:
    empty = TrainData(
        features=np.zeros((0, 8), dtype=np.float32),
        labels=np.zeros((0,), dtype=np.float32),
        groups=np.zeros((0,), dtype=np.int32),
    )
    r = train_reranker(empty)
    assert r.is_err()
    assert isinstance(r.err(), TrainFailed)


def test_train_rejects_mismatched_groups() -> None:
    data = TrainData(
        features=np.zeros((4, 8), dtype=np.float32),
        labels=np.zeros((4,), dtype=np.float32),
        groups=np.array([3], dtype=np.int32),  # 3 != 4
    )
    r = train_reranker(data)
    assert r.is_err()
    assert isinstance(r.err(), TrainFailed)


def test_save_and_load_roundtrip(tmp_path) -> None:  # type: ignore[no-untyped-def]
    data = _make_training_data()
    reranker = train_reranker(data, num_iters=10).unwrap()
    out = tmp_path / "reranker.joblib"
    reranker.save(out).unwrap()
    loaded = load_reranker(out).unwrap()
    feats = np.array([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    assert loaded.predict(feats).unwrap().shape == (1,)


def test_load_missing_returns_err(tmp_path) -> None:  # type: ignore[no-untyped-def]
    r = load_reranker(tmp_path / "nope.joblib")
    assert r.is_err()
    assert isinstance(r.err(), ModelMissing)
