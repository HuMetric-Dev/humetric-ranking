from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import lightgbm as lgb
import numpy as np
from humetric_core import Err, Ok, Result

from humetric_ranking.errors import (
    ModelLoadFailed,
    ModelMissing,
    PredictFailed,
    RankingError,
    TrainFailed,
)


@dataclass(slots=True)
class TrainData:
    """LambdaRank input: a flat feature matrix plus per-group sizes.

    `groups[i]` = number of rows belonging to query group `i`. Labels are
    relevance scores per row (we use 0/1 for v1: positive pairs = 1, all
    others = 0).
    """

    features: np.ndarray  # (n_rows, n_features)
    labels: np.ndarray  # (n_rows,)
    groups: np.ndarray  # (n_groups,)


@dataclass(slots=True)
class Reranker:
    model: Any  # lgb.Booster

    def predict(self, features: np.ndarray) -> Result[np.ndarray, RankingError]:
        if features.size == 0:
            return Ok(np.zeros((0,), dtype=np.float32))
        try:
            scores = self.model.predict(features)
        except (RuntimeError, ValueError) as e:
            return Err(PredictFailed(reason=str(e)))
        return Ok(np.asarray(scores, dtype=np.float32))

    def save(self, path: str | Path) -> Result[None, RankingError]:
        try:
            joblib.dump(self.model, str(path))
        except OSError as e:
            return Err(ModelLoadFailed(path=str(path), reason=str(e)))
        return Ok(None)


def train_reranker(
    data: TrainData, *, num_leaves: int = 31, num_iters: int = 100
) -> Result[Reranker, RankingError]:
    if data.features.size == 0 or data.groups.size == 0:
        return Err(TrainFailed(reason="empty training data"))
    if int(data.groups.sum()) != data.features.shape[0]:
        return Err(
            TrainFailed(
                reason=f"group sum {int(data.groups.sum())} != n_rows {data.features.shape[0]}"
            )
        )

    try:
        train_set = lgb.Dataset(data.features, label=data.labels, group=data.groups)
        booster = lgb.train(
            params={
                "objective": "lambdarank",
                "metric": "ndcg",
                "ndcg_at": [5, 10],
                "num_leaves": num_leaves,
                "learning_rate": 0.05,
                "min_data_in_leaf": 1,
                "verbosity": -1,
            },
            train_set=train_set,
            num_boost_round=num_iters,
        )
    except (RuntimeError, ValueError, lgb.basic.LightGBMError) as e:
        return Err(TrainFailed(reason=str(e)))
    return Ok(Reranker(model=booster))


def load_reranker(path: str | Path) -> Result[Reranker, RankingError]:
    p = Path(path)
    if not p.exists():
        return Err(ModelMissing(path=str(p)))
    try:
        booster = joblib.load(str(p))
    except (OSError, ValueError, RuntimeError) as e:
        return Err(ModelLoadFailed(path=str(p), reason=str(e)))
    return Ok(Reranker(model=booster))
