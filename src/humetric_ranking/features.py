from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final

import numpy as np
from humetric_core import ParsedQuery, Person, normalize_skill

FEATURE_NAMES: Final[tuple[str, ...]] = (
    "bm25",
    "text_cosine",
    "graph_cosine",
    "tower_cosine",
    "skill_overlap",
    "log_followers",
    "recency_recip",
    "history_cosine",
)


@dataclass(frozen=True, slots=True)
class FeatureRow:
    bm25: float = 0.0
    text_cosine: float = 0.0
    graph_cosine: float = 0.0
    tower_cosine: float = 0.0
    skill_overlap: float = 0.0
    log_followers: float = 0.0
    recency_recip: float = 0.0
    history_cosine: float = 0.0

    def to_array(self) -> np.ndarray:
        return np.array(
            [
                self.bm25,
                self.text_cosine,
                self.graph_cosine,
                self.tower_cosine,
                self.skill_overlap,
                self.log_followers,
                self.recency_recip,
                self.history_cosine,
            ],
            dtype=np.float32,
        )


def _skill_overlap(parsed: ParsedQuery, person: Person) -> float:
    targets = {normalize_skill(s) for s in (*parsed.must_skills, *parsed.nice_skills)}
    if not targets:
        return 0.0
    have = {s.normalized for s in person.skills}
    return float(len(targets & have)) / float(len(targets))


def build_feature_row(
    parsed: ParsedQuery,
    person: Person,
    *,
    bm25: float = 0.0,
    text_cosine: float = 0.0,
    graph_cosine: float = 0.0,
    tower_cosine: float = 0.0,
    history_cosine: float = 0.0,
) -> FeatureRow:
    log_f = math.log1p(max(0, person.follower_count))
    days = person.last_active_days_ago if person.last_active_days_ago is not None else 365
    recency = 1.0 / (1.0 + max(0, days))
    return FeatureRow(
        bm25=bm25,
        text_cosine=text_cosine,
        graph_cosine=graph_cosine,
        tower_cosine=tower_cosine,
        skill_overlap=_skill_overlap(parsed, person),
        log_followers=log_f,
        recency_recip=recency,
        history_cosine=history_cosine,
    )


def stack_features(rows: list[FeatureRow]) -> np.ndarray:
    if not rows:
        return np.zeros((0, len(FEATURE_NAMES)), dtype=np.float32)
    return np.stack([r.to_array() for r in rows])
