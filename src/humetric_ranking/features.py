from __future__ import annotations

import datetime
import math
from dataclasses import dataclass
from typing import Final

import numpy as np
from humetric_core import Organization, ParsedQuery, Person, normalize_skill

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

ORG_FEATURE_NAMES: Final[tuple[str, ...]] = (
    "bm25",
    "text_cosine",
    "graph_cosine",
    "tower_cosine",
    "industry_match",
    "log_employee_count",
    "founding_recency",
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


@dataclass(frozen=True, slots=True)
class OrgFeatureRow:
    bm25: float = 0.0
    text_cosine: float = 0.0
    graph_cosine: float = 0.0
    tower_cosine: float = 0.0
    industry_match: float = 0.0
    log_employee_count: float = 0.0
    founding_recency: float = 0.0
    history_cosine: float = 0.0

    def to_array(self) -> np.ndarray:
        return np.array(
            [
                self.bm25,
                self.text_cosine,
                self.graph_cosine,
                self.tower_cosine,
                self.industry_match,
                self.log_employee_count,
                self.founding_recency,
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


def _industry_match(parsed: ParsedQuery, org: Organization) -> float:
    """Treat parsed must/nice skills as industry-tag candidates. Overlap with
    `org.industries` (also lower-hyphenated) gives a soft signal without a
    dedicated industry field on ParsedQuery — which we can add when the LLM
    backend learns to surface it."""
    targets = {normalize_skill(s) for s in (*parsed.must_skills, *parsed.nice_skills)}
    if not targets:
        return 0.0
    have = {normalize_skill(i) for i in org.industries}
    return float(len(targets & have)) / float(len(targets))


def _founding_recency(org: Organization, *, today_year: int | None = None) -> float:
    """Higher score for younger orgs. Missing year → neutral 0.5."""
    if org.founding_year is None:
        return 0.5
    y = today_year if today_year is not None else datetime.date.today().year
    age = max(0, y - org.founding_year)
    return 1.0 / (1.0 + age)


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


def build_feature_row_organization(
    parsed: ParsedQuery,
    org: Organization,
    *,
    bm25: float = 0.0,
    text_cosine: float = 0.0,
    graph_cosine: float = 0.0,
    tower_cosine: float = 0.0,
    history_cosine: float = 0.0,
    today_year: int | None = None,
) -> OrgFeatureRow:
    log_employees = math.log1p(max(0, org.employee_count or 0))
    return OrgFeatureRow(
        bm25=bm25,
        text_cosine=text_cosine,
        graph_cosine=graph_cosine,
        tower_cosine=tower_cosine,
        industry_match=_industry_match(parsed, org),
        log_employee_count=log_employees,
        founding_recency=_founding_recency(org, today_year=today_year),
        history_cosine=history_cosine,
    )


def stack_features(rows: list[FeatureRow] | list[OrgFeatureRow]) -> np.ndarray:
    """Stack either person- or org-feature rows. Both produce float32 arrays
    of shape (n_rows, 8); the column semantics are bound to the entity-type
    of the trained model the array is fed to."""
    if not rows:
        return np.zeros((0, len(FEATURE_NAMES)), dtype=np.float32)
    return np.stack([r.to_array() for r in rows])
