from __future__ import annotations

from humetric_core import ParsedQuery, Person, Skill

from humetric_ranking import FEATURE_NAMES, build_feature_row, stack_features


def test_feature_row_shape() -> None:
    pq = ParsedQuery(free_text="rust", must_skills=("rust",), nice_skills=("kafka",))
    p = Person(
        id="x",
        source="github",
        name="x",
        skills=(Skill.of("rust"), Skill.of("python")),
        follower_count=100,
        last_active_days_ago=10,
    )
    row = build_feature_row(pq, p, bm25=2.5, text_cosine=0.7, graph_cosine=0.4)
    arr = row.to_array()
    assert arr.shape == (len(FEATURE_NAMES),)
    assert row.bm25 == 2.5
    # rust matches one of (rust, kafka) → 0.5 overlap
    assert row.skill_overlap == 0.5
    assert row.log_followers > 4.5
    assert row.recency_recip > 0.0


def test_recency_default_when_missing() -> None:
    pq = ParsedQuery(free_text="x")
    p = Person(id="x", source="github", name="x")  # no last_active_days_ago
    row = build_feature_row(pq, p)
    assert row.recency_recip < 0.01  # 1/(1+365)


def test_stack_features_handles_empty() -> None:
    out = stack_features([])
    assert out.shape == (0, len(FEATURE_NAMES))


def test_org_feature_row_shape_and_industry_match() -> None:
    from humetric_core import Organization

    from humetric_ranking import (
        ORG_FEATURE_NAMES,
        build_feature_row_organization,
        stack_features,
    )

    pq = ParsedQuery(free_text="ai labs", must_skills=("artificial-intelligence",))
    o = Organization(
        id="o:gh:anthropic",
        source="github",
        name="Anthropic",
        org_kind="company",
        industries=("artificial-intelligence", "research"),
        founding_year=2021,
        employee_count=500,
    )
    row = build_feature_row_organization(pq, o, bm25=3.1, text_cosine=0.6, today_year=2026)
    arr = row.to_array()
    assert arr.shape == (len(ORG_FEATURE_NAMES),)
    assert row.bm25 == 3.1
    assert row.industry_match == 1.0  # one target, one match
    assert row.log_employee_count > 0
    # founding_year=2021, today=2026 → age 5 → 1/(1+5) ≈ 0.166...
    assert 0.16 < row.founding_recency < 0.17
    stacked = stack_features([row, row])
    assert stacked.shape == (2, len(ORG_FEATURE_NAMES))


def test_org_founding_recency_neutral_when_year_missing() -> None:
    from humetric_core import Organization

    from humetric_ranking import build_feature_row_organization

    o = Organization(id="o:gh:x", source="github", name="x", org_kind="company")
    row = build_feature_row_organization(ParsedQuery(free_text="x"), o)
    assert row.founding_recency == 0.5
    assert row.log_employee_count == 0.0
