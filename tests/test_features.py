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
