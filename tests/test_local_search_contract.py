import math

import pytest

from verilog_causal_analysis import (
    FEATURE_SCHEMA,
    POLICY_REGISTRY,
    LocalSearchContractError,
    ScoreFeatures,
    SearchPolicyIdentity,
    make_search_summary,
    policy_identity,
    validate_search_summary,
)


def test_policy_payloads_and_hashes_are_frozen():
    assert list(POLICY_REGISTRY) == ["d2_backward_v1"]
    for policy_id, policy in POLICY_REGISTRY.items():
        assert policy.payload["policy_id"] == policy_id
        assert policy.payload["feature_schema"] == FEATURE_SCHEMA
        assert policy.payload["scheduler_kind"] == "deterministic_backward_dfs"
        assert policy.payload["max_interventions_per_candidate"] == 4


def test_policy_identity_rejects_unknown_or_hash_mismatch():
    identity = policy_identity()
    assert SearchPolicyIdentity.from_dict(identity.to_dict()) == identity
    with pytest.raises(LocalSearchContractError, match="mismatch"):
        SearchPolicyIdentity.from_dict(
            {**identity.to_dict(), "policy_sha256": "0" * 64}
        )
    with pytest.raises(LocalSearchContractError, match="one of"):
        SearchPolicyIdentity.from_dict(
            {**identity.to_dict(), "policy_id": "unregistered"}
        )


def test_score_features_distinguish_zero_from_unavailable():
    names = (
        "C_cf",
        "C_obs",
        "C_time",
        "C_ctrl",
        "C_sem",
        "C_structural",
        "P_unknown",
        "P_ambiguity",
        "P_temp",
        "P_fanout",
    )
    row = {
        "feature_vector": {name: 0.0 for name in names},
        "feature_availability": {name: "available" for name in names},
    }
    row["feature_vector"]["C_sem"] = None
    row["feature_availability"]["C_sem"] = "not_applicable"
    parsed = ScoreFeatures.from_dict(row)
    assert parsed.values["C_cf"] == 0.0
    assert parsed.availability["C_cf"] == "available"
    assert parsed.values["C_sem"] is None

    bad = {
        "feature_vector": dict(row["feature_vector"], C_cf=math.nan),
        "feature_availability": row["feature_availability"],
    }
    with pytest.raises(LocalSearchContractError, match="finite"):
        ScoreFeatures.from_dict(bad)


def test_search_summary_is_exact_hash_bound_and_path_free():
    identity = policy_identity()
    summary = make_search_summary(
        identity,
        termination_reason="frontier_exhausted",
        seed_count=1,
        expanded_nodes=3,
        candidate_evaluations=4,
        admitted_nodes=3,
        admitted_edges=2,
        exploit_expansions=3,
    )
    assert validate_search_summary(summary, expected_policy=identity) == summary
    assert summary["frontier_remaining"] == 0
    with pytest.raises(LocalSearchContractError, match="extra"):
        validate_search_summary({**summary, "absolute_path": "/tmp/leak"})
