from verilog_causal_analysis import (
    CONTRIBUTION_SCHEMA,
    ContributionWorkCounter,
    ContributionContractError,
    ContributionEvidence,
    LegacyContributionEvidence,
    adapt_legacy_contribution,
    contribution_cache_key,
    contribution_edge_fields,
    evaluate_interventions,
    generate_interventions,
    route_contribution,
    structural_evidence,
    toggle_evidence,
)
import pytest


def test_direct_one_bit_expression_has_exact_supported_score():
    evidence = evaluate_interventions(
        source_value="0",
        original_result="0",
        observed_target="0",
        intervention_values=("1",),
        intervention_results=("1",),
        target_basis="exact_lhs_slice",
    )
    assert evidence.schema_version == CONTRIBUTION_SCHEMA
    assert evidence.status == "supported"
    assert evidence.score == 1.0
    assert evidence.effects.max_impact == 1.0
    assert ContributionEvidence.from_dict(evidence.to_dict()) == evidence
    with pytest.raises(ContributionContractError, match="extra"):
        ContributionEvidence.from_dict({**evidence.to_dict(), "extra": True})


def test_wide_target_uses_only_exact_relevant_mask_without_half_floor():
    irrelevant = evaluate_interventions(
        source_value="0000",
        original_result="0000",
        observed_target="0000",
        intervention_values=("0001",),
        intervention_results=("1000",),
        target_basis="exact_endpoint_slice",
        relevant_target_mask=(0,),
    )
    assert irrelevant.status == "not_supported"
    assert irrelevant.score == 0.0

    relevant = evaluate_interventions(
        source_value="0000",
        original_result="0000",
        observed_target="0000",
        intervention_values=("0001",),
        intervention_results=("0001",),
        target_basis="exact_endpoint_slice",
        relevant_target_mask=(0,),
    )
    assert relevant.status == "supported"
    assert relevant.score == 1.0

    normalized = evaluate_interventions(
        source_value="0",
        original_result="0",
        observed_target="0000",
        intervention_values=("1",),
        intervention_results=("0001",),
        target_basis="full_exact_target",
    )
    assert normalized.fidelity.status == "normalized_match"
    assert normalized.relevant_target.known_bit_count == 4


def test_intervention_generator_is_canonical_bounded_and_deduplicated():
    assert generate_interventions("0") == ("1",)
    values = generate_interventions("0101", operator="==", literal="0011")
    assert values == ("0011", "0010", "0100", "1101")
    assert len(values) == len(set(values)) == 4
    assert generate_interventions("01x1") == ()


def test_complete_no_effect_is_negative_but_truncation_is_fail_closed():
    complete = evaluate_interventions(
        source_value="00",
        original_result="0",
        observed_target="0",
        intervention_values=("01", "10"),
        intervention_results=("0", "0"),
        target_basis="full_exact_target",
    )
    assert complete.status == "not_supported"
    assert complete.search_available is True

    truncated = evaluate_interventions(
        source_value="00",
        original_result="0",
        observed_target="0",
        intervention_values=("01", "10"),
        intervention_results=("0", None),
        target_basis="full_exact_target",
        global_budget_truncated=True,
    )
    assert truncated.status == "inconclusive"
    assert truncated.search_available is False


def test_partial_positive_keeps_supported_with_lower_coverage():
    partial = evaluate_interventions(
        source_value="00",
        original_result="0",
        observed_target="0",
        intervention_values=("01", "10"),
        intervention_results=("1", None),
        target_basis="full_exact_target",
        global_budget_truncated=True,
    )
    complete = evaluate_interventions(
        source_value="00",
        original_result="0",
        observed_target="0",
        intervention_values=("01", "10"),
        intervention_results=("1", "1"),
        target_basis="full_exact_target",
    )
    assert partial.status == "supported"
    assert partial.score < complete.score
    assert partial.interventions.complete is False


def test_original_result_mismatch_and_inactive_rule_fail_closed():
    mismatch = evaluate_interventions(
        source_value="0",
        original_result="1",
        observed_target="0",
        intervention_values=("1",),
        intervention_results=("0",),
        target_basis="exact_lhs_slice",
    )
    assert mismatch.status == "inconclusive"
    assert mismatch.fidelity.status == "mismatch"
    assert mismatch.reason_code == "original_result_mismatch"

    # An inactive data rule is represented by an unavailable original result;
    # LS-C supplies predicate results separately for guard-source evaluation.
    inactive = evaluate_interventions(
        source_value="0",
        original_result=None,
        observed_target="0",
        intervention_values=("1",),
        intervention_results=(None,),
        target_basis="exact_lhs_slice",
        method="active_rule_intervention",
        rule_active=False,
    )
    assert inactive.status == "inconclusive"
    assert inactive.score == 0.0
    assert inactive.reason_code == "inactive_data_rule"


def test_redundant_guard_has_no_blanket_score_and_routes_only_to_control():
    # a || b with a=b=1 remains true when only a is complemented.
    guard = evaluate_interventions(
        source_value="1",
        original_result="1",
        observed_target="1",
        intervention_values=("0",),
        intervention_results=("1",),
        target_basis="exact_predicate",
        method="branch_predicate_intervention",
    )
    routed = route_contribution(guard)
    assert guard.status == "not_supported"
    assert guard.score == 0.0
    assert routed.feature_name == "C_ctrl"
    assert routed.value == 0.0


def test_toggle_and_structural_reliability_caps_and_unique_routes():
    toggle = toggle_evidence(source_toggled=True, target_toggled=True)
    structural = structural_evidence()
    assert toggle.score == 0.45
    assert route_contribution(toggle).feature_name == "C_time"
    assert structural.score == 0.15
    assert route_contribution(structural).feature_name == "C_structural"


def test_work_counters_and_cache_key_dimensions_are_independent():
    counter = ContributionWorkCounter()
    counter.record_candidate()
    counter.record_intervention()
    counter.record_intervention()
    assert (counter.candidate_evaluations, counter.intervention_evaluations) == (1, 2)

    base = {
        "statement_key": "stmt",
        "source_identity": "top.a",
        "source_cycle": 3,
        "intervention_value": "1",
        "target_identity": "top.y",
        "target_cycle": 4,
        "relevant_target_basis": "exact_lhs_slice",
        "relevant_target_mask": (0,),
        "analyzer_revision": "rev-a",
    }
    digest = contribution_cache_key(**base)
    assert digest != contribution_cache_key(**{**base, "target_cycle": 5})
    assert digest != contribution_cache_key(**{**base, "relevant_target_mask": (1,)})
    assert digest != contribution_cache_key(**{**base, "analyzer_revision": "rev-b"})


def test_edge_projection_and_frozen_legacy_fixture():
    evidence = structural_evidence()
    edge = contribution_edge_fields(evidence)
    assert edge["contribution_score"] == edge["contribution_evidence"]["score"]

    legacy = adapt_legacy_contribution(
        legacy_method="expression_counterfactual",
        legacy_score=1.0,
        expression_evaluations=1,
        intervention_evaluations=1,
        change_examples=(
            {
                "type": "counterfactual",
                "source_original": "0",
                "source_perturbed": "1",
                "target_original": "0",
                "target_perturbed": "1",
            },
        ),
    )
    assert legacy.to_dict() == {
        "schema_version": "legacy_contribution_v1",
        "legacy_method": "expression_counterfactual",
        "legacy_score": 1.0,
        "expression_evaluations": 1,
        "intervention_evaluations": 1,
        "change_examples": [
            {
                "type": "counterfactual",
                "source_original": "0",
                "source_perturbed": "1",
                "target_original": "0",
                "target_perturbed": "1",
            }
        ],
    }
    legacy_edge = contribution_edge_fields(legacy)
    assert legacy_edge["contribution_score"] == legacy_edge["contribution_evidence"]["legacy_score"]
