from pathlib import Path

import pytest

from verilog_causal_analysis import (
    StructuralCausalRequest,
    StructuralContractError,
    STRUCTURAL_REQUEST_SCHEMA,
    build_structural_graph,
    validate_structural_graph,
    policy_identity,
)


def test_request_id_excludes_local_paths_but_binds_artifact_identity(counter_request):
    row = counter_request.to_dict()
    moved = dict(row)
    moved["trace"] = dict(row["trace"], path="/tmp/moved-trace.fst")
    moved["rtl_files"] = [
        dict(row["rtl_files"][0], path="/tmp/moved-rtl.sv")
    ]
    parsed = StructuralCausalRequest.from_dict(moved)
    assert parsed.request_id == counter_request.request_id
    assert parsed.identity_dict() == counter_request.identity_dict()


def test_contract_rejects_unknown_fields_and_non_exact_schema(counter_request):
    row = counter_request.to_dict()
    row["free_signal"] = "not allowed"
    with pytest.raises(StructuralContractError, match="extra"):
        StructuralCausalRequest.from_dict(row)

    row = counter_request.to_dict()
    row["schema_version"] = "unsupported"
    with pytest.raises(StructuralContractError, match=STRUCTURAL_REQUEST_SCHEMA):
        StructuralCausalRequest.from_dict(row)


def test_request_requires_hash_bound_policy_and_explicit_work_bounds(counter_request):
    row = counter_request.to_dict()
    row.pop("search_policy")
    with pytest.raises(StructuralContractError, match="missing"):
        StructuralCausalRequest.from_dict(row)

    row = counter_request.to_dict()
    row["bounds"].pop("max_candidate_evaluations")
    with pytest.raises(StructuralContractError, match="missing"):
        StructuralCausalRequest.from_dict(row)

    row = counter_request.to_dict()
    row["search_policy"]["policy_id"] = "retired_policy"
    with pytest.raises(StructuralContractError, match="one of"):
        StructuralCausalRequest.from_dict_without_id(row)

def test_production_requires_exact_endpoint_clock_cycle_and_strict(counter_request):
    row = counter_request.to_dict()
    row["endpoint"] = {"signal": "", "cycle": 1}
    with pytest.raises(StructuralContractError, match="endpoint.signal"):
        StructuralCausalRequest.from_dict(row)

    row = counter_request.to_dict()
    row["strict"] = False
    row["request_id"] = StructuralCausalRequest.from_dict_without_id(
        row
    ).computed_request_id()
    with pytest.raises(StructuralContractError, match="strict=true"):
        StructuralCausalRequest.from_dict(row)


def test_artifact_paths_must_be_absolute(counter_request):
    row = counter_request.to_dict()
    row["rtl_files"][0]["path"] = str(Path("relative.sv"))
    with pytest.raises(StructuralContractError, match="absolute"):
        StructuralCausalRequest.from_dict(row)


def test_result_contract_rejects_unknown_fields_and_path_leaks(counter_request):
    graph = build_structural_graph(counter_request)
    assert validate_structural_graph(graph)["graph_id"] == graph["graph_id"]

    graph["nodes"][0]["free_path"] = "/tmp/leak"
    with pytest.raises(StructuralContractError, match="extra"):
        validate_structural_graph(graph)
