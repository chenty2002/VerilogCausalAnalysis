from copy import deepcopy
import json

import pytest

from verilog_causal_analysis import (
    ContractError,
    build_causal_graph,
    get_waitfor_component,
    make_protocol_adapter,
    make_request,
    sha256_file,
    validate_protocol_adapter,
)
from verilog_causal_analysis.identity import stable_set_sha256
from verilog_causal_analysis.waitfor_graph import (
    WaitForError,
    build_c5_waitfor_layer,
    load_protocol_adapter,
)


def _base_nodes():
    return [
        {
            "semantic_id": "handshake",
            "type": "handshake",
            "instance_path": "Top.pipe",
        },
        {
            "semantic_id": "stall",
            "type": "stall_interval",
            "handshake_id": "handshake",
            "start_cycle": 2,
            "end_cycle": 10,
            "last_accept_cycle": 1,
            "evidence_strength": "fully_observed",
        },
        {
            "semantic_id": "aggregate",
            "type": "aggregate",
            "instance_path": "Top.pipe",
        },
        {
            "semantic_id": "pipeline",
            "type": "pipeline",
            "instance_path": "Top.pipe",
        },
        {
            "semantic_id": "blocker",
            "type": "blocking_relation",
            "instance_path": "Top.pipe",
            "statement_ids": ["stmt_block"],
        },
        {
            "semantic_id": "missing",
            "type": "missing_expected_completion",
            "register_id": "mshr_active",
            "start_cycle": 2,
            "end_cycle": 10,
            "evidence_refs": ["active_interval", "clear_rule"],
        },
        {
            "semantic_id": "mshr_active",
            "type": "register_transition",
        },
    ]


def _base_edges():
    return [
        {
            "edge_id": "stall_handshake",
            "src_semantic_id": "stall",
            "dst_semantic_id": "handshake",
            "relation": "persistent_stall",
        },
        {
            "edge_id": "aggregate_handshake",
            "src_semantic_id": "aggregate",
            "dst_semantic_id": "handshake",
            "relation": "ready_valid_semantics",
        },
        {
            "edge_id": "aggregate_pipeline",
            "src_semantic_id": "aggregate",
            "dst_semantic_id": "pipeline",
            "relation": "pipeline_stage",
        },
        {
            "edge_id": "pipeline_blocker",
            "src_semantic_id": "pipeline",
            "dst_semantic_id": "blocker",
            "relation": "pipeline_blocker",
        },
    ]


def _build(nodes=None, edges=None, roots=None, adapter=None):
    return build_c5_waitfor_layer(
        semantic_nodes=deepcopy(nodes if nodes is not None else _base_nodes()),
        edges=deepcopy(edges if edges is not None else _base_edges()),
        root_candidates=deepcopy(
            roots
            if roots is not None
            else [{"semantic_id": "blocker", "semantic_path": ["blocker"]}]
        ),
        endpoint_cycle=10,
        max_waitfor_nodes=120,
        max_waitfor_edges=240,
        max_scc_candidates=8,
        protocol_adapter=adapter,
        rtl_set_sha256=("5" * 64 if adapter is not None else None),
    )


def test_generic_waitfor_keeps_handshake_pipeline_and_mshr_evidence_open():
    nodes, edges, roots, diagnostics, work = _build()
    assert diagnostics == []
    waits = [row for row in edges if row.get("relation") == "waits_for"]
    assert {
        row["inference_rule"] for row in waits
    } == {
        "active_state_waits_for_completion",
        "pipeline_admission_waits_for_blocker",
        "pipeline_blocker_waits_for_release",
    }
    assert any(
        row["waiter_id"] == "stall" and row["awaited_id"] == "blocker"
        for row in waits
    )
    components = [
        row for row in nodes if row["type"] == "waitfor_component"
    ]
    pipeline_component = next(
        row for row in components if "blocker" in row["members"]
    )
    assert pipeline_component["closed"] is False
    assert pipeline_component["classification"] == "incomplete"
    assert pipeline_component["external_dependencies"]
    assert all(
        row["formal_verdict"] == "not_established" for row in components
    )
    assert roots[0]["waitfor_membership"] is True
    assert work["protocol_adapter_used"] is False


def test_persistent_pipeline_seed_materializes_open_wait_without_stall():
    nodes = [
        {
            "semantic_id": "pipeline",
            "type": "pipeline",
        },
        {
            "semantic_id": "persistent_blocker",
            "type": "blocking_relation",
            "blockers": [
                {
                    "pipeline_id": "pipeline",
                    "stage_index": 1,
                    "member_signals": ["Top.valid_s1"],
                }
            ],
        },
        {
            "semantic_id": "related_root",
            "type": "blocking_relation",
        },
    ]
    roots = [
        {
            "candidate_id": "seed_candidate",
            "semantic_id": "persistent_blocker",
            "semantic_path": ["persistent_blocker"],
            "seed": {
                "derivation_rule": "persistent_pipeline_blocker",
                "interval": [2, 10],
                "evidence_refs": ["pipeline"],
            },
        },
        {
            "candidate_id": "related_candidate",
            "semantic_id": "related_root",
            "semantic_path": [
                "related_root",
                "pipeline",
                "persistent_blocker",
            ],
        },
    ]
    result_nodes, result_edges, result_roots, diagnostics, _work = _build(
        nodes=nodes, edges=[], roots=roots
    )
    assert diagnostics == []
    wait = next(
        row
        for row in result_edges
        if row.get("inference_rule")
        == "persistent_pipeline_blocker_waits_for_release"
    )
    assert wait["waiter_id"] == "persistent_blocker"
    assert wait["evidence_strength"] == "exact_rtl_waveform"
    component = next(
        row for row in result_nodes if row["type"] == "waitfor_component"
    )
    assert {
        "persistent_blocker",
        "pipeline",
        "related_root",
    } <= set(component["members"])
    assert component["closed"] is False
    assert component["classification"] == "incomplete"
    assert all(row["waitfor_membership"] for row in result_roots)


def test_closed_cycle_is_only_a_deadlock_candidate_not_a_formal_verdict():
    nodes = [
        {
            "semantic_id": "resource_a",
            "type": "allocation_wait",
            "awaited_semantic_id": "resource_b",
            "start_cycle": 2,
            "end_cycle": 10,
            "evidence_strength": "transition_supported",
        },
        {
            "semantic_id": "resource_b",
            "type": "allocation_wait",
            "awaited_semantic_id": "resource_a",
            "start_cycle": 2,
            "end_cycle": 10,
            "evidence_strength": "transition_supported",
        },
    ]
    result_nodes, _edges, _roots, diagnostics, work = _build(
        nodes=nodes, edges=[], roots=[]
    )
    assert diagnostics == []
    scc = next(row for row in result_nodes if row["type"] == "waitfor_scc")
    assert scc["members"] == ["resource_a", "resource_b"]
    assert scc["closed"] is True
    assert scc["classification"] == "deadlock_candidate"
    assert scc["formal_verdict"] == "not_established"
    assert work["scc_candidates"] == 1


def test_open_cycle_discloses_external_dependency_and_stays_candidate():
    nodes = [
        {
            "semantic_id": "resource_a",
            "type": "allocation_wait",
            "awaited_semantic_id": "resource_b",
            "start_cycle": 2,
            "end_cycle": 10,
            "evidence_strength": "transition_supported",
        },
        {
            "semantic_id": "resource_b",
            "type": "allocation_wait",
            "awaited_semantic_id": "resource_a",
            "start_cycle": 2,
            "end_cycle": 10,
            "evidence_strength": "transition_supported",
        },
        {
            "semantic_id": "external_response",
            "type": "unknown_external_completion",
            "external": True,
            "reason": "environment response remains possible",
        },
    ]
    adapter = make_protocol_adapter(
        protocol="tilelink",
        rtl_set_sha256="5" * 64,
        review={
            "status": "approved",
            "reviewer": "codex",
            "evidence_refs": ["review_record_1"],
        },
        channels=[],
        dependencies=[
            {
                "waiter_ref": "semantic:resource_a",
                "awaited_ref": "semantic:external_response",
                "inference_rule": "reviewed_tilelink.external_response",
                "evidence_refs": ["environment_contract_1"],
            }
        ],
    )
    result_nodes, _edges, _roots, diagnostics, _work = _build(
        nodes=nodes, edges=[], roots=[], adapter=adapter
    )
    assert diagnostics == []
    scc = next(row for row in result_nodes if row["type"] == "waitfor_scc")
    assert scc["closed"] is False
    assert scc["classification"] == "deadlock_candidate"
    assert scc["external_dependencies"] == ["external_response"]
    assert scc["formal_verdict"] == "not_established"


def test_starvation_requires_observed_other_progress():
    nodes = [
        {
            "semantic_id": "request",
            "type": "arbiter_wait",
            "awaited_semantic_id": "grant",
            "start_cycle": 2,
            "end_cycle": 10,
            "other_progress_observed": True,
            "evidence_strength": "transition_supported",
        },
        {
            "semantic_id": "grant",
            "type": "arbiter_grant",
        },
    ]
    result_nodes, _edges, _roots, diagnostics, _work = _build(
        nodes=nodes, edges=[], roots=[]
    )
    assert diagnostics == []
    component = next(
        row for row in result_nodes if row["type"] == "waitfor_component"
    )
    assert component["classification"] == "starvation_candidate"
    assert component["closed"] is False
    assert not any(row["type"] == "waitfor_scc" for row in result_nodes)


def test_cleared_backpressure_and_unrelated_activity_are_not_waits():
    nodes = [
        {
            "semantic_id": "old_stall",
            "type": "stall_interval",
            "handshake_id": "old_handshake",
            "start_cycle": 1,
            "end_cycle": 4,
            "evidence_strength": "fully_observed",
        },
        {
            "semantic_id": "old_handshake",
            "type": "handshake",
        },
        {
            "semantic_id": "unrelated_valid",
            "type": "handshake",
        },
    ]
    result_nodes, result_edges, _roots, diagnostics, work = _build(
        nodes=nodes, edges=[], roots=[]
    )
    assert diagnostics == []
    assert not any(
        row.get("relation") == "waits_for" for row in result_edges
    )
    assert not any(
        row["type"] in {"waitfor_component", "waitfor_scc"}
        for row in result_nodes
    )
    assert work["waitfor_edges"] == 0


def test_reviewed_adapter_is_hash_bound_and_optional(tmp_path):
    nodes = _base_nodes()
    adapter = make_protocol_adapter(
        protocol="tilelink",
        rtl_set_sha256="5" * 64,
        review={
            "status": "approved",
            "reviewer": "codex",
            "evidence_refs": ["review_record_1"],
        },
        channels=[
            {
                "channel_id": "probe_ack",
                "handshake_id": "handshake",
                "role": "response",
                "event": "ProbeAckData",
            }
        ],
        dependencies=[
            {
                "waiter_ref": "semantic:missing",
                "awaited_ref": "channel:probe_ack",
                "inference_rule": "reviewed_tilelink.mshr_waits_probe_ack",
                "evidence_refs": ["pair_review_1"],
            }
        ],
    )
    validated = validate_protocol_adapter(
        adapter,
        rtl_set_sha256="5" * 64,
        known_semantic_ids={row["semantic_id"] for row in nodes},
    )
    adapter_path = tmp_path / "adapter.json"
    adapter_path.write_text(json.dumps(validated, sort_keys=True))
    adapter_sha256, adapter_bytes = sha256_file(adapter_path)
    assert (
        load_protocol_adapter(
            str(adapter_path),
            sha256=adapter_sha256,
            bytes=adapter_bytes,
            rtl_set_sha256="5" * 64,
            known_semantic_ids={row["semantic_id"] for row in nodes},
        )
        == validated
    )
    result_nodes, result_edges, _roots, diagnostics, work = _build(
        adapter=validated
    )
    assert diagnostics == []
    protocol = next(
        row for row in result_nodes if row["type"] == "protocol_transaction"
    )
    assert protocol["event"] == "ProbeAckData"
    assert any(
        row.get("inference_rule")
        == "reviewed_tilelink.mshr_waits_probe_ack"
        for row in result_edges
    )
    assert work["protocol_adapter_used"] is True

    with pytest.raises(WaitForError, match="rtl_set_sha256 mismatch"):
        validate_protocol_adapter(
            adapter,
            rtl_set_sha256="6" * 64,
            known_semantic_ids={row["semantic_id"] for row in nodes},
        )


def test_waitfor_bounds_fail_closed_and_component_query_is_canonical():
    nodes, edges, _roots, diagnostics, work = build_c5_waitfor_layer(
        semantic_nodes=_base_nodes(),
        edges=_base_edges(),
        root_candidates=[],
        endpoint_cycle=10,
        max_waitfor_nodes=2,
        max_waitfor_edges=1,
        max_scc_candidates=1,
    )
    assert work["waitfor_edges_reached"] is True
    assert any(row["breaks_complete"] for row in diagnostics)
    component = next(
        row for row in nodes if row["type"] == "waitfor_component"
    )
    graph = {
        "graph_id": "graph",
        "semantic_nodes": nodes,
        "edges": edges,
    }
    first = get_waitfor_component(graph, component["semantic_id"])
    second = get_waitfor_component(graph, component["semantic_id"])
    assert first == second
    assert len(first["result_sha256"]) == 64


def test_c5_contract_requires_full_temporal_semantic_stack(counter_request):
    rtl = counter_request.rtl_files[0]
    common = {
        "trace": {
            "artifact_id": "trace_0001",
            **counter_request.trace.to_dict(),
        },
        "rtl_files": [rtl.to_dict()],
        "clock": {
            "signal": counter_request.clock_signal,
            "edge": "rising",
        },
        "endpoint": {
            "signal": counter_request.endpoint_signal,
            "cycle": counter_request.endpoint_cycle,
            "projection": None,
        },
        "semantic_inputs": [],
        "search_policy": counter_request.search_policy.to_dict(),
        "bounds": {
            "max_signal_depth": 12,
            "max_signal_nodes": 120,
            "max_expanded_nodes": 120,
            "max_candidate_evaluations": 960,
            "max_intervention_evaluations": 3840,
            "max_semantic_nodes": 64,
            "max_edges": 240,
            "max_seed_count": 8,
            "max_intervals_per_signal": 64,
            "max_temporal_samples": 64000,
            "max_waitfor_nodes": 120,
            "max_waitfor_edges": 240,
            "max_scc_candidates": 8,
        },
        "random_seed": 0,
        "strict": True,
    }
    with pytest.raises(ContractError, match="waitfor requires"):
        make_request(
            **common,
            semantic_profile={
                "name": "chisel",
                "version": "chisel-semantic-profile",
                "features": ["instance_graph", "waitfor"],
            },
        )


def test_engine_materializes_c5_revision_and_waitfor_work(
    counter_request, tmp_path
):
    rtl = counter_request.rtl_files[0]
    member = "Counter.bit1.value"
    projection_path = tmp_path / "projection.json"
    projection_path.write_text(
        json.dumps(
            {
                "schema_version": "assertion_endpoint_projection",
                "endpoint_signal": counter_request.endpoint_signal,
                "endpoint_cycle": counter_request.endpoint_cycle,
                "clock_signal": counter_request.clock_signal,
                "predicate_members": [member],
                "rtl_set_sha256": stable_set_sha256(
                    [
                        {
                            "artifact_id": rtl.artifact_id,
                            "sha256": rtl.sha256,
                            "bytes": rtl.bytes,
                        }
                    ]
                ),
                "trace_sha256": counter_request.trace.sha256,
            },
            sort_keys=True,
        )
    )
    projection_sha256, projection_bytes = sha256_file(projection_path)
    request = make_request(
        trace={
            "artifact_id": "trace_0001",
            **counter_request.trace.to_dict(),
        },
        rtl_files=[rtl.to_dict()],
        semantic_profile={
            "name": "chisel",
            "version": "chisel-semantic-profile",
            "features": [
                "instance_graph",
                "endpoint_projection",
                "compiler_net_normalization",
                "register_transition",
                "aggregate",
                "handshake",
                "pipeline",
                "temporal_interval",
                "waitfor",
            ],
        },
        clock={"signal": counter_request.clock_signal, "edge": "rising"},
        endpoint={
            "signal": counter_request.endpoint_signal,
            "cycle": counter_request.endpoint_cycle,
            "projection": {
                "mode": "controller_supplied_exact",
                "predicate_members": [member],
                "evidence_ref": "projection_0001",
            },
        },
        semantic_inputs=[
            {
                "artifact_id": "projection_0001",
                "kind": "assertion_endpoint_projection",
                "path": str(projection_path),
                "sha256": projection_sha256,
                "bytes": projection_bytes,
            }
        ],
        search_policy=counter_request.search_policy.to_dict(),
        bounds={
            "max_signal_depth": 12,
            "max_signal_nodes": 120,
            "max_expanded_nodes": 120,
            "max_candidate_evaluations": 960,
            "max_intervention_evaluations": 3840,
            "max_semantic_nodes": 128,
            "max_edges": 320,
            "max_seed_count": 8,
            "max_intervals_per_signal": 64,
            "max_temporal_samples": 64000,
            "max_waitfor_nodes": 120,
            "max_waitfor_edges": 240,
            "max_scc_candidates": 8,
        },
        random_seed=0,
        strict=True,
    )
    first = build_causal_graph(request)
    second = build_causal_graph(request)
    assert first == second
    assert first["identity"]["analyzer_revision"].endswith("+c5")
    assert "waitfor_work" in first["bounds"]
    assert first["graph_id"] == second["graph_id"]
