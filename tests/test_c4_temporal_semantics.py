from copy import deepcopy
import json

from verilog_causal_analysis import (
    build_causal_graph,
    build_transition_intervals,
    get_semantic_paths,
    make_request,
    sha256_file,
)
from verilog_causal_analysis.identity import stable_set_sha256
from verilog_causal_analysis.temporal_semantics import (
    build_c4_temporal_layer,
    c4_enabled,
)


class TransitionWaveform:
    def __init__(self):
        self.values = {
            "Top.timer": ["000", "001", "010", "011", "100", "100"],
        }

    def get_signal_value(self, signal, cycle):
        return self.values.get(signal, [None] * 6)[cycle]

    def get_value_changes_bounded(
        self, signal, start_cycle, end_cycle, *, max_changes
    ):
        values = self.values[signal]
        rows = []
        for cycle in range(max(1, start_cycle), end_cycle + 1):
            if values[cycle] != values[cycle - 1]:
                rows.append((cycle, values[cycle - 1], values[cycle]))
        return rows[:max_changes]


def _semantic_fixture():
    register = {
        "semantic_id": "reg_timer",
        "type": "register_transition",
        "signal": "Top.timer",
        "counter_pattern": "bounded_progress_counter",
        "reset_rules": [
            {
                "rule_id": "rule_reset",
                "update_kind": "reset",
            }
        ],
        "update_rules": [
            {
                "rule_id": "rule_clear",
                "update_kind": "clear",
            },
            {
                "rule_id": "rule_increment",
                "update_kind": "increment",
            },
        ],
    }
    nodes = [
        {
            "semantic_id": "predicate",
            "type": "assertion_predicate",
            "member_signals": ["Top.timer"],
        },
        register,
        {
            "semantic_id": "active_interval",
            "type": "persistent_interval",
            "register_id": "reg_timer",
            "subject_id": "rule_increment",
            "signal": "Top.timer",
            "start_cycle": 1,
            "end_cycle": 5,
            "rule": "increment",
            "observation": "fully_observed",
            "dynamic_score": 1.0,
        },
        {
            "semantic_id": "aggregate",
            "type": "aggregate",
            "instance_path": "Top",
        },
        {
            "semantic_id": "handshake",
            "type": "handshake",
            "instance_path": "Top",
        },
        {
            "semantic_id": "stall",
            "type": "stall_interval",
            "handshake_id": "handshake",
            "start_cycle": 2,
            "end_cycle": 5,
            "last_accept_cycle": 1,
            "evidence_strength": "fully_observed",
        },
        {
            "semantic_id": "pipeline",
            "type": "pipeline",
            "instance_path": "Top",
        },
        {
            "semantic_id": "blocker",
            "type": "blocking_relation",
            "instance_path": "Top",
            "target_signal": "Top.blockB_s1",
        },
    ]
    edges = [
        {
            "edge_id": "edge_interval_register",
            "src_semantic_id": "active_interval",
            "dst_semantic_id": "reg_timer",
            "relation": "persistent_update_rule",
        },
        {
            "edge_id": "edge_register_predicate",
            "src_semantic_id": "reg_timer",
            "dst_semantic_id": "predicate",
            "relation": "predicate_member",
        },
        {
            "edge_id": "edge_stall_handshake",
            "src_semantic_id": "stall",
            "dst_semantic_id": "handshake",
            "relation": "persistent_stall",
        },
        {
            "edge_id": "edge_aggregate_handshake",
            "src_semantic_id": "aggregate",
            "dst_semantic_id": "handshake",
            "relation": "ready_valid_semantics",
        },
        {
            "edge_id": "edge_aggregate_pipeline",
            "src_semantic_id": "aggregate",
            "dst_semantic_id": "pipeline",
            "relation": "pipeline_stage",
        },
        {
            "edge_id": "edge_pipeline_blocker",
            "src_semantic_id": "pipeline",
            "dst_semantic_id": "blocker",
            "relation": "pipeline_blocker",
        },
    ]
    return nodes, edges


def test_transition_index_builder_reports_exact_boundaries_without_cycle_scan():
    result = build_transition_intervals(
        TransitionWaveform(),
        "Top.timer",
        start_cycle=0,
        end_cycle=5,
        max_transition_values=8,
    )
    assert result["available"] is True
    assert result["truncated"] is False
    assert result["boundary_values"] == {"start": "000", "end": "100"}
    assert [row["cycle"] for row in result["changes"]] == [1, 2, 3, 4]
    assert result["intervals"][-1] == {
        "start_cycle": 4,
        "end_cycle": 5,
        "value": "100",
        "coverage": "exact",
    }
    assert result["work"] == {
        "transition_values": 4,
        "value_misses": 0,
    }


def test_transition_interval_reports_xz_coverage_as_unknown():
    waveform = TransitionWaveform()
    waveform.values["Top.timer"] = [
        "000",
        "001",
        "xxx",
        "xxx",
        "010",
        "010",
    ]
    result = build_transition_intervals(
        waveform,
        "Top.timer",
        start_cycle=0,
        end_cycle=5,
        max_transition_values=8,
    )
    assert result["unknown_spans"] == [[2, 3]]
    assert next(
        row for row in result["intervals"] if row["start_cycle"] == 2
    )["coverage"] == "unknown"


def test_c4_derives_bounded_multi_seed_last_progress_and_ranked_blocker():
    original_nodes, original_edges = _semantic_fixture()
    nodes, edges, roots, diagnostics, work = build_c4_temporal_layer(
        normalized_design={"register_transitions": []},
        waveform=TransitionWaveform(),
        endpoint_cycle=5,
        semantic_nodes=deepcopy(original_nodes),
        edges=deepcopy(original_edges),
        max_seed_count=8,
        max_transition_values=16,
    )
    assert diagnostics == []
    assert work == {
        "transition_values": 4,
        "waveform_value_misses": 0,
        "dependency_evaluations": 20,
        "seed_candidates": 6,
        "seeds_retained": 6,
        "semantic_paths_evaluated": 7,
        "transition_values_reached": False,
    }
    node_types = {row["type"] for row in nodes}
    assert {
        "threshold_crossing",
        "missing_expected_completion",
        "last_progress_event",
    } <= node_types
    crossing = next(row for row in nodes if row["type"] == "threshold_crossing")
    assert crossing["cycle"] == 4
    assert crossing["threshold"] == 4
    progress = next(row for row in nodes if row["type"] == "last_progress_event")
    assert progress["cycle"] == 1
    assert progress["event_type"] == "handshake_accepted"
    assert len(
        {
            row["seed"]["candidate_id"]
            for row in roots
            if "seed" in row
        }
    ) <= 8
    blocker = next(row for row in roots if row["semantic_id"] == "blocker")
    assert blocker["causal_distance"] == 4
    assert blocker["semantic_path"] == [
        "blocker",
        "pipeline",
        "aggregate",
        "handshake",
        "stall",
    ]
    assert any(
        row["relation"] == "prevents_completion" for row in edges
    )

    graph = {
        "graph_id": "graph",
        "semantic_nodes": nodes,
        "root_candidates": roots,
    }
    query = get_semantic_paths(
        graph, "blocker", max_paths=1, max_length=8
    )
    assert query["paths"][0]["semantic_path"][-1] == "stall"
    assert len(query["result_sha256"]) == 64


def test_seed_truncation_is_priority_then_stable_id_and_repeatable():
    nodes, edges = _semantic_fixture()
    first = build_c4_temporal_layer(
        normalized_design={"register_transitions": []},
        waveform=TransitionWaveform(),
        endpoint_cycle=5,
        semantic_nodes=deepcopy(nodes),
        edges=deepcopy(edges),
        max_seed_count=3,
        max_transition_values=16,
    )
    second = build_c4_temporal_layer(
        normalized_design={"register_transitions": []},
        waveform=TransitionWaveform(),
        endpoint_cycle=5,
        semantic_nodes=deepcopy(nodes),
        edges=deepcopy(edges),
        max_seed_count=3,
        max_transition_values=16,
    )
    assert first == second
    assert first[4]["seed_candidates"] == 6
    assert first[4]["seeds_retained"] == 3
    retained_rules = {
        row["seed"]["derivation_rule"]
        for row in first[2]
        if "seed" in row
    }
    assert retained_rules == {
        "predicate_counter_operand",
        "threshold_crossing",
        "persistent_active_guard",
    }
    assert c4_enabled(["instance_graph", "temporal_interval"]) is True


def test_semantic_c4_is_explicit_hash_bound_and_canonical(counter_request, tmp_path):
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
                "temporal_interval",
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
            "max_semantic_nodes": 64,
            "max_edges": 240,
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
    assert first["identity"]["analyzer_revision"].endswith("+c4")
    assert first["bounds"]["temporal_work"]["seeds_retained"] <= 8
    assert first["graph_id"] == second["graph_id"]
