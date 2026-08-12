from pathlib import Path
import json

from verilog_causal_analysis import (
    build_causal_graph,
    get_raw_members,
    get_register_transition,
    make_request,
    sha256_file,
)
from verilog_causal_analysis.chisel_semantics import (
    build_normalized_design,
    persistent_intervals,
)
from verilog_causal_analysis.instance_graph import InstanceGraph
from verilog_causal_analysis.verilog_parser import VerilogParser
from verilog_causal_analysis.identity import stable_set_sha256


ROOT = Path(__file__).resolve().parent
FIXTURE = (ROOT / "c2" / "register_semantics.sv").resolve()


def _normalized():
    parser = VerilogParser(strict=True)
    parser.parse_files_strict([str(FIXTURE)])
    graph = InstanceGraph.from_parser(
        parser, {str(FIXTURE): "rtl_0001"}
    )
    return build_normalized_design(
        graph,
        rtl_set_sha256="0" * 64,
        clock_signal="C2Top.clock",
        features=[
            "instance_graph",
            "compiler_net_normalization",
            "register_transition",
        ],
    )


def test_alias_expression_and_register_rules_are_exact_and_reversible():
    normalized = _normalized()
    assert normalized["schema_version"] == "chisel_normalized_design"
    assert any(
        {"C2Top._T_1", "C2Top.data"} <= set(row["members"])
        for row in normalized["alias_classes"]
    )
    expression = next(
        row
        for row in normalized["expression_groups"]
        if row["target_signal"] == "C2Top._GEN_0"
    )
    assert "C2Top._T_1" in expression["member_signals"]
    assert "C2Top.data" in expression["leaf_inputs"]
    assert "C2Top.active" in expression["leaf_inputs"]
    assert set(expression["member_signals"]) == {
        "C2Top._GEN_0",
        "C2Top._T_1",
    }

    timer = next(
        row
        for row in normalized["register_transitions"]
        if row["signal"] == "C2Top.timer"
    )
    assert [row["update_kind"] for row in timer["reset_rules"]] == ["reset"]
    assert {"clear", "increment"} <= {
        row["update_kind"] for row in timer["update_rules"]
    }
    assert timer["counter_pattern"] == "bounded_progress_counter"
    increment = next(
        row for row in timer["update_rules"]
        if row["update_kind"] == "increment"
    )
    assert increment["guard_members"] == ["C2Top.active"]
    assert increment["value_members"] == ["C2Top.timer"]
    expanded = get_raw_members(
        normalized, [expression["expression_id"]], max_members=8
    )
    assert expanded["objects"][0]["raw_members"] == [
        "C2Top._GEN_0",
        "C2Top._T_1",
    ]
    assert len(expanded["result_sha256"]) == 64
    queried_timer = get_register_transition(
        normalized, timer["register_id"]
    )
    assert (
        queried_timer["register_transition"]["counter_pattern"]
        == "bounded_progress_counter"
    )


def test_register_rules_survive_prepared_design_cache_round_trip():
    parser = VerilogParser(strict=True)
    parser.parse_files_strict([str(FIXTURE)])
    payload = parser.to_prepared_design({str(FIXTURE): "rtl_0001"})
    restored = VerilogParser.from_prepared_design(
        payload, {"rtl_0001": str(FIXTURE)}, strict=True
    )
    graph = InstanceGraph.from_parser(
        restored, {str(FIXTURE): "rtl_0001"}
    )
    normalized = build_normalized_design(
        graph,
        rtl_set_sha256="0" * 64,
        clock_signal="C2Top.clock",
        features=["instance_graph", "register_transition"],
    )
    timer = next(
        row
        for row in normalized["register_transitions"]
        if row["signal"] == "C2Top.timer"
    )
    assert any(
        row["update_kind"] == "increment"
        for row in timer["update_rules"]
    )


def test_persistent_active_increment_is_one_interval_and_inactive_is_zero_score():
    normalized = _normalized()

    class Waveform:
        def get_signal_value(self, signal, cycle):
            values = {
                "C2Top.reset": "0",
                "C2Top.clear": "0",
                "C2Top.active": "1" if cycle < 4 else "0",
                "C2Top.timer": f"{min(cycle + 1, 4):08b}",
            }
            return values.get(signal)

    intervals, diagnostics = persistent_intervals(
        normalized,
        Waveform(),
        end_cycle=5,
        max_intervals=16,
        max_temporal_samples=64,
    )
    assert diagnostics == []
    active = next(
        row for row in intervals
        if row["signal"] == "C2Top.timer" and row["rule"] == "increment"
    )
    assert (active["start_cycle"], active["end_cycle"]) == (0, 3)
    assert active["sample_count"] == 4
    assert active["value_summary"]["monotonic"] is True
    assert active["dynamic_score"] == 1.0
    assert all(
        row["dynamic_score"] == 0.0
        for row in intervals
        if row["observation"] == "structural_only"
    )


def test_unknown_intervals_keep_register_scoped_stable_ids():
    normalized = json.loads(json.dumps(_normalized()))
    duplicate = dict(normalized["register_transitions"][0])
    duplicate["register_id"] = "vcr_second_register"
    duplicate["signal"] = "C2Top.second_timer"
    normalized["register_transitions"].append(duplicate)

    class MissingWaveform:
        def get_signal_value(self, _signal, _cycle):
            return None

    intervals, diagnostics = persistent_intervals(
        normalized,
        MissingWaveform(),
        end_cycle=2,
        max_intervals=64,
        max_temporal_samples=256,
    )
    semantic_ids = [row["semantic_id"] for row in intervals]
    assert diagnostics == []
    assert len(semantic_ids) == len(set(semantic_ids))
    assert len({row["register_id"] for row in intervals}) > 1


def test_semantic_c2_graph_links_projected_register_to_persistent_interval(
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
    graph = build_causal_graph(
        make_request(
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
    )
    register = next(
        row
        for row in graph["semantic_nodes"]
        if row["type"] == "register_transition"
        and row["signal"] == member
    )
    predicate = next(
        row
        for row in graph["semantic_nodes"]
        if row["type"] == "assertion_predicate"
    )
    assert any(
        row.get("src_semantic_id") == register["semantic_id"]
        and row.get("dst_semantic_id") == predicate["semantic_id"]
        and row.get("relation") == "predicate_member"
        for row in graph["edges"]
    )
    assert any(
        row["type"] == "persistent_interval"
        and row["register_id"] == register["semantic_id"]
        for row in graph["semantic_nodes"]
    )
