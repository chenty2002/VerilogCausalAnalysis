from pathlib import Path

from verilog_causal_analysis import get_raw_members
from verilog_causal_analysis.chisel_protocol_semantics import (
    project_c3_waveform_scope,
    stall_intervals,
)
from verilog_causal_analysis.chisel_semantics import build_normalized_design
from verilog_causal_analysis.instance_graph import InstanceGraph
from verilog_causal_analysis.verilog_parser import VerilogParser


ROOT = Path(__file__).resolve().parent
FIXTURE = (ROOT / "c3" / "pipeline_semantics.sv").resolve()
FEATURES = [
    "instance_graph",
    "compiler_net_normalization",
    "register_transition",
    "aggregate",
    "handshake",
    "pipeline",
]


def _normalized():
    parser = VerilogParser(strict=True)
    parser.parse_files_strict([str(FIXTURE)])
    graph = InstanceGraph.from_parser(
        parser, {str(FIXTURE): "rtl_0001"}
    )
    return build_normalized_design(
        graph,
        rtl_set_sha256="3" * 64,
        clock_signal="C3Top.clock",
        features=FEATURES,
    )


def test_ready_valid_aggregates_are_exact_and_never_cross_instances():
    normalized = _normalized()
    handshakes = normalized["handshakes"]
    left = next(
        row
        for row in handshakes
        if row["instance_path"] == "C3Top.left"
        and row["valid_signal"] == "C3Top.left.io_in_valid"
    )
    right = next(
        row
        for row in handshakes
        if row["instance_path"] == "C3Top.right"
        and row["valid_signal"] == "C3Top.right.io_in_valid"
    )
    assert left["ready_signal"] == "C3Top.left.io_in_ready"
    assert right["ready_signal"] == "C3Top.right.io_in_ready"
    assert left["handshake_id"] != right["handshake_id"]
    assert all(
        member.startswith(row["instance_path"] + ".")
        for row in (left, right)
        for member in row["member_signals"]
    )
    assert {
        row["base_name"]
        for row in normalized["aggregates"]
        if row["kind"] == "vec"
    } == {"vec"}


def test_pipeline_requires_sequential_transfer_and_summarizes_blocker():
    normalized = _normalized()
    left_pipeline = next(
        row
        for row in normalized["pipelines"]
        if row["instance_path"] == "C3Top.left"
        and row["base_name"] == "task"
    )
    assert [(row["from_stage"], row["to_stage"]) for row in left_pipeline["transfers"]] == [
        (1, 2)
    ]
    assert left_pipeline["identity_strength"] == "exact_sequential_dependency"

    blocker = next(
        row
        for row in normalized["blocking_relations"]
        if row["target_signal"] == "C3Top.left.blockB_s1"
    )
    assert blocker["blocked_resource"] == "rtl_signal:C3Top.left.blockB_s1"
    assert {
        row["stage_index"] for row in blocker["blockers"]
    } == {1, 2}
    blocker_members = {
        member
        for row in blocker["blockers"]
        for member in row["member_signals"]
    }
    assert {
        "C3Top.left.task_s1_valid",
        "C3Top.left.task_s1_bits_set",
        "C3Top.left.task_s2_valid",
        "C3Top.left.task_s2_bits_set",
    } <= blocker_members
    assert not any(".right." in item for item in blocker["member_signals"])

    expanded = get_raw_members(
        normalized, [blocker["blocking_id"]], max_members=16
    )
    assert "C3Top.left.blockB_s1" in expanded["objects"][0]["raw_members"]


def test_stall_payload_xz_degrades_identity_without_fabricating_transaction():
    normalized = _normalized()
    left = next(
        row
        for row in normalized["handshakes"]
        if row["instance_path"] == "C3Top.left"
    )

    class Waveform:
        def get_signal_value(self, signal, cycle):
            values = {
                "C3Top.left.io_in_valid": ["1", "1", "1", "1", "0"],
                "C3Top.left.io_in_ready": ["1", "0", "0", "0", "1"],
                "C3Top.left.io_in_bits_set": [
                    "0011",
                    "0011",
                    "0011",
                    "0011",
                    "0011",
                ],
                "C3Top.left.io_in_bits_tag": [
                    "00001111",
                    "00001111",
                    "xxxxxxxx",
                    "00001111",
                    "00001111",
                ],
            }
            return values.get(signal, [None] * 5)[cycle]

    intervals, diagnostics = stall_intervals(
        [left],
        Waveform(),
        end_cycle=4,
        max_intervals=8,
        max_temporal_samples=64,
    )
    assert diagnostics == []
    assert len(intervals) == 1
    stall = intervals[0]
    assert (stall["start_cycle"], stall["end_cycle"]) == (1, 3)
    assert stall["last_accept_cycle"] == 0
    assert stall["payload_identity"]["strength"] == "partial"
    assert stall["payload_identity"]["stable_members"] == {
        "C3Top.left.io_in_bits_set": "0011"
    }
    assert stall["payload_identity"]["unknown_members"] == [
        "C3Top.left.io_in_bits_tag"
    ]
    assert stall["payload_identity"]["unstable_members"] == []


def test_exact_waveform_only_scope_projects_pipeline_without_basename_join():
    parser = VerilogParser(strict=True)
    parser.parse_files_strict([str(FIXTURE)])
    graph = InstanceGraph.from_parser(
        parser, {str(FIXTURE): "rtl_0001"}
    )

    class Signals:
        by_name = {
            f"C3Top.elaborationAlias.{local}": object()
            for local in parser.modules["Pipe"].signals
        }

    class Waveform:
        signals = Signals()

    graph.bind_waveform(Waveform())
    scopes = graph.exact_waveform_scopes_for_modules({"Pipe"})
    assert [row[0] for row in scopes] == ["C3Top.elaborationAlias"]
    assert scopes[0][1].startswith("C3Top.elaborationAlias.")
    normalized = build_normalized_design(
        graph,
        rtl_set_sha256="3" * 64,
        clock_signal="C3Top.clock",
        features=FEATURES,
    )
    project_c3_waveform_scope(
        normalized,
        graph,
        "C3Top.elaborationAlias.blockB_s1",
        rtl_set_sha256="3" * 64,
    )
    relation = next(
        row
        for row in normalized["blocking_relations"]
        if row["target_signal"]
        == "C3Top.elaborationAlias.blockB_s1"
    )
    assert relation["instance_path"] == "C3Top.elaborationAlias"
    assert (
        relation["blocked_resource"]
        == "rtl_signal:C3Top.elaborationAlias.blockB_s1"
    )
    assert all(
        member.startswith("C3Top.elaborationAlias.")
        for member in relation["member_signals"]
    )
    assert all(
        row["instance_path"] == "C3Top.elaborationAlias"
        for row in normalized["pipelines"]
        if row["pipeline_id"]
        in {blocker["pipeline_id"] for blocker in relation["blockers"]}
    )
