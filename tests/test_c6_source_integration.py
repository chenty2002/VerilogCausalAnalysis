from __future__ import annotations

import json
from types import SimpleNamespace

from verilog_causal_analysis import (
    build_causal_graph,
    get_handshake_timeline,
    get_interval_evidence,
    get_pipeline_occupancy,
    get_semantic_overview,
    make_request,
    prepare_causal_session,
    sha256_file,
)
from verilog_causal_analysis.identity import stable_set_sha256
from verilog_causal_analysis.engine import (
    _causal_provenance_hints,
    _classify_internal_waveform_frontiers,
)
from verilog_causal_analysis.instance_graph import InstanceGraph
from verilog_causal_analysis.provenance import (
    build_provenance_hints,
    load_source_annotations,
)
from verilog_causal_analysis.verilog_parser import VerilogParser


BOUNDS = {
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
}


def test_provenance_budget_keeps_only_active_slice_and_selected_transition():
    parser = SimpleNamespace(
        _statement_evidence={
            "active": SimpleNamespace(line_start=10, line_end=10),
            "transition": SimpleNamespace(line_start=20, line_end=20),
            "unrelated": SimpleNamespace(line_start=30, line_end=30),
        }
    )
    hints = [
        {"rtl_statement_id": name, "rtl_artifact_id": "rtl_0001"}
        for name in ("active", "transition", "unrelated")
    ]
    assert [
        row["rtl_statement_id"]
        for row in _causal_provenance_hints(
            hints,
            parser,
            [{"rtl_evidence": {"artifact_id": "rtl_0001", "line_start": 10, "line_end": 10}}],
            {"transition"},
        )
    ] == ["active", "transition"]


def test_internal_waveform_frontiers_do_not_hide_hard_incomplete_diagnostics():
    rows = _classify_internal_waveform_frontiers(
        [
            {
                "code": "waveform_exact_instance_missing",
                "message": "missing",
                "severity": "error",
                "breaks_complete": True,
            },
            {
                "code": "waveform_value_unknown",
                "message": "unknown",
                "severity": "error",
                "breaks_complete": True,
            },
            {
                "code": "graph_max_work_reached",
                "message": "bound",
                "severity": "error",
                "breaks_complete": True,
            },
        ]
    )
    assert all(
        row["frontier"] and not row["breaks_complete"]
        for row in rows[:2]
    )
    assert rows[2]["breaks_complete"] is True
    assert "frontier" not in rows[2]


def _request(counter_request, endpoint_signal=None):
    rtl = counter_request.rtl_files[0]
    return make_request(
        trace={
            "artifact_id": "trace_0001",
            **counter_request.trace.to_dict(),
        },
        rtl_files=[rtl.to_dict()],
        semantic_profile={
            "name": "chisel",
            "version": "chisel-semantic-profile",
            "features": ["instance_graph", "source_provenance"],
        },
        clock={"signal": counter_request.clock_signal, "edge": "rising"},
        endpoint={
            "signal": endpoint_signal or counter_request.endpoint_signal,
            "cycle": counter_request.endpoint_cycle,
            "projection": None,
        },
        semantic_inputs=[],
        search_policy=counter_request.search_policy.to_dict(),
        bounds=BOUNDS,
        random_seed=0,
        strict=True,
    )


def test_locator_and_hash_bound_annotation_remain_non_authoritative(tmp_path):
    rtl = tmp_path / "Top.sv"
    rtl.write_text(
        "module Top(input logic a, output logic y);\n"
        "  // @[src/main/scala/Foo.scala 7:3]\n"
        "  assign y = a;\n"
        "endmodule\n"
    )
    parser = VerilogParser(strict=True)
    parser.parse_files_strict([str(rtl)])
    graph = InstanceGraph.from_parser(
        parser, {str(rtl.resolve()): "rtl_0001"}
    )
    hints, diagnostics = build_provenance_hints(
        graph, rtl_set_sha256="6" * 64
    )
    assert diagnostics == []
    comment = next(
        row for row in hints if row["inference_rule"] == "firrtl_locator_comment"
    )
    assert comment["reported_path"] == "src/main/scala/Foo.scala"
    assert comment["status"] == "unverified_hint"
    assert comment["authority"] == "non_authoritative"

    annotation_path = tmp_path / "annotation.json"
    annotation_path.write_text(
        json.dumps(
            {
                "schema_version": "chisel_source_annotations",
                "rtl_set_sha256": "6" * 64,
                "mappings": [
                    {
                        "statement_id": comment["rtl_statement_id"],
                        "reported_path": "src/main/scala/Foo.scala",
                        "reported_locator": "7:3",
                    }
                ],
            },
            sort_keys=True,
        )
    )
    digest, size = sha256_file(annotation_path)
    loaded = load_source_annotations(
        str(annotation_path),
        sha256=digest,
        bytes=size,
        rtl_set_sha256="6" * 64,
        known_statement_ids=set(parser._statement_evidence),
    )
    annotated, diagnostics = build_provenance_hints(
        graph, rtl_set_sha256="6" * 64, annotations=loaded
    )
    assert diagnostics == []
    sidecar = next(
        row
        for row in annotated
        if row["inference_rule"] == "hash_bound_annotation_sidecar"
    )
    assert sidecar["status"] == "source_projection_candidate"
    assert sidecar["authority"] == "non_authoritative"
    assert sidecar["annotation_sha256"] == digest


def test_current_circt_locator_comment_is_collected(tmp_path):
    rtl = tmp_path / "Top.sv"
    rtl.write_text(
        "module Top(input logic a, output logic y);\n"
        "  assign y = a; // src/main/scala/Foo.scala:7:3, :9:5\n"
        "endmodule\n"
    )
    parser = VerilogParser(strict=True)
    parser.parse_files_strict([str(rtl)])
    graph = InstanceGraph.from_parser(parser, {str(rtl.resolve()): "rtl_0001"})
    hints, diagnostics = build_provenance_hints(graph, rtl_set_sha256="7" * 64)
    assert diagnostics == []
    assert [(row["reported_path"], row["reported_locator"]) for row in hints] == [
        ("src/main/scala/Foo.scala", "9:5")
    ]


def test_c6_is_explicit_canonical_and_prepared_session_reuses_inputs(
    counter_request,
):
    request = _request(counter_request)
    first = build_causal_graph(request)
    second = build_causal_graph(request)
    assert first == second
    assert first["identity"]["analyzer_revision"].endswith("+c6")
    with prepare_causal_session(request) as session:
        assert session.build(request) == first
        alternate = _request(counter_request, "Counter.bit1.value")
        alternate_graph = session.build(alternate)
        assert (
            alternate_graph["identity"]["request_sha256"]
            == alternate.request_sha256
        )
        assert alternate_graph["endpoint"]["signal"] == "Counter.bit1.value"
        assert session.normalized_design["features"] == [
            "instance_graph",
            "source_provenance",
        ]


def test_c6_semantic_queries_are_id_only_bounded_and_canonical():
    graph = {
        "graph_id": "vcsg_query",
        "status": "complete",
        "diagnostics": [],
        "root_candidates": [{"candidate_id": "root", "semantic_id": "pipe"}],
        "semantic_nodes": [
            {"semantic_id": "hs", "type": "handshake"},
            {
                "semantic_id": "stall",
                "type": "stall_interval",
                "handshake_id": "hs",
                "start_cycle": 2,
                "end_cycle": 5,
            },
            {"semantic_id": "pipe", "type": "pipeline"},
            {
                "semantic_id": "occ",
                "type": "pipeline_occupancy",
                "pipeline_id": "pipe",
                "start_cycle": 3,
                "end_cycle": 4,
            },
        ],
    }
    assert get_semantic_overview(graph, top_k=1) == get_semantic_overview(
        graph, top_k=1
    )
    assert get_interval_evidence(graph, ["stall"])["intervals"][0][
        "semantic_id"
    ] == "stall"
    assert get_handshake_timeline(graph, "hs", max_events=2)["events"][0][
        "semantic_id"
    ] == "stall"
    assert get_pipeline_occupancy(
        graph, "pipe", start_cycle=2, end_cycle=5
    )["occupancy"][0]["semantic_id"] == "occ"
