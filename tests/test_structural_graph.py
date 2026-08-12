from dataclasses import replace
from pathlib import Path

from verilog_causal_analysis import build_structural_graph
from verilog_causal_analysis.engine import _convert_graph

from conftest import request_for


def diagnostic_codes(graph):
    return {row["code"] for row in graph["diagnostics"]}


def test_counter_fixture_builds_complete_typed_graph(counter_request):
    graph = build_structural_graph(counter_request)
    assert graph["schema_version"] == "verilog_causal_graph"
    assert graph["status"] == "complete"
    assert graph["nodes"] and graph["edges"]
    assert all(node["node_id"].startswith("vcn_") for node in graph["nodes"])
    assert all(edge["edge_id"].startswith("vce_") for edge in graph["edges"])
    assert all("path" not in edge["rtl_evidence"] for edge in graph["edges"])
    assert {node["identity_strength"] for node in graph["nodes"]} <= {
        "exact",
        "hierarchy_inferred",
        "unresolved",
    }


def test_hash_drift_fails_closed_before_parser(counter_request):
    request = replace(
        counter_request,
        trace=replace(counter_request.trace, sha256="0" * 64),
    )
    graph = build_structural_graph(request)
    assert graph["status"] == "incomplete"
    assert graph["nodes"] == []
    assert "waveform_hash_mismatch" in diagnostic_codes(graph)


def test_missing_or_partial_endpoint_is_not_exact_authority(counter_request):
    request = replace(counter_request, endpoint_signal="value_should_toggle")
    graph = build_structural_graph(request)
    assert graph["status"] == "incomplete"
    assert "endpoint_not_exact" in diagnostic_codes(graph)


def test_bounds_truncation_is_explicit(counter_request):
    graph = _convert_graph(
        counter_request,
        [
            {
                "id": "endpoint",
                "signal": counter_request.endpoint_signal,
                "cycle": counter_request.endpoint_cycle,
                "value": "0",
                "depth": 0,
                "is_endpoint": True,
                "rtl_context_missing": False,
                "identity_strength": "exact",
                "suspect_score": 1.0,
            }
        ],
        [],
        {"max_nodes_reached": True},
        [],
        {},
    )
    assert graph["status"] == "incomplete"
    assert graph["bounds"]["max_nodes_reached"] is True
    assert "graph_max_nodes_reached" in diagnostic_codes(graph)


def test_unknown_waveform_value_breaks_complete(counter_request):
    graph = _convert_graph(
        counter_request,
        [
            {
                "id": "input_node",
                "signal": counter_request.endpoint_signal,
                "cycle": 1,
                "value": "x",
                "depth": 0,
                "is_endpoint": True,
                "rtl_context_missing": False,
                "identity_strength": "exact",
                "suspect_score": 1.0,
            }
        ],
        [],
        {},
        [],
        {},
    )
    assert graph["status"] == "incomplete"
    assert "waveform_value_unknown" in diagnostic_codes(graph)


def test_fsm_and_sva_window_fixture_is_bounded_and_ordered():
    request = request_for(
        "philo4",
        "philo4.Philosopher_0_should_eventually_eat_when_hungry.fst",
        clock="philo4.clock",
        cycle=448,
    )
    graph = build_structural_graph(request)
    assert graph["status"] in {"complete", "incomplete"}
    assert len(graph["nodes"]) <= request.max_nodes
    assert graph["nodes"] == sorted(
        graph["nodes"],
        key=lambda row: (
            row["depth"],
            -row["cycle"],
            row["signal"],
            row["node_id"],
        ),
    )
    assert all(0 <= node["cycle"] <= request.endpoint_cycle for node in graph["nodes"])
