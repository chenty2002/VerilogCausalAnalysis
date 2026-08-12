import json
from types import SimpleNamespace

from verilog_causal_analysis import (
    CausalAnalysisRequest,
    build_structural_graph,
    build_causal_graph,
    make_structural_request,
    make_request,
    sha256_file,
)
from verilog_causal_analysis.causal_slicer import BackwardSlicer
from verilog_causal_analysis.identity import stable_set_sha256
from verilog_causal_analysis.instance_graph import InstanceGraph
from verilog_causal_analysis.verilog_parser import VerilogParser


BOUNDS = {
    "max_signal_depth": 12,
    "max_signal_nodes": 120,
    "max_expanded_nodes": 120,
    "max_candidate_evaluations": 960,
    "max_intervention_evaluations": 3840,
    "max_semantic_nodes": 16,
    "max_edges": 240,
    "max_seed_count": 8,
    "max_intervals_per_signal": 64,
    "max_temporal_samples": 64000,
    "max_waitfor_nodes": 120,
    "max_waitfor_edges": 240,
    "max_scc_candidates": 8,
}


def test_missing_endpoint_rtl_context_is_not_complete(counter_request):
    request = make_structural_request(
        trace=counter_request.trace.to_dict(),
        rtl_files=[item.to_dict() for item in counter_request.rtl_files],
        clock_signal=counter_request.clock_signal,
        endpoint_signal="Counter.:jasper_formal_clock",
        endpoint_cycle=1,
        search_policy=counter_request.search_policy.to_dict(),
        max_depth=4,
        max_nodes=8,
    )
    graph = build_structural_graph(request)
    assert graph["status"] == "incomplete"
    assert len(graph["nodes"]) == 1
    assert graph["nodes"][0]["rtl_context_status"] == "missing"
    assert {
        row["code"] for row in graph["diagnostics"]
    } >= {"endpoint_rtl_context_missing"}


def test_semantic_projection_is_request_bound_and_enters_exact_predicate_member(
    counter_request, tmp_path
):
    rtl = counter_request.rtl_files[0]
    rtl_set_sha256 = stable_set_sha256(
        [
            {
                "artifact_id": rtl.artifact_id,
                "sha256": rtl.sha256,
                "bytes": rtl.bytes,
            }
        ]
    )
    projection_path = tmp_path / "assertion_projection.json"
    projection_path.write_text(
        json.dumps(
            {
                "schema_version": "assertion_endpoint_projection",
                "endpoint_signal": counter_request.endpoint_signal,
                "endpoint_cycle": counter_request.endpoint_cycle,
                "clock_signal": counter_request.clock_signal,
                "predicate_members": ["Counter.bit1.value"],
                "rtl_set_sha256": rtl_set_sha256,
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
            "features": ["endpoint_projection", "instance_graph"],
        },
        clock={"signal": counter_request.clock_signal, "edge": "rising"},
        endpoint={
            "signal": counter_request.endpoint_signal,
            "cycle": counter_request.endpoint_cycle,
            "projection": {
                "mode": "controller_supplied_exact",
                "predicate_members": ["Counter.bit1.value"],
                "evidence_ref": "assertion_projection_0001",
            },
        },
        semantic_inputs=[
            {
                "artifact_id": "assertion_projection_0001",
                "kind": "assertion_endpoint_projection",
                "path": str(projection_path),
                "sha256": projection_sha256,
                "bytes": projection_bytes,
            }
        ],
        search_policy=counter_request.search_policy.to_dict(),
        bounds=BOUNDS,
        random_seed=0,
        strict=True,
    )
    assert (
        CausalAnalysisRequest.from_dict(request.to_dict()).request_sha256
        == request.request_sha256
    )
    graph = build_causal_graph(request)
    assert graph["status"] == "complete"
    assert graph["endpoint"]["projection_id"].startswith("vcp_")
    assert graph["semantic_nodes"] == [
        {
            "semantic_id": graph["semantic_nodes"][0]["semantic_id"],
            "type": "assertion_predicate",
            "endpoint_signal": counter_request.endpoint_signal,
            "cycle": counter_request.endpoint_cycle,
            "member_signals": ["Counter.bit1.value"],
            "evidence_ref": "assertion_projection_0001",
            "inference_rule": "controller_supplied_exact",
        }
    ]
    assert any(
        row["signal"] == "Counter.bit1.value" and row["depth"] == 1
        for row in graph["signal_nodes"]
    )


def test_missing_internal_waveform_value_is_a_nonrecursive_frontier(tmp_path):
    rtl = tmp_path / "Top.sv"
    rtl.write_text(
        "module Leaf(input logic in, output logic out);\n"
        "  assign out = in;\n"
        "endmodule\n"
        "module Top(input logic a, output logic y0);\n"
        "  Leaf left(.in(a), .out(y0));\n"
        "endmodule\n"
    )
    parser = VerilogParser(strict=True)
    parser.parse_files_strict([str(rtl)])
    instance_graph = InstanceGraph.from_parser(
        parser, {str(rtl.resolve()): "rtl_0001"}
    )

    class Waveform:
        exact_clock = True

        @staticmethod
        def resolve_signal(signal, hierarchy="", *, prefer_hierarchy=True):
            if signal == "Top.y0":
                return SimpleNamespace(
                    resolved_signal=signal,
                    candidates=(signal,),
                    identity_strength="exact",
                    ambiguous=False,
                )
            return SimpleNamespace(
                resolved_signal=None,
                candidates=(),
                identity_strength="unresolved",
                ambiguous=False,
            )

        @staticmethod
        def get_signal_value(signal, cycle):
            return "1" if signal == "Top.y0" else None

    slicer = BackwardSlicer(
        parser,
        Waveform(),
        max_depth=8,
        max_nodes=20,
        dependency_provider=instance_graph,
    )
    nodes, _edges = slicer.slice_from_endpoint("Top.y0", 0)
    signals = {row.signal for row in nodes.values()}
    assert "Top.y0" in signals
    assert "Top.left.out" in signals
    assert "Top.left.in" not in signals
    assert slicer.get_statistics()["exact_instance_waveform_misses"] == []
