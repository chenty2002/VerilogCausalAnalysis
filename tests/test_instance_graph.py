from pathlib import Path
from types import SimpleNamespace

import pytest

from verilog_causal_analysis.causal_slicer import BackwardSlicer
from verilog_causal_analysis.instance_graph import (
    InstanceGraph,
    InstanceGraphError,
)
from verilog_causal_analysis.verilog_parser import VerilogParser


ROOT = Path(__file__).resolve().parent
FIXTURE = (ROOT / "c1" / "duplicate_instances.sv").resolve()


def _graph() -> InstanceGraph:
    parser = VerilogParser(strict=True)
    parser.parse_files_strict([str(FIXTURE)])
    return InstanceGraph.from_parser(
        parser, {str(FIXTURE): "rtl_0001"}
    )


def test_duplicate_module_instances_have_distinct_exact_identities():
    graph = _graph()
    by_path = {row.instance_path: row for row in graph.instances}
    assert set(by_path) == {"Top", "Top.left", "Top.right"}
    assert by_path["Top.left"].module_name == "Leaf"
    assert by_path["Top.right"].module_name == "Leaf"
    assert by_path["Top.left"].instance_id != by_path["Top.right"].instance_id

    left = graph.resolve_signal("Top.left._T_1")
    right = graph.resolve_signal("Top.right._T_1")
    assert left.exact and right.exact
    assert left.instance_id != right.instance_id
    assert graph.resolve_signal("Other.left._T_1").status == "unresolved"


def test_instance_local_dependencies_and_port_directions_do_not_cross():
    graph = _graph()
    left_internal = graph.get_dependencies_for_signal("Top.left._T_1")
    right_internal = graph.get_dependencies_for_signal("Top.right._T_1")
    assert {row.source for row in left_internal} == {"Top.left.in"}
    assert {row.source for row in right_internal} == {"Top.right.in"}

    left_input = graph.get_dependencies_for_signal("Top.left.in")
    right_input = graph.get_dependencies_for_signal("Top.right.in")
    assert {row.source for row in left_input} == {"Top.a"}
    assert {row.source for row in right_input} == {"Top.b"}

    y0 = graph.get_dependencies_for_signal("Top.y0")
    y1 = graph.get_dependencies_for_signal("Top.y1")
    assert {row.source for row in y0} == {"Top.left.out"}
    assert {row.source for row in y1} == {"Top.right.out"}
    assert all(row.identity_strength == "exact" for row in (
        graph.lookup_dependencies("Top.left._T_1"),
        graph.lookup_dependencies("Top.right._T_1"),
    ))


def test_same_named_ports_cross_exact_instance_boundaries(tmp_path):
    rtl = tmp_path / "same_ports.sv"
    rtl.write_text(
        "module Leaf(input logic in, output logic out);\n"
        "  assign out = in;\n"
        "endmodule\n"
        "module Top(input logic in, output logic out);\n"
        "  Leaf leaf(.in(in), .out(out));\n"
        "endmodule\n"
    )
    parser = VerilogParser(strict=True)
    parser.parse_files_strict([str(rtl)])
    graph = InstanceGraph.from_parser(
        parser, {str(rtl.resolve()): "rtl_0001"}
    )

    class Waveform:
        exact_clock = True

        @staticmethod
        def resolve_signal(signal, hierarchy="", *, prefer_hierarchy=True):
            return SimpleNamespace(
                resolved_signal=signal,
                candidates=(signal,),
                identity_strength="exact",
                ambiguous=False,
            )

        @staticmethod
        def get_signal_value(signal, cycle):
            return "1"

    slicer = BackwardSlicer(
        parser,
        Waveform(),
        max_depth=8,
        max_nodes=20,
        dependency_provider=graph,
        search_policy="d2_backward_v1",
    )
    nodes, _edges = slicer.slice_from_endpoint("Top.out", 0)
    assert {row.signal for row in nodes.values()} == {
        "Top.out",
        "Top.leaf.out",
        "Top.leaf.in",
        "Top.in",
    }


def test_ambiguous_top_fails_closed(tmp_path):
    rtl = tmp_path / "two_tops.sv"
    rtl.write_text("module A(input x); endmodule\nmodule B(input y); endmodule\n")
    parser = VerilogParser(strict=True)
    parser.parse_files_strict([str(rtl)])
    with pytest.raises(InstanceGraphError, match="exactly one top"):
        InstanceGraph.from_parser(parser, {str(rtl): "rtl_0001"})


def test_unique_waveform_scope_signature_preserves_unlisted_instance_path():
    graph = _graph()

    class Signals:
        by_name = {
            "Top.elaborationAlias.in": object(),
            "Top.elaborationAlias.out": object(),
            "Top.elaborationAlias._T_1": object(),
        }

    class Waveform:
        signals = Signals()

    graph.bind_waveform(Waveform())
    resolution = graph.resolve_signal(
        "Top.elaborationAlias._T_1"
    )
    assert resolution.exact
    assert resolution.instance_path == "Top.elaborationAlias"
    assert resolution.module_name == "Leaf"
    assert {
        row.source
        for row in graph.get_dependencies_for_signal(
            "Top.elaborationAlias._T_1"
        )
    } == {"Top.elaborationAlias.in"}
