import json
import subprocess
from pathlib import Path

import pytest

from verilog_causal_analysis import (
    ContractError,
    build_causal_graph,
    build_rtl_candidates,
    make_request,
    sha256_file,
)
from verilog_causal_analysis.identity import stable_set_sha256

from test_verilog_profile import BOUNDS


def _request(
    counter_request,
    rtl_files,
    *,
    trace=None,
    clock=None,
    endpoint=None,
    cycle=None,
):
    return make_request(
        trace=trace
        or {"artifact_id": "trace_0001", **counter_request.trace.to_dict()},
        rtl_files=rtl_files,
        semantic_profile={
            "name": "verilog",
            "version": "verilog-semantic-profile",
            "features": ["instance_graph"],
        },
        clock={"signal": clock or counter_request.clock_signal, "edge": "rising"},
        endpoint={
            "signal": endpoint or counter_request.endpoint_signal,
            "cycle": counter_request.endpoint_cycle if cycle is None else cycle,
            "projection": None,
        },
        semantic_inputs=[],
        search_policy=counter_request.search_policy.to_dict(),
        bounds=BOUNDS,
        random_seed=0,
        strict=True,
    )


def _pilot_request(counter_request, case, *, cycle=None):
    case_root = (
        Path(__file__).resolve().parents[2]
        / "runs/verilogcause/20260812-native-pilot-v5/cases"
        / case
    )
    rtl = case_root / "model_inputs/sanitized_faulty/design.v"
    trace = rtl.parent / "dump.fst"
    graph = json.loads(
        (case_root / "model_inputs/causal_graph.json").read_text()
    )
    rtl_hash, rtl_bytes = sha256_file(rtl)
    trace_hash, trace_bytes = sha256_file(trace)
    return _request(
        counter_request,
        [{
            "artifact_id": "rtl_0001",
            "path": str(rtl),
            "sha256": rtl_hash,
            "bytes": rtl_bytes,
        }],
        trace={
            "artifact_id": "trace_0001",
            "path": str(trace),
            "format": "fst",
            "sha256": trace_hash,
            "bytes": trace_bytes,
        },
        clock="testbench.clk",
        endpoint=graph["endpoint"]["signal"],
        cycle=cycle,
    )


def test_candidates_are_complete_exact_and_line_shift_stable(
    counter_request, tmp_path
):
    source = (
        "module Leaf(input wire a, output wire y);\n"
        "  assign y = a;\n"
        "endmodule\n"
        "module Top(input wire clk, input wire a, output wire y);\n"
        "  wire unused;\n"
        "  Leaf leaf(.a(a), .y(y));\n"
        "  assign unused = a;\n"
        "  reg q;\n"
        "  always @(posedge clk) q <= a;\n"
        "endmodule\n"
    )
    rtl = tmp_path / "native.sv"
    rtl.write_text(source)
    digest, size = sha256_file(rtl)
    artifact = {
        "artifact_id": "rtl_0001",
        "path": str(rtl.resolve()),
        "sha256": digest,
        "bytes": size,
    }
    first = build_rtl_candidates(_request(counter_request, [artifact]))
    assert {row["statement_kind"] for row in first["candidates"]} == {
        "assignment",
        "port_binding",
        "register_update",
    }
    assert any(row["line_start"] == 7 for row in first["candidates"])
    assert first["rtl_set_sha256"] == stable_set_sha256(
        [{key: artifact[key] for key in ("artifact_id", "sha256", "bytes")}]
    )

    rtl.write_text("\n" + source)
    shifted_digest, shifted_size = sha256_file(rtl)
    shifted = build_rtl_candidates(
        _request(
            counter_request,
            [{**artifact, "sha256": shifted_digest, "bytes": shifted_size}],
        )
    )
    assert {row["statement_id"] for row in first["candidates"]} == {
        row["statement_id"] for row in shifted["candidates"]
    }


def test_blocking_assignments_have_exact_lines(counter_request, tmp_path):
    rtl = tmp_path / "case.v"
    rtl.write_text(
        "module Top(input wire clk, input wire a, output reg y);\n"
        "  always @(*) begin\n"
        "    if (a) y = 1'b1;\n"
        "    else y = 1'b0;\n"
        "  end\n"
        "endmodule\n"
    )
    digest, size = sha256_file(rtl)
    candidates = build_rtl_candidates(
        _request(
            counter_request,
            [{
                "artifact_id": "rtl_0001",
                "path": str(rtl.resolve()),
                "sha256": digest,
                "bytes": size,
            }],
        )
    )["candidates"]

    assignments = [
        row for row in candidates if row["statement_kind"] == "assignment"
    ]
    guards = [
        row for row in candidates
        if row["statement_kind"] == "conditional_guard"
    ]
    assert [row["line_start"] for row in assignments] == [3, 4]
    assert sorted(row["line_start"] for row in guards) == [3, 4]


def test_native_graph_edges_join_candidates_and_hash_drift_fails(
    counter_request, tmp_path
):
    rtl = tmp_path / "Top.sv"
    rtl.write_text(
        "module Top(input wire clk, input wire a, output wire y);\n"
        "  wire mid;\n"
        "  assign mid = a;\n"
        "  assign y = mid;\n"
        "endmodule\n"
    )
    tb = tmp_path / "tb.sv"
    tb.write_text(
        "module tb; reg clk = 0; reg a = 0; wire y;\n"
        "Top dut(.clk(clk), .a(a), .y(y));\n"
        "always #5 clk = ~clk;\n"
        "initial begin $dumpfile(\"trace.vcd\"); $dumpvars(0, tb); "
        "#2 a = 1; #20 $finish; end endmodule\n"
    )
    subprocess.run(
        ["iverilog", "-g2012", "-o", "sim", str(rtl), str(tb)],
        cwd=tmp_path,
        check=True,
        capture_output=True,
    )
    subprocess.run([str(tmp_path / "sim")], cwd=tmp_path, check=True, capture_output=True)
    fst = tmp_path / "trace.fst"
    subprocess.run(
        ["vcd2fst", str(tmp_path / "trace.vcd"), str(fst)],
        check=True,
        capture_output=True,
    )
    rtl_hash, rtl_bytes = sha256_file(rtl)
    fst_hash, fst_bytes = sha256_file(fst)
    request = _request(
        counter_request,
        [{
            "artifact_id": "rtl_0001",
            "path": str(rtl),
            "sha256": rtl_hash,
            "bytes": rtl_bytes,
        }],
        trace={
            "artifact_id": "trace_0001",
            "path": str(fst),
            "format": "fst",
            "sha256": fst_hash,
            "bytes": fst_bytes,
        },
        clock="tb.clk",
        endpoint="tb.dut.y",
    )
    candidates = build_rtl_candidates(request)
    graph = build_causal_graph(request)
    activations = [
        row for row in graph["edges"]
        if row.get("relation") == "active_statement_write"
    ]
    assert activations
    assert all(
        row["activation_status"] == "active_exact"
        and row["dst_node_id"] == row["target_node_id"]
        and row.get("src_semantic_id")
        for row in activations
    )
    activation_ids = {row["src_semantic_id"] for row in activations}
    assert activation_ids <= {
        row["semantic_id"]
        for row in graph["semantic_nodes"]
        if row["type"] == "rtl_statement_activation"
    }
    universe = {
        (row["artifact_id"], row["statement_id"])
        for row in candidates["candidates"]
    }
    mapped = [
        row["rtl_evidence"]
        for row in graph["edges"]
        if row.get("rtl_evidence", {}).get("statement_id")
    ]
    assert mapped
    candidates_by_id = {
        (row["artifact_id"], row["statement_id"]): row
        for row in candidates["candidates"]
    }
    assert all(
        (row["artifact_id"], row["statement_id"]) in universe for row in mapped
    )
    assert all(
        row["line_start"]
        == candidates_by_id[(row["artifact_id"], row["statement_id"])][
            "line_start"
        ]
        for row in mapped
    )
    assert graph["identity"]["rtl_set_sha256"] == candidates["rtl_set_sha256"]

    bad = request.to_dict()
    bad["rtl_files"][0]["sha256"] = "f" * 64
    with pytest.raises(ContractError):
        build_rtl_candidates(
            make_request(
                **{
                    key: value
                    for key, value in bad.items()
                    if key not in {"schema_version", "request_id"}
                }
            )
        )


def test_wit_hw_guard_candidates_and_exact_active_writes(counter_request):
    expected_candidates = {
        "alu_2": {("assignment", 22)},
        "counter-3": {("conditional_guard", 38)},
        "fsm_16-3": {("conditional_guard", 88)},
        "fsm_16-4": {
            ("conditional_guard", 45),
            ("conditional_guard", 52),
        },
    }
    for case, expected in expected_candidates.items():
        universe = build_rtl_candidates(
            _pilot_request(counter_request, case)
        )
        assert universe["schema_version"] == "rtl_candidate_universe.v2"
        actual = {
            (row["statement_kind"], row["line_start"])
            for row in universe["candidates"]
        }
        assert expected <= actual

    for case, cycle, expected_line, rejected_line in (
        ("alu_3", 0, 26, 28),
        ("alu_5", 1, 28, 26),
        ("alu_6", 0, 34, 32),
        ("fsm_16-1", 4, 79, 34),
        ("fsm_16-2", 4, 101, 34),
    ):
        request = _pilot_request(counter_request, case, cycle=cycle)
        graph = build_causal_graph(request)
        endpoint_id = next(
            row["node_id"]
            for row in graph["signal_nodes"]
            if row["signal"] == request.endpoint.signal
            and row["cycle"] == cycle
        )
        candidates = {
            row["statement_id"]: row
            for row in build_rtl_candidates(request)["candidates"]
        }
        active_lines = {
            candidates[row["statement_id"]]["line_start"]
            for row in graph["edges"]
            if row.get("relation") == "active_statement_write"
            and row["target_node_id"] == endpoint_id
        }
        assert expected_line in active_lines
        assert rejected_line not in active_lines
