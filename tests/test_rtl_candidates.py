import subprocess

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


def _request(counter_request, rtl_files, *, trace=None, clock=None, endpoint=None):
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
            "cycle": counter_request.endpoint_cycle,
            "projection": None,
        },
        semantic_inputs=[],
        search_policy=counter_request.search_policy.to_dict(),
        bounds=BOUNDS,
        random_seed=0,
        strict=True,
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

    assert [row["line_start"] for row in candidates] == [3, 4]


def test_native_graph_edges_join_candidates_and_hash_drift_fails(
    counter_request, tmp_path
):
    rtl = tmp_path / "Top.sv"
    rtl.write_text(
        "module Top(input wire clk, input wire a, output wire y);\n"
        "  assign y = a;\n"
        "  wire unused;\n"
        "  assign unused = ~a;\n"
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
    assert all(
        (row["artifact_id"], row["statement_id"]) in universe for row in mapped
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
