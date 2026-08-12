from pathlib import Path
from types import SimpleNamespace

import pytest

from verilog_causal_analysis.cycle_waveform import CycleAlignedWaveform
from verilog_causal_analysis.verilog_parser import VerilogParser


ROOT = Path(__file__).resolve().parents[1]


def test_strict_parser_returns_dependencies_for_valid_rtl():
    parser = VerilogParser(strict=True)
    modules = parser.parse_file(str(ROOT / "tests" / "counter" / "TestTop.sv"))
    assert modules
    assert parser.all_dependencies
    assert not any(row["code"] == "rtl_parse_failed" for row in parser.diagnostics)


def test_parse_failure_is_machine_readable_and_fail_closed(tmp_path):
    invalid = tmp_path / "invalid.sv"
    invalid.write_text("this is not a module")
    parser = VerilogParser(strict=True)
    with pytest.raises(RuntimeError):
        parser.parse_file(str(invalid))
    assert parser.diagnostics
    assert parser.diagnostics[-1]["code"] == "rtl_parse_failed"
    assert parser.diagnostics[-1]["breaks_complete"] is True


def test_clock_exact_mode_rejects_partial_match():
    fst = next((ROOT / "tests" / "counter").glob("*.fst"))
    with pytest.raises(ValueError, match="Clock signal not found"):
        CycleAlignedWaveform(str(fst), "clock", exact_clock=True)


def test_unique_full_signal_wins_over_conflicting_parent_hierarchy():
    waveform = object.__new__(CycleAlignedWaveform)
    waveform.fst = None
    waveform.signals = SimpleNamespace(by_name={})
    waveform._signal_names = ("Top.unit.data [1:0]",)
    waveform._normalized_names = {"top.unit.data": ("Top.unit.data [1:0]",)}
    waveform._resolution_cache = {}
    waveform._resolution_hits = 0
    waveform._resolution_misses = 0
    resolution = waveform.resolve_signal("Top.unit.data", "Top.other")
    assert resolution.resolved_signal == "Top.unit.data [1:0]"
    assert resolution.ambiguous is False


def test_dependency_lookup_is_module_scoped_and_ignores_constants(tmp_path):
    rtl = tmp_path / "scoped_constants.sv"
    rtl.write_text(
        "module A #(parameter P = 1)(input logic a, output logic y);\n"
        "  localparam K = 1;\n"
        "  assign y = a & P & K;\n"
        "endmodule\n"
        "module B(input logic b, output logic y);\n"
        "  assign y = b;\n"
        "endmodule\n"
    )
    parser = VerilogParser(strict=True)
    parser.parse_files_strict([str(rtl)])

    a_dependencies = parser.lookup_dependencies("y", "A").dependencies
    b_dependencies = parser.lookup_dependencies("y", "B").dependencies
    assert {row.module_name for row in a_dependencies} == {"A"}
    assert {row.source for row in a_dependencies} == {"a"}
    assert {row.module_name for row in b_dependencies} == {"B"}
    assert {row.source for row in b_dependencies} == {"b"}
    assert {"P", "K"}.isdisjoint(parser.modules["A"].signals)
