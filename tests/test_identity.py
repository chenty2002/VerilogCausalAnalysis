from verilog_causal_analysis import canonical_json_bytes, canonical_sha256
from verilog_causal_analysis.structural_engine import _redact_absolute_paths
from verilog_causal_analysis.identity import stable_id


def test_canonical_json_and_ids_ignore_mapping_insertion_order():
    left = {"z": [3, 2, 1], "a": {"y": 2, "x": 1}}
    right = {"a": {"x": 1, "y": 2}, "z": [3, 2, 1]}
    assert canonical_json_bytes(left) == canonical_json_bytes(right)
    assert canonical_sha256(left) == canonical_sha256(right)
    assert stable_id("id_", left) == stable_id("id_", right)


def test_absolute_path_redaction_preserves_relative_chisel_locator():
    text = (
        "assign state = stateReg; // src/main/scala/Fsm16.scala:62:13 "
        "from /private/build/SpecFlowOverlay.sv"
    )
    redacted = _redact_absolute_paths(text)
    assert "src/main/scala/Fsm16.scala:62:13" in redacted
    assert "/private/build" not in redacted
    assert "<redacted-path>" in redacted
