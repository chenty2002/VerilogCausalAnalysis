"""Deterministic production engine for the V2 causal graph contract."""

from __future__ import annotations

import os
import re
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

from .causal_slicer import BackwardSlicer
from .contracts import CausalAnalysisRequestV2, GRAPH_SCHEMA
from .cycle_waveform import CycleAlignedWaveform
from .identity import (
    ANALYZER_REVISION,
    HDLCONVERTOR_REVISION,
    canonical_sha256,
    sha256_file,
    stable_id,
    stable_set_sha256,
)
from .verilog_parser import VerilogParser


_ABSOLUTE_PATH_RE = re.compile(
    r"(?<![A-Za-z0-9_.-])/(?:[A-Za-z0-9_.-]+/)+[A-Za-z0-9_.-]+"
)


def _redact_absolute_paths(text: str) -> str:
    return _ABSOLUTE_PATH_RE.sub("<redacted-path>", text)


def _diagnostic(
    code: str,
    message: str,
    *,
    severity: str = "error",
    breaks_complete: bool = True,
    artifact_id: Optional[str] = None,
) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "code": code,
        "severity": severity,
        "breaks_complete": breaks_complete,
        "message": message,
    }
    if artifact_id is not None:
        row["artifact_id"] = artifact_id
    return row


def _graph_identity(request: CausalAnalysisRequestV2) -> Dict[str, Any]:
    rtl_rows = [
        {
            "artifact_id": artifact.artifact_id,
            "sha256": artifact.sha256,
            "bytes": artifact.bytes,
        }
        for artifact in request.rtl_files
    ]
    return {
        "request_sha256": request.request_sha256,
        "trace_sha256": request.trace.sha256,
        "rtl_set_sha256": stable_set_sha256(rtl_rows),
        "analyzer_revision": ANALYZER_REVISION,
        "hdlconvertor_revision": HDLCONVERTOR_REVISION,
        "random_seed": request.random_seed,
    }


def _sort_diagnostics(rows: Iterable[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    unique = {
        canonical_sha256(dict(row)): dict(row)
        for row in rows
    }
    return sorted(
        unique.values(),
        key=lambda row: (
            row["code"],
            row.get("artifact_id") or "",
            row["message"],
        ),
    )


def _empty_graph(
    request: CausalAnalysisRequestV2,
    diagnostics: List[Dict[str, Any]],
    *,
    status: str = "incomplete",
) -> Dict[str, Any]:
    identity = _graph_identity(request)
    return {
        "schema_version": GRAPH_SCHEMA,
        "graph_id": stable_id("vcg_", identity),
        "status": status,
        "identity": identity,
        "bounds": {
            "max_depth": request.max_depth,
            "max_nodes": request.max_nodes,
            "max_depth_reached": False,
            "max_nodes_reached": False,
        },
        "nodes": [],
        "edges": [],
        "diagnostics": _sort_diagnostics(diagnostics),
    }


def _evidence_strength(edge: Mapping[str, Any]) -> Tuple[str, str]:
    examples = edge.get("change_examples") or []
    types = {item.get("type") for item in examples if isinstance(item, dict)}
    evidence = edge.get("evidence") or {}
    if "counterfactual" in types:
        return "expression_counterfactual", "counterfactual_changed_target"
    if "sva_antecedent" in types or evidence.get("type") == "sva_trigger":
        return "branch_observed", "observed_branch_or_trigger"
    if "toggle_correlation" in types:
        return "toggle_supported", "source_and_target_toggled"
    if "structural" in types or not examples:
        return "structural_only", "rtl_dependency_only"
    return "unresolved", "causal_evidence_unresolved"


def _dependency_type(edge: Mapping[str, Any]) -> str:
    reason = str(edge.get("reason", ""))
    if " via " in reason:
        return reason.rsplit(" via ", 1)[1]
    evidence_type = (edge.get("evidence") or {}).get("type")
    return str(evidence_type or edge.get("contribution_type") or "unknown")


def _convert_graph(
    request: CausalAnalysisRequestV2,
    legacy_nodes: Iterable[Mapping[str, Any]],
    legacy_edges: Iterable[Mapping[str, Any]],
    stats: Mapping[str, Any],
    diagnostics: List[Dict[str, Any]],
    artifact_by_path: Mapping[str, str],
) -> Dict[str, Any]:
    identity = _graph_identity(request)
    graph_id = stable_id("vcg_", identity)
    old_nodes = list(legacy_nodes)
    old_edges = list(legacy_edges)
    incoming = {str(edge["dst_node_id"]) for edge in old_edges}
    node_id_map: Dict[str, str] = {}
    nodes: List[Dict[str, Any]] = []

    for node in old_nodes:
        signal = str(node["signal"])
        cycle = int(node["cycle"])
        value = str(node["value"])
        node_id = stable_id("vcn_", identity, signal, cycle, value, length=24)
        node_id_map[str(node["id"])] = node_id
        nodes.append(
            {
                "node_id": node_id,
                "signal_id": stable_id("sig_", signal),
                "signal": signal,
                "cycle": cycle,
                "value": value,
                "depth": int(node.get("depth", 0)),
                "is_endpoint": bool(node.get("is_endpoint")),
                "is_slice_leaf": str(node["id"]) not in incoming
                and not bool(node.get("is_endpoint")),
                "rtl_context_status": (
                    "missing" if node.get("rtl_context_missing") else "exact"
                ),
                "identity_strength": node.get("identity_strength", "unresolved"),
                "suspect_score": round(
                    max(0.0, min(1.0, float(node.get("suspect_score", 0.0)))),
                    6,
                ),
            }
        )
        if any(char in value.lower() for char in ("x", "z")):
            diagnostics.append(
                _diagnostic(
                    "waveform_value_unknown",
                    f"unknown value for signal {signal} at cycle {cycle}",
                )
            )

    node_strength = {row["node_id"]: row["identity_strength"] for row in nodes}
    edges: List[Dict[str, Any]] = []
    for edge in old_edges:
        src_id = node_id_map[str(edge["src_node_id"])]
        dst_id = node_id_map[str(edge["dst_node_id"])]
        evidence = dict(edge.get("evidence") or {})
        source_path = str(evidence.get("file") or "")
        artifact_id = (
            artifact_by_path.get(os.path.abspath(source_path)) if source_path else None
        )
        lines = evidence.get("lines") or [0, 0]
        line_start = int(lines[0] or 0)
        line_end = int(lines[1] or line_start)
        snippet = _redact_absolute_paths(str(evidence.get("code_snippet") or ""))
        rtl_evidence = {
            "artifact_id": artifact_id,
            "line_start": line_start,
            "line_end": line_end,
            "snippet": snippet,
            "snippet_sha256": canonical_sha256(snippet),
            "expression": _redact_absolute_paths(
                str(evidence.get("expression") or "")
            ),
            "condition": _redact_absolute_paths(
                str(evidence.get("condition") or "")
            ),
        }
        if artifact_id is None or line_start <= 0:
            diagnostics.append(
                _diagnostic(
                    "dependency_position_missing",
                    "dependency has no exact required-RTL source position",
                    severity="warning",
                    breaks_complete=False,
                    artifact_id=artifact_id,
                )
            )
        strength, reason_code = _evidence_strength(edge)
        dependency_type = _dependency_type(edge)
        edge_identity = {
            "artifact_id": artifact_id,
            "line_start": line_start,
            "line_end": line_end,
            "snippet_sha256": rtl_evidence["snippet_sha256"],
        }
        edges.append(
            {
                "edge_id": stable_id(
                    "vce_",
                    identity,
                    src_id,
                    dst_id,
                    dependency_type,
                    edge_identity,
                    length=24,
                ),
                "src_node_id": src_id,
                "dst_node_id": dst_id,
                "dependency_type": dependency_type,
                "identity_strength": node_strength.get(src_id, "unresolved"),
                "evidence_strength": strength,
                "contribution_score": round(
                    max(0.0, min(1.0, float(edge.get("contribution_score", 0.0)))),
                    6,
                ),
                "reason_code": reason_code,
                "rtl_evidence": rtl_evidence,
                "change_examples": edge.get("change_examples") or [],
            }
        )

    max_depth_reached = bool(stats.get("max_depth_reached"))
    max_nodes_reached = bool(stats.get("max_nodes_reached"))
    max_work_reached = bool(stats.get("candidate_evaluation_budget_reached"))
    for ambiguity in stats.get("identity_ambiguities") or []:
        diagnostics.append(
            _diagnostic(
                "dependency_identity_ambiguous",
                (
                    f"signal {ambiguity['signal']} at cycle "
                    f"{ambiguity['cycle']} has "
                    f"{ambiguity['candidate_count']} waveform candidates"
                ),
            )
        )
    if max_depth_reached:
        diagnostics.append(
            _diagnostic(
                "graph_max_depth_reached",
                f"causal slice reached max_depth={request.max_depth}",
            )
        )
    if max_nodes_reached:
        diagnostics.append(
            _diagnostic(
                "graph_max_nodes_reached",
                f"causal slice reached max_nodes={request.max_nodes}",
            )
        )
    if max_work_reached:
        diagnostics.append(
            _diagnostic(
                "graph_max_work_reached",
                (
                    "causal slice reached deterministic "
                    f"candidate_evaluation_budget="
                    f"{stats.get('candidate_evaluation_budget')}"
                ),
            )
        )
    return {
        "schema_version": GRAPH_SCHEMA,
        "graph_id": graph_id,
        "status": (
            "incomplete"
            if any(item.get("breaks_complete") for item in diagnostics)
            else "complete"
        ),
        "identity": identity,
        "bounds": {
            "max_depth": request.max_depth,
            "max_nodes": request.max_nodes,
            "max_depth_reached": max_depth_reached,
            "max_nodes_reached": max_nodes_reached,
        },
        "nodes": sorted(
            nodes,
            key=lambda row: (
                row["depth"],
                -row["cycle"],
                row["signal"],
                row["node_id"],
            ),
        ),
        "edges": sorted(
            edges,
            key=lambda row: (
                row["dst_node_id"],
                row["src_node_id"],
                row["dependency_type"],
                row["edge_id"],
            ),
        ),
        "diagnostics": _sort_diagnostics(diagnostics),
    }


def _build_causal_graph_v2(
    request: CausalAnalysisRequestV2 | Mapping[str, Any],
    *,
    production: bool,
) -> Dict[str, Any]:
    if not isinstance(request, CausalAnalysisRequestV2):
        request = CausalAnalysisRequestV2.from_dict(request, production=production)
    if production and not request.strict:
        raise ValueError("production V2 API requires strict=true")
    if not production and request.strict:
        raise ValueError("diagnostic heuristic requests require strict=false")

    diagnostics: List[Dict[str, Any]] = []
    try:
        actual_hash, actual_bytes = sha256_file(request.trace.path)
    except OSError as error:
        diagnostics.append(
            _diagnostic("waveform_hash_mismatch", "trace file is unavailable")
        )
        return _empty_graph(request, diagnostics)
    if (actual_hash, actual_bytes) != (request.trace.sha256, request.trace.bytes):
        diagnostics.append(
            _diagnostic(
                "waveform_hash_mismatch",
                "trace bytes or SHA-256 differ from the request",
            )
        )

    artifact_by_path: Dict[str, str] = {}
    for artifact in request.rtl_files:
        artifact_by_path[os.path.abspath(artifact.path)] = artifact.artifact_id
        try:
            actual_hash, actual_bytes = sha256_file(artifact.path)
        except OSError as error:
            diagnostics.append(
                _diagnostic(
                    "rtl_file_hash_mismatch",
                    "required RTL file is unavailable",
                    artifact_id=artifact.artifact_id,
                )
            )
            continue
        if (actual_hash, actual_bytes) != (artifact.sha256, artifact.bytes):
            diagnostics.append(
                _diagnostic(
                    "rtl_file_hash_mismatch",
                    "required RTL bytes or SHA-256 differ from the request",
                    artifact_id=artifact.artifact_id,
                )
            )
    if diagnostics:
        return _empty_graph(request, diagnostics)

    parser = VerilogParser(strict=True)
    try:
        for artifact in sorted(request.rtl_files, key=lambda item: item.artifact_id):
            parser.parse_file(artifact.path)
    except Exception as error:
        artifact_id = next(
            (
                artifact.artifact_id
                for artifact in request.rtl_files
                if artifact.path in str(error)
            ),
            None,
        )
        diagnostics.append(
            _diagnostic(
                "rtl_parse_failed",
                f"required RTL parse failed ({type(error).__name__})",
                artifact_id=artifact_id,
            )
        )
        return _empty_graph(request, diagnostics)

    waveform: Optional[CycleAlignedWaveform] = None
    try:
        waveform = CycleAlignedWaveform(
            request.trace.path, request.clock_signal, exact_clock=True
        )
    except ValueError as error:
        diagnostics.append(_diagnostic("clock_not_exact", str(error)))
        return _empty_graph(request, diagnostics)
    except Exception as error:
        diagnostics.append(
            _diagnostic(
                "rtl_construct_unsupported",
                f"waveform open failed ({type(error).__name__})",
            )
        )
        return _empty_graph(request, diagnostics, status="unsupported")

    try:
        if not waveform.has_exact_signal(request.endpoint_signal):
            diagnostics.append(
                _diagnostic(
                    "endpoint_not_exact",
                    f"endpoint is not an exact waveform signal: {request.endpoint_signal}",
                )
            )
            return _empty_graph(request, diagnostics)
        if request.endpoint_cycle >= waveform.get_cycle_count():
            diagnostics.append(
                _diagnostic(
                    "endpoint_not_exact",
                    f"endpoint cycle {request.endpoint_cycle} is outside the waveform",
                )
            )
            return _empty_graph(request, diagnostics)
        if (
            waveform.get_signal_value(
                request.endpoint_signal, request.endpoint_cycle
            )
            is None
        ):
            diagnostics.append(
                _diagnostic(
                    "waveform_signal_missing",
                    "endpoint has no sampled waveform value",
                )
            )
            return _empty_graph(request, diagnostics)

        slicer = BackwardSlicer(
            parser,
            waveform,
            max_depth=request.max_depth,
            max_nodes=request.max_nodes,
        )
        nodes, edges = slicer.slice_from_endpoint(
            request.endpoint_signal, request.endpoint_cycle
        )
        if not nodes:
            diagnostics.append(
                _diagnostic(
                    "waveform_signal_missing",
                    "causal slicer produced no endpoint node",
                )
            )
            return _empty_graph(request, diagnostics)
        return _convert_graph(
            request,
            (node.to_dict() for node in nodes.values()),
            (edge.to_dict() for edge in edges),
            slicer.get_statistics(),
            diagnostics,
            artifact_by_path,
        )
    finally:
        waveform.close()


def build_causal_graph_v2(
    request: CausalAnalysisRequestV2 | Mapping[str, Any],
) -> Dict[str, Any]:
    """Build a strict V2 graph without any auto-detection or path guessing."""
    return _build_causal_graph_v2(request, production=True)


def _build_diagnostic_graph_v2(
    request: CausalAnalysisRequestV2 | Mapping[str, Any],
) -> Dict[str, Any]:
    """CLI-only engine entry for explicitly downgraded heuristic inputs."""
    return _build_causal_graph_v2(request, production=False)
