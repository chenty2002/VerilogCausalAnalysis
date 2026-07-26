"""Deterministic production engine for the V2 causal graph contract."""

from __future__ import annotations

import json
import os
import re
import tempfile
from collections import OrderedDict
from pathlib import Path
from threading import RLock
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

from .causal_slicer import BackwardSlicer
from .contracts import CausalAnalysisRequestV2, GRAPH_SCHEMA
from .cycle_waveform import CycleAlignedWaveform
from .identity import (
    ANALYZER_REVISION,
    HDLCONVERTOR_REVISION,
    canonical_sha256,
    contains_absolute_path,
    sha256_file,
    stable_id,
    stable_set_sha256,
)
from .verilog_parser import VerilogParser


_ABSOLUTE_PATH_RE = re.compile(
    r"(?<![A-Za-z0-9_.-])/(?:[A-Za-z0-9_.-]+/)+[A-Za-z0-9_.-]+"
)
_PARSED_DESIGN_CACHE_MAX = 8
_PARSED_DESIGN_CACHE: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()
_PARSED_DESIGN_CACHE_LOCK = RLock()


def _redact_absolute_paths(text: str) -> str:
    return _ABSOLUTE_PATH_RE.sub("<redacted-path>", text)


def _contains_absolute_path_fragment(value: Any) -> bool:
    if isinstance(value, str):
        return bool(_ABSOLUTE_PATH_RE.search(value))
    if isinstance(value, Mapping):
        return any(
            _contains_absolute_path_fragment(item)
            for item in value.values()
        )
    if isinstance(value, (list, tuple)):
        return any(
            _contains_absolute_path_fragment(item) for item in value
        )
    return False


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
    temporal_work_reached = bool(
        stats.get("temporal_work_budget_reached")
    )
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
    if stats.get("sva_exact_trigger_missing"):
        diagnostics.append(
            _diagnostic(
                "sva_exact_trigger_missing",
                (
                    "SVA implication requires hash-bound exact trigger "
                    "evidence; heuristic trigger search is diagnostic-only"
                ),
            )
        )
    if temporal_work_reached:
        diagnostics.append(
            _diagnostic(
                "sva_temporal_work_reached",
                (
                    "SVA temporal analysis reached its deterministic "
                    f"lookback/value budget "
                    f"({stats.get('temporal_lookback_budget')}/"
                    f"{stats.get('temporal_value_budget')})"
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


class CausalPreparationError(RuntimeError):
    """Preparation failed before an authoritative graph could be built."""

    def __init__(
        self,
        diagnostics: List[Dict[str, Any]],
        *,
        status: str = "incomplete",
    ):
        super().__init__(
            diagnostics[0]["message"]
            if diagnostics
            else "causal analysis preparation failed"
        )
        self.diagnostics = diagnostics
        self.status = status


def _validate_file_identities(
    request: CausalAnalysisRequestV2,
) -> Tuple[List[Dict[str, Any]], Dict[str, str]]:
    """Revalidate every trust-boundary byte before consulting caches."""
    diagnostics: List[Dict[str, Any]] = []
    try:
        actual_hash, actual_bytes = sha256_file(request.trace.path)
    except OSError:
        diagnostics.append(
            _diagnostic(
                "waveform_hash_mismatch",
                "trace file is unavailable",
            )
        )
    else:
        if (actual_hash, actual_bytes) != (
            request.trace.sha256,
            request.trace.bytes,
        ):
            diagnostics.append(
                _diagnostic(
                    "waveform_hash_mismatch",
                    "trace bytes or SHA-256 differ from the request",
                )
            )

    artifact_by_path: Dict[str, str] = {}
    for artifact in request.rtl_files:
        absolute_path = os.path.abspath(artifact.path)
        artifact_by_path[absolute_path] = artifact.artifact_id
        try:
            actual_hash, actual_bytes = sha256_file(artifact.path)
        except OSError:
            diagnostics.append(
                _diagnostic(
                    "rtl_file_hash_mismatch",
                    "required RTL file is unavailable",
                    artifact_id=artifact.artifact_id,
                )
            )
            continue
        if (actual_hash, actual_bytes) != (
            artifact.sha256,
            artifact.bytes,
        ):
            diagnostics.append(
                _diagnostic(
                    "rtl_file_hash_mismatch",
                    "required RTL bytes or SHA-256 differ from the request",
                    artifact_id=artifact.artifact_id,
                )
            )
    return diagnostics, artifact_by_path


def _parsed_design_cache_key(
    request: CausalAnalysisRequestV2,
) -> str:
    return canonical_sha256(
        {
            "schema_version": "parsed_design_cache_key.v1",
            "rtl_files": sorted(
                (
                    {
                        "artifact_id": artifact.artifact_id,
                        "sha256": artifact.sha256,
                        "bytes": artifact.bytes,
                    }
                    for artifact in request.rtl_files
                ),
                key=lambda row: row["artifact_id"],
            ),
            "analyzer_revision": ANALYZER_REVISION,
            "hdlconvertor_revision": HDLCONVERTOR_REVISION,
            "parser_language_policy": "closure-system-verilog-if-any-v1",
            "include_policy": "required-artifact-directories-v1",
            "strict": request.strict,
        }
    )


def _persistent_cache_path(cache_key: str) -> Optional[Path]:
    cache_root = os.environ.get("VCA_PARSED_DESIGN_CACHE_DIR")
    if not cache_root:
        return None
    return Path(cache_root) / f"{cache_key}.json"


def _valid_cache_entry(
    entry: Any,
    cache_key: str,
) -> bool:
    if not isinstance(entry, Mapping):
        return False
    payload = entry.get("payload")
    return bool(
        entry.get("schema_version")
        == "parsed_design_cache_entry.v1"
        and entry.get("cache_key") == cache_key
        and isinstance(payload, Mapping)
        and entry.get("payload_sha256") == canonical_sha256(payload)
        and not contains_absolute_path(payload)
        and not _contains_absolute_path_fragment(payload)
    )


def _load_persistent_cache(
    cache_key: str,
) -> Optional[Dict[str, Any]]:
    cache_path = _persistent_cache_path(cache_key)
    if cache_path is None:
        return None
    try:
        entry = json.loads(cache_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    return entry if _valid_cache_entry(entry, cache_key) else None


def _write_persistent_cache(
    cache_key: str,
    payload: Mapping[str, Any],
) -> None:
    cache_path = _persistent_cache_path(cache_key)
    if cache_path is None:
        return
    entry = {
        "schema_version": "parsed_design_cache_entry.v1",
        "cache_key": cache_key,
        "payload_sha256": canonical_sha256(payload),
        "payload": payload,
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_path = tempfile.mkstemp(
        prefix=f".{cache_key}.",
        suffix=".tmp",
        dir=cache_path.parent,
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(
                entry,
                handle,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, cache_path)
    finally:
        if os.path.exists(temporary_path):
            os.remove(temporary_path)


def _prepare_parser(
    request: CausalAnalysisRequestV2,
    artifact_by_path: Mapping[str, str],
) -> Tuple[VerilogParser, str]:
    cache_key = _parsed_design_cache_key(request)
    artifact_paths = {
        artifact.artifact_id: artifact.path
        for artifact in request.rtl_files
    }
    artifact_path_identity = tuple(sorted(artifact_paths.items()))
    cached: Optional[Dict[str, Any]]
    with _PARSED_DESIGN_CACHE_LOCK:
        live_entry = _PARSED_DESIGN_CACHE.get(cache_key)
        cached = None
        if live_entry is not None:
            cached = {
                "cache_key": live_entry.get("cache_key"),
                "payload_sha256": live_entry.get("payload_sha256"),
                # Cache payloads are immutable after publication.
                "payload": live_entry.get("payload"),
                "prepared_parser": (
                    live_entry.get("prepared_parser")
                    if live_entry.get("artifact_path_identity")
                    == artifact_path_identity
                    else None
                ),
            }
            _PARSED_DESIGN_CACHE.move_to_end(cache_key)
    if cached is not None:
        payload = cached.get("payload")
        prepared_parser = cached.get("prepared_parser")
        if (
            cached.get("cache_key") == cache_key
            and isinstance(prepared_parser, VerilogParser)
        ):
            # The same-process object was created only after byte/hash
            # validation and is never mutated structurally after publication.
            return prepared_parser, "hit"
        if (
            cached.get("cache_key") == cache_key
            and isinstance(payload, Mapping)
            and cached.get("payload_sha256")
            == canonical_sha256(payload)
            and not contains_absolute_path(payload)
            and not _contains_absolute_path_fragment(payload)
        ):
            try:
                return (
                    VerilogParser.from_prepared_design(
                        payload,
                        artifact_paths,
                        strict=True,
                    ),
                    "hit",
                )
            except (KeyError, TypeError, ValueError):
                pass
        with _PARSED_DESIGN_CACHE_LOCK:
            _PARSED_DESIGN_CACHE.pop(cache_key, None)

    persistent = _load_persistent_cache(cache_key)
    if persistent is not None:
        try:
            parser = VerilogParser.from_prepared_design(
                persistent["payload"],
                artifact_paths,
                strict=True,
            )
        except (KeyError, TypeError, ValueError):
            pass
        else:
            with _PARSED_DESIGN_CACHE_LOCK:
                _PARSED_DESIGN_CACHE[cache_key] = {
                    **persistent,
                    "artifact_path_identity": artifact_path_identity,
                    "prepared_parser": parser,
                }
                _PARSED_DESIGN_CACHE.move_to_end(cache_key)
                while (
                    len(_PARSED_DESIGN_CACHE)
                    > _PARSED_DESIGN_CACHE_MAX
                ):
                    _PARSED_DESIGN_CACHE.popitem(last=False)
            return parser, "hit"

    parser = VerilogParser(strict=True)
    parser.parse_files_strict(
        artifact.path
        for artifact in sorted(
            request.rtl_files,
            key=lambda item: item.artifact_id,
        )
    )
    payload = parser.to_prepared_design(artifact_by_path)
    if (
        not contains_absolute_path(payload)
        and not _contains_absolute_path_fragment(payload)
    ):
        entry = {
            "schema_version": "parsed_design_cache_entry.v1",
            "cache_key": cache_key,
            "payload_sha256": canonical_sha256(payload),
            "payload": payload,
            "artifact_path_identity": artifact_path_identity,
            "prepared_parser": parser,
        }
        with _PARSED_DESIGN_CACHE_LOCK:
            _PARSED_DESIGN_CACHE[cache_key] = entry
            _PARSED_DESIGN_CACHE.move_to_end(cache_key)
            while len(_PARSED_DESIGN_CACHE) > _PARSED_DESIGN_CACHE_MAX:
                _PARSED_DESIGN_CACHE.popitem(last=False)
        try:
            _write_persistent_cache(cache_key, payload)
        except OSError:
            # Cache availability is never part of semantic authority.
            pass
        cache_status = "miss"
    else:
        cache_status = "disabled"
    return parser, cache_status


def _session_identity(
    request: CausalAnalysisRequestV2,
) -> Tuple[Any, ...]:
    return (
        request.trace.path,
        request.trace.sha256,
        request.trace.bytes,
        tuple(
            (
                artifact.artifact_id,
                artifact.path,
                artifact.sha256,
                artifact.bytes,
            )
            for artifact in request.rtl_files
        ),
        request.clock_signal,
        request.strict,
    )


class PreparedCausalAnalysis:
    """Verified parser/waveform state reusable across typed endpoints."""

    def __init__(
        self,
        request: CausalAnalysisRequestV2 | Mapping[str, Any],
        *,
        production: bool = True,
    ):
        if not isinstance(request, CausalAnalysisRequestV2):
            request = CausalAnalysisRequestV2.from_dict(
                request, production=production
            )
        if production and not request.strict:
            raise ValueError("production V2 API requires strict=true")
        if not production and request.strict:
            raise ValueError(
                "diagnostic heuristic requests require strict=false"
            )

        diagnostics, artifact_by_path = _validate_file_identities(request)
        if diagnostics:
            raise CausalPreparationError(diagnostics)
        try:
            parser, cache_status = _prepare_parser(
                request, artifact_by_path
            )
        except Exception as error:
            artifact_id = next(
                (
                    artifact.artifact_id
                    for artifact in request.rtl_files
                    if artifact.path in str(error)
                ),
                None,
            )
            raise CausalPreparationError(
                [
                    _diagnostic(
                        "rtl_parse_failed",
                        (
                            "required RTL parse failed "
                            f"({type(error).__name__})"
                        ),
                        artifact_id=artifact_id,
                    )
                ]
            ) from error

        try:
            waveform = CycleAlignedWaveform(
                request.trace.path,
                request.clock_signal,
                exact_clock=True,
            )
        except ValueError as error:
            raise CausalPreparationError(
                [_diagnostic("clock_not_exact", str(error))]
            ) from error
        except Exception as error:
            raise CausalPreparationError(
                [
                    _diagnostic(
                        "rtl_construct_unsupported",
                        (
                            "waveform open failed "
                            f"({type(error).__name__})"
                        ),
                    )
                ],
                status="unsupported",
            ) from error

        self.request = request
        self.parser = parser
        self.waveform = waveform
        self.artifact_by_path = dict(artifact_by_path)
        self.cache_status = cache_status
        self._identity = _session_identity(request)
        self._production = production
        self._closed = False

    def build(
        self,
        request: CausalAnalysisRequestV2 | Mapping[str, Any],
    ) -> Dict[str, Any]:
        """Build one graph while reusing verified frontend state."""
        if self._closed:
            raise RuntimeError("prepared causal analysis is closed")
        if not isinstance(request, CausalAnalysisRequestV2):
            request = CausalAnalysisRequestV2.from_dict(
                request, production=self._production
            )
        if _session_identity(request) != self._identity:
            raise ValueError(
                "request does not match prepared RTL/trace/clock identity"
            )

        diagnostics: List[Dict[str, Any]] = []
        if not self.waveform.has_exact_signal(request.endpoint_signal):
            diagnostics.append(
                _diagnostic(
                    "endpoint_not_exact",
                    (
                        "endpoint is not an exact waveform signal: "
                        f"{request.endpoint_signal}"
                    ),
                )
            )
            return _empty_graph(request, diagnostics)
        if request.endpoint_cycle >= self.waveform.get_cycle_count():
            diagnostics.append(
                _diagnostic(
                    "endpoint_not_exact",
                    (
                        f"endpoint cycle {request.endpoint_cycle} "
                        "is outside the waveform"
                    ),
                )
            )
            return _empty_graph(request, diagnostics)
        if self.waveform.get_signal_value(
            request.endpoint_signal,
            request.endpoint_cycle,
        ) is None:
            diagnostics.append(
                _diagnostic(
                    "waveform_signal_missing",
                    "endpoint has no sampled waveform value",
                )
            )
            return _empty_graph(request, diagnostics)

        slicer = BackwardSlicer(
            self.parser,
            self.waveform,
            max_depth=request.max_depth,
            max_nodes=request.max_nodes,
            allow_heuristic_sva_trigger=not self._production,
        )
        nodes, edges = slicer.slice_from_endpoint(
            request.endpoint_signal,
            request.endpoint_cycle,
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
            self.artifact_by_path,
        )

    def close(self) -> None:
        if not self._closed:
            self.waveform.close()
            self._closed = True

    def __enter__(self) -> "PreparedCausalAnalysis":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        self.close()
        return False


def prepare_causal_analysis(
    request: CausalAnalysisRequestV2 | Mapping[str, Any],
) -> PreparedCausalAnalysis:
    """Prepare one verified production session for multiple endpoints."""
    return PreparedCausalAnalysis(request, production=True)


def _build_causal_graph_v2(
    request: CausalAnalysisRequestV2 | Mapping[str, Any],
    *,
    production: bool,
) -> Dict[str, Any]:
    if not isinstance(request, CausalAnalysisRequestV2):
        request = CausalAnalysisRequestV2.from_dict(
            request, production=production
        )
    try:
        with PreparedCausalAnalysis(
            request, production=production
        ) as prepared:
            return prepared.build(request)
    except CausalPreparationError as error:
        return _empty_graph(
            request,
            error.diagnostics,
            status=error.status,
        )


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
