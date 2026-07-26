"""Strict, schema-only contracts for Verilog Causal Analysis V2."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Tuple

from .identity import canonical_sha256, contains_absolute_path, stable_id


REQUEST_SCHEMA = "verilog_causal_request.v2"
GRAPH_SCHEMA = "verilog_causal_graph.v2"
GRAPH_STATUSES = frozenset({"complete", "incomplete", "unsupported"})
IDENTITY_STRENGTHS = frozenset(
    {"exact", "hierarchy_inferred", "basename_fallback", "unresolved"}
)
EVIDENCE_STRENGTHS = frozenset(
    {
        "expression_counterfactual",
        "branch_observed",
        "toggle_supported",
        "structural_only",
        "unresolved",
    }
)
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class ContractError(ValueError):
    """Raised when a typed V2 contract is malformed."""


def _exact_keys(row: Mapping[str, Any], required: Iterable[str], where: str) -> None:
    expected = set(required)
    actual = set(row)
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    if missing or extra:
        raise ContractError(f"{where} keys mismatch: missing={missing}, extra={extra}")


def _sha256(value: Any, where: str) -> str:
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
        raise ContractError(f"{where} must be a lowercase SHA-256")
    return value


def _positive_int(value: Any, where: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ContractError(f"{where} must be a positive integer")
    return value


@dataclass(frozen=True)
class FileArtifactV2:
    artifact_id: str
    path: str
    sha256: str
    bytes: int

    @classmethod
    def from_dict(cls, row: Mapping[str, Any], *, where: str) -> "FileArtifactV2":
        _exact_keys(row, {"artifact_id", "path", "sha256", "bytes"}, where)
        artifact_id = row["artifact_id"]
        if not isinstance(artifact_id, str) or not artifact_id:
            raise ContractError(f"{where}.artifact_id must be a non-empty string")
        path = row["path"]
        if not isinstance(path, str) or not Path(path).is_absolute():
            raise ContractError(f"{where}.path must be an absolute local path")
        size = row["bytes"]
        if isinstance(size, bool) or not isinstance(size, int) or size < 0:
            raise ContractError(f"{where}.bytes must be a non-negative integer")
        return cls(artifact_id, path, _sha256(row["sha256"], f"{where}.sha256"), size)

    def to_dict(self, *, include_path: bool = True) -> Dict[str, Any]:
        row: Dict[str, Any] = {
            "artifact_id": self.artifact_id,
            "sha256": self.sha256,
            "bytes": self.bytes,
        }
        if include_path:
            row["path"] = self.path
        return row


@dataclass(frozen=True)
class TraceArtifactV2:
    path: str
    format: str
    sha256: str
    bytes: int

    @classmethod
    def from_dict(cls, row: Mapping[str, Any]) -> "TraceArtifactV2":
        _exact_keys(row, {"path", "format", "sha256", "bytes"}, "trace")
        path = row["path"]
        if not isinstance(path, str) or not Path(path).is_absolute():
            raise ContractError("trace.path must be an absolute local path")
        if row["format"] != "fst":
            raise ContractError("trace.format must be 'fst'")
        size = row["bytes"]
        if isinstance(size, bool) or not isinstance(size, int) or size < 0:
            raise ContractError("trace.bytes must be a non-negative integer")
        return cls(path, "fst", _sha256(row["sha256"], "trace.sha256"), size)

    def to_dict(self, *, include_path: bool = True) -> Dict[str, Any]:
        row: Dict[str, Any] = {
            "format": self.format,
            "sha256": self.sha256,
            "bytes": self.bytes,
        }
        if include_path:
            row["path"] = self.path
        return row


@dataclass(frozen=True)
class CausalAnalysisRequestV2:
    request_id: str
    trace: TraceArtifactV2
    rtl_files: Tuple[FileArtifactV2, ...]
    clock_signal: str
    endpoint_signal: str
    endpoint_cycle: int
    max_depth: int
    max_nodes: int
    random_seed: int
    strict: bool

    @classmethod
    def from_dict(
        cls, row: Mapping[str, Any], *, production: bool = True
    ) -> "CausalAnalysisRequestV2":
        _exact_keys(
            row,
            {
                "schema_version",
                "request_id",
                "trace",
                "rtl_files",
                "clock",
                "endpoint",
                "bounds",
                "random_seed",
                "strict",
            },
            "request",
        )
        if row["schema_version"] != REQUEST_SCHEMA:
            raise ContractError(f"schema_version must be {REQUEST_SCHEMA!r}")
        trace = TraceArtifactV2.from_dict(row["trace"])
        rtl_rows = row["rtl_files"]
        if not isinstance(rtl_rows, list) or not rtl_rows:
            raise ContractError("rtl_files must be a non-empty list")
        rtl_files = tuple(
            FileArtifactV2.from_dict(item, where=f"rtl_files[{index}]")
            for index, item in enumerate(rtl_rows)
        )
        ids = [item.artifact_id for item in rtl_files]
        if len(ids) != len(set(ids)):
            raise ContractError("rtl_files artifact_id values must be unique")
        _exact_keys(row["clock"], {"signal", "edge"}, "clock")
        if row["clock"]["edge"] != "rising":
            raise ContractError("only a rising clock edge is supported")
        clock_signal = row["clock"]["signal"]
        if not isinstance(clock_signal, str) or not clock_signal:
            raise ContractError("clock.signal must be a non-empty exact signal")
        _exact_keys(row["endpoint"], {"signal", "cycle"}, "endpoint")
        endpoint_signal = row["endpoint"]["signal"]
        if not isinstance(endpoint_signal, str) or not endpoint_signal:
            raise ContractError("endpoint.signal must be a non-empty exact signal")
        endpoint_cycle = row["endpoint"]["cycle"]
        if isinstance(endpoint_cycle, bool) or not isinstance(endpoint_cycle, int) or endpoint_cycle < 0:
            raise ContractError("endpoint.cycle must be a non-negative integer")
        _exact_keys(row["bounds"], {"max_depth", "max_nodes"}, "bounds")
        max_depth = _positive_int(row["bounds"]["max_depth"], "bounds.max_depth")
        max_nodes = _positive_int(row["bounds"]["max_nodes"], "bounds.max_nodes")
        seed = row["random_seed"]
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ContractError("random_seed must be a non-negative integer")
        strict = row["strict"]
        if not isinstance(strict, bool):
            raise ContractError("strict must be boolean")
        if production and not strict:
            raise ContractError("production requests require strict=true")
        request = cls(
            str(row["request_id"]),
            trace,
            rtl_files,
            clock_signal,
            endpoint_signal,
            endpoint_cycle,
            max_depth,
            max_nodes,
            seed,
            strict,
        )
        expected = request.computed_request_id()
        if request.request_id != expected:
            raise ContractError(
                f"request_id mismatch: expected {expected}, got {request.request_id}"
            )
        return request

    def identity_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": REQUEST_SCHEMA,
            "trace": self.trace.to_dict(include_path=False),
            "rtl_files": [
                item.to_dict(include_path=False)
                for item in sorted(self.rtl_files, key=lambda value: value.artifact_id)
            ],
            "clock": {"signal": self.clock_signal, "edge": "rising"},
            "endpoint": {
                "signal": self.endpoint_signal,
                "cycle": self.endpoint_cycle,
            },
            "bounds": {
                "max_depth": self.max_depth,
                "max_nodes": self.max_nodes,
            },
            "random_seed": self.random_seed,
            "strict": self.strict,
        }

    def computed_request_id(self) -> str:
        return stable_id("vcr_", self.identity_dict())

    @property
    def request_sha256(self) -> str:
        return canonical_sha256(self.identity_dict())

    def to_dict(self) -> Dict[str, Any]:
        row = self.identity_dict()
        row["request_id"] = self.request_id
        row["trace"] = self.trace.to_dict()
        row["rtl_files"] = [item.to_dict() for item in self.rtl_files]
        return row


def make_request_v2(
    *,
    trace: Mapping[str, Any],
    rtl_files: Iterable[Mapping[str, Any]],
    clock_signal: str,
    endpoint_signal: str,
    endpoint_cycle: int,
    max_depth: int = 12,
    max_nodes: int = 120,
    random_seed: int = 0,
    strict: bool = True,
) -> CausalAnalysisRequestV2:
    """Create a request while deriving its path-free canonical request ID."""
    provisional = {
        "schema_version": REQUEST_SCHEMA,
        "request_id": "pending",
        "trace": dict(trace),
        "rtl_files": [dict(item) for item in rtl_files],
        "clock": {"signal": clock_signal, "edge": "rising"},
        "endpoint": {"signal": endpoint_signal, "cycle": endpoint_cycle},
        "bounds": {"max_depth": max_depth, "max_nodes": max_nodes},
        "random_seed": random_seed,
        "strict": strict,
    }
    unchecked = CausalAnalysisRequestV2.from_dict_without_id(provisional)
    provisional["request_id"] = unchecked.computed_request_id()
    return CausalAnalysisRequestV2.from_dict(provisional, production=strict)


def _from_dict_without_id(
    cls: type[CausalAnalysisRequestV2], row: Mapping[str, Any]
) -> CausalAnalysisRequestV2:
    copied = dict(row)
    copied["request_id"] = "vcr_" + "0" * 64
    expected = ""
    try:
        return cls.from_dict(copied, production=False)
    except ContractError as error:
        if not str(error).startswith("request_id mismatch:"):
            raise
        expected = str(error).split("expected ", 1)[1].split(",", 1)[0]
    # Parse once more using the request ID mentioned by the validator.
    copied["request_id"] = expected
    return cls.from_dict(copied, production=False)


CausalAnalysisRequestV2.from_dict_without_id = classmethod(_from_dict_without_id)  # type: ignore[attr-defined]


def validate_graph_v2(graph: Mapping[str, Any]) -> Dict[str, Any]:
    """Fail closed if a purported V2 graph violates the durable result schema."""
    _exact_keys(
        graph,
        {
            "schema_version",
            "graph_id",
            "status",
            "identity",
            "bounds",
            "nodes",
            "edges",
            "diagnostics",
        },
        "graph",
    )
    if graph["schema_version"] != GRAPH_SCHEMA:
        raise ContractError(f"graph.schema_version must be {GRAPH_SCHEMA}")
    if graph["status"] not in GRAPH_STATUSES:
        raise ContractError(f"graph.status must be one of {sorted(GRAPH_STATUSES)}")
    if not isinstance(graph["graph_id"], str) or not graph["graph_id"].startswith("vcg_"):
        raise ContractError("graph.graph_id must start with vcg_")
    _exact_keys(
        graph["identity"],
        {
            "request_sha256",
            "trace_sha256",
            "rtl_set_sha256",
            "analyzer_revision",
            "hdlconvertor_revision",
            "random_seed",
        },
        "graph.identity",
    )
    for key in ("request_sha256", "trace_sha256", "rtl_set_sha256"):
        _sha256(graph["identity"][key], f"graph.identity.{key}")
    _exact_keys(
        graph["bounds"],
        {
            "max_depth",
            "max_nodes",
            "max_depth_reached",
            "max_nodes_reached",
        },
        "graph.bounds",
    )
    _positive_int(graph["bounds"]["max_depth"], "graph.bounds.max_depth")
    _positive_int(graph["bounds"]["max_nodes"], "graph.bounds.max_nodes")
    for key in ("max_depth_reached", "max_nodes_reached"):
        if not isinstance(graph["bounds"][key], bool):
            raise ContractError(f"graph.bounds.{key} must be boolean")
    if not isinstance(graph["nodes"], list) or not isinstance(graph["edges"], list):
        raise ContractError("graph nodes and edges must be lists")
    node_ids = set()
    for index, node in enumerate(graph["nodes"]):
        _exact_keys(
            node,
            {
                "node_id",
                "signal_id",
                "signal",
                "cycle",
                "value",
                "depth",
                "is_endpoint",
                "is_slice_leaf",
                "rtl_context_status",
                "identity_strength",
                "suspect_score",
            },
            f"graph.nodes[{index}]",
        )
        if node["node_id"] in node_ids:
            raise ContractError("graph node IDs must be unique")
        node_ids.add(node["node_id"])
        if node["identity_strength"] not in IDENTITY_STRENGTHS:
            raise ContractError("node identity_strength is invalid")
        if not 0 <= float(node["suspect_score"]) <= 1:
            raise ContractError("node suspect_score must be in [0, 1]")
    edge_ids = set()
    for index, edge in enumerate(graph["edges"]):
        _exact_keys(
            edge,
            {
                "edge_id",
                "src_node_id",
                "dst_node_id",
                "dependency_type",
                "identity_strength",
                "evidence_strength",
                "contribution_score",
                "reason_code",
                "rtl_evidence",
                "change_examples",
            },
            f"graph.edges[{index}]",
        )
        if edge["edge_id"] in edge_ids:
            raise ContractError("graph edge IDs must be unique")
        edge_ids.add(edge["edge_id"])
        if edge["src_node_id"] not in node_ids or edge["dst_node_id"] not in node_ids:
            raise ContractError("graph edge references an unknown node ID")
        if edge["identity_strength"] not in IDENTITY_STRENGTHS:
            raise ContractError("edge identity_strength is invalid")
        if edge["evidence_strength"] not in EVIDENCE_STRENGTHS:
            raise ContractError("edge evidence_strength is invalid")
        if not 0 <= float(edge["contribution_score"]) <= 1:
            raise ContractError("edge contribution_score must be in [0, 1]")
        _exact_keys(
            edge["rtl_evidence"],
            {
                "artifact_id",
                "line_start",
                "line_end",
                "snippet",
                "snippet_sha256",
                "expression",
                "condition",
            },
            f"graph.edges[{index}].rtl_evidence",
        )
        _sha256(
            edge["rtl_evidence"]["snippet_sha256"],
            f"graph.edges[{index}].rtl_evidence.snippet_sha256",
        )
    if not isinstance(graph["diagnostics"], list):
        raise ContractError("graph.diagnostics must be a list")
    if contains_absolute_path(graph):
        raise ContractError("graph must not contain absolute paths")
    return dict(graph)
