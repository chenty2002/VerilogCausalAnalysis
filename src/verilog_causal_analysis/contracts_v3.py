"""Typed opt-in request contract for the C1 Chisel semantic profile."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple

from .identity import canonical_sha256, stable_id


REQUEST_SCHEMA_V3 = "verilog_causal_request.v3"
SEMANTIC_GRAPH_SCHEMA = "verilog_causal_semantic_graph.v1"
CHISEL_PROFILE_VERSION = "chisel-semantic-profile.v1"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_C1_FEATURES = frozenset({"instance_graph", "endpoint_projection"})


class ContractV3Error(ValueError):
    pass


def _keys(row: Mapping[str, Any], expected: Iterable[str], where: str) -> None:
    expected_set = set(expected)
    if set(row) != expected_set:
        raise ContractV3Error(
            f"{where} keys mismatch: missing={sorted(expected_set - set(row))}, "
            f"extra={sorted(set(row) - expected_set)}"
        )


def _hash(value: Any, where: str) -> str:
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
        raise ContractV3Error(f"{where} must be a lowercase SHA-256")
    return value


@dataclass(frozen=True)
class FileArtifactV3:
    artifact_id: str
    path: str
    sha256: str
    bytes: int
    kind: Optional[str] = None

    @classmethod
    def from_dict(
        cls, row: Mapping[str, Any], *, where: str, semantic: bool = False
    ) -> "FileArtifactV3":
        expected = {"artifact_id", "path", "sha256", "bytes"}
        if semantic:
            expected.add("kind")
        _keys(row, expected, where)
        artifact_id = row["artifact_id"]
        path = row["path"]
        size = row["bytes"]
        kind = row.get("kind")
        if not isinstance(artifact_id, str) or not artifact_id:
            raise ContractV3Error(f"{where}.artifact_id must be non-empty")
        if not isinstance(path, str) or not Path(path).is_absolute():
            raise ContractV3Error(f"{where}.path must be absolute")
        if isinstance(size, bool) or not isinstance(size, int) or size < 0:
            raise ContractV3Error(f"{where}.bytes must be non-negative")
        if semantic and (
            not isinstance(kind, str)
            or kind != "assertion_endpoint_projection"
        ):
            raise ContractV3Error(
                f"{where}.kind must be assertion_endpoint_projection"
            )
        return cls(
            artifact_id,
            path,
            _hash(row["sha256"], f"{where}.sha256"),
            size,
            kind,
        )

    def identity_dict(self) -> Dict[str, Any]:
        row: Dict[str, Any] = {
            "artifact_id": self.artifact_id,
            "sha256": self.sha256,
            "bytes": self.bytes,
        }
        if self.kind is not None:
            row["kind"] = self.kind
        return row

    def to_dict(self) -> Dict[str, Any]:
        return {**self.identity_dict(), "path": self.path}


@dataclass(frozen=True)
class SemanticProfileV3:
    name: str
    version: str
    features: Tuple[str, ...]


@dataclass(frozen=True)
class EndpointV3:
    signal: str
    cycle: int
    projection_mode: str
    evidence_ref: Optional[str]
    predicate_members: Tuple[str, ...]


@dataclass(frozen=True)
class CausalAnalysisRequestV3:
    request_id: str
    trace: FileArtifactV3
    rtl_files: Tuple[FileArtifactV3, ...]
    semantic_profile: SemanticProfileV3
    clock_signal: str
    endpoint: EndpointV3
    semantic_inputs: Tuple[FileArtifactV3, ...]
    bounds: Mapping[str, int]
    random_seed: int
    strict: bool

    @classmethod
    def from_dict(
        cls, row: Mapping[str, Any], *, production: bool = True
    ) -> "CausalAnalysisRequestV3":
        _keys(
            row,
            {
                "schema_version",
                "request_id",
                "trace",
                "rtl_files",
                "semantic_profile",
                "clock",
                "endpoint",
                "semantic_inputs",
                "bounds",
                "random_seed",
                "strict",
            },
            "request",
        )
        if row["schema_version"] != REQUEST_SCHEMA_V3:
            raise ContractV3Error(
                f"schema_version must be {REQUEST_SCHEMA_V3}"
            )
        trace_row = dict(row["trace"])
        _keys(
            trace_row,
            {"artifact_id", "path", "format", "sha256", "bytes"},
            "trace",
        )
        if trace_row.pop("format") != "fst":
            raise ContractV3Error("trace.format must be fst")
        trace = FileArtifactV3.from_dict(trace_row, where="trace")
        rtl_rows = row["rtl_files"]
        if not isinstance(rtl_rows, list) or not rtl_rows:
            raise ContractV3Error("rtl_files must be non-empty")
        rtl_files = tuple(
            FileArtifactV3.from_dict(item, where=f"rtl_files[{index}]")
            for index, item in enumerate(rtl_rows)
        )
        profile_row = row["semantic_profile"]
        _keys(profile_row, {"name", "version", "features"}, "semantic_profile")
        features = profile_row["features"]
        if (
            profile_row["name"] != "chisel"
            or profile_row["version"] != CHISEL_PROFILE_VERSION
            or not isinstance(features, list)
            or not features
            or any(item not in _C1_FEATURES for item in features)
        ):
            raise ContractV3Error(
                "C1 supports only the explicit chisel profile features "
                f"{sorted(_C1_FEATURES)}"
            )
        canonical_features = tuple(sorted(set(features)))
        if "instance_graph" not in canonical_features:
            raise ContractV3Error("C1 requires feature instance_graph")
        profile = SemanticProfileV3(
            "chisel", CHISEL_PROFILE_VERSION, canonical_features
        )
        _keys(row["clock"], {"signal", "edge"}, "clock")
        if row["clock"]["edge"] != "rising":
            raise ContractV3Error("only rising clock is supported")
        clock_signal = row["clock"]["signal"]
        if not isinstance(clock_signal, str) or not clock_signal:
            raise ContractV3Error("clock.signal must be exact and non-empty")

        endpoint_row = row["endpoint"]
        _keys(endpoint_row, {"signal", "cycle", "projection"}, "endpoint")
        signal = endpoint_row["signal"]
        cycle = endpoint_row["cycle"]
        if not isinstance(signal, str) or not signal:
            raise ContractV3Error("endpoint.signal must be non-empty")
        if isinstance(cycle, bool) or not isinstance(cycle, int) or cycle < 0:
            raise ContractV3Error("endpoint.cycle must be non-negative")
        projection = endpoint_row["projection"]
        if projection is None:
            projection_mode = "none"
            evidence_ref = None
            predicate_members: Tuple[str, ...] = ()
        else:
            _keys(
                projection,
                {"mode", "predicate_members", "evidence_ref"},
                "endpoint.projection",
            )
            if projection["mode"] != "controller_supplied_exact":
                raise ContractV3Error(
                    "endpoint projection mode must be controller_supplied_exact"
                )
            evidence_ref = projection["evidence_ref"]
            members = projection["predicate_members"]
            if (
                not isinstance(evidence_ref, str)
                or not evidence_ref
                or not isinstance(members, list)
                or not members
                or any(not isinstance(item, str) or not item for item in members)
            ):
                raise ContractV3Error(
                    "projection requires an evidence_ref and predicate members"
                )
            projection_mode = projection["mode"]
            predicate_members = tuple(sorted(set(members)))
        endpoint = EndpointV3(
            signal, cycle, projection_mode, evidence_ref, predicate_members
        )

        semantic_rows = row["semantic_inputs"]
        if not isinstance(semantic_rows, list):
            raise ContractV3Error("semantic_inputs must be a list")
        semantic_inputs = tuple(
            FileArtifactV3.from_dict(
                item, where=f"semantic_inputs[{index}]", semantic=True
            )
            for index, item in enumerate(semantic_rows)
        )
        semantic_by_id = {item.artifact_id: item for item in semantic_inputs}
        if len(semantic_by_id) != len(semantic_inputs):
            raise ContractV3Error("semantic input IDs must be unique")
        if evidence_ref is not None and evidence_ref not in semantic_by_id:
            raise ContractV3Error(
                "endpoint projection references an undeclared semantic input"
            )
        if evidence_ref is None and semantic_inputs:
            raise ContractV3Error(
                "unreferenced semantic inputs are not allowed"
            )

        required_bounds = {
            "max_signal_nodes",
            "max_semantic_nodes",
            "max_edges",
            "max_seed_count",
            "max_intervals_per_signal",
            "max_temporal_samples",
            "max_waitfor_nodes",
            "max_waitfor_edges",
            "max_scc_candidates",
        }
        _keys(row["bounds"], required_bounds, "bounds")
        bounds: Dict[str, int] = {}
        for key in sorted(required_bounds):
            value = row["bounds"][key]
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ContractV3Error(f"bounds.{key} must be positive")
            bounds[key] = value
        seed = row["random_seed"]
        strict = row["strict"]
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ContractV3Error("random_seed must be non-negative")
        if not isinstance(strict, bool) or (production and not strict):
            raise ContractV3Error("production requests require strict=true")
        request = cls(
            str(row["request_id"]),
            trace,
            rtl_files,
            profile,
            clock_signal,
            endpoint,
            semantic_inputs,
            bounds,
            seed,
            strict,
        )
        if request.request_id != request.computed_request_id():
            raise ContractV3Error("request_id mismatch")
        return request

    def identity_dict(self) -> Dict[str, Any]:
        projection: Optional[Dict[str, Any]]
        if self.endpoint.projection_mode == "none":
            projection = None
        else:
            projection = {
                "mode": self.endpoint.projection_mode,
                "predicate_members": list(self.endpoint.predicate_members),
                "evidence_ref": self.endpoint.evidence_ref,
            }
        return {
            "schema_version": REQUEST_SCHEMA_V3,
            "trace": {**self.trace.identity_dict(), "format": "fst"},
            "rtl_files": [
                item.identity_dict()
                for item in sorted(
                    self.rtl_files, key=lambda artifact: artifact.artifact_id
                )
            ],
            "semantic_profile": {
                "name": self.semantic_profile.name,
                "version": self.semantic_profile.version,
                "features": list(self.semantic_profile.features),
            },
            "clock": {"signal": self.clock_signal, "edge": "rising"},
            "endpoint": {
                "signal": self.endpoint.signal,
                "cycle": self.endpoint.cycle,
                "projection": projection,
            },
            "semantic_inputs": [
                item.identity_dict()
                for item in sorted(
                    self.semantic_inputs,
                    key=lambda artifact: artifact.artifact_id,
                )
            ],
            "bounds": dict(self.bounds),
            "random_seed": self.random_seed,
            "strict": self.strict,
        }

    def computed_request_id(self) -> str:
        return stable_id("vcr3_", self.identity_dict())

    @property
    def request_sha256(self) -> str:
        return canonical_sha256(self.identity_dict())

    def to_dict(self) -> Dict[str, Any]:
        row = self.identity_dict()
        row["request_id"] = self.request_id
        row["trace"] = {**self.trace.to_dict(), "format": "fst"}
        row["rtl_files"] = [item.to_dict() for item in self.rtl_files]
        row["semantic_inputs"] = [
            item.to_dict() for item in self.semantic_inputs
        ]
        return row


def make_request_v3(**kwargs: Any) -> CausalAnalysisRequestV3:
    provisional = {
        "schema_version": REQUEST_SCHEMA_V3,
        "request_id": "pending",
        **kwargs,
    }
    provisional["request_id"] = "vcr3_" + "0" * 64
    try:
        CausalAnalysisRequestV3.from_dict(provisional)
    except ContractV3Error as error:
        if str(error) != "request_id mismatch":
            raise
    provisional["request_id"] = _computed_id_without_validation(provisional)
    return CausalAnalysisRequestV3.from_dict(provisional)


def _computed_id_without_validation(row: Mapping[str, Any]) -> str:
    copied = dict(row)
    copied["request_id"] = "vcr3_" + "0" * 64
    try:
        CausalAnalysisRequestV3.from_dict(copied)
    except ContractV3Error as error:
        if str(error) != "request_id mismatch":
            raise
    identity = {key: value for key, value in copied.items() if key != "request_id"}
    identity["semantic_profile"] = dict(identity["semantic_profile"])
    identity["semantic_profile"]["features"] = sorted(
        set(identity["semantic_profile"]["features"])
    )
    projection = identity["endpoint"]["projection"]
    if projection is not None:
        projection = dict(projection)
        projection["predicate_members"] = sorted(
            set(projection["predicate_members"])
        )
        identity["endpoint"] = dict(identity["endpoint"])
        identity["endpoint"]["projection"] = projection
    for key in ("rtl_files", "semantic_inputs"):
        identity[key] = [
            {name: value for name, value in item.items() if name != "path"}
            for item in sorted(identity[key], key=lambda item: item["artifact_id"])
        ]
    identity["trace"] = {
        name: value
        for name, value in identity["trace"].items()
        if name != "path"
    }
    identity["bounds"] = {
        key: identity["bounds"][key] for key in sorted(identity["bounds"])
    }
    return stable_id("vcr3_", identity)
