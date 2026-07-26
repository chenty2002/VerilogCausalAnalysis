"""Hash-bound controller assertion projection for C1 causal analysis."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

from .identity import sha256_file, stable_id
from .instance_graph import InstanceGraph
from .verilog_parser import (
    Dependency,
    DependencyLookupResult,
    DependencyType,
    StatementEvidence,
)


ASSERTION_PROJECTION_SCHEMA = "assertion_endpoint_projection.v1"


class EndpointProjectionError(ValueError):
    """A projection is malformed, stale, or does not match the request."""


@dataclass(frozen=True)
class AssertionEndpointProjection:
    projection_id: str
    artifact_id: str
    endpoint_signal: str
    endpoint_cycle: int
    clock_signal: str
    predicate_members: Tuple[str, ...]
    rtl_set_sha256: str
    trace_sha256: str

    def identity_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": ASSERTION_PROJECTION_SCHEMA,
            "artifact_id": self.artifact_id,
            "endpoint_signal": self.endpoint_signal,
            "endpoint_cycle": self.endpoint_cycle,
            "clock_signal": self.clock_signal,
            "predicate_members": list(self.predicate_members),
            "rtl_set_sha256": self.rtl_set_sha256,
            "trace_sha256": self.trace_sha256,
        }


def validate_assertion_projection(
    row: Mapping[str, Any],
    *,
    artifact_id: str,
    endpoint_signal: str,
    endpoint_cycle: int,
    clock_signal: str,
    rtl_set_sha256: str,
    trace_sha256: str,
) -> AssertionEndpointProjection:
    required = {
        "schema_version",
        "endpoint_signal",
        "endpoint_cycle",
        "clock_signal",
        "predicate_members",
        "rtl_set_sha256",
        "trace_sha256",
    }
    if set(row) != required:
        raise EndpointProjectionError(
            "assertion projection keys mismatch: "
            f"missing={sorted(required - set(row))}, "
            f"extra={sorted(set(row) - required)}"
        )
    if row["schema_version"] != ASSERTION_PROJECTION_SCHEMA:
        raise EndpointProjectionError(
            f"projection schema must be {ASSERTION_PROJECTION_SCHEMA}"
        )
    expected = {
        "endpoint_signal": endpoint_signal,
        "endpoint_cycle": endpoint_cycle,
        "clock_signal": clock_signal,
        "rtl_set_sha256": rtl_set_sha256,
        "trace_sha256": trace_sha256,
    }
    for key, value in expected.items():
        if row[key] != value:
            raise EndpointProjectionError(
                f"assertion projection {key} does not match the request"
            )
    members = row["predicate_members"]
    if (
        not isinstance(members, list)
        or not members
        or any(not isinstance(item, str) or not item for item in members)
        or len(set(members)) != len(members)
    ):
        raise EndpointProjectionError(
            "predicate_members must be a non-empty unique signal list"
        )
    canonical_members = tuple(sorted(members))
    identity = {
        "artifact_id": artifact_id,
        **{key: row[key] for key in sorted(required)},
        "predicate_members": list(canonical_members),
    }
    return AssertionEndpointProjection(
        projection_id=stable_id("vcp_", identity, length=24),
        artifact_id=artifact_id,
        endpoint_signal=endpoint_signal,
        endpoint_cycle=endpoint_cycle,
        clock_signal=clock_signal,
        predicate_members=canonical_members,
        rtl_set_sha256=rtl_set_sha256,
        trace_sha256=trace_sha256,
    )


def load_assertion_projection(
    path: str,
    *,
    artifact_id: str,
    sha256: str,
    bytes: int,
    endpoint_signal: str,
    endpoint_cycle: int,
    clock_signal: str,
    rtl_set_sha256: str,
    trace_sha256: str,
) -> AssertionEndpointProjection:
    actual_sha256, actual_bytes = sha256_file(path)
    if (actual_sha256, actual_bytes) != (sha256, bytes):
        raise EndpointProjectionError(
            "assertion projection bytes or SHA-256 differ from the request"
        )
    try:
        row = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise EndpointProjectionError(
            "assertion projection is not valid UTF-8 JSON"
        ) from error
    if not isinstance(row, dict):
        raise EndpointProjectionError(
            "assertion projection root must be an object"
        )
    return validate_assertion_projection(
        row,
        artifact_id=artifact_id,
        endpoint_signal=endpoint_signal,
        endpoint_cycle=endpoint_cycle,
        clock_signal=clock_signal,
        rtl_set_sha256=rtl_set_sha256,
        trace_sha256=trace_sha256,
    )


class ProjectedDependencyProvider:
    """Compose exact instance dependencies with one reviewed endpoint join."""

    def __init__(
        self,
        instance_graph: InstanceGraph,
        projection: Optional[AssertionEndpointProjection] = None,
    ):
        self.instance_graph = instance_graph
        self.projection = projection

    def infer_module_from_signal(
        self, signal: str, hierarchy: Optional[str] = None
    ) -> Optional[str]:
        return self.instance_graph.infer_module_from_signal(signal, hierarchy)

    def get_dependencies_for_signal(
        self, signal: str, module_name: Optional[str] = None
    ) -> List[Dependency]:
        if (
            self.projection is not None
            and self.instance_graph._clean(signal)
            == self.instance_graph._clean(self.projection.endpoint_signal)
        ):
            return [
                self._projection_dependency(member)
                for member in self.projection.predicate_members
            ]
        return self.instance_graph.get_dependencies_for_signal(
            signal, module_name
        )

    def lookup_dependencies(
        self, signal: str, module_name: Optional[str] = None
    ) -> DependencyLookupResult:
        dependencies = tuple(
            self.get_dependencies_for_signal(signal, module_name)
        )
        return DependencyLookupResult(
            dependencies,
            "exact" if dependencies else "unresolved",
            False,
        )

    def get_signal_sources(
        self, signal: str, module_name: Optional[str] = None
    ) -> List[Tuple[str, DependencyType]]:
        return [
            (dep.source, dep.dep_type)
            for dep in self.get_dependencies_for_signal(signal, module_name)
        ]

    def get_rtl_context(
        self, signal: str, module_name: Optional[str] = None
    ) -> Dict[str, Any]:
        if (
            self.projection is not None
            and self.instance_graph._clean(signal)
            == self.instance_graph._clean(self.projection.endpoint_signal)
        ):
            return {
                "signal_name": signal,
                "found": True,
                "definition": {
                    "type": "assertion_projection",
                    "width": 1,
                    "module": self.infer_module_from_signal(signal),
                    "file": "",
                    "line": 0,
                },
                "dependencies": [],
                "rtl_refs": [],
            }
        return self.instance_graph.get_rtl_context(signal, module_name)

    def _projection_dependency(self, member: str) -> Dependency:
        assert self.projection is not None
        identity = {
            "projection_id": self.projection.projection_id,
            "endpoint_signal": self.projection.endpoint_signal,
            "predicate_member": member,
        }
        statement = StatementEvidence(
            statement_id=stable_id("vcs_", identity, length=24),
            expression=member,
            module_name=(
                self.infer_module_from_signal(
                    self.projection.endpoint_signal
                )
                or ""
            ),
            target=self.projection.endpoint_signal,
            target_qualified=self.projection.endpoint_signal,
        )
        return Dependency(
            source=member,
            target=self.projection.endpoint_signal,
            dep_type=DependencyType.ASSERTION,
            source_qualified=member,
            target_qualified=self.projection.endpoint_signal,
            statement=statement,
        )
