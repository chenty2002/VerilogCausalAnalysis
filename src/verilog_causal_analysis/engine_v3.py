"""C0-C2 execution for the explicit V3 Chisel profile."""

from __future__ import annotations

import re
from typing import Any, Dict, Mapping, Optional

from .causal_slicer import BackwardSlicer
from .chisel_semantics import (
    build_normalized_design,
    c2_enabled,
    persistent_intervals,
)
from .contracts import make_request_v2
from .contracts_v3 import (
    CausalAnalysisRequestV3,
    CHISEL_PROFILE_VERSION,
    SEMANTIC_GRAPH_SCHEMA,
)
from .endpoint_projection import (
    EndpointProjectionError,
    ProjectedDependencyProvider,
    load_assertion_projection,
)
from .engine import PreparedCausalAnalysis, _convert_graph, _diagnostic
from .identity import ANALYZER_REVISION, stable_id, stable_set_sha256
from .instance_graph import InstanceGraph, InstanceGraphError


class _SemanticBoundaryProvider:
    """Stop raw recursion at registers represented by C2 semantic objects."""

    def __init__(self, provider: Any, register_signals: set[str]):
        self._provider = provider
        self._register_signals = register_signals

    @staticmethod
    def _clean(signal: str) -> str:
        return re.sub(r"\s*\[\d+:\d+\]$", "", signal)

    def get_dependencies_for_signal(
        self, signal: str, module_name: Optional[str] = None
    ):
        if self._clean(signal) in self._register_signals:
            return []
        return self._provider.get_dependencies_for_signal(
            signal, module_name
        )

    def __getattr__(self, name: str) -> Any:
        return getattr(self._provider, name)


def _v2_request(request: CausalAnalysisRequestV3):
    return make_request_v2(
        trace={
            "path": request.trace.path,
            "format": "fst",
            "sha256": request.trace.sha256,
            "bytes": request.trace.bytes,
        },
        rtl_files=[
            {
                "artifact_id": item.artifact_id,
                "path": item.path,
                "sha256": item.sha256,
                "bytes": item.bytes,
            }
            for item in request.rtl_files
        ],
        clock_signal=request.clock_signal,
        endpoint_signal=request.endpoint.signal,
        endpoint_cycle=request.endpoint.cycle,
        max_depth=request.bounds["max_signal_nodes"],
        max_nodes=request.bounds["max_signal_nodes"],
        random_seed=request.random_seed,
        strict=request.strict,
    )


def _identity(request: CausalAnalysisRequestV3) -> Dict[str, Any]:
    return {
        "request_sha256": request.request_sha256,
        "rtl_set_sha256": stable_set_sha256(
            [item.identity_dict() for item in request.rtl_files]
        ),
        "trace_sha256": request.trace.sha256,
        "analyzer_revision": (
            f"{ANALYZER_REVISION}+c2"
            if c2_enabled(request.semantic_profile.features)
            else f"{ANALYZER_REVISION}+c1"
        ),
        "profile_version": CHISEL_PROFILE_VERSION,
    }


def _empty_graph(
    request: CausalAnalysisRequestV3,
    diagnostics: list[dict[str, Any]],
) -> Dict[str, Any]:
    identity = _identity(request)
    return {
        "schema_version": SEMANTIC_GRAPH_SCHEMA,
        "graph_id": stable_id("vcsg_", identity),
        "status": "incomplete",
        "identity": identity,
        "endpoint": {
            "signal": request.endpoint.signal,
            "cycle": request.endpoint.cycle,
            "projection_id": None,
        },
        "signal_nodes": [],
        "semantic_nodes": [],
        "edges": [],
        "root_candidates": [],
        "bounds": {
            **dict(request.bounds),
            "signal_nodes_reached": False,
            "semantic_nodes_reached": False,
            "edges_reached": False,
        },
        "diagnostics": sorted(
            diagnostics,
            key=lambda row: (row["code"], row.get("message", "")),
        ),
    }


class PreparedCausalSessionV3:
    """Verified V3 session with reusable parser, waveform, and instance graph."""

    def __init__(
        self,
        request: CausalAnalysisRequestV3 | Mapping[str, Any],
        *,
        top_module: Optional[str] = None,
    ):
        if not isinstance(request, CausalAnalysisRequestV3):
            request = CausalAnalysisRequestV3.from_dict(request)
        self.request = request
        self._v2 = _v2_request(request)
        self._prepared = PreparedCausalAnalysis(self._v2, production=True)
        try:
            self.instance_graph = InstanceGraph.from_parser(
                self._prepared.parser,
                self._prepared.artifact_by_path,
                top_module=top_module,
            ).bind_waveform(self._prepared.waveform)
        except InstanceGraphError:
            self._prepared.close()
            raise
        self._normalized_design = (
            build_normalized_design(
                self.instance_graph,
                rtl_set_sha256=_identity(request)["rtl_set_sha256"],
                clock_signal=request.clock_signal,
                features=request.semantic_profile.features,
            )
            if c2_enabled(request.semantic_profile.features)
            else None
        )
        self._closed = False

    @property
    def normalized_design(self) -> Dict[str, Any]:
        if self._normalized_design is not None:
            return self._normalized_design
        return self.instance_graph.to_dict()

    def build(
        self,
        request: CausalAnalysisRequestV3 | Mapping[str, Any] | None = None,
    ) -> Dict[str, Any]:
        if self._closed:
            raise RuntimeError("prepared V3 session is closed")
        if request is None:
            request = self.request
        elif not isinstance(request, CausalAnalysisRequestV3):
            request = CausalAnalysisRequestV3.from_dict(request)
        if request.request_sha256 != self.request.request_sha256:
            raise ValueError("request does not match prepared V3 session")

        diagnostics = [
            _diagnostic(
                row["code"],
                row["message"],
            )
            for row in self.instance_graph.diagnostics
        ]
        projection = None
        if request.endpoint.evidence_ref is not None:
            artifact = next(
                item
                for item in request.semantic_inputs
                if item.artifact_id == request.endpoint.evidence_ref
            )
            try:
                projection = load_assertion_projection(
                    artifact.path,
                    artifact_id=artifact.artifact_id,
                    sha256=artifact.sha256,
                    bytes=artifact.bytes,
                    endpoint_signal=request.endpoint.signal,
                    endpoint_cycle=request.endpoint.cycle,
                    clock_signal=request.clock_signal,
                    rtl_set_sha256=_identity(request)["rtl_set_sha256"],
                    trace_sha256=request.trace.sha256,
                )
            except EndpointProjectionError as error:
                diagnostics.append(
                    _diagnostic(
                        "assertion_projection_invalid",
                        str(error),
                    )
                )
                return _empty_graph(request, diagnostics)
            if (
                projection.predicate_members
                != request.endpoint.predicate_members
            ):
                diagnostics.append(
                    _diagnostic(
                        "assertion_projection_mismatch",
                        (
                            "projection predicate members differ from the "
                            "request"
                        ),
                    )
                )
                return _empty_graph(request, diagnostics)

        provider: Any = ProjectedDependencyProvider(
            self.instance_graph, projection
        )
        projection_members = {
            re.sub(r"\s*\[\d+:\d+\]$", "", member)
            for member in (
                projection.predicate_members
                if projection is not None
                else ()
            )
        }
        selected_transitions = []
        if self._normalized_design is not None:
            all_transitions = self._normalized_design[
                "register_transitions"
            ]
            selected_transitions = [
                row
                for row in all_transitions
                if not projection_members
                or re.sub(r"\s*\[\d+:\d+\]$", "", row["signal"])
                in projection_members
            ]
            provider = _SemanticBoundaryProvider(
                provider,
                {
                    re.sub(r"\s*\[\d+:\d+\]$", "", row["signal"])
                    for row in selected_transitions
                },
            )
        if not self._prepared.waveform.has_exact_signal(
            request.endpoint.signal
        ):
            diagnostics.append(
                _diagnostic(
                    "endpoint_not_exact",
                    "endpoint is not an exact waveform signal",
                )
            )
            return _empty_graph(request, diagnostics)
        if request.endpoint.cycle >= self._prepared.waveform.get_cycle_count():
            diagnostics.append(
                _diagnostic(
                    "endpoint_not_exact",
                    "endpoint cycle is outside the waveform",
                )
            )
            return _empty_graph(request, diagnostics)

        slicer = BackwardSlicer(
            self._prepared.parser,
            self._prepared.waveform,
            max_depth=request.bounds["max_signal_nodes"],
            max_nodes=request.bounds["max_signal_nodes"],
            dependency_provider=provider,
        )
        nodes, edges = slicer.slice_from_endpoint(
            request.endpoint.signal, request.endpoint.cycle
        )
        stats = dict(slicer.get_statistics())
        ambiguity_by_identity: Dict[tuple[str, int], Dict[str, Any]] = {}
        for row in stats.get("identity_ambiguities") or []:
            key = (str(row["signal"]), int(row["candidate_count"]))
            previous = ambiguity_by_identity.get(key)
            if previous is None or int(row["cycle"]) < int(
                previous["cycle"]
            ):
                ambiguity_by_identity[key] = dict(row)
        stats["identity_ambiguities"] = [
            ambiguity_by_identity[key]
            for key in sorted(ambiguity_by_identity)
        ]
        raw = _convert_graph(
            self._v2,
            (node.to_dict() for node in nodes.values()),
            (edge.to_dict() for edge in edges),
            stats,
            diagnostics,
            self._prepared.artifact_by_path,
        )
        identity = _identity(request)
        node_ids: Dict[str, str] = {}
        signal_nodes = []
        for node in raw["nodes"]:
            new_id = stable_id(
                "vcn3_",
                identity,
                node["signal"],
                node["cycle"],
                node["value"],
                length=24,
            )
            node_ids[node["node_id"]] = new_id
            signal_nodes.append({**node, "node_id": new_id})
        graph_edges = []
        for edge in raw["edges"]:
            src = node_ids[edge["src_node_id"]]
            dst = node_ids[edge["dst_node_id"]]
            graph_edges.append(
                {
                    **edge,
                    "edge_id": stable_id(
                        "vce3_",
                        identity,
                        src,
                        dst,
                        edge["dependency_type"],
                        edge["rtl_evidence"],
                        length=24,
                    ),
                    "src_node_id": src,
                    "dst_node_id": dst,
                }
            )
        semantic_nodes = []
        projection_id = None
        if projection is not None:
            projection_id = projection.projection_id
            semantic_nodes.append(
                {
                    "semantic_id": stable_id(
                        "vcs_", identity, projection.projection_id, length=24
                    ),
                    "type": "assertion_predicate",
                    "endpoint_signal": projection.endpoint_signal,
                    "cycle": projection.endpoint_cycle,
                    "member_signals": list(projection.predicate_members),
                    "evidence_ref": projection.artifact_id,
                    "inference_rule": "controller_supplied_exact",
                }
            )
        interval_diagnostics: list[Dict[str, Any]] = []
        if self._normalized_design is not None:
            for transition in selected_transitions:
                semantic_nodes.append(
                    {
                        "semantic_id": transition["register_id"],
                        "type": "register_transition",
                        "signal": transition["signal"],
                        "clock_signal": transition["clock_signal"],
                        "reset_rules": transition["reset_rules"],
                        "update_rules": transition["update_rules"],
                        "hold_rule": transition["hold_rule"],
                        "counter_pattern": transition["counter_pattern"],
                        "statement_ids": transition["statement_ids"],
                        "inference_rule": transition["inference_rule"],
                    }
                )
            intervals, interval_diagnostics = persistent_intervals(
                self._normalized_design,
                self._prepared.waveform,
                end_cycle=request.endpoint.cycle,
                max_intervals=request.bounds["max_intervals_per_signal"],
                max_temporal_samples=request.bounds["max_temporal_samples"],
                subject_signals=(
                    projection.predicate_members
                    if projection is not None
                    else None
                ),
            )
            semantic_nodes.extend(intervals)
            semantic_nodes.sort(key=lambda row: row["semantic_id"])
            semantic_by_id = {
                row["semantic_id"]: row for row in semantic_nodes
            }
            predicate_nodes = [
                row
                for row in semantic_nodes
                if row["type"] == "assertion_predicate"
            ]
            register_nodes = [
                row
                for row in semantic_nodes
                if row["type"] == "register_transition"
            ]
            for interval in (
                row
                for row in semantic_nodes
                if row["type"] == "persistent_interval"
            ):
                if interval["register_id"] not in semantic_by_id:
                    continue
                graph_edges.append(
                    {
                        "edge_id": stable_id(
                            "vcse_",
                            identity,
                            interval["semantic_id"],
                            interval["register_id"],
                            "persistent_update_rule",
                            length=24,
                        ),
                        "src_semantic_id": interval["semantic_id"],
                        "dst_semantic_id": interval["register_id"],
                        "relation": "persistent_update_rule",
                        "evidence_strength": interval["observation"],
                        "dynamic_score": interval["dynamic_score"],
                    }
                )
            for predicate in predicate_nodes:
                predicate_members = {
                    re.sub(r"\s*\[\d+:\d+\]$", "", member)
                    for member in predicate["member_signals"]
                }
                for register in register_nodes:
                    if (
                        re.sub(
                            r"\s*\[\d+:\d+\]$",
                            "",
                            register["signal"],
                        )
                        not in predicate_members
                    ):
                        continue
                    graph_edges.append(
                        {
                            "edge_id": stable_id(
                                "vcse_",
                                identity,
                                register["semantic_id"],
                                predicate["semantic_id"],
                                "predicate_member",
                                length=24,
                            ),
                            "src_semantic_id": register["semantic_id"],
                            "dst_semantic_id": predicate["semantic_id"],
                            "relation": "predicate_member",
                            "evidence_strength": "exact",
                            "dynamic_score": 1.0,
                        }
                    )
        all_diagnostics = raw["diagnostics"]
        all_diagnostics.extend(interval_diagnostics)
        max_semantic_nodes_reached = (
            len(semantic_nodes) > request.bounds["max_semantic_nodes"]
        )
        if max_semantic_nodes_reached:
            semantic_nodes = semantic_nodes[
                : request.bounds["max_semantic_nodes"]
            ]
            retained_semantic_ids = {
                row["semantic_id"] for row in semantic_nodes
            }
            graph_edges = [
                row
                for row in graph_edges
                if "src_semantic_id" not in row
                or (
                    row["src_semantic_id"] in retained_semantic_ids
                    and row["dst_semantic_id"] in retained_semantic_ids
                )
            ]
            all_diagnostics.append(
                _diagnostic(
                    "graph_max_semantic_nodes_reached",
                    "semantic graph reached max_semantic_nodes",
                )
            )
        max_edges_reached = len(graph_edges) > request.bounds["max_edges"]
        if max_edges_reached:
            graph_edges = graph_edges[: request.bounds["max_edges"]]
            all_diagnostics.append(
                _diagnostic(
                    "graph_max_edges_reached",
                    "semantic graph reached max_edges",
                )
            )
        result = {
            "schema_version": SEMANTIC_GRAPH_SCHEMA,
            "graph_id": stable_id("vcsg_", identity),
            "status": (
                "incomplete"
                if any(row.get("breaks_complete") for row in all_diagnostics)
                else "complete"
            ),
            "identity": identity,
            "endpoint": {
                "signal": request.endpoint.signal,
                "cycle": request.endpoint.cycle,
                "projection_id": projection_id,
            },
            "signal_nodes": signal_nodes,
            "semantic_nodes": semantic_nodes,
            "edges": graph_edges,
            "root_candidates": [],
            "bounds": {
                **dict(request.bounds),
                "signal_nodes_reached": raw["bounds"]["max_nodes_reached"],
                "semantic_nodes_reached": max_semantic_nodes_reached,
                "edges_reached": max_edges_reached,
            },
            "diagnostics": sorted(
                all_diagnostics,
                key=lambda row: (row["code"], row.get("message", "")),
            ),
        }
        result["graph_id"] = stable_id(
            "vcsg_",
            identity,
            {
                "endpoint": result["endpoint"],
                "signal_nodes": result["signal_nodes"],
                "semantic_nodes": result["semantic_nodes"],
                "edges": result["edges"],
                "diagnostics": result["diagnostics"],
            },
        )
        return result

    def close(self) -> None:
        if not self._closed:
            self._prepared.close()
            self._closed = True

    def __enter__(self) -> "PreparedCausalSessionV3":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        self.close()
        return False


def prepare_causal_session_v3(
    request: CausalAnalysisRequestV3 | Mapping[str, Any],
    *,
    top_module: Optional[str] = None,
) -> PreparedCausalSessionV3:
    return PreparedCausalSessionV3(request, top_module=top_module)


def build_causal_graph_v3(
    request: CausalAnalysisRequestV3 | Mapping[str, Any],
    *,
    top_module: Optional[str] = None,
) -> Dict[str, Any]:
    with prepare_causal_session_v3(
        request, top_module=top_module
    ) as session:
        return session.build()
