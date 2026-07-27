"""C0-C5 execution for the explicit V3 Chisel profile."""

from __future__ import annotations

import re
from typing import Any, Dict, Mapping, Optional

from .causal_slicer import BackwardSlicer
from .chisel_semantics import (
    build_normalized_design,
    c2_enabled,
    c3_enabled,
    persistent_intervals,
)
from .chisel_protocol_semantics import (
    project_c3_waveform_scope,
    stall_intervals,
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
from .temporal_semantics import build_c4_temporal_layer, c4_enabled
from .waitfor_graph import (
    WaitForError,
    build_c5_waitfor_layer,
    c5_enabled,
    load_protocol_adapter,
)


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
            f"{ANALYZER_REVISION}+c5"
            if c5_enabled(request.semantic_profile.features)
            else (
                f"{ANALYZER_REVISION}+c4"
                if c4_enabled(request.semantic_profile.features)
                else (
                    f"{ANALYZER_REVISION}+c3"
                    if c3_enabled(request.semantic_profile.features)
                    else (
                        f"{ANALYZER_REVISION}+c2"
                        if c2_enabled(request.semantic_profile.features)
                        else f"{ANALYZER_REVISION}+c1"
                    )
                )
            )
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
            if (
                c2_enabled(request.semantic_profile.features)
                or c3_enabled(request.semantic_profile.features)
            )
            else None
        )
        if (
            self._normalized_design is not None
            and c3_enabled(request.semantic_profile.features)
        ):
            project_c3_waveform_scope(
                self._normalized_design,
                self.instance_graph,
                request.endpoint.signal,
                rtl_set_sha256=_identity(request)["rtl_set_sha256"],
            )
            if c4_enabled(request.semantic_profile.features):
                pipeline_paths = {
                    str(row["instance_path"])
                    for row in self._normalized_design["pipelines"]
                }
                pipeline_modules = {
                    row.module_name
                    for row in self.instance_graph.instances
                    if row.instance_path in pipeline_paths
                }
                projected_paths = {
                    str(row["instance_path"])
                    for row in self._normalized_design["pipelines"]
                }
                for scope, representative in (
                    self.instance_graph.exact_waveform_scopes_for_modules(
                        pipeline_modules
                    )
                ):
                    if scope in projected_paths:
                        continue
                    project_c3_waveform_scope(
                        self._normalized_design,
                        self.instance_graph,
                        representative,
                        rtl_set_sha256=_identity(request)["rtl_set_sha256"],
                    )
                    projected_paths.add(scope)
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
        raw_seed_signals = [request.endpoint.signal]
        if (
            self._normalized_design is not None
            and c4_enabled(request.semantic_profile.features)
        ):
            derived_seed_signals = sorted(
                {
                    str(member)
                    for transition in selected_transitions
                    for rule in list(transition["update_rules"])
                    + list(transition["reset_rules"])
                    for member in rule.get("guard_members", [])
                    if self._prepared.waveform.has_exact_signal(str(member))
                    and str(member) != request.endpoint.signal
                }
            )[: max(0, request.bounds["max_seed_count"] - 1)]
            raw_seed_signals.extend(derived_seed_signals)
            raw_nodes = {
                str(row["node_id"]): dict(row) for row in raw["nodes"]
            }
            raw_edges = {
                str(row["edge_id"]): dict(row) for row in raw["edges"]
            }
            for seed_signal in derived_seed_signals:
                seed_slicer = BackwardSlicer(
                    self._prepared.parser,
                    self._prepared.waveform,
                    max_depth=request.bounds["max_signal_nodes"],
                    max_nodes=request.bounds["max_signal_nodes"],
                    dependency_provider=provider,
                )
                seed_nodes, seed_edges = seed_slicer.slice_from_endpoint(
                    seed_signal, request.endpoint.cycle
                )
                seed_raw = _convert_graph(
                    self._v2,
                    (node.to_dict() for node in seed_nodes.values()),
                    (edge.to_dict() for edge in seed_edges),
                    dict(seed_slicer.get_statistics()),
                    [],
                    self._prepared.artifact_by_path,
                )
                for row in seed_raw["nodes"]:
                    raw_nodes.setdefault(str(row["node_id"]), dict(row))
                for row in seed_raw["edges"]:
                    raw_edges.setdefault(str(row["edge_id"]), dict(row))
                raw["diagnostics"].extend(seed_raw["diagnostics"])
                raw["bounds"]["max_depth_reached"] = bool(
                    raw["bounds"]["max_depth_reached"]
                    or seed_raw["bounds"]["max_depth_reached"]
                )
                raw["bounds"]["max_nodes_reached"] = bool(
                    raw["bounds"]["max_nodes_reached"]
                    or seed_raw["bounds"]["max_nodes_reached"]
                )
            retained_raw_ids = {
                str(row["node_id"])
                for row in sorted(
                    raw_nodes.values(),
                    key=lambda row: (
                        int(row.get("depth", 0)),
                        str(row["node_id"]),
                    ),
                )[: request.bounds["max_signal_nodes"]]
            }
            if len(raw_nodes) > len(retained_raw_ids):
                raw["bounds"]["max_nodes_reached"] = True
                raw["diagnostics"].append(
                    _diagnostic(
                        "graph_max_nodes_reached",
                        "C4 merged multi-seed graph reached max_signal_nodes",
                    )
                )
            raw["nodes"] = sorted(
                (
                    row
                    for node_id, row in raw_nodes.items()
                    if node_id in retained_raw_ids
                ),
                key=lambda row: str(row["node_id"]),
            )
            raw["edges"] = sorted(
                (
                    row
                    for row in raw_edges.values()
                    if str(row["src_node_id"]) in retained_raw_ids
                    and str(row["dst_node_id"]) in retained_raw_ids
                ),
                key=lambda row: str(row["edge_id"]),
            )
            raw["diagnostics"] = sorted(
                {
                    (
                        str(row["code"]),
                        str(row.get("message", "")),
                        bool(row.get("breaks_complete", False)),
                    ): row
                    for row in raw["diagnostics"]
                }.values(),
                key=lambda row: (row["code"], row.get("message", "")),
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
        if self._normalized_design is not None and not projection_members:
            raw_signal_names = {
                _SemanticBoundaryProvider._clean(str(row["signal"]))
                for row in signal_nodes
            }
            selected_transitions = [
                row
                for row in selected_transitions
                if _SemanticBoundaryProvider._clean(str(row["signal"]))
                in raw_signal_names
            ]
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
        c3_diagnostics: list[Dict[str, Any]] = []
        if (
            self._normalized_design is not None
            and c3_enabled(request.semantic_profile.features)
        ):
            clean = _SemanticBoundaryProvider._clean
            raw_signals = {
                clean(str(row["signal"])) for row in signal_nodes
            }
            raw_signals.update(projection_members)
            if c4_enabled(request.semantic_profile.features):
                raw_signals.update(
                    clean(str(member))
                    for transition in selected_transitions
                    for rule in list(transition["reset_rules"])
                    + list(transition["update_rules"])
                    for member in list(rule.get("guard_members", []))
                    + list(rule.get("value_members", []))
                )
            handshakes = [
                row
                for row in self._normalized_design["handshakes"]
                if bool(
                    raw_signals
                    & {clean(str(item)) for item in row["member_signals"]}
                )
            ]
            selected_aggregate_ids = {
                str(row["aggregate_id"]) for row in handshakes
            }
            pipelines = [
                row
                for row in self._normalized_design["pipelines"]
                if bool(
                    raw_signals
                    & {clean(str(item)) for item in row["member_signals"]}
                )
                or any(
                    str(stage["aggregate_id"]) in selected_aggregate_ids
                    for stage in row["stages"]
                )
            ]
            selected_pipeline_ids = {
                str(row["pipeline_id"]) for row in pipelines
            }
            blocking_relations = [
                row
                for row in self._normalized_design["blocking_relations"]
                if clean(str(row["target_signal"])) in raw_signals
                or bool(
                    raw_signals
                    & {clean(str(item)) for item in row["member_signals"]}
                )
                or any(
                    str(blocker["pipeline_id"]) in selected_pipeline_ids
                    for blocker in row["blockers"]
                )
                or (
                    c4_enabled(request.semantic_profile.features)
                    and row["inference_rule"].startswith(
                        "exact_waveform_scope_module_signature.v1+"
                    )
                    and self._prepared.waveform.has_exact_signal(
                        str(row["target_signal"])
                    )
                    and self._prepared.waveform.get_signal_value(
                        str(row["target_signal"]), request.endpoint.cycle
                    )
                    == "1"
                )
            ]
            selected_pipeline_ids.update(
                str(blocker["pipeline_id"])
                for relation in blocking_relations
                for blocker in relation["blockers"]
            )
            pipelines = [
                row
                for row in self._normalized_design["pipelines"]
                if row["pipeline_id"] in selected_pipeline_ids
            ]
            selected_aggregate_ids.update(
                str(stage["aggregate_id"])
                for pipeline in pipelines
                for stage in pipeline["stages"]
            )
            aggregates = [
                row
                for row in self._normalized_design["aggregates"]
                if row["aggregate_id"] in selected_aggregate_ids
                or bool(
                    raw_signals
                    & {clean(str(item)) for item in row["member_signals"]}
                )
            ]

            for row in aggregates:
                semantic_nodes.append(
                    {
                        **row,
                        "semantic_id": row["aggregate_id"],
                        "type": "aggregate",
                    }
                )
            for row in handshakes:
                semantic_nodes.append(
                    {
                        **row,
                        "semantic_id": row["handshake_id"],
                        "type": "handshake",
                    }
                )
            for row in pipelines:
                semantic_nodes.append(
                    {
                        **row,
                        "semantic_id": row["pipeline_id"],
                        "type": "pipeline",
                    }
                )
            for row in blocking_relations:
                semantic_nodes.append(
                    {
                        **row,
                        "semantic_id": row["blocking_id"],
                        "type": "blocking_relation",
                    }
                )

            stalls, c3_diagnostics = stall_intervals(
                handshakes,
                self._prepared.waveform,
                end_cycle=request.endpoint.cycle,
                max_intervals=request.bounds[
                    "max_intervals_per_signal"
                ],
                max_temporal_samples=request.bounds[
                    "max_temporal_samples"
                ],
            )
            semantic_nodes.extend(stalls)

            aggregate_ids = {
                row["aggregate_id"] for row in aggregates
            }
            pipeline_ids = {row["pipeline_id"] for row in pipelines}
            for handshake in handshakes:
                if handshake["aggregate_id"] not in aggregate_ids:
                    continue
                graph_edges.append(
                    {
                        "edge_id": stable_id(
                            "vcse_",
                            identity,
                            handshake["aggregate_id"],
                            handshake["handshake_id"],
                            "ready_valid_semantics",
                            length=24,
                        ),
                        "src_semantic_id": handshake["aggregate_id"],
                        "dst_semantic_id": handshake["handshake_id"],
                        "relation": "ready_valid_semantics",
                        "evidence_strength": "exact",
                        "dynamic_score": 1.0,
                    }
                )
            for pipeline in pipelines:
                for stage in pipeline["stages"]:
                    if stage["aggregate_id"] not in aggregate_ids:
                        continue
                    graph_edges.append(
                        {
                            "edge_id": stable_id(
                                "vcse_",
                                identity,
                                stage["aggregate_id"],
                                pipeline["pipeline_id"],
                                "pipeline_stage",
                                length=24,
                            ),
                            "src_semantic_id": stage["aggregate_id"],
                            "dst_semantic_id": pipeline["pipeline_id"],
                            "relation": "pipeline_stage",
                            "evidence_strength": "exact",
                            "dynamic_score": 1.0,
                        }
                    )
            for relation in blocking_relations:
                for blocker in relation["blockers"]:
                    if blocker["pipeline_id"] not in pipeline_ids:
                        continue
                    graph_edges.append(
                        {
                            "edge_id": stable_id(
                                "vcse_",
                                identity,
                                blocker["pipeline_id"],
                                relation["blocking_id"],
                                blocker["stage_index"],
                                "pipeline_blocker",
                                length=24,
                            ),
                            "src_semantic_id": blocker["pipeline_id"],
                            "dst_semantic_id": relation["blocking_id"],
                            "relation": "pipeline_blocker",
                            "stage_index": blocker["stage_index"],
                            "evidence_strength": "exact",
                            "dynamic_score": 1.0,
                        }
                    )
            for stall in stalls:
                graph_edges.append(
                    {
                        "edge_id": stable_id(
                            "vcse_",
                            identity,
                            stall["semantic_id"],
                            stall["handshake_id"],
                            "persistent_stall",
                            length=24,
                        ),
                        "src_semantic_id": stall["semantic_id"],
                        "dst_semantic_id": stall["handshake_id"],
                        "relation": "persistent_stall",
                        "evidence_strength": stall["evidence_strength"],
                        "dynamic_score": 1.0,
                    }
                )

            raw_by_signal: Dict[str, list[str]] = {}
            for node in signal_nodes:
                raw_by_signal.setdefault(
                    clean(str(node["signal"])), []
                ).append(str(node["node_id"]))
            for semantic in semantic_nodes:
                for member in sorted(
                    {
                        clean(str(item))
                        for item in semantic.get("member_signals", [])
                    }
                ):
                    for raw_node_id in sorted(raw_by_signal.get(member, [])):
                        graph_edges.append(
                            {
                                "edge_id": stable_id(
                                    "vcse_",
                                    identity,
                                    raw_node_id,
                                    semantic["semantic_id"],
                                    "semantic_member",
                                    length=24,
                                ),
                                "src_node_id": raw_node_id,
                                "dst_semantic_id": semantic[
                                    "semantic_id"
                                ],
                                "relation": "semantic_member",
                                "evidence_strength": "exact",
                                "dynamic_score": 1.0,
                            }
                        )
        temporal_diagnostics: list[Dict[str, Any]] = []
        root_candidates: list[Dict[str, Any]] = []
        temporal_work: Dict[str, Any] = {}
        if (
            self._normalized_design is not None
            and c4_enabled(request.semantic_profile.features)
        ):
            (
                semantic_nodes,
                graph_edges,
                root_candidates,
                temporal_diagnostics,
                temporal_work,
            ) = build_c4_temporal_layer(
                normalized_design=self._normalized_design,
                waveform=self._prepared.waveform,
                endpoint_cycle=request.endpoint.cycle,
                semantic_nodes=semantic_nodes,
                edges=graph_edges,
                max_seed_count=request.bounds["max_seed_count"],
                max_transition_values=request.bounds["max_temporal_samples"],
            )
            temporal_work["raw_seed_count"] = len(raw_seed_signals)
            temporal_work["raw_seed_signals"] = raw_seed_signals
        waitfor_diagnostics: list[Dict[str, Any]] = []
        waitfor_work: Dict[str, Any] = {}
        if c5_enabled(request.semantic_profile.features):
            protocol_adapter = None
            adapter_artifacts = [
                item
                for item in request.semantic_inputs
                if item.kind == "reviewed_protocol_adapter"
            ]
            if adapter_artifacts:
                adapter_artifact = adapter_artifacts[0]
                try:
                    protocol_adapter = load_protocol_adapter(
                        adapter_artifact.path,
                        sha256=adapter_artifact.sha256,
                        bytes=adapter_artifact.bytes,
                        rtl_set_sha256=_identity(request)["rtl_set_sha256"],
                        known_semantic_ids={
                            str(row["semantic_id"])
                            for row in semantic_nodes
                        },
                    )
                except WaitForError as error:
                    waitfor_diagnostics.append(
                        {
                            "code": "protocol_adapter_invalid",
                            "message": str(error),
                            "breaks_complete": True,
                        }
                    )
            if not waitfor_diagnostics:
                (
                    semantic_nodes,
                    graph_edges,
                    root_candidates,
                    waitfor_diagnostics,
                    waitfor_work,
                ) = build_c5_waitfor_layer(
                    semantic_nodes=semantic_nodes,
                    edges=graph_edges,
                    root_candidates=root_candidates,
                    endpoint_cycle=request.endpoint.cycle,
                    max_waitfor_nodes=request.bounds[
                        "max_waitfor_nodes"
                    ],
                    max_waitfor_edges=request.bounds[
                        "max_waitfor_edges"
                    ],
                    max_scc_candidates=request.bounds[
                        "max_scc_candidates"
                    ],
                    protocol_adapter=protocol_adapter,
                    rtl_set_sha256=_identity(request)["rtl_set_sha256"],
                    max_total_edges=request.bounds["max_edges"],
                )
        semantic_nodes.sort(key=lambda row: row["semantic_id"])
        all_diagnostics = raw["diagnostics"]
        all_diagnostics.extend(interval_diagnostics)
        all_diagnostics.extend(c3_diagnostics)
        all_diagnostics.extend(temporal_diagnostics)
        all_diagnostics.extend(waitfor_diagnostics)
        max_semantic_nodes_reached = (
            len(semantic_nodes) > request.bounds["max_semantic_nodes"]
        )
        if max_semantic_nodes_reached:
            ranked_semantic_ids: Dict[str, int] = {}
            for rank, candidate in enumerate(root_candidates):
                for semantic_id in candidate["semantic_path"]:
                    ranked_semantic_ids.setdefault(str(semantic_id), rank)
            essential_types = {
                "assertion_predicate",
                "register_transition",
                "persistent_interval",
                "threshold_crossing",
                "missing_expected_completion",
                "last_progress_event",
                "stall_interval",
                "unknown_external_completion",
                "protocol_transaction",
                "resource_wait",
                "waitfor_component",
                "waitfor_scc",
            }
            semantic_nodes = sorted(
                semantic_nodes,
                key=lambda row: (
                    0
                    if str(row["semantic_id"]) in ranked_semantic_ids
                    else (
                        1 if row["type"] in essential_types else 2
                    ),
                    ranked_semantic_ids.get(str(row["semantic_id"]), 10**9),
                    str(row["semantic_id"]),
                ),
            )[: request.bounds["max_semantic_nodes"]]
            semantic_nodes.sort(key=lambda row: row["semantic_id"])
            retained_semantic_ids = {
                row["semantic_id"] for row in semantic_nodes
            }
            graph_edges = [
                row
                for row in graph_edges
                if (
                    (
                        "src_semantic_id" not in row
                        or row["src_semantic_id"] in retained_semantic_ids
                    )
                    and (
                        "dst_semantic_id" not in row
                        or row["dst_semantic_id"] in retained_semantic_ids
                    )
                )
            ]
            root_candidates = [
                row
                for row in root_candidates
                if row["semantic_id"] in retained_semantic_ids
                and all(
                    semantic_id in retained_semantic_ids
                    for semantic_id in row["semantic_path"]
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
        result_bounds = {
            **dict(request.bounds),
            "signal_nodes_reached": raw["bounds"]["max_nodes_reached"],
            "semantic_nodes_reached": max_semantic_nodes_reached,
            "edges_reached": max_edges_reached,
            "temporal_work": temporal_work,
        }
        if c5_enabled(request.semantic_profile.features):
            result_bounds["waitfor_work"] = waitfor_work
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
            "root_candidates": root_candidates,
            "bounds": result_bounds,
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
                "root_candidates": result["root_candidates"],
                "bounds": result["bounds"],
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
