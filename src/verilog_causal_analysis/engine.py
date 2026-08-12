"""C0-C6 execution for the explicit semantic Chisel profile."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
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
from .structural_contract import make_structural_request
from .contracts import (
    CausalAnalysisRequest,
    ContractError,
    GRAPH_SCHEMA,
    VERILOG_PROFILE,
)
from .endpoint_projection import (
    EndpointProjectionError,
    ProjectedDependencyProvider,
    load_assertion_projection,
)
from .structural_engine import PreparedCausalAnalysis, _convert_graph, _diagnostic
from .identity import (
    ANALYZER_REVISION,
    canonical_sha256,
    sha256_file,
    stable_id,
    stable_set_sha256,
)
from .local_search import SearchSeed, make_search_summary
from .instance_graph import InstanceGraph, InstanceGraphError
from .temporal_semantics import build_c4_temporal_layer, c4_enabled
from .waitfor_graph import (
    WaitForError,
    build_c5_waitfor_layer,
    c5_enabled,
    load_protocol_adapter,
)
from .provenance import (
    ProvenanceError,
    build_heuristic_feature_index,
    build_provenance_hints,
    c6_enabled,
    load_source_annotations,
)
from .verilog_parser import DependencyType, VerilogParser

_MAX_RAW_SLICE_DEPTH = 256


def _clean_signal(signal: str) -> str:
    return re.sub(r"\s*\[\d+:\d+\]$", "", signal)


def _causal_provenance_hints(
    hints: list[Dict[str, Any]],
    parser: Any,
    raw_edges: list[Mapping[str, Any]],
    selected_statement_ids: set[str],
) -> list[Dict[str, Any]]:
    """Keep source hints attached to the active slice before budgeting."""

    causal_lines: Dict[str, list[tuple[int, int]]] = {}
    for edge in raw_edges:
        evidence = edge.get("rtl_evidence")
        if not isinstance(evidence, Mapping):
            continue
        artifact_id = str(evidence.get("artifact_id", ""))
        start = int(evidence.get("line_start", 0) or 0)
        end = int(evidence.get("line_end", start) or start)
        if artifact_id and start > 0:
            causal_lines.setdefault(artifact_id, []).append((start, max(start, end)))

    result = []
    statements = parser._statement_evidence
    for hint in hints:
        statement_id = str(hint["rtl_statement_id"])
        if statement_id in selected_statement_ids:
            result.append(hint)
            continue
        statement = statements.get(statement_id)
        if statement is None:
            continue
        ranges = causal_lines.get(str(hint.get("rtl_artifact_id", "")), ())
        if any(
            statement.line_start <= end and statement.line_end >= start
            for start, end in ranges
        ):
            result.append(hint)
    return result


def _structural_request(request: CausalAnalysisRequest):
    return make_structural_request(
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
        search_policy=request.search_policy.to_dict(),
        max_depth=min(request.bounds["max_signal_depth"], _MAX_RAW_SLICE_DEPTH),
        max_nodes=request.bounds["max_signal_nodes"],
        max_expanded_nodes=request.bounds["max_expanded_nodes"],
        max_candidate_evaluations=request.bounds["max_candidate_evaluations"],
        max_intervention_evaluations=request.bounds["max_intervention_evaluations"],
        random_seed=request.random_seed,
        strict=request.strict,
    )


def _identity(request: CausalAnalysisRequest) -> Dict[str, Any]:
    return {
        "request_sha256": request.request_sha256,
        "rtl_set_sha256": stable_set_sha256(
            [item.identity_dict() for item in request.rtl_files]
        ),
        "trace_sha256": request.trace.sha256,
        "analyzer_revision": (
            f"{ANALYZER_REVISION}+c6"
            if c6_enabled(request.semantic_profile.features)
            else (
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
            )
        ),
        "profile_version": request.semantic_profile.version,
    }


def _session_identity(request: CausalAnalysisRequest) -> Dict[str, Any]:
    """Identity of parser/waveform/normalization state reusable by endpoints."""

    row = request.identity_dict()
    row.pop("endpoint")
    row["semantic_inputs"] = [
        item
        for item in row["semantic_inputs"]
        if item.get("kind") != "assertion_endpoint_projection"
    ]
    return row


def _empty_graph(
    request: CausalAnalysisRequest,
    diagnostics: list[dict[str, Any]],
) -> Dict[str, Any]:
    identity = _identity(request)
    return {
        "schema_version": GRAPH_SCHEMA,
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
        "search_summary": make_search_summary(
            request.search_policy,
            termination_reason="frontier_exhausted",
            seed_count=1,
        ),
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


class PreparedCausalSession:
    """Verified semantic session with reusable parser, waveform, and instance graph."""

    def __init__(
        self,
        request: CausalAnalysisRequest | Mapping[str, Any],
        *,
        top_module: Optional[str] = None,
    ):
        if not isinstance(request, CausalAnalysisRequest):
            request = CausalAnalysisRequest.from_dict(request)
        self.request = request
        self._structural = _structural_request(request)
        self._prepared = PreparedCausalAnalysis(self._structural)
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
                or c6_enabled(request.semantic_profile.features)
            )
            else None
        )
        self._provenance_diagnostics: list[Dict[str, Any]] = []
        if (
            self._normalized_design is not None
            and c6_enabled(request.semantic_profile.features)
        ):
            annotation = None
            annotation_artifacts = [
                item
                for item in request.semantic_inputs
                if item.kind == "chisel_source_annotations"
            ]
            try:
                if annotation_artifacts:
                    artifact = annotation_artifacts[0]
                    annotation = load_source_annotations(
                        artifact.path,
                        sha256=artifact.sha256,
                        bytes=artifact.bytes,
                        rtl_set_sha256=_identity(request)["rtl_set_sha256"],
                        known_statement_ids=set(
                            self._prepared.parser._statement_evidence
                        ),
                    )
                hints, self._provenance_diagnostics = build_provenance_hints(
                    self.instance_graph,
                    rtl_set_sha256=_identity(request)["rtl_set_sha256"],
                    annotations=annotation,
                )
                self._normalized_design["provenance_hints"] = hints
                self._normalized_design["diagnostics"].extend(
                    self._provenance_diagnostics
                )
                self._normalized_design["normalized_design_id"] = stable_id(
                    "vcnd_",
                    {
                        key: value
                        for key, value in self._normalized_design.items()
                        if key != "normalized_design_id"
                    },
                )
            except ProvenanceError as error:
                self._provenance_diagnostics = [
                    {
                        "code": "source_annotation_invalid",
                        "message": str(error),
                        "breaks_complete": True,
                    }
                ]
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
        self.search_trace: list[Dict[str, Any]] = []
        self._heuristic_features = build_heuristic_feature_index(
            self._normalized_design
        )

    @property
    def normalized_design(self) -> Dict[str, Any]:
        if self._normalized_design is not None:
            return self._normalized_design
        return self.instance_graph.to_dict()

    def build(
        self,
        request: CausalAnalysisRequest | Mapping[str, Any] | None = None,
    ) -> Dict[str, Any]:
        if self._closed:
            raise RuntimeError("prepared semantic session is closed")
        if request is None:
            request = self.request
        elif not isinstance(request, CausalAnalysisRequest):
            request = CausalAnalysisRequest.from_dict(request)
        if _session_identity(request) != _session_identity(self.request):
            raise ValueError("request does not match prepared semantic session inputs")
        current_structural = _structural_request(request)

        diagnostics = [
            _diagnostic(
                row["code"],
                row["message"],
            )
            for row in self.instance_graph.diagnostics
        ]
        diagnostics.extend(
            _diagnostic(row["code"], row["message"])
            for row in self._provenance_diagnostics
        )
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

        raw_seed_signals = [request.endpoint.signal]
        seeds = [
            SearchSeed(
                seed_id=stable_id(
                    "vcss_", request.endpoint.signal, request.endpoint.cycle,
                    "exact_endpoint", length=24,
                ),
                signal=request.endpoint.signal,
                cycle=request.endpoint.cycle,
                seed_kind="exact_endpoint",
                seed_prior=1.0,
                seed_rank=0,
            )
        ]
        extra_seeds: list[tuple[str, str, float]] = []
        extra_seeds.extend(
            (member, "exact_predicate_member", 0.9)
            for member in sorted(projection_members)
            if member != request.endpoint.signal
            and self._prepared.waveform.has_exact_signal(member)
        )
        if self._normalized_design is not None and c4_enabled(
            request.semantic_profile.features
        ):
            extra_seeds.extend(
                (member, "derived_active_guard", 0.75)
                for member in sorted(
                    {
                        str(member)
                        for transition in selected_transitions
                        for rule in list(transition["update_rules"])
                        + list(transition["reset_rules"])
                        for member in rule.get("guard_members", [])
                        if self._prepared.waveform.has_exact_signal(str(member))
                    }
                )
            )
        seen_seed_signals = {request.endpoint.signal}
        for signal, kind, prior in extra_seeds:
            if signal in seen_seed_signals or len(seeds) >= request.bounds["max_seed_count"]:
                continue
            seen_seed_signals.add(signal)
            raw_seed_signals.append(signal)
            seeds.append(
                SearchSeed(
                    seed_id=stable_id(
                        "vcss_", signal, request.endpoint.cycle, kind, length=24
                    ),
                    signal=signal,
                    cycle=request.endpoint.cycle,
                    seed_kind=kind,
                    seed_prior=prior,
                    seed_rank=len(seeds),
                )
            )

        slicer = BackwardSlicer(
            self._prepared.parser,
            self._prepared.waveform,
            max_depth=min(
                request.bounds["max_signal_depth"], _MAX_RAW_SLICE_DEPTH
            ),
            max_nodes=request.bounds["max_signal_nodes"],
            dependency_provider=provider,
            search_policy=request.search_policy.policy_id,
            heuristic_context=self._heuristic_features,
            max_expanded_nodes=request.bounds["max_expanded_nodes"],
            max_candidate_evaluations=request.bounds["max_candidate_evaluations"],
            max_intervention_evaluations=request.bounds["max_intervention_evaluations"],
        )
        nodes, edges = slicer.slice_from_seeds(seeds)
        stats = dict(slicer.get_statistics())
        raw_search_trace = slicer.get_search_trace()
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
        termination_reason = str(stats.get("termination_reason", "frontier_exhausted"))
        if termination_reason != "frontier_exhausted":
            diagnostics.append(
                _diagnostic(
                    "search_termination_incomplete",
                    f"local search terminated at {termination_reason}",
                )
            )
        raw = _convert_graph(
            current_structural,
            (node.to_dict() for node in nodes.values()),
            (edge.to_dict() for edge in edges),
            stats,
            diagnostics,
            self._prepared.artifact_by_path,
        )
        raw["search_summary"] = make_search_summary(
            request.search_policy,
            termination_reason=(
                "source_projection_ambiguous"
                if any(
                    row.get("code") == "source_projection_ambiguous"
                    for row in diagnostics
                )
                else str(stats["termination_reason"])
            ),
            seed_count=len(raw_seed_signals),
            expanded_nodes=int(stats["expanded_nodes"]),
            candidate_evaluations=int(stats["candidate_evaluations"]),
            intervention_evaluations=int(stats["intervention_evaluations"]),
            admitted_nodes=len(raw["nodes"]),
            admitted_edges=len(raw["edges"]),
            exploit_expansions=int(stats["exploit_expansions"]),
            explore_expansions=int(stats["explore_expansions"]),
            frontier_ids=stats.get("frontier_ids", ()),
        )
        identity = _identity(request)
        node_ids: Dict[str, str] = {}
        slicer_node_ids: Dict[str, str] = {}
        signal_nodes = []
        for slicer_node, node in zip(nodes.values(), raw["nodes"]):
            new_id = stable_id(
                "vcn3_",
                identity,
                node["signal"],
                node["cycle"],
                node["value"],
                length=24,
            )
            node_ids[node["node_id"]] = new_id
            slicer_node_ids[slicer_node.id] = new_id
            signal_nodes.append({**node, "node_id": new_id})
        if self._normalized_design is not None and not projection_members:
            raw_signal_names = {
                _clean_signal(str(row["signal"]))
                for row in signal_nodes
            }
            selected_transitions = [
                row
                for row in selected_transitions
                if _clean_signal(str(row["signal"]))
                in raw_signal_names
            ]
        graph_edges = []
        edge_ids: Dict[str, str] = {}
        for slicer_edge, edge in zip(edges, raw["edges"]):
            src = node_ids[edge["src_node_id"]]
            dst = node_ids[edge["dst_node_id"]]
            rtl_evidence = dict(edge["rtl_evidence"])
            if request.semantic_profile.version == VERILOG_PROFILE:
                rtl_evidence["statement_id"] = slicer_edge.evidence.get(
                    "statement_id"
                )
            edge_id = stable_id(
                "vce3_",
                identity,
                src,
                dst,
                edge["dependency_type"],
                rtl_evidence,
                length=24,
            )
            trace_edge_id = hashlib.md5(
                f"{slicer_edge.src_node_id}->{slicer_edge.dst_node_id}".encode()
            ).hexdigest()[:12]
            edge_ids[trace_edge_id] = edge_id
            graph_edges.append(
                {
                    **edge,
                    "edge_id": edge_id,
                    "src_node_id": src,
                    "dst_node_id": dst,
                    "rtl_evidence": rtl_evidence,
                }
            )
        self.search_trace = []
        for raw_event in raw_search_trace:
            event = dict(raw_event)
            for field in ("node_id", "parent_node_id", "target_node_id"):
                if field in event:
                    event[field] = slicer_node_ids.get(event[field], event[field])
            if "edge_id" in event:
                event["edge_id"] = edge_ids.get(event["edge_id"], event["edge_id"])
            self.search_trace.append(event)
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
            if c6_enabled(request.semantic_profile.features):
                selected_statement_ids = {
                    str(statement_id)
                    for transition in selected_transitions
                    for statement_id in transition["statement_ids"]
                }
                hints = _causal_provenance_hints(
                    self._normalized_design["provenance_hints"],
                    self._prepared.parser,
                    raw["edges"],
                    selected_statement_ids,
                )
                for hint in hints:
                    semantic_nodes.append(
                        {
                            **hint,
                            "semantic_id": hint["hint_id"],
                            "type": "source_provenance_hint",
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
            clean = _clean_signal
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
                        "exact_waveform_scope_module_signature+"
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
            selected_aggregate_ids.update(
                str(row["aggregate_id"]) for row in handshakes
            )
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
        all_diagnostics = list(raw["diagnostics"])
        all_diagnostics.extend(interval_diagnostics)
        all_diagnostics.extend(c3_diagnostics)
        all_diagnostics.extend(temporal_diagnostics)
        all_diagnostics.extend(waitfor_diagnostics)
        all_diagnostics = _classify_internal_waveform_frontiers(
            all_diagnostics
        )
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
        all_diagnostics = _aggregate_diagnostics(all_diagnostics)
        result_bounds["waveform_frontiers"] = {
            "exact_instance_missing": sum(
                row["code"] == "waveform_exact_instance_missing"
                for row in all_diagnostics
            ),
            "unknown_value": sum(
                row["code"] == "waveform_value_unknown"
                for row in all_diagnostics
            ),
        }
        result = {
            "schema_version": GRAPH_SCHEMA,
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
            "search_summary": raw["search_summary"],
            "bounds": result_bounds,
            "diagnostics": all_diagnostics,
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
                "search_summary": result["search_summary"],
                "bounds": result["bounds"],
                "diagnostics": result["diagnostics"],
            },
        )
        return result

    def close(self) -> None:
        if not self._closed:
            self._prepared.close()
            self._closed = True

    def __enter__(self) -> "PreparedCausalSession":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        self.close()
        return False


def prepare_causal_session(
    request: CausalAnalysisRequest | Mapping[str, Any],
    *,
    top_module: Optional[str] = None,
) -> PreparedCausalSession:
    return PreparedCausalSession(request, top_module=top_module)


def build_rtl_candidates(
    request: CausalAnalysisRequest | Mapping[str, Any],
) -> Dict[str, Any]:
    """Build the exact executable statement universe for native Verilog RTL."""

    if not isinstance(request, CausalAnalysisRequest):
        request = CausalAnalysisRequest.from_dict(request)
    if request.semantic_profile.version != VERILOG_PROFILE:
        raise ContractError("RTL candidates require the verilog profile")

    artifact_by_path = {}
    for artifact in request.rtl_files:
        if sha256_file(artifact.path) != (artifact.sha256, artifact.bytes):
            raise ContractError(
                f"RTL bytes or SHA-256 differ for {artifact.artifact_id}"
            )
        artifact_by_path[str(Path(artifact.path).resolve())] = artifact.artifact_id

    parser = VerilogParser(strict=True)
    parser.parse_files_strict(
        item.path for item in sorted(request.rtl_files, key=lambda row: row.artifact_id)
    )
    records = {
        row.statement_id: row.is_sequential
        for module in parser.modules.values()
        for row in module.assignment_records
    }
    dependency_types: Dict[str, set[DependencyType]] = {}
    for dependency in parser.all_dependencies:
        dependency_types.setdefault(dependency.statement_id, set()).add(
            dependency.dep_type
        )

    candidates = []
    for statement_id, statement in sorted(parser._statement_evidence.items()):
        types = dependency_types.get(statement_id, set())
        if statement_id not in records and types == {DependencyType.ASSERTION}:
            continue
        try:
            artifact_id = artifact_by_path[str(Path(statement.file_path).resolve())]
        except KeyError as error:
            raise ContractError("statement references an unbound RTL artifact") from error
        if statement.line_start <= 0 or not statement.code_snippet:
            raise ContractError(
                f"statement {statement_id} lacks exact source identity"
            )
        kind = (
            "port_binding"
            if types & {DependencyType.PORT_INPUT, DependencyType.PORT_OUTPUT}
            else "register_update"
            if records.get(statement_id)
            else "assignment"
        )
        candidates.append(
            {
                "artifact_id": artifact_id,
                "statement_id": statement_id,
                "line_start": statement.line_start,
                "line_end": statement.line_end,
                "statement_kind": kind,
                "executable": True,
                "snippet_sha256": canonical_sha256(statement.code_snippet),
            }
        )
    return {
        "schema_version": "rtl_candidate_universe.v1",
        "rtl_set_sha256": _identity(request)["rtl_set_sha256"],
        "candidates": sorted(
            candidates,
            key=lambda row: (row["artifact_id"], row["statement_id"]),
        ),
    }


def _classify_internal_waveform_frontiers(
    diagnostics: list[Mapping[str, Any]],
) -> list[Dict[str, Any]]:
    """Keep internal waveform gaps explicit without failing the computation.

    Endpoint identity and value are hard-gated before slicing. Internal X/Z
    or absent exact-instance values terminate raw recursion and retain only
    weak structural evidence, so they are coverage frontiers rather than an
    unfinished analyzer operation.
    """

    result = []
    for diagnostic in diagnostics:
        row = dict(diagnostic)
        if row.get("code") in {
            "waveform_exact_instance_missing",
            "waveform_value_unknown",
        }:
            row.update(
                {
                    "severity": "warning",
                    "breaks_complete": False,
                    "frontier": True,
                }
            )
        result.append(row)
    return result


def _aggregate_diagnostics(
    diagnostics: list[Mapping[str, Any]],
) -> list[Dict[str, Any]]:
    """Canonicalize repeated cycle diagnostics by semantic identity."""

    grouped: Dict[tuple[Any, ...], Dict[str, Any]] = {}
    counts: Dict[tuple[Any, ...], int] = {}
    for raw in diagnostics:
        row = dict(raw)
        code = str(row.get("code"))
        message = str(row.get("message", ""))
        if code == "dependency_identity_ambiguous":
            match = re.fullmatch(
                r"signal (.+) at cycle [0-9]+ has ([0-9]+) waveform candidates",
                message,
            )
            if match:
                key = (code, match.group(1), int(match.group(2)))
                row["message"] = (
                    f"signal {match.group(1)} has {match.group(2)} waveform "
                    "candidates across sampled cycles"
                )
            else:
                key = (code, message)
        elif code == "waveform_value_unknown":
            match = re.fullmatch(
                r"unknown value for signal (.+) at cycle [0-9]+", message
            )
            key = (code, match.group(1) if match else message)
            if match:
                row["message"] = (
                    f"unknown values observed for signal {match.group(1)}"
                )
        else:
            key = (
                code,
                message,
                row.get("artifact_id"),
                bool(row.get("breaks_complete")),
            )
        grouped.setdefault(key, row)
        counts[key] = counts.get(key, 0) + 1
    result = []
    for key in sorted(grouped, key=lambda item: tuple(map(str, item))):
        row = grouped[key]
        if counts[key] > 1:
            row["occurrence_count"] = counts[key]
        result.append(row)
    return result


def build_causal_graph(
    request: CausalAnalysisRequest | Mapping[str, Any],
    *,
    top_module: Optional[str] = None,
) -> Dict[str, Any]:
    with prepare_causal_session(
        request, top_module=top_module
    ) as session:
        return session.build()
