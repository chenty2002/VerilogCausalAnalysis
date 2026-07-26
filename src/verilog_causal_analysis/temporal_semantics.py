"""Deterministic C4 temporal events, seed derivation, and path ranking."""

from __future__ import annotations

from collections import deque
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

from .causal_slicer import parse_binary_value
from .identity import canonical_sha256, stable_id


TEMPORAL_FEATURE = "temporal_interval"
_CLEAN_SUFFIX = " "
_TYPE_PRIORITY = {
    "missing_expected_completion": 0,
    "blocking_relation": 1,
    "stall_interval": 2,
    "persistent_interval": 3,
    "register_transition": 4,
    "threshold_crossing": 5,
    "last_progress_event": 6,
}
_SEED_PRIORITY = {
    "predicate_counter_operand.v1": 0,
    "threshold_crossing.v1": 1,
    "persistent_active_guard.v1": 2,
    "missing_expected_completion.v1": 3,
    "overlapping_stall.v1": 4,
    "persistent_pipeline_blocker.v1": 5,
    "last_progress_predecessor.v1": 6,
}


def c4_enabled(features: Iterable[str]) -> bool:
    return TEMPORAL_FEATURE in set(features)


def _clean(signal: str) -> str:
    head, separator, tail = signal.rpartition(_CLEAN_SUFFIX)
    if (
        separator
        and tail.startswith("[")
        and tail.endswith("]")
        and ":" in tail
    ):
        return head
    return signal


def _waveform_signal(waveform: Any, signal: str) -> Optional[str]:
    if not hasattr(waveform, "signals"):
        return signal
    if signal in waveform.signals.by_name:
        return signal
    clean = _clean(signal)
    matches = [
        str(item)
        for item in waveform.signals.by_name
        if _clean(str(item)) == clean
    ]
    return matches[0] if len(matches) == 1 else None


def _transition_series(
    waveform: Any,
    signal: str,
    *,
    start_cycle: int,
    end_cycle: int,
    max_values: int,
) -> Dict[str, Any]:
    """Use the prepared transition index, failing closed on unavailable data."""
    observed = _waveform_signal(waveform, signal)
    if observed is None:
        return {
            "signal": signal,
            "waveform_signal": None,
            "available": False,
            "truncated": False,
            "changes": [],
            "boundary_values": {"start": None, "end": None},
            "unknown_spans": [[start_cycle, end_cycle]],
            "work": {"transition_values": 0, "value_misses": 2},
        }
    if hasattr(waveform, "get_transition_series_bounded"):
        return waveform.get_transition_series_bounded(
            observed,
            start_cycle,
            end_cycle,
            max_values=max_values,
        )
    changes = waveform.get_value_changes_bounded(
        observed,
        start_cycle,
        end_cycle,
        max_changes=max_values + 1,
    )
    start_value = waveform.get_signal_value(observed, start_cycle)
    end_value = waveform.get_signal_value(observed, end_cycle)
    if changes is None:
        return {
            "signal": signal,
            "waveform_signal": observed,
            "available": False,
            "truncated": False,
            "changes": [],
            "boundary_values": {"start": start_value, "end": end_value},
            "unknown_spans": [[start_cycle, end_cycle]],
            "work": {
                "transition_values": 0,
                "value_misses": int(start_value is None) + int(end_value is None),
            },
        }
    truncated = len(changes) > max_values
    bounded = changes[:max_values]
    unknown_spans = (
        [[bounded[-1][0] if bounded else start_cycle, end_cycle]]
        if truncated
        else []
    )
    return {
        "signal": signal,
        "waveform_signal": observed,
        "available": True,
        "truncated": truncated,
        "changes": [
            {"cycle": cycle, "old": old, "new": new}
            for cycle, old, new in bounded
        ],
        "boundary_values": {"start": start_value, "end": end_value},
        "unknown_spans": unknown_spans,
        "work": {
            "transition_values": len(bounded),
            "value_misses": int(start_value is None) + int(end_value is None),
        },
    }


def build_transition_intervals(
    waveform: Any,
    signal: str,
    *,
    start_cycle: int,
    end_cycle: int,
    max_transition_values: int,
) -> Dict[str, Any]:
    """Return exact constant-value intervals from a bounded transition series."""
    series = _transition_series(
        waveform,
        signal,
        start_cycle=start_cycle,
        end_cycle=end_cycle,
        max_values=max_transition_values,
    )
    intervals: list[Dict[str, Any]] = []
    current_start = start_cycle
    current_value = series["boundary_values"]["start"]
    for change in series["changes"]:
        cycle = int(change["cycle"])
        if cycle > current_start:
            intervals.append(
                {
                    "start_cycle": current_start,
                    "end_cycle": cycle - 1,
                    "value": current_value,
                    "coverage": (
                        "exact"
                        if parse_binary_value(current_value) is not None
                        else "unknown"
                    ),
                }
            )
        current_start = cycle
        current_value = change["new"]
    if not series["truncated"] and current_start <= end_cycle:
        intervals.append(
            {
                "start_cycle": current_start,
                "end_cycle": end_cycle,
                "value": current_value,
                "coverage": (
                    "exact"
                    if parse_binary_value(current_value) is not None
                    else "unknown"
                ),
            }
        )
    unknown_spans = [
        [int(row["start_cycle"]), int(row["end_cycle"])]
        for row in intervals
        if row["coverage"] == "unknown"
    ]
    unknown_spans.extend(
        list(row) for row in series["unknown_spans"]
    )
    return {
        **series,
        "unknown_spans": [
            list(row)
            for row in sorted({tuple(row) for row in unknown_spans})
        ],
        "intervals": intervals,
    }


def _edge_endpoints(edge: Mapping[str, Any]) -> Tuple[Optional[str], Optional[str]]:
    source = edge.get("src_semantic_id")
    target = edge.get("dst_semantic_id")
    return (
        str(source) if source is not None else None,
        str(target) if target is not None else None,
    )


def _semantic_path(
    start_id: str,
    targets: set[str],
    adjacency: Mapping[str, set[str]],
    *,
    max_length: int = 8,
) -> Optional[list[str]]:
    queue = deque([(start_id, [start_id])])
    visited = {start_id}
    while queue:
        current, path = queue.popleft()
        if current in targets and current != start_id:
            return path
        if len(path) - 1 >= max_length:
            continue
        for neighbor in sorted(adjacency.get(current, ())):
            if neighbor in visited:
                continue
            visited.add(neighbor)
            queue.append((neighbor, path + [neighbor]))
    return None


def build_c4_temporal_layer(
    *,
    normalized_design: Mapping[str, Any],
    waveform: Any,
    endpoint_cycle: int,
    semantic_nodes: Sequence[Mapping[str, Any]],
    edges: Sequence[Mapping[str, Any]],
    max_seed_count: int,
    max_transition_values: int,
) -> Tuple[
    list[Dict[str, Any]],
    list[Dict[str, Any]],
    list[Dict[str, Any]],
    list[Dict[str, Any]],
    Dict[str, int | bool],
]:
    """Build the bounded C4 delta without wait-for or source-level authority."""
    nodes = [dict(row) for row in semantic_nodes]
    graph_edges = [dict(row) for row in edges]
    diagnostics: list[Dict[str, Any]] = []
    counters: Dict[str, int | bool] = {
        "transition_values": 0,
        "waveform_value_misses": 0,
        "dependency_evaluations": 0,
        "seed_candidates": 0,
        "seeds_retained": 0,
        "semantic_paths_evaluated": 0,
        "transition_values_reached": False,
    }
    node_by_id = {str(row["semantic_id"]): row for row in nodes}
    predicates = [
        row for row in nodes if row["type"] == "assertion_predicate"
    ]
    predicate_ids = {str(row["semantic_id"]) for row in predicates}
    predicate_members = {
        _clean(str(member))
        for predicate in predicates
        for member in predicate.get("member_signals", [])
    }
    registers = [
        row
        for row in nodes
        if row["type"] == "register_transition"
        and _clean(str(row["signal"])) in predicate_members
        and row.get("counter_pattern") == "bounded_progress_counter"
    ]
    register_ids = {str(row["semantic_id"]) for row in registers}
    persistent = [
        row
        for row in nodes
        if row["type"] == "persistent_interval"
        and str(row.get("register_id")) in register_ids
        and row.get("rule") in {"increment", "decrement"}
        and int(row["end_cycle"]) >= endpoint_cycle
    ]
    temporal_nodes: list[Dict[str, Any]] = []
    temporal_edges: list[Dict[str, Any]] = []
    seed_rows: list[Dict[str, Any]] = []

    def add_seed(
        semantic_id: str,
        rule: str,
        *,
        cycle: Optional[int] = None,
        interval: Optional[Sequence[int]] = None,
        evidence_refs: Sequence[str] = (),
        identity_strength: str = "exact",
        structural_priority: Sequence[int] = (),
        component_identity: Optional[str] = None,
    ) -> None:
        seed_rows.append(
            {
                "candidate_id": stable_id(
                    "vseed_",
                    semantic_id,
                    rule,
                    cycle,
                    list(interval) if interval is not None else None,
                    sorted(set(evidence_refs)),
                    length=24,
                ),
                "semantic_id": semantic_id,
                "derivation_rule": rule,
                "cycle": cycle,
                "interval": list(interval) if interval is not None else None,
                "evidence_refs": sorted(set(evidence_refs)),
                "identity_strength": identity_strength,
                "structural_priority": list(structural_priority),
                "component_identity": component_identity,
            }
        )

    for register in sorted(registers, key=lambda row: row["semantic_id"]):
        register_id = str(register["semantic_id"])
        add_seed(
            register_id,
            "predicate_counter_operand.v1",
            cycle=endpoint_cycle,
            evidence_refs=sorted(predicate_ids),
        )
        remaining = max(
            0,
            max_transition_values - int(counters["transition_values"]),
        )
        if remaining == 0:
            counters["transition_values_reached"] = True
            break
        series = build_transition_intervals(
            waveform,
            str(register["signal"]),
            start_cycle=0,
            end_cycle=endpoint_cycle,
            max_transition_values=remaining,
        )
        counters["transition_values"] = int(counters["transition_values"]) + int(
            series["work"]["transition_values"]
        )
        counters["waveform_value_misses"] = int(
            counters["waveform_value_misses"]
        ) + int(series["work"]["value_misses"])
        if series["truncated"]:
            counters["transition_values_reached"] = True
            diagnostics.append(
                {
                    "code": "transition_value_budget_reached",
                    "message": "C4 transition index reached max_temporal_samples",
                    "breaks_complete": True,
                }
            )
        endpoint_value = parse_binary_value(
            series["boundary_values"]["end"]
        )
        crossing = None
        if endpoint_value is not None:
            for change in series["changes"]:
                old_value = parse_binary_value(change["old"])
                new_value = parse_binary_value(change["new"])
                if old_value is None or new_value is None:
                    continue
                if (
                    new_value >= endpoint_value > old_value
                    or new_value <= endpoint_value < old_value
                ):
                    crossing = change
                    break
        if crossing is not None:
            identity = {
                "type": "threshold_crossing",
                "register_id": register_id,
                "cycle": crossing["cycle"],
                "threshold": endpoint_value,
                "threshold_source": "endpoint_observed_counter_value",
            }
            crossing_id = stable_id("vct_", identity, length=24)
            crossing_node = {
                "semantic_id": crossing_id,
                **identity,
                "old_value": parse_binary_value(crossing["old"]),
                "new_value": parse_binary_value(crossing["new"]),
                "boundary_transitions": [dict(crossing)],
                "unknown_spans": series["unknown_spans"],
                "evidence_strength": "transition_supported",
                "inference_rule": "observed_counter_threshold_crossing.v1",
            }
            temporal_nodes.append(crossing_node)
            add_seed(
                crossing_id,
                "threshold_crossing.v1",
                cycle=int(crossing["cycle"]),
                evidence_refs=[register_id],
            )
            for predicate_id in sorted(predicate_ids):
                temporal_edges.append(
                    {
                        "edge_id": stable_id(
                            "vcse_",
                            crossing_id,
                            predicate_id,
                            "causes_transition",
                            length=24,
                        ),
                        "src_semantic_id": crossing_id,
                        "dst_semantic_id": predicate_id,
                        "relation": "causes_transition",
                        "evidence_strength": "transition_supported",
                        "dynamic_score": 1.0,
                    }
                )

    for interval in sorted(persistent, key=lambda row: row["semantic_id"]):
        interval_id = str(interval["semantic_id"])
        add_seed(
            interval_id,
            "persistent_active_guard.v1",
            interval=[int(interval["start_cycle"]), int(interval["end_cycle"])],
            evidence_refs=[
                str(interval["register_id"]),
                str(interval["subject_id"]),
            ],
            identity_strength=(
                "exact"
                if interval.get("observation") == "fully_observed"
                else "structural_only"
            ),
        )
        register = node_by_id[str(interval["register_id"])]
        completion_rule_ids = sorted(
            str(rule["rule_id"])
            for rule in list(register.get("reset_rules", []))
            + list(register.get("update_rules", []))
            if rule.get("update_kind") in {"clear", "reset", "load"}
        )
        if not completion_rule_ids:
            continue
        identity = {
            "type": "missing_expected_completion",
            "register_id": interval["register_id"],
            "start_cycle": interval["start_cycle"],
            "end_cycle": endpoint_cycle,
            "expected_rule_ids": completion_rule_ids,
        }
        missing_id = stable_id("vct_", identity, length=24)
        temporal_nodes.append(
            {
                "semantic_id": missing_id,
                **identity,
                "evidence_refs": [interval_id, *completion_rule_ids],
                "evidence_strength": "interval_rule_derived",
                "inference_rule": "persistent_counter_without_completion.v1",
            }
        )
        add_seed(
            missing_id,
            "missing_expected_completion.v1",
            interval=[int(interval["start_cycle"]), endpoint_cycle],
            evidence_refs=[interval_id, *completion_rule_ids],
            identity_strength="derived",
        )
        temporal_edges.append(
            {
                "edge_id": stable_id(
                    "vcse_",
                    interval_id,
                    missing_id,
                    "prevents_completion",
                    length=24,
                ),
                "src_semantic_id": interval_id,
                "dst_semantic_id": missing_id,
                "relation": "prevents_completion",
                "evidence_strength": "interval_rule_derived",
                "dynamic_score": float(interval.get("dynamic_score", 0.0)),
            }
        )
        for predicate_id in sorted(predicate_ids):
            temporal_edges.append(
                {
                    "edge_id": stable_id(
                        "vcse_",
                        missing_id,
                        predicate_id,
                        "persists_through",
                        length=24,
                    ),
                    "src_semantic_id": missing_id,
                    "dst_semantic_id": predicate_id,
                    "relation": "persists_through",
                    "evidence_strength": "interval_rule_derived",
                    "dynamic_score": float(interval.get("dynamic_score", 0.0)),
                }
            )

    failure_start = min(
        [int(row["start_cycle"]) for row in persistent] or [endpoint_cycle]
    )
    stalls = [
        row
        for row in nodes
        if row["type"] == "stall_interval"
        and int(row["start_cycle"]) <= endpoint_cycle
        and int(row["end_cycle"]) >= failure_start
    ]
    for stall in sorted(stalls, key=lambda row: row["semantic_id"]):
        add_seed(
            str(stall["semantic_id"]),
            "overlapping_stall.v1",
            interval=[
                max(failure_start, int(stall["start_cycle"])),
                min(endpoint_cycle, int(stall["end_cycle"])),
            ],
            evidence_refs=[str(stall["handshake_id"])],
            identity_strength=str(stall.get("evidence_strength", "unknown")),
        )

    projected_blockers = [
        row
        for row in nodes
        if row["type"] == "blocking_relation"
        and str(row.get("inference_rule", "")).startswith(
            "exact_waveform_scope_module_signature.v1+"
        )
    ]
    pipeline_stage_valids = {
        str(row["semantic_id"]): {
            int(stage["index"]): str(stage["valid_signal"])
            for stage in row.get("stages", [])
        }
        for row in nodes
        if row["type"] == "pipeline"
    }
    for blocker in sorted(
        projected_blockers, key=lambda row: row["semantic_id"]
    ):
        remaining = max(
            0,
            max_transition_values - int(counters["transition_values"]),
        )
        if remaining == 0:
            counters["transition_values_reached"] = True
            break
        series = build_transition_intervals(
            waveform,
            str(blocker["target_signal"]),
            start_cycle=failure_start,
            end_cycle=endpoint_cycle,
            max_transition_values=remaining,
        )
        counters["transition_values"] = int(counters["transition_values"]) + int(
            series["work"]["transition_values"]
        )
        counters["waveform_value_misses"] = int(
            counters["waveform_value_misses"]
        ) + int(series["work"]["value_misses"])
        start_value = parse_binary_value(
            series["boundary_values"]["start"]
        )
        end_value = parse_binary_value(series["boundary_values"]["end"])
        if (
            not series["available"]
            or series["truncated"]
            or series["unknown_spans"]
            or start_value != 1
            or end_value != 1
            or series["changes"]
        ):
            continue
        blocker_stage_count = len(blocker.get("blockers", []))
        guarded_stage_count = sum(
            pipeline_stage_valids.get(str(item["pipeline_id"]), {}).get(
                int(item["stage_index"])
            )
            in set(item.get("member_signals", []))
            for item in blocker.get("blockers", [])
        )
        add_seed(
            str(blocker["semantic_id"]),
            "persistent_pipeline_blocker.v1",
            interval=[failure_start, endpoint_cycle],
            evidence_refs=[
                str(item["pipeline_id"])
                for item in blocker.get("blockers", [])
            ],
            identity_strength="exact",
            structural_priority=[
                guarded_stage_count,
                blocker_stage_count,
                len(blocker.get("statement_ids", [])),
            ],
            component_identity=str(blocker["instance_path"]),
        )

    progress_rows: list[Tuple[int, str, str]] = []
    for stall in stalls:
        if stall.get("last_accept_cycle") is not None:
            progress_rows.append(
                (
                    int(stall["last_accept_cycle"]),
                    str(stall["handshake_id"]),
                    "handshake_accepted",
                )
            )
    for interval in persistent:
        if int(interval["start_cycle"]) > 0:
            progress_rows.append(
                (
                    int(interval["start_cycle"]) - 1,
                    str(interval["register_id"]),
                    "counter_progress_before_persistence",
                )
            )
    if progress_rows:
        progress_cycle, source_id, event_type = sorted(
            progress_rows, key=lambda row: (row[0], row[1], row[2])
        )[-1]
        identity = {
            "type": "last_progress_event",
            "source_semantic_id": source_id,
            "cycle": progress_cycle,
            "event_type": event_type,
        }
        progress_id = stable_id("vct_", identity, length=24)
        temporal_nodes.append(
            {
                "semantic_id": progress_id,
                **identity,
                "evidence_strength": "transition_supported",
                "inference_rule": "latest_observed_progress.v1",
            }
        )
        add_seed(
            progress_id,
            "last_progress_predecessor.v1",
            cycle=progress_cycle,
            evidence_refs=[source_id],
        )
        temporal_edges.append(
            {
                "edge_id": stable_id(
                    "vcse_",
                    progress_id,
                    source_id,
                    "starts_before",
                    length=24,
                ),
                "src_semantic_id": progress_id,
                "dst_semantic_id": source_id,
                "relation": "starts_before",
                "evidence_strength": "transition_supported",
                "dynamic_score": 1.0,
            }
        )

    nodes.extend(temporal_nodes)
    graph_edges.extend(temporal_edges)
    counters["seed_candidates"] = len(seed_rows)
    ordered_seeds = sorted(
        seed_rows,
        key=lambda row: (
            _SEED_PRIORITY[row["derivation_rule"]],
            tuple(-int(value) for value in row["structural_priority"]),
            str(row["semantic_id"]),
            str(row["candidate_id"]),
        ),
    )
    retained_seeds: list[Dict[str, Any]] = []
    deferred_blockers: list[Dict[str, Any]] = []
    retained_pipeline_components: set[Tuple[str, ...]] = set()
    for row in ordered_seeds:
        if row["derivation_rule"] == "persistent_pipeline_blocker.v1":
            component = (str(row["component_identity"]),)
            if component in retained_pipeline_components:
                deferred_blockers.append(row)
                continue
            retained_pipeline_components.add(component)
        retained_seeds.append(row)
        if len(retained_seeds) >= max_seed_count:
            break
    if len(retained_seeds) < max_seed_count:
        retained_seeds.extend(
            deferred_blockers[: max_seed_count - len(retained_seeds)]
        )
    counters["seeds_retained"] = len(retained_seeds)
    seed_ids = {str(row["semantic_id"]) for row in retained_seeds}

    adjacency: Dict[str, set[str]] = {}
    for edge in graph_edges:
        source, target = _edge_endpoints(edge)
        if source is None or target is None:
            continue
        adjacency.setdefault(source, set()).add(target)
        adjacency.setdefault(target, set()).add(source)
    counters["dependency_evaluations"] = sum(
        len(neighbors) for neighbors in adjacency.values()
    )
    candidates: list[Dict[str, Any]] = []
    for node in sorted(nodes, key=lambda row: row["semantic_id"]):
        if node["type"] not in _TYPE_PRIORITY:
            continue
        semantic_id = str(node["semantic_id"])
        counters["semantic_paths_evaluated"] = int(
            counters["semantic_paths_evaluated"]
        ) + 1
        path = _semantic_path(
            semantic_id,
            seed_ids | predicate_ids,
            adjacency,
            max_length=8,
        )
        if semantic_id in seed_ids:
            path = [semantic_id]
        if path is None:
            continue
        distance = max(0, len(path) - 1)
        start_cycle = int(node.get("start_cycle", endpoint_cycle))
        end_cycle = int(node.get("end_cycle", endpoint_cycle))
        overlap = max(0, min(end_cycle, endpoint_cycle) - max(start_cycle, failure_start) + 1)
        window = max(1, endpoint_cycle - failure_start + 1)
        persistence_ratio = min(1.0, overlap / window)
        dynamic_support = str(
            node.get(
                "evidence_strength",
                node.get("observation", "exact_structural"),
            )
        )
        exactness = 1.0 if node.get("instance_path") is not None else 0.9
        structural_support = 0.0
        stage_coverage = 0.0
        if node["type"] == "blocking_relation":
            blocker_rows = list(node.get("blockers", []))
            guarded = sum(
                pipeline_stage_valids.get(
                    str(item["pipeline_id"]), {}
                ).get(int(item["stage_index"]))
                in set(item.get("member_signals", []))
                for item in blocker_rows
            )
            structural_support = guarded / max(1, len(blocker_rows))
            stage_coverage = min(1.0, len(blocker_rows) / 3.0)
        score = round(
            0.30 * exactness
            + 0.30 * persistence_ratio
            + 0.25 * (1.0 / (1.0 + distance))
            + 0.15 * (1.0 if semantic_id in seed_ids else 0.0)
            + 0.12 * structural_support
            + 0.08 * stage_coverage,
            6,
        )
        candidates.append(
            {
                "candidate_id": stable_id(
                    "vsrc_",
                    semantic_id,
                    path,
                    score,
                    length=24,
                ),
                "semantic_id": semantic_id,
                "semantic_type": node["type"],
                "instance_exactness": exactness,
                "temporal_overlap": round(overlap / window, 6),
                "persistence_ratio": round(persistence_ratio, 6),
                "causal_distance": distance,
                "dynamic_support": dynamic_support,
                "structural_support": round(structural_support, 6),
                "stage_coverage": round(stage_coverage, 6),
                "source_projection_status": "rtl_only",
                "semantic_path": path,
                "ranking_score": score,
            }
        )
    candidates.sort(
        key=lambda row: (
            -float(row["ranking_score"]),
            int(row["causal_distance"]),
            _TYPE_PRIORITY[str(row["semantic_type"])],
            str(row["semantic_id"]),
        )
    )
    root_candidates = candidates[:20]
    seed_by_id = {str(row["semantic_id"]): row for row in retained_seeds}
    for candidate in root_candidates:
        seed = seed_by_id.get(str(candidate["semantic_id"]))
        if seed is not None:
            candidate["seed"] = seed
    return (
        sorted(nodes, key=lambda row: row["semantic_id"]),
        sorted(graph_edges, key=lambda row: row["edge_id"]),
        root_candidates,
        diagnostics,
        counters,
    )


def get_semantic_paths(
    graph: Mapping[str, Any],
    target_id: str,
    *,
    max_paths: int,
    max_length: int,
) -> Dict[str, Any]:
    """Return only canonical, already-materialized C4 paths for one graph ID."""
    if max_paths < 1 or max_paths > 3:
        raise ValueError("max_paths must be in [1, 3]")
    if max_length < 1 or max_length > 8:
        raise ValueError("max_length must be in [1, 8]")
    known = {
        str(row["semantic_id"]) for row in graph.get("semantic_nodes", [])
    }
    if target_id not in known:
        raise ValueError("target_id is absent from the semantic graph")
    rows = [
        {
            "candidate_id": row["candidate_id"],
            "semantic_path": row["semantic_path"][: max_length + 1],
            "ranking_score": row["ranking_score"],
        }
        for row in graph.get("root_candidates", [])
        if row["semantic_id"] == target_id
        and len(row["semantic_path"]) - 1 <= max_length
    ][:max_paths]
    result = {
        "schema_version": "chisel_semantic_paths_query.v1",
        "graph_id": graph["graph_id"],
        "target_id": target_id,
        "paths": rows,
    }
    result["result_sha256"] = canonical_sha256(result)
    return result
