"""Stable-ID queries over current semantic graphs."""

from __future__ import annotations

from typing import Any, Dict, Mapping

from .identity import canonical_sha256


class SemanticGraphQueryError(ValueError):
    pass


def _node(graph: Mapping[str, Any], semantic_id: str, types: set[str]) -> Dict[str, Any]:
    rows = [
        dict(row)
        for row in graph.get("semantic_nodes", [])
        if row.get("semantic_id") == semantic_id and row.get("type") in types
    ]
    if len(rows) != 1:
        raise SemanticGraphQueryError("semantic ID is absent or has the wrong type")
    return rows[0]


def _result(schema: str, graph: Mapping[str, Any], payload: Dict[str, Any]) -> Dict[str, Any]:
    row = {"schema_version": schema, "graph_id": graph["graph_id"], **payload}
    row["result_sha256"] = canonical_sha256(row)
    return row


def get_semantic_overview(graph: Mapping[str, Any], *, top_k: int) -> Dict[str, Any]:
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
        raise SemanticGraphQueryError("top_k must be positive")
    counts: Dict[str, int] = {}
    for row in graph.get("semantic_nodes", []):
        key = str(row.get("type"))
        counts[key] = counts.get(key, 0) + 1
    return _result(
        "chisel_semantic_overview_query",
        graph,
        {
            "status": graph.get("status"),
            "semantic_type_counts": dict(sorted(counts.items())),
            "root_candidates": list(graph.get("root_candidates", []))[:top_k],
            "diagnostics": list(graph.get("diagnostics", []))[:top_k],
        },
    )


def get_interval_evidence(
    graph: Mapping[str, Any], interval_ids: list[str]
) -> Dict[str, Any]:
    if not interval_ids or len(interval_ids) != len(set(interval_ids)):
        raise SemanticGraphQueryError("interval_ids must be non-empty and unique")
    rows = [
        _node(graph, item, {"persistent_interval", "stall_interval", "pipeline_occupancy"})
        for item in sorted(interval_ids)
    ]
    return _result("chisel_interval_evidence_query", graph, {"intervals": rows})


def get_handshake_timeline(
    graph: Mapping[str, Any], handshake_id: str, *, max_events: int
) -> Dict[str, Any]:
    if isinstance(max_events, bool) or not isinstance(max_events, int) or max_events <= 0:
        raise SemanticGraphQueryError("max_events must be positive")
    handshake = _node(graph, handshake_id, {"handshake"})
    events = sorted(
        (
            dict(row)
            for row in graph.get("semantic_nodes", [])
            if row.get("handshake_id") == handshake_id
            and row.get("type") in {"stall_interval", "handshake_event", "last_progress_event"}
        ),
        key=lambda row: (
            int(row.get("start_cycle", row.get("cycle", -1))),
            str(row["semantic_id"]),
        ),
    )[:max_events]
    return _result(
        "chisel_handshake_timeline_query",
        graph,
        {"handshake": handshake, "events": events, "max_events": max_events},
    )


def get_pipeline_occupancy(
    graph: Mapping[str, Any],
    pipeline_id: str,
    *,
    start_cycle: int,
    end_cycle: int,
) -> Dict[str, Any]:
    if (
        isinstance(start_cycle, bool)
        or isinstance(end_cycle, bool)
        or not isinstance(start_cycle, int)
        or not isinstance(end_cycle, int)
        or start_cycle < 0
        or end_cycle < start_cycle
    ):
        raise SemanticGraphQueryError("cycle range is invalid")
    pipeline = _node(graph, pipeline_id, {"pipeline"})
    intervals = [
        dict(row)
        for row in graph.get("semantic_nodes", [])
        if row.get("type") == "pipeline_occupancy"
        and row.get("pipeline_id") == pipeline_id
        and row.get("start_cycle", end_cycle + 1) <= end_cycle
        and row.get("end_cycle", start_cycle - 1) >= start_cycle
    ]
    return _result(
        "chisel_pipeline_occupancy_query",
        graph,
        {
            "pipeline": pipeline,
            "requested_interval": [start_cycle, end_cycle],
            "occupancy": sorted(
                intervals,
                key=lambda row: (row["start_cycle"], row["semantic_id"]),
            ),
        },
    )
