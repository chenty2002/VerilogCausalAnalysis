"""Pure, bounded queries over an immutable ``verilog_causal_graph.v2``."""

from __future__ import annotations

import copy
from collections import OrderedDict
from dataclasses import dataclass
from threading import RLock
from types import MappingProxyType
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Set, Tuple

from .contracts import EVIDENCE_STRENGTHS, validate_graph_v2
from .identity import canonical_sha256, contains_absolute_path


MAX_OVERVIEW_NODES = 20
MAX_EXPAND_HOPS = 2
MAX_EXPAND_NODES = 20
MAX_EXPAND_EDGES = 30
MAX_PATHS = 3
MAX_PATH_LENGTH = 8
MAX_EVIDENCE_EDGES = 8

_STRENGTH_RANK = {
    "unresolved": 0,
    "structural_only": 1,
    "toggle_supported": 2,
    "branch_observed": 3,
    "expression_counterfactual": 4,
}


class QueryError(ValueError):
    """Raised for unknown IDs, invalid graphs, or requests over hard bounds."""


def _deep_freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {str(key): _deep_freeze(item) for key, item in value.items()}
        )
    if isinstance(value, list):
        return tuple(_deep_freeze(item) for item in value)
    return value


def _deep_thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _deep_thaw(item) for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_deep_thaw(item) for item in value]
    return value


@dataclass(frozen=True)
class GraphQueryView:
    """Validated immutable indexes bound to canonical graph bytes."""

    graph_sha256: str
    graph: Mapping[str, Any]
    nodes: Mapping[str, Mapping[str, Any]]
    edges: Mapping[str, Mapping[str, Any]]
    incoming: Mapping[str, Tuple[Mapping[str, Any], ...]]
    outgoing: Mapping[str, Tuple[Mapping[str, Any], ...]]

    @classmethod
    def from_graph(cls, graph: Mapping[str, Any]) -> "GraphQueryView":
        try:
            validated = validate_graph_v2(graph)
        except Exception as error:
            raise QueryError(str(error)) from error
        if contains_absolute_path(validated):
            raise QueryError("graph contains an absolute path")
        copied_graph = copy.deepcopy(validated)
        graph_sha256 = canonical_sha256(copied_graph)
        frozen_graph = _deep_freeze(copied_graph)
        nodes = {
            str(row["node_id"]): row
            for row in frozen_graph.get("nodes", [])
        }
        edges = {
            str(row["edge_id"]): row
            for row in frozen_graph.get("edges", [])
        }
        incoming_rows: Dict[str, List[Mapping[str, Any]]] = {}
        outgoing_rows: Dict[str, List[Mapping[str, Any]]] = {}
        for edge in edges.values():
            incoming_rows.setdefault(
                str(edge["dst_node_id"]), []
            ).append(edge)
            outgoing_rows.setdefault(
                str(edge["src_node_id"]), []
            ).append(edge)
        incoming = {
            node_id: tuple(
                sorted(
                    rows,
                    key=lambda row: (
                        str(row["src_node_id"]),
                        str(row["edge_id"]),
                    ),
                )
            )
            for node_id, rows in incoming_rows.items()
        }
        outgoing = {
            node_id: tuple(
                sorted(
                    rows,
                    key=lambda row: (
                        str(row["dst_node_id"]),
                        str(row["edge_id"]),
                    ),
                )
            )
            for node_id, rows in outgoing_rows.items()
        }
        return cls(
            graph_sha256=graph_sha256,
            graph=frozen_graph,
            nodes=MappingProxyType(nodes),
            edges=MappingProxyType(edges),
            incoming=MappingProxyType(incoming),
            outgoing=MappingProxyType(outgoing),
        )


_QUERY_VIEW_CACHE_MAX = 16
_QUERY_VIEW_CACHE: "OrderedDict[str, GraphQueryView]" = OrderedDict()
_QUERY_VIEW_CACHE_LOCK = RLock()
_QUERY_VIEW_BUILDS = 0
_QUERY_VIEW_HITS = 0


def prepare_query_view(
    graph: Mapping[str, Any] | GraphQueryView,
) -> GraphQueryView:
    """Return a canonical-hash-bound reusable query view."""
    global _QUERY_VIEW_BUILDS, _QUERY_VIEW_HITS
    if isinstance(graph, GraphQueryView):
        return graph
    graph_sha256 = canonical_sha256(graph)
    with _QUERY_VIEW_CACHE_LOCK:
        cached = _QUERY_VIEW_CACHE.get(graph_sha256)
        if cached is not None:
            _QUERY_VIEW_HITS += 1
            _QUERY_VIEW_CACHE.move_to_end(graph_sha256)
            return cached
    view = GraphQueryView.from_graph(graph)
    if view.graph_sha256 != graph_sha256:
        raise QueryError("graph changed while constructing query view")
    with _QUERY_VIEW_CACHE_LOCK:
        existing = _QUERY_VIEW_CACHE.get(graph_sha256)
        if existing is not None:
            _QUERY_VIEW_HITS += 1
            _QUERY_VIEW_CACHE.move_to_end(graph_sha256)
            return existing
        _QUERY_VIEW_CACHE[graph_sha256] = view
        _QUERY_VIEW_CACHE.move_to_end(graph_sha256)
        _QUERY_VIEW_BUILDS += 1
        while len(_QUERY_VIEW_CACHE) > _QUERY_VIEW_CACHE_MAX:
            _QUERY_VIEW_CACHE.popitem(last=False)
    return view


def get_query_cache_statistics() -> Dict[str, int]:
    """Return non-authoritative process-local query cache counters."""
    with _QUERY_VIEW_CACHE_LOCK:
        return {
            "entries": len(_QUERY_VIEW_CACHE),
            "builds": _QUERY_VIEW_BUILDS,
            "hits": _QUERY_VIEW_HITS,
            "capacity": _QUERY_VIEW_CACHE_MAX,
        }


def _view(
    graph: Mapping[str, Any] | GraphQueryView,
) -> GraphQueryView:
    return prepare_query_view(graph)


def _result(
    graph: Mapping[str, Any], query: Mapping[str, Any], payload: Mapping[str, Any]
) -> Dict[str, Any]:
    normalized_payload = _deep_thaw(payload)
    row = {"graph_id": graph["graph_id"], **normalized_payload}
    row["query_sha256"] = canonical_sha256(
        {
            "graph_id": graph["graph_id"],
            "query": dict(query),
            "result": normalized_payload,
        }
    )
    return row


def _known_ids(
    requested: Iterable[str], available: Mapping[str, Any], what: str
) -> List[str]:
    values = list(requested)
    if not values:
        raise QueryError(f"{what} must not be empty")
    if len(values) != len(set(values)):
        raise QueryError(f"{what} must not contain duplicates")
    unknown = sorted(set(values) - set(available))
    if unknown:
        raise QueryError(f"unknown {what}: {unknown}")
    return values


def get_overview(
    graph: Mapping[str, Any] | GraphQueryView, *, top_k: int
) -> Dict[str, Any]:
    view = _view(graph)
    graph = view.graph
    nodes, edges = view.nodes, view.edges
    if isinstance(top_k, bool) or not isinstance(top_k, int) or not 1 <= top_k <= MAX_OVERVIEW_NODES:
        raise QueryError(f"top_k must be in [1, {MAX_OVERVIEW_NODES}]")
    ranked = sorted(
        nodes.values(),
        key=lambda row: (
            -float(row.get("suspect_score", 0.0)),
            int(row.get("depth", 0)),
            str(row["node_id"]),
        ),
    )[:top_k]
    payload = {
        "status": graph.get("status"),
        "node_count": len(nodes),
        "edge_count": len(edges),
        "top_nodes": [dict(row) for row in ranked],
        "bounds": dict(graph.get("bounds") or {}),
        "diagnostics": list(graph.get("diagnostics") or []),
    }
    return _result(graph, {"name": "overview", "top_k": top_k}, payload)


def expand_predecessors(
    graph: Mapping[str, Any] | GraphQueryView,
    node_ids: Sequence[str],
    *,
    max_hops: int,
    max_nodes: int,
) -> Dict[str, Any]:
    view = _view(graph)
    graph = view.graph
    nodes = view.nodes
    seeds = _known_ids(node_ids, nodes, "node_ids")
    if max_hops not in (1, 2):
        raise QueryError("max_hops must be 1 or 2")
    if isinstance(max_nodes, bool) or not isinstance(max_nodes, int) or not 1 <= max_nodes <= MAX_EXPAND_NODES:
        raise QueryError(f"max_nodes must be in [1, {MAX_EXPAND_NODES}]")
    selected_nodes: Set[str] = set(seeds)
    selected_edges: Dict[str, Mapping[str, Any]] = {}
    frontier = sorted(seeds)
    for _ in range(max_hops):
        next_frontier: Set[str] = set()
        for node_id in frontier:
            for edge in view.incoming.get(node_id, ()):
                source = str(edge["src_node_id"])
                selected_edges[str(edge["edge_id"])] = edge
                selected_nodes.add(source)
                next_frontier.add(source)
        frontier = sorted(next_frontier)
    if len(selected_nodes) > max_nodes or len(selected_edges) > MAX_EXPAND_EDGES:
        raise QueryError(
            "predecessor expansion exceeds the deterministic node/edge budget"
        )
    payload = {
        "nodes": [dict(nodes[node_id]) for node_id in sorted(selected_nodes)],
        "edges": [
            dict(selected_edges[edge_id]) for edge_id in sorted(selected_edges)
        ],
    }
    return _result(
        graph,
        {
            "name": "expand_predecessors",
            "node_ids": sorted(seeds),
            "max_hops": max_hops,
            "max_nodes": max_nodes,
        },
        payload,
    )


def get_ranked_paths(
    graph: Mapping[str, Any] | GraphQueryView,
    target_node_id: str,
    *,
    max_paths: int,
    max_path_length: int,
    minimum_evidence_strength: str,
) -> Dict[str, Any]:
    view = _view(graph)
    graph = view.graph
    nodes = view.nodes
    _known_ids([target_node_id], nodes, "target_node_id")
    if isinstance(max_paths, bool) or not isinstance(max_paths, int) or not 1 <= max_paths <= MAX_PATHS:
        raise QueryError(f"max_paths must be in [1, {MAX_PATHS}]")
    if (
        isinstance(max_path_length, bool)
        or not isinstance(max_path_length, int)
        or not 1 <= max_path_length <= MAX_PATH_LENGTH
    ):
        raise QueryError(f"max_path_length must be in [1, {MAX_PATH_LENGTH}]")
    if minimum_evidence_strength not in EVIDENCE_STRENGTHS:
        raise QueryError("minimum_evidence_strength is invalid")
    threshold = _STRENGTH_RANK[minimum_evidence_strength]
    incoming = {
        node_id: tuple(
            sorted(
                (
                    edge
                    for edge in rows
                    if _STRENGTH_RANK[str(edge["evidence_strength"])]
                    >= threshold
                ),
                key=lambda row: (
                    -float(row.get("contribution_score", 0.0)),
                    str(row["edge_id"]),
                ),
            )
        )
        for node_id, rows in view.incoming.items()
    }

    paths: List[Dict[str, Any]] = []

    def visit(
        node_id: str,
        node_path: List[str],
        edge_path: List[str],
        scores: List[float],
    ) -> None:
        if len(paths) >= max_paths * 8:
            return
        parents = incoming.get(node_id, [])
        if not parents or len(edge_path) >= max_path_length:
            paths.append(
                {
                    "node_ids": list(node_path),
                    "edge_ids": list(edge_path),
                    "path_score": round(min(scores) if scores else 0.0, 6),
                }
            )
            return
        for edge in parents:
            source = str(edge["src_node_id"])
            if source in node_path:
                continue
            visit(
                source,
                node_path + [source],
                edge_path + [str(edge["edge_id"])],
                scores + [float(edge.get("contribution_score", 0.0))],
            )

    visit(target_node_id, [target_node_id], [], [])
    ranked = sorted(
        paths,
        key=lambda row: (
            -float(row["path_score"]),
            len(row["edge_ids"]),
            tuple(row["edge_ids"]),
        ),
    )[:max_paths]
    return _result(
        graph,
        {
            "name": "get_ranked_paths",
            "target_node_id": target_node_id,
            "max_paths": max_paths,
            "max_path_length": max_path_length,
            "minimum_evidence_strength": minimum_evidence_strength,
        },
        {"paths": ranked},
    )


def get_edge_evidence(
    graph: Mapping[str, Any] | GraphQueryView,
    edge_ids: Sequence[str],
) -> Dict[str, Any]:
    view = _view(graph)
    graph = view.graph
    edges = view.edges
    selected = _known_ids(edge_ids, edges, "edge_ids")
    if len(selected) > MAX_EVIDENCE_EDGES:
        raise QueryError(f"at most {MAX_EVIDENCE_EDGES} edge IDs are allowed")
    payload = {
        "edges": [dict(edges[edge_id]) for edge_id in sorted(selected)],
    }
    return _result(
        graph,
        {"name": "get_edge_evidence", "edge_ids": sorted(selected)},
        payload,
    )
