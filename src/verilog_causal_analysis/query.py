"""Pure, bounded queries over an immutable ``verilog_causal_graph.v2``."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Sequence, Set, Tuple

from .contracts import EVIDENCE_STRENGTHS, GRAPH_SCHEMA
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


def _index(
    graph: Mapping[str, Any],
) -> Tuple[Dict[str, Mapping[str, Any]], Dict[str, Mapping[str, Any]]]:
    if graph.get("schema_version") != GRAPH_SCHEMA:
        raise QueryError(f"graph must use {GRAPH_SCHEMA}")
    if contains_absolute_path(graph):
        raise QueryError("graph contains an absolute path")
    nodes = {str(row["node_id"]): row for row in graph.get("nodes", [])}
    edges = {str(row["edge_id"]): row for row in graph.get("edges", [])}
    if len(nodes) != len(graph.get("nodes", [])):
        raise QueryError("graph contains duplicate node IDs")
    if len(edges) != len(graph.get("edges", [])):
        raise QueryError("graph contains duplicate edge IDs")
    return nodes, edges


def _result(
    graph: Mapping[str, Any], query: Mapping[str, Any], payload: Mapping[str, Any]
) -> Dict[str, Any]:
    row = {"graph_id": graph["graph_id"], **dict(payload)}
    row["query_sha256"] = canonical_sha256(
        {"graph_id": graph["graph_id"], "query": dict(query), "result": payload}
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


def get_overview(graph: Mapping[str, Any], *, top_k: int) -> Dict[str, Any]:
    nodes, edges = _index(graph)
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
    graph: Mapping[str, Any],
    node_ids: Sequence[str],
    *,
    max_hops: int,
    max_nodes: int,
) -> Dict[str, Any]:
    nodes, edges = _index(graph)
    seeds = _known_ids(node_ids, nodes, "node_ids")
    if max_hops not in (1, 2):
        raise QueryError("max_hops must be 1 or 2")
    if isinstance(max_nodes, bool) or not isinstance(max_nodes, int) or not 1 <= max_nodes <= MAX_EXPAND_NODES:
        raise QueryError(f"max_nodes must be in [1, {MAX_EXPAND_NODES}]")
    incoming: Dict[str, List[Mapping[str, Any]]] = {}
    for edge in edges.values():
        incoming.setdefault(str(edge["dst_node_id"]), []).append(edge)
    for rows in incoming.values():
        rows.sort(key=lambda row: (str(row["src_node_id"]), str(row["edge_id"])))

    selected_nodes: Set[str] = set(seeds)
    selected_edges: Dict[str, Mapping[str, Any]] = {}
    frontier = sorted(seeds)
    for _ in range(max_hops):
        next_frontier: Set[str] = set()
        for node_id in frontier:
            for edge in incoming.get(node_id, []):
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
    graph: Mapping[str, Any],
    target_node_id: str,
    *,
    max_paths: int,
    max_path_length: int,
    minimum_evidence_strength: str,
) -> Dict[str, Any]:
    nodes, edges = _index(graph)
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
    incoming: Dict[str, List[Mapping[str, Any]]] = {}
    for edge in edges.values():
        if _STRENGTH_RANK[str(edge["evidence_strength"])] >= threshold:
            incoming.setdefault(str(edge["dst_node_id"]), []).append(edge)
    for rows in incoming.values():
        rows.sort(
            key=lambda row: (
                -float(row.get("contribution_score", 0.0)),
                str(row["edge_id"]),
            )
        )

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
    graph: Mapping[str, Any], edge_ids: Sequence[str]
) -> Dict[str, Any]:
    _, edges = _index(graph)
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
