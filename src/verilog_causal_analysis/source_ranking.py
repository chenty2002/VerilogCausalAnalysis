"""Aggregate causal evidence into a deterministic Chisel source ranking."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

from .identity import canonical_sha256, stable_id


_LOCATOR_RE = re.compile(
    r"(?P<path>[A-Za-z0-9_./-]+\.scala):"
    r"(?P<line>[0-9]+):(?P<column>[0-9]+|\{[0-9,]+\})"
)
_SHORT_LOCATOR_RE = re.compile(
    r"(?:^|,)\s*:(?P<line>[0-9]+):(?P<column>[0-9]+|\{[0-9,]+\})"
)


def build_source_ranking(
    graph: Mapping[str, Any],
    elaboration: Mapping[str, Any],
    *,
    case_id: str,
    method: str,
    source_root: str | Path,
) -> Dict[str, Any]:
    """Project graph evidence after graph construction; no gold input is accepted."""

    candidates: dict[tuple[str, int], dict[str, Any]] = {}
    for locator in elaboration.get("source_locators", []):
        _merge(
            candidates,
            locator,
            score=0.0,
            instance="",
            evidence_ids=(),
            source_authority="authoritative",
            allow_new=True,
        )

    object_locators: dict[str, list[Mapping[str, Any]]] = {}
    for obj in elaboration.get("objects", []):
        locator = obj.get("source_locator")
        if not isinstance(locator, Mapping):
            continue
        object_locators.setdefault(str(obj.get("name", "")), []).append(locator)
    # The method-independent universe is the set of executable Chisel
    # statements retained by elaboration, never a scan of arbitrary source lines.
    _ = source_root

    nodes = {
        str(row["node_id"]): row
        for row in graph.get("signal_nodes", [])
        if isinstance(row, Mapping) and row.get("node_id")
    }
    for node_id, node in nodes.items():
        signal = re.sub(r"\s*\[[0-9]+:[0-9]+\]$", "", str(node.get("signal", "")))
        name = signal.rsplit(".", 1)[-1]
        instance = signal.rsplit(".", 1)[0] if "." in signal else ""
        for locator in object_locators.get(name, []):
            _merge(
                candidates,
                locator,
                score=float(node.get("suspect_score", 0.0) or 0.0),
                instance=instance,
                evidence_ids=(node_id,),
                semantic_role="state" if not node.get("is_slice_leaf") else "data",
                source_authority="authoritative",
            )

    for edge in graph.get("edges", []):
        if not isinstance(edge, Mapping):
            continue
        evidence = edge.get("rtl_evidence")
        if not isinstance(evidence, Mapping):
            continue
        evidence_ids = [str(edge.get("edge_id", ""))]
        evidence_ids.extend(
            str(edge[key])
            for key in ("src_node_id", "dst_node_id", "src_semantic_id", "dst_semantic_id")
            if edge.get(key)
        )
        instance = ""
        for key in ("src_node_id", "dst_node_id"):
            node = nodes.get(str(edge.get(key, "")))
            if node:
                signal = re.sub(r"\s*\[[0-9]+:[0-9]+\]$", "", str(node["signal"]))
                instance = signal.rsplit(".", 1)[0] if "." in signal else ""
                break
        score = float(
            edge.get("contribution_score", edge.get("dynamic_score", 0.0)) or 0.0
        )
        role = (
            "control"
            if str(evidence.get("condition", ""))
            else "state"
            if edge.get("dependency_type") == "sequential"
            else "data"
        )
        for locator in _locators(str(evidence.get("snippet", ""))):
            _merge(
                candidates,
                locator,
                score=score,
                instance=instance,
                evidence_ids=evidence_ids,
                semantic_role=role,
                source_authority="authoritative",
            )

    for semantic in graph.get("semantic_nodes", []):
        if not isinstance(semantic, Mapping) or semantic.get("type") != "source_provenance_hint":
            continue
        locator = str(semantic.get("reported_locator", ""))
        match = re.fullmatch(r"(?P<line>[0-9]+):(?P<column>[0-9]+|\{[0-9,]+\})", locator)
        if match:
            _merge(
                candidates,
                {
                    "path": semantic.get("reported_path"),
                    "line": int(match.group("line")),
                    "column": _column(match.group("column")),
                },
                score=(
                    0.4
                    if semantic.get("status") == "unverified_hint"
                    else 1.0
                    if semantic.get("status") == "source_projection_candidate"
                    else 0.0
                ),
                instance="",
                evidence_ids=(str(semantic.get("semantic_id", "")),),
                semantic_role="source",
                source_authority=str(semantic.get("authority", "non_authoritative")),
            )

    coverage_frontiers = sorted(
        str(node_id)
        for node_id, node in nodes.items()
        if node.get("is_slice_leaf") and node.get("rtl_context_status") == "missing"
    )
    trigger_candidates = sorted(
        str(node_id)
        for node_id, node in nodes.items()
        if node.get("is_slice_leaf") and node.get("rtl_context_status") != "missing"
    )
    for row in candidates.values():
        groups = sorted(row.pop("_groups").values(), key=lambda item: item["evidence_group_id"])
        role_best: Dict[str, float] = {}
        for group in groups:
            role_best[group["semantic_role"]] = max(
                role_best.get(group["semantic_role"], 0.0), group["score"]
            )
        best = max(role_best.values(), default=0.0)
        top_roles = sorted(role_best.values(), reverse=True)[:2]
        role_mean = sum(top_roles) / len(top_roles) if top_roles else 0.0
        rootness = best
        row["score"] = round(min(1.0, 0.65 * best + 0.25 * role_mean + 0.10 * rootness), 6)
        row["evidence_groups"] = groups
        row["evidence_group_ids"] = [item["evidence_group_id"] for item in groups]
        row["semantic_roles"] = sorted(role_best)
        row["best_path_ids"] = []
        row["positive_authoritative_evidence"] = bool(
            row["score"] > 0.0 and row["authoritative_evidence_ids"]
        )

    rows = sorted(
        candidates.values(),
        key=lambda row: (
            -row["score"],
            -len(row["authoritative_evidence_ids"]),
            row["statement_id"],
        ),
    )
    for position, row in enumerate(rows, 1):
        row["position"] = position
    for start in range(len(rows)):
        if start and rows[start - 1]["score"] == rows[start]["score"]:
            continue
        stop = start
        while stop + 1 < len(rows) and rows[stop + 1]["score"] == rows[start]["score"]:
            stop += 1
        average_rank = (start + 1 + stop + 1) / 2
        for index in range(start, stop + 1):
            rows[index]["rank"] = average_rank
            rows[index]["tie_size"] = stop - start + 1

    ambiguous_projection = any(
        item.get("type") == "source_provenance_hint"
        and item.get("status") == "ambiguous"
        for item in graph.get("semantic_nodes", [])
    )
    complete_source_projection = bool(rows) and not ambiguous_projection
    return {
        "schema_version": "chiselcause_source_ranking.v2",
        "case_id": case_id,
        "method": method,
        "status": "complete" if complete_source_projection else "incomplete",
        "complete_graph": graph.get("status") == "complete",
        "complete_source_projection": complete_source_projection,
        "graph_id": graph.get("graph_id"),
        "policy_id": (graph.get("search_summary") or {}).get("policy_id"),
        "source_projection_status": (
            "incomplete"
            if any(
                item.get("type") == "source_provenance_hint"
                and item.get("status") == "ambiguous"
                for item in graph.get("semantic_nodes", [])
            )
            else "candidate"
            if any(row["evidence_group_ids"] for row in rows)
            else "rtl_only"
        ),
        "ordering": rows,
        "candidates": rows,
        "statement_candidate_count": len(rows),
        "authoritative_candidate_count": sum(
            row["source_authority"] == "authoritative" for row in rows
        ),
        "positive_authoritative_candidate_count": sum(
            row["positive_authoritative_evidence"] for row in rows
        ),
        "coverage_frontiers": coverage_frontiers,
        "trigger_candidates": trigger_candidates,
        "tie_rule": "average_rank_by_score; position_by_authoritative_evidence_then_statement_id",
        "ranking_sha256": canonical_sha256(rows),
    }


def _locators(text: str) -> list[dict[str, Any]]:
    matches = list(_LOCATOR_RE.finditer(text))
    rows = [
        {
            "path": match.group("path"),
            "line": int(match.group("line")),
            "column": _column(match.group("column")),
        }
        for match in matches
    ]
    if matches:
        last = matches[-1]
        rows.extend(
            {
                "path": last.group("path"),
                "line": int(match.group("line")),
                "column": _column(match.group("column")),
            }
            for match in _SHORT_LOCATOR_RE.finditer(text, last.end())
        )
    return rows


def _column(value: Any) -> int:
    match = re.search(r"[0-9]+", str(value))
    return int(match.group()) if match else 1


def _merge(
    candidates: dict[tuple[str, int], dict[str, Any]],
    locator: Mapping[str, Any],
    *,
    score: float,
    instance: str,
    evidence_ids: Sequence[str],
    semantic_role: str = "data",
    source_authority: str = "non_authoritative",
    allow_new: bool = False,
) -> None:
    path = str(locator.get("path", ""))
    line = int(locator.get("line", 0) or 0)
    column = _column(locator.get("column", 1))
    if not path.endswith(".scala") or line <= 0:
        return
    key = (path, line)
    if key not in candidates and not allow_new:
        return
    statement_id = stable_id(
        "vcs_", {"path": path, "line": line}, length=24
    )
    row = candidates.setdefault(
        key,
        {
            "rank": 0,
            "position": 0,
            "tie_size": 0,
            "statement_id": statement_id,
            "score": 0.0,
            "instance": instance,
            "file": path,
            "line": line,
            "column": column,
            "evidence_node_ids": [],
            "authoritative_evidence_ids": [],
            "source_authority": source_authority,
            "origin_evidence": [
                {
                    "kind": "elaboration_source_locator",
                    "authority": source_authority,
                    "path": path,
                    "line": line,
                    "column": column,
                }
            ],
            "_groups": {},
        },
    )
    group_identity = {
        "instance": instance,
        "file": path,
        "line": line,
        "semantic_role": semantic_role,
    }
    group_id = stable_id("vceg_", group_identity, length=24)
    group = row["_groups"].setdefault(
        group_id,
        {
            "evidence_group_id": group_id,
            "semantic_role": semantic_role,
            "score": 0.0,
            "evidence_ids": [],
        },
    )
    # Repeated compiler nets/cycles in one statement-role group contribute once.
    group["score"] = round(max(group["score"], max(0.0, min(1.0, score))), 6)
    group["evidence_ids"] = sorted(
        set(group["evidence_ids"]) | {item for item in evidence_ids if item}
    )
    row["score"] = max(row["score"], group["score"])
    if instance and (not row["instance"] or instance < row["instance"]):
        row["instance"] = instance
    row["evidence_node_ids"] = sorted(
        set(row["evidence_node_ids"]) | {item for item in evidence_ids if item}
    )
    if source_authority == "authoritative":
        row["source_authority"] = "authoritative"
        row["authoritative_evidence_ids"] = sorted(
            set(row["authoritative_evidence_ids"])
            | {item for item in evidence_ids if item}
        )
