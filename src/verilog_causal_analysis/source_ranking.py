"""Project causal evidence onto indexed executable Chisel statements."""

from __future__ import annotations

import difflib
import re
from pathlib import Path
from typing import Any, Dict, Mapping

from .identity import canonical_sha256


_LOCATOR_RE = re.compile(
    r"(?:@\[)?(?P<path>[A-Za-z0-9_./-]+\.scala)(?:\s+|:)"
    r"(?P<line>[0-9]+):(?P<column>[0-9]+|\{[0-9,]+\})"
)
_SHORT_LOCATOR_RE = re.compile(
    r"(?:^|,)\s*:(?P<line>[0-9]+):(?P<column>[0-9]+|\{[0-9,]+\})"
)


def build_source_ranking(
    graph: Mapping[str, Any],
    elaboration: Mapping[str, Any],
    *,
    source_index: Mapping[str, Any],
    case_id: str,
    method: str,
    source_root: str | Path,
    clean_rtl: str | Path | None = None,
    faulty_rtl: str | Path | None = None,
) -> Dict[str, Any]:
    """Rank one shared, source-indexed statement universe without reading gold."""

    _ = source_root
    candidates = {
        str(statement["statement_id"]): _candidate(statement)
        for statement in source_index.get("statements", [])
        if isinstance(statement, Mapping) and statement.get("statement_id")
    }
    by_object: dict[str, set[str]] = {}
    object_names: dict[str, str] = {}
    object_ids_by_name: dict[str, set[str]] = {}
    for row in source_index.get("objects", []):
        if not isinstance(row, Mapping) or not row.get("object_id"):
            continue
        object_id = str(row["object_id"])
        name = str(row.get("name", ""))
        object_names[object_id] = name
        object_ids_by_name.setdefault(name, set()).add(object_id)
    for statement_id, row in candidates.items():
        for object_id in row["semantic_object_ids"]:
            name = object_names.get(object_id)
            if name and len(object_ids_by_name[name]) == 1:
                by_object.setdefault(name, set()).add(statement_id)

    delta_statement_ids = _differential_statement_ids(
        candidates,
        elaboration=elaboration,
        clean_rtl=clean_rtl,
        faulty_rtl=faulty_rtl,
    )
    for statement_id in delta_statement_ids:
        candidates[statement_id]["in_rtl_delta"] = True

    nodes = {
        str(row["node_id"]): row
        for row in graph.get("signal_nodes", [])
        if isinstance(row, Mapping) and row.get("node_id")
    }
    distances = _endpoint_distances(nodes, graph.get("edges", []))
    unmapped: list[dict[str, str]] = []

    differential_endpoints = [
        node_id
        for node_id, node in nodes.items()
        if node.get("is_endpoint")
        and str(node.get("signal", "")).endswith("chiselcause_mismatch_any")
    ]
    if graph.get("status") == "complete" and len(delta_statement_ids) == 1 and differential_endpoints:
        statement_id = next(iter(delta_statement_ids))
        evidence_id = f"differential-unique-delta:{differential_endpoints[0]}"
        candidates[statement_id]["differential_evidence_ids"] = [evidence_id]
        _add_evidence(
            candidates[statement_id],
            evidence_id=evidence_id,
            score=1.0,
            distance=distances.get(differential_endpoints[0]),
            authority="authoritative",
        )

    nodes_by_name: dict[str, set[str]] = {}
    for node_id, node in nodes.items():
        signal = re.sub(r"\s*\[[0-9]+:[0-9]+\]$", "", str(node.get("signal", "")))
        nodes_by_name.setdefault(signal.rsplit(".", 1)[-1], set()).add(node_id)
    for statement_id in sorted(delta_statement_ids):
        child_objects: set[str] = set()
        for child_id in candidates[statement_id]["child_statement_ids"]:
            if child_id in candidates:
                child_objects.update(candidates[child_id]["semantic_object_ids"])
        related = child_objects | set(candidates[statement_id]["semantic_object_ids"])
        for object_id in sorted(related):
            name = object_names.get(object_id)
            if not name or len(object_ids_by_name.get(name, ())) != 1:
                continue
            for node_id in sorted(nodes_by_name.get(name, ())):
                evidence_id = f"differential-node:{node_id}"
                candidates[statement_id]["differential_evidence_ids"] = sorted(
                    set(candidates[statement_id]["differential_evidence_ids"])
                    | {evidence_id}
                )
                _add_evidence(
                    candidates[statement_id],
                    evidence_id=evidence_id,
                    score=1.0,
                    distance=distances.get(node_id),
                    authority="authoritative",
                )

    if graph.get("status") == "complete" and differential_endpoints:
        for statement_id in sorted(delta_statement_ids):
            row = candidates[statement_id]
            expression = str(row["exact_origin_spec"].get("value_expression", ""))
            if row["entity_kind"] != "blackbox_parameter" or not expression:
                continue
            consumers = [
                candidate
                for candidate in candidates.values()
                if candidate["execution_phase"] == "runtime"
                and candidate["in_rtl_delta"]
                and expression in candidate["syntax"]
            ]
            if len(consumers) == 1:
                _add_evidence(
                    consumers[0],
                    evidence_id=f"differential-elaboration-value:{statement_id}",
                    score=1.0,
                    distance=distances.get(differential_endpoints[0]),
                    authority="authoritative",
                )

    for node_id, node in nodes.items():
        signal = re.sub(r"\s*\[[0-9]+:[0-9]+\]$", "", str(node.get("signal", "")))
        name = signal.rsplit(".", 1)[-1]
        for statement_id in sorted(by_object.get(name, ())):
            _add_evidence(
                candidates[statement_id],
                evidence_id=node_id,
                score=0.0,
                distance=distances.get(node_id),
                authority="non_authoritative",
            )

    for row in candidates.values():
        if row["execution_phase"] == "elaboration":
            for origin in row["exact_origins"]:
                _add_evidence(
                    row,
                    evidence_id=f"origin:{origin.get('path')}:{origin.get('line')}",
                    score=0.0,
                    distance=None,
                    authority="authoritative",
                )

    for edge in graph.get("edges", []):
        if not isinstance(edge, Mapping):
            continue
        edge_id = str(edge.get("edge_id", ""))
        evidence = edge.get("rtl_evidence")
        locators = _locators(str(evidence.get("snippet", ""))) if isinstance(evidence, Mapping) else []
        matched = set()
        for locator in locators:
            matched.update(_match_candidates(candidates, locator))
        matched = _with_ancestors(candidates, matched)
        if not matched and edge_id:
            unmapped.append({"evidence_id": edge_id, "reason": "no_exact_statement_origin"})
        score = float(edge.get("contribution_score", 0.0) or 0.0)
        distance = min(
            (distances[node_id] for node_id in (str(edge.get("src_node_id", "")), str(edge.get("dst_node_id", ""))) if node_id in distances),
            default=None,
        )
        for statement_id in sorted(matched):
            _add_evidence(
                candidates[statement_id],
                evidence_id=edge_id,
                score=score,
                distance=distance,
                authority="authoritative",
            )
            if statement_id in delta_statement_ids:
                evidence_id = f"differential:{edge_id}"
                candidates[statement_id]["differential_evidence_ids"] = sorted(
                    set(candidates[statement_id]["differential_evidence_ids"]) | {evidence_id}
                )
                _add_evidence(
                    candidates[statement_id],
                    evidence_id=evidence_id,
                    score=1.0,
                    distance=distance,
                    authority="authoritative",
                )

    for semantic in graph.get("semantic_nodes", []):
        if not isinstance(semantic, Mapping) or semantic.get("type") != "source_provenance_hint":
            continue
        locator = _reported_locator(semantic)
        matched = _match_statements(candidates, locator) if locator else []
        if not matched:
            unmapped.append(
                {"evidence_id": str(semantic.get("semantic_id", "")), "reason": "locator_not_in_statement_index"}
            )
        for statement_id in matched:
            _add_evidence(
                candidates[statement_id],
                evidence_id=str(semantic.get("semantic_id", "")),
                score=0.0,
                distance=None,
                authority="non_authoritative",
            )

    rows = list(candidates.values())
    for row in rows:
        row["positive_authoritative_evidence"] = bool(
            row["max_contribution_score"] > 0.0 and row["authoritative_evidence_ids"]
        )
    rows.sort(key=_ordering_key)
    for position, row in enumerate(rows, 1):
        row["position"] = position
    for start, row in enumerate(rows):
        if start and _tie_key(rows[start - 1]) == _tie_key(row):
            continue
        stop = start
        while stop + 1 < len(rows) and _tie_key(rows[stop + 1]) == _tie_key(row):
            stop += 1
        average_rank = (start + stop + 2) / 2
        for index in range(start, stop + 1):
            rows[index]["rank"] = average_rank
            rows[index]["tie_size"] = stop - start + 1

    ambiguous = any(
        row.get("type") == "source_provenance_hint" and row.get("status") == "ambiguous"
        for row in graph.get("semantic_nodes", [])
        if isinstance(row, Mapping)
    )
    complete_projection = bool(rows) and not ambiguous
    return {
        "schema_version": "chiselcause_source_ranking.v3",
        "case_id": case_id,
        "method": method,
        "status": "complete" if complete_projection else "incomplete",
        "complete_graph": graph.get("status") == "complete",
        "complete_source_projection": complete_projection,
        "graph_id": graph.get("graph_id"),
        "policy_id": (graph.get("search_summary") or {}).get("policy_id"),
        "source_projection_status": "incomplete" if ambiguous else "statement_origin",
        "ordering": rows,
        "candidates": rows,
        "statement_candidate_count": len(rows),
        "source_entity_count": len(rows),
        "authoritative_candidate_count": sum(row["source_authority"] == "authoritative" for row in rows),
        "positive_authoritative_candidate_count": sum(row["positive_authoritative_evidence"] for row in rows),
        "differential_candidate_count": len(delta_statement_ids),
        "unmapped_evidence": sorted(unmapped, key=lambda row: (row["evidence_id"], row["reason"])),
        "tie_rule": "average_rank_by_positive_distance_contribution; statement_id_breaks_order_only",
        "ranking_sha256": canonical_sha256(rows),
    }


def _candidate(statement: Mapping[str, Any]) -> dict[str, Any]:
    anchor = statement.get("source_anchor") or {}
    return {
        "rank": 0,
        "position": 0,
        "tie_size": 0,
        "statement_id": str(statement["statement_id"]),
        "statement_kind": str(statement.get("statement_kind", "")),
        "entity_kind": str(statement.get("entity_kind", statement.get("statement_kind", ""))),
        "execution_phase": str(statement.get("execution_phase", "runtime")),
        "file": str(anchor.get("path", "")),
        "line": int(anchor.get("line_start", 0) or 0),
        "line_end": int(anchor.get("line_end", anchor.get("line_start", 0)) or 0),
        "column": int(statement.get("column_start", 1) or 1),
        "syntax": str(statement.get("syntax", "")),
        "semantic_object_ids": sorted(str(value) for value in statement.get("semantic_object_ids", [])),
        "parent_statement_id": statement.get("parent_statement_id"),
        "ancestor_statement_ids": sorted(str(value) for value in statement.get("ancestor_statement_ids", [])),
        "child_statement_ids": sorted(str(value) for value in statement.get("child_statement_ids", [])),
        "exact_origins": [dict(value) for value in statement.get("exact_origins", [])],
        "exact_origin_spec": dict(statement.get("exact_origin_spec") or {}),
        "origin_evidence": [{"kind": "scala_source_index", "authority": "authoritative", "source_anchor": dict(anchor)}],
        "source_authority": "non_authoritative",
        "evidence_node_ids": [],
        "authoritative_evidence_ids": [],
        "non_authoritative_evidence_ids": [],
        "differential_evidence_ids": [],
        "in_rtl_delta": False,
        "shortest_endpoint_distance": None,
        "max_contribution_score": 0.0,
        "score": 0.0,
    }


def _add_evidence(row: dict[str, Any], *, evidence_id: str, score: float, distance: int | None, authority: str) -> None:
    if not evidence_id:
        return
    bucket = "authoritative_evidence_ids" if authority == "authoritative" else "non_authoritative_evidence_ids"
    row[bucket] = sorted(set(row[bucket]) | {evidence_id})
    row["evidence_node_ids"] = sorted(set(row["evidence_node_ids"]) | {evidence_id})
    if authority == "authoritative":
        row["source_authority"] = "authoritative"
        row["max_contribution_score"] = round(max(row["max_contribution_score"], max(0.0, min(1.0, score))), 6)
        row["score"] = row["max_contribution_score"]
        if distance is not None:
            current = row["shortest_endpoint_distance"]
            row["shortest_endpoint_distance"] = distance if current is None else min(current, distance)


def _endpoint_distances(nodes: Mapping[str, Mapping[str, Any]], edges: Any) -> dict[str, int]:
    distances = {node_id: 0 for node_id, row in nodes.items() if row.get("is_endpoint")}
    pending = list(sorted(distances))
    incoming: dict[str, list[str]] = {}
    for edge in edges:
        if isinstance(edge, Mapping):
            incoming.setdefault(str(edge.get("dst_node_id", "")), []).append(str(edge.get("src_node_id", "")))
    while pending:
        target = pending.pop(0)
        for source in sorted(incoming.get(target, [])):
            distance = distances[target] + 1
            if source and (source not in distances or distance < distances[source]):
                distances[source] = distance
                pending.append(source)
    return distances


def _match_candidates(candidates: Mapping[str, Mapping[str, Any]], locator: Mapping[str, Any]) -> list[str]:
    phase = "elaboration" if str(locator.get("path", "")).endswith((".sv", ".v")) else "runtime"
    if phase == "elaboration":
        return sorted(
            row["statement_id"]
            for row in candidates.values()
            if any(
                origin.get("path") == locator.get("path")
                and origin.get("line") == locator.get("line")
                for origin in row["exact_origins"]
            )
        )
    matches = set(_match_statements(candidates, locator))
    for row in candidates.values():
        if row["execution_phase"] != "elaboration":
            continue
        if any(
            source.get("path") == locator.get("path")
            and source.get("line") == locator.get("line")
            for origin in row["exact_origins"]
            for source in origin.get("source_locators", [])
            if isinstance(source, Mapping)
        ):
            matches.add(str(row["statement_id"]))
    return sorted(matches)


def _with_ancestors(
    candidates: Mapping[str, Mapping[str, Any]], statement_ids: set[str]
) -> set[str]:
    result = set(statement_ids)
    for statement_id in list(statement_ids):
        result.update(
            ancestor
            for ancestor in candidates[statement_id].get("ancestor_statement_ids", [])
            if ancestor in candidates
        )
    return result


def _differential_statement_ids(
    candidates: Mapping[str, dict[str, Any]],
    *,
    elaboration: Mapping[str, Any],
    clean_rtl: str | Path | None,
    faulty_rtl: str | Path | None,
) -> set[str]:
    if clean_rtl is None and faulty_rtl is None:
        return set()
    if clean_rtl is None or faulty_rtl is None:
        raise ValueError("clean_rtl and faulty_rtl must be provided together")
    locators = _rtl_delta_locators(Path(clean_rtl), Path(faulty_rtl))
    matches: set[str] = set()
    for locator in locators:
        matches.update(_match_candidates(candidates, locator))
    selected = _selected_table_update(candidates, elaboration)
    if selected is not None:
        origins = {
            (str(row["path"]), int(row["line"]), int(row["column"]))
            for row in locators
            if str(row["path"]).endswith((".sv", ".v"))
        }
        if origins:
            matches.add(selected)
            candidates[selected]["exact_origins"] = [
                {
                    "authority": "authoritative",
                    "kind": "differential_table_update",
                    "path": path,
                    "line": line,
                    "column": column,
                    "source_locators": [],
                }
                for path, line, column in sorted(origins)
            ]
    return _with_ancestors(candidates, matches)


def _selected_table_update(
    candidates: Mapping[str, Mapping[str, Any]], elaboration: Mapping[str, Any]
) -> str | None:
    selected = []
    for statement_id, row in candidates.items():
        spec = row.get("exact_origin_spec") or {}
        updates = spec.get("updates") or []
        parameter = spec.get("selection_parameter")
        if (
            row.get("entity_kind") == "table_update"
            and not row.get("exact_origins")
            and isinstance(parameter, str)
            and isinstance(spec.get("selection_value"), int)
            and isinstance(spec.get("row_width"), int)
            and spec["row_width"] > 0
            and updates
            and all(
                isinstance(update, Mapping)
                and re.fullmatch(r"[0-9]+", str(update.get("row_expression", "")))
                for update in updates
            )
            and _selected_parameter(elaboration, parameter) == spec["selection_value"]
        ):
            selected.append(statement_id)
    return selected[0] if len(selected) == 1 else None


def _selected_parameter(elaboration: Mapping[str, Any], parameter: str) -> int | None:
    for values in (elaboration.get("commands") or {}).values():
        if not isinstance(values, list):
            continue
        for value in values:
            match = re.search(
                rf"(?:^|\s){re.escape(parameter)}=([0-9]+)(?:\s|$)", str(value)
            )
            if match:
                return int(match.group(1))
    return None


def _rtl_delta_locators(clean_rtl: Path, faulty_rtl: Path) -> list[dict[str, Any]]:
    clean = clean_rtl.read_text(encoding="utf-8").splitlines()
    faulty = faulty_rtl.read_text(encoding="utf-8").splitlines()
    rows: list[dict[str, Any]] = []
    matcher = difflib.SequenceMatcher(a=clean, b=faulty, autojunk=False)
    for tag, clean_start, clean_stop, faulty_start, faulty_stop in matcher.get_opcodes():
        if tag == "equal":
            continue
        for path, lines, start, stop in (
            (clean_rtl, clean, clean_start, clean_stop),
            (faulty_rtl, faulty, faulty_start, faulty_stop),
        ):
            for index in range(start, stop):
                rows.extend(_locators(lines[index]))
                rows.append(
                    {
                        "path": _generated_rtl_path(path),
                        "line": index + 1,
                        "column": 1,
                    }
                )
    unique = {
        (row["path"], row["line"], row["column"]): row
        for row in rows
    }
    return [unique[key] for key in sorted(unique)]


def _generated_rtl_path(path: Path) -> str:
    parts = path.parts
    return (
        Path(*parts[parts.index("specflow-generated") :]).as_posix()
        if "specflow-generated" in parts
        else path.name
    )


def _match_statements(candidates: Mapping[str, Mapping[str, Any]], locator: Mapping[str, Any]) -> list[str]:
    path = str(locator.get("path", ""))
    line = int(locator.get("line", 0) or 0)
    column = int(locator.get("column", 1) or 1)
    matches = [
        row for row in candidates.values()
        if row["execution_phase"] == "runtime"
        and row["file"] == path and row["line"] <= line <= row["line_end"]
    ]
    if not matches:
        return []
    exact_line = [row for row in matches if row["line"] == line]
    pool = exact_line or matches
    best = min((row["line_end"] - row["line"], abs(row["column"] - column)) for row in pool)
    return sorted(row["statement_id"] for row in pool if (row["line_end"] - row["line"], abs(row["column"] - column)) == best)


def _locators(text: str) -> list[dict[str, Any]]:
    matches = list(_LOCATOR_RE.finditer(text))
    rows = [{"path": match.group("path"), "line": int(match.group("line")), "column": _column(match.group("column"))} for match in matches]
    if matches:
        last = matches[-1]
        rows.extend({"path": last.group("path"), "line": int(match.group("line")), "column": _column(match.group("column"))} for match in _SHORT_LOCATOR_RE.finditer(text, last.end()))
    return rows


def _reported_locator(row: Mapping[str, Any]) -> dict[str, Any] | None:
    match = re.fullmatch(r"(?P<line>[0-9]+):(?P<column>[0-9]+|\{[0-9,]+\})", str(row.get("reported_locator", "")))
    if not match:
        return None
    return {"path": row.get("reported_path"), "line": int(match.group("line")), "column": _column(match.group("column"))}


def _column(value: Any) -> int:
    match = re.search(r"[0-9]+", str(value))
    return int(match.group()) if match else 1


def _tie_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        not row["positive_authoritative_evidence"],
        row["shortest_endpoint_distance"] if row["shortest_endpoint_distance"] is not None else 10**9,
        -row["max_contribution_score"],
    )


def _ordering_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return (*_tie_key(row), row["statement_id"])
