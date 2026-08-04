"""Aggregate causal evidence into a deterministic Chisel source ranking."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence


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
    object_locators: dict[str, list[Mapping[str, Any]]] = {}
    for obj in elaboration.get("objects", []):
        locator = obj.get("source_locator")
        if not isinstance(locator, Mapping):
            continue
        object_locators.setdefault(str(obj.get("name", "")), []).append(locator)
        _merge(candidates, locator, score=0.0, instance="", evidence_ids=())

    # Freeze a method-independent source candidate universe before scoring.
    root = Path(source_root)
    for locator in elaboration.get("source_locators", []):
        path = str(locator.get("path", ""))
        source = root / path
        if not source.is_file():
            continue
        for number, text in enumerate(source.read_text(encoding="utf-8").splitlines(), 1):
            stripped = text.strip()
            if stripped and not stripped.startswith("//") and stripped not in {"{", "}"}:
                _merge(
                    candidates,
                    {"path": path, "line": number, "column": 1},
                    score=0.0,
                    instance="",
                    evidence_ids=(),
                )

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
        for locator in _locators(str(evidence.get("snippet", ""))):
            _merge(
                candidates,
                locator,
                score=score,
                instance=instance,
                evidence_ids=evidence_ids,
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
                score=0.0,
                instance="",
                evidence_ids=(str(semantic.get("semantic_id", "")),),
            )

    rows = sorted(
        candidates.values(),
        key=lambda row: (
            -row["score"],
            -len(row["evidence_node_ids"]),
            row["file"],
            row["line"],
            row["column"],
        ),
    )
    previous = None
    rank = 0
    for index, row in enumerate(rows, 1):
        key = (row["score"], len(row["evidence_node_ids"]))
        if key != previous:
            rank = index
            previous = key
        row["rank"] = rank
    return {
        "schema_version": "chiselcause_source_ranking.v1",
        "case_id": case_id,
        "method": method,
        "status": "complete" if any(row["evidence_node_ids"] for row in rows) else "rtl_only",
        "graph_id": graph.get("graph_id"),
        "ordering": rows,
        "tie_rule": "score_then_evidence_count_then_source_location",
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
) -> None:
    path = str(locator.get("path", ""))
    line = int(locator.get("line", 0) or 0)
    if not path.endswith(".scala") or line <= 0:
        return
    key = (path, line)
    row = candidates.setdefault(
        key,
        {
            "rank": 0,
            "score": 0.0,
            "instance": instance,
            "file": path,
            "line": line,
            "column": _column(locator.get("column", 1)),
            "evidence_node_ids": [],
        },
    )
    # Compiler temporaries that map to one statement contribute once.
    if score > row["score"]:
        row["score"] = round(score, 6)
    if instance and (not row["instance"] or instance < row["instance"]):
        row["instance"] = instance
    row["evidence_node_ids"] = sorted(
        set(row["evidence_node_ids"]) | {item for item in evidence_ids if item}
    )
