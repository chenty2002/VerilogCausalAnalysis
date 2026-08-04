"""Fail-closed generated-RTL provenance hints for the C6 Chisel profile."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional

from .identity import canonical_sha256, sha256_file, stable_id


SOURCE_ANNOTATION_SCHEMA = "chisel_source_annotations"
SOURCE_PROVENANCE_FEATURE = "source_provenance"
_LOCATOR_RE = re.compile(
    r"@\[(?P<path>[A-Za-z0-9_./-]+\.scala)\s+"
    r"(?P<locator>[0-9]+:(?:[0-9]+|\{[0-9,]+\}))\]"
)
_CIRCT_LOCATOR_RE = re.compile(
    r"(?P<path>[A-Za-z0-9_./-]+\.scala):"
    r"(?P<line>[0-9]+):(?P<column>[0-9]+|\{[0-9,]+\})"
)
_CIRCT_SHORT_LOCATOR_RE = re.compile(
    r"(?:^|,)\s*:(?P<line>[0-9]+):(?P<column>[0-9]+|\{[0-9,]+\})"
)


class ProvenanceError(ValueError):
    """A source sidecar or locator cannot be joined without guessing."""


def c6_enabled(features: Iterable[str]) -> bool:
    return SOURCE_PROVENANCE_FEATURE in set(features)


def load_source_annotations(
    path: str,
    *,
    sha256: str,
    bytes: int,
    rtl_set_sha256: str,
    known_statement_ids: set[str],
) -> Dict[str, Any]:
    actual_sha256, actual_bytes = sha256_file(path)
    if (actual_sha256, actual_bytes) != (sha256, bytes):
        raise ProvenanceError("source annotation artifact hash or size mismatch")
    try:
        row = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ProvenanceError("source annotation artifact is not valid JSON") from error
    if not isinstance(row, Mapping) or set(row) != {
        "schema_version",
        "rtl_set_sha256",
        "mappings",
    }:
        raise ProvenanceError("source annotation artifact has invalid exact fields")
    if row["schema_version"] != SOURCE_ANNOTATION_SCHEMA:
        raise ProvenanceError("source annotation schema is unsupported")
    if row["rtl_set_sha256"] != rtl_set_sha256:
        raise ProvenanceError("source annotation rtl_set_sha256 mismatch")
    mappings = row["mappings"]
    if not isinstance(mappings, list):
        raise ProvenanceError("source annotation mappings must be a list")
    normalized = []
    identities = set()
    for index, item in enumerate(mappings):
        if not isinstance(item, Mapping) or set(item) != {
            "statement_id",
            "reported_path",
            "reported_locator",
        }:
            raise ProvenanceError(
                f"source annotation mapping {index} has invalid exact fields"
            )
        statement_id = item["statement_id"]
        reported_path = item["reported_path"]
        reported_locator = item["reported_locator"]
        if statement_id not in known_statement_ids:
            raise ProvenanceError("source annotation references an unknown statement")
        if (
            not isinstance(reported_path, str)
            or not reported_path
            or Path(reported_path).is_absolute()
            or ".." in Path(reported_path).parts
            or not reported_path.endswith(".scala")
        ):
            raise ProvenanceError("source annotation path must be safe and relative")
        if (
            not isinstance(reported_locator, str)
            or not re.fullmatch(r"[0-9]+:(?:[0-9]+|\{[0-9,]+\})", reported_locator)
        ):
            raise ProvenanceError("source annotation locator is invalid")
        identity = (statement_id, reported_path, reported_locator)
        if identity in identities:
            raise ProvenanceError("source annotation contains a duplicate mapping")
        identities.add(identity)
        normalized.append(dict(item))
    return {
        "schema_version": SOURCE_ANNOTATION_SCHEMA,
        "rtl_set_sha256": rtl_set_sha256,
        "mappings": sorted(
            normalized,
            key=lambda item: (
                item["statement_id"],
                item["reported_path"],
                item["reported_locator"],
            ),
        ),
        "artifact_sha256": sha256,
    }


def build_provenance_hints(
    instance_graph: Any,
    *,
    rtl_set_sha256: str,
    annotations: Optional[Mapping[str, Any]] = None,
) -> tuple[list[Dict[str, Any]], list[Dict[str, Any]]]:
    """Collect reversible hints without granting Chisel source authority."""

    parser = instance_graph._parser
    artifact_by_module = {
        row.module_name: row.rtl_artifact_id for row in instance_graph.instances
    }
    hints: Dict[tuple[str, str, str, str], Dict[str, Any]] = {}
    diagnostics: list[Dict[str, Any]] = []
    source_lines: Dict[Path, Optional[list[str]]] = {}
    for statement_id, statement in sorted(parser._statement_evidence.items()):
        if statement.line_start <= 0 or not statement.file_path:
            continue
        path = Path(statement.file_path)
        if path not in source_lines:
            try:
                source_lines[path] = path.read_text(
                    encoding="utf-8", errors="replace"
                ).splitlines()
            except OSError:
                source_lines[path] = None
        lines = source_lines[path]
        if lines is None:
            diagnostics.append(
                {
                    "code": "provenance_rtl_unreadable",
                    "message": statement_id,
                    "breaks_complete": True,
                }
            )
            continue
        # FIRRTL/CIRCT locators normally precede the generated statement.
        start = max(0, statement.line_start - 4)
        end = min(len(lines), max(statement.line_end, statement.line_start))
        window = "\n".join(lines[start:end])
        matches = list(_locator_matches(window))
        if not matches:
            continue
        nearest_offset = max(match[0] for match in matches)
        for match in matches:
            if match[0] != nearest_offset:
                continue
            _add_hint(
                hints,
                rtl_set_sha256=rtl_set_sha256,
                rtl_artifact_id=artifact_by_module.get(statement.module_name),
                statement_id=statement_id,
                reported_path=match[1],
                reported_locator=match[2],
                status="unverified_hint",
                inference_rule="firrtl_locator_comment",
                annotation_sha256=None,
            )
    if annotations is not None:
        for item in annotations["mappings"]:
            statement = parser._statement_evidence[item["statement_id"]]
            _add_hint(
                hints,
                rtl_set_sha256=rtl_set_sha256,
                rtl_artifact_id=artifact_by_module.get(statement.module_name),
                statement_id=item["statement_id"],
                reported_path=item["reported_path"],
                reported_locator=item["reported_locator"],
                status="source_projection_candidate",
                inference_rule="hash_bound_annotation_sidecar",
                annotation_sha256=annotations["artifact_sha256"],
            )
    by_statement: Dict[str, set[tuple[str, str]]] = {}
    for hint in hints.values():
        by_statement.setdefault(hint["rtl_statement_id"], set()).add(
            (hint["reported_path"], hint["reported_locator"])
        )
    for statement_id, locators in sorted(by_statement.items()):
        if len(locators) > 1:
            diagnostics.append(
                {
                    "code": "source_projection_ambiguous",
                    "message": statement_id,
                    "breaks_complete": True,
                }
            )
            for hint in hints.values():
                if hint["rtl_statement_id"] == statement_id:
                    hint["status"] = "ambiguous"
    return (
        sorted(hints.values(), key=lambda item: item["hint_id"]),
        diagnostics,
    )


def _locator_matches(text: str) -> list[tuple[int, str, str]]:
    """Return both legacy FIRRTL and current CIRCT source locators."""

    rows = [
        (match.start(), match.group("path"), match.group("locator"))
        for match in _LOCATOR_RE.finditer(text)
    ]
    rows.extend(
        (
            match.start(),
            match.group("path"),
            f"{match.group('line')}:{match.group('column')}",
        )
        for match in _CIRCT_LOCATOR_RE.finditer(text)
    )
    circt = list(_CIRCT_LOCATOR_RE.finditer(text))
    if circt:
        last = circt[-1]
        rows.extend(
            (
                match.start(),
                last.group("path"),
                f"{match.group('line')}:{match.group('column')}",
            )
            for match in _CIRCT_SHORT_LOCATOR_RE.finditer(text, last.end())
        )
    return sorted(set(rows))


def _add_hint(
    hints: Dict[tuple[str, str, str, str], Dict[str, Any]],
    *,
    rtl_set_sha256: str,
    rtl_artifact_id: Optional[str],
    statement_id: str,
    reported_path: str,
    reported_locator: str,
    status: str,
    inference_rule: str,
    annotation_sha256: Optional[str],
) -> None:
    key = (statement_id, reported_path, reported_locator, inference_rule)
    identity = {
        "rtl_set_sha256": rtl_set_sha256,
        "rtl_statement_id": statement_id,
        "reported_path": reported_path,
        "reported_locator": reported_locator,
        "inference_rule": inference_rule,
        "annotation_sha256": annotation_sha256,
    }
    hints[key] = {
        "hint_id": stable_id("vcph_", identity, length=24),
        "rtl_artifact_id": rtl_artifact_id,
        "rtl_statement_id": statement_id,
        "reported_path": reported_path,
        "reported_locator": reported_locator,
        "status": status,
        "inference_rule": inference_rule,
        "annotation_sha256": annotation_sha256,
        "authority": "non_authoritative",
        "evidence_sha256": canonical_sha256(identity),
    }
