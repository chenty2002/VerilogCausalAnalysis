"""Canonical identities used by the deterministic structural API."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Iterable, Mapping


ANALYZER_REVISION = "verilog-causal-analysis-ls-e-v2"
HDLCONVERTOR_REVISION = "03c081a307850dc3c438df36592fc67b1ef6cfc0"


def canonical_json_bytes(value: Any) -> bytes:
    """Return the single canonical JSON representation used by structural artifacts."""
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: str | Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            size += len(chunk)
            digest.update(chunk)
    return digest.hexdigest(), size


def canonical_sha256(value: Any) -> str:
    return sha256_bytes(canonical_json_bytes(value))


def stable_id(prefix: str, *parts: Any, length: int | None = None) -> str:
    digest = canonical_sha256(list(parts))
    return prefix + (digest if length is None else digest[:length])


def stable_set_sha256(rows: Iterable[Mapping[str, Any]]) -> str:
    normalized = sorted((dict(row) for row in rows), key=canonical_json_bytes)
    return canonical_sha256(normalized)


def contains_absolute_path(value: Any) -> bool:
    """Conservatively detect leaked POSIX absolute paths in a JSON value."""
    if isinstance(value, str):
        return bool(re.fullmatch(r"/(?:[A-Za-z0-9_.-]+/)+[^\s]*", value))
    if isinstance(value, Mapping):
        return any(contains_absolute_path(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(contains_absolute_path(item) for item in value)
    return False
