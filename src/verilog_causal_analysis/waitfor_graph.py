"""Bounded C5 wait-for construction and fail-closed candidate classification.

The generic rules in this module only compose semantic objects that already
carry structural or temporal evidence.  Protocol labels and request/response
pairs are accepted only from a hash-bound, Codex-reviewed adapter asset.
Neither an SCC nor a ``deadlock_candidate`` is a formal deadlock verdict.
"""

from __future__ import annotations

import json
from collections import deque
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

from .identity import canonical_sha256, sha256_file, stable_id


WAITFOR_FEATURE = "waitfor"
PROTOCOL_ADAPTER_SCHEMA = "reviewed_protocol_adapter.v1"
_PROTOCOL_EVENTS = {
    "Probe",
    "ProbeAck",
    "ProbeAckData",
    "AcquireBlock",
    "Grant",
    "GrantData",
    "Release",
    "ReleaseData",
}


class WaitForError(ValueError):
    pass


def c5_enabled(features: Sequence[str]) -> bool:
    return WAITFOR_FEATURE in set(features)


def _exact_keys(
    row: Mapping[str, Any], expected: Iterable[str], where: str
) -> None:
    expected_set = set(expected)
    if set(row) != expected_set:
        raise WaitForError(
            f"{where} keys mismatch: "
            f"missing={sorted(expected_set - set(row))}, "
            f"extra={sorted(set(row) - expected_set)}"
        )


def validate_protocol_adapter(
    row: Mapping[str, Any],
    *,
    rtl_set_sha256: str,
    known_semantic_ids: Iterable[str],
) -> Dict[str, Any]:
    """Validate one optional reviewed protocol asset against this exact graph."""
    _exact_keys(
        row,
        {
            "schema_version",
            "adapter_id",
            "protocol",
            "rtl_set_sha256",
            "review",
            "channels",
            "dependencies",
        },
        "protocol_adapter",
    )
    if row["schema_version"] != PROTOCOL_ADAPTER_SCHEMA:
        raise WaitForError(
            f"protocol adapter schema must be {PROTOCOL_ADAPTER_SCHEMA}"
        )
    if row["protocol"] != "tilelink":
        raise WaitForError("C5 supports only the optional TileLink adapter")
    if row["rtl_set_sha256"] != rtl_set_sha256:
        raise WaitForError("protocol adapter rtl_set_sha256 mismatch")
    review = row["review"]
    if not isinstance(review, Mapping):
        raise WaitForError("protocol_adapter.review must be an object")
    _exact_keys(review, {"status", "reviewer", "evidence_refs"}, "review")
    if review["status"] != "approved" or review["reviewer"] != "codex":
        raise WaitForError(
            "protocol adapter must be approved by reviewer codex"
        )
    if (
        not isinstance(review["evidence_refs"], list)
        or not review["evidence_refs"]
        or any(
            not isinstance(item, str) or not item
            for item in review["evidence_refs"]
        )
    ):
        raise WaitForError("protocol adapter review requires evidence_refs")

    known = set(known_semantic_ids)
    channels = row["channels"]
    if not isinstance(channels, list):
        raise WaitForError("protocol_adapter.channels must be a list")
    channel_by_id: Dict[str, Dict[str, Any]] = {}
    for index, item in enumerate(channels):
        if not isinstance(item, Mapping):
            raise WaitForError(f"channels[{index}] must be an object")
        _exact_keys(
            item,
            {"channel_id", "handshake_id", "role", "event"},
            f"channels[{index}]",
        )
        channel_id = item["channel_id"]
        handshake_id = item["handshake_id"]
        if (
            not isinstance(channel_id, str)
            or not channel_id
            or channel_id in channel_by_id
        ):
            raise WaitForError("protocol adapter channel IDs must be unique")
        if handshake_id not in known:
            raise WaitForError(
                f"protocol adapter handshake is absent: {handshake_id}"
            )
        if item["role"] not in {"request", "response"}:
            raise WaitForError("protocol channel role must be request/response")
        if item["event"] not in _PROTOCOL_EVENTS:
            raise WaitForError("unsupported reviewed TileLink event")
        channel_by_id[channel_id] = dict(item)

    dependencies = row["dependencies"]
    if not isinstance(dependencies, list):
        raise WaitForError("protocol_adapter.dependencies must be a list")
    normalized_dependencies = []
    for index, item in enumerate(dependencies):
        if not isinstance(item, Mapping):
            raise WaitForError(f"dependencies[{index}] must be an object")
        _exact_keys(
            item,
            {
                "waiter_ref",
                "awaited_ref",
                "inference_rule",
                "evidence_refs",
            },
            f"dependencies[{index}]",
        )
        for key in ("waiter_ref", "awaited_ref"):
            ref = item[key]
            if not isinstance(ref, str) or ":" not in ref:
                raise WaitForError(f"{key} must be channel:<id> or semantic:<id>")
            kind, value = ref.split(":", 1)
            if (
                (kind == "channel" and value not in channel_by_id)
                or (kind == "semantic" and value not in known)
                or kind not in {"channel", "semantic"}
            ):
                raise WaitForError(f"unresolved protocol dependency ref: {ref}")
        if (
            not isinstance(item["inference_rule"], str)
            or not item["inference_rule"].startswith("reviewed_tilelink.")
        ):
            raise WaitForError(
                "protocol dependency rule must use reviewed_tilelink.*"
            )
        if (
            not isinstance(item["evidence_refs"], list)
            or not item["evidence_refs"]
        ):
            raise WaitForError(
                "protocol dependency requires non-empty evidence_refs"
            )
        normalized_dependencies.append(
            {
                **dict(item),
                "evidence_refs": sorted(set(item["evidence_refs"])),
            }
        )

    normalized = {
        **dict(row),
        "review": {
            **dict(review),
            "evidence_refs": sorted(set(review["evidence_refs"])),
        },
        "channels": [
            channel_by_id[key] for key in sorted(channel_by_id)
        ],
        "dependencies": sorted(
            normalized_dependencies,
            key=lambda item: (
                item["waiter_ref"],
                item["awaited_ref"],
                item["inference_rule"],
            ),
        ),
    }
    expected_id = stable_id(
        "vcpa_",
        {key: value for key, value in normalized.items() if key != "adapter_id"},
    )
    if normalized["adapter_id"] != expected_id:
        raise WaitForError("protocol adapter_id mismatch")
    return normalized


def load_protocol_adapter(
    path: str,
    *,
    sha256: str,
    bytes: int,
    rtl_set_sha256: str,
    known_semantic_ids: Iterable[str],
) -> Dict[str, Any]:
    actual_sha256, actual_bytes = sha256_file(Path(path))
    if actual_sha256 != sha256 or actual_bytes != bytes:
        raise WaitForError("protocol adapter artifact hash/size mismatch")
    try:
        row = json.loads(Path(path).read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise WaitForError(f"protocol adapter is not valid JSON: {error}") from error
    if not isinstance(row, Mapping):
        raise WaitForError("protocol adapter must contain one object")
    return validate_protocol_adapter(
        row,
        rtl_set_sha256=rtl_set_sha256,
        known_semantic_ids=known_semantic_ids,
    )


def make_protocol_adapter(**kwargs: Any) -> Dict[str, Any]:
    provisional = {
        "schema_version": PROTOCOL_ADAPTER_SCHEMA,
        "adapter_id": "pending",
        **kwargs,
    }
    if isinstance(provisional.get("review"), Mapping):
        provisional["review"] = {
            **dict(provisional["review"]),
            "evidence_refs": sorted(
                set(provisional["review"].get("evidence_refs", []))
            ),
        }
    provisional["channels"] = sorted(
        [dict(item) for item in provisional.get("channels", [])],
        key=lambda item: str(item.get("channel_id", "")),
    )
    provisional["dependencies"] = sorted(
        [
            {
                **dict(item),
                "evidence_refs": sorted(
                    set(item.get("evidence_refs", []))
                ),
            }
            for item in provisional.get("dependencies", [])
        ],
        key=lambda item: (
            str(item.get("waiter_ref", "")),
            str(item.get("awaited_ref", "")),
            str(item.get("inference_rule", "")),
        ),
    )
    provisional["adapter_id"] = stable_id(
        "vcpa_",
        {
            key: value
            for key, value in provisional.items()
            if key != "adapter_id"
        },
    )
    return provisional


def _edge_endpoints(edge: Mapping[str, Any]) -> Tuple[Optional[str], Optional[str]]:
    source = edge.get("src_semantic_id")
    target = edge.get("dst_semantic_id")
    return (
        str(source) if source is not None else None,
        str(target) if target is not None else None,
    )


def _overlaps(row: Mapping[str, Any], start: int, end: int) -> bool:
    row_start = int(row.get("start_cycle", end))
    row_end = int(row.get("end_cycle", end))
    return row_start <= end and row_end >= start


def _shortest_path(
    start: str,
    targets: set[str],
    adjacency: Mapping[str, set[str]],
    *,
    max_length: int = 8,
) -> Optional[list[str]]:
    queue = deque([(start, [start])])
    visited = {start}
    while queue:
        current, path = queue.popleft()
        if current in targets and current != start:
            return path
        if len(path) - 1 >= max_length:
            continue
        for neighbor in sorted(adjacency.get(current, ())):
            if neighbor in visited:
                continue
            visited.add(neighbor)
            queue.append((neighbor, path + [neighbor]))
    return None


def _strong_components(
    members: Iterable[str], adjacency: Mapping[str, set[str]]
) -> list[list[str]]:
    """Deterministic Tarjan SCC."""
    index = 0
    indexes: Dict[str, int] = {}
    lowlinks: Dict[str, int] = {}
    stack: list[str] = []
    on_stack: set[str] = set()
    result: list[list[str]] = []

    def visit(node: str) -> None:
        nonlocal index
        indexes[node] = index
        lowlinks[node] = index
        index += 1
        stack.append(node)
        on_stack.add(node)
        for neighbor in sorted(adjacency.get(node, ())):
            if neighbor not in indexes:
                visit(neighbor)
                lowlinks[node] = min(lowlinks[node], lowlinks[neighbor])
            elif neighbor in on_stack:
                lowlinks[node] = min(lowlinks[node], indexes[neighbor])
        if lowlinks[node] != indexes[node]:
            return
        component = []
        while stack:
            member = stack.pop()
            on_stack.remove(member)
            component.append(member)
            if member == node:
                break
        result.append(sorted(component))

    for member in sorted(set(members)):
        if member not in indexes:
            visit(member)
    return sorted(result, key=lambda row: tuple(row))


def _weak_components(
    members: Iterable[str], edges: Sequence[Mapping[str, Any]]
) -> list[list[str]]:
    adjacency: Dict[str, set[str]] = {}
    for edge in edges:
        waiter = str(edge["waiter_id"])
        awaited = str(edge["awaited_id"])
        adjacency.setdefault(waiter, set()).add(awaited)
        adjacency.setdefault(awaited, set()).add(waiter)
    result = []
    seen: set[str] = set()
    for member in sorted(set(members)):
        if member in seen:
            continue
        queue = deque([member])
        seen.add(member)
        component = []
        while queue:
            current = queue.popleft()
            component.append(current)
            for neighbor in sorted(adjacency.get(current, ())):
                if neighbor not in seen:
                    seen.add(neighbor)
                    queue.append(neighbor)
        result.append(sorted(component))
    return result


def build_c5_waitfor_layer(
    *,
    semantic_nodes: Sequence[Mapping[str, Any]],
    edges: Sequence[Mapping[str, Any]],
    root_candidates: Sequence[Mapping[str, Any]],
    endpoint_cycle: int,
    max_waitfor_nodes: int,
    max_waitfor_edges: int,
    max_scc_candidates: int,
    protocol_adapter: Optional[Mapping[str, Any]] = None,
    rtl_set_sha256: Optional[str] = None,
    max_total_edges: Optional[int] = None,
) -> Tuple[
    list[Dict[str, Any]],
    list[Dict[str, Any]],
    list[Dict[str, Any]],
    list[Dict[str, Any]],
    Dict[str, int | bool],
]:
    """Compose C0-C4 evidence into bounded wait-for components."""
    nodes = [dict(row) for row in semantic_nodes]
    graph_edges = [dict(row) for row in edges]
    roots = [dict(row) for row in root_candidates]
    diagnostics: list[Dict[str, Any]] = []
    node_by_id = {str(row["semantic_id"]): row for row in nodes}
    structural: Dict[str, set[str]] = {}
    for edge in graph_edges:
        source, target = _edge_endpoints(edge)
        if source is None or target is None:
            continue
        structural.setdefault(source, set()).add(target)
        structural.setdefault(target, set()).add(source)

    failure_start = min(
        [
            int(row["start_cycle"])
            for row in nodes
            if row.get("type")
            in {
                "persistent_interval",
                "missing_expected_completion",
                "stall_interval",
                "arbiter_wait",
                "allocation_wait",
            }
            and int(row.get("end_cycle", endpoint_cycle)) >= endpoint_cycle
        ]
        or [endpoint_cycle]
    )
    wait_edges: list[Dict[str, Any]] = []
    wait_nodes: set[str] = set()
    generated_nodes: Dict[str, Dict[str, Any]] = {}
    edge_bound_hit = False
    node_bound_hit = False
    reserved_protocol_edges = (
        len(protocol_adapter.get("channels", []))
        if isinstance(protocol_adapter, Mapping)
        and isinstance(protocol_adapter.get("channels"), list)
        else 0
    )
    total_edge_capacity = (
        max_waitfor_edges
        if max_total_edges is None
        else max(
            0,
            min(
                max_waitfor_edges,
                max_total_edges
                - len(graph_edges)
                - reserved_protocol_edges,
            ),
        )
    )
    protocol_materialization_allowed = (
        max_total_edges is None
        or len(graph_edges) + reserved_protocol_edges <= max_total_edges
    )
    if not protocol_materialization_allowed:
        edge_bound_hit = True

    def add_external(kind: str, owner_id: str, reason: str) -> str:
        external_id = stable_id(
            "vcs_",
            "unknown_external_completion",
            kind,
            owner_id,
            failure_start,
            endpoint_cycle,
            reason,
            length=24,
        )
        generated_nodes.setdefault(
            external_id,
            {
                "semantic_id": external_id,
                "type": "unknown_external_completion",
                "completion_kind": kind,
                "owner_id": owner_id,
                "start_cycle": failure_start,
                "end_cycle": endpoint_cycle,
                "external": True,
                "reason": reason,
                "evidence_strength": "unresolved",
                "inference_rule": "unresolved_external_completion.v1",
            },
        )
        return external_id

    def add_wait(
        waiter_id: str,
        awaited_id: str,
        *,
        start_cycle: int,
        end_cycle: int,
        guard_semantic_ids: Sequence[str],
        evidence_refs: Sequence[str],
        evidence_strength: str,
        inference_rule: str,
    ) -> None:
        nonlocal edge_bound_hit, node_bound_hit
        if len(wait_edges) >= total_edge_capacity:
            edge_bound_hit = True
            return
        new_members = {waiter_id, awaited_id} - wait_nodes
        if len(wait_nodes) + len(new_members) > max_waitfor_nodes:
            node_bound_hit = True
            return
        wait_nodes.update(new_members)
        identity = {
            "waiter_id": waiter_id,
            "awaited_id": awaited_id,
            "start_cycle": start_cycle,
            "end_cycle": end_cycle,
            "guard_semantic_ids": sorted(set(guard_semantic_ids)),
            "evidence_refs": sorted(set(evidence_refs)),
            "evidence_strength": evidence_strength,
            "inference_rule": inference_rule,
        }
        edge_id = stable_id("vcw_", identity, length=24)
        if any(row["edge_id"] == edge_id for row in wait_edges):
            return
        wait_edges.append({"edge_id": edge_id, **identity})

    missing_rows = [
        row
        for row in nodes
        if row.get("type") == "missing_expected_completion"
        and _overlaps(row, failure_start, endpoint_cycle)
    ]
    for row in sorted(missing_rows, key=lambda item: item["semantic_id"]):
        missing_id = str(row["semantic_id"])
        waiter_id = str(row["register_id"])
        if waiter_id not in node_by_id:
            waiter_id = missing_id
        external = add_external(
            "register_clear_or_response",
            missing_id,
            "completion provider is not proven by generic RTL semantics",
        )
        add_wait(
            waiter_id,
            external,
            start_cycle=max(failure_start, int(row["start_cycle"])),
            end_cycle=min(endpoint_cycle, int(row["end_cycle"])),
            guard_semantic_ids=[missing_id],
            evidence_refs=[
                missing_id,
                *[str(item) for item in row.get("evidence_refs", [])],
            ],
            evidence_strength="interval_rule_derived",
            inference_rule="active_state_waits_for_completion.v1",
        )

    stalls = [
        row
        for row in nodes
        if row.get("type") == "stall_interval"
        and _overlaps(row, failure_start, endpoint_cycle)
    ]
    blockers = {
        str(row["semantic_id"]): row
        for row in nodes
        if row.get("type") == "blocking_relation"
    }
    blocker_ids = set(blockers)
    blocker_release_waiters: set[str] = set()
    for row in sorted(stalls, key=lambda item: item["semantic_id"]):
        stall_id = str(row["semantic_id"])
        path = _shortest_path(stall_id, blocker_ids, structural)
        if path is not None:
            blocker_id = path[-1]
            add_wait(
                stall_id,
                blocker_id,
                start_cycle=max(failure_start, int(row["start_cycle"])),
                end_cycle=min(endpoint_cycle, int(row["end_cycle"])),
                guard_semantic_ids=path[1:-1],
                evidence_refs=path,
                evidence_strength=str(
                    row.get("evidence_strength", "transition_supported")
                ),
                inference_rule="pipeline_admission_waits_for_blocker.v1",
            )
            external = add_external(
                "pipeline_blocker_release",
                blocker_id,
                "blocker release has no closed generic completion proof",
            )
            add_wait(
                blocker_id,
                external,
                start_cycle=max(failure_start, int(row["start_cycle"])),
                end_cycle=min(endpoint_cycle, int(row["end_cycle"])),
                guard_semantic_ids=[blocker_id],
                evidence_refs=[
                    blocker_id,
                    *[
                        str(item)
                        for item in blockers[blocker_id].get(
                            "statement_ids", []
                        )
                    ],
                ],
                evidence_strength="exact_structural",
                inference_rule="pipeline_blocker_waits_for_release.v1",
            )
            blocker_release_waiters.add(blocker_id)
        else:
            external = add_external(
                "ready_assertion",
                stall_id,
                "ready source is outside the recovered semantic component",
            )
            add_wait(
                stall_id,
                external,
                start_cycle=max(failure_start, int(row["start_cycle"])),
                end_cycle=min(endpoint_cycle, int(row["end_cycle"])),
                guard_semantic_ids=[str(row["handshake_id"])],
                evidence_refs=[stall_id, str(row["handshake_id"])],
                evidence_strength=str(
                    row.get("evidence_strength", "transition_supported")
                ),
                inference_rule="valid_ready_stall_waits_for_ready.v1",
            )

    # C4 certifies a small deterministic subset of blockers whose exact
    # waveform value stayed asserted throughout the failure window.  C5 may
    # materialize those as open resource waits even when no ready/valid stall
    # object exists.  Related ranked paths are evidence guards only; they do
    # not become wait edges or SCC proof.
    for root in sorted(
        roots,
        key=lambda item: (
            str(item.get("candidate_id", "")),
            str(item["semantic_id"]),
        ),
    ):
        seed = root.get("seed") or {}
        if seed.get("derivation_rule") != "persistent_pipeline_blocker.v1":
            continue
        blocker_id = str(root["semantic_id"])
        blocker = blockers.get(blocker_id)
        if blocker is None or blocker_id in blocker_release_waiters:
            continue
        interval = seed.get("interval")
        if (
            not isinstance(interval, list)
            or len(interval) != 2
            or any(not isinstance(item, int) for item in interval)
        ):
            continue
        pipeline_ids = sorted(
            {
                str(item["pipeline_id"])
                for item in blocker.get("blockers", [])
                if str(item["pipeline_id"]) in node_by_id
            }
        )
        related_paths = sorted(
            {
                str(member)
                for candidate in roots
                for member in candidate.get("semantic_path", [])
                if blocker_id in candidate.get("semantic_path", [])
                and str(member) in node_by_id
            }
        )
        external = add_external(
            "pipeline_blocker_release",
            blocker_id,
            "persistent blocker release remains externally unresolved",
        )
        add_wait(
            blocker_id,
            external,
            start_cycle=max(failure_start, int(interval[0])),
            end_cycle=min(endpoint_cycle, int(interval[1])),
            guard_semantic_ids=sorted(
                set(pipeline_ids) | set(related_paths)
            ),
            evidence_refs=[
                blocker_id,
                *[str(item) for item in seed.get("evidence_refs", [])],
            ],
            evidence_strength="exact_rtl_waveform",
            inference_rule=(
                "persistent_pipeline_blocker_waits_for_release.v1"
            ),
        )
        blocker_release_waiters.add(blocker_id)

    for row in sorted(nodes, key=lambda item: item["semantic_id"]):
        row_type = row.get("type")
        if row_type not in {"arbiter_wait", "allocation_wait"} or not _overlaps(
            row, failure_start, endpoint_cycle
        ):
            continue
        waiter_id = str(row["semantic_id"])
        awaited_id = str(row.get("awaited_semantic_id") or "")
        if awaited_id not in node_by_id:
            awaited_id = add_external(
                "arbiter_grant" if row_type == "arbiter_wait" else "free_slot",
                waiter_id,
                "awaited resource is not present in the semantic graph",
            )
        add_wait(
            waiter_id,
            awaited_id,
            start_cycle=max(failure_start, int(row["start_cycle"])),
            end_cycle=min(endpoint_cycle, int(row["end_cycle"])),
            guard_semantic_ids=[
                str(item) for item in row.get("guard_semantic_ids", [])
            ],
            evidence_refs=[
                waiter_id,
                *[str(item) for item in row.get("evidence_refs", [])],
            ],
            evidence_strength=str(
                row.get("evidence_strength", "transition_supported")
            ),
            inference_rule=(
                "arbiter_request_waits_for_grant.v1"
                if row_type == "arbiter_wait"
                else "allocation_waits_for_free_slot.v1"
            ),
        )

    adapter_channel_nodes: Dict[str, str] = {}
    if protocol_adapter is not None:
        if rtl_set_sha256 is None:
            raise WaitForError(
                "rtl_set_sha256 is required with a protocol adapter"
            )
        adapter = validate_protocol_adapter(
            protocol_adapter,
            rtl_set_sha256=rtl_set_sha256,
            known_semantic_ids=node_by_id,
        )
        for channel in (
            adapter["channels"] if protocol_materialization_allowed else []
        ):
            protocol_id = stable_id(
                "vcs_",
                adapter["adapter_id"],
                channel,
                length=24,
            )
            adapter_channel_nodes[str(channel["channel_id"])] = protocol_id
            generated_nodes[protocol_id] = {
                "semantic_id": protocol_id,
                "type": "protocol_transaction",
                "protocol": "tilelink",
                "role": channel["role"],
                "event": channel["event"],
                "handshake_id": channel["handshake_id"],
                "adapter_id": adapter["adapter_id"],
                "evidence_refs": adapter["review"]["evidence_refs"],
                "evidence_strength": "exact_structural",
                "inference_rule": "reviewed_tilelink.channel_label.v1",
            }
            graph_edges.append(
                {
                    "edge_id": stable_id(
                        "vcse_",
                        channel["handshake_id"],
                        protocol_id,
                        "reviewed_protocol_label",
                        length=24,
                    ),
                    "src_semantic_id": channel["handshake_id"],
                    "dst_semantic_id": protocol_id,
                    "relation": "reviewed_protocol_label",
                    "evidence_strength": "exact_structural",
                    "dynamic_score": 1.0,
                }
            )

        def resolve(ref: str) -> str:
            kind, value = ref.split(":", 1)
            return (
                adapter_channel_nodes[value]
                if kind == "channel"
                else value
            )

        for dependency in (
            adapter["dependencies"] if protocol_materialization_allowed else []
        ):
            add_wait(
                resolve(dependency["waiter_ref"]),
                resolve(dependency["awaited_ref"]),
                start_cycle=failure_start,
                end_cycle=endpoint_cycle,
                guard_semantic_ids=[],
                evidence_refs=[
                    adapter["adapter_id"],
                    *dependency["evidence_refs"],
                ],
                evidence_strength="exact_structural",
                inference_rule=dependency["inference_rule"],
            )

    if edge_bound_hit:
        diagnostics.append(
            {
                "code": "waitfor_edge_budget_reached",
                "message": "C5 wait-for graph reached max_waitfor_edges",
                "breaks_complete": True,
            }
        )
    if node_bound_hit:
        diagnostics.append(
            {
                "code": "waitfor_node_budget_reached",
                "message": "C5 wait-for graph reached max_waitfor_nodes",
                "breaks_complete": True,
            }
        )

    wait_edges.sort(key=lambda row: row["edge_id"])
    directed: Dict[str, set[str]] = {}
    for edge in wait_edges:
        directed.setdefault(str(edge["waiter_id"]), set()).add(
            str(edge["awaited_id"])
        )
    strong = _strong_components(wait_nodes, directed)
    cyclic_sets = [
        set(component)
        for component in strong
        if len(component) > 1
        or (
            len(component) == 1
            and component[0] in directed.get(component[0], set())
        )
    ]
    weak = _weak_components(wait_nodes, wait_edges)
    component_rows = []
    scc_rows = []
    for members in weak:
        member_set = set(members)
        component_edges = [
            row
            for row in wait_edges
            if row["waiter_id"] in member_set
            and row["awaited_id"] in member_set
        ]
        display_members = sorted(
            member_set
            | {
                str(guard)
                for row in component_edges
                for guard in row["guard_semantic_ids"]
                if str(guard) in node_by_id
            }
        )

        def semantic_row(member: str) -> Mapping[str, Any]:
            return generated_nodes.get(member) or node_by_id.get(member, {})

        external_dependencies = sorted(
            (
                {
                    "semantic_id": member,
                    "type": "unknown_external_completion",
                    "reason": semantic_row(member).get(
                        "reason", "external completion remains unresolved"
                    ),
                }
                for member in members
                if semantic_row(member).get("external")
            ),
            key=lambda row: row["semantic_id"],
        )
        cyclic = any(component <= member_set for component in cyclic_sets)
        has_starvation_evidence = any(
            node_by_id.get(member, {}).get("type") == "arbiter_wait"
            and node_by_id[member].get("other_progress_observed") is True
            for member in members
        )
        classification = (
            "deadlock_candidate"
            if cyclic
            else (
                "starvation_candidate"
                if has_starvation_evidence
                else "incomplete"
            )
        )
        interval_start = max(
            failure_start,
            max(
                [int(row["start_cycle"]) for row in component_edges]
                or [failure_start]
            ),
        )
        interval_end = min(
            endpoint_cycle,
            min(
                [int(row["end_cycle"]) for row in component_edges]
                or [endpoint_cycle]
            ),
        )
        component_identity = {
            "members": display_members,
            "edge_ids": [row["edge_id"] for row in component_edges],
            "interval": [interval_start, interval_end],
            "classification": classification,
            "external_dependencies": external_dependencies,
        }
        component_id = stable_id(
            "vwc_", component_identity, length=24
        )
        component_rows.append(
            {
                "semantic_id": component_id,
                "type": "waitfor_component",
                "members": display_members,
                "edges": [row["edge_id"] for row in component_edges],
                "interval": {
                    "start_cycle": interval_start,
                    "end_cycle": interval_end,
                },
                "closed": cyclic and not external_dependencies,
                "external_dependencies": external_dependencies,
                "classification": classification,
                "formal_verdict": "not_established",
                "inference_rule": "bounded_failure_window_waitfor.v1",
            }
        )
        for edge in component_edges:
            edge["component_id"] = component_id

        for scc_members in cyclic_sets:
            if not scc_members <= member_set:
                continue
            scc_edges = [
                row
                for row in component_edges
                if row["waiter_id"] in scc_members
                and row["awaited_id"] in scc_members
            ]
            outgoing = [
                row
                for row in wait_edges
                if row["waiter_id"] in scc_members
                and row["awaited_id"] not in scc_members
            ]
            scc_external = sorted(
                {
                    str(row["awaited_id"]) for row in outgoing
                }
                | {
                    member
                    for member in scc_members
                    if semantic_row(member).get("external")
                }
            )
            scc_identity = {
                "members": sorted(scc_members),
                "edge_ids": [row["edge_id"] for row in scc_edges],
                "interval": [interval_start, interval_end],
            }
            scc_rows.append(
                {
                    "semantic_id": stable_id(
                        "vwscc_", scc_identity, length=24
                    ),
                    "type": "waitfor_scc",
                    "members": sorted(scc_members),
                    "edges": [row["edge_id"] for row in scc_edges],
                    "interval": {
                        "start_cycle": interval_start,
                        "end_cycle": interval_end,
                    },
                    "closed": not scc_external,
                    "external_dependencies": scc_external,
                    "classification": "deadlock_candidate",
                    "formal_verdict": "not_established",
                    "component_id": component_id,
                    "inference_rule": "tarjan_failure_window_scc.v1",
                }
            )

    scc_rows.sort(key=lambda row: row["semantic_id"])
    scc_bound_hit = len(scc_rows) > max_scc_candidates
    if scc_bound_hit:
        scc_rows = scc_rows[:max_scc_candidates]
        diagnostics.append(
            {
                "code": "waitfor_scc_budget_reached",
                "message": "C5 wait-for graph reached max_scc_candidates",
                "breaks_complete": True,
            }
        )

    wait_object_nodes = []
    for edge in wait_edges:
        wait_object_nodes.append(
            {
                "semantic_id": stable_id(
                    "vcs_", edge["edge_id"], "resource_wait", length=24
                ),
                "type": "resource_wait",
                "wait_edge_id": edge["edge_id"],
                "component_id": edge["component_id"],
                "waiter_id": edge["waiter_id"],
                "awaited_id": edge["awaited_id"],
                "start_cycle": edge["start_cycle"],
                "end_cycle": edge["end_cycle"],
                "guard_semantic_ids": edge["guard_semantic_ids"],
                "evidence_refs": edge["evidence_refs"],
                "evidence_strength": edge["evidence_strength"],
                "inference_rule": edge["inference_rule"],
            }
        )
        graph_edges.append(
            {
                "edge_id": edge["edge_id"],
                "src_semantic_id": edge["waiter_id"],
                "dst_semantic_id": edge["awaited_id"],
                "waiter_id": edge["waiter_id"],
                "awaited_id": edge["awaited_id"],
                "relation": "waits_for",
                "start_cycle": edge["start_cycle"],
                "end_cycle": edge["end_cycle"],
                "guard_semantic_ids": edge["guard_semantic_ids"],
                "evidence_refs": edge["evidence_refs"],
                "evidence_strength": edge["evidence_strength"],
                "inference_rule": edge["inference_rule"],
                "component_id": edge["component_id"],
                "dynamic_score": 1.0,
            }
        )

    generated_nodes = {
        key: value
        for key, value in generated_nodes.items()
        if value.get("type") != "unknown_external_completion"
        or key in wait_nodes
    }
    generated_nodes.update(
        {str(row["semantic_id"]): row for row in wait_object_nodes}
    )
    generated_nodes.update(
        {str(row["semantic_id"]): row for row in component_rows}
    )
    generated_nodes.update(
        {str(row["semantic_id"]): row for row in scc_rows}
    )
    nodes.extend(generated_nodes.values())
    membership: Dict[str, list[str]] = {}
    classification_by_component = {
        str(row["semantic_id"]): str(row["classification"])
        for row in component_rows
    }
    for row in component_rows:
        component_id = str(row["semantic_id"])
        for member in row["members"]:
            membership.setdefault(str(member), []).append(component_id)
    for root in roots:
        component_ids = sorted(
            membership.get(str(root["semantic_id"]), [])
        )
        root["waitfor_membership"] = bool(component_ids)
        root["waitfor_component_ids"] = component_ids
        root["waitfor_classifications"] = sorted(
            {
                classification_by_component[item]
                for item in component_ids
            }
        )

    work: Dict[str, int | bool] = {
        "waitfor_nodes": len(wait_nodes),
        "waitfor_edges": len(wait_edges),
        "waitfor_components": len(component_rows),
        "scc_candidates": len(scc_rows),
        "protocol_adapter_used": protocol_adapter is not None,
        "waitfor_nodes_reached": node_bound_hit,
        "waitfor_edges_reached": edge_bound_hit,
        "scc_candidates_reached": scc_bound_hit,
    }
    return (
        sorted(nodes, key=lambda row: row["semantic_id"]),
        sorted(graph_edges, key=lambda row: row["edge_id"]),
        roots,
        sorted(
            diagnostics,
            key=lambda row: (row["code"], row.get("message", "")),
        ),
        work,
    )


def get_waitfor_component(
    graph: Mapping[str, Any], component_id: str
) -> Dict[str, Any]:
    """Return one already-materialized component without graph expansion."""
    component = next(
        (
            row
            for row in graph.get("semantic_nodes", [])
            if row.get("semantic_id") == component_id
            and row.get("type") in {"waitfor_component", "waitfor_scc"}
        ),
        None,
    )
    if component is None:
        raise WaitForError("component_id is absent from the semantic graph")
    member_ids = set(component["members"])
    edge_ids = set(component["edges"])
    result = {
        "schema_version": "chisel_waitfor_component_query.v1",
        "graph_id": graph["graph_id"],
        "component": dict(component),
        "members": [
            dict(row)
            for row in graph.get("semantic_nodes", [])
            if row.get("semantic_id") in member_ids
        ],
        "edges": [
            dict(row)
            for row in graph.get("edges", [])
            if row.get("edge_id") in edge_ids
        ],
    }
    result["result_sha256"] = canonical_sha256(result)
    return result
