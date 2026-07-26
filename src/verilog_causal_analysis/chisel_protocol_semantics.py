"""C3 aggregate, ready/valid, and pipeline semantics.

All relationships in this module are instance-local and statement-backed.
Chisel-like names are grouping hints only: handshake truth requires exact
declared members, and pipeline truth additionally requires a sequential RTL
dependency.
"""

from __future__ import annotations

import copy
import re
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

from .cycle_waveform import parse_binary_value
from .identity import stable_id
from .instance_graph import InstanceGraph


_HANDSHAKE_RE = re.compile(r"^(?P<base>.+)_(?P<role>valid|ready)$")
_BITS_RE = re.compile(r"^(?P<base>.+)_bits(?:_(?P<field>.+))?$")
_VEC_RE = re.compile(r"^(?P<base>.+)_(?P<index>\d+)_(?P<field>.+)$")
_STAGE_RE = re.compile(
    r"^(?P<base>.+?)(?:_s|_stage)(?P<index>\d+)(?:_(?P<field>.+))?$"
)
_COMPILER_NET_RE = re.compile(
    r"^(?:_T(?:_\d+)?|_GEN(?:_\d+)?|_.+_T(?:_\d+)?)$"
)
_RANGE_RE = re.compile(r"\s*\[\d+:\d+\]$")


def _clean(signal: str) -> str:
    return _RANGE_RE.sub("", signal)


def _qualify(instance_path: str, local: str) -> str:
    local = _clean(local)
    return local if local.startswith(instance_path + ".") else f"{instance_path}.{local}"


def _object_id(
    prefix: str,
    rtl_set_sha256: str,
    object_type: str,
    instance_path: str,
    members: Iterable[str],
    *extra: Any,
) -> str:
    return stable_id(
        prefix,
        {
            "rtl_set_sha256": rtl_set_sha256,
            "profile_version": "chisel-semantic-profile.v1",
            "object_type": object_type,
            "instance_path": instance_path,
            "members": sorted(set(members)),
            "extra": extra,
        },
        length=24,
    )


def _signal_statements(graph: InstanceGraph, instance_path: str) -> Dict[str, set[str]]:
    instance = next(row for row in graph.instances if row.instance_path == instance_path)
    module = graph._parser.modules[instance.module_name]
    result: Dict[str, set[str]] = {name: set() for name in module.signals}
    for dep in module.dependencies:
        result.setdefault(dep.source, set()).add(dep.statement_id)
        result.setdefault(dep.target, set()).add(dep.statement_id)
    return result


def build_c3_semantics(
    graph: InstanceGraph,
    *,
    rtl_set_sha256: str,
    register_transitions: Sequence[Mapping[str, Any]],
    features: Sequence[str],
) -> Dict[str, list[Dict[str, Any]]]:
    """Recover deterministic, instance-local C3 semantic objects."""
    enabled = set(features)
    if not (enabled & {"aggregate", "handshake", "pipeline"}):
        return {
            "aggregates": [],
            "handshakes": [],
            "pipelines": [],
            "blocking_relations": [],
        }

    aggregates: list[Dict[str, Any]] = []
    handshakes: list[Dict[str, Any]] = []
    pipelines: list[Dict[str, Any]] = []
    blocking_relations: list[Dict[str, Any]] = []
    aggregate_by_key: Dict[Tuple[str, str], Dict[str, Any]] = {}

    for instance in graph.instances:
        module = graph._parser.modules[instance.module_name]
        locals_ = sorted(module.signals)
        statements = _signal_statements(graph, instance.instance_path)
        shaped: Dict[str, Dict[str, str]] = {}
        vecs: Dict[str, list[Tuple[int, str, str]]] = {}
        records: Dict[str, list[Tuple[str, str]]] = {}
        for local in locals_:
            match = _HANDSHAKE_RE.fullmatch(local)
            if match:
                shaped.setdefault(match["base"], {})[match["role"]] = local
                continue
            match = _BITS_RE.fullmatch(local)
            if match:
                field = match["field"] or "bits"
                shaped.setdefault(match["base"], {})[f"bits.{field}"] = local
                continue
            match = _VEC_RE.fullmatch(local)
            if match:
                vecs.setdefault(match["base"], []).append(
                    (int(match["index"]), match["field"], local)
                )
                continue
            if "_" in local:
                base, field = local.rsplit("_", 1)
                records.setdefault(base, []).append((field, local))

        # A valid member is sufficient for a pipeline payload aggregate; both
        # valid and ready are required before it can become Decoupled.
        for base, roles in sorted(shaped.items()):
            if "valid" not in roles and len(roles) < 2:
                continue
            members = [_qualify(instance.instance_path, item) for item in roles.values()]
            kind = (
                "decoupled"
                if {"valid", "ready"} <= set(roles)
                else "pipeline_payload"
            )
            identity_strength = (
                "exact_rtl_members"
                if {"valid", "ready"} <= set(roles)
                else "inferred_shape"
            )
            member_rows = [
                {
                    "signal": _qualify(instance.instance_path, local),
                    "role": role,
                    "width": module.signals[local].width,
                }
                for role, local in sorted(roles.items())
            ]
            statement_ids = sorted(
                {
                    statement
                    for local in roles.values()
                    for statement in statements.get(local, ())
                }
            )
            aggregate_id = _object_id(
                "vcs_",
                rtl_set_sha256,
                "aggregate",
                instance.instance_path,
                members,
                base,
                kind,
            )
            aggregate = {
                "aggregate_id": aggregate_id,
                "instance_path": instance.instance_path,
                "base_name": base,
                "kind": kind,
                "members": member_rows,
                "member_signals": sorted(members),
                "identity_strength": identity_strength,
                "inference_rule": "exact_instance_flattened_fields.v1",
                "statement_ids": statement_ids,
            }
            aggregates.append(aggregate)
            aggregate_by_key[(instance.instance_path, base)] = aggregate
            if (
                "handshake" in enabled
                and {"valid", "ready"} <= set(roles)
            ):
                payload = sorted(
                    _qualify(instance.instance_path, local)
                    for role, local in roles.items()
                    if role.startswith("bits.")
                )
                handshake_id = _object_id(
                    "vcs_",
                    rtl_set_sha256,
                    "handshake",
                    instance.instance_path,
                    members,
                    aggregate_id,
                )
                handshakes.append(
                    {
                        "handshake_id": handshake_id,
                        "aggregate_id": aggregate_id,
                        "instance_path": instance.instance_path,
                        "valid_signal": _qualify(
                            instance.instance_path, roles["valid"]
                        ),
                        "ready_signal": _qualify(
                            instance.instance_path, roles["ready"]
                        ),
                        "payload_members": payload,
                        "member_signals": sorted(members),
                        "identity_strength": "exact_rtl_members",
                        "statement_ids": statement_ids,
                        "inference_rule": "instance_local_ready_valid.v1",
                    }
                )

        for base, rows in sorted(vecs.items()):
            if len({index for index, _field, _local in rows}) < 2:
                continue
            members = [_qualify(instance.instance_path, local) for _, _, local in rows]
            aggregate_id = _object_id(
                "vcs_", rtl_set_sha256, "aggregate", instance.instance_path, members, base, "vec"
            )
            aggregates.append(
                {
                    "aggregate_id": aggregate_id,
                    "instance_path": instance.instance_path,
                    "base_name": base,
                    "kind": "vec",
                    "members": [
                        {
                            "signal": _qualify(instance.instance_path, local),
                            "role": f"[{index}].{field}",
                            "width": module.signals[local].width,
                        }
                        for index, field, local in sorted(rows)
                    ],
                    "member_signals": sorted(members),
                    "identity_strength": "inferred_shape",
                    "inference_rule": "flattened_vec_shape.v1",
                    "statement_ids": sorted(
                        {
                            statement
                            for _index, _field, local in rows
                            for statement in statements.get(local, ())
                        }
                    ),
                }
            )

        # Record grouping is intentionally limited to declared port fields and
        # is weaker than the exact ready/valid grouping above.
        for base, rows in sorted(records.items()):
            port_rows = [
                (field, local)
                for field, local in rows
                if module.signals[local].signal_type in {"input", "output", "inout"}
            ]
            if len(port_rows) < 2 or (instance.instance_path, base) in aggregate_by_key:
                continue
            members = [_qualify(instance.instance_path, local) for _, local in port_rows]
            aggregates.append(
                {
                    "aggregate_id": _object_id(
                        "vcs_", rtl_set_sha256, "aggregate", instance.instance_path, members, base, "record"
                    ),
                    "instance_path": instance.instance_path,
                    "base_name": base,
                    "kind": "record",
                    "members": [
                        {
                            "signal": _qualify(instance.instance_path, local),
                            "role": field,
                            "width": module.signals[local].width,
                        }
                        for field, local in sorted(port_rows)
                    ],
                    "member_signals": sorted(members),
                    "identity_strength": "inferred_shape",
                    "inference_rule": "flattened_port_record_shape.v1",
                    "statement_ids": sorted(
                        {
                            statement
                            for _field, local in port_rows
                            for statement in statements.get(local, ())
                        }
                    ),
                }
            )

    if "pipeline" in enabled:
        pipeline_groups: Dict[Tuple[str, str], Dict[int, Dict[str, Any]]] = {}
        for aggregate in aggregates:
            match = _STAGE_RE.fullmatch(aggregate["base_name"])
            if not match:
                continue
            pipeline_groups.setdefault(
                (aggregate["instance_path"], match["base"]), {}
            )[int(match["index"])] = aggregate

        transition_by_signal = {
            _clean(str(row["signal"])): row for row in register_transitions
        }
        for (instance_path, base), stages in sorted(pipeline_groups.items()):
            transfers = []
            ordered_indexes = sorted(stages)
            for left, right in zip(ordered_indexes, ordered_indexes[1:]):
                if right != left + 1:
                    continue
                left_members = set(stages[left]["member_signals"])
                right_members = set(stages[right]["member_signals"])
                statement_ids: set[str] = set()
                member_pairs = []
                for target in sorted(right_members):
                    transition = transition_by_signal.get(_clean(target))
                    if transition is None:
                        continue
                    for rule in list(transition["reset_rules"]) + list(
                        transition["update_rules"]
                    ):
                        sources = left_members & {
                            _clean(str(item)) for item in rule["value_members"]
                        }
                        for source in sorted(sources):
                            member_pairs.append(
                                {"source": source, "target": target}
                            )
                            statement_ids.update(rule["statement_ids"])
                if member_pairs:
                    transfers.append(
                        {
                            "from_stage": left,
                            "to_stage": right,
                            "member_pairs": member_pairs,
                            "statement_ids": sorted(statement_ids),
                        }
                    )
            if not transfers:
                continue
            members = sorted(
                {
                    member
                    for aggregate in stages.values()
                    for member in aggregate["member_signals"]
                }
            )
            pipelines.append(
                {
                    "pipeline_id": _object_id(
                        "vcs_", rtl_set_sha256, "pipeline", instance_path, members, base
                    ),
                    "instance_path": instance_path,
                    "base_name": base,
                    "stages": [
                        {
                            "index": index,
                            "aggregate_id": stages[index]["aggregate_id"],
                            "valid_signal": next(
                                (
                                    row["signal"]
                                    for row in stages[index]["members"]
                                    if row["role"] == "valid"
                                ),
                                None,
                            ),
                            "payload_members": sorted(
                                row["signal"]
                                for row in stages[index]["members"]
                                if row["role"].startswith("bits.")
                            ),
                        }
                        for index in ordered_indexes
                    ],
                    "transfers": transfers,
                    "member_signals": members,
                    "identity_strength": "exact_sequential_dependency",
                    "statement_ids": sorted(
                        {
                            statement
                            for transfer in transfers
                            for statement in transfer["statement_ids"]
                        }
                    ),
                    "inference_rule": "stage_shape_sequential_transfer.v1",
                }
            )

        pipeline_stage_members: Dict[str, Tuple[Dict[str, Any], Dict[str, Any]]] = {}
        for pipeline in pipelines:
            for stage in pipeline["stages"]:
                aggregate = next(
                    row
                    for row in aggregates
                    if row["aggregate_id"] == stage["aggregate_id"]
                )
                for signal in aggregate["member_signals"]:
                    pipeline_stage_members[signal] = (pipeline, stage)
        for instance in graph.instances:
            module = graph._parser.modules[instance.module_name]
            grouped: Dict[Tuple[str, str, int, int], list[Any]] = {}
            combinational_by_target: Dict[str, list[Any]] = {}
            for dep in module.dependencies:
                if dep.dep_type.value != "combinational":
                    continue
                key = (dep.target, dep.expression, dep.line_start, dep.line_end)
                grouped.setdefault(key, []).append(dep)
                combinational_by_target.setdefault(dep.target, []).append(dep)
            for (target, expression, _start, _end), deps in sorted(grouped.items()):
                source_locals = {dep.source for dep in deps}
                statement_ids = {dep.statement_id for dep in deps}
                pending = sorted(source_locals)
                visited: set[str] = set()
                while pending:
                    source = pending.pop()
                    if source in visited:
                        continue
                    visited.add(source)
                    nested = combinational_by_target.get(source, [])
                    if not nested or not _COMPILER_NET_RE.fullmatch(source):
                        continue
                    source_locals.discard(source)
                    source_locals.update(dep.source for dep in nested)
                    statement_ids.update(dep.statement_id for dep in nested)
                    pending.extend(
                        sorted(
                            dep.source
                            for dep in nested
                            if dep.source not in visited
                        )
                    )
                sources = {
                    _qualify(instance.instance_path, source)
                    for source in source_locals
                }
                stage_hits: Dict[Tuple[str, int], set[str]] = {}
                pipeline_by_id: Dict[str, Dict[str, Any]] = {}
                for source in sources:
                    hit = pipeline_stage_members.get(source)
                    if hit is None:
                        continue
                    pipeline, stage = hit
                    if pipeline["instance_path"] != instance.instance_path:
                        continue
                    pipeline_by_id[pipeline["pipeline_id"]] = pipeline
                    stage_hits.setdefault(
                        (pipeline["pipeline_id"], int(stage["index"])), set()
                    ).add(source)
                if not stage_hits:
                    continue
                qualified_target = _qualify(instance.instance_path, target)
                member_signals = sorted(sources | {qualified_target})
                blockers = [
                    {
                        "pipeline_id": pipeline_id,
                        "pipeline_base": pipeline_by_id[pipeline_id]["base_name"],
                        "stage_index": stage_index,
                        "member_signals": sorted(members),
                    }
                    for (pipeline_id, stage_index), members in sorted(stage_hits.items())
                ]
                blocking_relations.append(
                    {
                        "blocking_id": _object_id(
                            "vcs_",
                            rtl_set_sha256,
                            "blocking_relation",
                            instance.instance_path,
                            member_signals,
                            qualified_target,
                            expression,
                        ),
                        "instance_path": instance.instance_path,
                        "blocked_resource": f"rtl_signal:{qualified_target}",
                        "target_signal": qualified_target,
                        "blockers": blockers,
                        "member_signals": member_signals,
                        "expression": expression,
                        "statement_ids": sorted(statement_ids),
                        "identity_strength": "exact_rtl_expression",
                        "inference_rule": "pipeline_member_combinational_blocker.v1",
                    }
                )

    return {
        "aggregates": sorted(aggregates, key=lambda row: row["aggregate_id"]),
        "handshakes": sorted(handshakes, key=lambda row: row["handshake_id"]),
        "pipelines": sorted(pipelines, key=lambda row: row["pipeline_id"]),
        "blocking_relations": sorted(
            blocking_relations, key=lambda row: row["blocking_id"]
        ),
    }


def project_c3_waveform_scope(
    normalized_design: Dict[str, Any],
    graph: InstanceGraph,
    endpoint_signal: str,
    *,
    rtl_set_sha256: str,
) -> Dict[str, Any]:
    """Project C3 objects onto one exact waveform-only instance scope.

    Jasper may expose elaboration aliases absent from the emitted top-level
    instance declarations. InstanceGraph already joins such a scope to a
    module definition only when the immediate waveform signature has a unique
    best match. This function reuses that exact join and never selects a
    module or source instance from a basename.
    """
    resolution = graph.resolve_signal(endpoint_signal)
    if (
        not resolution.exact
        or resolution.instance_path is None
        or resolution.module_name is None
        or any(
            row.instance_path == resolution.instance_path
            for row in graph.instances
        )
    ):
        return normalized_design
    source_paths = sorted(
        row.instance_path
        for row in graph.instances
        if row.module_name == resolution.module_name
    )
    if not source_paths:
        return normalized_design
    source_path = source_paths[0]
    target_path = resolution.instance_path

    def replace(value: Any) -> Any:
        if isinstance(value, str):
            if value == source_path:
                return target_path
            if value.startswith(source_path + "."):
                return target_path + value[len(source_path) :]
            if value.startswith("rtl_signal:" + source_path + "."):
                return (
                    "rtl_signal:"
                    + target_path
                    + value[len("rtl_signal:" + source_path) :]
                )
            return value
        if isinstance(value, list):
            return [replace(item) for item in value]
        if isinstance(value, dict):
            return {key: replace(item) for key, item in value.items()}
        return value

    aggregates = [
        replace(copy.deepcopy(row))
        for row in normalized_design["aggregates"]
        if row["instance_path"] == source_path
    ]
    aggregate_ids: Dict[str, str] = {}
    source_aggregates = [
        row
        for row in normalized_design["aggregates"]
        if row["instance_path"] == source_path
    ]
    for source, projected in zip(source_aggregates, aggregates):
        projected_id = _object_id(
            "vcs_",
            rtl_set_sha256,
            "aggregate",
            target_path,
            projected["member_signals"],
            projected["base_name"],
            projected["kind"],
        )
        aggregate_ids[source["aggregate_id"]] = projected_id
        projected["aggregate_id"] = projected_id
        projected["inference_rule"] = (
            "exact_waveform_scope_module_signature.v1+"
            + projected["inference_rule"]
        )

    handshakes = [
        replace(copy.deepcopy(row))
        for row in normalized_design["handshakes"]
        if row["instance_path"] == source_path
    ]
    handshake_ids: Dict[str, str] = {}
    source_handshakes = [
        row
        for row in normalized_design["handshakes"]
        if row["instance_path"] == source_path
    ]
    for source, projected in zip(source_handshakes, handshakes):
        projected["aggregate_id"] = aggregate_ids[
            source["aggregate_id"]
        ]
        projected_id = _object_id(
            "vcs_",
            rtl_set_sha256,
            "handshake",
            target_path,
            projected["member_signals"],
            projected["aggregate_id"],
        )
        handshake_ids[source["handshake_id"]] = projected_id
        projected["handshake_id"] = projected_id
        projected["inference_rule"] = (
            "exact_waveform_scope_module_signature.v1+"
            + projected["inference_rule"]
        )

    pipelines = [
        replace(copy.deepcopy(row))
        for row in normalized_design["pipelines"]
        if row["instance_path"] == source_path
    ]
    pipeline_ids: Dict[str, str] = {}
    source_pipelines = [
        row
        for row in normalized_design["pipelines"]
        if row["instance_path"] == source_path
    ]
    for source, projected in zip(source_pipelines, pipelines):
        for source_stage, projected_stage in zip(
            source["stages"], projected["stages"]
        ):
            projected_stage["aggregate_id"] = aggregate_ids[
                source_stage["aggregate_id"]
            ]
        projected_id = _object_id(
            "vcs_",
            rtl_set_sha256,
            "pipeline",
            target_path,
            projected["member_signals"],
            projected["base_name"],
        )
        pipeline_ids[source["pipeline_id"]] = projected_id
        projected["pipeline_id"] = projected_id
        projected["inference_rule"] = (
            "exact_waveform_scope_module_signature.v1+"
            + projected["inference_rule"]
        )

    relations = [
        replace(copy.deepcopy(row))
        for row in normalized_design["blocking_relations"]
        if row["instance_path"] == source_path
    ]
    source_relations = [
        row
        for row in normalized_design["blocking_relations"]
        if row["instance_path"] == source_path
    ]
    for source, projected in zip(source_relations, relations):
        for source_blocker, projected_blocker in zip(
            source["blockers"], projected["blockers"]
        ):
            projected_blocker["pipeline_id"] = pipeline_ids[
                source_blocker["pipeline_id"]
            ]
        projected["blocking_id"] = _object_id(
            "vcs_",
            rtl_set_sha256,
            "blocking_relation",
            target_path,
            projected["member_signals"],
            projected["target_signal"],
            projected["expression"],
        )
        projected["inference_rule"] = (
            "exact_waveform_scope_module_signature.v1+"
            + projected["inference_rule"]
        )

    normalized_design["aggregates"].extend(aggregates)
    normalized_design["handshakes"].extend(handshakes)
    normalized_design["pipelines"].extend(pipelines)
    normalized_design["blocking_relations"].extend(relations)
    for key, id_key in (
        ("aggregates", "aggregate_id"),
        ("handshakes", "handshake_id"),
        ("pipelines", "pipeline_id"),
        ("blocking_relations", "blocking_id"),
    ):
        normalized_design[key].sort(key=lambda row: row[id_key])
    normalized_design["diagnostics"].append(
        {
            "code": "waveform_scope_semantics_projected",
            "instance_path": target_path,
            "module_name": resolution.module_name,
            "inference_rule": "exact_waveform_scope_module_signature.v1",
        }
    )
    identity_payload = {
        key: value
        for key, value in normalized_design.items()
        if key != "normalized_design_id"
    }
    normalized_design["normalized_design_id"] = stable_id(
        "vcnd_", identity_payload
    )
    return normalized_design


def _waveform_value(waveform: Any, signal: str, cycle: int) -> Optional[str]:
    value = waveform.get_signal_value(signal, cycle)
    if value is not None:
        return value
    if not hasattr(waveform, "signals"):
        return None
    clean = _clean(signal)
    matches = [
        str(item)
        for item in waveform.signals.by_name
        if _clean(str(item)) == clean
    ]
    return (
        waveform.get_signal_value(matches[0], cycle)
        if len(matches) == 1
        else None
    )


def _bit(value: Optional[str]) -> Optional[int]:
    parsed = parse_binary_value(value) if value is not None else None
    return parsed if parsed in (0, 1) else None


def stall_intervals(
    handshakes: Sequence[Mapping[str, Any]],
    waveform: Any,
    *,
    end_cycle: int,
    max_intervals: int,
    max_temporal_samples: int,
) -> Tuple[list[Dict[str, Any]], list[Dict[str, Any]]]:
    """Build bounded exact ready/valid stall intervals.

    X/Z in control ends an observed interval. X/Z or changing payload fields
    degrade payload identity and are reported explicitly.
    """
    result: list[Dict[str, Any]] = []
    diagnostics: list[Dict[str, Any]] = []
    samples = 0
    for handshake in sorted(handshakes, key=lambda row: row["handshake_id"]):
        start: Optional[int] = None
        last_accept: Optional[int] = None
        last_accept_before_stall: Optional[int] = None
        payload_values: Dict[str, list[Optional[str]]] = {
            signal: [] for signal in handshake["payload_members"]
        }
        for cycle in range(0, end_cycle + 1):
            if samples + 2 > max_temporal_samples:
                diagnostics.append(
                    {
                        "code": "temporal_sample_budget_reached",
                        "message": "C3 stall interval sampling reached its bound",
                        "breaks_complete": True,
                    }
                )
                return result, diagnostics
            valid = _bit(_waveform_value(waveform, handshake["valid_signal"], cycle))
            ready = _bit(_waveform_value(waveform, handshake["ready_signal"], cycle))
            samples += 2
            if valid == 1 and ready == 1:
                last_accept = cycle
            stalled = valid == 1 and ready == 0
            if stalled:
                if start is None:
                    start = cycle
                    last_accept_before_stall = last_accept
                    payload_values = {
                        signal: [] for signal in handshake["payload_members"]
                    }
                for signal in payload_values:
                    if samples >= max_temporal_samples:
                        diagnostics.append(
                            {
                                "code": "temporal_sample_budget_reached",
                                "message": "C3 payload sampling reached its bound",
                                "breaks_complete": True,
                            }
                        )
                        return result, diagnostics
                    payload_values[signal].append(
                        _waveform_value(waveform, signal, cycle)
                    )
                    samples += 1
                continue
            if start is not None:
                _append_stall(
                    result,
                    handshake,
                    start,
                    cycle - 1,
                    last_accept_before_stall,
                    payload_values,
                )
                if len(result) >= max_intervals:
                    return result, diagnostics
                start = None
        if start is not None:
            _append_stall(
                result,
                handshake,
                start,
                end_cycle,
                last_accept_before_stall,
                payload_values,
            )
            if len(result) >= max_intervals:
                return result, diagnostics
    return result, diagnostics


def _append_stall(
    result: list[Dict[str, Any]],
    handshake: Mapping[str, Any],
    start_cycle: int,
    end_cycle: int,
    last_accept_cycle: Optional[int],
    payload_values: Mapping[str, Sequence[Optional[str]]],
) -> None:
    stable: Dict[str, str] = {}
    unknown: list[str] = []
    unstable: list[str] = []
    for signal, values in sorted(payload_values.items()):
        if not values or any(parse_binary_value(value) is None for value in values):
            unknown.append(signal)
        elif len(set(values)) == 1:
            stable[signal] = str(values[0])
        else:
            unstable.append(signal)
    if not payload_values:
        strength = "none"
    elif len(stable) == len(payload_values):
        strength = "exact"
    elif stable:
        strength = "partial"
    else:
        strength = "unknown"
    identity = {
        "type": "stall_interval",
        "handshake_id": handshake["handshake_id"],
        "start_cycle": start_cycle,
        "end_cycle": end_cycle,
        "payload_identity": {
            "strength": strength,
            "stable_members": stable,
            "unknown_members": unknown,
            "unstable_members": unstable,
        },
    }
    result.append(
        {
            "semantic_id": stable_id("vct_", identity, length=24),
            "type": "stall_interval",
            "handshake_id": handshake["handshake_id"],
            "start_cycle": start_cycle,
            "end_cycle": end_cycle,
            "sample_count": end_cycle - start_cycle + 1,
            "valid": "1",
            "ready": "0",
            "payload_identity": identity["payload_identity"],
            "last_accept_cycle": last_accept_cycle,
            "evidence_strength": "fully_observed",
            "inference_rule": "exact_ready_valid_stall_interval.v1",
        }
    )
