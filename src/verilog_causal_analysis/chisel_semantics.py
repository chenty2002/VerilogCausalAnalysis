"""Deterministic C2/C3 Chisel lowering semantics.

This module deliberately consumes parser and instance identities rather than
temporary-name heuristics alone.  The resulting objects are reusable for every
endpoint in one prepared semantic session.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

from .causal_slicer import ExpressionEvaluator
from .cycle_waveform import parse_binary_value
from .identity import canonical_sha256, stable_id
from .instance_graph import InstanceGraph
from .verilog_parser import Dependency, DependencyType


NORMALIZED_DESIGN_SCHEMA = "chisel_normalized_design"
_C2_FEATURES = frozenset(
    {"compiler_net_normalization", "register_transition"}
)
_C3_FEATURES = frozenset({"aggregate", "handshake", "pipeline"})
_TEMPORARY_RE = re.compile(
    r"^(?:_T(?:_\d+)?|_GEN(?:_\d+)?|.+_(?:T|GEN)(?:_\d+)?)$"
)
_IDENTIFIER_RE = re.compile(r"^[A-Za-z_$][A-Za-z0-9_.$]*(?:\s*\[\d+:\d+\])?$")
_NUMBER_RE = re.compile(r"^\s*(?:\d+|\d+'[bhdBHD][0-9a-fA-F_xXzZ]+)\s*$")


class SemanticQueryError(ValueError):
    pass


def c2_enabled(features: Iterable[str]) -> bool:
    return bool(_C2_FEATURES & set(features))


def c3_enabled(features: Iterable[str]) -> bool:
    return bool(_C3_FEATURES & set(features))


def _local_members(parser: Any, text: str) -> Tuple[str, ...]:
    return tuple(sorted(parser._extract_signals_from_text(text)))


def _qualify(instance_path: str, signal: str) -> str:
    clean = re.sub(r"\s*\[\d+:\d+\]$", "", signal)
    if clean.startswith(instance_path + "."):
        return clean
    return f"{instance_path}.{clean}"


def _statement_groups(
    graph: InstanceGraph,
    *,
    dependency_type: Optional[DependencyType] = None,
) -> Dict[Tuple[str, str, str, str, int, int], Tuple[Dependency, ...]]:
    groups: Dict[
        Tuple[str, str, str, str, int, int], list[Dependency]
    ] = {}
    for instance in graph.instances:
        module = graph._parser.modules[instance.module_name]
        for dep in module.dependencies:
            if dependency_type is not None and dep.dep_type != dependency_type:
                continue
            key = (
                instance.instance_path,
                dep.target,
                dep.expression,
                dep.condition,
                dep.line_start,
                dep.line_end,
            )
            groups.setdefault(key, []).append(dep)
    return {
        key: tuple(
            sorted(rows, key=lambda row: (row.source, row.statement_id))
        )
        for key, rows in sorted(groups.items())
    }


class _UnionFind:
    def __init__(self) -> None:
        self.parent: Dict[str, str] = {}

    def find(self, item: str) -> str:
        self.parent.setdefault(item, item)
        if self.parent[item] != item:
            self.parent[item] = self.find(self.parent[item])
        return self.parent[item]

    def union(self, left: str, right: str) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root != right_root:
            self.parent[max(left_root, right_root)] = min(left_root, right_root)


def _alias_classes(
    graph: InstanceGraph, rtl_set_sha256: str
) -> list[Dict[str, Any]]:
    union = _UnionFind()
    evidence: Dict[frozenset[str], set[str]] = {}
    widths: Dict[str, Optional[int]] = {}
    for key, deps in _statement_groups(
        graph, dependency_type=DependencyType.COMBINATIONAL
    ).items():
        instance_path, target, expression, condition, _line_start, _line_end = key
        if condition or len(deps) != 1 or not _IDENTIFIER_RE.fullmatch(expression):
            continue
        dep = deps[0]
        expression_clean = re.sub(r"\s*\[\d+:\d+\]$", "", expression)
        if expression_clean != dep.source:
            continue
        left = _qualify(instance_path, target)
        right = _qualify(instance_path, dep.source)
        module = graph._parser.modules[
            graph.resolve_signal(left).module_name
        ]
        left_info = module.signals.get(target)
        right_info = module.signals.get(dep.source)
        if (
            left_info is not None
            and right_info is not None
            and left_info.width != right_info.width
        ):
            continue
        widths[left] = left_info.width if left_info is not None else None
        widths[right] = right_info.width if right_info is not None else None
        union.union(left, right)
        evidence.setdefault(frozenset((left, right)), set()).add(
            dep.statement_id
        )
    for binding in graph.port_bindings:
        if (
            len(binding.actual_member_signals) != 1
            or not _IDENTIFIER_RE.fullmatch(binding.actual_expression)
        ):
            continue
        left = binding.source_signal
        right = binding.target_signal
        union.union(left, right)
        evidence.setdefault(frozenset((left, right)), set()).add(
            binding.statement_id
        )

    members_by_root: Dict[str, list[str]] = {}
    for member in sorted(union.parent):
        members_by_root.setdefault(union.find(member), []).append(member)
    result = []
    for members in sorted(members_by_root.values()):
        if len(members) < 2:
            continue
        statement_ids = sorted(
            {
                statement_id
                for pair, rows in evidence.items()
                if pair <= set(members)
                for statement_id in rows
            }
        )
        canonical = min(
            members,
            key=lambda item: (
                bool(_TEMPORARY_RE.fullmatch(item.rsplit(".", 1)[-1])),
                item,
            ),
        )
        known_widths = {widths[item] for item in members if widths.get(item)}
        identity = {
            "rtl_set_sha256": rtl_set_sha256,
            "profile_version": "chisel-semantic-profile",
            "object_type": "alias_class",
            "members": members,
            "statement_ids": statement_ids,
        }
        result.append(
            {
                "alias_id": stable_id("vca_", identity, length=24),
                "canonical_signal": canonical,
                "members": members,
                "exact_equivalence": True,
                "evidence_statements": statement_ids,
                "width": next(iter(known_widths)) if len(known_widths) == 1 else None,
                "signed": None,
                "inference_rule": "single_driver_pure_wire_alias",
            }
        )
    return sorted(result, key=lambda row: row["alias_id"])


def _expression_groups(
    graph: InstanceGraph, rtl_set_sha256: str
) -> list[Dict[str, Any]]:
    combinational = _statement_groups(
        graph, dependency_type=DependencyType.COMBINATIONAL
    )
    by_target: Dict[Tuple[str, str], Tuple[Dependency, ...]] = {}
    for key, deps in combinational.items():
        instance_path, target, _expression, _condition, _start, _end = key
        by_target.setdefault((instance_path, target), deps)

    groups = []
    for (instance_path, target), direct in sorted(by_target.items()):
        if not _TEMPORARY_RE.fullmatch(target):
            continue
        pending = [target]
        member_locals: set[str] = set()
        leaf_locals: set[str] = set()
        guards: set[str] = set()
        statements: set[str] = set()
        expressions: list[str] = []
        while pending:
            current = pending.pop()
            if current in member_locals:
                continue
            rows = by_target.get((instance_path, current))
            if rows is None:
                leaf_locals.add(current)
                continue
            member_locals.add(current)
            expressions.extend(sorted({row.expression for row in rows if row.expression}))
            statements.update(row.statement_id for row in rows)
            for row in rows:
                guards.update(_local_members(graph._parser, row.condition))
                if "?" in row.expression:
                    guards.update(
                        _local_members(
                            graph._parser, row.expression.split("?", 1)[0]
                        )
                    )
                if _TEMPORARY_RE.fullmatch(row.source) and (
                    instance_path, row.source
                ) in by_target:
                    pending.append(row.source)
                else:
                    leaf_locals.add(row.source)
        if not member_locals:
            continue
        members = sorted(_qualify(instance_path, item) for item in member_locals)
        leaves = sorted(_qualify(instance_path, item) for item in leaf_locals)
        qualified_guards = sorted(
            _qualify(instance_path, item) for item in guards
        )
        identity = {
            "rtl_set_sha256": rtl_set_sha256,
            "profile_version": "chisel-semantic-profile",
            "object_type": "expression_group",
            "instance_path": instance_path,
            "target_signal": _qualify(instance_path, target),
            "members": members,
            "leaf_inputs": leaves,
            "statement_ids": sorted(statements),
        }
        groups.append(
            {
                "expression_id": stable_id("vcx_", identity, length=24),
                "target_signal": _qualify(instance_path, target),
                "member_signals": members,
                "leaf_inputs": leaves,
                "expression": " ; ".join(dict.fromkeys(expressions)),
                "guards": qualified_guards,
                "leaf_roles": [
                    {
                        "signal": leaf,
                        "role": "guard" if leaf in qualified_guards else "data",
                    }
                    for leaf in leaves
                ],
                "statement_ids": sorted(statements),
                "provenance_hint_ids": [],
                "inference_rule": "pure_combinational_temporary_dag",
            }
        )
    return sorted(groups, key=lambda row: row["expression_id"])


def _update_kind(target: str, expression: str, condition: str) -> str:
    compact = re.sub(r"\s+", "", expression)
    if "reset" in condition.lower():
        return "reset"
    if re.fullmatch(rf"\(?{re.escape(target)}\+[^)]+\)?", compact):
        return "increment"
    if re.fullmatch(rf"\(?{re.escape(target)}-[^)]+\)?", compact):
        return "decrement"
    if _NUMBER_RE.fullmatch(expression):
        value = ExpressionEvaluator({}).evaluate(expression)
        integer = parse_binary_value(value) if value is not None else None
        if integer == 0:
            return "clear"
        return "load"
    if expression == target:
        return "hold"
    if "?" in expression:
        return "muxed_update"
    return "load" if expression else "unknown"


def _register_transitions(
    graph: InstanceGraph, rtl_set_sha256: str, clock_signal: str
) -> list[Dict[str, Any]]:
    by_register: Dict[Tuple[str, str], list[Any]] = {}
    combinational_by_target: Dict[Tuple[str, str], list[Any]] = {}
    for instance in graph.instances:
        module = graph._parser.modules[instance.module_name]
        for assignment in module.assignment_records:
            if assignment.is_sequential:
                by_register.setdefault(
                    (instance.instance_path, assignment.target), []
                ).append(assignment)
            else:
                combinational_by_target.setdefault(
                    (instance.instance_path, assignment.target), []
                ).append(assignment)
    for key, deps in _statement_groups(
        graph, dependency_type=DependencyType.COMBINATIONAL
    ).items():
        instance_path, target, expression, condition, start, end = key
        bucket = combinational_by_target.setdefault(
            (instance_path, target), []
        )
        if any(
            row.expression == expression and row.condition == condition
            for row in bucket
        ):
            continue
        bucket.append(
            SimpleNamespace(
                target=target,
                expression=expression,
                condition=condition,
                is_sequential=False,
                statement_id=deps[0].statement_id,
                line_start=start,
                line_end=end,
            )
        )
    result = []
    for (instance_path, target), rows in sorted(by_register.items()):
        rules = []
        statement_ids: set[str] = set()
        for priority, assignment in enumerate(
            sorted(
                rows,
                key=lambda item: (
                    item.line_start,
                    item.line_end,
                    item.condition,
                    item.expression,
                ),
            )
        ):
            expression = assignment.expression
            condition = assignment.condition
            statement_ids.add(assignment.statement_id)
            expanded_statement_ids = [assignment.statement_id]
            expression_identifier = re.sub(
                r"\s*\[\d+:\d+\]$", "", expression.strip()
            )
            inline_rows = combinational_by_target.get(
                (instance_path, expression_identifier), []
            )
            if (
                _IDENTIFIER_RE.fullmatch(expression)
                and len(inline_rows) == 1
                and not inline_rows[0].condition
            ):
                expression = inline_rows[0].expression
                expanded_statement_ids.append(
                    inline_rows[0].statement_id
                )
                statement_ids.add(inline_rows[0].statement_id)
            expression_members = set(
                _local_members(graph._parser, expression)
            )
            guard_members = set(_local_members(graph._parser, condition))
            value_members = expression_members - guard_members
            kind = _update_kind(target, expression, condition)
            rules.append(
                {
                    "rule_id": stable_id(
                        "vcr_",
                        rtl_set_sha256,
                        instance_path,
                        target,
                        priority,
                        condition,
                        expression,
                        length=24,
                    ),
                    "priority": priority,
                    "guard_expression": condition or "1'b1",
                    "guard_members": sorted(
                        _qualify(instance_path, item) for item in guard_members
                    ),
                    "value_expression": expression,
                    "source_expression": assignment.expression,
                    "value_members": sorted(
                        _qualify(instance_path, item) for item in value_members
                    ),
                    "update_kind": kind,
                    "statement_ids": sorted(expanded_statement_ids),
                }
            )
        reset_rules = [
            row for row in rules
            if row["update_kind"] == "reset"
            or "reset" in row["guard_expression"].lower()
        ]
        update_rules = [row for row in rules if row not in reset_rules]
        hold_rule = next(
            (row for row in update_rules if row["update_kind"] == "hold"),
            None,
        )
        if hold_rule is None:
            guard_members = sorted(
                {
                    member
                    for row in reset_rules + update_rules
                    for member in row["guard_members"]
                }
            )
            hold_rule = {
                "rule_id": stable_id(
                    "vcr_",
                    rtl_set_sha256,
                    instance_path,
                    target,
                    "implicit_hold",
                    length=24,
                ),
                "priority": len(rules),
                "guard_expression": "implicit_no_update",
                "guard_members": guard_members,
                "value_expression": target,
                "value_members": [_qualify(instance_path, target)],
                "update_kind": "hold",
                "statement_ids": [],
                "inference_rule": "implicit_sequential_hold",
            }
        width = None
        module_name = graph.resolve_signal(
            _qualify(instance_path, target)
        ).module_name
        if module_name is not None:
            info = graph._parser.modules[module_name].signals.get(target)
            width = info.width if info is not None else None
        identity = {
            "rtl_set_sha256": rtl_set_sha256,
            "profile_version": "chisel-semantic-profile",
            "object_type": "register_transition",
            "instance_path": instance_path,
            "signal": _qualify(instance_path, target),
            "clock_signal": clock_signal,
            "statement_ids": sorted(statement_ids),
        }
        result.append(
            {
                "register_id": stable_id("vcr_", identity, length=24),
                "instance_path": instance_path,
                "signal": _qualify(instance_path, target),
                "clock_signal": clock_signal,
                "width": width,
                "reset_rules": reset_rules,
                "update_rules": update_rules,
                "hold_rule": hold_rule,
                "statement_ids": sorted(statement_ids),
                "counter_pattern": (
                    "bounded_progress_counter"
                    if any(
                        row["update_kind"] == "increment"
                        for row in update_rules
                    )
                    and any(
                        row["update_kind"] in {"reset", "clear", "load"}
                        for row in reset_rules + update_rules
                    )
                    and width is not None
                    else None
                ),
                "inference_rule": "sequential_assignment_rules",
            }
        )
    return sorted(result, key=lambda row: row["register_id"])


def build_normalized_design(
    graph: InstanceGraph,
    *,
    rtl_set_sha256: str,
    clock_signal: str,
    features: Sequence[str],
) -> Dict[str, Any]:
    enabled = set(features)
    aliases = (
        _alias_classes(graph, rtl_set_sha256)
        if "compiler_net_normalization" in enabled
        else []
    )
    expressions = (
        _expression_groups(graph, rtl_set_sha256)
        if "compiler_net_normalization" in enabled
        else []
    )
    registers = (
        _register_transitions(graph, rtl_set_sha256, clock_signal)
        if {"register_transition", "pipeline"} & enabled
        else []
    )
    from .chisel_protocol_semantics import build_c3_semantics

    c3 = build_c3_semantics(
        graph,
        rtl_set_sha256=rtl_set_sha256,
        register_transitions=registers,
        features=features,
    )
    payload = {
        "schema_version": NORMALIZED_DESIGN_SCHEMA,
        "rtl_set_sha256": rtl_set_sha256,
        "profile_version": "chisel-semantic-profile",
        "features": sorted(features),
        "instances": graph.to_dict()["instances"],
        "alias_classes": aliases,
        "expression_groups": expressions,
        "register_transitions": registers,
        "aggregates": c3["aggregates"],
        "handshakes": c3["handshakes"],
        "pipelines": c3["pipelines"],
        "blocking_relations": c3["blocking_relations"],
        "provenance_hints": [],
        "diagnostics": [],
    }
    payload["normalized_design_id"] = stable_id("vcnd_", payload)
    return payload


def get_raw_members(
    normalized_design: Mapping[str, Any],
    semantic_ids: Iterable[str],
    *,
    max_members: int,
) -> Dict[str, Any]:
    """Expand only exact IDs already present in a normalized design."""
    if (
        normalized_design.get("schema_version")
        != NORMALIZED_DESIGN_SCHEMA
    ):
        raise SemanticQueryError("unsupported normalized design schema")
    if isinstance(max_members, bool) or max_members <= 0:
        raise SemanticQueryError("max_members must be positive")
    objects: Dict[str, Dict[str, Any]] = {}
    for collection, id_key, member_key, object_type in (
        ("alias_classes", "alias_id", "members", "alias_class"),
        (
            "expression_groups",
            "expression_id",
            "member_signals",
            "expression_group",
        ),
        (
            "register_transitions",
            "register_id",
            "signal",
            "register_transition",
        ),
        ("aggregates", "aggregate_id", "member_signals", "aggregate"),
        ("handshakes", "handshake_id", "member_signals", "handshake"),
        ("pipelines", "pipeline_id", "member_signals", "pipeline"),
        (
            "blocking_relations",
            "blocking_id",
            "member_signals",
            "blocking_relation",
        ),
    ):
        for row in normalized_design[collection]:
            members = (
                [row[member_key]]
                if isinstance(row[member_key], str)
                else list(row[member_key])
            )
            objects[row[id_key]] = {
                "semantic_id": row[id_key],
                "object_type": object_type,
                "raw_members": sorted(members),
            }
    requested = list(semantic_ids)
    if (
        not requested
        or any(not isinstance(item, str) or not item for item in requested)
        or len(set(requested)) != len(requested)
    ):
        raise SemanticQueryError(
            "semantic_ids must be a non-empty unique ID list"
        )
    missing = sorted(set(requested) - set(objects))
    if missing:
        raise SemanticQueryError(
            f"semantic IDs are absent from normalized design: {missing}"
        )
    rows = [objects[item] for item in sorted(requested)]
    member_count = sum(len(row["raw_members"]) for row in rows)
    if member_count > max_members:
        raise SemanticQueryError(
            "raw member expansion exceeds max_members"
        )
    result = {
        "schema_version": "chisel_raw_members_query",
        "normalized_design_id": normalized_design[
            "normalized_design_id"
        ],
        "objects": rows,
        "member_count": member_count,
        "max_members": max_members,
    }
    result["result_sha256"] = canonical_sha256(result)
    return result


def get_register_transition(
    normalized_design: Mapping[str, Any],
    register_id: str,
) -> Dict[str, Any]:
    if (
        normalized_design.get("schema_version")
        != NORMALIZED_DESIGN_SCHEMA
    ):
        raise SemanticQueryError("unsupported normalized design schema")
    matches = [
        row
        for row in normalized_design["register_transitions"]
        if row["register_id"] == register_id
    ]
    if len(matches) != 1:
        raise SemanticQueryError(
            "register_id is absent from normalized design"
        )
    result = {
        "schema_version": "chisel_register_transition_query",
        "normalized_design_id": normalized_design[
            "normalized_design_id"
        ],
        "register_transition": matches[0],
    }
    result["result_sha256"] = canonical_sha256(result)
    return result


def _environment(
    waveform: Any,
    instance_path: str,
    members: Iterable[str],
    cycle: int,
) -> Dict[str, str]:
    env: Dict[str, str] = {}
    for local in sorted(set(members)):
        qualified = _qualify(instance_path, local)
        value = waveform.get_signal_value(qualified, cycle)
        if value is not None:
            env[local] = value
            env[qualified] = value
    return env


def _waveform_signal(waveform: Any, signal: str) -> Optional[str]:
    if not hasattr(waveform, "signals"):
        return signal
    if signal in waveform.signals.by_name:
        return signal
    clean = re.sub(r"\s*\[\d+:\d+\]$", "", signal)
    matches = [
        str(item)
        for item in waveform.signals.by_name
        if re.sub(r"\s*\[\d+:\d+\]$", "", str(item)) == clean
    ]
    return matches[0] if len(matches) == 1 else None


def _active_rule(
    waveform: Any, transition: Mapping[str, Any], cycle: int
) -> Tuple[Optional[Mapping[str, Any]], Dict[str, str], str]:
    instance_path = str(transition["instance_path"])
    for rule in list(transition["reset_rules"]) + list(
        transition["update_rules"]
    ):
        guard = str(rule["guard_expression"])
        local_members = [
            item.rsplit(".", 1)[-1] for item in rule["guard_members"]
        ]
        env = _environment(waveform, instance_path, local_members, cycle)
        evaluated = ExpressionEvaluator(env).evaluate(guard)
        if evaluated is None:
            return None, env, "structural_only"
        if parse_binary_value(evaluated) not in (None, 0):
            return rule, env, "fully_observed"
    return transition.get("hold_rule"), {}, "fully_observed"


def persistent_intervals(
    normalized_design: Mapping[str, Any],
    waveform: Any,
    *,
    end_cycle: int,
    max_intervals: int,
    max_temporal_samples: int,
    subject_signals: Optional[Iterable[str]] = None,
) -> Tuple[list[Dict[str, Any]], list[Dict[str, Any]]]:
    """Compress consecutive activations of one exact update rule.

    Sampling is bounded by the request contract.  Unknown guard evaluation
    fails closed to ``structural_only`` and never receives a dynamic score.
    """
    intervals: list[Dict[str, Any]] = []
    diagnostics: list[Dict[str, Any]] = []
    samples = 0
    selected_signals = (
        {re.sub(r"\s*\[\d+:\d+\]$", "", item) for item in subject_signals}
        if subject_signals is not None
        else None
    )
    transitions = [
        row
        for row in normalized_design["register_transitions"]
        if selected_signals is None
        or re.sub(r"\s*\[\d+:\d+\]$", "", row["signal"])
        in selected_signals
    ]
    for transition in transitions:
        if len(intervals) >= max_intervals:
            break
        start = max(0, end_cycle - max_temporal_samples + samples + 1)
        current: Optional[Tuple[str, int, Dict[str, str], str]] = None
        for cycle in range(start, end_cycle + 1):
            if samples >= max_temporal_samples:
                diagnostics.append(
                    {
                        "code": "temporal_sample_budget_reached",
                        "message": "C2 persistent interval sampling reached its bound",
                        "breaks_complete": True,
                    }
                )
                return intervals, diagnostics
            samples += 1
            rule, guard_values, observation = _active_rule(
                waveform, transition, cycle
            )
            rule_id = str(rule["rule_id"]) if rule is not None else "unknown"
            if current is None:
                current = (rule_id, cycle, guard_values, observation)
                continue
            if (rule_id, observation) == (current[0], current[3]):
                continue
            _append_interval(
                intervals, transition, waveform, current, cycle - 1
            )
            if len(intervals) >= max_intervals:
                return intervals, diagnostics
            current = (rule_id, cycle, guard_values, observation)
        if current is not None:
            _append_interval(
                intervals, transition, waveform, current, end_cycle
            )
    return intervals, diagnostics


def _append_interval(
    intervals: list[Dict[str, Any]],
    transition: Mapping[str, Any],
    waveform: Any,
    current: Tuple[str, int, Dict[str, str], str],
    end_cycle: int,
) -> None:
    rule_id, start_cycle, guard_values, observation = current
    if end_cycle < start_cycle:
        return
    signal = str(transition["signal"])
    observed_signal = _waveform_signal(waveform, signal)
    start_value = (
        waveform.get_signal_value(observed_signal, start_cycle)
        if observed_signal is not None
        else None
    )
    end_value = (
        waveform.get_signal_value(observed_signal, end_cycle)
        if observed_signal is not None
        else None
    )
    start_int = parse_binary_value(start_value) if start_value is not None else None
    end_int = parse_binary_value(end_value) if end_value is not None else None
    selected = next(
        (
            row
            for row in list(transition["reset_rules"])
            + list(transition["update_rules"])
            if row["rule_id"] == rule_id
        ),
        None,
    )
    kind = selected["update_kind"] if selected is not None else "unknown"
    monotonic = (
        start_int is not None
        and end_int is not None
        and (
            (kind == "increment" and end_int >= start_int)
            or (kind == "decrement" and end_int <= start_int)
        )
    )
    identity = {
        "type": "persistent_interval",
        "register_id": transition["register_id"],
        "subject_id": rule_id,
        "start_cycle": start_cycle,
        "end_cycle": end_cycle,
        "observation": observation,
    }
    intervals.append(
        {
            "semantic_id": stable_id("vct_", identity, length=24),
            "type": "persistent_interval",
            "subject_id": rule_id,
            "register_id": transition["register_id"],
            "signal": signal,
            "waveform_signal": observed_signal,
            "start_cycle": start_cycle,
            "end_cycle": end_cycle,
            "sample_count": end_cycle - start_cycle + 1,
            "rule": kind,
            "guard_values": dict(sorted(guard_values.items())),
            "value_summary": {
                "start": start_int,
                "end": end_int,
                "monotonic": monotonic,
            },
            "observation": observation,
            "dynamic_score": 1.0 if observation == "fully_observed" else 0.0,
            "inference_rule": "persistent_update_rule_interval",
        }
    )
