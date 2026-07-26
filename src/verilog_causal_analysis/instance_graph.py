"""Deterministic elaborated instance identities for parsed Verilog designs.

This module is intentionally independent from Chisel naming conventions.  It
elaborates module definitions into exact instance paths, materializes
formal/actual port bindings, and provides instance-local dependency lookup.
Unsupported or ambiguous hierarchy is reported instead of falling back to a
global basename match.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

from .identity import stable_id
from .verilog_parser import (
    Dependency,
    DependencyLookupResult,
    DependencyType,
    StatementEvidence,
    VerilogParser,
)


INSTANCE_GRAPH_SCHEMA = "verilog_instance_graph.v1"


@dataclass(frozen=True)
class InstanceNode:
    instance_id: str
    instance_path: str
    parent_instance_id: Optional[str]
    instance_name: str
    module_name: str
    rtl_artifact_id: str
    declaration_statement_id: str


@dataclass(frozen=True)
class PortBinding:
    binding_id: str
    parent_instance_id: str
    child_instance_id: str
    formal_port: str
    direction: str
    actual_expression: str
    actual_member_signals: Tuple[str, ...]
    statement_id: str
    source_signal: str
    target_signal: str


@dataclass(frozen=True)
class InstanceSignalResolution:
    requested_signal: str
    instance_id: Optional[str]
    instance_path: Optional[str]
    module_name: Optional[str]
    local_signal: Optional[str]
    status: str

    @property
    def exact(self) -> bool:
        return self.status == "exact"


class InstanceGraphError(ValueError):
    """The parsed closure cannot be elaborated without guessing."""


class InstanceGraph:
    """Immutable exact instance graph derived from a :class:`VerilogParser`."""

    def __init__(
        self,
        *,
        top_module: str,
        instances: Iterable[InstanceNode],
        port_bindings: Iterable[PortBinding],
        diagnostics: Iterable[Mapping[str, Any]] = (),
    ):
        self.top_module = top_module
        self.instances = tuple(
            sorted(instances, key=lambda row: (row.instance_path, row.instance_id))
        )
        self.port_bindings = tuple(
            sorted(
                port_bindings,
                key=lambda row: (
                    row.target_signal,
                    row.source_signal,
                    row.binding_id,
                ),
            )
        )
        self.diagnostics = tuple(
            sorted(
                (dict(row) for row in diagnostics),
                key=lambda row: (str(row.get("code")), str(row.get("message"))),
            )
        )
        self._by_id = {row.instance_id: row for row in self.instances}
        self._by_path = {row.instance_path: row for row in self.instances}
        self._bindings_by_target: Dict[str, List[PortBinding]] = {}
        for binding in self.port_bindings:
            self._bindings_by_target.setdefault(binding.target_signal, []).append(
                binding
            )
        self._waveform_scope_signals: Optional[Dict[str, set[str]]] = None
        self._waveform_scope_modules: Dict[str, Optional[str]] = {}

    @staticmethod
    def _clean(signal: str) -> str:
        return re.sub(r"\s*\[\d+:\d+\]$", "", signal.strip())

    @classmethod
    def build(
        cls,
        parser: VerilogParser,
        artifact_by_path: Mapping[str, str],
        *,
        top_module: Optional[str] = None,
    ) -> "InstanceGraph":
        """Elaborate one exact top; missing/duplicate roots fail closed."""
        instantiated_modules = {
            child_module
            for module in parser.modules.values()
            for _instance_name, child_module, _line in module.submodule_instances
            if child_module in parser.modules
        }
        roots = sorted(set(parser.modules) - instantiated_modules)
        if top_module is None:
            if len(roots) != 1:
                raise InstanceGraphError(
                    "instance graph requires exactly one top module; "
                    f"candidates={roots}"
                )
            top_module = roots[0]
        elif top_module not in parser.modules:
            raise InstanceGraphError(f"unknown top module: {top_module}")

        normalized_artifacts = {
            os.path.abspath(path): artifact_id
            for path, artifact_id in artifact_by_path.items()
        }
        instances: List[InstanceNode] = []
        bindings: List[PortBinding] = []
        diagnostics: List[Dict[str, Any]] = []
        unresolved_instances: Dict[str, List[str]] = {}
        recursive_instances: Dict[str, List[str]] = {}
        by_path: Dict[str, InstanceNode] = {}

        def artifact_id(path: str) -> str:
            try:
                return normalized_artifacts[os.path.abspath(path)]
            except KeyError as error:
                raise InstanceGraphError(
                    "instance declaration references an unbound RTL artifact"
                ) from error

        def add_instance(
            module_name: str,
            instance_name: str,
            instance_path: str,
            parent: Optional[InstanceNode],
            line: int,
            active_modules: Tuple[str, ...],
        ) -> None:
            if instance_path in by_path:
                raise InstanceGraphError(
                    f"duplicate elaborated instance path: {instance_path}"
                )
            module = parser.modules[module_name]
            declaration_identity = {
                "parent_instance_id": parent.instance_id if parent else None,
                "instance_name": instance_name,
                "module_name": module_name,
                "artifact_id": artifact_id(module.file_path),
                "line": line,
            }
            node = InstanceNode(
                instance_id=stable_id("vci_", declaration_identity, length=24),
                instance_path=instance_path,
                parent_instance_id=parent.instance_id if parent else None,
                instance_name=instance_name,
                module_name=module_name,
                rtl_artifact_id=artifact_id(module.file_path),
                declaration_statement_id=stable_id(
                    "vcs_", declaration_identity, length=24
                ),
            )
            instances.append(node)
            by_path[instance_path] = node

            for child_name, child_module, child_line in sorted(
                module.submodule_instances
            ):
                child_path = f"{instance_path}.{child_name}"
                if child_module not in parser.modules:
                    unresolved_instances.setdefault(
                        child_module, []
                    ).append(child_path)
                    continue
                if child_module in active_modules:
                    recursive_instances.setdefault(
                        child_module, []
                    ).append(child_path)
                    continue
                add_instance(
                    child_module,
                    child_name,
                    child_path,
                    node,
                    child_line,
                    active_modules + (child_module,),
                )

        add_instance(
            top_module,
            top_module,
            top_module,
            None,
            parser.modules[top_module].line_start,
            (top_module,),
        )
        for child_module, paths in sorted(unresolved_instances.items()):
            diagnostics.append(
                {
                    "code": "instance_module_unresolved",
                    "message": (
                        f"{len(paths)} instance paths reference unavailable "
                        f"module {child_module}; first={min(paths)}"
                    ),
                    "module_name": child_module,
                    "instance_count": len(paths),
                }
            )
        for child_module, paths in sorted(recursive_instances.items()):
            diagnostics.append(
                {
                    "code": "recursive_instance_unsupported",
                    "message": (
                        f"{len(paths)} recursive instance paths for module "
                        f"{child_module}; first={min(paths)}"
                    ),
                    "module_name": child_module,
                    "instance_count": len(paths),
                }
            )

        for child in sorted(
            (row for row in instances if row.parent_instance_id is not None),
            key=lambda row: row.instance_path,
        ):
            parent = next(
                row for row in instances if row.instance_id == child.parent_instance_id
            )
            parent_module = parser.modules[parent.module_name]
            child_module = parser.modules[child.module_name]
            port_prefix = f"{child.instance_name}."
            for dep in parent_module.dependencies:
                port_side = None
                if dep.target.startswith(port_prefix):
                    port_side = dep.target
                elif dep.source.startswith(port_prefix):
                    port_side = dep.source
                if port_side is None:
                    continue
                formal = port_side[len(port_prefix) :]
                port = child_module.ports.get(formal)
                if port is None:
                    continue
                direction = port.signal_type
                child_formal = f"{child.instance_path}.{formal}"
                if parser._is_input_direction(direction):
                    actual = dep.source
                    source_signal = cls._qualify(parent.instance_path, actual)
                    target_signal = child_formal
                elif parser._is_output_direction(direction):
                    actual = dep.target
                    source_signal = child_formal
                    target_signal = cls._qualify(parent.instance_path, actual)
                else:
                    diagnostics.append(
                        {
                            "code": "instance_port_direction_unsupported",
                            "message": (
                                f"port direction for {child_formal} is unsupported"
                            ),
                            "instance_path": child.instance_path,
                        }
                    )
                    continue
                identity = {
                    "parent_instance_id": parent.instance_id,
                    "child_instance_id": child.instance_id,
                    "formal_port": formal,
                    "direction": direction,
                    "actual_expression": actual,
                    "statement_id": dep.statement_id,
                }
                binding = PortBinding(
                    binding_id=stable_id("vcb_", identity, length=24),
                    parent_instance_id=parent.instance_id,
                    child_instance_id=child.instance_id,
                    formal_port=formal,
                    direction=direction,
                    actual_expression=actual,
                    actual_member_signals=(source_signal if parser._is_input_direction(direction) else target_signal,),
                    statement_id=dep.statement_id,
                    source_signal=source_signal,
                    target_signal=target_signal,
                )
                if binding not in bindings:
                    bindings.append(binding)

        return cls(
            top_module=top_module,
            instances=instances,
            port_bindings=bindings,
            diagnostics=diagnostics,
        ).bind_parser(parser)

    @staticmethod
    def _qualify(instance_path: str, local_signal: str) -> str:
        clean = InstanceGraph._clean(local_signal)
        if clean.startswith(instance_path + "."):
            return clean
        return f"{instance_path}.{clean}"

    def resolve_signal(self, signal: str) -> InstanceSignalResolution:
        """Join a signal only to an exact elaborated instance path."""
        clean = self._clean(signal)
        hierarchy, separator, local = clean.rpartition(".")
        if separator and hierarchy not in self._by_path:
            module_name = self._resolve_waveform_scope(hierarchy)
            if module_name is not None:
                instance_id = stable_id(
                    "vci_",
                    {
                        "instance_path": hierarchy,
                        "module_name": module_name,
                        "inference_rule": (
                            "exact_waveform_scope_module_signature.v1"
                        ),
                    },
                    length=24,
                )
                return InstanceSignalResolution(
                    signal,
                    instance_id,
                    hierarchy,
                    module_name,
                    local,
                    "exact",
                )
        matches = [
            node
            for node in self.instances
            if clean.startswith(node.instance_path + ".")
        ]
        if not matches:
            module_name = (
                self._resolve_waveform_scope(hierarchy)
                if separator
                else None
            )
            if module_name is None:
                return InstanceSignalResolution(
                    signal, None, None, None, None, "unresolved"
                )
            instance_id = stable_id(
                "vci_",
                {
                    "instance_path": hierarchy,
                    "module_name": module_name,
                    "inference_rule": "exact_waveform_scope_module_signature.v1",
                },
                length=24,
            )
            return InstanceSignalResolution(
                signal,
                instance_id,
                hierarchy,
                module_name,
                local,
                "exact",
            )
        max_length = max(len(row.instance_path) for row in matches)
        longest = [row for row in matches if len(row.instance_path) == max_length]
        if len(longest) != 1:
            return InstanceSignalResolution(
                signal, None, None, None, None, "ambiguous"
            )
        instance = longest[0]
        local = clean[len(instance.instance_path) + 1 :]
        if not local:
            return InstanceSignalResolution(
                signal,
                instance.instance_id,
                instance.instance_path,
                instance.module_name,
                None,
                "unresolved",
            )
        return InstanceSignalResolution(
            signal,
            instance.instance_id,
            instance.instance_path,
            instance.module_name,
            local,
            "exact",
        )

    def bind_waveform(self, waveform: Any) -> "InstanceGraph":
        """Index exact waveform scopes for structural RTL-module joins.

        Some formal traces expose elaboration scopes which are absent from the
        emitted RTL's top-level instance declarations.  Such a scope is joined
        only when its immediate signal set has one uniquely best module
        definition with strong exact-name coverage.  Ties remain unresolved.
        """
        scopes: Dict[str, set[str]] = {}
        for raw_name in waveform.signals.by_name:
            clean = self._clean(str(raw_name))
            hierarchy, separator, local = clean.rpartition(".")
            if separator and hierarchy and local:
                scopes.setdefault(hierarchy, set()).add(local)
        self._waveform_scope_signals = scopes
        self._waveform_scope_modules.clear()
        return self

    def _resolve_waveform_scope(self, hierarchy: str) -> Optional[str]:
        if self._waveform_scope_signals is None:
            return None
        if hierarchy in self._waveform_scope_modules:
            return self._waveform_scope_modules[hierarchy]
        local_signals = self._waveform_scope_signals.get(hierarchy, set())
        if len(local_signals) < 3:
            self._waveform_scope_modules[hierarchy] = None
            return None
        scores: List[Tuple[int, float, str]] = []
        for module_name, module in self._parser.modules.items():
            declared = set(module.signals)
            intersection = len(local_signals & declared)
            coverage = intersection / max(1, len(declared))
            required_intersection = min(4, len(declared))
            if (
                intersection >= required_intersection
                and coverage >= 0.75
            ):
                scores.append((intersection, coverage, module_name))
        scores.sort(reverse=True)
        resolved: Optional[str] = None
        if scores:
            best = scores[0]
            runner = scores[1] if len(scores) > 1 else None
            if runner is None or (best[0], best[1]) > (
                runner[0],
                runner[1],
            ):
                resolved = best[2]
        self._waveform_scope_modules[hierarchy] = resolved
        return resolved

    def infer_module_from_signal(
        self, signal: str, hierarchy: Optional[str] = None
    ) -> Optional[str]:
        resolution = self.resolve_signal(signal)
        if not resolution.exact and hierarchy:
            resolution = self.resolve_signal(
                self._qualify(hierarchy, signal)
            )
        return resolution.module_name if resolution.exact else None

    def lookup_dependencies(
        self,
        signal: str,
        module_name: Optional[str] = None,
    ) -> DependencyLookupResult:
        """Return dependencies translated into exact instance identities."""
        resolution = self.resolve_signal(signal)
        if not resolution.exact:
            return DependencyLookupResult((), "unresolved", False)
        assert resolution.instance_path is not None
        assert resolution.module_name is not None
        assert resolution.local_signal is not None
        if module_name and module_name != resolution.module_name:
            return DependencyLookupResult((), "unresolved", False)

        local_lookup = self._parser.lookup_dependencies(
            resolution.local_signal, resolution.module_name
        )
        translated: List[Dependency] = []
        for dep in local_lookup.dependencies:
            translated.append(
                self._translated_dependency(dep, resolution.instance_path)
            )
        for binding in self._bindings_by_target.get(self._clean(signal), ()):
            translated.append(self._binding_dependency(binding))
        identities = {
            (dep.source_qualified, dep.target_qualified, dep.statement_id)
            for dep in translated
        }
        return DependencyLookupResult(
            tuple(translated),
            "exact" if translated else "unresolved",
            len(identities) != len(translated),
        )

    def get_dependencies_for_signal(
        self, signal: str, module_name: Optional[str] = None
    ) -> List[Dependency]:
        return list(self.lookup_dependencies(signal, module_name).dependencies)

    def get_signal_sources(
        self, signal: str, module_name: Optional[str] = None
    ) -> List[Tuple[str, DependencyType]]:
        return [
            (dep.source, dep.dep_type)
            for dep in self.get_dependencies_for_signal(signal, module_name)
        ]

    def get_rtl_context(
        self, signal: str, module_name: Optional[str] = None
    ) -> Dict[str, Any]:
        resolution = self.resolve_signal(signal)
        if not resolution.exact:
            return {
                "signal_name": signal,
                "found": False,
                "definition": None,
                "dependencies": [],
                "rtl_refs": [],
            }
        assert resolution.local_signal is not None
        assert resolution.module_name is not None
        return self._parser.get_rtl_context(
            resolution.local_signal, resolution.module_name
        )

    def bind_parser(self, parser: VerilogParser) -> "InstanceGraph":
        """Attach parser state after cache-safe graph construction."""
        self._parser = parser
        return self

    def _translated_dependency(
        self, dep: Dependency, instance_path: str
    ) -> Dependency:
        source = self._qualify(instance_path, dep.source)
        target = self._qualify(instance_path, dep.target)
        return Dependency(
            source=source,
            target=target,
            dep_type=dep.dep_type,
            source_qualified=source,
            target_qualified=target,
            statement=dep.statement,
        )

    def _binding_dependency(self, binding: PortBinding) -> Dependency:
        child = self._by_id[binding.child_instance_id]
        statement = self._parser._statement_evidence.get(
            binding.statement_id
        )
        if statement is None:
            statement = StatementEvidence(
                statement_id=binding.statement_id,
                expression=(
                    f".{binding.formal_port}"
                    f"({binding.actual_expression})"
                ),
                module_name=self._by_id[
                    binding.parent_instance_id
                ].module_name,
                target=binding.target_signal,
                target_qualified=binding.target_signal,
            )
        dep_type = (
            DependencyType.PORT_INPUT
            if binding.target_signal.startswith(child.instance_path + ".")
            else DependencyType.PORT_OUTPUT
        )
        return Dependency(
            source=binding.source_signal,
            target=binding.target_signal,
            dep_type=dep_type,
            source_qualified=binding.source_signal,
            target_qualified=binding.target_signal,
            statement=statement,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": INSTANCE_GRAPH_SCHEMA,
            "top_module": self.top_module,
            "instances": [
                {
                    "instance_id": row.instance_id,
                    "instance_path": row.instance_path,
                    "parent_instance_id": row.parent_instance_id,
                    "instance_name": row.instance_name,
                    "module_name": row.module_name,
                    "rtl_artifact_id": row.rtl_artifact_id,
                    "declaration_statement_id": row.declaration_statement_id,
                }
                for row in self.instances
            ],
            "port_bindings": [
                {
                    "binding_id": row.binding_id,
                    "parent_instance_id": row.parent_instance_id,
                    "child_instance_id": row.child_instance_id,
                    "formal_port": row.formal_port,
                    "direction": row.direction,
                    "actual_expression": row.actual_expression,
                    "actual_member_signals": list(row.actual_member_signals),
                    "statement_id": row.statement_id,
                    "source_signal": row.source_signal,
                    "target_signal": row.target_signal,
                }
                for row in self.port_bindings
            ],
            "diagnostics": [dict(row) for row in self.diagnostics],
        }

    @classmethod
    def from_parser(
        cls,
        parser: VerilogParser,
        artifact_by_path: Mapping[str, str],
        *,
        top_module: Optional[str] = None,
    ) -> "InstanceGraph":
        return cls.build(
            parser, artifact_by_path, top_module=top_module
        )
