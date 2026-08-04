"""Pure contribution-evidence contracts and scoring for local search.

The module intentionally does not read RTL, waveforms, or parser objects.  A
caller supplies exact observed values and intervention results; LS-C will wire
those callbacks into the slicer without duplicating the scoring semantics.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

from .identity import canonical_sha256


CONTRIBUTION_SCHEMA = "contribution_evidence_v2"
LEGACY_CONTRIBUTION_SCHEMA = "legacy_contribution_v1"
FIDELITY_STATUSES = frozenset(
    {"exact_match", "normalized_match", "uncheckable", "mismatch"}
)
TARGET_BASES = frozenset(
    {
        "exact_lhs_slice",
        "exact_endpoint_slice",
        "exact_predicate",
        "full_exact_target",
        "unknown",
    }
)
METHOD_RELIABILITY = {
    "expression_intervention": 1.0,
    "active_rule_intervention": 0.95,
    "branch_predicate_intervention": 0.85,
    "toggle_correlation": 0.45,
    "synthetic_temporal_evidence": 1.0,
    "structural_dependency": 0.15,
}
METHOD_DOMAINS = {
    "expression_intervention": "target_effect",
    "active_rule_intervention": "target_effect",
    "branch_predicate_intervention": "branch_activation",
    "toggle_correlation": "temporal_correlation",
    "synthetic_temporal_evidence": "temporal_correlation",
    "structural_dependency": "structural_dependency",
}
REASON_CODES = frozenset(
    {
        "counterfactual_changed_relevant_target",
        "branch_activation_changed",
        "no_intervention_changed_relevant_target",
        "no_valid_intervention",
        "intervention_budget_exhausted",
        "target_value_unknown",
        "original_result_unknown",
        "original_result_mismatch",
        "target_alignment_uncheckable",
        "inactive_data_rule",
        "selected_rule_ambiguous",
        "toggle_correlation_observed",
        "toggle_correlation_not_observed",
        "synthetic_temporal_evidence",
        "exact_structural_dependency",
    }
)
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class ContributionContractError(ValueError):
    """Raised when contribution evidence violates the frozen LS-CS contract."""


class ContributionStatus(str, Enum):
    SUPPORTED = "supported"
    NOT_SUPPORTED = "not_supported"
    INCONCLUSIVE = "inconclusive"
    STRUCTURAL_ONLY = "structural_only"


class EvidenceDomain(str, Enum):
    TARGET_EFFECT = "target_effect"
    BRANCH_ACTIVATION = "branch_activation"
    TEMPORAL_CORRELATION = "temporal_correlation"
    STRUCTURAL_DEPENDENCY = "structural_dependency"


class EvidenceMethod(str, Enum):
    EXPRESSION_INTERVENTION = "expression_intervention"
    ACTIVE_RULE_INTERVENTION = "active_rule_intervention"
    BRANCH_PREDICATE_INTERVENTION = "branch_predicate_intervention"
    TOGGLE_CORRELATION = "toggle_correlation"
    SYNTHETIC_TEMPORAL_EVIDENCE = "synthetic_temporal_evidence"
    STRUCTURAL_DEPENDENCY = "structural_dependency"


def _enum_value(value: Any, enum_type: type[Enum], where: str) -> str:
    try:
        return enum_type(value).value
    except (TypeError, ValueError) as error:
        raise ContributionContractError(f"{where} is invalid") from error


def _unit(value: Any, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ContributionContractError(f"{where} must be a finite number in [0, 1]")
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ContributionContractError(f"{where} must be a finite number in [0, 1]")
    return round(result, 6)


def _count(value: Any, where: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ContributionContractError(f"{where} must be a non-negative integer")
    return value


def _exact_keys(row: Mapping[str, Any], expected: Iterable[str], where: str) -> None:
    actual = set(row)
    expected_set = set(expected)
    if actual != expected_set:
        raise ContributionContractError(
            f"{where} keys mismatch: missing={sorted(expected_set - actual)}, "
            f"extra={sorted(actual - expected_set)}"
        )


def _bits(value: Any) -> Optional[str]:
    if not isinstance(value, str):
        return None
    normalized = value.replace("_", "").lower()
    return normalized if normalized and set(normalized) <= {"0", "1"} else None


def _literal_bits(value: int | str, width: int) -> Optional[str]:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return format(value, f"0{width}b") if 0 <= value < (1 << width) else None
    raw = value.replace("_", "").lower()
    if raw.startswith("0b"):
        raw = raw[2:]
    if set(raw) <= {"0", "1"} and 0 < len(raw) <= width:
        return raw.zfill(width)
    return None


@dataclass(frozen=True)
class Fidelity:
    status: str
    observed_target_basis: str
    original_result_matches_observed: Optional[bool]

    def __post_init__(self) -> None:
        if self.status not in FIDELITY_STATUSES:
            raise ContributionContractError("fidelity.status is invalid")
        if self.observed_target_basis not in TARGET_BASES:
            raise ContributionContractError("fidelity.observed_target_basis is invalid")
        if self.original_result_matches_observed not in (True, False, None):
            raise ContributionContractError(
                "fidelity.original_result_matches_observed must be boolean or null"
            )
        if self.status in {"exact_match", "normalized_match"}:
            if self.original_result_matches_observed is not True:
                raise ContributionContractError("matched fidelity requires an exact comparison")
        elif self.status == "mismatch" and self.original_result_matches_observed is not False:
            raise ContributionContractError("mismatch fidelity requires a failed comparison")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "observed_target_basis": self.observed_target_basis,
            "original_result_matches_observed": self.original_result_matches_observed,
        }


@dataclass(frozen=True)
class RelevantTarget:
    basis: str
    known_bit_count: int

    def __post_init__(self) -> None:
        if self.basis not in TARGET_BASES:
            raise ContributionContractError("relevant_target.basis is invalid")
        _count(self.known_bit_count, "relevant_target.known_bit_count")
        if self.basis != "unknown" and self.known_bit_count == 0:
            raise ContributionContractError("known relevant target must contain a bit")

    def to_dict(self) -> Dict[str, Any]:
        return {"basis": self.basis, "known_bit_count": self.known_bit_count}


@dataclass(frozen=True)
class InterventionSummary:
    planned: int
    evaluated: int
    changing: int
    complete: bool
    global_budget_truncated: bool
    results_sha256: str

    def __post_init__(self) -> None:
        for name in ("planned", "evaluated", "changing"):
            _count(getattr(self, name), f"interventions.{name}")
        if self.changing > self.evaluated or self.evaluated > self.planned:
            raise ContributionContractError("intervention counts are inconsistent")
        if self.complete != (self.evaluated == self.planned):
            raise ContributionContractError("interventions.complete disagrees with counts")
        if self.global_budget_truncated and self.complete:
            raise ContributionContractError("a complete intervention set is not truncated")
        if not isinstance(self.results_sha256, str) or not _SHA256_RE.fullmatch(
            self.results_sha256
        ):
            raise ContributionContractError("interventions.results_sha256 is invalid")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "planned": self.planned,
            "evaluated": self.evaluated,
            "changing": self.changing,
            "complete": self.complete,
            "global_budget_truncated": self.global_budget_truncated,
            "results_sha256": self.results_sha256,
        }


@dataclass(frozen=True)
class ContributionEffects:
    max_impact: float
    mean_impact: float
    change_rate: float
    min_intervention_cost: float

    def __post_init__(self) -> None:
        for name in (
            "max_impact",
            "mean_impact",
            "change_rate",
            "min_intervention_cost",
        ):
            object.__setattr__(self, name, _unit(getattr(self, name), f"effects.{name}"))

    def to_dict(self) -> Dict[str, float]:
        return {
            "max_impact": self.max_impact,
            "mean_impact": self.mean_impact,
            "change_rate": self.change_rate,
            "min_intervention_cost": self.min_intervention_cost,
        }


@dataclass(frozen=True)
class ContributionEvidence:
    status: str
    domain: str
    method: str
    score: float
    search_available: bool
    fidelity: Fidelity
    relevant_target: RelevantTarget
    interventions: InterventionSummary
    effects: ContributionEffects
    reason_code: str
    schema_version: str = field(default=CONTRIBUTION_SCHEMA, init=False)

    def __post_init__(self) -> None:
        status = _enum_value(self.status, ContributionStatus, "status")
        domain = _enum_value(self.domain, EvidenceDomain, "domain")
        method = _enum_value(self.method, EvidenceMethod, "method")
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "domain", domain)
        object.__setattr__(self, "method", method)
        object.__setattr__(self, "score", _unit(self.score, "score"))
        if METHOD_DOMAINS[method] != domain:
            raise ContributionContractError("method does not belong to evidence domain")
        if self.score > METHOD_RELIABILITY[method]:
            raise ContributionContractError("score exceeds method reliability cap")
        if self.reason_code not in REASON_CODES:
            raise ContributionContractError("reason_code is invalid")
        if status == "inconclusive":
            if self.search_available or self.score != 0.0:
                raise ContributionContractError("inconclusive evidence must be unavailable")
        else:
            if not self.search_available:
                raise ContributionContractError("conclusive evidence must be search available")
        if status == "not_supported":
            if self.score != 0.0 or not self.interventions.complete or self.interventions.planned == 0:
                raise ContributionContractError("not_supported requires a complete non-empty zero result")
        if status == "supported" and self.score <= 0.0:
            raise ContributionContractError("supported evidence requires a positive score")
        if status == "structural_only":
            if domain != "structural_dependency" or method != "structural_dependency" or self.score != 0.15:
                raise ContributionContractError("structural_only evidence must have score 0.15")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "status": self.status,
            "domain": self.domain,
            "method": self.method,
            "score": self.score,
            "search_available": self.search_available,
            "fidelity": self.fidelity.to_dict(),
            "relevant_target": self.relevant_target.to_dict(),
            "interventions": self.interventions.to_dict(),
            "effects": self.effects.to_dict(),
            "reason_code": self.reason_code,
        }

    @classmethod
    def from_dict(cls, row: Mapping[str, Any]) -> "ContributionEvidence":
        if not isinstance(row, Mapping):
            raise ContributionContractError("contribution_evidence must be an object")
        _exact_keys(
            row,
            {
                "schema_version",
                "status",
                "domain",
                "method",
                "score",
                "search_available",
                "fidelity",
                "relevant_target",
                "interventions",
                "effects",
                "reason_code",
            },
            "contribution_evidence",
        )
        if row["schema_version"] != CONTRIBUTION_SCHEMA:
            raise ContributionContractError(
                f"schema_version must be {CONTRIBUTION_SCHEMA}"
            )
        fidelity = row["fidelity"]
        target = row["relevant_target"]
        interventions = row["interventions"]
        effects = row["effects"]
        for value, keys, where in (
            (
                fidelity,
                {"status", "observed_target_basis", "original_result_matches_observed"},
                "fidelity",
            ),
            (target, {"basis", "known_bit_count"}, "relevant_target"),
            (
                interventions,
                {
                    "planned",
                    "evaluated",
                    "changing",
                    "complete",
                    "global_budget_truncated",
                    "results_sha256",
                },
                "interventions",
            ),
            (
                effects,
                {"max_impact", "mean_impact", "change_rate", "min_intervention_cost"},
                "effects",
            ),
        ):
            if not isinstance(value, Mapping):
                raise ContributionContractError(f"{where} must be an object")
            _exact_keys(value, keys, where)
        if not isinstance(row["search_available"], bool):
            raise ContributionContractError("search_available must be boolean")
        return cls(
            status=row["status"],
            domain=row["domain"],
            method=row["method"],
            score=row["score"],
            search_available=row["search_available"],
            fidelity=Fidelity(**fidelity),
            relevant_target=RelevantTarget(**target),
            interventions=InterventionSummary(**interventions),
            effects=ContributionEffects(**effects),
            reason_code=row["reason_code"],
        )


def generate_interventions(
    observed_value: str,
    *,
    operator: Optional[str] = None,
    literal: Optional[int | str] = None,
    boolean_selector: bool = False,
    max_interventions: int = 4,
) -> Tuple[str, ...]:
    """Return a canonical, bounded set of exact-width binary interventions."""

    observed = _bits(observed_value)
    if observed is None or max_interventions <= 0:
        return ()
    width = len(observed)
    maximum = (1 << width) - 1
    candidates: list[tuple[int, str]] = []

    def add(priority: int, value: Optional[str]) -> None:
        if value is not None and len(value) == width and value != observed:
            candidates.append((priority, value))

    directed = _literal_bits(literal, width) if literal is not None else None
    if operator in {"==", "!="} and directed is not None:
        add(0, directed)
        directed_int = int(directed, 2)
        add(0, format(directed_int ^ 1, f"0{width}b"))
    elif operator in {"<", "<=", ">", ">="} and directed is not None:
        boundary = int(directed, 2)
        for value in (boundary, boundary - 1, boundary + 1):
            add(0, format(value, f"0{width}b") if 0 <= value <= maximum else None)
    if boolean_selector or width == 1:
        add(0, "1" if observed == "0" else "0")

    low_flip = list(observed)
    low_flip[-1] = "1" if low_flip[-1] == "0" else "0"
    add(1, "".join(low_flip))
    high_flip = list(observed)
    high_flip[0] = "1" if high_flip[0] == "0" else "0"
    add(1, "".join(high_flip))
    add(2, "0" * width)
    add(2, "1" * width)
    add(3, "".join("1" if bit == "0" else "0" for bit in observed))

    unique: Dict[str, int] = {}
    for priority, value in candidates:
        unique[value] = min(priority, unique.get(value, priority))
    ordered = sorted(
        unique,
        key=lambda value: (
            unique[value],
            sum(left != right for left, right in zip(observed, value)),
            int(value, 2),
        ),
    )
    return tuple(ordered[:max_interventions])


def fidelity_gate(
    original_result: Optional[str],
    observed_target: Optional[str],
    *,
    target_basis: str,
) -> Fidelity:
    """Check exact or explicit zero-extension alignment before scoring effects."""

    if target_basis not in TARGET_BASES or target_basis == "unknown":
        return Fidelity("uncheckable", "unknown", None)
    original = _bits(original_result)
    observed = _bits(observed_target)
    if original is None or observed is None:
        return Fidelity("uncheckable", target_basis, None)
    if len(original) == len(observed):
        matches = original == observed
        return Fidelity("exact_match" if matches else "mismatch", target_basis, matches)
    width = max(len(original), len(observed))
    matches = original.zfill(width) == observed.zfill(width)
    return Fidelity("normalized_match" if matches else "mismatch", target_basis, matches)


def _masked_bits(value: str, mask: Optional[Sequence[int]]) -> Optional[str]:
    bits = _bits(value)
    if bits is None:
        return None
    if mask is None:
        return bits
    if not mask or len(set(mask)) != len(mask) or any(
        isinstance(index, bool) or not isinstance(index, int) or index < 0 or index >= len(bits)
        for index in mask
    ):
        return None
    # Mask indices use normal hardware convention: bit 0 is the rightmost bit.
    return "".join(bits[-1 - index] for index in sorted(mask, reverse=True))


def _zero_extend(value: Optional[str], width: int) -> Optional[str]:
    bits = _bits(value)
    if bits is None or len(bits) > width:
        return None
    return bits.zfill(width)


def evaluate_interventions(
    *,
    source_value: str,
    original_result: Optional[str],
    observed_target: Optional[str],
    intervention_values: Sequence[str],
    intervention_results: Sequence[Optional[str]],
    target_basis: str,
    relevant_target_mask: Optional[Sequence[int]] = None,
    method: str = "expression_intervention",
    global_budget_truncated: bool = False,
    rule_active: Optional[bool] = True,
    selected_rule_exact: bool = True,
) -> ContributionEvidence:
    """Apply the C0 fidelity gate and aggregate supplied intervention results."""

    method = _enum_value(method, EvidenceMethod, "method")
    if method not in {
        "expression_intervention",
        "active_rule_intervention",
        "branch_predicate_intervention",
    }:
        raise ContributionContractError("evaluate_interventions requires an intervention method")
    if len(intervention_results) != len(intervention_values):
        raise ContributionContractError("each planned intervention requires a result slot")
    source = _bits(source_value)
    fidelity = fidelity_gate(original_result, observed_target, target_basis=target_basis)
    original = _bits(original_result)
    observed = _bits(observed_target)
    target_width = max(len(original or ""), len(observed or ""))
    aligned_original = _zero_extend(original, target_width)
    masked_original = (
        _masked_bits(aligned_original, relevant_target_mask)
        if aligned_original is not None
        else None
    )
    relevant = RelevantTarget(
        target_basis if masked_original is not None else "unknown",
        len(masked_original) if masked_original is not None else 0,
    )

    def inconclusive(reason: str) -> ContributionEvidence:
        rows = [
            {"intervention": value, "result": result}
            for value, result in zip(intervention_values, intervention_results)
            if result is not None
        ]
        evaluated = len(rows)
        return ContributionEvidence(
            status="inconclusive",
            domain=METHOD_DOMAINS[method],
            method=method,
            score=0.0,
            search_available=False,
            fidelity=fidelity,
            relevant_target=relevant,
            interventions=InterventionSummary(
                len(intervention_values),
                evaluated,
                0,
                evaluated == len(intervention_values),
                global_budget_truncated,
                canonical_sha256(rows),
            ),
            effects=ContributionEffects(0.0, 0.0, 0.0, 0.0),
            reason_code=reason,
        )

    if method == "active_rule_intervention":
        if not selected_rule_exact or rule_active is None:
            return inconclusive("selected_rule_ambiguous")
        if rule_active is False:
            return inconclusive("inactive_data_rule")
    if source is None:
        return inconclusive("no_valid_intervention")
    if original is None:
        return inconclusive("original_result_unknown")
    if _bits(observed_target) is None:
        return inconclusive("target_value_unknown")
    if fidelity.status == "mismatch":
        return inconclusive("original_result_mismatch")
    if fidelity.status == "uncheckable" or masked_original is None:
        return inconclusive("target_alignment_uncheckable")
    if not intervention_values:
        return inconclusive("no_valid_intervention")

    rows: list[Dict[str, Any]] = []
    impacts: list[float] = []
    costs: list[float] = []
    changing = 0
    for intervention, result in zip(intervention_values, intervention_results):
        changed_source = _bits(intervention)
        aligned_result = _zero_extend(result, target_width)
        masked_result = (
            _masked_bits(aligned_result, relevant_target_mask)
            if aligned_result is not None
            else None
        )
        if changed_source is None or len(changed_source) != len(source) or masked_result is None:
            continue
        changed_source_bits = sum(a != b for a, b in zip(source, changed_source))
        cost = max(0, changed_source_bits - 1) / max(1, len(source) - 1)
        comparable = len(masked_original)
        changed_target_bits = sum(a != b for a, b in zip(masked_original, masked_result))
        effect = changed_target_bits / comparable
        impact = effect * (1.0 - 0.25 * cost)
        if effect > 0.0:
            changing += 1
        impacts.append(impact)
        costs.append(cost)
        rows.append(
            {
                "intervention": changed_source,
                "result": result,
                "impact": round(impact, 6),
                "cost": round(cost, 6),
            }
        )

    planned = len(intervention_values)
    evaluated = len(rows)
    complete = evaluated == planned
    truncated = global_budget_truncated or not complete
    effects = ContributionEffects(
        max(impacts, default=0.0),
        sum(impacts) / planned,
        changing / planned,
        min(costs, default=0.0),
    )
    summary = InterventionSummary(
        planned,
        evaluated,
        changing,
        complete,
        truncated,
        canonical_sha256(rows),
    )
    if changing == 0:
        if not complete:
            return ContributionEvidence(
                "inconclusive",
                METHOD_DOMAINS[method],
                method,
                0.0,
                False,
                fidelity,
                relevant,
                summary,
                effects,
                "intervention_budget_exhausted",
            )
        return ContributionEvidence(
            "not_supported",
            METHOD_DOMAINS[method],
            method,
            0.0,
            True,
            fidelity,
            relevant,
            summary,
            effects,
            "no_intervention_changed_relevant_target",
        )

    effect_strength = (
        0.35
        + 0.35 * effects.max_impact
        + 0.15 * effects.mean_impact
        + 0.15 * effects.change_rate
    )
    coverage = 0.75 + 0.25 * evaluated / planned
    fidelity_quality = 1.0 if fidelity.status == "exact_match" else 0.9
    score = round(
        min(1.0, effect_strength * coverage * fidelity_quality * METHOD_RELIABILITY[method]),
        6,
    )
    reason = (
        "branch_activation_changed"
        if METHOD_DOMAINS[method] == "branch_activation"
        else "counterfactual_changed_relevant_target"
    )
    return ContributionEvidence(
        "supported",
        METHOD_DOMAINS[method],
        method,
        score,
        True,
        fidelity,
        relevant,
        summary,
        effects,
        reason,
    )


def structural_evidence() -> ContributionEvidence:
    empty_digest = canonical_sha256([])
    return ContributionEvidence(
        "structural_only",
        "structural_dependency",
        "structural_dependency",
        0.15,
        True,
        Fidelity("uncheckable", "unknown", None),
        RelevantTarget("unknown", 0),
        InterventionSummary(0, 0, 0, True, False, empty_digest),
        ContributionEffects(0.0, 0.0, 0.0, 0.0),
        "exact_structural_dependency",
    )


def toggle_evidence(*, source_toggled: bool, target_toggled: bool) -> ContributionEvidence:
    supported = source_toggled and target_toggled
    return ContributionEvidence(
        "supported" if supported else "inconclusive",
        "temporal_correlation",
        "toggle_correlation",
        0.45 if supported else 0.0,
        supported,
        Fidelity("uncheckable", "unknown", None),
        RelevantTarget("unknown", 0),
        InterventionSummary(0, 0, 0, True, False, canonical_sha256([])),
        ContributionEffects(0.0, 0.0, 0.0, 0.0),
        "toggle_correlation_observed" if supported else "toggle_correlation_not_observed",
    )


@dataclass(frozen=True)
class RoutedContribution:
    feature_name: str
    value: Optional[float]
    availability: str


def route_contribution(evidence: ContributionEvidence) -> RoutedContribution:
    """Route one evidence record to exactly one primary local-search feature."""

    if evidence.domain == "target_effect":
        return RoutedContribution(
            "C_cf",
            evidence.score if evidence.search_available else None,
            "available" if evidence.search_available else "not_available",
        )
    if evidence.domain == "branch_activation":
        return RoutedContribution(
            "C_ctrl",
            evidence.score if evidence.search_available else None,
            "available" if evidence.search_available else "not_available",
        )
    if evidence.domain == "temporal_correlation":
        return RoutedContribution(
            "C_time",
            evidence.score if evidence.search_available else None,
            "available" if evidence.search_available else "not_available",
        )
    return RoutedContribution("C_structural", evidence.score, "available")


def contribution_cache_key(
    *,
    statement_key: str,
    source_identity: str,
    source_cycle: int,
    intervention_value: str,
    target_identity: str,
    target_cycle: int,
    relevant_target_basis: str,
    relevant_target_mask: Optional[Sequence[int]],
    analyzer_revision: str,
) -> str:
    return canonical_sha256(
        {
            "schema_version": CONTRIBUTION_SCHEMA,
            "statement_key": statement_key,
            "source_identity": source_identity,
            "source_cycle": source_cycle,
            "intervention_value": intervention_value,
            "target_identity": target_identity,
            "target_cycle": target_cycle,
            "relevant_target_basis": relevant_target_basis,
            "relevant_target_mask": None if relevant_target_mask is None else list(relevant_target_mask),
            "analyzer_revision": analyzer_revision,
        }
    )


@dataclass
class ContributionWorkCounter:
    candidate_evaluations: int = 0
    intervention_evaluations: int = 0

    def record_candidate(self) -> None:
        self.candidate_evaluations += 1

    def record_intervention(self) -> None:
        self.intervention_evaluations += 1


@dataclass(frozen=True)
class LegacyContributionEvidence:
    legacy_method: str
    legacy_score: float
    expression_evaluations: int
    intervention_evaluations: int
    change_examples: Tuple[Mapping[str, Any], ...] = ()
    schema_version: str = field(default=LEGACY_CONTRIBUTION_SCHEMA, init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.legacy_method, str) or not self.legacy_method:
            raise ContributionContractError("legacy_method must be non-empty")
        object.__setattr__(self, "legacy_score", _unit(self.legacy_score, "legacy_score"))
        _count(self.expression_evaluations, "expression_evaluations")
        _count(self.intervention_evaluations, "intervention_evaluations")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "legacy_method": self.legacy_method,
            "legacy_score": self.legacy_score,
            "expression_evaluations": self.expression_evaluations,
            "intervention_evaluations": self.intervention_evaluations,
            "change_examples": [dict(row) for row in self.change_examples],
        }


def adapt_legacy_contribution(
    *,
    legacy_method: str,
    legacy_score: float,
    expression_evaluations: int,
    intervention_evaluations: int,
    change_examples: Iterable[Mapping[str, Any]] = (),
) -> LegacyContributionEvidence:
    """Wrap a frozen v1 evaluator result without presenting it as v2 evidence."""

    return LegacyContributionEvidence(
        legacy_method=legacy_method,
        legacy_score=legacy_score,
        expression_evaluations=expression_evaluations,
        intervention_evaluations=intervention_evaluations,
        change_examples=tuple(dict(row) for row in change_examples),
    )


def contribution_edge_fields(
    evidence: ContributionEvidence | LegacyContributionEvidence,
) -> Dict[str, Any]:
    """Create the exact scalar-plus-envelope projection used by graph edges."""

    score = evidence.score if isinstance(evidence, ContributionEvidence) else evidence.legacy_score
    return {"contribution_score": score, "contribution_evidence": evidence.to_dict()}
