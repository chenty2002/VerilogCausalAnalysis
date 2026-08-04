"""Frozen local-search policy, feature, and summary contracts.

This module deliberately contains no traversal or scoring implementation.  LS-A
owns the durable identities consumed by the later local-search work packages.
"""

from __future__ import annotations

import math
import re
import heapq
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

from .identity import canonical_sha256


FEATURE_SCHEMA = "vca_search_features_v1"
SEARCH_SUMMARY_SCHEMA = "vca_search_summary_v1"
POLICY_IDS = (
    "legacy_dfs_v1",
    "legacy_scalar_best_first_v1",
    "edge_best_first_v1",
    "chisel_hybrid_best_first_v1",
)
POSITIVE_FEATURES = (
    "C_cf",
    "C_obs",
    "C_time",
    "C_ctrl",
    "C_sem",
    "C_structural",
)
PENALTY_FEATURES = ("P_unknown", "P_ambiguity", "P_temp", "P_fanout")
FEATURE_AVAILABILITIES = frozenset(
    {"available", "not_available", "not_applicable"}
)
TERMINATION_REASONS = frozenset(
    {
        "frontier_exhausted",
        "max_signal_nodes",
        "max_expanded_nodes",
        "max_candidate_evaluations",
        "max_intervention_evaluations",
        "max_signal_depth",
        "unknown_value_frontier",
        "identity_ambiguous",
        "source_projection_ambiguous",
    }
)
CONTRIBUTION_STATUSES = (
    "supported",
    "not_supported",
    "inconclusive",
    "structural_only",
)
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class LocalSearchContractError(ValueError):
    """Raised when a frozen local-search contract is malformed."""


def _exact_keys(row: Mapping[str, Any], expected: Iterable[str], where: str) -> None:
    expected_set = set(expected)
    actual = set(row)
    if actual != expected_set:
        raise LocalSearchContractError(
            f"{where} keys mismatch: missing={sorted(expected_set - actual)}, "
            f"extra={sorted(actual - expected_set)}"
        )


def _finite_unit(value: Any, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise LocalSearchContractError(f"{where} must be a finite number in [0, 1]")
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise LocalSearchContractError(f"{where} must be a finite number in [0, 1]")
    return result


def _non_negative_int(value: Any, where: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise LocalSearchContractError(f"{where} must be a non-negative integer")
    return value


@dataclass(frozen=True)
class SearchPolicyIdentity:
    policy_id: str
    feature_schema: str
    policy_sha256: str

    @classmethod
    def from_dict(cls, row: Mapping[str, Any]) -> "SearchPolicyIdentity":
        if not isinstance(row, Mapping):
            raise LocalSearchContractError("search_policy must be an object")
        _exact_keys(
            row,
            {"policy_id", "feature_schema", "policy_sha256"},
            "search_policy",
        )
        policy_id = row["policy_id"]
        if policy_id not in POLICY_REGISTRY:
            raise LocalSearchContractError(
                f"search_policy.policy_id must be one of {list(POLICY_IDS)}"
            )
        if row["feature_schema"] != FEATURE_SCHEMA:
            raise LocalSearchContractError(
                f"search_policy.feature_schema must be {FEATURE_SCHEMA}"
            )
        digest = row["policy_sha256"]
        if not isinstance(digest, str) or not _SHA256_RE.fullmatch(digest):
            raise LocalSearchContractError(
                "search_policy.policy_sha256 must be a lowercase SHA-256"
            )
        expected = POLICY_REGISTRY[policy_id].policy_sha256
        if digest != expected:
            raise LocalSearchContractError(
                f"search_policy.policy_sha256 mismatch for {policy_id}"
            )
        return cls(policy_id, FEATURE_SCHEMA, digest)

    def to_dict(self) -> Dict[str, str]:
        return {
            "policy_id": self.policy_id,
            "feature_schema": self.feature_schema,
            "policy_sha256": self.policy_sha256,
        }


@dataclass(frozen=True)
class SearchPolicy:
    policy_id: str
    payload: Mapping[str, Any]
    policy_sha256: str

    @property
    def identity(self) -> SearchPolicyIdentity:
        return SearchPolicyIdentity(self.policy_id, FEATURE_SCHEMA, self.policy_sha256)


def _feature_value_tables() -> Dict[str, Any]:
    return {
        "observation_identity": {
            "exact_known": 1.0,
            "hierarchy_inferred_known": 0.6,
            "unresolved_ambiguous_or_unknown": 0.0,
        },
        "temporal_alignment": {
            "aligned_transition": 1.0,
            "exact_sequential_predecessor": 0.8,
            "same_cycle_combinational": 0.4,
        },
        "control_relevance": {
            "active_reset_update_or_guard": 1.0,
            "exact_sequential_state": 0.85,
            "exact_conditional": 0.7,
            "assertion_antecedent_member": 0.6,
            "ordinary_data": 0.3,
            "port_or_alias_passthrough": 0.1,
        },
        "chisel_semantic_specificity": {
            "exact_rule_protocol_or_pipeline_unique_source": 1.0,
            "normalized_object_without_source_authority": 0.8,
            "expression_group_or_aggregate_member": 0.65,
            "hash_bound_ordinary_rtl_statement": 0.55,
            "unverified_locator_hint": 0.3,
            "unmerged_compiler_temporary": 0.1,
            "no_semantic_match": 0.0,
        },
        "structural_support": {
            "exact_dependency_dynamic_unconfirmed": 0.15,
            "rtl_context_missing": 0.0,
        },
        "legacy_contribution": {
            "sva_antecedent_fallback": 0.85,
            "toggle_correlation": 0.7,
            "weak_structural": 0.3,
        },
    }


def _policy_payload(policy_id: str) -> Dict[str, Any]:
    hybrid = policy_id != "legacy_dfs_v1"
    if policy_id == "chisel_hybrid_best_first_v1":
        positive = {
            "C_cf": 0.35,
            "C_obs": 0.15,
            "C_time": 0.15,
            "C_ctrl": 0.15,
            "C_sem": 0.2,
            "C_structural": 0.1,
        }
    else:
        positive = {
            "C_cf": 0.5,
            "C_obs": 0.2,
            "C_time": 0.15,
            "C_ctrl": 0.15,
            "C_sem": None,
            "C_structural": 0.1,
        }
    return {
        "policy_id": policy_id,
        "feature_schema": FEATURE_SCHEMA,
        "scheduler_kind": "hybrid_best_first" if hybrid else "legacy_lifo_dfs",
        "exploration_period": 5,
        "weak_beam_width": 2,
        "max_support_paths_per_node": 3,
        "max_interventions_per_candidate": 4,
        "score_round_digits": 6,
        "path_epsilon": 0.000001,
        "positive_feature_weights": positive,
        "penalty_weights": {
            "P_unknown": 0.2,
            "P_ambiguity": 0.15,
            "P_temp": 0.1,
            "P_fanout": 0.1,
        },
        "path_mix": {"geometric_mean": 0.75, "weakest_edge": 0.25},
        "frontier_mix": {"local": 0.65, "path": 0.25, "seed": 0.1},
        "feature_value_tables": _feature_value_tables(),
    }


def _build_registry() -> Mapping[str, SearchPolicy]:
    def freeze(value: Any) -> Any:
        if isinstance(value, dict):
            return MappingProxyType(
                {key: freeze(item) for key, item in value.items()}
            )
        if isinstance(value, list):
            return tuple(freeze(item) for item in value)
        return value

    rows: Dict[str, SearchPolicy] = {}
    for policy_id in POLICY_IDS:
        payload = _policy_payload(policy_id)
        rows[policy_id] = SearchPolicy(
            policy_id=policy_id,
            payload=freeze(payload),
            policy_sha256=canonical_sha256(payload),
        )
    return MappingProxyType(rows)


POLICY_REGISTRY = _build_registry()


def policy_identity(policy_id: str) -> SearchPolicyIdentity:
    try:
        return POLICY_REGISTRY[policy_id].identity
    except KeyError as error:
        raise LocalSearchContractError(f"unknown search policy {policy_id!r}") from error


def make_search_summary(
    policy: SearchPolicyIdentity,
    *,
    termination_reason: str,
    seed_count: int,
    expanded_nodes: int = 0,
    candidate_evaluations: int = 0,
    intervention_evaluations: int = 0,
    admitted_nodes: int = 0,
    admitted_edges: int = 0,
    exploit_expansions: int = 0,
    explore_expansions: int = 0,
    frontier_ids: Iterable[str] = (),
    unevaluated_candidate_ids: Iterable[str] = (),
    unevaluated_intervention_ids: Iterable[str] = (),
) -> Dict[str, Any]:
    """Build a canonical minimal summary for producers during LS-A migration."""

    frontier = sorted(set(frontier_ids))
    candidates = sorted(set(unevaluated_candidate_ids))
    interventions = sorted(set(unevaluated_intervention_ids))
    row = {
        "schema_version": SEARCH_SUMMARY_SCHEMA,
        **policy.to_dict(),
        "termination_reason": termination_reason,
        "seed_count": seed_count,
        "expanded_nodes": expanded_nodes,
        "candidate_evaluations": candidate_evaluations,
        "intervention_evaluations": intervention_evaluations,
        "admitted_nodes": admitted_nodes,
        "admitted_edges": admitted_edges,
        "exploit_expansions": exploit_expansions,
        "explore_expansions": explore_expansions,
        "frontier_remaining": len(frontier),
        "frontier_sha256": canonical_sha256(frontier),
        "unevaluated_candidate_count": len(candidates),
        "unevaluated_sha256": canonical_sha256(candidates),
        "unevaluated_intervention_count": len(interventions),
        "unevaluated_intervention_sha256": canonical_sha256(interventions),
        "contribution_status_counts": {
            status: 0 for status in CONTRIBUTION_STATUSES
        },
        "contribution_method_counts": {},
        "feature_missing_counts": {},
        "rejection_counts": {},
        "score_summary": {"min": 0.0, "max": 0.0, "mean": 0.0},
    }
    return validate_search_summary(row, expected_policy=policy)


@dataclass(frozen=True)
class ScoreFeatures:
    """Exact feature vector with availability distinct from observed zero."""

    values: Mapping[str, Optional[float]]
    availability: Mapping[str, str]

    @classmethod
    def from_dict(cls, row: Mapping[str, Any]) -> "ScoreFeatures":
        _exact_keys(row, {"feature_vector", "feature_availability"}, "score_features")
        values = row["feature_vector"]
        availability = row["feature_availability"]
        feature_names = POSITIVE_FEATURES + PENALTY_FEATURES
        if not isinstance(values, Mapping) or not isinstance(availability, Mapping):
            raise LocalSearchContractError("score feature fields must be objects")
        _exact_keys(values, feature_names, "score_features.feature_vector")
        _exact_keys(
            availability, feature_names, "score_features.feature_availability"
        )
        parsed_values: Dict[str, Optional[float]] = {}
        parsed_availability: Dict[str, str] = {}
        for name in feature_names:
            state = availability[name]
            if state not in FEATURE_AVAILABILITIES:
                raise LocalSearchContractError(f"{name} availability is invalid")
            value = values[name]
            if state == "available":
                parsed_values[name] = _finite_unit(value, f"score_features.{name}")
            elif value is not None:
                raise LocalSearchContractError(
                    f"score_features.{name} must be null when {state}"
                )
            else:
                parsed_values[name] = None
            parsed_availability[name] = state
        return cls(MappingProxyType(parsed_values), MappingProxyType(parsed_availability))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "feature_vector": {name: self.values[name] for name in sorted(self.values)},
            "feature_availability": {
                name: self.availability[name] for name in sorted(self.availability)
            },
        }


@dataclass(frozen=True)
class ScoreResult:
    """Availability-aware local score used only for search scheduling."""

    local_score: float
    positive_score: float
    penalty_score: float
    missing_positive_features: Tuple[str, ...]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "local_score": self.local_score,
            "positive_score": self.positive_score,
            "penalty_score": self.penalty_score,
            "missing_positive_features": list(self.missing_positive_features),
        }


def score_features(policy: SearchPolicy | str, features: ScoreFeatures) -> ScoreResult:
    """Compute the frozen v1 local score without treating missing data as zero."""

    if isinstance(policy, str):
        try:
            policy = POLICY_REGISTRY[policy]
        except KeyError as error:
            raise LocalSearchContractError(f"unknown search policy {policy!r}") from error
    positive_weights = policy.payload["positive_feature_weights"]
    weighted_sum = 0.0
    weight_sum = 0.0
    missing = []
    for name in POSITIVE_FEATURES:
        weight = positive_weights[name]
        if weight is None:
            continue
        if features.availability[name] != "available":
            missing.append(name)
            continue
        weighted_sum += float(weight) * float(features.values[name])
        weight_sum += float(weight)
    positive = weighted_sum / weight_sum if weight_sum else 0.0
    penalties = sum(
        float(policy.payload["penalty_weights"][name])
        * float(features.values[name])
        for name in PENALTY_FEATURES
        if features.availability[name] == "available"
    )
    digits = int(policy.payload["score_round_digits"])
    return ScoreResult(
        local_score=round(min(1.0, max(0.0, positive - penalties)), digits),
        positive_score=round(positive, digits),
        penalty_score=round(penalties, digits),
        missing_positive_features=tuple(sorted(missing)),
    )


def fanout_penalty(fanout: int) -> float:
    if isinstance(fanout, bool) or not isinstance(fanout, int) or fanout < 0:
        raise LocalSearchContractError("fanout must be a non-negative integer")
    return round(
        min(1.0, max(0.0, math.log2(max(fanout, 1)) - 1.0) / 4.0),
        6,
    )


def path_support(edge_scores: Sequence[float], policy: SearchPolicy | str) -> float:
    """Combine path evidence without the depth decay of score multiplication."""

    if isinstance(policy, str):
        policy = POLICY_REGISTRY[policy]
    if not edge_scores:
        return 1.0
    scores = tuple(_finite_unit(value, "path edge score") for value in edge_scores)
    epsilon = float(policy.payload["path_epsilon"])
    geometric = math.exp(sum(math.log(max(value, epsilon)) for value in scores) / len(scores))
    mix = policy.payload["path_mix"]
    result = float(mix["geometric_mean"]) * geometric + float(mix["weakest_edge"]) * min(scores)
    return round(result, int(policy.payload["score_round_digits"]))


def frontier_priority(
    local_score: float,
    path_score: float,
    seed_prior: float,
    policy: SearchPolicy | str,
) -> float:
    if isinstance(policy, str):
        policy = POLICY_REGISTRY[policy]
    local = _finite_unit(local_score, "local_score")
    path = _finite_unit(path_score, "path_score")
    seed = _finite_unit(seed_prior, "seed_prior")
    mix = policy.payload["frontier_mix"]
    result = float(mix["local"]) * local + float(mix["path"]) * path + float(mix["seed"]) * seed
    return round(result, int(policy.payload["score_round_digits"]))


@dataclass(frozen=True)
class FrontierItem:
    node_id: str
    incoming_edge_id: str
    depth: int
    seed_id: str
    seed_rank: int
    seed_prior: float
    local_score: float
    path_score: float
    frontier_priority: float
    support_scores: Tuple[float, ...] = ()
    source_group: str = ""

    def __post_init__(self) -> None:
        if not self.node_id or not self.seed_id:
            raise LocalSearchContractError("frontier node_id and seed_id must be non-empty")
        _non_negative_int(self.depth, "frontier.depth")
        _non_negative_int(self.seed_rank, "frontier.seed_rank")
        for name in ("seed_prior", "local_score", "path_score", "frontier_priority"):
            _finite_unit(getattr(self, name), f"frontier.{name}")
        for value in self.support_scores:
            _finite_unit(value, "frontier.support_scores")

    @property
    def group_id(self) -> str:
        return self.source_group or self.seed_id


@dataclass(frozen=True)
class FrontierSelection:
    item: FrontierItem
    lane: str


class FrontierScheduler:
    """Deterministic LIFO or hybrid exploit/explore frontier."""

    def __init__(self, policy: SearchPolicy | str):
        if isinstance(policy, str):
            try:
                policy = POLICY_REGISTRY[policy]
            except KeyError as error:
                raise LocalSearchContractError(f"unknown search policy {policy!r}") from error
        self.policy = policy
        self._best: Dict[str, FrontierItem] = {}
        self._generation: Dict[str, int] = {}
        self._expanded: set[str] = set()
        self._exploit: list[tuple[Any, ...]] = []
        self._explore: list[tuple[Any, ...]] = []
        self._stack: list[tuple[str, int]] = []
        self._group_expansions: Dict[str, int] = {}
        self.expansion_count = 0

    def __len__(self) -> int:
        return sum(node_id not in self._expanded for node_id in self._best)

    @staticmethod
    def _better(new: FrontierItem, old: FrontierItem) -> bool:
        return (
            -new.frontier_priority,
            new.depth,
            new.seed_rank,
            new.node_id,
            new.incoming_edge_id,
        ) < (
            -old.frontier_priority,
            old.depth,
            old.seed_rank,
            old.node_id,
            old.incoming_edge_id,
        )

    def push(self, item: FrontierItem) -> bool:
        if item.node_id in self._expanded:
            return False
        current = self._best.get(item.node_id)
        if current is not None and not self._better(item, current):
            return False
        generation = self._generation.get(item.node_id, 0) + 1
        self._generation[item.node_id] = generation
        self._best[item.node_id] = item
        if self.policy.payload["scheduler_kind"] == "legacy_lifo_dfs":
            self._stack.append((item.node_id, generation))
            return True
        heapq.heappush(
            self._exploit,
            (-item.frontier_priority, item.depth, item.seed_rank, item.node_id, item.incoming_edge_id, generation),
        )
        heapq.heappush(
            self._explore,
            (item.depth, self._group_expansions.get(item.group_id, 0), -item.frontier_priority, item.seed_rank, item.node_id, item.incoming_edge_id, generation),
        )
        return True

    def _valid(self, node_id: str, generation: int) -> Optional[FrontierItem]:
        if node_id in self._expanded or self._generation.get(node_id) != generation:
            return None
        return self._best.get(node_id)

    def _pop_stack(self) -> Optional[FrontierItem]:
        while self._stack:
            node_id, generation = self._stack.pop()
            item = self._valid(node_id, generation)
            if item is not None:
                return item
        return None

    def _pop_exploit(self) -> Optional[FrontierItem]:
        while self._exploit:
            *_key, node_id, _edge_id, generation = heapq.heappop(self._exploit)
            item = self._valid(node_id, generation)
            if item is not None:
                return item
        return None

    def _pop_explore(self) -> Optional[FrontierItem]:
        while self._explore:
            depth, group_count, _priority, seed_rank, node_id, edge_id, generation = heapq.heappop(self._explore)
            item = self._valid(node_id, generation)
            if item is None:
                continue
            current_count = self._group_expansions.get(item.group_id, 0)
            if group_count != current_count:
                heapq.heappush(
                    self._explore,
                    (depth, current_count, -item.frontier_priority, seed_rank, node_id, edge_id, generation),
                )
                continue
            return item
        return None

    def pop(self) -> Optional[FrontierSelection]:
        if self.policy.payload["scheduler_kind"] == "legacy_lifo_dfs":
            item = self._pop_stack()
            lane = "legacy"
        else:
            period = int(self.policy.payload["exploration_period"])
            explore = (self.expansion_count + 1) % period == 0
            if explore:
                item = self._pop_explore() or self._pop_exploit()
                lane = "explore" if item is not None else "exploit"
            else:
                item = self._pop_exploit() or self._pop_explore()
                lane = "exploit" if item is not None else "explore"
        if item is None:
            return None
        self._expanded.add(item.node_id)
        self._group_expansions[item.group_id] = self._group_expansions.get(item.group_id, 0) + 1
        self.expansion_count += 1
        return FrontierSelection(item, lane)


def validate_search_summary(
    row: Mapping[str, Any], *, expected_policy: Optional[SearchPolicyIdentity] = None
) -> Dict[str, Any]:
    """Validate the bounded, path-free summary included in graph identity."""

    expected_keys = {
        "schema_version",
        "policy_id",
        "feature_schema",
        "policy_sha256",
        "termination_reason",
        "seed_count",
        "expanded_nodes",
        "candidate_evaluations",
        "intervention_evaluations",
        "admitted_nodes",
        "admitted_edges",
        "exploit_expansions",
        "explore_expansions",
        "frontier_remaining",
        "frontier_sha256",
        "unevaluated_candidate_count",
        "unevaluated_sha256",
        "unevaluated_intervention_count",
        "unevaluated_intervention_sha256",
        "contribution_status_counts",
        "contribution_method_counts",
        "feature_missing_counts",
        "rejection_counts",
        "score_summary",
    }
    if not isinstance(row, Mapping):
        raise LocalSearchContractError("search_summary must be an object")
    _exact_keys(row, expected_keys, "search_summary")
    if row["schema_version"] != SEARCH_SUMMARY_SCHEMA:
        raise LocalSearchContractError(
            f"search_summary.schema_version must be {SEARCH_SUMMARY_SCHEMA}"
        )
    identity = SearchPolicyIdentity.from_dict(
        {
            "policy_id": row["policy_id"],
            "feature_schema": row["feature_schema"],
            "policy_sha256": row["policy_sha256"],
        }
    )
    if expected_policy is not None and identity != expected_policy:
        raise LocalSearchContractError("search_summary policy does not match request")
    if row["termination_reason"] not in TERMINATION_REASONS:
        raise LocalSearchContractError("search_summary.termination_reason is invalid")
    count_fields = expected_keys - {
        "schema_version",
        "policy_id",
        "feature_schema",
        "policy_sha256",
        "termination_reason",
        "frontier_sha256",
        "unevaluated_sha256",
        "unevaluated_intervention_sha256",
        "contribution_status_counts",
        "contribution_method_counts",
        "feature_missing_counts",
        "rejection_counts",
        "score_summary",
    }
    for name in count_fields:
        _non_negative_int(row[name], f"search_summary.{name}")
    if row["seed_count"] == 0:
        raise LocalSearchContractError("search_summary.seed_count must be positive")
    for name in ("frontier_sha256", "unevaluated_sha256", "unevaluated_intervention_sha256"):
        value = row[name]
        if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
            raise LocalSearchContractError(f"search_summary.{name} must be a lowercase SHA-256")
    status_counts = row["contribution_status_counts"]
    if not isinstance(status_counts, Mapping):
        raise LocalSearchContractError("contribution_status_counts must be an object")
    _exact_keys(status_counts, CONTRIBUTION_STATUSES, "contribution_status_counts")
    for name, value in status_counts.items():
        _non_negative_int(value, f"contribution_status_counts.{name}")
    for field in ("contribution_method_counts", "feature_missing_counts", "rejection_counts"):
        counts = row[field]
        if not isinstance(counts, Mapping) or any(
            not isinstance(key, str) or not key or _non_negative_int(value, field) < 0
            for key, value in counts.items()
        ):
            raise LocalSearchContractError(f"search_summary.{field} must contain named counts")
    scores = row["score_summary"]
    if not isinstance(scores, Mapping):
        raise LocalSearchContractError("search_summary.score_summary must be an object")
    _exact_keys(scores, {"min", "max", "mean"}, "search_summary.score_summary")
    parsed_scores = {name: _finite_unit(value, f"score_summary.{name}") for name, value in scores.items()}
    if not parsed_scores["min"] <= parsed_scores["mean"] <= parsed_scores["max"]:
        raise LocalSearchContractError("search_summary score ordering is invalid")
    return dict(row)
