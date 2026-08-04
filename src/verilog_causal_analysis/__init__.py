"""Public API for structural baselines and the current Chisel-aware analysis."""

from .contracts import (
    CHISEL_PROFILE,
    GRAPH_SCHEMA,
    REQUEST_SCHEMA,
    CausalAnalysisRequest,
    ContractError,
    make_request,
    validate_graph,
)
from .structural_contract import (
    EVIDENCE_STRENGTHS,
    GRAPH_STATUSES,
    IDENTITY_STRENGTHS,
    STRUCTURAL_GRAPH_SCHEMA,
    STRUCTURAL_REQUEST_SCHEMA,
    StructuralCausalRequest,
    StructuralContractError,
    make_structural_request,
    validate_structural_graph,
)
from .source_ranking import build_source_ranking
from .local_search import (
    FEATURE_SCHEMA,
    POLICY_IDS,
    POLICY_REGISTRY,
    SEARCH_SUMMARY_SCHEMA,
    LocalSearchContractError,
    FrontierItem,
    FrontierScheduler,
    FrontierSelection,
    ScoreFeatures,
    ScoreResult,
    SearchPolicy,
    SearchPolicyIdentity,
    make_search_summary,
    fanout_penalty,
    frontier_priority,
    path_support,
    policy_identity,
    score_features,
    validate_search_summary,
)
from .contribution import (
    CONTRIBUTION_SCHEMA,
    LEGACY_CONTRIBUTION_SCHEMA,
    ContributionContractError,
    ContributionEvidence,
    ContributionEffects,
    ContributionStatus,
    ContributionWorkCounter,
    EvidenceDomain,
    EvidenceMethod,
    Fidelity,
    InterventionSummary,
    LegacyContributionEvidence,
    RelevantTarget,
    RoutedContribution,
    adapt_legacy_contribution,
    contribution_cache_key,
    contribution_edge_fields,
    evaluate_interventions,
    fidelity_gate,
    generate_interventions,
    route_contribution,
    structural_evidence,
    toggle_evidence,
)
from .identity import (
    ANALYZER_REVISION,
    HDLCONVERTOR_REVISION,
    canonical_json_bytes,
    canonical_sha256,
    sha256_file,
)
from .endpoint_projection import (
    ASSERTION_PROJECTION_SCHEMA,
    AssertionEndpointProjection,
    EndpointProjectionError,
)
from .instance_graph import (
    INSTANCE_GRAPH_SCHEMA,
    InstanceGraph,
    InstanceGraphError,
    InstanceNode,
    PortBinding,
)
from .chisel_semantics import (
    NORMALIZED_DESIGN_SCHEMA,
    SemanticQueryError,
    get_raw_members,
    get_register_transition,
)
from .temporal_semantics import (
    TEMPORAL_FEATURE,
    build_transition_intervals,
    get_semantic_paths,
)
from .waitfor_graph import (
    PROTOCOL_ADAPTER_SCHEMA,
    WAITFOR_FEATURE,
    WaitForError,
    get_waitfor_component,
    make_protocol_adapter,
    validate_protocol_adapter,
)
from .provenance import (
    SOURCE_ANNOTATION_SCHEMA,
    SOURCE_PROVENANCE_FEATURE,
    ProvenanceError,
)
from .semantic_query import (
    SemanticGraphQueryError,
    get_handshake_timeline,
    get_interval_evidence,
    get_pipeline_occupancy,
    get_semantic_overview,
)

__version__ = "2.3.0"


def build_structural_graph(request):
    """Build the structural baseline graph."""
    from .structural_engine import build_structural_graph as build

    return build(request)


def prepare_structural_analysis(request):
    """Prepare shared state for structural baseline queries."""
    from .structural_engine import prepare_structural_analysis as prepare

    return prepare(request)


def build_causal_graph(request, *, top_module=None):
    """Build the current Chisel-aware causal graph."""
    from .engine import build_causal_graph as build

    return build(request, top_module=top_module)


def prepare_causal_session(request, *, top_module=None):
    """Prepare shared state for Chisel-aware causal queries."""
    from .engine import prepare_causal_session as prepare

    return prepare(request, top_module=top_module)


__all__ = [
    "ANALYZER_REVISION",
    "ASSERTION_PROJECTION_SCHEMA",
    "AssertionEndpointProjection",
    "CHISEL_PROFILE",
    "CONTRIBUTION_SCHEMA",
    "CausalAnalysisRequest",
    "ContributionContractError",
    "ContributionEffects",
    "ContributionEvidence",
    "ContributionStatus",
    "ContributionWorkCounter",
    "ContractError",
    "EVIDENCE_STRENGTHS",
    "EvidenceDomain",
    "EvidenceMethod",
    "EndpointProjectionError",
    "GRAPH_SCHEMA",
    "GRAPH_STATUSES",
    "HDLCONVERTOR_REVISION",
    "IDENTITY_STRENGTHS",
    "INSTANCE_GRAPH_SCHEMA",
    "InstanceGraph",
    "InstanceGraphError",
    "InstanceNode",
    "NORMALIZED_DESIGN_SCHEMA",
    "FEATURE_SCHEMA",
    "Fidelity",
    "InterventionSummary",
    "LEGACY_CONTRIBUTION_SCHEMA",
    "LegacyContributionEvidence",
    "POLICY_IDS",
    "POLICY_REGISTRY",
    "SEARCH_SUMMARY_SCHEMA",
    "LocalSearchContractError",
    "FrontierItem",
    "FrontierScheduler",
    "FrontierSelection",
    "PROTOCOL_ADAPTER_SCHEMA",
    "PortBinding",
    "ProvenanceError",
    "REQUEST_SCHEMA",
    "RelevantTarget",
    "RoutedContribution",
    "SOURCE_ANNOTATION_SCHEMA",
    "SOURCE_PROVENANCE_FEATURE",
    "STRUCTURAL_GRAPH_SCHEMA",
    "STRUCTURAL_REQUEST_SCHEMA",
    "SemanticQueryError",
    "SemanticGraphQueryError",
    "StructuralCausalRequest",
    "StructuralContractError",
    "ScoreFeatures",
    "ScoreResult",
    "SearchPolicy",
    "SearchPolicyIdentity",
    "TEMPORAL_FEATURE",
    "WAITFOR_FEATURE",
    "WaitForError",
    "build_causal_graph",
    "build_structural_graph",
    "build_source_ranking",
    "build_transition_intervals",
    "adapt_legacy_contribution",
    "canonical_json_bytes",
    "canonical_sha256",
    "contribution_cache_key",
    "contribution_edge_fields",
    "evaluate_interventions",
    "fidelity_gate",
    "generate_interventions",
    "get_raw_members",
    "get_register_transition",
    "get_handshake_timeline",
    "get_interval_evidence",
    "get_pipeline_occupancy",
    "get_semantic_overview",
    "get_semantic_paths",
    "get_waitfor_component",
    "make_protocol_adapter",
    "make_request",
    "make_search_summary",
    "fanout_penalty",
    "frontier_priority",
    "path_support",
    "make_structural_request",
    "prepare_causal_session",
    "prepare_structural_analysis",
    "policy_identity",
    "score_features",
    "route_contribution",
    "sha256_file",
    "structural_evidence",
    "toggle_evidence",
    "validate_graph",
    "validate_protocol_adapter",
    "validate_structural_graph",
    "validate_search_summary",
    "__version__",
]
