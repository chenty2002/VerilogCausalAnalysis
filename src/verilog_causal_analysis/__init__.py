"""Public production surface for Verilog Causal Analysis V2.

Auto-detection remains available only to :mod:`verilog_causal_analysis.cli`
through the private diagnostic module and is intentionally not re-exported.
"""

from .contracts import (
    CausalAnalysisRequestV2,
    ContractError,
    EVIDENCE_STRENGTHS,
    GRAPH_SCHEMA,
    GRAPH_STATUSES,
    IDENTITY_STRENGTHS,
    REQUEST_SCHEMA,
    make_request_v2,
    validate_graph_v2,
)
from .identity import (
    ANALYZER_REVISION,
    HDLCONVERTOR_REVISION,
    canonical_json_bytes,
    canonical_sha256,
    sha256_file,
)
from .contracts_v3 import (
    CHISEL_PROFILE_VERSION,
    CausalAnalysisRequestV3,
    ContractV3Error,
    REQUEST_SCHEMA_V3,
    SEMANTIC_GRAPH_SCHEMA,
    make_request_v3,
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
from .query import (
    GraphQueryView,
    QueryError,
    expand_predecessors,
    get_edge_evidence,
    get_overview,
    get_query_cache_statistics,
    get_ranked_paths,
    prepare_query_view,
)

__version__ = "2.3.0"


def build_causal_graph_v2(request):
    """Lazily load the parser/waveform backend for a production build."""
    from .engine import build_causal_graph_v2 as _build

    return _build(request)


def prepare_causal_analysis(request):
    """Prepare verified RTL/waveform state for multiple V2 graph builds."""
    from .engine import prepare_causal_analysis as _prepare

    return _prepare(request)


def build_causal_graph_v3(request, *, top_module=None):
    """Build one opt-in C0-C2 Chisel semantic graph."""
    from .engine_v3 import build_causal_graph_v3 as _build

    return _build(request, top_module=top_module)


def prepare_causal_session_v3(request, *, top_module=None):
    """Prepare a reusable C0-C2 Chisel semantic session."""
    from .engine_v3 import prepare_causal_session_v3 as _prepare

    return _prepare(request, top_module=top_module)


__all__ = [
    "ANALYZER_REVISION",
    "ASSERTION_PROJECTION_SCHEMA",
    "AssertionEndpointProjection",
    "CHISEL_PROFILE_VERSION",
    "CausalAnalysisRequestV2",
    "CausalAnalysisRequestV3",
    "ContractError",
    "ContractV3Error",
    "EVIDENCE_STRENGTHS",
    "GRAPH_SCHEMA",
    "GRAPH_STATUSES",
    "GraphQueryView",
    "HDLCONVERTOR_REVISION",
    "IDENTITY_STRENGTHS",
    "INSTANCE_GRAPH_SCHEMA",
    "NORMALIZED_DESIGN_SCHEMA",
    "InstanceGraph",
    "InstanceGraphError",
    "InstanceNode",
    "EndpointProjectionError",
    "PortBinding",
    "QueryError",
    "REQUEST_SCHEMA",
    "REQUEST_SCHEMA_V3",
    "SEMANTIC_GRAPH_SCHEMA",
    "SemanticQueryError",
    "build_causal_graph_v2",
    "build_causal_graph_v3",
    "canonical_json_bytes",
    "canonical_sha256",
    "expand_predecessors",
    "get_edge_evidence",
    "get_overview",
    "get_query_cache_statistics",
    "get_ranked_paths",
    "get_raw_members",
    "get_register_transition",
    "make_request_v2",
    "make_request_v3",
    "prepare_causal_analysis",
    "prepare_causal_session_v3",
    "prepare_query_view",
    "sha256_file",
    "validate_graph_v2",
    "__version__",
]
