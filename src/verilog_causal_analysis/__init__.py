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
from .query import (
    QueryError,
    expand_predecessors,
    get_edge_evidence,
    get_overview,
    get_ranked_paths,
)

__version__ = "2.0.0"


def build_causal_graph_v2(request):
    """Lazily load the parser/waveform backend for a production build."""
    from .engine import build_causal_graph_v2 as _build

    return _build(request)

__all__ = [
    "ANALYZER_REVISION",
    "CausalAnalysisRequestV2",
    "ContractError",
    "EVIDENCE_STRENGTHS",
    "GRAPH_SCHEMA",
    "GRAPH_STATUSES",
    "HDLCONVERTOR_REVISION",
    "IDENTITY_STRENGTHS",
    "QueryError",
    "REQUEST_SCHEMA",
    "build_causal_graph_v2",
    "canonical_json_bytes",
    "canonical_sha256",
    "expand_predecessors",
    "get_edge_evidence",
    "get_overview",
    "get_ranked_paths",
    "make_request_v2",
    "sha256_file",
    "validate_graph_v2",
    "__version__",
]
