import pytest

from verilog_causal_analysis import ContractError, make_request, policy_identity


BOUNDS = {
    "max_signal_depth": 8,
    "max_signal_nodes": 32,
    "max_expanded_nodes": 32,
    "max_candidate_evaluations": 256,
    "max_intervention_evaluations": 1024,
    "max_semantic_nodes": 32,
    "max_edges": 64,
    "max_seed_count": 4,
    "max_intervals_per_signal": 16,
    "max_temporal_samples": 256,
    "max_waitfor_nodes": 32,
    "max_waitfor_edges": 64,
    "max_scc_candidates": 4,
}


def _request(**changes):
    values = {
        "trace": {
            "artifact_id": "trace_0001",
            "path": "/tmp/native.fst",
            "format": "fst",
            "sha256": "0" * 64,
            "bytes": 0,
        },
        "rtl_files": [
            {
                "artifact_id": "rtl_0001",
                "path": "/tmp/native.sv",
                "sha256": "1" * 64,
                "bytes": 1,
            }
        ],
        "semantic_profile": {
            "name": "verilog",
            "version": "verilog-semantic-profile",
            "features": [
                "temporal_interval",
                "instance_graph",
                "register_transition",
            ],
        },
        "clock": {"signal": "Top.clk", "edge": "rising"},
        "endpoint": {"signal": "Top.out", "cycle": 2, "projection": None},
        "semantic_inputs": [],
        "search_policy": policy_identity().to_dict(),
        "bounds": BOUNDS,
        "random_seed": 0,
        "strict": True,
    }
    values.update(changes)
    return make_request(**values)


def test_verilog_profile_accepts_only_generic_features():
    request = _request()
    assert request.semantic_profile.features == (
        "instance_graph",
        "register_transition",
        "temporal_interval",
    )

    profile = dict(request.to_dict()["semantic_profile"])
    profile["features"] = ["instance_graph", "source_provenance"]
    with pytest.raises(ContractError, match="unsupported"):
        _request(semantic_profile=profile)


def test_verilog_profile_rejects_chisel_semantic_inputs():
    annotation = {
        "artifact_id": "annotations",
        "kind": "chisel_source_annotations",
        "path": "/tmp/annotations.json",
        "sha256": "2" * 64,
        "bytes": 1,
    }
    with pytest.raises(ContractError, match="rejects Chisel"):
        _request(semantic_inputs=[annotation])
