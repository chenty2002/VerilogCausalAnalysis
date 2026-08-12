import json

import pytest

from verilog_causal_analysis.endpoint_projection import (
    EndpointProjectionError,
    load_assertion_projection,
)
from verilog_causal_analysis.identity import sha256_file


def _projection_row():
    return {
        "schema_version": "assertion_endpoint_projection",
        "endpoint_signal": "Top.left._assert_1",
        "endpoint_cycle": 10,
        "clock_signal": "Top.clock",
        "predicate_members": ["Top.left.timer"],
        "rtl_set_sha256": "1" * 64,
        "trace_sha256": "2" * 64,
    }


def test_projection_is_hash_bound_and_exact(tmp_path):
    path = tmp_path / "projection.json"
    path.write_text(json.dumps(_projection_row(), sort_keys=True))
    digest, size = sha256_file(path)
    projection = load_assertion_projection(
        str(path),
        artifact_id="assertion_projection_0001",
        sha256=digest,
        bytes=size,
        endpoint_signal="Top.left._assert_1",
        endpoint_cycle=10,
        clock_signal="Top.clock",
        rtl_set_sha256="1" * 64,
        trace_sha256="2" * 64,
    )
    assert projection.predicate_members == ("Top.left.timer",)
    assert projection.projection_id.startswith("vcp_")


def test_projection_rejects_stale_endpoint_and_bytes(tmp_path):
    path = tmp_path / "projection.json"
    path.write_text(json.dumps(_projection_row(), sort_keys=True))
    digest, size = sha256_file(path)
    with pytest.raises(EndpointProjectionError, match="endpoint_cycle"):
        load_assertion_projection(
            str(path),
            artifact_id="assertion_projection_0001",
            sha256=digest,
            bytes=size,
            endpoint_signal="Top.left._assert_1",
            endpoint_cycle=11,
            clock_signal="Top.clock",
            rtl_set_sha256="1" * 64,
            trace_sha256="2" * 64,
        )
    with pytest.raises(EndpointProjectionError, match="bytes or SHA"):
        load_assertion_projection(
            str(path),
            artifact_id="assertion_projection_0001",
            sha256="0" * 64,
            bytes=size,
            endpoint_signal="Top.left._assert_1",
            endpoint_cycle=10,
            clock_signal="Top.clock",
            rtl_set_sha256="1" * 64,
            trace_sha256="2" * 64,
        )
