from pathlib import Path

import pytest

from verilog_causal_analysis import (
    make_structural_request,
    policy_identity,
    sha256_file,
)


ROOT = Path(__file__).resolve().parents[1]


def request_for(
    family: str,
    fst_name: str,
    *,
    clock: str,
    cycle: int,
    max_depth: int = 12,
    max_nodes: int = 120,
):
    directory = ROOT / "tests" / family
    fst = (directory / fst_name).resolve()
    rtl = (directory / "TestTop.sv").resolve()
    trace_hash, trace_bytes = sha256_file(fst)
    rtl_hash, rtl_bytes = sha256_file(rtl)
    return make_structural_request(
        trace={
            "path": str(fst),
            "format": "fst",
            "sha256": trace_hash,
            "bytes": trace_bytes,
        },
        rtl_files=[
            {
                "artifact_id": "rtl_0001",
                "path": str(rtl),
                "sha256": rtl_hash,
                "bytes": rtl_bytes,
            }
        ],
        clock_signal=clock,
        endpoint_signal=fst.stem,
        endpoint_cycle=cycle,
        search_policy=policy_identity().to_dict(),
        max_depth=max_depth,
        max_nodes=max_nodes,
        random_seed=0,
    )


@pytest.fixture
def counter_request():
    return request_for(
        "counter",
        "Counter.bit1.value_should_toggle_when_carry_in_is_true2C_stay_stable_otherwise.fst",
        clock="Counter.clock",
        cycle=1,
    )
