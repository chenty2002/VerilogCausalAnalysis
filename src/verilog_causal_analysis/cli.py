"""Exact-input structural graph CLI."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Sequence

from .identity import canonical_json_bytes, sha256_file
from .structural_contract import make_structural_request
from .structural_engine import build_structural_graph


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a structural causal graph from exact FST and RTL inputs"
    )
    parser.add_argument("--fst", "-f", required=True)
    parser.add_argument("--verilog", required=True, nargs="+")
    parser.add_argument("--clock", required=True)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--cycle", type=int, required=True)
    parser.add_argument("--output", "-o", required=True)
    parser.add_argument("--max-depth", type=int, default=12)
    parser.add_argument("--max-nodes", type=int, default=120)
    parser.add_argument("--random-seed", type=int, default=0)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    trace_path = Path(args.fst).resolve()
    trace_hash, trace_bytes = sha256_file(trace_path)
    rtl_files = []
    for index, value in enumerate(args.verilog, 1):
        path = Path(value).resolve()
        digest, size = sha256_file(path)
        rtl_files.append(
            {
                "artifact_id": f"rtl_{index:04d}",
                "path": str(path),
                "sha256": digest,
                "bytes": size,
            }
        )
    graph = build_structural_graph(
        make_structural_request(
            trace={
                "path": str(trace_path),
                "format": "fst",
                "sha256": trace_hash,
                "bytes": trace_bytes,
            },
            rtl_files=rtl_files,
            clock_signal=args.clock,
            endpoint_signal=args.endpoint,
            endpoint_cycle=args.cycle,
            max_depth=args.max_depth,
            max_nodes=args.max_nodes,
            random_seed=args.random_seed,
        )
    )
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(canonical_json_bytes(graph) + b"\n")
    print(
        json.dumps(
            {
                "graph_id": graph["graph_id"],
                "status": graph["status"],
                "output": str(output),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
