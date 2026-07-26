"""Diagnostic CLI that resolves heuristics before invoking the V2 engine."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Sequence

from .auto_detect import (
    detect_assertion_trigger_cycle,
    detect_clock_signal,
    extract_assertion_from_filename,
)
from .contracts import make_request_v2
from .engine import _build_diagnostic_graph_v2, build_causal_graph_v2
from .identity import canonical_json_bytes, sha256_file


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a deterministic V2 causal graph from FST and RTL"
    )
    parser.add_argument("--fst", "-f", required=True)
    parser.add_argument("--verilog", "-v", required=True, nargs="+")
    parser.add_argument("--clock", "-c")
    parser.add_argument("--endpoint", "-e")
    parser.add_argument("--cycle", "-n", type=int)
    parser.add_argument("--output", "-o", default="result")
    parser.add_argument("--max-depth", "-d", type=int, default=12)
    parser.add_argument("--max-nodes", "-m", type=int, default=120)
    parser.add_argument("--random-seed", type=int, default=0)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    trace_path = Path(args.fst).resolve()
    rtl_paths = [Path(path).resolve() for path in args.verilog]
    heuristic_fields = []
    clock = args.clock
    if clock is None:
        clock = detect_clock_signal(str(trace_path))
        heuristic_fields.append("clock")
    endpoint = args.endpoint
    if endpoint is None:
        endpoint = extract_assertion_from_filename(str(trace_path))
        heuristic_fields.append("endpoint")
    cycle = args.cycle
    if cycle is None:
        cycle = detect_assertion_trigger_cycle(str(trace_path), endpoint, clock)
        heuristic_fields.append("cycle")

    trace_hash, trace_bytes = sha256_file(trace_path)
    rtl_files = []
    for index, path in enumerate(rtl_paths, 1):
        digest, size = sha256_file(path)
        rtl_files.append(
            {
                "artifact_id": f"rtl_{index:04d}",
                "path": str(path),
                "sha256": digest,
                "bytes": size,
            }
        )
    request = make_request_v2(
        trace={
            "path": str(trace_path),
            "format": "fst",
            "sha256": trace_hash,
            "bytes": trace_bytes,
        },
        rtl_files=rtl_files,
        clock_signal=clock,
        endpoint_signal=endpoint,
        endpoint_cycle=cycle,
        max_depth=args.max_depth,
        max_nodes=args.max_nodes,
        random_seed=args.random_seed,
        strict=not heuristic_fields,
    )
    graph = (
        _build_diagnostic_graph_v2(request)
        if heuristic_fields
        else build_causal_graph_v2(request)
    )
    if heuristic_fields:
        graph["status"] = "incomplete"
        graph["diagnostics"].append(
            {
                "code": "diagnostic_heuristic_input",
                "severity": "warning",
                "breaks_complete": True,
                "message": (
                    "mode=diagnostic_heuristic; auto-detected fields have no "
                    f"SpecFlow identity authority: {','.join(heuristic_fields)}"
                ),
            }
        )
        graph["diagnostics"].sort(
            key=lambda row: (
                row["code"],
                row.get("artifact_id") or "",
                row["message"],
            )
        )
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "causal_graph.json").write_bytes(
        canonical_json_bytes(graph) + b"\n"
    )
    print(
        json.dumps(
            {
                "mode": (
                    "diagnostic_heuristic" if heuristic_fields else "production_exact"
                ),
                "graph_id": graph["graph_id"],
                "status": graph["status"],
                "output": str(output_dir / "causal_graph.json"),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
