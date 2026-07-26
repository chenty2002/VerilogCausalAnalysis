#!/usr/bin/env python3
"""Compatibility executable for the V2 diagnostic CLI."""

from verilog_causal_analysis.cli import main


if __name__ == "__main__":
    raise SystemExit(main())
