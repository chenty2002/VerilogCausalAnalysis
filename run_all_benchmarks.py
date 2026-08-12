#!/usr/bin/env python3
"""
Batch run causal analysis on all benchmarks in tests/ directory.
For each benchmark, pick one FST waveform and generate causal graph.
"""

import os
import sys
import glob

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Benchmark configurations: (benchmark_name, fst_file_pattern, verilog_file)
# We'll pick the first available FST for each benchmark
BENCHMARKS = [
    ("abp", "tests/abp/*.fst", "tests/abp/TestTop.sv"),
    ("counter", "tests/counter/*.fst", "tests/counter/TestTop.sv"),
    ("crc", "tests/crc/*.fst", "tests/crc/TestTop.sv"),
    ("gcd", "tests/gcd/*.fst", "tests/gcd/TestTop.sv"),
    ("gigamax", "tests/gigamax/*.fst", "tests/gigamax/TestTop.sv"),
    ("gray", "tests/gray/*.fst", "tests/gray/TestTop.sv"),
    ("itc99_b01", "tests/itc99_b01/*.fst", "tests/itc99_b01/TestTop.sv"),
    ("itc99_b02", "tests/itc99_b02/*.fst", "tests/itc99_b02/TestTop.sv"),
    ("lock", "tests/lock/*.fst", "tests/lock/TestTop.sv"),
    ("philo4", "tests/philo4/*.fst", "tests/philo4/TestTop.sv"),
    ("reset", "tests/reset/*.fst", "tests/reset/TestTop.sv"),
    ("short", "tests/short/*.fst", "tests/short/TestTop.sv"),
    ("swap", "tests/swap/*.fst", "tests/swap/TestTop.sv"),
]


def run_benchmark(name: str, fst_pattern: str, verilog_path: str, output_dir: str):
    """Run analysis for a single benchmark."""
    
    # Find first available FST file
    fst_files = sorted(glob.glob(fst_pattern))
    if not fst_files:
        print(f"  [!] No FST files found for {name}, skipping...")
        return False, "No FST files"
    
    fst_path = fst_files[0]
    
    # Check Verilog exists
    if not os.path.exists(verilog_path):
        print(f"  [!] Verilog file not found: {verilog_path}")
        return False, "Verilog not found"
    
    print(f"  [*] FST: {os.path.basename(fst_path)}")
    print(f"  [*] Verilog: {verilog_path}")
    
    # Keep the legacy harness in-process. The diagnostic CLI performs one
    # bounded heuristic resolution pass and writes the same V2 artifact.
    from verilog_causal_analysis.cli import main as cli_main

    argv = [
        "--fst", fst_path,
        "--verilog", verilog_path,
        "--output", output_dir,
        "--quiet"
    ]
    try:
        return_code = cli_main(argv)
        if return_code == 0:
            print("  [+] Success")
            return True, None
        error = f"diagnostic CLI returned {return_code}"
        print(f"  [!] Failed: {error}")
        return False, error
    except Exception as error:
        print(f"  [!] Exception: {error}")
        return False, str(error)


def main():
    print("=" * 60)
    print("Batch Causal Analysis for All Benchmarks")
    print("=" * 60)
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(base_dir)
    
    results = {}
    
    for name, fst_pattern, verilog in BENCHMARKS:
        print(f"\n[{name}]")
        output_dir = f"results/{name}"
        os.makedirs(output_dir, exist_ok=True)
        
        success, error = run_benchmark(name, fst_pattern, verilog, output_dir)
        results[name] = {"success": success, "error": error}
    
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    
    success_count = sum(1 for r in results.values() if r["success"])
    print(f"Successful: {success_count}/{len(BENCHMARKS)}")
    
    for name, result in results.items():
        status = "✓" if result["success"] else "✗"
        print(f"  {status} {name}")
    
    return 0 if success_count == len(BENCHMARKS) else 1


if __name__ == "__main__":
    sys.exit(main())
