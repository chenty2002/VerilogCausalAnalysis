"""
Auto-detection utilities for causal analysis parameters.

Provides automatic detection of:
- Endpoint signal from FST filename
- Assertion trigger cycle from waveform
- Clock signal from top-level module hierarchy
- SVA assertion labels from Verilog source files
"""

import os
import re
from bisect import bisect_right
from typing import List, Optional, Set, Tuple
import pylibfst


def _clock_from_signal_names(signal_names: List[str]) -> str:
    candidates = []
    for signal_name in signal_names:
        parts = signal_name.split(".")
        if len(parts) < 2:
            continue
        base = re.sub(r"\s*\[.*\]$", "", parts[-1].lower())
        if base in {"clock", "clk"}:
            candidates.append((len(parts), signal_name))
    if not candidates:
        raise RuntimeError(
            "Could not find clock signal (clock/clk) in waveform hierarchy. "
            "Please specify --clock manually."
        )
    return min(candidates)[1]


def resolve_diagnostic_inputs(
    fst_path: str,
    *,
    clock_signal: Optional[str],
    endpoint_signal: Optional[str],
    endpoint_cycle: Optional[int],
) -> Tuple[str, str, int]:
    """Resolve missing diagnostic fields with at most one FST open.

    This helper remains diagnostic-only. Its inferred values do not gain V2
    production identity authority.
    """
    endpoint = endpoint_signal or extract_assertion_from_filename(fst_path)
    if clock_signal is not None and endpoint_cycle is not None:
        return clock_signal, endpoint, endpoint_cycle

    fst = pylibfst.lib.fstReaderOpen(fst_path.encode("UTF-8"))
    if fst == pylibfst.ffi.NULL:
        raise RuntimeError(f"Failed to open FST file: {fst_path}")
    try:
        _, signals = pylibfst.get_scopes_signals2(fst)
        names = sorted(signals.by_name)
        if not names:
            raise RuntimeError("No signals found in waveform")
        clock = clock_signal or _clock_from_signal_names(names)
        clock_row = signals.by_name.get(clock)
        if clock_row is None:
            raise RuntimeError(f"Clock signal not found in waveform: {clock}")
        endpoint_row = signals.by_name.get(endpoint)
        if endpoint_cycle is None and endpoint_row is None:
            raise RuntimeError(
                f"Assertion signal not found in waveform: {endpoint}"
            )

        pylibfst.lib.fstReaderClrFacProcessMaskAll(fst)
        pylibfst.lib.fstReaderSetFacProcessMask(fst, clock_row.handle)
        if endpoint_cycle is None and endpoint_row is not None:
            pylibfst.lib.fstReaderSetFacProcessMask(
                fst, endpoint_row.handle
            )
        timestamps = pylibfst.lib.fstReaderGetTimestamps(fst)
        if timestamps.nvals == 0:
            pylibfst.lib.fstReaderFreeTimestamps(timestamps)
            raise RuntimeError("No timestamps found in waveform")
        buffer = pylibfst.ffi.new("char[256]")
        clock_boundaries: List[int] = []
        previous_clock = None
        previous_endpoint = None
        trigger_time = None
        for index in range(timestamps.nvals):
            time = int(timestamps.val[index])
            clock_value = pylibfst.helpers.string(
                pylibfst.lib.fstReaderGetValueFromHandleAtTime(
                    fst, time, clock_row.handle, buffer
                )
            )
            if (
                clock_value == "1"
                and previous_clock in ("0", "x", "X", None)
            ):
                clock_boundaries.append(time)
            previous_clock = clock_value
            if endpoint_cycle is None and endpoint_row is not None:
                endpoint_value = pylibfst.helpers.string(
                    pylibfst.lib.fstReaderGetValueFromHandleAtTime(
                        fst, time, endpoint_row.handle, buffer
                    )
                )
                if previous_endpoint == "1" and endpoint_value == "0":
                    trigger_time = time
                    break
                previous_endpoint = endpoint_value
        pylibfst.lib.fstReaderFreeTimestamps(timestamps)
        if endpoint_cycle is None:
            if trigger_time is None:
                raise RuntimeError(
                    f"Assertion signal '{endpoint}' did not transition "
                    "from 1 to 0. Cannot detect trigger cycle."
                )
            endpoint_cycle = max(
                0, bisect_right(clock_boundaries, trigger_time) - 1
            )
        return clock, endpoint, endpoint_cycle
    finally:
        pylibfst.lib.fstReaderClose(fst)


def extract_assertion_from_filename(fst_path: str) -> str:
    """
    Extract assertion signal name from FST filename.
    
    Args:
        fst_path: Path to FST waveform file
        
    Returns:
        Assertion signal name (basename without .fst extension)
        
    Example:
        "tests/philo4.System_should_not_deadlock_when_all_philosophers_are_hungry.fst"
        -> "philo4.System_should_not_deadlock_when_all_philosophers_are_hungry"
    """
    basename = os.path.basename(fst_path)
    # Remove .fst extension
    if basename.endswith('.fst'):
        assertion_name = basename[:-4]
    else:
        assertion_name = basename
    return assertion_name


def detect_assertion_trigger_cycle(fst_path: str, assertion_signal: str, clock_signal: str) -> int:
    """
    Detect the cycle when assertion triggered (value changed from 1 to 0).
    
    Args:
        fst_path: Path to FST waveform file
        assertion_signal: Full hierarchical assertion signal name
        clock_signal: Clock signal name for cycle alignment
        
    Returns:
        The cycle number when assertion failed
        
    Raises:
        RuntimeError: If assertion signal not found or didn't trigger
    """
    return resolve_diagnostic_inputs(
        fst_path,
        clock_signal=clock_signal,
        endpoint_signal=assertion_signal,
        endpoint_cycle=None,
    )[2]


def detect_clock_signal(fst_path: str) -> str:
    """
    Detect clock signal from the top-level module in the waveform.
    
    Looks for signals named 'clock' or 'clk' in the top-level hierarchy.
    
    Args:
        fst_path: Path to FST waveform file
        
    Returns:
        The full hierarchical clock signal name (prefers top-level clocks)
        
    Raises:
        RuntimeError: If no clock signal found
    """
    fst = pylibfst.lib.fstReaderOpen(fst_path.encode("UTF-8"))
    if fst == pylibfst.ffi.NULL:
        raise RuntimeError(f"Failed to open FST file: {fst_path}")
    
    try:
        _, signals = pylibfst.get_scopes_signals2(fst)
        return _clock_from_signal_names(sorted(signals.by_name))
        
    finally:
        pylibfst.lib.fstReaderClose(fst)


def extract_sva_assertions_from_verilog(verilog_paths: List[str]) -> List[str]:
    """
    Extract SVA assertion labels from Verilog/SystemVerilog source files.
    
    Parses files using regex to find "label: assert property" patterns.
    
    Args:
        verilog_paths: List of Verilog/SystemVerilog source file paths
        
    Returns:
        List of assertion labels found in source files
        
    Example:
        Input file contains:
            my_assertion: assert property (@(posedge clk) a |-> b);
        Returns:
            ["my_assertion"]
    """
    # SVA assertion pattern: label: assert property
    # Matches same-line and split-line Chisel/firtool output:
    #   label:
    #     assert property (@(...) ...)
    assertion_pattern = re.compile(
        r'^\s*([a-zA-Z_$][a-zA-Z0-9_$]*)\s*:\s*(?:\n\s*)?assert\s+property\b',
        re.MULTILINE
    )
    
    assertion_labels = []
    
    for vpath in verilog_paths:
        if not os.path.exists(vpath):
            continue
            
        try:
            with open(vpath, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                
            for match in assertion_pattern.finditer(content):
                label = match.group(1)
                assertion_labels.append(label)
                
        except Exception:
            # Skip files that can't be read
            continue
    
    return assertion_labels


def get_assertion_signals_from_waveform(fst_path: str, 
                                        assertion_labels: Optional[List[str]] = None) -> List[str]:
    """
    Find assertion signals in waveform that match extracted SVA labels.
    
    Args:
        fst_path: Path to FST waveform file
        assertion_labels: Optional list of SVA labels from source (for filtering)
        
    Returns:
        List of assertion signal names from waveform
    """
    fst = pylibfst.lib.fstReaderOpen(fst_path.encode("UTF-8"))
    if fst == pylibfst.ffi.NULL:
        return []
    
    try:
        _, signals = pylibfst.get_scopes_signals2(fst)
        all_sigs = list(signals.by_name.keys())
        
        # Filter for assertion-like signals
        assertion_keywords = ["assert", "fail", "error", "property", "should", "must", "valid", "check"]
        assertion_signals = []
        
        # First priority: match with SVA labels from source code
        if assertion_labels:
            label_set = set(assertion_labels)
            for sig in all_sigs:
                # Extract the signal name (last component)
                sig_name = sig.split('.')[-1]
                # Remove bit width if present
                sig_name = re.sub(r'\s*\[.*\]$', '', sig_name)
                
                if sig_name in label_set:
                    assertion_signals.append(sig)
        
        # Second priority: signals with assertion keywords
        if not assertion_signals:
            for sig in all_sigs:
                if any(kw in sig.lower() for kw in assertion_keywords):
                    assertion_signals.append(sig)
        
        return assertion_signals
        
    finally:
        pylibfst.lib.fstReaderClose(fst)


def build(fst_path: str,
          verilog_paths: List[str],
          clock_signal: Optional[str] = None,
          endpoint_signal: Optional[str] = None,
          endpoint_cycle: Optional[int] = None,
          max_depth: int = 20,
          max_nodes: int = 200,
          random_seed: Optional[int] = None,
          quiet: bool = False):
    """
    Build a CausalGraphBuilder with auto-detection of missing parameters.
    
    Auto-detects:
    - clock_signal: If None, searches for 'clock'/'clk' in top-level hierarchy
    - endpoint_signal: If None, extracts from FST filename
    - endpoint_cycle: If None, detects assertion trigger cycle from waveform
    
    Args:
        fst_path: Path to FST waveform file
        verilog_paths: List of Verilog/SystemVerilog source file paths
        clock_signal: Clock signal name (auto-detect if None)
        endpoint_signal: Endpoint signal name (auto-detect from filename if None)
        endpoint_cycle: Endpoint cycle number (auto-detect from assertion trigger if None)
        max_depth: Maximum traversal depth (default: 20)
        max_nodes: Maximum nodes in DAG (default: 200)
        random_seed: Random seed for reproducibility (default: deterministic zero)
        quiet: If True, suppress auto-detection messages
        
    Returns:
        Tuple of (CausalGraphBuilder, endpoint_signal, endpoint_cycle) with
        all parameters resolved (auto-detected or provided)
        
    Raises:
        RuntimeError: If auto-detection fails or inputs are invalid
        
    Example:
        builder, endpoint, cycle = build(
            fst_path="tests/philo4.System_should_not_deadlock.fst",
            verilog_paths=["tests/TestTop.sv"],
            quiet=False
        )
        # All parameters auto-detected from filename and waveform
    """
    from .causal_graph import CausalGraphBuilder
    
    def log(msg):
        if not quiet:
            print(msg)
    
    # Validate inputs
    if not os.path.exists(fst_path):
        raise RuntimeError(f"FST file not found: {fst_path}")
    
    for vpath in verilog_paths:
        if not os.path.exists(vpath):
            raise RuntimeError(f"Verilog file not found: {vpath}")
    
    # === Auto-detection Phase ===
    
    # 1. Auto-detect clock signal if not specified
    if clock_signal is None:
        log("[*] Auto-detecting clock signal from waveform hierarchy...")
        clock_signal = detect_clock_signal(fst_path)
        log(f"[+] Detected clock signal: {clock_signal}")
    
    # 2. Auto-detect endpoint signal from filename if not specified
    if endpoint_signal is None:
        log("[*] Auto-detecting endpoint signal from FST filename...")
        endpoint_signal = extract_assertion_from_filename(fst_path)
        log(f"[+] Extracted endpoint from filename: {endpoint_signal}")
        
        # Verify the signal exists in waveform
        fst = pylibfst.lib.fstReaderOpen(fst_path.encode("UTF-8"))
        if fst != pylibfst.ffi.NULL:
            try:
                _, signals = pylibfst.get_scopes_signals2(fst)
                if endpoint_signal not in signals.by_name:
                    raise RuntimeError(
                        f"Endpoint signal '{endpoint_signal}' (extracted from filename) "
                        f"not found in waveform. Use --endpoint to specify manually, "
                        f"or --list-signals to see available signals."
                    )
            finally:
                pylibfst.lib.fstReaderClose(fst)
    
    # 3. Auto-detect endpoint cycle if not specified
    if endpoint_cycle is None:
        log("[*] Auto-detecting assertion trigger cycle from waveform...")
        endpoint_cycle = detect_assertion_trigger_cycle(fst_path, endpoint_signal, clock_signal)
        log(f"[+] Detected assertion trigger at cycle: {endpoint_cycle}")
    
    # === Build CausalGraphBuilder ===
    
    log(f"[*] Loading FST: {fst_path}")
    log(f"[*] Loading Verilog: {', '.join(verilog_paths)}")
    log(f"[*] Clock signal: {clock_signal}")
    log(f"[*] Endpoint signal: {endpoint_signal}")
    log(f"[*] Endpoint cycle: {endpoint_cycle}")
    
    builder = CausalGraphBuilder(
        fst_path=fst_path,
        verilog_paths=verilog_paths,
        clock_signal=clock_signal,
        max_depth=max_depth,
        max_nodes=max_nodes,
        random_seed=random_seed
    )
    
    return builder, endpoint_signal, endpoint_cycle
