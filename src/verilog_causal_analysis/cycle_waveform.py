"""
Cycle-aligned Waveform Parser for Causal Graph Construction.

Parses FST waveform files and aligns signal values to clock cycles.
Provides discrete value(signal, cycle) table for causal analysis.

Key Features:
- Clock edge detection for cycle boundary identification
- Efficient signal value caching with O(1) lookup after first access
- Binary search for time-to-cycle conversion
"""

import os
import re
from bisect import bisect_left, bisect_right
from collections import OrderedDict, defaultdict
from dataclasses import dataclass, field
from threading import RLock
from typing import Dict, Iterable, List, Tuple, Optional, Any
import pylibfst


@dataclass
class SignalTransition:
    """A signal value transition."""
    signal_name: str
    time: int
    cycle: int
    old_value: str
    new_value: str


@dataclass
class CycleSnapshot:
    """Snapshot of all signal values at a specific cycle."""
    cycle: int
    time_start: int
    time_end: int
    values: Dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class SignalResolution:
    """Cycle-independent waveform identity resolution."""

    requested_signal: str
    hierarchy: str
    resolved_signal: Optional[str]
    candidates: Tuple[str, ...]
    identity_strength: str
    ambiguous: bool


class CycleAlignedWaveform:
    """
    Waveform parser that aligns signal values to clock cycles.
    
    Uses rising edges of a specified clock signal to define cycle boundaries.
    Provides discrete value(signal, cycle) lookups for causal analysis.
    """
    
    def __init__(
        self,
        fst_path: str,
        clock_signal: str = "clock",
        *,
        exact_clock: bool = False,
        value_cache_capacity: int = 16384,
        transition_cache_capacity: int = 131072,
    ):
        """
        Initialize cycle-aligned waveform parser.
        
        Args:
            fst_path: Path to FST waveform file
            clock_signal: Full hierarchical name of clock signal (e.g., "TestTop.clock")
        """
        if not os.path.exists(fst_path):
            raise FileNotFoundError(f"FST file not found: {fst_path}")
        
        self.fst_path = fst_path
        self.clock_signal = clock_signal
        self.exact_clock = exact_clock
        if value_cache_capacity <= 0 or transition_cache_capacity <= 0:
            raise ValueError("waveform cache capacities must be positive")
        self.value_cache_capacity = value_cache_capacity
        self.transition_cache_capacity = transition_cache_capacity
        self._reader_lock = RLock()
        
        # Open FST file
        self.fst = pylibfst.lib.fstReaderOpen(fst_path.encode("UTF-8"))
        if self.fst == pylibfst.ffi.NULL:
            raise RuntimeError(f"Failed to open FST file: {fst_path}")
        
        # Get scopes and signals
        self.scopes, self.signals = pylibfst.get_scopes_signals2(self.fst)
        self._signal_names = tuple(sorted(self.signals.by_name))
        self._normalized_names: Dict[str, Tuple[str, ...]] = {}
        self._hierarchy_base_names: Dict[Tuple[str, str], Tuple[str, ...]] = {}
        self._base_names: Dict[str, Tuple[str, ...]] = {}
        self._suffix_names: Dict[str, Tuple[str, ...]] = {}
        self._resolution_cache: Dict[
            Tuple[str, str, bool], SignalResolution
        ] = {}
        self._find_cache: Dict[Tuple[str, int], Tuple[str, ...]] = {}
        self._resolution_hits = 0
        self._resolution_misses = 0
        self._build_signal_index()
        
        # Get metadata
        self.start_time = pylibfst.lib.fstReaderGetStartTime(self.fst)
        self.end_time = pylibfst.lib.fstReaderGetEndTime(self.fst)
        self.timescale = pylibfst.lib.fstReaderGetTimescale(self.fst)
        
        # Cycle boundaries (list of rising edge times)
        self._cycle_boundaries: List[int] = []
        self._cycle_count: int = 0
        
        # Cached values: (signal_name, cycle) -> value
        self._value_cache: "OrderedDict[Tuple[str, int], str]" = OrderedDict()
        self._value_hits = 0
        self._value_misses = 0
        self._value_evictions = 0
        self._transition_cache: "OrderedDict[str, Tuple[Tuple[int, str], ...]]" = (
            OrderedDict()
        )
        self._transition_value_count = 0
        self._transition_batches = 0
        self._transition_hits = 0
        self._transition_misses = 0
        self._transition_evictions = 0
        self._transition_overflow_signals: set[str] = set()
        self._value_buffer = pylibfst.ffi.new("char[256]")
        
        # Initialize cycle boundaries
        self._build_cycle_boundaries()

    @staticmethod
    def _normalize_signal_name(signal_name: str) -> str:
        return re.sub(
            r"\s*\[\d+:\d+\]$",
            "",
            signal_name.strip(),
        ).lower()

    def _build_signal_index(self) -> None:
        """Build immutable lookup tables for cycle-independent resolution."""
        normalized: Dict[str, List[str]] = defaultdict(list)
        hierarchy_base: Dict[Tuple[str, str], List[str]] = defaultdict(list)
        base_names: Dict[str, List[str]] = defaultdict(list)
        suffix_names: Dict[str, List[str]] = defaultdict(list)

        for signal_name in self._signal_names:
            clean = self._normalize_signal_name(signal_name)
            normalized[clean].append(signal_name)
            parts = clean.split(".")
            hierarchy = ".".join(parts[:-1])
            base = parts[-1]
            hierarchy_base[(hierarchy, base)].append(signal_name)
            base_names[base].append(signal_name)
            for index in range(len(parts)):
                suffix_names[".".join(parts[index:])].append(signal_name)

        freeze = lambda rows: {
            key: tuple(sorted(set(values)))
            for key, values in rows.items()
        }
        self._normalized_names = freeze(normalized)
        self._hierarchy_base_names = freeze(hierarchy_base)
        self._base_names = freeze(base_names)
        self._suffix_names = freeze(suffix_names)
    
    def _build_cycle_boundaries(self):
        """Build cycle boundaries from clock rising edges."""
        clock = self.signals.by_name.get(self.clock_signal)
        if not clock and not self.exact_clock:
            # Try to find clock with partial match
            for sig_name in self.signals.by_name.keys():
                if 'clock' in sig_name.lower() or 'clk' in sig_name.lower():
                    clock = self.signals.by_name[sig_name]
                    self.clock_signal = sig_name
                    break
        
        if not clock:
            raise ValueError(f"Clock signal not found: {self.clock_signal}")
        
        # Set process mask for clock
        with self._reader_lock:
            pylibfst.lib.fstReaderClrFacProcessMaskAll(self.fst)
            pylibfst.lib.fstReaderSetFacProcessMask(self.fst, clock.handle)

            timestamps = pylibfst.lib.fstReaderGetTimestamps(self.fst)
            if timestamps.nvals == 0:
                pylibfst.lib.fstReaderFreeTimestamps(timestamps)
                raise RuntimeError("No timestamps found in waveform")

            prev_value = None
            for ts in range(timestamps.nvals):
                time = timestamps.val[ts]
                value = pylibfst.helpers.string(
                    pylibfst.lib.fstReaderGetValueFromHandleAtTime(
                        self.fst, time, clock.handle, self._value_buffer
                    )
                )

                if value == '1' and prev_value in ('0', 'x', 'X', None):
                    self._cycle_boundaries.append(int(time))
                prev_value = value

            pylibfst.lib.fstReaderFreeTimestamps(timestamps)
        self._cycle_count = len(self._cycle_boundaries)

    def has_exact_signal(self, signal_name: str) -> bool:
        return signal_name in self.signals.by_name

    def resolve_signal(
        self,
        signal_name: str,
        hierarchy: str = "",
        *,
        prefer_hierarchy: bool = True,
    ) -> SignalResolution:
        """Resolve a waveform signal once and cache the identity result.

        Resolution is independent of cycle. Only exact names and a unique
        normalized hierarchy match are accepted.
        """
        cache_key = (signal_name, hierarchy, prefer_hierarchy)
        cached = self._resolution_cache.get(cache_key)
        if cached is not None:
            self._resolution_hits += 1
            return cached
        self._resolution_misses += 1

        qualified = (
            f"{hierarchy}.{signal_name}"
            if hierarchy and not signal_name.startswith(hierarchy + ".")
            else signal_name
        )
        exact_order = (
            (qualified, signal_name)
            if prefer_hierarchy
            else (signal_name, qualified)
        )
        for candidate in dict.fromkeys(exact_order):
            if candidate in self.signals.by_name:
                strength = (
                    "hierarchy_inferred"
                    if candidate != signal_name
                    else "exact"
                )
                result = SignalResolution(
                    requested_signal=signal_name,
                    hierarchy=hierarchy,
                    resolved_signal=candidate,
                    candidates=(candidate,),
                    identity_strength=strength,
                    ambiguous=False,
                )
                self._resolution_cache[cache_key] = result
                return result

        normalized_signal = self._normalize_signal_name(signal_name)
        normalized_hierarchy = self._normalize_signal_name(hierarchy)
        normalized_qualified = (
            f"{normalized_hierarchy}.{normalized_signal}"
            if normalized_hierarchy
            and not normalized_signal.startswith(normalized_hierarchy + ".")
            else normalized_signal
        )
        indexed_candidates: List[str] = []
        for key in dict.fromkeys((normalized_qualified, normalized_signal)):
            indexed_candidates.extend(self._normalized_names.get(key, ()))

        candidates = tuple(sorted(set(indexed_candidates)))
        if candidates:
            direct = tuple(
                candidate
                for candidate in candidates
                if self._normalize_signal_name(candidate) == normalized_signal
            )
            if len(direct) == 1:
                result = SignalResolution(
                    requested_signal=signal_name,
                    hierarchy=hierarchy,
                    resolved_signal=direct[0],
                    candidates=direct,
                    identity_strength=(
                        "exact"
                        if direct[0] == signal_name
                        else "hierarchy_inferred"
                    ),
                    ambiguous=False,
                )
                self._resolution_cache[cache_key] = result
                return result
            preferred = tuple(
                candidate
                for candidate in candidates
                if self._normalize_signal_name(candidate) == normalized_qualified
            )
            if preferred:
                candidates = preferred
            exact_normalized = (
                len(candidates) == 1
                and self._normalize_signal_name(candidates[0])
                == normalized_qualified
            )
            result = SignalResolution(
                requested_signal=signal_name,
                hierarchy=hierarchy,
                resolved_signal=candidates[0] if exact_normalized else None,
                candidates=candidates,
                identity_strength=(
                    "hierarchy_inferred" if exact_normalized else "unresolved"
                ),
                ambiguous=not exact_normalized,
            )
        else:
            result = SignalResolution(
                requested_signal=signal_name,
                hierarchy=hierarchy,
                resolved_signal=None,
                candidates=(),
                identity_strength="unresolved",
                ambiguous=False,
            )
        self._resolution_cache[cache_key] = result
        return result

    def get_resolved_signal_value(
        self,
        resolution: SignalResolution,
        cycle: int,
    ) -> Optional[str]:
        """Read a value from a previously resolved signal identity."""
        if resolution.resolved_signal is None:
            return None
        return self.get_signal_value(resolution.resolved_signal, cycle)

    def get_resolved_signal_values(
        self,
        resolutions: Iterable[SignalResolution],
        cycle: int,
    ) -> Dict[str, str]:
        """Batch value lookup without repeating name resolution."""
        values: Dict[str, str] = {}
        for resolution in resolutions:
            value = self.get_resolved_signal_value(resolution, cycle)
            if value is not None and resolution.resolved_signal is not None:
                values[resolution.resolved_signal] = value
        return values
    
    def get_cycle_count(self) -> int:
        """Get total number of clock cycles."""
        return self._cycle_count
    
    def time_to_cycle(self, time: int) -> int:
        """Convert simulation time to cycle number (0-indexed)."""
        if not self._cycle_boundaries:
            return 0
        idx = bisect_right(self._cycle_boundaries, time) - 1
        return max(0, idx)
    
    def cycle_to_time(self, cycle: int) -> int:
        """Get the start time of a cycle (rising edge time)."""
        if cycle < 0:
            return self.start_time
        if cycle >= self._cycle_count:
            return self.end_time
        return self._cycle_boundaries[cycle]
    
    def get_cycle_time_range(self, cycle: int) -> Tuple[int, int]:
        """Get (start_time, end_time) for a cycle."""
        start = self.cycle_to_time(cycle)
        if cycle + 1 < self._cycle_count:
            end = self._cycle_boundaries[cycle + 1] - 1
        else:
            end = self.end_time
        return start, end
    
    def get_signal_value(self, signal_name: str, cycle: int) -> Optional[str]:
        """
        Get the value of a signal at a specific cycle.
        
        Uses the value at the end of the cycle (just before next rising edge).
        
        Args:
            signal_name: Full hierarchical signal name
            cycle: Cycle number
            
        Returns:
            Signal value as string, or None if not found
        """
        cache_key = (signal_name, cycle)
        if cache_key in self._value_cache:
            self._value_hits += 1
            move_to_end = getattr(self._value_cache, "move_to_end", None)
            if callable(move_to_end):
                move_to_end(cache_key)
            return self._value_cache[cache_key]
        self._value_misses += 1
        
        signal = self.signals.by_name.get(signal_name)
        if not signal:
            return None
        
        # Get time at end of cycle
        if cycle + 1 < self._cycle_count:
            sample_time = self._cycle_boundaries[cycle + 1] - 1
        else:
            sample_time = self.end_time
        
        with self._reader_lock:
            value = pylibfst.helpers.string(
                pylibfst.lib.fstReaderGetValueFromHandleAtTime(
                    self.fst, sample_time, signal.handle, self._value_buffer
                )
            )

        self._value_cache[cache_key] = value
        self._value_cache.move_to_end(cache_key)
        while len(self._value_cache) > self.value_cache_capacity:
            self._value_cache.popitem(last=False)
            self._value_evictions += 1
        return value
    
    def get_signal_value_at_cycle_start(self, signal_name: str, cycle: int) -> Optional[str]:
        """
        Get the value of a signal at the start of a cycle (just after rising edge).
        
        Args:
            signal_name: Full hierarchical signal name
            cycle: Cycle number
            
        Returns:
            Signal value as string, or None if not found
        """
        signal = self.signals.by_name.get(signal_name)
        if not signal:
            return None
        
        sample_time = self.cycle_to_time(cycle)
        
        with self._reader_lock:
            value = pylibfst.helpers.string(
                pylibfst.lib.fstReaderGetValueFromHandleAtTime(
                    self.fst, sample_time, signal.handle, self._value_buffer
                )
            )
        return value

    def prepare_transition_index(
        self,
        signal_names: Iterable[str],
    ) -> Tuple[str, ...]:
        """Stream selected signals into a bounded, per-reader transition index.

        The mutable FST process mask is protected by ``_reader_lock`` and never
        shared concurrently. Signals whose transition count exceeds the
        configured capacity are marked unavailable rather than falling back to
        an unbounded per-cycle scan.
        """
        requested = tuple(
            sorted(
                {
                    signal_name
                    for signal_name in signal_names
                    if signal_name in self.signals.by_name
                }
            )
        )
        ready: List[str] = []
        missing: List[str] = []
        for signal_name in requested:
            if signal_name in self._transition_cache:
                self._transition_hits += 1
                self._transition_cache.move_to_end(signal_name)
                ready.append(signal_name)
            elif signal_name in self._transition_overflow_signals:
                self._transition_misses += 1
            else:
                missing.append(signal_name)
        if not missing:
            return tuple(ready)

        self._transition_misses += len(missing)
        handles: Dict[int, Tuple[str, ...]] = defaultdict(tuple)
        names_by_handle: Dict[int, List[str]] = defaultdict(list)
        for signal_name in missing:
            names_by_handle[int(self.signals.by_name[signal_name].handle)].append(
                signal_name
            )
        handles = {
            handle: tuple(sorted(names))
            for handle, names in names_by_handle.items()
        }
        collected: Dict[str, List[Tuple[int, str]]] = {
            signal_name: [] for signal_name in missing
        }
        overflowed: set[str] = set()
        batch_values = 0
        batch_overflowed = False

        def record(_data, time, facidx, value, _length=None):
            nonlocal batch_overflowed, batch_values
            names = handles.get(int(facidx), ())
            if not names or batch_overflowed:
                return
            decoded = pylibfst.helpers.string(value)
            for signal_name in names:
                if signal_name in overflowed:
                    continue
                if batch_values >= self.transition_cache_capacity:
                    # A partially collected series is never authoritative.
                    # Fail the complete selected batch closed rather than
                    # caching a prefix for signals that happen not to toggle
                    # again after the global capacity is reached.
                    batch_overflowed = True
                    return
                row = (int(time), decoded)
                if (
                    collected[signal_name]
                    and collected[signal_name][-1] == row
                ):
                    continue
                collected[signal_name].append(row)
                batch_values += 1

        with self._reader_lock:
            pylibfst.lib.fstReaderClrFacProcessMaskAll(self.fst)
            for handle in sorted(handles):
                pylibfst.lib.fstReaderSetFacProcessMask(self.fst, handle)
            result = pylibfst.helpers.fstReaderIterBlocks2(
                self.fst,
                record,
                record,
                None,
            )
            pylibfst.lib.fstReaderClrFacProcessMaskAll(self.fst)
        self._transition_batches += 1
        if not result or batch_overflowed:
            overflowed.update(missing)
            for signal_name in missing:
                collected[signal_name].clear()

        for signal_name in missing:
            rows = tuple(collected[signal_name])
            if signal_name in overflowed:
                self._transition_overflow_signals.add(signal_name)
                continue
            while (
                self._transition_cache
                and self._transition_value_count + len(rows)
                > self.transition_cache_capacity
            ):
                _, evicted = self._transition_cache.popitem(last=False)
                self._transition_value_count -= len(evicted)
                self._transition_evictions += 1
            if len(rows) > self.transition_cache_capacity:
                self._transition_overflow_signals.add(signal_name)
                continue
            self._transition_cache[signal_name] = rows
            self._transition_value_count += len(rows)
            ready.append(signal_name)
        return tuple(sorted(set(ready)))

    def get_value_changes_bounded(
        self,
        signal_name: str,
        start_cycle: int,
        end_cycle: int,
        *,
        max_changes: int = 5,
    ) -> Optional[List[Tuple[int, str, str]]]:
        """Return cycle-sampled changes using transitions, never a cycle scan."""
        if max_changes <= 0:
            return []
        if signal_name not in self.prepare_transition_index([signal_name]):
            return None
        rows = self._transition_cache[signal_name]
        start_time = self.cycle_to_time(max(0, start_cycle))
        end_time = (
            self.cycle_to_time(end_cycle + 1) - 1
            if end_cycle + 1 < self._cycle_count
            else self.end_time
        )
        left = bisect_left(rows, (start_time, ""))
        right = bisect_right(rows, (end_time, "\U0010ffff"))
        candidate_cycles = set()
        for time, _value in rows[left:right]:
            cycle = self.time_to_cycle(time)
            if start_cycle <= cycle <= end_cycle:
                candidate_cycles.add(cycle)
        changes: List[Tuple[int, str, str]] = []
        for cycle in sorted(candidate_cycles):
            current = self.get_signal_value(signal_name, cycle)
            previous = (
                self.get_signal_value(signal_name, cycle - 1)
                if cycle > 0
                else None
            )
            if current is not None and previous is not None and current != previous:
                changes.append((cycle, previous, current))
                if len(changes) >= max_changes:
                    break
        return changes

    def get_transition_series_bounded(
        self,
        signal_name: str,
        start_cycle: int,
        end_cycle: int,
        *,
        max_values: int,
    ) -> Dict[str, Any]:
        """Return bounded cycle changes plus exact boundary/coverage metadata."""
        if max_values <= 0:
            raise ValueError("max_values must be positive")
        changes = self.get_value_changes_bounded(
            signal_name,
            start_cycle,
            end_cycle,
            max_changes=max_values + 1,
        )
        start_value = self.get_signal_value(signal_name, start_cycle)
        end_value = self.get_signal_value(signal_name, end_cycle)
        if changes is None:
            return {
                "signal": signal_name,
                "waveform_signal": signal_name,
                "available": False,
                "truncated": False,
                "changes": [],
                "boundary_values": {"start": start_value, "end": end_value},
                "unknown_spans": [[start_cycle, end_cycle]],
                "work": {
                    "transition_values": 0,
                    "value_misses": int(start_value is None)
                    + int(end_value is None),
                },
            }
        truncated = len(changes) > max_values
        bounded = changes[:max_values]
        return {
            "signal": signal_name,
            "waveform_signal": signal_name,
            "available": True,
            "truncated": truncated,
            "changes": [
                {"cycle": cycle, "old": old, "new": new}
                for cycle, old, new in bounded
            ],
            "boundary_values": {"start": start_value, "end": end_value},
            "unknown_spans": (
                [[bounded[-1][0] if bounded else start_cycle, end_cycle]]
                if truncated
                else []
            ),
            "work": {
                "transition_values": len(bounded),
                "value_misses": int(start_value is None)
                + int(end_value is None),
            },
        }
    
    def get_signal_transitions_in_cycle(self, signal_name: str, cycle: int) -> List[SignalTransition]:
        """
        Get all value transitions of a signal within a cycle.
        
        Args:
            signal_name: Signal name
            cycle: Cycle number
            
        Returns:
            List of SignalTransition objects
        """
        signal = self.signals.by_name.get(signal_name)
        if not signal:
            return []
        
        start_time, end_time = self.get_cycle_time_range(cycle)
        
        if signal_name not in self.prepare_transition_index([signal_name]):
            return []
        rows = self._transition_cache[signal_name]
        left = bisect_left(rows, (start_time, ""))
        right = bisect_right(rows, (end_time, "\U0010ffff"))
        transitions = []
        previous = rows[left - 1][1] if left > 0 else None
        for time, value in rows[left:right]:
            if previous is not None and value != previous:
                transitions.append(
                    SignalTransition(
                        signal_name=signal_name,
                        time=time,
                        cycle=cycle,
                        old_value=previous,
                        new_value=value,
                    )
                )
            previous = value
        return transitions
    
    def find_signal(self, pattern: str, max_results: int = 100) -> List[str]:
        """
        Find signals matching a pattern.
        
        Args:
            pattern: Substring to match
            max_results: Maximum number of results
            
        Returns:
            List of matching signal names
        """
        cache_key = (pattern.lower(), max_results)
        cached = self._find_cache.get(cache_key)
        if cached is None:
            cached = tuple(
                signal_name
                for signal_name in self._signal_names
                if pattern.lower() in signal_name.lower()
            )[:max_results]
            self._find_cache[cache_key] = cached
        return list(cached)

    def get_cache_statistics(self) -> Dict[str, int]:
        """Return non-authoritative waveform cache counters."""
        return {
            "signal_resolution_hits": self._resolution_hits,
            "signal_resolution_misses": self._resolution_misses,
            "signal_resolution_entries": len(self._resolution_cache),
            "waveform_value_entries": len(self._value_cache),
            "waveform_value_hits": getattr(self, "_value_hits", 0),
            "waveform_value_misses": getattr(self, "_value_misses", 0),
            "waveform_value_evictions": getattr(self, "_value_evictions", 0),
            "waveform_value_capacity": getattr(
                self, "value_cache_capacity", len(self._value_cache)
            ),
            "waveform_transition_entries": len(
                getattr(self, "_transition_cache", {})
            ),
            "waveform_transition_values": getattr(
                self, "_transition_value_count", 0
            ),
            "waveform_transition_batches": getattr(
                self, "_transition_batches", 0
            ),
            "waveform_transition_hits": getattr(
                self, "_transition_hits", 0
            ),
            "waveform_transition_misses": getattr(
                self, "_transition_misses", 0
            ),
            "waveform_transition_evictions": getattr(
                self, "_transition_evictions", 0
            ),
            "waveform_transition_overflow_signals": len(
                getattr(self, "_transition_overflow_signals", ())
            ),
            "waveform_transition_capacity": getattr(
                self, "transition_cache_capacity", 0
            ),
        }
    
    def get_all_signals(self) -> List[str]:
        """Get list of all signal names."""
        return list(self.signals.by_name.keys())
    
    def get_cycle_snapshot(self, cycle: int, signals: Optional[List[str]] = None) -> CycleSnapshot:
        """
        Get snapshot of signal values at a cycle.
        
        Args:
            cycle: Cycle number
            signals: Optional list of signals to include (all if None)
            
        Returns:
            CycleSnapshot with values for specified signals
        """
        start_time, end_time = self.get_cycle_time_range(cycle)
        snapshot = CycleSnapshot(
            cycle=cycle,
            time_start=start_time,
            time_end=end_time
        )
        
        if signals is None:
            signals = list(self.signals.by_name.keys())
        
        with self._reader_lock:
            for sig_name in signals:
                signal = self.signals.by_name.get(sig_name)
                if signal:
                    value = pylibfst.helpers.string(
                        pylibfst.lib.fstReaderGetValueFromHandleAtTime(
                            self.fst, end_time, signal.handle, self._value_buffer
                        )
                    )
                    snapshot.values[sig_name] = value
        
        return snapshot
    
    def get_value_changes(self, signal_name: str, 
                          start_cycle: int = 0, 
                          end_cycle: Optional[int] = None) -> List[Tuple[int, str, str]]:
        """
        Get all value changes for a signal between cycles.
        
        Args:
            signal_name: Signal name
            start_cycle: Starting cycle
            end_cycle: Ending cycle (inclusive), or None for end
            
        Returns:
            List of (cycle, old_value, new_value) tuples
        """
        if end_cycle is None:
            end_cycle = self._cycle_count - 1
        
        bounded = self.get_value_changes_bounded(
            signal_name,
            start_cycle,
            min(end_cycle, self._cycle_count - 1),
            max_changes=max(1, self._cycle_count),
        )
        return bounded or []
    
    def get_metadata(self) -> Dict[str, Any]:
        """Get waveform metadata."""
        return {
            "fst_path": self.fst_path,
            "clock_signal": self.clock_signal,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "timescale": self.timescale,
            "cycle_count": self._cycle_count,
            "signal_count": len(self.signals.by_name)
        }
    
    def build_value_table(self, signals: List[str], 
                          start_cycle: int = 0,
                          end_cycle: Optional[int] = None) -> Dict[str, Dict[int, str]]:
        """
        Build a complete value table for signals over cycles.
        
        Args:
            signals: List of signal names
            start_cycle: Starting cycle
            end_cycle: Ending cycle, or None for end
            
        Returns:
            Dictionary: signal_name -> {cycle: value}
        """
        if end_cycle is None:
            end_cycle = self._cycle_count - 1
        
        table = {}
        for sig_name in signals:
            table[sig_name] = {}
            for cycle in range(start_cycle, min(end_cycle + 1, self._cycle_count)):
                value = self.get_signal_value(sig_name, cycle)
                if value is not None:
                    table[sig_name][cycle] = value
        
        return table
    
    def close(self):
        """Close the FST file."""
        if self.fst:
            pylibfst.lib.fstReaderClose(self.fst)
            self.fst = None

    def __enter__(self) -> "CycleAlignedWaveform":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        self.close()
        return False
    
    def __del__(self):
        """Cleanup on deletion."""
        self.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False


def parse_binary_value(value: str) -> Optional[int]:
    """Parse binary string to int; returns None if contains x/z."""
    if not value or 'x' in value.lower() or 'z' in value.lower():
        return None
    try:
        return int(value, 2)
    except ValueError:
        return None


def invert_value(value: str) -> str:
    """Bitwise invert binary value (preserves x/z)."""
    if not value:
        return value

    return value.translate(str.maketrans('01', '10'))


def values_differ(val1: str, val2: str) -> bool:
    """Check if two binary values differ (ignoring x/z bits)."""
    if not val1 or not val2:
        return False
    
    # Pad to same length
    max_len = max(len(val1), len(val2))
    val1 = val1.zfill(max_len)
    val2 = val2.zfill(max_len)
    
    for c1, c2 in zip(val1, val2):
        if c1 in 'xXzZ' or c2 in 'xXzZ':
            continue  # Can't determine
        if c1 != c2:
            return True
    
    return False
