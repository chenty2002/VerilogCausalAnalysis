"""
Backward Causal Slicing Engine for Counterexample Analysis.

Performs backward slicing from counterexample endpoint to build
a causal DAG with counterfactual evaluation at expression level.
"""

import re
import hashlib
from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple, Optional, Any, Callable
from enum import Enum

from .verilog_parser import VerilogParser, Dependency, DependencyType
from .cycle_waveform import CycleAlignedWaveform, parse_binary_value, invert_value, values_differ


class ContributionType(Enum):
    """Type of causal contribution."""
    EXPR_EVAL = "expr_eval"        # Expression evaluation shows causality
    TOGGLE = "toggle"              # Signal toggle affects output
    STATE = "state"                # State machine transition
    DIRECT = "direct"              # Direct assignment
    CONDITIONAL = "conditional"    # Condition branch taken
    UNKNOWN = "unknown"            # Could not determine


@dataclass
class CausalNode:
    """A node in the causal DAG representing (signal, cycle, value)."""
    id: str                        # Unique node ID
    signal: str                    # Signal name
    cycle: int                     # Clock cycle
    value: str                     # Signal value at this cycle
    suspect_score: float = 0.0     # Suspiciousness score (0-1)
    rtl_refs: List[Dict] = field(default_factory=list)  # RTL file references
    rtl_context_missing: bool = False  # True if RTL context unavailable
    is_root: bool = False          # True if this is a root cause candidate
    is_endpoint: bool = False      # True if this is the analysis endpoint
    depth: int = 0                 # Distance from endpoint
    
    def __hash__(self):
        return hash(self.id)
    
    def __eq__(self, other):
        if not isinstance(other, CausalNode):
            return False
        return self.id == other.id
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "signal": self.signal,
            "cycle": self.cycle,
            "value": self.value,
            "suspect_score": self.suspect_score,
            "rtl_refs": self.rtl_refs,
            "rtl_context_missing": self.rtl_context_missing,
            "is_root": self.is_root,
            "is_endpoint": self.is_endpoint,
            "depth": self.depth
        }


@dataclass
class CausalEdge:
    """An edge in the causal DAG representing direct causality."""
    src_node_id: str               # Source node ID
    dst_node_id: str               # Destination node ID
    reason: str                    # Human-readable reason
    contribution_type: ContributionType
    contribution_score: float      # Strength of contribution (0-1)
    evidence: Dict[str, Any] = field(default_factory=dict)  # RTL evidence
    change_examples: List[Dict] = field(default_factory=list)  # Counterfactual examples
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "src_node_id": self.src_node_id,
            "dst_node_id": self.dst_node_id,
            "reason": self.reason,
            "contribution_type": self.contribution_type.value,
            "contribution_score": self.contribution_score,
            "evidence": self.evidence,
            "change_examples": self.change_examples
        }


class ExpressionEvaluator:
    """
    Simple expression evaluator for counterfactual analysis.
    
    Evaluates Verilog expressions with given signal values.
    """
    
    # Operator precedence (higher = binds tighter)
    PRECEDENCE = {
        '||': 1, '|': 2,
        '&&': 3, '&': 4,
        '^': 5, '~^': 5, '^~': 5,
        '==': 6, '!=': 6, '===': 6, '!==': 6,
        '<': 7, '>': 7, '<=': 7, '>=': 7,
        '<<': 8, '>>': 8, '<<<': 8, '>>>': 8,
        '+': 9, '-': 9,
        '*': 10, '/': 10, '%': 10,
        '!': 11, '~': 11
    }
    
    RE_NUMBER = re.compile(r"(\d+)'([bhd])([0-9a-fA-F_xXzZ]+)")
    RE_DECIMAL = re.compile(r'\b(\d+)\b')
    RE_TERNARY = re.compile(r'(.+?)\s*\?\s*(.+?)\s*:\s*(.+)')  
    # SVA implication pattern: antecedent |-> ##[n:m] consequent
    RE_SVA_IMPLICATION = re.compile(
        r'^(.+?)\s*\|->\s*(?:##\[\d+:\d+\]\s*)?(.+)$',
        re.DOTALL
    )
    
    def __init__(self, signal_values: Dict[str, str]):
        """
        Initialize evaluator with signal values.
        
        Args:
            signal_values: Dictionary of signal_name -> binary_value
        """
        self.signal_values = signal_values
    
    def evaluate(self, expr: str) -> Optional[str]:
        """
        Evaluate a Verilog expression.
        
        Args:
            expr: Verilog expression string
            
        Returns:
            Result as binary string, or None if cannot evaluate
        """
        expr = expr.strip()
        if not expr:
            return None
        
        try:
            # Check for SVA implication first
            sva_match = self.RE_SVA_IMPLICATION.match(expr)
            if sva_match:
                return self._eval_sva_implication(sva_match.group(1), sva_match.group(2))
            return self._eval_expr(expr)
        except Exception:
            return None
    
    def _eval_sva_implication(self, antecedent: str, consequent: str) -> Optional[str]:
        """
        Evaluate SVA implication: antecedent |-> consequent.
        
        For causality analysis:
        - If antecedent is true, we need consequent to be true for assertion to pass
        - If antecedent is true and consequent is false, assertion fails (return '0')
        - If antecedent is false, assertion vacuously passes (return '1')
        
        Note: This is a simplified evaluation for single-cycle analysis.
        The ##[n:m] delay is ignored here as we're analyzing at the trigger point.
        """
        ante_val = self._eval_expr(antecedent.strip())
        
        if ante_val is None:
            return None
        
        # If antecedent is false, implication vacuously true
        if not self._is_true(ante_val):
            return '1'
        
        # Antecedent is true, so we evaluate consequent at this point
        # For a failed assertion with delay, consequent would be false
        cons_val = self._eval_expr(consequent.strip())
        
        if cons_val is None:
            # Can't evaluate consequent - for liveness property violation,
            # if we're at the failure point and antecedent was true,
            # the consequent must have been false throughout the window
            # Return '0' to indicate failure
            return '0'
        
        return cons_val
    
    def _eval_expr(self, expr: str) -> Optional[str]:
        """Evaluate expression recursively."""
        expr = expr.strip()
        
        # Handle Verilog concatenation: {a, b, c}
        if expr.startswith('{') and expr.endswith('}'):
            inner = expr[1:-1]
            # Split by comma, respecting nesting
            parts = self._split_concat_parts(inner)
            if parts:
                result_bits = []
                for part in parts:
                    val = self._eval_expr(part.strip())
                    if val is None:
                        return None
                    result_bits.append(val)
                return ''.join(result_bits)
        
        # Handle parentheses
        if expr.startswith('(') and expr.endswith(')'):
            depth = 0
            for i, c in enumerate(expr):
                if c == '(':
                    depth += 1
                elif c == ')':
                    depth -= 1
                if depth == 0 and i < len(expr) - 1:
                    break
            else:
                return self._eval_expr(expr[1:-1])
        
        # Handle ternary operator
        ternary = self._parse_ternary(expr)
        if ternary:
            cond, then_expr, else_expr = ternary
            cond_val = self._eval_expr(cond)
            if cond_val is None:
                return None
            if self._is_true(cond_val):
                return self._eval_expr(then_expr)
            else:
                return self._eval_expr(else_expr)
        
        # Handle unary operators
        if expr.startswith('!'):
            val = self._eval_expr(expr[1:])
            if val is None:
                return None
            return '1' if not self._is_true(val) else '0'
        
        if expr.startswith('~'):
            val = self._eval_expr(expr[1:])
            if val is None:
                return None
            return self._bitwise_not(val)
        
        # Handle unary reduction operators (Verilog): &, |, ^, ~&, ~|, ~^
        # These reduce a multi-bit value to a single bit
        if expr.startswith('&') and len(expr) > 1 and expr[1] != '&':
            val = self._eval_expr(expr[1:])
            if val is None:
                return None
            # Reduction AND: all bits must be 1
            return '1' if all(c == '1' for c in val) else '0'
        
        if expr.startswith('|') and len(expr) > 1 and expr[1] != '|':
            val = self._eval_expr(expr[1:])
            if val is None:
                return None
            # Reduction OR: any bit must be 1
            return '1' if '1' in val else '0'
        
        if expr.startswith('^') and len(expr) > 1:
            val = self._eval_expr(expr[1:])
            if val is None:
                return None
            # Reduction XOR: count of 1s is odd
            count = sum(1 for c in val if c == '1')
            return '1' if count % 2 == 1 else '0'
        
        # Handle binary operators (find lowest precedence)
        op_pos, op = self._find_lowest_op(expr)
        if op:
            left = self._eval_expr(expr[:op_pos])
            right = self._eval_expr(expr[op_pos + len(op):])
            if left is None or right is None:
                return None
            return self._apply_binary_op(left, right, op)
        
        # Handle literals and signals
        return self._eval_atom(expr)
    
    def _split_concat_parts(self, inner: str) -> List[str]:
        """Split concatenation expression by commas, respecting nesting."""
        parts = []
        depth = 0
        current = []
        
        for c in inner:
            if c in '({':
                depth += 1
                current.append(c)
            elif c in ')}':
                depth -= 1
                current.append(c)
            elif c == ',' and depth == 0:
                parts.append(''.join(current))
                current = []
            else:
                current.append(c)
        
        if current:
            parts.append(''.join(current))
        
        return parts
    
    def _parse_ternary(self, expr: str) -> Optional[Tuple[str, str, str]]:
        """Parse ternary operator, respecting parentheses."""
        depth = 0
        q_pos = -1
        c_pos = -1
        
        for i, c in enumerate(expr):
            if c == '(':
                depth += 1
            elif c == ')':
                depth -= 1
            elif c == '?' and depth == 0 and q_pos < 0:
                q_pos = i
            elif c == ':' and depth == 0 and q_pos >= 0:
                c_pos = i
                break
        
        if q_pos > 0 and c_pos > q_pos:
            return (expr[:q_pos].strip(),
                    expr[q_pos+1:c_pos].strip(),
                    expr[c_pos+1:].strip())
        return None
    
    def _find_lowest_op(self, expr: str) -> Tuple[int, Optional[str]]:
        """Find the lowest precedence operator not inside parentheses."""
        depth = 0
        lowest_prec = 999
        lowest_pos = -1
        lowest_op = None
        
        i = 0
        while i < len(expr):
            c = expr[i]
            if c == '(':
                depth += 1
            elif c == ')':
                depth -= 1
            elif depth == 0:
                # Check for operators (longest match first)
                for op in sorted(self.PRECEDENCE.keys(), key=len, reverse=True):
                    if expr[i:i+len(op)] == op:
                        prec = self.PRECEDENCE[op]
                        if prec <= lowest_prec:
                            lowest_prec = prec
                            lowest_pos = i
                            lowest_op = op
                        break
            i += 1
        
        return lowest_pos, lowest_op
    
    def _eval_atom(self, expr: str) -> Optional[str]:
        """Evaluate an atomic expression (number or signal)."""
        expr = expr.strip()
        
        # Sized number (e.g., 4'b1010)
        match = self.RE_NUMBER.match(expr)
        if match:
            width = int(match.group(1))
            base = match.group(2)
            value = match.group(3).replace('_', '')
            
            if base == 'b':
                return value.zfill(width)[-width:]
            elif base == 'h':
                try:
                    int_val = int(value, 16)
                    return bin(int_val)[2:].zfill(width)[-width:]
                except:
                    return 'x' * width
            elif base == 'd':
                try:
                    int_val = int(value)
                    return bin(int_val)[2:].zfill(width)[-width:]
                except:
                    return 'x' * width
        
        # Decimal number
        if expr.isdigit():
            int_val = int(expr)
            if int_val == 0:
                return '0'
            return bin(int_val)[2:]
        
        # Signal reference
        if expr in self.signal_values:
            return self.signal_values[expr]
        
        # Try partial match (for hierarchical signals)
        for sig, val in self.signal_values.items():
            if sig.endswith('.' + expr) or sig.endswith('_' + expr):
                return val
        
        return None
    
    def _is_true(self, val: str) -> bool:
        """Check if a value is logically true (non-zero)."""
        if not val or 'x' in val.lower() or 'z' in val.lower():
            return False
        return '1' in val
    
    def _bitwise_not(self, val: str) -> str:
        """Bitwise NOT operation."""
        result = []
        for c in val:
            if c == '0':
                result.append('1')
            elif c == '1':
                result.append('0')
            else:
                result.append('x')
        return ''.join(result)
    
    def _apply_binary_op(self, left: str, right: str, op: str) -> Optional[str]:
        """Apply a binary operator."""
        # Pad to same length
        max_len = max(len(left), len(right))
        left = left.zfill(max_len)
        right = right.zfill(max_len)
        
        if op == '&' or op == '&&':
            result = []
            for l, r in zip(left, right):
                if l == '0' or r == '0':
                    result.append('0')
                elif l == '1' and r == '1':
                    result.append('1')
                else:
                    result.append('x')
            return ''.join(result)
        
        if op == '|' or op == '||':
            result = []
            for l, r in zip(left, right):
                if l == '1' or r == '1':
                    result.append('1')
                elif l == '0' and r == '0':
                    result.append('0')
                else:
                    result.append('x')
            return ''.join(result)
        
        if op == '^':
            result = []
            for l, r in zip(left, right):
                if l in 'xXzZ' or r in 'xXzZ':
                    result.append('x')
                elif l == r:
                    result.append('0')
                else:
                    result.append('1')
            return ''.join(result)
        
        if op == '==' or op == '===':
            return '1' if left == right else '0'
        
        if op == '!=' or op == '!==':
            return '1' if left != right else '0'
        
        # For arithmetic, try to convert to integers
        left_int = parse_binary_value(left)
        right_int = parse_binary_value(right)
        
        if left_int is not None and right_int is not None:
            if op == '+':
                return bin(left_int + right_int)[2:]
            if op == '-':
                result = left_int - right_int
                if result < 0:
                    return bin(result & ((1 << max_len) - 1))[2:].zfill(max_len)
                return bin(result)[2:]
            if op == '*':
                return bin(left_int * right_int)[2:]
            if op == '<':
                return '1' if left_int < right_int else '0'
            if op == '>':
                return '1' if left_int > right_int else '0'
            if op == '<=':
                return '1' if left_int <= right_int else '0'
            if op == '>=':
                return '1' if left_int >= right_int else '0'
            if op == '<<':
                return bin(left_int << right_int)[2:]
            if op == '>>':
                return bin(left_int >> right_int)[2:]
        
        return None


class BackwardSlicer:
    """
    Backward slicing engine for building causal DAG.
    
    Performs backward traversal from counterexample endpoint,
    using counterfactual evaluation to determine causality.
    """
    
    def __init__(self, 
                 verilog_parser: VerilogParser,
                 waveform: CycleAlignedWaveform,
                 max_depth: int = 20,
                 max_nodes: int = 200):
        """
        Initialize backward slicer.
        
        Args:
            verilog_parser: Parsed Verilog RTL
            waveform: Cycle-aligned waveform data
            max_depth: Maximum depth to traverse
            max_nodes: Maximum number of nodes in DAG
        """
        self.parser = verilog_parser
        self.waveform = waveform
        self.max_depth = max_depth
        self.max_nodes = max_nodes
        
        # Build static dependency graph
        self.dep_graph = verilog_parser.build_dependency_graph()
        
        # DAG construction state
        self.nodes: Dict[str, CausalNode] = {}
        self.edges: List[CausalEdge] = []
        self.visited: Set[str] = set()
        
        # SVA implication pattern for detecting trigger cycles
        self._re_sva_implication = re.compile(
            r'^(.+?)\s*\|->\s*(?:##\[\d+:\d+\]\s*)?(.+)$',
            re.DOTALL
        )
        
        # Statistics
        self.stats = {
            "nodes_created": 0,
            "edges_created": 0,
            "max_depth_reached": False,
            "max_nodes_reached": False,
            "undetermined_nodes": 0,
            "sva_trigger_cycle": None
        }
    
    def _make_node_id(self, signal: str, cycle: int, value: str) -> str:
        """Create a unique node ID."""
        content = f"{signal}@{cycle}={value}"
        return hashlib.md5(content.encode()).hexdigest()[:12]
    
    def _extract_base_signal_name(self, signal: str) -> str:
        """Extract base signal name without hierarchy prefix and width annotation."""
        # Remove width annotation like " [1:0]" or "[31:0]"
        base = re.sub(r'\s*\[\d+:\d+\]$', '', signal)
        # Get the last part after '.' (base signal name)
        base = base.split('.')[-1]
        return base
    
    def _extract_module_hierarchy(self, signal: str) -> str:
        """Extract module hierarchy prefix from signal name."""
        # Remove width annotation first
        clean = re.sub(r'\s*\[\d+:\d+\]$', '', signal)
        parts = clean.split('.')
        if len(parts) > 1:
            return '.'.join(parts[:-1])
        return ''
    
    def _find_sva_trigger_cycle(self, endpoint_signal: str, max_cycle: int) -> Optional[int]:
        """
        Find the cycle where an SVA implication's antecedent becomes true.
        
        For SVA property: antecedent |-> ##[n:m] consequent
        We need to find the cycle where antecedent is true, as that's where
        the assertion is "triggered".
        
        Args:
            endpoint_signal: The assertion signal name
            max_cycle: Maximum cycle to search
            
        Returns:
            Trigger cycle number, or None if not found
        """
        base_signal = self._extract_base_signal_name(endpoint_signal)
        hierarchy = self._extract_module_hierarchy(endpoint_signal)
        
        # Get the SVA expression from dependencies
        deps = self.parser.get_dependencies_for_signal(base_signal)
        if not deps:
            return None
        
        # Find the first dep that has an SVA implication expression
        sva_expr = None
        for dep in deps:
            if '|->' in dep.expression:
                sva_expr = dep.expression
                break
        
        if not sva_expr:
            return None
        
        # Extract antecedent
        match = self._re_sva_implication.match(sva_expr)
        if not match:
            return None
        
        antecedent = match.group(1).strip()
        
        # Get signal sources from antecedent
        sources = set()
        for dep in deps:
            sources.add(dep.source)
        
        # Scan backwards from max_cycle to find where antecedent is true
        for cycle in range(max_cycle, -1, -1):
            # Build environment for this cycle
            env = {}
            for src in sources:
                full_src = f"{hierarchy}.{src}" if hierarchy else src
                val = self.waveform.get_signal_value(full_src, cycle)
                if val:
                    env[src] = val
            
            # Evaluate antecedent
            evaluator = ExpressionEvaluator(env)
            result = evaluator.evaluate(antecedent)
            
            if result and evaluator._is_true(result):
                return cycle
        
        return None
    
    def _get_or_create_node(self, signal: str, cycle: int, depth: int, 
                            parent_hierarchy: str = '') -> Optional[CausalNode]:
        """Get existing node or create a new one."""
        original_signal = signal
        value = self.waveform.get_signal_value(signal, cycle)
        
        if value is None:
            # Try to find with partial match
            matches = self.waveform.find_signal(signal, max_results=10)
            for match in matches:
                # More flexible matching: remove width suffix for comparison
                match_base = re.sub(r'\s*\[\d+:\d+\]$', '', match)
                if match_base.endswith('.' + signal) or match_base.endswith(signal):
                    value = self.waveform.get_signal_value(match, cycle)
                    if value is not None:
                        signal = match
                        break
        
        if value is None and parent_hierarchy:
            # Try with parent hierarchy prefix
            full_signal = f"{parent_hierarchy}.{signal}"
            value = self.waveform.get_signal_value(full_signal, cycle)
            if value is not None:
                signal = full_signal
            else:
                # Try partial match with hierarchy - more flexible matching
                matches = self.waveform.find_signal(signal, max_results=10)
                for match in matches:
                    match_base = re.sub(r'\s*\[\d+:\d+\]$', '', match)
                    if match_base.endswith('.' + signal):
                        value = self.waveform.get_signal_value(match, cycle)
                        if value is not None:
                            signal = match
                            break
        
        if value is None:
            value = 'x'  # Unknown value
        
        node_id = self._make_node_id(signal, cycle, value)
        
        if node_id in self.nodes:
            return self.nodes[node_id]
        
        if len(self.nodes) >= self.max_nodes:
            self.stats["max_nodes_reached"] = True
            return None
        
        # Get RTL context using base signal name (without hierarchy prefix)
        base_signal = self._extract_base_signal_name(signal)
        rtl_context = self.parser.get_rtl_context(base_signal)
        
        # If not found, try with original signal name
        if not rtl_context.get("found", False):
            rtl_context = self.parser.get_rtl_context(original_signal)
        
        node = CausalNode(
            id=node_id,
            signal=signal,
            cycle=cycle,
            value=value,
            rtl_refs=rtl_context.get("rtl_refs", []),
            rtl_context_missing=not rtl_context.get("found", False),
            depth=depth
        )
        
        self.nodes[node_id] = node
        self.stats["nodes_created"] += 1
        
        if node.rtl_context_missing:
            self.stats["undetermined_nodes"] += 1
        
        return node
    
    def _get_parent_cycle(self, dep: Dependency, target_cycle: int) -> int:
        """
        Determine the parent cycle based on dependency type.
        
        Args:
            dep: Dependency information
            target_cycle: Target signal's cycle
            
        Returns:
            Source signal's relevant cycle
        """
        if dep.dep_type == DependencyType.COMBINATIONAL:
            return target_cycle  # Same cycle for combinational
        elif dep.dep_type == DependencyType.ASSERTION:
            return target_cycle  # Assertions are combinational - same cycle
        elif dep.dep_type == DependencyType.SEQUENTIAL:
            return max(0, target_cycle - 1)  # Previous cycle for sequential
        else:
            return max(0, target_cycle - 1)  # Default to previous
    
    def _evaluate_counterfactual(self,
                                  target_signal: str,
                                  target_cycle: int,
                                  source_signal: str,
                                  source_cycle: int,
                                  dep: Dependency) -> Tuple[bool, float, List[Dict]]:
        """
        Evaluate if source causally affects target via counterfactual analysis.
        
        Args:
            target_signal: Target signal name
            target_cycle: Target cycle
            source_signal: Source signal name
            source_cycle: Source cycle
            dep: Dependency information
            
        Returns:
            (is_causal, contribution_score, change_examples)
        """
        # Get current values
        target_value = self.waveform.get_signal_value(target_signal, target_cycle)
        source_value = self.waveform.get_signal_value(source_signal, source_cycle)
        
        if target_value is None or source_value is None:
            return False, 0.0, []
        
        # Build environment for expression evaluation
        # Use base signal name for RTL lookup, but full name for waveform lookup
        env = {}
        target_base = self._extract_base_signal_name(target_signal)
        target_hierarchy = self._extract_module_hierarchy(target_signal)
        source_base = self._extract_base_signal_name(source_signal)
        
        sources = self.parser.get_signal_sources(target_base)
        for src, _ in sources:
            # Try to get value with hierarchy prefix
            found_value = False
            if target_hierarchy:
                full_src = f"{target_hierarchy}.{src}"
                val = self.waveform.get_signal_value(full_src, source_cycle)
                if val:
                    env[src] = val
                    found_value = True
            
            if not found_value:
                # Fallback to direct lookup
                val = self.waveform.get_signal_value(src, source_cycle)
                if val:
                    env[src] = val
                    found_value = True
            
            if not found_value:
                # Try to find signal in waveform with partial matching
                matches = self.waveform.find_signal(src, max_results=10)
                for match in matches:
                    # More flexible matching: check if src is contained after a '.'
                    # Handle cases like "philo4._ph0_io_out [1:0]" matching "_ph0_io_out"
                    match_base = re.sub(r'\s*\[\d+:\d+\]$', '', match)  # Remove width suffix
                    if match_base.endswith('.' + src):
                        val = self.waveform.get_signal_value(match, source_cycle)
                        if val:
                            env[src] = val
                            break
        
        # Evaluate original expression
        evaluator = ExpressionEvaluator(env)
        orig_result = evaluator.evaluate(dep.expression)
        
        if orig_result is None:
            # Can't evaluate expression, use simple toggle test
            return self._simple_toggle_test(source_signal, source_cycle, target_signal, target_cycle)
        
        # Perturbation: invert source value
        # Use dep.source as the key since that's what's stored in env
        # (env uses the source names from RTL dependencies, not from waveform)
        perturbed_env = env.copy()
        # Try dep.source first (e.g., "ph0.io_out"), then source_base (e.g., "io_out")
        if dep.source in perturbed_env:
            perturbed_env[dep.source] = invert_value(source_value)
        elif source_base in perturbed_env:
            perturbed_env[source_base] = invert_value(source_value)
        else:
            # Source not in env, try adding it
            perturbed_env[dep.source] = invert_value(source_value)
        
        perturbed_evaluator = ExpressionEvaluator(perturbed_env)
        perturbed_result = perturbed_evaluator.evaluate(dep.expression)
        
        if perturbed_result is None:
            return False, 0.0, []
        
        # Check if perturbation changes output
        is_causal = values_differ(orig_result, perturbed_result)
        
        # If simple inversion doesn't show causality, try smarter perturbation
        # For equality comparisons (X == const), try setting X to the const value
        if not is_causal and '==' in dep.expression:
            smart_result = self._try_smart_perturbation(
                dep, env, source_value, source_base, orig_result
            )
            if smart_result is not None:
                is_causal, perturbed_result = smart_result
        
        # Calculate contribution score
        if is_causal:
            # Count differing bits
            diff_bits = sum(1 for a, b in zip(orig_result.zfill(len(perturbed_result)), 
                                              perturbed_result.zfill(len(orig_result)))
                           if a != b and a not in 'xXzZ' and b not in 'xXzZ')
            total_bits = max(len(orig_result), len(perturbed_result))
            score = min(1.0, diff_bits / max(1, total_bits) + 0.5)  # Base 0.5 + bit diff
        else:
            score = 0.0
        
        # Build change examples
        examples = []
        if is_causal:
            examples.append({
                "type": "counterfactual",
                "source_original": source_value,
                "source_perturbed": invert_value(source_value),
                "target_original": orig_result,
                "target_perturbed": perturbed_result,
                "expression": dep.expression
            })
        
        return is_causal, score, examples
    
    def _try_smart_perturbation(self,
                                 dep: Dependency,
                                 env: Dict[str, str],
                                 source_value: str,
                                 source_base: str,
                                 orig_result: str) -> Optional[Tuple[bool, str]]:
        """
        Try smarter perturbation for equality comparisons.
        
        For expressions like (X == const), simple inversion may not change the result.
        Instead, try setting X to the const value to flip the comparison.
        
        Args:
            dep: Dependency information
            env: Current signal environment
            source_value: Current source signal value
            source_base: Base name of source signal
            orig_result: Original expression result
            
        Returns:
            (is_causal, perturbed_result) or None if cannot apply
        """
        expr = dep.expression
        
        # Pattern: source == literal
        # Try to extract the comparison value
        patterns = [
            # Match: signal == N'hX or signal == N'bX or signal == N'dX
            re.compile(rf'\b{re.escape(source_base)}\s*==\s*(\d+\'[bhd][0-9a-fA-F_]+)'),
            re.compile(rf'\b{re.escape(dep.source)}\s*==\s*(\d+\'[bhd][0-9a-fA-F_]+)'),
            # Match: N'hX == signal (reversed order)
            re.compile(rf'(\d+\'[bhd][0-9a-fA-F_]+)\s*==\s*{re.escape(source_base)}\b'),
            re.compile(rf'(\d+\'[bhd][0-9a-fA-F_]+)\s*==\s*{re.escape(dep.source)}\b'),
        ]
        
        target_value = None
        for pattern in patterns:
            match = pattern.search(expr)
            if match:
                # Parse the literal value
                lit = match.group(1)
                # Use ExpressionEvaluator to parse the literal
                evaluator = ExpressionEvaluator({})
                target_value = evaluator.evaluate(lit)
                break
        
        if target_value is None:
            return None
        
        # If current value already equals target, try a different value
        if source_value == target_value:
            # The comparison should be true, set to something different
            perturb_val = invert_value(target_value)
        else:
            # The comparison is false, set to target to make it true
            perturb_val = target_value
        
        # Apply perturbation
        perturbed_env = env.copy()
        if dep.source in perturbed_env:
            perturbed_env[dep.source] = perturb_val
        elif source_base in perturbed_env:
            perturbed_env[source_base] = perturb_val
        else:
            perturbed_env[dep.source] = perturb_val
        
        perturbed_evaluator = ExpressionEvaluator(perturbed_env)
        perturbed_result = perturbed_evaluator.evaluate(dep.expression)
        
        if perturbed_result is None:
            return None
        
        is_causal = values_differ(orig_result, perturbed_result)
        return (is_causal, perturbed_result)

    def _simple_toggle_test(self, 
                            source_signal: str, 
                            source_cycle: int,
                            target_signal: str,
                            target_cycle: int) -> Tuple[bool, float, List[Dict]]:
        """
        Simple toggle test: check if source change correlates with target change.
        
        Args:
            source_signal: Source signal name
            source_cycle: Source cycle
            target_signal: Target signal name
            target_cycle: Target cycle
            
        Returns:
            (is_causal, score, examples)
        """
        # Initialize values
        src_prev: Optional[str] = None
        src_curr: Optional[str] = None
        tgt_prev: Optional[str] = None
        tgt_curr: Optional[str] = None
        
        # Check if source changed in source_cycle
        if source_cycle > 0:
            src_prev = self.waveform.get_signal_value(source_signal, source_cycle - 1)
            src_curr = self.waveform.get_signal_value(source_signal, source_cycle)
            source_changed = values_differ(src_prev or 'x', src_curr or 'x')
        else:
            source_changed = False
        
        # Check if target changed in target_cycle
        if target_cycle > 0:
            tgt_prev = self.waveform.get_signal_value(target_signal, target_cycle - 1)
            tgt_curr = self.waveform.get_signal_value(target_signal, target_cycle)
            target_changed = values_differ(tgt_prev or 'x', tgt_curr or 'x')
        else:
            target_changed = False
        
        # Simple heuristic: if both changed, likely causal
        is_causal = source_changed and target_changed
        score = 0.7 if is_causal else 0.0
        
        examples = []
        if is_causal:
            examples.append({
                "type": "toggle_correlation",
                "source_before": src_prev,
                "source_after": src_curr,
                "target_before": tgt_prev,
                "target_after": tgt_curr
            })
        
        return is_causal, score, examples
    
    def _slice_node(self, node: CausalNode, depth: int):
        """
        Perform backward slicing from a node.
        
        Args:
            node: Current node to slice from
            depth: Current depth
        """
        if depth > self.max_depth:
            self.stats["max_depth_reached"] = True
            return
        
        if node.id in self.visited:
            return
        
        self.visited.add(node.id)
        
        # Extract base signal name and hierarchy for lookup
        base_signal = self._extract_base_signal_name(node.signal)
        parent_hierarchy = self._extract_module_hierarchy(node.signal)
        
        # Get dependencies from RTL
        deps = self.parser.get_dependencies_for_signal(base_signal)
        
        if not deps:
            # Check if we can find with the full name (no width annotation)
            clean_signal = re.sub(r'\s*\[\d+:\d+\]$', '', node.signal)
            deps = self.parser.get_dependencies_for_signal(clean_signal)
        
        if not deps:
            # No RTL dependencies found, mark as potential root
            node.is_root = True
            return
        
        for dep in deps:
            # Determine parent cycle
            parent_cycle = self._get_parent_cycle(dep, node.cycle)
            
            if parent_cycle < 0:
                continue
            
            # Check for self-dependency before creating node
            # A signal depending on itself in the same cycle is not valid causality
            source_base = self._extract_base_signal_name(dep.source)
            target_base = self._extract_base_signal_name(node.signal)
            
            # Skip if source and target are the same signal (avoid self-loops)
            if source_base == target_base:
                continue
            
            # Also check if full signal names match (with hierarchy)
            if dep.source == node.signal:
                continue
            
            # Check if the source (with hierarchy) matches node signal
            full_source = f"{parent_hierarchy}.{dep.source}" if parent_hierarchy else dep.source
            clean_node_signal = re.sub(r'\s*\[\d+:\d+\]$', '', node.signal)
            if full_source == clean_node_signal or full_source.endswith('.' + target_base):
                if dep.dep_type == DependencyType.COMBINATIONAL:
                    # Combinational self-dependency is not allowed
                    continue
            
            # Create parent node with hierarchy context
            parent_node = self._get_or_create_node(
                dep.source, parent_cycle, depth + 1, parent_hierarchy
            )
            if parent_node is None:
                continue
            
            # Final self-loop check using node IDs
            if parent_node.id == node.id:
                continue
            
            # Check for duplicate edges
            edge_key = (parent_node.id, node.id)
            if not hasattr(self, '_edge_set'):
                self._edge_set = set()
            if edge_key in self._edge_set:
                continue  # Skip duplicate edge
            
            # Evaluate causality using full signal names from nodes
            is_causal, score, examples = self._evaluate_counterfactual(
                node.signal, node.cycle,
                parent_node.signal, parent_cycle,
                dep
            )
            
            # For SVA assertions at trigger cycle, if antecedent is true,
            # all signals in antecedent are causally contributing
            is_sva_antecedent_signal = False
            if not is_causal and '|->' in dep.expression:
                # Check if this is an SVA trigger cycle (antecedent is true)
                trigger_cycle = self.stats.get("sva_trigger_cycle")
                if trigger_cycle is not None and node.cycle == trigger_cycle:
                    # At trigger cycle, all antecedent signals contribute to assertion
                    is_sva_antecedent_signal = True
                    is_causal = True
                    score = 0.85  # High score for antecedent signals
                    examples = [{
                        "type": "sva_antecedent",
                        "trigger_cycle": trigger_cycle,
                        "expression": dep.expression,
                        "signal": parent_node.signal,
                        "value": parent_node.value
                    }]
            
            if not is_causal and not node.rtl_context_missing:
                # Counterfactual did not show causality, but we may still want to track
                # this dependency based on RTL structure for deeper exploration
                # Use a lower score to indicate it's structural dependency only
                if depth < self.max_depth // 2:
                    # For shallow depths, still create edges for structural deps
                    # This helps build a more complete causal picture
                    is_causal = True
                    score = 0.3  # Lower score for structural-only dependency
                    examples = [{
                        "type": "structural",
                        "reason": "RTL dependency exists but counterfactual not conclusive"
                    }]
                else:
                    # For deeper levels, skip to avoid graph explosion
                    continue
            
            # Determine contribution type
            if dep.dep_type == DependencyType.SEQUENTIAL:
                contrib_type = ContributionType.STATE
            elif dep.condition:
                contrib_type = ContributionType.CONDITIONAL
            elif examples and examples[0].get("type") == "toggle_correlation":
                contrib_type = ContributionType.TOGGLE
            else:
                contrib_type = ContributionType.EXPR_EVAL
            
            # Create edge
            edge = CausalEdge(
                src_node_id=parent_node.id,
                dst_node_id=node.id,
                reason=f"{dep.source} affects {node.signal} via {dep.dep_type.value}",
                contribution_type=contrib_type,
                contribution_score=score,
                evidence={
                    "file": dep.file_path,
                    "lines": [dep.line_start, dep.line_end],
                    "code_snippet": dep.code_snippet,
                    "expression": dep.expression,
                    "condition": dep.condition
                },
                change_examples=examples
            )
            
            self.edges.append(edge)
            self._edge_set.add(edge_key)  # Track edge to prevent duplicates
            self.stats["edges_created"] += 1
            
            # Update parent node suspect score
            parent_node.suspect_score = max(parent_node.suspect_score, score * 0.9)
            
            # Recurse
            self._slice_node(parent_node, depth + 1)
    
    def slice_from_endpoint(self, 
                            endpoint_signal: str, 
                            endpoint_cycle: int) -> Tuple[Dict[str, CausalNode], List[CausalEdge]]:
        """
        Perform backward slicing starting from endpoint.
        
        For SVA assertions with implication (|->), automatically finds
        the cycle where the antecedent becomes true (trigger point).
        
        Args:
            endpoint_signal: Signal that triggered the counterexample
            endpoint_cycle: Cycle when counterexample was triggered
            
        Returns:
            (nodes dict, edges list)
        """
        # Reset state
        self.nodes = {}
        self.edges = []
        self.visited = set()
        self._edge_set = set()  # Track edges to prevent duplicates
        self.stats = {
            "nodes_created": 0,
            "edges_created": 0,
            "max_depth_reached": False,
            "max_nodes_reached": False,
            "undetermined_nodes": 0,
            "sva_trigger_cycle": None
        }
        
        # For SVA assertions, try to find the actual trigger cycle
        # where the antecedent became true
        trigger_cycle = self._find_sva_trigger_cycle(endpoint_signal, endpoint_cycle)
        if trigger_cycle is not None:
            self.stats["sva_trigger_cycle"] = trigger_cycle
            # Use trigger cycle as the analysis point for the endpoint
            endpoint_cycle = trigger_cycle
        
        # Create endpoint node
        endpoint_node = self._get_or_create_node(endpoint_signal, endpoint_cycle, 0)
        if endpoint_node is None:
            return {}, []
        
        endpoint_node.is_endpoint = True
        endpoint_node.suspect_score = 1.0
        
        # Perform backward slicing
        self._slice_node(endpoint_node, 0)
        
        # Mark leaf nodes as roots
        nodes_with_incoming = set(e.dst_node_id for e in self.edges)
        for node in self.nodes.values():
            if node.id not in nodes_with_incoming and not node.is_endpoint:
                node.is_root = True
        
        return self.nodes, self.edges
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get slicing statistics."""
        return self.stats.copy()
