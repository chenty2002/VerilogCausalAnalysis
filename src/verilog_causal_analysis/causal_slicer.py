"""
Backward Causal Slicing Engine for Counterexample Analysis.

Performs backward slicing from counterexample endpoint to build
a causal DAG with counterfactual evaluation at expression level.
"""

import re
import hashlib
from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple, Optional, Any
from enum import Enum

from .verilog_parser import VerilogParser, Dependency, DependencyType
from .cycle_waveform import CycleAlignedWaveform, parse_binary_value, invert_value, values_differ
from .contribution import (
    ContributionEvidence,
    LegacyContributionEvidence,
    adapt_legacy_contribution,
    evaluate_interventions,
    generate_interventions,
    route_contribution,
    structural_evidence,
    toggle_evidence,
)
from .local_search import (
    FrontierItem,
    FrontierScheduler,
    POLICY_REGISTRY,
    ScoreFeatures,
    SearchPolicy,
    frontier_priority,
    path_support,
    score_features,
)


class ContributionType(Enum):
    """Type of causal contribution."""
    EXPR_EVAL = "expr_eval"        # Expression evaluation verified causality
    TOGGLE = "toggle"              # Signal toggle correlation
    STATE = "state"                # State machine transition
    DIRECT = "direct"              # Direct assignment
    CONDITIONAL = "conditional"    # Condition branch taken
    UNKNOWN = "unknown"            # Could not determine


@dataclass
class CausalNode:
    """A node in the causal DAG: (signal, cycle, value) tuple."""
    id: str
    signal: str
    cycle: int
    value: str
    suspect_score: float = 0.0
    rtl_refs: List[Dict] = field(default_factory=list)
    rtl_context_missing: bool = False
    is_root: bool = False
    is_endpoint: bool = False
    depth: int = 0
    identity_strength: str = "unresolved"

    def __hash__(self): return hash(self.id)
    def __eq__(self, other): return isinstance(other, CausalNode) and self.id == other.id

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
            "depth": self.depth,
            "identity_strength": self.identity_strength,
        }


@dataclass
class CausalEdge:
    """An edge representing direct causality in the DAG."""
    src_node_id: str
    dst_node_id: str
    reason: str
    contribution_type: ContributionType
    contribution_score: float
    evidence: Dict[str, Any] = field(default_factory=dict)
    change_examples: List[Dict] = field(default_factory=list)
    contribution_evidence: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        row = {
            "src_node_id": self.src_node_id,
            "dst_node_id": self.dst_node_id,
            "reason": self.reason,
            "contribution_type": self.contribution_type.value,
            "contribution_score": self.contribution_score,
            "evidence": self.evidence,
            "change_examples": self.change_examples
        }
        if self.contribution_evidence is not None:
            row["contribution_evidence"] = self.contribution_evidence
        return row


# Operator precedence table (higher = binds tighter)
# NOTE: Only binary operators belong here. Unary !, ~ are handled separately.
_PRECEDENCE = {
    '||': 1, '|': 2, '&&': 3, '&': 4,
    '^': 5, '~^': 5, '^~': 5,
    '==': 6, '!=': 6, '===': 6, '!==': 6,
    '<': 7, '>': 7, '<=': 7, '>=': 7,
    '<<': 8, '>>': 8, '<<<': 8, '>>>': 8,
    '+': 9, '-': 9, '*': 10, '/': 10, '%': 10,
}

# Signal names to ignore during causal analysis (assertion-helper signals)
_IGNORED_SIGNALS = frozenset({'hasBeenReset', 'hasBeenResetReg', 'reset'})


class ExpressionEvaluator:
    """Evaluates Verilog expressions for counterfactual analysis."""
    
    RE_NUMBER = re.compile(r"(\d+)'([bhd])([0-9a-fA-F_xXzZ]+)")
    RE_DECIMAL = re.compile(r'\b(\d+)\b')
    RE_TERNARY = re.compile(r'(.+?)\s*\?\s*(.+?)\s*:\s*(.+)')  
    RE_SVA_IMPLICATION = re.compile(r'^(.+?)\s*\|->\s*(?:##\[\d+:\d+\]\s*)?(.+)$', re.DOTALL)

    def __init__(self, signal_values: Dict[str, str]):
        """Initialize with signal_name -> binary_value mapping."""
        self.signal_values = signal_values

    def evaluate(self, expr: str) -> Optional[str]:
        """Evaluate Verilog expression; returns binary string or None."""
        expr = expr.strip()
        if not expr:
            return None
        try:
            sva_match = self.RE_SVA_IMPLICATION.match(expr)
            if sva_match:
                return self._eval_sva_implication(sva_match.group(1), sva_match.group(2))
            return self._eval_expr(expr)
        except Exception:
            return None
    
    def _eval_sva_implication(self, antecedent: str, consequent: str) -> Optional[str]:
        """Evaluate SVA implication: if antecedent true, check consequent; else vacuously true."""
        ante_val = self._eval_expr(antecedent.strip())
        if ante_val is None:
            return None
        if not self._is_true(ante_val):
            return '1'  # Vacuously true
        cons_val = self._eval_expr(consequent.strip())
        return cons_val if cons_val is not None else '0'
    
    def _eval_expr(self, expr: str) -> Optional[str]:
        """Evaluate expression recursively."""
        expr = expr.strip()
        if not expr:
            return None
        
        # Handle Verilog concatenation: {a, b, c}
        if expr.startswith('{') and expr.endswith('}'):
            # Verify the outer braces actually match (not {a} & {b})
            depth = 0
            for ci, cc in enumerate(expr):
                if cc == '{':
                    depth += 1
                elif cc == '}':
                    depth -= 1
                if depth == 0 and ci < len(expr) - 1:
                    break  # Outer { closes before end — not a single concat
            else:
                parts = self._split_concat_parts(expr[1:-1])
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
        
        # Binary operators first (find lowest precedence outside parens)
        # This must come before unary checks so that "!a & b" correctly
        # splits at '&' rather than treating '!' as consuming the whole expr.
        op_pos, op = self._find_lowest_op(expr)
        if op is not None and op_pos > 0:
            left = self._eval_expr(expr[:op_pos])
            right = self._eval_expr(expr[op_pos + len(op):])
            if left is None or right is None:
                return None
            return self._apply_binary_op(left, right, op)
        
        # Handle unary operators: !, ~, -
        if expr.startswith('!'):
            val = self._eval_expr(expr[1:])
            if val is None:
                return None
            return '1' if not self._is_true(val) else '0'
        
        if expr.startswith('~'):
            val = self._eval_expr(expr[1:])
            return self._bitwise_not(val) if val else None
        
        if expr.startswith('-') and len(expr) > 1:
            val = self._eval_expr(expr[1:])
            if val is None:
                return None
            int_val = parse_binary_value(val)
            if int_val is not None:
                neg_val = (-int_val) & ((1 << len(val)) - 1)
                return bin(neg_val)[2:].zfill(len(val))
            return None
        
        # Reduction operators: &x, |x, ^x (reduce multi-bit to single bit)
        if len(expr) > 1:
            if expr[0] == '&' and expr[1] != '&':
                val = self._eval_expr(expr[1:])
                return ('1' if all(c == '1' for c in val) else '0') if val else None
            if expr[0] == '|' and expr[1] != '|':
                val = self._eval_expr(expr[1:])
                return ('1' if '1' in val else '0') if val else None
            if expr[0] == '^':
                val = self._eval_expr(expr[1:])
                return ('1' if val.count('1') % 2 else '0') if val else None
        
        return self._eval_atom(expr)
    
    def _split_concat_parts(self, inner: str) -> List[str]:
        """Split concatenation by commas, respecting nesting."""
        parts, depth, current = [], 0, []
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
        """Parse ternary operator (cond ? then : else)."""
        depth, q_pos, c_pos = 0, -1, -1
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
            return expr[:q_pos].strip(), expr[q_pos+1:c_pos].strip(), expr[c_pos+1:].strip()
        return None
    
    def _find_lowest_op(self, expr: str) -> Tuple[int, Optional[str]]:
        """Find lowest precedence binary operator outside parentheses/brackets/braces."""
        depth = 0
        lowest = (999, -1, None)  # (prec, pos, op)
        sorted_ops = sorted(_PRECEDENCE.keys(), key=len, reverse=True)
        i = 0
        while i < len(expr):
            c = expr[i]
            if c in '([{':
                depth += 1
                i += 1
            elif c in ')]}':  
                depth -= 1
                i += 1
            elif depth == 0:
                # Check for operators (longest match first)
                matched = False
                for op in sorted_ops:
                    if expr[i:i+len(op)] == op:
                        if _PRECEDENCE[op] <= lowest[0]:
                            lowest = (_PRECEDENCE[op], i, op)
                        i += len(op)  # Skip past operator to avoid substring matches
                        matched = True
                        break
                if not matched:
                    i += 1
            else:
                i += 1
        return lowest[1], lowest[2]
    
    def _eval_atom(self, expr: str) -> Optional[str]:
        """Evaluate atomic expression (number or signal)."""
        expr = expr.strip()
        
        # Sized number (e.g., 4'b1010, 8'hFF)
        match = self.RE_NUMBER.match(expr)
        if match:
            width, base, value = int(match.group(1)), match.group(2), match.group(3).replace('_', '')
            try:
                int_val = int(value, {'b': 2, 'h': 16, 'd': 10}[base])
                return bin(int_val)[2:].zfill(width)[-width:]
            except:
                return 'x' * width
        
        # Decimal number
        if expr.isdigit():
            n = int(expr)
            return bin(n)[2:] if n else '0'
        
        # Direct signal lookup
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
        """Bitwise NOT: 0->1, 1->0, x->x."""
        return val.translate(str.maketrans('01', '10'))
    
    def _apply_binary_op(self, left: str, right: str, op: str) -> Optional[str]:
        """Apply binary operator to two values."""
        max_len = max(len(left), len(right))
        left, right = left.zfill(max_len), right.zfill(max_len)
        
        # Logical operators (always return single-bit '0' or '1')
        if op == '&&':
            return '1' if self._is_true(left) and self._is_true(right) else '0'
        
        if op == '||':
            return '1' if self._is_true(left) or self._is_true(right) else '0'
        
        # Bitwise AND
        if op == '&':
            result = []
            for l, r in zip(left, right):
                if l == '0' or r == '0':
                    result.append('0')
                elif l == '1' and r == '1':
                    result.append('1')
                else:
                    result.append('x')
            return ''.join(result)
        
        # Bitwise OR
        if op == '|':
            result = []
            for l, r in zip(left, right):
                if l == '1' or r == '1':
                    result.append('1')
                elif l == '0' and r == '0':
                    result.append('0')
                else:
                    result.append('x')
            return ''.join(result)
        
        # Bitwise XOR
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
        
        # Bitwise XNOR
        if op in ('~^', '^~'):
            result = []
            for l, r in zip(left, right):
                if l in 'xXzZ' or r in 'xXzZ':
                    result.append('x')
                elif l == r:
                    result.append('1')
                else:
                    result.append('0')
            return ''.join(result)
        
        if op == '==' or op == '===':
            return '1' if left == right else '0'
        
        if op == '!=' or op == '!==':
            return '1' if left != right else '0'
        
        # Arithmetic and shift (requires integer conversion)
        left_int, right_int = parse_binary_value(left), parse_binary_value(right)
        if left_int is not None and right_int is not None:
            mask = (1 << max_len) - 1
            ops = {
                '+': lambda: bin((left_int + right_int) & mask)[2:].zfill(max_len),
                '-': lambda: bin((left_int - right_int) & mask)[2:].zfill(max_len),
                '*': lambda: bin((left_int * right_int) & mask)[2:].zfill(max_len),
                '/': lambda: bin(left_int // right_int)[2:].zfill(max_len) if right_int != 0 else None,
                '%': lambda: bin(left_int % right_int)[2:].zfill(max_len) if right_int != 0 else None,
                '<': lambda: '1' if left_int < right_int else '0',
                '>': lambda: '1' if left_int > right_int else '0',
                '<=': lambda: '1' if left_int <= right_int else '0',
                '>=': lambda: '1' if left_int >= right_int else '0',
                '<<': lambda: bin((left_int << right_int) & mask)[2:].zfill(max_len),
                '>>': lambda: bin(left_int >> right_int)[2:].zfill(max_len),
                '<<<': lambda: bin((left_int << right_int) & mask)[2:].zfill(max_len),
                '>>>': lambda: bin(left_int >> right_int)[2:].zfill(max_len),
            }
            if op in ops:
                result = ops[op]()
                return result
        return None


@dataclass(frozen=True)
class CompiledExpression:
    """Parsed expression tree reusable across counterfactual environments."""

    expression: str
    root: Any

    @classmethod
    def compile(cls, expression: str) -> "CompiledExpression":
        parser = ExpressionEvaluator({})
        text = expression.strip()
        if not text:
            return cls(expression=expression, root=("invalid",))
        sva_match = parser.RE_SVA_IMPLICATION.match(text)
        if sva_match:
            root = (
                "sva",
                cls._compile_expr(parser, sva_match.group(1)),
                cls._compile_expr(parser, sva_match.group(2)),
            )
        else:
            root = cls._compile_expr(parser, text)
        return cls(expression=expression, root=root)

    @classmethod
    def _compile_expr(
        cls,
        parser: ExpressionEvaluator,
        expression: str,
    ) -> Any:
        expr = expression.strip()
        if not expr:
            return ("invalid",)

        if expr.startswith("{") and expr.endswith("}"):
            depth = 0
            for index, char in enumerate(expr):
                if char == "{":
                    depth += 1
                elif char == "}":
                    depth -= 1
                if depth == 0 and index < len(expr) - 1:
                    break
            else:
                parts = parser._split_concat_parts(expr[1:-1])
                if parts:
                    return (
                        "concat",
                        tuple(cls._compile_expr(parser, part) for part in parts),
                    )

        if expr.startswith("(") and expr.endswith(")"):
            depth = 0
            for index, char in enumerate(expr):
                if char == "(":
                    depth += 1
                elif char == ")":
                    depth -= 1
                if depth == 0 and index < len(expr) - 1:
                    break
            else:
                return cls._compile_expr(parser, expr[1:-1])

        ternary = parser._parse_ternary(expr)
        if ternary:
            condition, then_expr, else_expr = ternary
            return (
                "ternary",
                cls._compile_expr(parser, condition),
                cls._compile_expr(parser, then_expr),
                cls._compile_expr(parser, else_expr),
            )

        op_pos, operator = parser._find_lowest_op(expr)
        if operator is not None and op_pos > 0:
            return (
                "binary",
                operator,
                cls._compile_expr(parser, expr[:op_pos]),
                cls._compile_expr(parser, expr[op_pos + len(operator):]),
            )

        if expr.startswith("!"):
            return ("unary", "!", cls._compile_expr(parser, expr[1:]))
        if expr.startswith("~"):
            return ("unary", "~", cls._compile_expr(parser, expr[1:]))
        if expr.startswith("-") and len(expr) > 1:
            return ("unary", "-", cls._compile_expr(parser, expr[1:]))
        if len(expr) > 1 and expr[0] in "&|^":
            if expr[0] not in "&|" or expr[1] != expr[0]:
                return (
                    "reduction",
                    expr[0],
                    cls._compile_expr(parser, expr[1:]),
                )

        if parser.RE_NUMBER.match(expr) or expr.isdigit():
            return ("literal", parser._eval_atom(expr))
        return ("signal", expr)

    def evaluate(self, signal_values: Dict[str, str]) -> Optional[str]:
        evaluator = ExpressionEvaluator(signal_values)
        try:
            return self._evaluate_node(self.root, evaluator)
        except Exception:
            return None

    @classmethod
    def _evaluate_node(
        cls,
        node: Any,
        evaluator: ExpressionEvaluator,
    ) -> Optional[str]:
        kind = node[0]
        if kind == "invalid":
            return None
        if kind == "literal":
            return node[1]
        if kind == "signal":
            return evaluator._eval_atom(node[1])
        if kind == "concat":
            values = [
                cls._evaluate_node(child, evaluator)
                for child in node[1]
            ]
            return None if any(value is None for value in values) else "".join(values)
        if kind == "ternary":
            condition = cls._evaluate_node(node[1], evaluator)
            if condition is None:
                return None
            branch = node[2] if evaluator._is_true(condition) else node[3]
            return cls._evaluate_node(branch, evaluator)
        if kind == "binary":
            left = cls._evaluate_node(node[2], evaluator)
            right = cls._evaluate_node(node[3], evaluator)
            if left is None or right is None:
                return None
            return evaluator._apply_binary_op(left, right, node[1])
        if kind == "unary":
            value = cls._evaluate_node(node[2], evaluator)
            if value is None:
                return None
            if node[1] == "!":
                return "1" if not evaluator._is_true(value) else "0"
            if node[1] == "~":
                return evaluator._bitwise_not(value)
            int_value = parse_binary_value(value)
            if int_value is None:
                return None
            negated = (-int_value) & ((1 << len(value)) - 1)
            return bin(negated)[2:].zfill(len(value))
        if kind == "reduction":
            value = cls._evaluate_node(node[2], evaluator)
            if value is None:
                return None
            if node[1] == "&":
                return "1" if all(char == "1" for char in value) else "0"
            if node[1] == "|":
                return "1" if "1" in value else "0"
            return "1" if value.count("1") % 2 else "0"
        if kind == "sva":
            antecedent = cls._evaluate_node(node[1], evaluator)
            if antecedent is None:
                return None
            if not evaluator._is_true(antecedent):
                return "1"
            consequent = cls._evaluate_node(node[2], evaluator)
            return consequent if consequent is not None else "0"
        return None


@dataclass(frozen=True)
class StatementEvaluationKey:
    """Identity of one target statement sampled at one target cycle."""

    file_path: str
    line_start: int
    line_end: int
    module_name: str
    target: str
    target_signal: str
    target_cycle: int
    expression: str
    condition: str


@dataclass(frozen=True)
class _StatementEvaluationContext:
    environment: Dict[str, str]
    source_values: Dict[str, str]
    compiled: CompiledExpression
    original_result: Optional[str]


@dataclass
class CandidateExpansion:
    dep: Dependency
    parent_cycle: int
    parent_node: CausalNode
    parent_is_new: bool
    is_causal: bool
    weak_candidate: bool
    contribution_score: float
    contribution_evidence: ContributionEvidence | LegacyContributionEvidence
    change_examples: List[Dict[str, Any]]
    local_score: float
    score_features: Optional[ScoreFeatures]
    edge: Optional[CausalEdge] = None


class BackwardSlicer:
    """Backward slicing engine for building causal DAG.
    
    Performs backward traversal from counterexample endpoint,
    using counterfactual evaluation to determine causality.
    """

    def __init__(self, 
                 verilog_parser: VerilogParser, 
                 waveform: CycleAlignedWaveform,
                 max_depth: int = 20, 
                 max_nodes: int = 200,
                 *,
                 exact_sva_trigger_cycle: Optional[int] = None,
                 dependency_provider: Optional[Any] = None,
                 search_policy: Optional[SearchPolicy | str] = None,
                 contribution_evaluator: Optional[Any] = None,
                 max_intervention_evaluations: Optional[int] = None,
                 ):
        """Initialize backward slicer with RTL parser and waveform data."""
        self.parser = verilog_parser
        self.dependency_provider = dependency_provider or verilog_parser
        self.instance_local_dependencies = dependency_provider is not None
        self.waveform = waveform
        self.max_depth = max_depth
        self.max_nodes = max_nodes
        # structural.2 derives deterministic work limits from max_nodes.  In
        # particular, none of these limits expands when max_depth changes.
        self.weak_edge_budget = max(0, (max_nodes - 1) // 4)
        self.weak_beam_width = 2
        self.candidate_evaluation_budget = max_nodes * 8
        self.temporal_lookback_budget = max(64, max_nodes * 8)
        self.temporal_value_budget = max(256, max_nodes * 32)
        self.exact_sva_trigger_cycle = exact_sva_trigger_cycle
        if isinstance(search_policy, str):
            try:
                search_policy = POLICY_REGISTRY[search_policy]
            except KeyError as error:
                raise ValueError(f"unknown search policy {search_policy!r}") from error
        self.search_policy = search_policy
        self.contribution_evaluator = contribution_evaluator
        self.max_intervention_evaluations = (
            max_nodes * 32
            if max_intervention_evaluations is None
            else max_intervention_evaluations
        )
        if self.max_intervention_evaluations < 0:
            raise ValueError("max_intervention_evaluations must be non-negative")
        self.dep_graph = verilog_parser.build_dependency_graph()
        self._signal_sources_cache: Dict[str, List[str]] = {}
        self._statement_evaluation_cache: Dict[
            StatementEvaluationKey, _StatementEvaluationContext
        ] = {}
        self._compiled_expression_cache: Dict[str, CompiledExpression] = {}
        self._equality_literal_cache: Dict[
            Tuple[str, str, str], Optional[str]
        ] = {}
        
        # DAG state
        self.nodes: Dict[str, CausalNode] = {}
        self.edges: List[CausalEdge] = []
        self.visited: Set[str] = set()
        self._edge_set: Set[Tuple[str, str]] = set()
        
        # SVA pattern for trigger cycle detection
        self._re_sva_implication = re.compile(r'^(.+?)\s*\|->\s*(?:##\[\d+:\d+\]\s*)?(.+)$', re.DOTALL)
        # SVA time window pattern: ##[min:max] or ##N
        self._re_sva_time_window = re.compile(r'##\[(\d+):(\d+)\]|##(\d+)')
        
        self.stats = self._new_stats()

    def _new_stats(self) -> Dict[str, Any]:
        return {
            "nodes_created": 0,
            "edges_created": 0,
            "max_depth_reached": False,
            "max_nodes_reached": False,
            "candidate_evaluations": 0,
            "candidate_evaluation_budget": self.candidate_evaluation_budget,
            "candidate_evaluation_budget_reached": False,
            "intervention_evaluations": 0,
            "intervention_evaluation_budget": self.max_intervention_evaluations,
            "intervention_evaluation_budget_reached": False,
            "expanded_nodes": 0,
            "exploit_expansions": 0,
            "explore_expansions": 0,
            "statement_evaluations": 0,
            "statement_environment_builds": 0,
            "compiled_expression_hits": 0,
            "compiled_expression_misses": 0,
            "rejected_candidates": 0,
            "weak_edges_admitted": 0,
            "weak_edge_budget": self.weak_edge_budget,
            "weak_beam_width": self.weak_beam_width,
            "undetermined_nodes": 0,
            "temporal_lookback_budget": self.temporal_lookback_budget,
            "temporal_value_budget": self.temporal_value_budget,
            "temporal_cycles_evaluated": 0,
            "temporal_values_loaded": 0,
            "temporal_work_budget_reached": False,
            "sva_trigger_evidence": "not_applicable",
            "sva_exact_trigger_missing": False,
            "sva_trigger_cycle": None,
            "sva_time_window": None,  # (min_delay, max_delay) if SVA has time window
            "sva_window_end_cycle": None,  # The cycle when assertion failed (end of window)
            "sva_consequent_signals": None,  # Signals in the consequent part
            "exact_instance_waveform_misses": [],
        }
    
    @staticmethod
    def _make_node_id(signal: str, cycle: int, value: str) -> str:
        """Create unique node ID from signal@cycle=value."""
        return hashlib.md5(f"{signal}@{cycle}={value}".encode()).hexdigest()[:12]
    
    @staticmethod
    def _extract_base_signal_name(signal: str) -> str:
        """Extract base signal name (no hierarchy/width)."""
        return re.sub(r'\s*\[\d+:\d+\]$', '', signal).split('.')[-1]
    
    @staticmethod
    def _extract_module_hierarchy(signal: str) -> str:
        """Extract module hierarchy prefix."""
        clean = re.sub(r'\s*\[\d+:\d+\]$', '', signal)
        parts = clean.split('.')
        if len(parts) > 1:
            return '.'.join(parts[:-1])
        return ''

    def _infer_module_name(self, signal: str, hierarchy: str = '') -> Optional[str]:
        """Infer RTL module name for a waveform signal when parser supports it."""
        infer = getattr(
            self.dependency_provider, "infer_module_from_signal", None
        )
        if callable(infer):
            return infer(signal, hierarchy=hierarchy)
        return None

    def _get_signal_sources_cached(self, signal_name: str, module_name: Optional[str] = None) -> List[str]:
        """Get cached source signals for a target signal name."""
        cache_key = f"{module_name or ''}:{signal_name}"
        if cache_key in self._signal_sources_cache:
            return self._signal_sources_cache[cache_key]

        graph_key = f"{module_name}.{signal_name}" if module_name else signal_name
        sources = self.dep_graph.get(graph_key)
        if sources is None:
            sources = self.dep_graph.get(signal_name)
        if sources is None:
            source_list = [s for s, _ in self.parser.get_signal_sources(signal_name, module_name)]
        else:
            if module_name:
                source_list = [
                    s for s, _, dep in sources
                    if not getattr(dep, "module_name", None) or dep.module_name == module_name
                ]
            else:
                source_list = [s for s, _, _ in sources]

        # De-duplicate while preserving order
        seen: Set[str] = set()
        deduped: List[str] = []
        for src in source_list:
            if src not in seen:
                seen.add(src)
                deduped.append(src)

        self._signal_sources_cache[cache_key] = deduped
        return deduped

    def _resolve_signal_value(self,
                              signal: str,
                              cycle: int,
                              hierarchy: str = '',
                              match_cache: Optional[Dict[str, List[str]]] = None,
                              prefer_hierarchy: bool = True
                              ) -> Tuple[Optional[str], str]:
        """Resolve a signal value using exact or normalized hierarchy identity."""
        resolver = getattr(self.waveform, "resolve_signal", None)
        if callable(resolver):
            resolution = resolver(
                signal,
                hierarchy,
                prefer_hierarchy=prefer_hierarchy,
            )
            if resolution.ambiguous and getattr(
                self.waveform, "exact_clock", False
            ):
                ambiguity = {
                    "signal": signal,
                    "cycle": cycle,
                    "candidate_count": len(resolution.candidates),
                }
                rows = self.stats.setdefault("identity_ambiguities", [])
                if ambiguity not in rows:
                    rows.append(ambiguity)
                return None, signal
            resolved_signal = resolution.resolved_signal
            if resolved_signal is not None:
                value = self.waveform.get_signal_value(
                    resolved_signal, cycle
                )
                if value is not None:
                    return value, resolved_signal
            return None, signal

        if prefer_hierarchy and hierarchy and not signal.startswith(hierarchy + '.'):
            full_signal = f"{hierarchy}.{signal}"
            value = self.waveform.get_signal_value(full_signal, cycle)
            if value is not None:
                return value, full_signal

        value = self.waveform.get_signal_value(signal, cycle)
        if value is not None:
            return value, signal

        if not prefer_hierarchy and hierarchy and not signal.startswith(hierarchy + '.'):
            full_signal = f"{hierarchy}.{signal}"
            value = self.waveform.get_signal_value(full_signal, cycle)
            if value is not None:
                return value, full_signal

        if match_cache is not None and signal in match_cache:
            matches = match_cache[signal]
        else:
            matches = self.waveform.find_signal(signal, max_results=10)
            if match_cache is not None:
                match_cache[signal] = matches

        usable_matches = []
        for match in matches:
            match_base = re.sub(r'\s*\[\d+:\d+\]$', '', match)
            if match_base.endswith('.' + signal) or match_base.endswith(signal):
                value = self.waveform.get_signal_value(match, cycle)
                if value is not None:
                    usable_matches.append((value, match))

        if len(usable_matches) > 1 and getattr(self.waveform, "exact_clock", False):
            self.stats.setdefault("identity_ambiguities", []).append(
                {
                    "signal": signal,
                    "cycle": cycle,
                    "candidate_count": len(usable_matches),
                }
            )
            return None, signal
        if usable_matches:
            return usable_matches[0]

        return None, signal
    
    def _parse_sva_time_window(self, sva_expr: str) -> Tuple[Optional[int], Optional[int]]:
        """
        Parse SVA time window from expression.
        
        Args:
            sva_expr: SVA expression like "antecedent |-> ##[1:200] consequent"
            
        Returns:
            (min_delay, max_delay) tuple, or (None, None) if no time window
        """
        match = self._re_sva_time_window.search(sva_expr)
        if not match:
            return None, None
        
        if match.group(1) and match.group(2):
            # ##[min:max] format
            return int(match.group(1)), int(match.group(2))
        elif match.group(3):
            # ##N format (fixed delay)
            delay = int(match.group(3))
            return delay, delay
        
        return None, None
    
    def _extract_consequent_signals(self, sva_expr: str, dep_sources: Optional[Set[str]] = None) -> Set[str]:
        """
        Extract signal names from SVA consequent part.
        
        Args:
            sva_expr: Full SVA expression
            dep_sources: Optional set of all source signals from dependencies
            
        Returns:
            Set of signal names in the consequent
        """
        match = self._re_sva_implication.match(sva_expr)
        if not match:
            # If expression is truncated, try to use dep_sources to infer consequent signals
            if dep_sources:
                # Heuristic: consequent signals are often named with "state", "eating", "done", etc.
                consequent_keywords = {'eating', 'done', 'ready', 'valid', 'complete', 'finish', 'state'}
                signals = set()
                for src in dep_sources:
                    src_lower = src.lower()
                    for kw in consequent_keywords:
                        if kw in src_lower:
                            signals.add(src)
                            break
                return signals
            return set()
        
        consequent = match.group(2).strip()
        # Remove time window specification if present
        consequent = re.sub(r'##\[\d+:\d+\]\s*', '', consequent)
        consequent = re.sub(r'##\d+\s*', '', consequent)
        
        # If consequent is empty or just "...", try using dep_sources
        if not consequent or consequent == '...' or consequent.strip() == '':
            if dep_sources:
                consequent_keywords = {'eating', 'done', 'ready', 'valid', 'complete', 'finish', 'state'}
                signals = set()
                for src in dep_sources:
                    src_lower = src.lower()
                    for kw in consequent_keywords:
                        if kw in src_lower:
                            signals.add(src)
                            break
                return signals
            return set()
        
        # Extract signal references
        signals = set()
        for sig_match in re.finditer(r'\b([a-zA-Z_][a-zA-Z0-9_]*)\b', consequent):
            sig = sig_match.group(1)
            # Filter out keywords and numbers
            if sig.lower() not in {'if', 'else', 'and', 'or', 'not', 'h', 'b', 'd', 'o'} and not sig.isdigit():
                signals.add(sig)
        
        return signals
    
    def _find_sva_trigger_cycle(self, endpoint_signal: str, max_cycle: int) -> Optional[int]:
        """
        Find cycle where SVA antecedent becomes true (trigger point).
        
        For assertions like: antecedent |-> ##[min:max] consequent
        The trigger cycle is the first cycle (before failure) where antecedent is true.
        This is typically max_cycle - max_delay for time-windowed assertions.
        """
        base_signal = self._extract_base_signal_name(endpoint_signal)
        hierarchy = self._extract_module_hierarchy(endpoint_signal)
        module_hint = self._infer_module_name(endpoint_signal, hierarchy)
        
        lookup_signal = (
            endpoint_signal if self.instance_local_dependencies else base_signal
        )
        deps = self.dependency_provider.get_dependencies_for_signal(
            lookup_signal, module_hint
        )
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
        
        match = self._re_sva_implication.match(sva_expr)
        if not match:
            return None
        
        antecedent = match.group(1).strip()
        sources = {d.source for d in deps}
        
        # Parse time window
        min_delay, max_delay = self._parse_sva_time_window(sva_expr)
        if min_delay is not None:
            self.stats["sva_time_window"] = (min_delay, max_delay)
            self.stats["sva_window_end_cycle"] = max_cycle
            # Pass dep sources to help extract consequent signals from truncated expressions
            self.stats["sva_consequent_signals"] = self._extract_consequent_signals(sva_expr, sources)

        if self.exact_sva_trigger_cycle is not None:
            if not 0 <= self.exact_sva_trigger_cycle <= max_cycle:
                self.stats["temporal_work_budget_reached"] = True
                self.stats["sva_trigger_evidence"] = "invalid_exact"
                return None
            self.stats["sva_trigger_evidence"] = "exact"
            return self.exact_sva_trigger_cycle
        self.stats["sva_trigger_evidence"] = "missing_exact"
        self.stats["sva_exact_trigger_missing"] = True
        return None
    
    def _prepare_node(self, signal: str, cycle: int, depth: int,
                      parent_hierarchy: str = '') -> Tuple[CausalNode, bool]:
        """Resolve a candidate node without consuming graph node budget."""
        original_signal = signal
        value, resolved_signal = self._resolve_signal_value(
            signal, cycle, parent_hierarchy, prefer_hierarchy=False
        )
        signal = resolved_signal
        
        value_missing = value is None
        if value is None:
            value = 'x'  # Unknown value
        
        node_id = self._make_node_id(signal, cycle, value)
        
        if node_id in self.nodes:
            return self.nodes[node_id], False
        
        # Get RTL context using base signal name (without hierarchy prefix)
        base_signal = self._extract_base_signal_name(signal)
        signal_hierarchy = self._extract_module_hierarchy(signal)
        module_hint = self._infer_module_name(signal, signal_hierarchy or parent_hierarchy)
        lookup_signal = (
            signal if self.instance_local_dependencies else base_signal
        )
        rtl_context = self.dependency_provider.get_rtl_context(
            lookup_signal, module_hint
        )
        
        node = CausalNode(
            id=node_id,
            signal=signal,
            cycle=cycle,
            value=value,
            rtl_refs=rtl_context.get("rtl_refs", []),
            rtl_context_missing=not rtl_context.get("found", False),
            depth=depth,
            identity_strength=(
                "unresolved"
                if value_missing
                else "exact"
                if original_signal == resolved_signal
                else "hierarchy_inferred"
                if parent_hierarchy
                and resolved_signal == f"{parent_hierarchy}.{original_signal}"
                else "unresolved"
            ),
        )
        
        return node, True

    def _commit_node(self, node: CausalNode, is_new: bool) -> Optional[CausalNode]:
        """Commit a prepared node after admission, if graph capacity permits."""
        existing = self.nodes.get(node.id)
        if existing is not None:
            return existing
        if not is_new:
            return node
        if len(self.nodes) >= self.max_nodes:
            self.stats["max_nodes_reached"] = True
            return None
        self.nodes[node.id] = node
        self.stats["nodes_created"] += 1
        if node.rtl_context_missing:
            self.stats["undetermined_nodes"] += 1
        return node

    def _get_or_create_node(self, signal: str, cycle: int, depth: int,
                            parent_hierarchy: str = '') -> Optional[CausalNode]:
        """Create an explicitly admitted endpoint or synthetic SVA node."""
        node, is_new = self._prepare_node(
            signal, cycle, depth, parent_hierarchy
        )
        return self._commit_node(node, is_new)

    def _commit_candidate(
        self,
        parent_node: CausalNode,
        parent_is_new: bool,
        edge: CausalEdge,
    ) -> Optional[CausalNode]:
        """Atomically admit the source endpoint and its causal edge."""
        committed = self._commit_node(parent_node, parent_is_new)
        if committed is None:
            return None
        edge.src_node_id = committed.id
        self.edges.append(edge)
        self._edge_set.add((committed.id, edge.dst_node_id))
        self.stats["edges_created"] += 1
        return committed
    
    def _get_parent_cycle(self, dep: Dependency, target_cycle: int) -> int:
        """
        Determine the parent cycle based on dependency type.
        
        Args:
            dep: Dependency information
            target_cycle: Target signal's cycle
            
        Returns:
            Source signal's relevant cycle
        """
        if dep.dep_type in (
            DependencyType.COMBINATIONAL,
            DependencyType.ASSERTION,
            DependencyType.PORT_INPUT,
            DependencyType.PORT_OUTPUT,
            DependencyType.WIRE,
        ):
            return target_cycle  # Same cycle for combinational
        elif dep.dep_type == DependencyType.SEQUENTIAL:
            return max(0, target_cycle - 1)  # Previous cycle for sequential
        else:
            return max(0, target_cycle - 1)  # Default to previous

    @staticmethod
    def _statement_key(
        dep: Dependency,
        target_signal: str,
        target_cycle: int,
    ) -> StatementEvaluationKey:
        return StatementEvaluationKey(
            file_path=dep.file_path,
            line_start=dep.line_start,
            line_end=dep.line_end,
            module_name=dep.module_name,
            target=dep.target_qualified or dep.target,
            target_signal=target_signal,
            target_cycle=target_cycle,
            expression=dep.expression,
            condition=dep.condition,
        )

    def _get_compiled_expression(
        self,
        expression: str,
    ) -> CompiledExpression:
        compiled = self._compiled_expression_cache.get(expression)
        if compiled is not None:
            self.stats["compiled_expression_hits"] += 1
            return compiled
        self.stats["compiled_expression_misses"] += 1
        compiled = CompiledExpression.compile(expression)
        self._compiled_expression_cache[expression] = compiled
        return compiled

    def _get_statement_evaluation(
        self,
        target_signal: str,
        target_cycle: int,
        dep: Dependency,
        statement_dependencies: Tuple[Dependency, ...],
    ) -> _StatementEvaluationContext:
        """Build one base environment for every source of a statement."""
        key = self._statement_key(dep, target_signal, target_cycle)
        cached = self._statement_evaluation_cache.get(key)
        if cached is not None:
            return cached

        target_hierarchy = self._extract_module_hierarchy(target_signal)
        environment: Dict[str, str] = {}
        source_values: Dict[str, str] = {}
        for source_dep in sorted(
            statement_dependencies,
            key=lambda row: (
                row.source,
                row.dep_type.value,
                row.source_qualified,
            ),
        ):
            source_cycle = self._get_parent_cycle(source_dep, target_cycle)
            value, _resolved = self._resolve_signal_value(
                source_dep.source,
                source_cycle,
                target_hierarchy,
                prefer_hierarchy=True,
            )
            if value is not None:
                environment[source_dep.source] = value
                source_values[source_dep.source] = value

        compiled = self._get_compiled_expression(dep.expression)
        original_result = (
            compiled.evaluate(environment) if environment else None
        )
        context = _StatementEvaluationContext(
            environment=environment,
            source_values=source_values,
            compiled=compiled,
            original_result=original_result,
        )
        self._statement_evaluation_cache[key] = context
        self.stats["statement_evaluations"] += 1
        self.stats["statement_environment_builds"] += 1
        return context

    def _evaluate_counterfactual(self,
                                  target_signal: str,
                                  target_cycle: int,
                                  source_signal: str,
                                  source_cycle: int,
                                  dep: Dependency,
                                  statement_dependencies: Optional[
                                      Tuple[Dependency, ...]
                                  ] = None
                                  ) -> Tuple[bool, float, List[Dict]]:
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
        target_value = self.waveform.get_signal_value(target_signal, target_cycle)
        source_value = self.waveform.get_signal_value(
            source_signal, source_cycle
        )
        if target_value is None:
            return False, 0.0, []

        if not dep.expression.strip():
            return self._simple_toggle_test(source_signal, source_cycle, target_signal, target_cycle)

        source_base = self._extract_base_signal_name(source_signal)
        context = self._get_statement_evaluation(
            target_signal,
            target_cycle,
            dep,
            statement_dependencies or (dep,),
        )
        env = context.environment
        if source_value is None:
            source_value = context.source_values.get(dep.source)
        if source_value is None:
            return False, 0.0, []

        if not env:
            return self._simple_toggle_test(source_signal, source_cycle, target_signal, target_cycle)

        orig_result = context.original_result
        if orig_result is None:
            return self._simple_toggle_test(source_signal, source_cycle, target_signal, target_cycle)

        perturbed_env = dict(env)
        perturb_key = dep.source if dep.source in perturbed_env else source_base if source_base in perturbed_env else dep.source
        perturb_value = invert_value(source_value)
        perturbed_env[perturb_key] = perturb_value

        perturbed_result = context.compiled.evaluate(perturbed_env)
        if perturbed_result is None:
            return False, 0.0, []

        is_causal = values_differ(orig_result, perturbed_result)

        if not is_causal and '==' in dep.expression:
            smart_result = self._try_smart_perturbation(
                dep,
                env,
                source_value,
                source_base,
                orig_result,
                context.compiled,
            )
            if smart_result is not None:
                is_causal, perturbed_result, perturb_value = smart_result

        score = 0.0
        if is_causal:
            max_len = max(len(orig_result), len(perturbed_result))
            left = orig_result.zfill(max_len)
            right = perturbed_result.zfill(max_len)
            diff_bits = sum(
                1 for a, b in zip(left, right)
                if a != b and a not in 'xXzZ' and b not in 'xXzZ'
            )
            score = min(1.0, diff_bits / max(1, max_len) + 0.5)

        examples = []
        if is_causal:
            examples.append({
                "type": "counterfactual",
                "source_original": source_value,
                "source_perturbed": perturb_value,
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
                                 orig_result: str,
                                 compiled: Optional[CompiledExpression] = None
                                 ) -> Optional[Tuple[bool, str, str]]:
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
            (is_causal, perturbed_result, perturbed_source_value) or None if cannot apply
        """
        expr = dep.expression
        
        # Pattern: source == literal
        # Try to extract the comparison value
        cache_key = (expr, dep.source, source_base)
        if cache_key in self._equality_literal_cache:
            target_value = self._equality_literal_cache[cache_key]
        else:
            signal_names = sorted(
                {dep.source, source_base},
                key=lambda name: (-len(name), name),
            )
            signal_pattern = "|".join(
                re.escape(name) for name in signal_names
            )
            literal_pattern = r"(\d+'[bhd][0-9a-fA-F_]+)"
            patterns = (
                re.compile(
                    rf"\b(?:{signal_pattern})\s*==\s*{literal_pattern}"
                ),
                re.compile(
                    rf"{literal_pattern}\s*==\s*(?:{signal_pattern})\b"
                ),
            )
            target_value = None
            for pattern in patterns:
                match = pattern.search(expr)
                if match:
                    target_value = ExpressionEvaluator({}).evaluate(
                        match.group(1)
                    )
                    break
            self._equality_literal_cache[cache_key] = target_value
        
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
        
        expression = compiled or self._get_compiled_expression(dep.expression)
        perturbed_result = expression.evaluate(perturbed_env)
        
        if perturbed_result is None:
            return None
        
        is_causal = values_differ(orig_result, perturbed_result)
        return (is_causal, perturbed_result, perturb_val)

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

    def _expandable_parent_cycle(
        self,
        node: CausalNode,
        dep: Dependency,
        parent_hierarchy: str,
    ) -> Optional[int]:
        """Return the parent cycle only for a dependency eligible to expand."""
        parent_cycle = self._get_parent_cycle(dep, node.cycle)
        if parent_cycle < 0:
            return None
        source_base = self._extract_base_signal_name(dep.source)
        target_base = self._extract_base_signal_name(node.signal)
        if source_base == target_base or source_base in _IGNORED_SIGNALS:
            return None
        if dep.source == node.signal:
            return None
        full_source = (
            f"{parent_hierarchy}.{dep.source}"
            if parent_hierarchy
            and not dep.source.startswith(parent_hierarchy + ".")
            else dep.source
        )
        clean_node_signal = re.sub(r'\s*\[\d+:\d+\]$', '', node.signal)
        full_source_base = full_source.rsplit('.', 1)[-1]
        if (
            (full_source == clean_node_signal or full_source_base == target_base)
            and dep.dep_type == DependencyType.COMBINATIONAL
        ):
            return None
        return parent_cycle

    def _prepare_direct_candidates(
        self, node: CausalNode, depth: int
    ) -> Tuple[str, List[Tuple[Dependency, int]], Dict[StatementEvaluationKey, Tuple[Dependency, ...]]]:
        """Resolve and group direct dependencies without committing graph state."""
        base_signal = self._extract_base_signal_name(node.signal)
        parent_hierarchy = self._extract_module_hierarchy(node.signal)
        module_hint = self._infer_module_name(node.signal, parent_hierarchy)
        lookup_signal = node.signal if self.instance_local_dependencies else base_signal
        deps = self.dependency_provider.get_dependencies_for_signal(lookup_signal, module_hint)
        if not deps:
            clean_signal = re.sub(r'\s*\[\d+:\d+\]$', '', node.signal)
            deps = self.dependency_provider.get_dependencies_for_signal(clean_signal, module_hint)
        candidates = [
            (dep, parent_cycle)
            for dep in deps
            for parent_cycle in [self._expandable_parent_cycle(node, dep, parent_hierarchy)]
            if parent_cycle is not None
        ]
        if self.search_policy is not None and self.search_policy.policy_id != "legacy_dfs_v1":
            candidates.sort(
                key=lambda row: (
                    row[0].source,
                    row[1],
                    row[0].dep_type.value,
                    row[0].file_path,
                    row[0].line_start,
                    row[0].line_end,
                    row[0].expression,
                )
            )
        grouped: Dict[StatementEvaluationKey, List[Dependency]] = {}
        for dep, _cycle in candidates:
            grouped.setdefault(self._statement_key(dep, node.signal, node.cycle), []).append(dep)
        return parent_hierarchy, candidates, {
            key: tuple(rows) for key, rows in grouped.items()
        }

    @staticmethod
    def _legacy_method(examples: List[Dict[str, Any]], is_causal: bool) -> str:
        if examples:
            return str(examples[0].get("type", "counterfactual"))
        return "not_supported" if not is_causal else "counterfactual"

    def _evaluate_typed_contribution(
        self,
        node: CausalNode,
        parent_node: CausalNode,
        dep: Dependency,
        statement_dependencies: Tuple[Dependency, ...],
    ) -> ContributionEvidence:
        remaining = max(
            0,
            self.max_intervention_evaluations
            - self.stats["intervention_evaluations"],
        )
        if self.contribution_evaluator is not None:
            evidence = self.contribution_evaluator(
                target_node=node,
                parent_node=parent_node,
                dependency=dep,
                statement_dependencies=statement_dependencies,
                remaining_interventions=remaining,
            )
            if not isinstance(evidence, ContributionEvidence):
                raise TypeError("contribution_evaluator must return ContributionEvidence")
            if evidence.interventions.evaluated > remaining:
                raise ValueError("contribution_evaluator exceeded the intervention budget")
            self.stats["intervention_evaluations"] += evidence.interventions.evaluated
            return evidence

        if not dep.expression.strip():
            causal, _score, _examples = self._simple_toggle_test(
                parent_node.signal, parent_node.cycle, node.signal, node.cycle
            )
            return toggle_evidence(source_toggled=causal, target_toggled=causal) if causal else structural_evidence()

        context = self._get_statement_evaluation(
            node.signal, node.cycle, dep, statement_dependencies
        )
        source_value = parent_node.value
        planned = generate_interventions(
            source_value,
            max_interventions=int(self.search_policy.payload["max_interventions_per_candidate"]),
        )
        evaluated_values = planned[:remaining]
        perturb_key = dep.source if dep.source in context.environment else self._extract_base_signal_name(parent_node.signal)
        results: List[Optional[str]] = []
        for value in evaluated_values:
            environment = dict(context.environment)
            environment[perturb_key] = value
            results.append(context.compiled.evaluate(environment))
        self.stats["intervention_evaluations"] += len(evaluated_values)
        truncated = len(evaluated_values) < len(planned)
        if truncated:
            self.stats["intervention_evaluation_budget_reached"] = True
        results.extend([None] * (len(planned) - len(results)))
        rule_active: Optional[bool] = True
        method = "expression_intervention"
        if dep.condition:
            condition_result = self._get_compiled_expression(dep.condition).evaluate(context.environment)
            rule_active = None if condition_result is None else ExpressionEvaluator(context.environment)._is_true(condition_result)
            method = "active_rule_intervention"
        return evaluate_interventions(
            source_value=source_value,
            original_result=context.original_result,
            observed_target=node.value,
            intervention_values=planned,
            intervention_results=results,
            target_basis="full_exact_target",
            method=method,
            global_budget_truncated=truncated,
            rule_active=rule_active,
        )

    def _extract_score_features(
        self,
        parent_node: CausalNode,
        dep: Dependency,
        evidence: ContributionEvidence | LegacyContributionEvidence,
    ) -> ScoreFeatures:
        values: Dict[str, Optional[float]] = {
            name: None
            for name in ("C_cf", "C_obs", "C_time", "C_ctrl", "C_sem", "C_structural", "P_unknown", "P_ambiguity", "P_temp", "P_fanout")
        }
        availability = {name: "not_available" for name in values}

        if isinstance(evidence, LegacyContributionEvidence):
            values["C_cf"] = evidence.legacy_score
            availability["C_cf"] = "available"
        else:
            routed = route_contribution(evidence)
            values[routed.feature_name] = routed.value
            availability[routed.feature_name] = routed.availability

        values["C_obs"] = {
            "exact": 1.0,
            "hierarchy_inferred": 0.6,
        }.get(parent_node.identity_strength, 0.0)
        availability["C_obs"] = "available"
        if availability["C_time"] != "available":
            values["C_time"] = 0.8 if dep.dep_type == DependencyType.SEQUENTIAL else 0.4
            availability["C_time"] = "available"
        if availability["C_ctrl"] != "available":
            values["C_ctrl"] = 0.85 if dep.dep_type == DependencyType.SEQUENTIAL else 0.7 if dep.condition else 0.3
            availability["C_ctrl"] = "available"
        if self.search_policy.policy_id == "chisel_hybrid_best_first_v1":
            values["C_sem"] = 0.0
            availability["C_sem"] = "available"
        else:
            availability["C_sem"] = "not_applicable"
        if availability["C_structural"] == "not_available":
            availability["C_structural"] = "not_applicable"

        unknown = any(char in parent_node.value.lower() for char in ("x", "z"))
        penalty_values = {
            "P_unknown": 1.0 if unknown else 0.0,
            "P_ambiguity": 0.0 if parent_node.identity_strength == "exact" else 0.5 if parent_node.identity_strength == "hierarchy_inferred" else 1.0,
            "P_temp": 0.0,
            "P_fanout": 0.0,
        }
        for name, value in penalty_values.items():
            values[name] = value
            availability[name] = "available"
        return ScoreFeatures.from_dict(
            {"feature_vector": values, "feature_availability": availability}
        )

    def _evaluate_candidate(
        self,
        node: CausalNode,
        depth: int,
        parent_hierarchy: str,
        dep: Dependency,
        parent_cycle: int,
        statement_dependencies: Tuple[Dependency, ...],
    ) -> Optional[CandidateExpansion]:
        parent_node, parent_is_new = self._prepare_node(
            dep.source, parent_cycle, depth + 1, parent_hierarchy
        )
        if parent_node.id == node.id or (parent_node.id, node.id) in self._edge_set:
            self.stats["rejected_candidates"] += 1
            return None
        self.stats["candidate_evaluations"] += 1

        if self.search_policy is None or self.search_policy.policy_id in {
            "legacy_dfs_v1", "legacy_scalar_best_first_v1"
        }:
            is_causal, score, examples = self._evaluate_counterfactual(
                node.signal, node.cycle, parent_node.signal, parent_cycle, dep, statement_dependencies
            )
            if not is_causal and '|->' in dep.expression:
                trigger_cycle = self.stats.get("sva_trigger_cycle")
                if trigger_cycle is not None and node.cycle == trigger_cycle:
                    is_causal, score = True, 0.85
                    examples = [{
                        "type": "sva_antecedent", "trigger_cycle": trigger_cycle,
                        "expression": dep.expression, "signal": parent_node.signal,
                        "value": parent_node.value,
                    }]
            weak = not is_causal
            if weak and not node.rtl_context_missing:
                score = 0.3
                examples = [{"type": "structural", "reason": "RTL dependency exists but counterfactual not conclusive"}]
            envelope = adapt_legacy_contribution(
                legacy_method=self._legacy_method(examples, is_causal),
                legacy_score=score,
                expression_evaluations=1,
                intervention_evaluations=1 if any(row.get("type") == "counterfactual" for row in examples) else 0,
                change_examples=examples,
            )
            if self.search_policy is not None and self.search_policy.policy_id == "legacy_scalar_best_first_v1":
                feature_row = self._extract_score_features(parent_node, dep, envelope)
                local_score = score_features(self.search_policy, feature_row).local_score
            else:
                feature_row = None
                local_score = score
        else:
            raw_evidence = self._evaluate_typed_contribution(
                node, parent_node, dep, statement_dependencies
            )
            is_causal = raw_evidence.status == "supported"
            weak = not is_causal
            envelope = raw_evidence if is_causal else structural_evidence()
            score = envelope.score
            examples = []
            feature_row = self._extract_score_features(parent_node, dep, envelope)
            local_score = score_features(self.search_policy, feature_row).local_score
        return CandidateExpansion(
            dep, parent_cycle, parent_node, parent_is_new, is_causal, weak,
            score, envelope, examples, local_score, feature_row,
        )

    def _admit_evaluated_candidates(
        self, node: CausalNode, evaluated: List[CandidateExpansion]
    ) -> List[CandidateExpansion]:
        if self.search_policy is not None and self.search_policy.policy_id != "legacy_dfs_v1":
            evaluated.sort(
                key=lambda row: (
                    row.weak_candidate,
                    -row.local_score,
                    row.parent_node.id,
                    row.dep.source,
                )
            )
        admitted: List[CandidateExpansion] = []
        weak_for_target = 0
        for candidate in evaluated:
            if (candidate.parent_node.id, node.id) in self._edge_set:
                continue
            if candidate.weak_candidate and (
                self.stats["weak_edges_admitted"] >= self.weak_edge_budget
                or weak_for_target >= self.weak_beam_width
            ):
                self.stats["rejected_candidates"] += 1
                continue
            dep = candidate.dep
            if dep.dep_type == DependencyType.SEQUENTIAL:
                contribution_type = ContributionType.STATE
            elif dep.condition:
                contribution_type = ContributionType.CONDITIONAL
            elif candidate.contribution_evidence.to_dict().get("method") == "toggle_correlation" or (
                candidate.change_examples and candidate.change_examples[0].get("type") == "toggle_correlation"
            ):
                contribution_type = ContributionType.TOGGLE
            else:
                contribution_type = ContributionType.EXPR_EVAL
            edge = CausalEdge(
                src_node_id=candidate.parent_node.id,
                dst_node_id=node.id,
                reason=f"{dep.source} affects {node.signal} via {dep.dep_type.value}",
                contribution_type=contribution_type,
                contribution_score=candidate.contribution_score,
                evidence={
                    "file": dep.file_path,
                    "lines": [dep.line_start, dep.line_end],
                    "code_snippet": dep.code_snippet,
                    "expression": dep.expression,
                    "condition": dep.condition,
                },
                change_examples=candidate.change_examples,
                contribution_evidence=(
                    candidate.contribution_evidence.to_dict()
                    if self.search_policy is not None
                    else None
                ),
            )
            committed = self._commit_candidate(candidate.parent_node, candidate.parent_is_new, edge)
            if committed is None:
                continue
            candidate.parent_node = committed
            candidate.edge = edge
            if candidate.weak_candidate:
                weak_for_target += 1
                self.stats["weak_edges_admitted"] += 1
            committed.suspect_score = max(committed.suspect_score, candidate.contribution_score * 0.9)
            admitted.append(candidate)
        return admitted

    def _expand_node(self, node: CausalNode, depth: int) -> List[CandidateExpansion]:
        if node.id in self.visited:
            return []
        self.visited.add(node.id)
        self.stats["expanded_nodes"] += 1
        parent_hierarchy, candidates, statement_groups = self._prepare_direct_candidates(node, depth)
        if not candidates:
            node.is_root = True
            return []
        if depth >= self.max_depth:
            self.stats["max_depth_reached"] = True
            return []
        evaluated: List[CandidateExpansion] = []
        for dep, parent_cycle in candidates:
            if self.stats["candidate_evaluations"] >= self.candidate_evaluation_budget:
                self.stats["candidate_evaluation_budget_reached"] = True
                break
            result = self._evaluate_candidate(
                node, depth, parent_hierarchy, dep, parent_cycle,
                statement_groups[self._statement_key(dep, node.signal, node.cycle)],
            )
            if result is not None:
                evaluated.append(result)
        return self._admit_evaluated_candidates(node, evaluated)

    def _slice_node(self, node: CausalNode, depth: int):
        """Legacy recursive traversal over the shared expansion implementation."""
        for candidate in self._expand_node(node, depth):
            if not any(char in candidate.parent_node.value.lower() for char in ("x", "z")):
                self._slice_node(candidate.parent_node, depth + 1)

    def _run_frontier(self, endpoint_node: CausalNode) -> None:
        scheduler = FrontierScheduler(self.search_policy)
        scheduler.push(
            FrontierItem(
                node_id=endpoint_node.id,
                incoming_edge_id="seed",
                depth=0,
                seed_id="endpoint",
                seed_rank=0,
                seed_prior=1.0,
                local_score=1.0,
                path_score=1.0,
                frontier_priority=1.0,
                support_scores=(),
                source_group=self._extract_module_hierarchy(endpoint_node.signal),
            )
        )
        while True:
            selection = scheduler.pop()
            if selection is None:
                break
            self.stats[f"{selection.lane}_expansions"] += 1
            node = self.nodes.get(selection.item.node_id)
            if node is None:
                continue
            for candidate in self._expand_node(node, selection.item.depth):
                parent = candidate.parent_node
                if any(char in parent.value.lower() for char in ("x", "z")):
                    continue
                support_scores = selection.item.support_scores + (candidate.local_score,)
                path_score = path_support(support_scores, self.search_policy)
                priority = frontier_priority(
                    candidate.local_score, path_score, selection.item.seed_prior,
                    self.search_policy,
                )
                edge_id = hashlib.md5(
                    f"{parent.id}->{node.id}".encode()
                ).hexdigest()[:12]
                scheduler.push(
                    FrontierItem(
                        node_id=parent.id,
                        incoming_edge_id=edge_id,
                        depth=selection.item.depth + 1,
                        seed_id=selection.item.seed_id,
                        seed_rank=selection.item.seed_rank,
                        seed_prior=selection.item.seed_prior,
                        local_score=candidate.local_score,
                        path_score=path_score,
                        frontier_priority=priority,
                        support_scores=support_scores,
                        source_group=self._extract_module_hierarchy(parent.signal),
                    )
                )
    
    def _analyze_sva_time_window(self, 
                                  trigger_cycle: int,
                                  window_end_cycle: int,
                                  consequent_signals: Set[str],
                                  hierarchy: str,
                                  depth: int) -> List[CausalNode]:
        """
        Analyze SVA time window to find why consequent never became true.
        
        For assertions like: antecedent |-> ##[1:200] consequent
        We need to analyze why 'consequent' was never true during cycles
        [trigger_cycle + 1, window_end_cycle].
        
        Strategy:
        1. Sample key cycles within the window (start, middle, end)
        2. For each consequent signal, find cycles where it was closest to becoming true
        3. Trace back why it didn't toggle to true
        
        Args:
            trigger_cycle: Cycle where antecedent became true
            window_end_cycle: Cycle where assertion failed (end of window)
            consequent_signals: Set of signal names in the consequent
            hierarchy: Module hierarchy prefix
            depth: Current depth for node creation
            
        Returns:
            List of nodes created for window analysis
        """
        window_nodes = []
        min_delay, max_delay = self.stats.get("sva_time_window", (1, 1))
        
        # Calculate actual window range
        window_start = trigger_cycle + min_delay
        window_end = min(trigger_cycle + max_delay, window_end_cycle)
        
        if window_start > window_end:
            return window_nodes
        
        # Sample cycles: start, a few intermediate points, and end
        sample_cycles = [window_start]
        window_size = window_end - window_start
        if window_size > 10:
            # Add intermediate sample points
            for i in [0.25, 0.5, 0.75]:
                sample_cycles.append(window_start + int(window_size * i))
        sample_cycles.append(window_end)
        sample_cycles = sorted(set(sample_cycles))
        
        # For each consequent signal, analyze at sample cycles
        for sig in sorted(consequent_signals):
            resolution = self.waveform.resolve_signal(
                sig, hierarchy, prefer_hierarchy=True
            )
            resolved = resolution.resolved_signal
            interesting_cycles: List[int] = []
            if resolved is not None:
                changes = self.waveform.get_value_changes_bounded(
                    resolved,
                    window_start,
                    window_end,
                    max_changes=5,
                )
                if changes is None:
                    self.stats["temporal_work_budget_reached"] = True
                else:
                    interesting_cycles = [
                        cycle for cycle, _old, _new in changes
                    ]
            
            # Combine sample cycles with interesting cycles (limit to avoid explosion)
            cycles_to_analyze = list(set(sample_cycles + interesting_cycles[:5]))
            cycles_to_analyze = sorted(cycles_to_analyze)[:8]  # Limit to 8 cycles
            
            for cycle in cycles_to_analyze:
                # Create node for this signal at this cycle
                node = self._get_or_create_node(
                    resolved or sig, cycle, depth, hierarchy
                )
                if node is not None:
                    # Mark as part of window analysis
                    if "window_analysis" not in [ref.get("type") for ref in node.rtl_refs]:
                        node.rtl_refs.append({
                            "type": "window_analysis",
                            "window_start": window_start,
                            "window_end": window_end,
                            "trigger_cycle": trigger_cycle
                        })
                    window_nodes.append(node)
        
        return window_nodes
    
    def slice_from_endpoint(self, 
                            endpoint_signal: str, 
                            endpoint_cycle: int) -> Tuple[Dict[str, CausalNode], List[CausalEdge]]:
        """
        Perform backward slicing starting from endpoint.
        
        For SVA assertions with implication (|->), automatically finds
        the cycle where the antecedent becomes true (trigger point).
        
        For assertions with time windows (##[min:max]), also analyzes
        why the consequent never became true during the window.
        
        Args:
            endpoint_signal: Signal that triggered the counterexample
            endpoint_cycle: Cycle when counterexample was triggered (assertion failure)
            
        Returns:
            (nodes dict, edges list)
        """
        # Reset state
        self.nodes = {}
        self.edges = []
        self.visited = set()
        self._edge_set = set()  # Track edges to prevent duplicates
        self._statement_evaluation_cache = {}
        self.stats = self._new_stats()
        
        original_endpoint_cycle = endpoint_cycle  # This is the failure cycle
        hierarchy = self._extract_module_hierarchy(endpoint_signal)
        
        # For SVA assertions, try to find the actual trigger cycle
        # where the antecedent became true
        trigger_cycle = self._find_sva_trigger_cycle(endpoint_signal, endpoint_cycle)
        if trigger_cycle is not None:
            self.stats["sva_trigger_cycle"] = trigger_cycle
        
        # Create endpoint node at the FAILURE cycle (not trigger cycle)
        # This is the assertion failure point - causality flows backward from here
        endpoint_node = self._get_or_create_node(endpoint_signal, original_endpoint_cycle, 0)
        if endpoint_node is None:
            return {}, []
        
        endpoint_node.is_endpoint = True
        endpoint_node.suspect_score = 1.0
        
        # If SVA has time window, analyze the causal chain properly
        time_window = self.stats.get("sva_time_window")
        consequent_signals = self.stats.get("sva_consequent_signals")
        
        if time_window is not None and consequent_signals and trigger_cycle is not None:
            # For time-windowed SVA: antecedent |-> ##[min:max] consequent
            # 
            # The causal chain should be:
            # 1. antecedent_signals@trigger_cycle -> ... -> consequent_signals@window
            # 2. consequent_signals@window (being false) -> assertion_fail@failure_cycle
            #
            # So we need to:
            # a) Analyze consequent signals in the window (they are direct causes of failure)
            # b) Then trace back from consequent signals to find why they stayed false
            # c) Also trace the antecedent signals to understand the trigger condition
            
            # First, analyze consequent signals in the time window
            window_nodes = self._analyze_sva_time_window(
                trigger_cycle=trigger_cycle,
                window_end_cycle=original_endpoint_cycle,
                consequent_signals=consequent_signals,
                hierarchy=hierarchy,
                depth=1
            )
            
            # Create edges from window nodes to endpoint (causality: earlier -> later)
            # Only include nodes that are BEFORE or AT the failure cycle
            for window_node in window_nodes:
                if window_node.cycle <= original_endpoint_cycle:
                    edge_key = (window_node.id, endpoint_node.id)
                    if edge_key not in self._edge_set:
                        edge = CausalEdge(
                            src_node_id=window_node.id,
                            dst_node_id=endpoint_node.id,
                            reason=f"{window_node.signal}@{window_node.cycle} stayed false in window [{time_window[0]}:{time_window[1]}]",
                            contribution_type=ContributionType.STATE,
                            contribution_score=0.8,
                            evidence={
                                "type": "sva_time_window",
                                "trigger_cycle": trigger_cycle,
                                "window": time_window,
                                "window_cycle": window_node.cycle,
                                "failure_cycle": original_endpoint_cycle
                            },
                            change_examples=[{
                                "type": "window_consequent",
                                "signal": window_node.signal,
                                "cycle": window_node.cycle,
                                "value": window_node.value,
                                "expected": "should become true within window"
                            }]
                        )
                        self.edges.append(edge)
                        self._edge_set.add(edge_key)
                        self.stats["edges_created"] += 1
            
            # Slice backward from each window node to find root causes
            for window_node in window_nodes:
                if window_node.id not in self.visited:
                    self._slice_node(window_node, window_node.depth)
            
            # Also create a node for the antecedent at trigger cycle and trace back
            if trigger_cycle is not None and trigger_cycle < original_endpoint_cycle:
                # Create antecedent analysis node at trigger cycle
                trigger_node = self._get_or_create_node(endpoint_signal, trigger_cycle, 1)
                if trigger_node is not None and trigger_node.id != endpoint_node.id:
                    # Edge from trigger to failure (trigger happened before failure)
                    edge_key = (trigger_node.id, endpoint_node.id)
                    if edge_key not in self._edge_set:
                        edge = CausalEdge(
                            src_node_id=trigger_node.id,
                            dst_node_id=endpoint_node.id,
                            reason=f"SVA triggered at cycle {trigger_cycle}, failed at cycle {original_endpoint_cycle}",
                            contribution_type=ContributionType.CONDITIONAL,
                            contribution_score=1.0,
                            evidence={
                                "type": "sva_trigger",
                                "trigger_cycle": trigger_cycle,
                                "failure_cycle": original_endpoint_cycle,
                                "window": time_window
                            },
                            change_examples=[]
                        )
                        self.edges.append(edge)
                        self._edge_set.add(edge_key)
                        self.stats["edges_created"] += 1
                    
                    # Slice backward from trigger node
                    if trigger_node.id not in self.visited:
                        self._slice_node(trigger_node, trigger_node.depth)
        else:
            # No time window - just do normal backward slicing from endpoint
            if self.search_policy is not None and self.search_policy.policy_id != "legacy_dfs_v1":
                self._run_frontier(endpoint_node)
            else:
                self._slice_node(endpoint_node, 0)
        
        # Mark leaf nodes as roots
        nodes_with_incoming = set(e.dst_node_id for e in self.edges)
        for node in self.nodes.values():
            if node.id not in nodes_with_incoming and not node.is_endpoint:
                node.is_root = True
        
        return self.nodes, self.edges
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get slicing statistics."""
        result = self.stats.copy()
        cache_stats = getattr(self.waveform, "get_cache_statistics", None)
        if callable(cache_stats):
            result.update(cache_stats())
        return result
