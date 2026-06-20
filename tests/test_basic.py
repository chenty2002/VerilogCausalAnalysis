#!/usr/bin/env python3
"""
Basic tests for verilog_causal_analysis module.
"""

import sys
import os
import tempfile

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

def test_imports():
    """Test that all main classes can be imported."""
    from verilog_causal_analysis import (
        CausalGraphBuilder,
        CausalGraphResult,
        CausalGraphMeta,
        build_causal_graph,
        VerilogParser,
        DependencyType,
        Dependency,
        SignalInfo,
        ModuleInfo,
        CycleAlignedWaveform,
        SignalTransition,
        CycleSnapshot,
        parse_binary_value,
        invert_value,
        values_differ,
        BackwardSlicer,
        CausalNode,
        CausalEdge,
        ContributionType,
        ExpressionEvaluator,
        __version__
    )
    
    print(f"✓ All imports successful")
    print(f"  Version: {__version__}")


def test_verilog_parser_basic():
    """Test VerilogParser basic functionality."""
    from verilog_causal_analysis import VerilogParser
    
    parser = VerilogParser()
    assert parser is not None
    print("✓ VerilogParser instantiation successful")


def test_verilog_parser_dependencies():
    """Test direction-aware ports, control deps, and full multiline SVA capture."""
    from verilog_causal_analysis import VerilogParser, DependencyType

    source = r'''
module Child(input logic in, output logic out);
  assign out = in;
endmodule

module Parent(input logic clk, input logic a, input logic en, output logic y, output logic q);
  Child u (.in(a), .out(y));
  always_ff @(posedge clk) begin
    if (en) q <= a;
    else q <= 1'b0;
  end
  my_assert: assert property (@(posedge clk) disable iff (en)
    a |-> ##[1:20] (y && q)
  );
endmodule
'''

    with tempfile.NamedTemporaryFile('w', suffix='.sv', delete=False) as tmp:
        tmp.write(source)
        path = tmp.name

    try:
        parser = VerilogParser()
        parser.parse_file(path)

        input_port_deps = parser.get_dependencies_for_signal('u.in', 'Parent')
        assert any(
            dep.source == 'a'
            and dep.target == 'u.in'
            and dep.dep_type == DependencyType.PORT_INPUT
            for dep in input_port_deps
        )

        output_port_deps = parser.get_dependencies_for_signal('y', 'Parent')
        assert any(
            dep.source == 'u.out'
            and dep.target == 'y'
            and dep.dep_type == DependencyType.PORT_OUTPUT
            for dep in output_port_deps
        )

        q_deps = parser.get_dependencies_for_signal('q', 'Parent')
        assert any(dep.source == 'en' and dep.condition for dep in q_deps)

        sva_deps = parser.get_dependencies_for_signal('my_assert', 'Parent')
        assert any(
            '##[1:20]' in dep.expression and 'y && q' in dep.expression
            for dep in sva_deps
        )
    finally:
        os.unlink(path)

    print("✓ VerilogParser dependency extraction tests passed")


def test_sva_label_auto_detect_split_line():
    """Test SVA label extraction when label and assert property are split."""
    from verilog_causal_analysis import extract_sva_assertions_from_verilog

    source = r'''
module M(input logic clk, input logic a);
  split_label:
    assert property (@(posedge clk) a);
endmodule
'''

    with tempfile.NamedTemporaryFile('w', suffix='.sv', delete=False) as tmp:
        tmp.write(source)
        path = tmp.name

    try:
        labels = extract_sva_assertions_from_verilog([path])
        assert labels == ['split_label']
    finally:
        os.unlink(path)

    print("✓ SVA label auto-detection tests passed")


def test_sva_signal_extraction_preserves_source_order():
    """Test SVA dependency extraction keeps operand order stable."""
    from verilog_causal_analysis import VerilogParser

    parser = VerilogParser()

    signals = parser._extract_signals_from_text('~_controllerA_io_ack | _clientA_io_req')

    assert signals == ['_controllerA_io_ack', '_clientA_io_req']


def test_endpoint_direct_dependencies_are_preserved_before_recursion():
    """Test endpoint parents are not lost when the first branch exhausts max_nodes."""
    from verilog_causal_analysis import BackwardSlicer, Dependency, DependencyType

    class FakeParser:
        def build_dependency_graph(self):
            return {}

        def infer_module_from_signal(self, signal_name, hierarchy=''):
            return 'Top'

        def get_dependencies_for_signal(self, signal_name, module_name=None):
            if signal_name == 'fail':
                return [
                    Dependency(
                        source='req',
                        target='fail',
                        dep_type=DependencyType.ASSERTION,
                        expression='~ack | req',
                    ),
                    Dependency(
                        source='ack',
                        target='fail',
                        dep_type=DependencyType.ASSERTION,
                        expression='~ack | req',
                    ),
                ]
            if signal_name == 'req':
                return [
                    Dependency(
                        source='deep',
                        target='req',
                        dep_type=DependencyType.COMBINATIONAL,
                        expression='deep',
                    )
                ]
            return []

        def get_signal_sources(self, signal_name, module_name=None):
            return [(dep.source, dep.dep_type) for dep in self.get_dependencies_for_signal(signal_name, module_name)]

        def get_rtl_context(self, signal_name, module_name=None):
            return {'found': True, 'rtl_refs': []}

    class FakeWaveform:
        def __init__(self):
            self.values = {
                ('Top.fail', 1): '0',
                ('Top.req', 1): '0',
                ('Top.ack', 1): '1',
                ('Top.deep', 1): '0',
            }

        def get_signal_value(self, signal, cycle):
            return self.values.get((signal, cycle))

        def find_signal(self, signal, max_results=10):
            return [f'Top.{signal}']

    slicer = BackwardSlicer(FakeParser(), FakeWaveform(), max_depth=20, max_nodes=3)

    nodes, edges = slicer.slice_from_endpoint('Top.fail', 1)

    node_by_id = {node.id: node for node in nodes.values()}
    endpoint_edges = [
        edge for edge in edges
        if node_by_id[edge.dst_node_id].signal == 'Top.fail'
    ]
    endpoint_sources = {node_by_id[edge.src_node_id].signal for edge in endpoint_edges}

    assert endpoint_sources == {'Top.req', 'Top.ack'}


def test_expression_evaluator():
    """Test ExpressionEvaluator."""
    from verilog_causal_analysis import ExpressionEvaluator
    
    env = {
        'a': '1010',
        'b': '0011',
        'sel': '1'
    }
    
    evaluator = ExpressionEvaluator(env)
    
    # Test basic AND
    result = evaluator.evaluate('a & b')
    assert result == '0010', f"Expected '0010', got '{result}'"
    
    # Test basic OR
    result = evaluator.evaluate('a | b')
    assert result == '1011', f"Expected '1011', got '{result}'"
    
    # Test ternary
    result = evaluator.evaluate('sel ? a : b')
    assert result == '1010', f"Expected '1010', got '{result}'"
    
    print("✓ ExpressionEvaluator tests passed")


def test_utility_functions():
    """Test utility functions."""
    from verilog_causal_analysis import parse_binary_value, invert_value, values_differ
    
    # Test parse_binary_value
    assert parse_binary_value('1010') == 10
    assert parse_binary_value('0011') == 3
    assert parse_binary_value('x') is None
    
    # Test invert_value
    assert invert_value('1010') == '0101'
    assert invert_value('0') == '1'
    
    # Test values_differ
    assert values_differ('1010', '1011') == True
    assert values_differ('1010', '1010') == False
    
    print("✓ Utility function tests passed")


if __name__ == '__main__':
    print("Running verilog_causal_analysis tests...\n")
    
    test_imports()
    test_verilog_parser_basic()
    test_verilog_parser_dependencies()
    test_sva_label_auto_detect_split_line()
    test_expression_evaluator()
    test_utility_functions()
    
    print("\n✓ All tests passed!")
