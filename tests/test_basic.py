#!/usr/bin/env python3
"""
Basic tests for verilog_causal_analysis module.
"""

import sys
import os

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
    test_expression_evaluator()
    test_utility_functions()
    
    print("\n✓ All tests passed!")
