"""
Causal Graph Builder for Counterexample Analysis.

Main module that orchestrates the construction of a causal DAG from
FST waveforms and Verilog RTL code. The DAG represents causality relationships
where nodes are (signal, cycle, value) tuples and edges represent direct
causal influence under the counterexample execution.

Usage:
    from verilog_causal_analysis import CausalGraphBuilder
    
    builder = CausalGraphBuilder(
        fst_path="counterexample.fst",
        verilog_paths=["design.v", "testbench.v"],
        clock_signal="TestTop.clock"
    )
    
    result = builder.build(
        endpoint_signal="assertion_fail",
        endpoint_cycle=100  # or None to use last cycle
    )
    
    # Export to different formats
    builder.export_json("causal_graph.json")
    builder.export_dot("causal_graph.dot")
    builder.export_networkx("causal_graph_edges.csv")
"""

import os
import json
import time
import random
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Set, Tuple, Optional, Any

import graphviz

from .verilog_parser import VerilogParser, DependencyType
from .cycle_waveform import CycleAlignedWaveform
from .causal_slicer import BackwardSlicer, CausalNode, CausalEdge, ContributionType


# Version info for reproducibility
__version__ = "1.0.0"


@dataclass
class CausalGraphMeta:
    """Metadata for the causal graph."""
    fst_path: str
    verilog_paths: List[str]
    clock_signal: str
    endpoint_signal: str
    endpoint_cycle: int
    max_depth: int
    max_nodes: int
    generation_time: str
    runtime_seconds: float
    tool_version: str
    random_seed: int
    cycle_count: int
    timescale: int
    total_nodes: int
    total_edges: int
    root_nodes: int
    max_depth_reached: bool
    max_nodes_reached: bool
    undetermined_nodes: int
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class CausalGraphResult:
    """Result of causal graph construction."""
    nodes: List[Dict[str, Any]]
    edges: List[Dict[str, Any]]
    meta: CausalGraphMeta
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "nodes": self.nodes,
            "edges": self.edges,
            "meta": self.meta.to_dict()
        }


class CausalGraphBuilder:
    """
    Main class for constructing causal DAGs from waveforms and RTL.
    
    Given:
    - FST waveform file (counterexample trace)
    - Verilog source files (RTL design)
    - Clock signal name
    
    Builds a DAG where:
    - Nodes are (signal, cycle, value) events
    - Edges represent "direct cause under this counterexample"
    
    The graph is built via backward slicing from the counterexample
    endpoint, using expression-level counterfactual evaluation.
    """
    
    DEFAULT_MAX_DEPTH = 20
    DEFAULT_MAX_NODES = 200
    
    def __init__(self,
                 fst_path: str,
                 verilog_paths: List[str],
                 clock_signal: str = "clock",
                 max_depth: int = DEFAULT_MAX_DEPTH,
                 max_nodes: int = DEFAULT_MAX_NODES,
                 random_seed: Optional[int] = None):
        """
        Initialize the causal graph builder.
        
        Args:
            fst_path: Path to FST waveform file
            verilog_paths: List of paths to Verilog source files
            clock_signal: Hierarchical name of clock signal (e.g., "TestTop.clock")
            max_depth: Maximum backward traversal depth (default: 20)
            max_nodes: Maximum nodes in the DAG (default: 200)
            random_seed: Random seed for reproducibility
        """
        self.fst_path = os.path.abspath(fst_path)
        self.verilog_paths = [os.path.abspath(p) for p in verilog_paths]
        self.clock_signal = clock_signal
        self.max_depth = max_depth
        self.max_nodes = max_nodes
        
        # Set random seed for reproducibility
        self.random_seed = random_seed if random_seed is not None else int(time.time())
        random.seed(self.random_seed)
        
        # Initialize parsers (lazy)
        self._verilog_parser: Optional[VerilogParser] = None
        self._waveform: Optional[CycleAlignedWaveform] = None
        self._slicer: Optional[BackwardSlicer] = None
        
        # Result storage
        self._result: Optional[CausalGraphResult] = None
        self._nodes: Dict[str, CausalNode] = {}
        self._edges: List[CausalEdge] = []
    
    def _ensure_initialized(self):
        """Ensure all parsers are initialized."""
        if self._verilog_parser is None:
            self._verilog_parser = VerilogParser()
            for vpath in self.verilog_paths:
                if os.path.exists(vpath):
                    try:
                        self._verilog_parser.parse_file(vpath)
                    except Exception as e:
                        print(f"Warning: Failed to parse {vpath}: {e}")
        
        if self._waveform is None:
            self._waveform = CycleAlignedWaveform(self.fst_path, self.clock_signal)
        
        if self._slicer is None:
            self._slicer = BackwardSlicer(
                self._verilog_parser,
                self._waveform,
                max_depth=self.max_depth,
                max_nodes=self.max_nodes
            )
    
    def find_endpoint_signal(self, pattern: str = "assert") -> List[str]:
        """
        Find potential endpoint signals (assertion failures).
        
        Args:
            pattern: Pattern to match signal names
            
        Returns:
            List of matching signal names
        """
        self._ensure_initialized()
        assert self._waveform is not None  # Guaranteed by _ensure_initialized
        return self._waveform.find_signal(pattern, max_results=20)
    
    def get_last_cycle(self) -> int:
        """Get the last cycle in the waveform."""
        self._ensure_initialized()
        assert self._waveform is not None  # Guaranteed by _ensure_initialized
        return self._waveform.get_cycle_count() - 1
    
    def build(self,
              endpoint_signal: str,
              endpoint_cycle: Optional[int] = None) -> CausalGraphResult:
        """
        Build the causal graph from the endpoint.
        
        Args:
            endpoint_signal: Signal that triggered the counterexample
            endpoint_cycle: Cycle when triggered (None = last cycle)
            
        Returns:
            CausalGraphResult with nodes, edges, and metadata
        """
        start_time = time.time()
        
        self._ensure_initialized()
        assert self._waveform is not None  # Guaranteed by _ensure_initialized
        assert self._slicer is not None  # Guaranteed by _ensure_initialized
        
        # Determine endpoint cycle
        if endpoint_cycle is None:
            endpoint_cycle = self.get_last_cycle()
        
        # Perform backward slicing
        self._nodes, self._edges = self._slicer.slice_from_endpoint(
            endpoint_signal, endpoint_cycle
        )
        
        stats = self._slicer.get_statistics()
        runtime = time.time() - start_time
        
        # Convert to output format
        nodes_list = [node.to_dict() for node in self._nodes.values()]
        edges_list = [edge.to_dict() for edge in self._edges]
        
        # Sort nodes by depth (endpoint first) then by cycle
        nodes_list.sort(key=lambda n: (n["depth"], -n["cycle"]))
        
        # Count root nodes
        root_count = sum(1 for n in nodes_list if n.get("is_root", False))
        
        # Build metadata
        meta = CausalGraphMeta(
            fst_path=self.fst_path,
            verilog_paths=self.verilog_paths,
            clock_signal=self.clock_signal,
            endpoint_signal=endpoint_signal,
            endpoint_cycle=endpoint_cycle,
            max_depth=self.max_depth,
            max_nodes=self.max_nodes,
            generation_time=datetime.now().isoformat(),
            runtime_seconds=round(runtime, 3),
            tool_version=__version__,
            random_seed=self.random_seed,
            cycle_count=self._waveform.get_cycle_count(),
            timescale=self._waveform.timescale,
            total_nodes=len(nodes_list),
            total_edges=len(edges_list),
            root_nodes=root_count,
            max_depth_reached=stats["max_depth_reached"],
            max_nodes_reached=stats["max_nodes_reached"],
            undetermined_nodes=stats["undetermined_nodes"]
        )
        
        self._result = CausalGraphResult(
            nodes=nodes_list,
            edges=edges_list,
            meta=meta
        )
        
        return self._result
    
    def export_json(self, output_path: str, indent: int = 2) -> str:
        """
        Export the causal graph to JSON format.
        
        Args:
            output_path: Path to write JSON file
            indent: JSON indentation level
            
        Returns:
            Path to written file
        """
        if self._result is None:
            raise ValueError("No result to export. Call build() first.")
        
        output_path = os.path.abspath(output_path)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self._result.to_dict(), f, indent=indent, ensure_ascii=False)
        
        return output_path
    
    def export_dot(self, output_path: str) -> str:
        """
        Export the causal graph to GraphViz DOT format.
        
        Args:
            output_path: Path to write DOT file
            
        Returns:
            Path to written file
        """
        if self._result is None:
            raise ValueError("No result to export. Call build() first.")
        
        output_path = os.path.abspath(output_path)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        lines = [
            'digraph CausalGraph {',
            '    rankdir=BT;',  # Bottom to top (causes flow up to endpoint)
            '    node [shape=box, fontsize=10];',
            '    edge [fontsize=8];',
            '',
            '    // Legend subgraph',
            '    subgraph cluster_legend {',
            '        label="Legend";',
            '        fontsize=12;',
            '        style=dashed;',
            '        color=gray;',
            '',
            '        // Node style legend',
            '        legend_endpoint [label="Endpoint\\n(assertion fail)", style=filled, fillcolor=red, fontcolor=white];',
            '        legend_root [label="Root Cause\\nCandidate", style=filled, fillcolor=green];',
            '        legend_high_suspect [label="High Suspicion\\n(score > 0.7)", style=filled, fillcolor=orange];',
            '        legend_medium_suspect [label="Medium Suspicion\\n(score > 0.4)", style=filled, fillcolor=yellow];',
            '        legend_rtl_missing [label="RTL Context\\nMissing", style=dashed, color=gray];',
            '        legend_normal [label="Normal Node"];',
            '',
            '        // Edge style legend',
            '        legend_edge_expr [label="expr_eval", shape=plaintext];',
            '        legend_edge_toggle [label="toggle", shape=plaintext];',
            '        legend_edge_state [label="state", shape=plaintext];',
            '        legend_edge_cond [label="conditional", shape=plaintext];',
            '        legend_edge_other [label="other", shape=plaintext];',
            '',
            '        // Dummy edges for edge color legend (invisible nodes as targets)',
            '        legend_edge_expr -> legend_edge_expr_dst [label="expr_eval", color=blue, penwidth=2];',
            '        legend_edge_toggle -> legend_edge_toggle_dst [label="toggle", color=green, penwidth=2];',
            '        legend_edge_state -> legend_edge_state_dst [label="state", color=purple, penwidth=2];',
            '        legend_edge_cond -> legend_edge_cond_dst [label="conditional", color=orange, penwidth=2];',
            '        legend_edge_other -> legend_edge_other_dst [label="other", color=black, penwidth=2];',
            '',
            '        // Invisible target nodes for edge legend',
            '        legend_edge_expr_dst [label="", shape=point, width=0.1];',
            '        legend_edge_toggle_dst [label="", shape=point, width=0.1];',
            '        legend_edge_state_dst [label="", shape=point, width=0.1];',
            '        legend_edge_cond_dst [label="", shape=point, width=0.1];',
            '        legend_edge_other_dst [label="", shape=point, width=0.1];',
            '    }',
            ''
        ]
        
        # Add nodes
        for node in self._result.nodes:
            node_id = node["id"]
            label = f'{node["signal"]}\\n@{node["cycle"]}={node["value"]}'
            
            # Color coding
            if node.get("is_endpoint"):
                style = 'style=filled, fillcolor=red, fontcolor=white'
            elif node.get("is_root"):
                style = 'style=filled, fillcolor=green'
            elif node.get("rtl_context_missing"):
                style = 'style=dashed, color=gray'
            else:
                # Color by suspect score
                score = node.get("suspect_score", 0)
                if score > 0.7:
                    style = 'style=filled, fillcolor=orange'
                elif score > 0.4:
                    style = 'style=filled, fillcolor=yellow'
                else:
                    style = ''
            
            lines.append(f'    "{node_id}" [label="{label}", {style}];')
        
        lines.append('')
        
        # Add edges
        for edge in self._result.edges:
            src = edge["src_node_id"]
            dst = edge["dst_node_id"]
            score = edge.get("contribution_score", 0)
            contrib_type = edge.get("contribution_type", "")
            
            # Edge styling based on contribution type
            if contrib_type == "expr_eval":
                color = "blue"
            elif contrib_type == "toggle":
                color = "green"
            elif contrib_type == "state":
                color = "purple"
            elif contrib_type == "conditional":
                color = "orange"
            else:
                color = "black"
            
            # Edge thickness based on score
            penwidth = 1 + score * 2
            
            label = f'{contrib_type}\\n{score:.2f}'
            lines.append(f'    "{src}" -> "{dst}" [label="{label}", color={color}, penwidth={penwidth}];')
        
        lines.append('}')
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))
        
        return output_path
    
    def export_graph(self, output_path: str, format: str = 'png', dpi: int = 300) -> str:
        """
        Export the causal graph directly to an image file using Graphviz.
        
        This function uses the same logic as export_dot() but additionally
        renders the graph to a visual format (PNG, PDF, SVG, etc.) using
        the graphviz Python library.
        
        Args:
            output_path: Path to write image file (extension will be replaced by format)
            format: Output format ('png', 'pdf', 'svg', 'jpg', etc.)
            dpi: Resolution for raster formats (default: 300)
            
        Returns:
            Path to written file
        """
        if self._result is None:
            raise ValueError("No result to export. Call build() first.")
        
        output_path = os.path.abspath(output_path)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Remove extension from output_path as graphviz adds it
        base_path = os.path.splitext(output_path)[0]
        
        # Create Digraph object
        dot = graphviz.Digraph(
            'CausalGraph',
            comment='Causal Graph Analysis',
            format=format,
            engine='dot'
        )
        
        # Set graph attributes
        dot.attr(rankdir='BT')  # Bottom to top (causes flow up to endpoint)
        dot.attr('node', shape='box', fontsize='10')
        dot.attr('edge', fontsize='8')
        dot.attr(dpi=str(dpi))
        
        # Add nodes
        for node in self._result.nodes:
            node_id = node["id"]
            label = f'{node["signal"]}\\n@{node["cycle"]}={node["value"]}'
            
            # Determine node styling
            attrs = {}
            if node.get("is_endpoint"):
                attrs = {'style': 'filled', 'fillcolor': 'red', 'fontcolor': 'white'}
            elif node.get("is_root"):
                attrs = {'style': 'filled', 'fillcolor': 'green'}
            elif node.get("rtl_context_missing"):
                attrs = {'style': 'dashed', 'color': 'gray'}
            else:
                # Color by suspect score
                score = node.get("suspect_score", 0)
                if score > 0.7:
                    attrs = {'style': 'filled', 'fillcolor': 'orange'}
                elif score > 0.4:
                    attrs = {'style': 'filled', 'fillcolor': 'yellow'}
            
            dot.node(node_id, label=label, **attrs)
        
        # Add edges
        for edge in self._result.edges:
            src = edge["src_node_id"]
            dst = edge["dst_node_id"]
            score = edge.get("contribution_score", 0)
            contrib_type = edge.get("contribution_type", "")
            
            # Edge styling based on contribution type
            if contrib_type == "expr_eval":
                color = "blue"
            elif contrib_type == "toggle":
                color = "green"
            elif contrib_type == "state":
                color = "purple"
            elif contrib_type == "conditional":
                color = "orange"
            else:
                color = "black"
            
            # Edge thickness based on score
            penwidth = str(1 + score * 2)
            
            label = f'{contrib_type}\\n{score:.2f}'
            dot.edge(src, dst, label=label, color=color, penwidth=penwidth)
        
        # Render to file
        try:
            rendered_path = dot.render(base_path, cleanup=True)
            return rendered_path
        except Exception as e:
            raise RuntimeError(f"Failed to render graph: {e}")
    
    def export_networkx(self, output_path: str) -> str:
        """
        Export edges in CSV format for NetworkX loading.
        
        Format: src_id,dst_id,weight,type
        
        Args:
            output_path: Path to write CSV file
            
        Returns:
            Path to written file
        """
        if self._result is None:
            raise ValueError("No result to export. Call build() first.")
        
        output_path = os.path.abspath(output_path)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        lines = ['source,target,weight,type,reason']
        
        for edge in self._result.edges:
            src = edge["src_node_id"]
            dst = edge["dst_node_id"]
            weight = edge.get("contribution_score", 0)
            etype = edge.get("contribution_type", "unknown")
            reason = edge.get("reason", "").replace(',', ';').replace('\n', ' ')
            
            lines.append(f'{src},{dst},{weight},{etype},"{reason}"')
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))
        
        return output_path
    
    def export_node_attributes(self, output_path: str) -> str:
        """
        Export node attributes in CSV format for NetworkX.
        
        Format: id,signal,cycle,value,suspect_score,is_root,is_endpoint
        
        Args:
            output_path: Path to write CSV file
            
        Returns:
            Path to written file
        """
        if self._result is None:
            raise ValueError("No result to export. Call build() first.")
        
        output_path = os.path.abspath(output_path)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        lines = ['id,signal,cycle,value,suspect_score,is_root,is_endpoint,depth,rtl_missing']
        
        for node in self._result.nodes:
            node_id = node["id"]
            signal = node["signal"].replace(',', ';')
            cycle = node["cycle"]
            value = node["value"]
            score = node.get("suspect_score", 0)
            is_root = 1 if node.get("is_root") else 0
            is_endpoint = 1 if node.get("is_endpoint") else 0
            depth = node.get("depth", 0)
            rtl_missing = 1 if node.get("rtl_context_missing") else 0
            
            lines.append(f'{node_id},{signal},{cycle},{value},{score},{is_root},{is_endpoint},{depth},{rtl_missing}')
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))
        
        return output_path
    
    def get_natural_language_summary(self) -> str:
        """
        Generate a natural language summary of the causal graph.
        
        Returns:
            Human-readable summary string
        """
        if self._result is None:
            raise ValueError("No result to summarize. Call build() first.")
        
        meta = self._result.meta
        
        summary = [
            f"# Causal Graph Analysis Summary",
            f"",
            f"## Overview",
            f"- Endpoint: `{meta.endpoint_signal}` at cycle {meta.endpoint_cycle}",
            f"- Total nodes: {meta.total_nodes}",
            f"- Total edges: {meta.total_edges}",
            f"- Root cause candidates: {meta.root_nodes}",
            f"- Analysis depth: {meta.max_depth} (reached: {meta.max_depth_reached})",
            f"- Undetermined nodes (missing RTL): {meta.undetermined_nodes}",
            f"",
            f"## Root Cause Candidates",
        ]
        
        # List root nodes
        roots = [n for n in self._result.nodes if n.get("is_root")]
        roots.sort(key=lambda n: -n.get("suspect_score", 0))
        
        for i, root in enumerate(roots[:10], 1):
            summary.append(
                f"{i}. `{root['signal']}` @ cycle {root['cycle']} = {root['value']} "
                f"(score: {root.get('suspect_score', 0):.2f})"
            )
        
        if len(roots) > 10:
            summary.append(f"   ... and {len(roots) - 10} more")
        
        summary.extend([
            f"",
            f"## High-Suspicion Paths",
        ])
        
        # Find paths with high contribution scores
        high_score_edges = [e for e in self._result.edges if e.get("contribution_score", 0) > 0.7]
        high_score_edges.sort(key=lambda e: -e.get("contribution_score", 0))
        
        for edge in high_score_edges[:5]:
            # Find source and dest nodes
            src_node = next((n for n in self._result.nodes if n["id"] == edge["src_node_id"]), None)
            dst_node = next((n for n in self._result.nodes if n["id"] == edge["dst_node_id"]), None)
            
            if src_node and dst_node:
                summary.append(
                    f"- `{src_node['signal']}@{src_node['cycle']}` → "
                    f"`{dst_node['signal']}@{dst_node['cycle']}` "
                    f"(score: {edge.get('contribution_score', 0):.2f}, type: {edge.get('contribution_type')})"
                )
                
                # Add evidence if available
                evidence = edge.get("evidence", {})
                if evidence.get("code_snippet"):
                    snippet = evidence["code_snippet"].strip().split('\n')[0][:60]
                    summary.append(f"  RTL: `{snippet}...`")
        
        summary.extend([
            f"",
            f"## Generation Info",
            f"- Runtime: {meta.runtime_seconds:.3f}s",
            f"- Tool version: {meta.tool_version}",
            f"- Random seed: {meta.random_seed}",
            f"- Generated: {meta.generation_time}",
        ])
        
        return '\n'.join(summary)
    
    def get_evidence_for_node(self, node_id: str) -> Dict[str, Any]:
        """
        Get detailed evidence for a specific node.
        
        Args:
            node_id: Node ID
            
        Returns:
            Dictionary with evidence details
        """
        if self._result is None:
            raise ValueError("No result available. Call build() first.")
        
        node = next((n for n in self._result.nodes if n["id"] == node_id), None)
        if not node:
            return {"error": f"Node not found: {node_id}"}
        
        # Find all edges involving this node
        incoming = [e for e in self._result.edges if e["dst_node_id"] == node_id]
        outgoing = [e for e in self._result.edges if e["src_node_id"] == node_id]
        
        return {
            "node": node,
            "incoming_edges": incoming,
            "outgoing_edges": outgoing,
            "rtl_refs": node.get("rtl_refs", []),
            "is_root": node.get("is_root", False),
            "is_endpoint": node.get("is_endpoint", False)
        }
    
    def close(self):
        """Clean up resources."""
        if self._waveform is not None:
            self._waveform.close()
            self._waveform = None
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False


def build_causal_graph(fst_path: str,
                       verilog_paths: List[str],
                       clock_signal: str,
                       endpoint_signal: Optional[str] = None,
                       endpoint_cycle: Optional[int] = None,
                       max_depth: int = CausalGraphBuilder.DEFAULT_MAX_DEPTH,
                       max_nodes: int = CausalGraphBuilder.DEFAULT_MAX_NODES,
                       output_dir: Optional[str] = None) -> CausalGraphResult:
    """
    Convenience function to build a causal graph.
    
    Args:
        fst_path: Path to FST waveform file
        verilog_paths: List of Verilog source files
        clock_signal: Clock signal name
        endpoint_signal: Signal that triggered counterexample (auto-detect if None)
        endpoint_cycle: Cycle of trigger (last cycle if None)
        max_depth: Maximum traversal depth
        max_nodes: Maximum nodes in DAG
        output_dir: Directory to write output files (None = no file output)
        
    Returns:
        CausalGraphResult
    """
    with CausalGraphBuilder(
        fst_path=fst_path,
        verilog_paths=verilog_paths,
        clock_signal=clock_signal,
        max_depth=max_depth,
        max_nodes=max_nodes
    ) as builder:
        
        # Auto-detect endpoint if not specified
        if endpoint_signal is None:
            candidates = builder.find_endpoint_signal("assert")
            if not candidates:
                candidates = builder.find_endpoint_signal("fail")
            if not candidates:
                candidates = builder.find_endpoint_signal("error")
            if candidates:
                endpoint_signal = candidates[0]
            else:
                raise ValueError("Could not auto-detect endpoint signal. Please specify endpoint_signal.")
        
        # Build the graph
        result = builder.build(endpoint_signal, endpoint_cycle)
        
        # Export files if output directory specified
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            builder.export_json(os.path.join(output_dir, "causal_graph.json"))
            builder.export_dot(os.path.join(output_dir, "causal_graph.dot"))
            builder.export_networkx(os.path.join(output_dir, "causal_edges.csv"))
            builder.export_node_attributes(os.path.join(output_dir, "causal_nodes.csv"))
            
            # Write summary
            summary = builder.get_natural_language_summary()
            with open(os.path.join(output_dir, "summary.md"), 'w') as f:
                f.write(summary)
        
        return result
