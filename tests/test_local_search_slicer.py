from verilog_causal_analysis.causal_slicer import BackwardSlicer
from verilog_causal_analysis import SearchSeed
from verilog_causal_analysis.verilog_parser import Dependency, DependencyType


class FakeParser:
    def __init__(self, reverse=False):
        self.reverse = reverse

    def build_dependency_graph(self):
        return {}

    def get_dependencies_for_signal(self, signal, module_name=None):
        base = signal.rsplit(".", 1)[-1]
        rows = {
            "fail": [
                Dependency(source="a", target="fail", dep_type=DependencyType.COMBINATIONAL, expression="a & b"),
                Dependency(source="b", target="fail", dep_type=DependencyType.COMBINATIONAL, expression="a & b"),
            ],
            "a": [
                Dependency(source="root_a", target="a", dep_type=DependencyType.COMBINATIONAL, expression="root_a"),
            ],
            "b": [
                Dependency(source="root_b", target="b", dep_type=DependencyType.COMBINATIONAL, expression="root_b"),
            ],
        }.get(base, [])
        return list(reversed(rows)) if self.reverse else rows

    def get_signal_sources(self, signal_name, module_name=None):
        return [(row.source, row.dep_type) for row in self.get_dependencies_for_signal(signal_name, module_name)]

    def get_rtl_context(self, signal_name, module_name=None):
        return {"found": True, "rtl_refs": []}


class FakeWaveform:
    def __init__(self):
        self.values = {
            ("Top.fail", 1): "1",
            ("Top.a", 1): "1",
            ("Top.b", 1): "1",
            ("Top.root_a", 1): "1",
            ("Top.root_b", 1): "1",
        }

    def get_signal_value(self, signal, cycle):
        return self.values.get((signal, cycle))

    def find_signal(self, signal, max_results=10):
        candidate = f"Top.{signal}"
        return [candidate] if (candidate, 1) in self.values else []


def topology(nodes, edges):
    signals = {node_id: node.signal for node_id, node in nodes.items()}
    return sorted((signals[edge.src_node_id], signals[edge.dst_node_id], edge.contribution_score) for edge in edges)


def test_d2_backward_is_independent_of_dependency_enumeration():
    outputs = []
    for reverse in (False, True):
        slicer = BackwardSlicer(
            FakeParser(reverse), FakeWaveform(), max_nodes=8,
            search_policy="d2_backward_v1",
        )
        nodes, edges = slicer.slice_from_endpoint("Top.fail", 1)
        outputs.append((topology(nodes, edges), slicer.get_statistics()))
        assert all(edge.contribution_evidence["schema_version"] == "contribution_evidence_v2" for edge in edges)
    assert outputs[0][0] == outputs[1][0]
    for key in ("expanded_nodes", "candidate_evaluations", "intervention_evaluations"):
        assert outputs[0][1][key] == outputs[1][1][key]


def test_intervention_budget_truncation_is_inconclusive():
    slicer = BackwardSlicer(
        FakeParser(), FakeWaveform(), max_nodes=9,
        search_policy="d2_backward_v1", max_intervention_evaluations=0,
    )
    _nodes, edges = slicer.slice_from_endpoint("Top.fail", 1)
    stats = slicer.get_statistics()
    assert stats["intervention_evaluation_budget_reached"] is True
    assert edges
    assert all(edge.contribution_evidence["status"] == "inconclusive" for edge in edges)


def test_multi_seed_search_shares_one_candidate_budget():
    slicer = BackwardSlicer(
        FakeParser(), FakeWaveform(), max_nodes=9,
        search_policy="d2_backward_v1", max_candidate_evaluations=1,
    )
    slicer.slice_from_seeds(
        [
            SearchSeed("endpoint", "Top.fail", 1, "exact_endpoint", 1.0, 0),
            SearchSeed("member", "Top.a", 1, "exact_predicate_member", 0.9, 1),
        ]
    )
    stats = slicer.get_statistics()
    assert stats["seed_count"] == 2
    assert stats["candidate_evaluations"] == 1
    assert stats["termination_reason"] == "max_candidate_evaluations"


def test_inactive_case_dependencies_do_not_consume_candidate_budget():
    class Parser(FakeParser):
        def get_dependencies_for_signal(self, signal, module_name=None):
            if signal.rsplit(".", 1)[-1] != "fail":
                return []
            return [
                Dependency(
                    source="data",
                    target="fail",
                    dep_type=DependencyType.COMBINATIONAL,
                    expression="data",
                    condition="sel",
                ),
                Dependency(
                    source="sel",
                    target="fail",
                    dep_type=DependencyType.COMBINATIONAL,
                    expression="data",
                    condition="sel",
                ),
                Dependency(
                    source="seq_guard",
                    target="fail",
                    dep_type=DependencyType.SEQUENTIAL,
                    expression="1'b1",
                    condition="seq_guard",
                ),
            ]

    waveform = FakeWaveform()
    waveform.values.update(
        {
            ("Top.data", 1): "1",
            ("Top.sel", 1): "0",
            ("Top.seq_guard", 0): "0",
            ("Top.fail", 0): "0",
        }
    )
    slicer = BackwardSlicer(
        Parser(), waveform, max_nodes=4,
        search_policy="d2_backward_v1", max_candidate_evaluations=1,
    )
    nodes, _edges = slicer.slice_from_endpoint("Top.fail", 1)
    stats = slicer.get_statistics()
    assert stats["candidate_evaluations"] == 1
    assert stats["termination_reason"] == "frontier_exhausted"
    assert "Top.data" not in {node.signal for node in nodes.values()}


def test_sequential_guard_intervention_evaluates_update_or_hold():
    class Parser(FakeParser):
        def get_dependencies_for_signal(self, signal, module_name=None):
            if signal.rsplit(".", 1)[-1] != "counter":
                return []
            return [
                Dependency(
                    source="enable",
                    target="counter",
                    dep_type=DependencyType.SEQUENTIAL,
                    expression="counter + 4'h2",
                    condition="enable",
                )
            ]

    waveform = FakeWaveform()
    waveform.values.update(
        {
            ("Top.counter", 0): "0000",
            ("Top.counter", 1): "0010",
            ("Top.enable", 0): "1",
            ("Top.enable", 1): "1",
        }
    )
    slicer = BackwardSlicer(
        Parser(), waveform, max_nodes=4, search_policy="d2_backward_v1"
    )
    nodes, edges = slicer.slice_from_endpoint("Top.counter", 1)

    edge = edges[0]
    assert nodes[edge.src_node_id].cycle == 0
    assert edge.contribution_evidence["method"] == "active_rule_intervention"
    assert edge.contribution_evidence["fidelity"]["status"] == "exact_match"
    assert edge.contribution_score > 0
