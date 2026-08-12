from verilog_causal_analysis import build_structural_graph, canonical_json_bytes


def test_same_request_produces_byte_identical_graph(counter_request):
    first = build_structural_graph(counter_request)
    second = build_structural_graph(counter_request)
    assert canonical_json_bytes(first) == canonical_json_bytes(second)


def test_node_and_edge_ids_are_unique_and_stably_sorted(counter_request):
    graph = build_structural_graph(counter_request)
    node_ids = [row["node_id"] for row in graph["nodes"]]
    edge_ids = [row["edge_id"] for row in graph["edges"]]
    assert len(node_ids) == len(set(node_ids))
    assert len(edge_ids) == len(set(edge_ids))
    assert edge_ids == [
        row["edge_id"]
        for row in sorted(
            graph["edges"],
            key=lambda row: (
                row["dst_node_id"],
                row["src_node_id"],
                row["dependency_type"],
                row["edge_id"],
            ),
        )
    ]
