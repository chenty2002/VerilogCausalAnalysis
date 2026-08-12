from verilog_causal_analysis import (
    build_heuristic_feature_index,
    build_source_ranking,
)
from src.chiselspecflow.source_index import _attach_exact_origins


def test_feature_index_collapses_compiler_members_and_keeps_register_specificity():
    index = build_heuristic_feature_index(
        {
            "alias_classes": [],
            "expression_groups": [
                {
                    "expression_id": "expr",
                    "member_signals": ["Top._T_1", "Top._T_2"],
                    "leaf_inputs": ["Top.in"],
                }
            ],
            "register_transitions": [
                {
                    "register_id": "reg",
                    "signal": "Top.state",
                    "reset_rules": [],
                    "update_rules": [],
                }
            ],
            "aggregates": [],
            "handshakes": [],
            "pipelines": [],
        }
    )
    assert index.get("Top._T_1").source_group == "expr"
    assert index.get("Top._T_1").compiler_temporary is True
    assert index.get("Top.state").semantic_score == 1.0


def test_source_ranking_deduplicates_same_statement_role(tmp_path):
    source = tmp_path / "Foo.scala"
    source.write_text("val y = a\n")
    graph = {
        "graph_id": "g",
        "status": "complete",
        "search_summary": {"policy_id": "d2_backward_v1"},
        "signal_nodes": [],
        "semantic_nodes": [],
        "edges": [
            {
                "edge_id": "e1",
                "contribution_score": 0.8,
                "dependency_type": "combinational",
                "rtl_evidence": {"snippet": "x // Foo.scala:1:1", "condition": ""},
            },
            {
                "edge_id": "e2",
                "contribution_score": 0.8,
                "dependency_type": "combinational",
                "rtl_evidence": {"snippet": "y // Foo.scala:1:1", "condition": ""},
            },
        ],
    }
    ranking = build_source_ranking(
        graph,
        {"objects": [], "source_locators": [{"path": "Foo.scala", "line": 1, "column": 1}]},
        source_index={
            "objects": [],
            "statements": [
                {
                    "statement_id": "stmt_foo",
                    "statement_kind": "assignment",
                    "source_anchor": {"path": "Foo.scala", "line_start": 1, "line_end": 1},
                    "column_start": 1,
                    "syntax": "val y = a",
                    "semantic_object_ids": [],
                }
            ],
        },
        case_id="case",
        method="method",
        source_root=tmp_path,
    )
    row = ranking["ordering"][0]
    assert row["score"] == 0.8
    assert row["evidence_node_ids"] == ["e1", "e2"]
    assert row["max_contribution_score"] == 0.8
    assert row["statement_id"] == "stmt_foo"
    assert row["positive_authoritative_evidence"] is True


def test_object_name_fanout_is_discovery_only(tmp_path):
    ranking = build_source_ranking(
        {
            "graph_id": "g",
            "status": "complete",
            "search_summary": {"policy_id": "d2_backward_v1"},
            "signal_nodes": [
                {"node_id": "n", "signal": "Top.state", "suspect_score": 1.0}
            ],
            "semantic_nodes": [],
            "edges": [],
        },
        {},
        source_index={
            "objects": [{"object_id": "o", "name": "state"}],
            "statements": [
                {
                    "statement_id": statement_id,
                    "statement_kind": "assignment",
                    "source_anchor": {"path": "Foo.scala", "line_start": line, "line_end": line},
                    "semantic_object_ids": ["o"],
                }
                for statement_id, line in (("s1", 1), ("s2", 2))
            ],
        },
        case_id="case",
        method="d2",
        source_root=tmp_path,
    )
    assert all(not row["positive_authoritative_evidence"] for row in ranking["ordering"])
    assert ranking["authoritative_candidate_count"] == 0


def test_differential_slice_evidence_projects_to_parent_guard(tmp_path):
    clean = tmp_path / "clean.sv"
    faulty = tmp_path / "faulty.sv"
    clean.write_text("x = 1'b0; // Foo.scala:2:3\n")
    faulty.write_text("x = 1'b1; // Foo.scala:2:3\n")
    ranking = build_source_ranking(
        {
            "graph_id": "g",
            "status": "complete",
            "search_summary": {"policy_id": "d1"},
            "signal_nodes": [
                {"node_id": "n", "signal": "Top.counter", "cycle": 0, "is_endpoint": True}
            ],
            "semantic_nodes": [],
            "edges": [],
        },
        {},
        source_index={
            "objects": [{"object_id": "counter_object", "name": "counter"}],
            "statements": [
                {
                    "statement_id": "guard",
                    "statement_kind": "elaboration_guard",
                    "execution_phase": "elaboration",
                    "source_anchor": {"path": "Foo.scala", "line_start": 1, "line_end": 3},
                    "ancestor_statement_ids": [],
                    "child_statement_ids": ["update"],
                },
                {
                    "statement_id": "update",
                    "statement_kind": "register_update",
                    "execution_phase": "runtime",
                    "source_anchor": {"path": "Foo.scala", "line_start": 2, "line_end": 2},
                    "parent_statement_id": "guard",
                    "ancestor_statement_ids": ["guard"],
                    "child_statement_ids": [],
                    "semantic_object_ids": ["counter_object"],
                },
            ],
        },
        case_id="case",
        method="d1",
        source_root=tmp_path,
        clean_rtl=clean,
        faulty_rtl=faulty,
    )
    rows = {row["statement_id"]: row for row in ranking["ordering"]}
    assert rows["update"]["positive_authoritative_evidence"] is True
    assert rows["guard"]["positive_authoritative_evidence"] is True
    assert rows["guard"]["differential_evidence_ids"] == ["differential-node:n"]

    (tmp_path / "clean").mkdir()
    (tmp_path / "faulty").mkdir()
    clean = tmp_path / "clean/design.sv"
    faulty = tmp_path / "faulty/design.sv"
    clean.write_text(".MODE(0)\n")
    faulty.write_text(".MODE(1)\n")
    ranking = build_source_ranking(
        {
            "graph_id": "g",
            "status": "complete",
            "search_summary": {"policy_id": "d1"},
            "signal_nodes": [
                {
                    "node_id": "endpoint",
                    "signal": "Top.chiselcause_mismatch_any",
                    "is_endpoint": True,
                }
            ],
            "semantic_nodes": [],
            "edges": [],
        },
        {},
        source_index={
            "objects": [],
            "statements": [
                {
                    "statement_id": "parameter",
                    "statement_kind": "blackbox_parameter",
                    "execution_phase": "elaboration",
                    "source_anchor": {
                        "path": "Foo.scala",
                        "line_start": 4,
                        "line_end": 4,
                    },
                    "exact_origins": [{"path": "design.sv", "line": 1}],
                }
            ],
        },
        case_id="case",
        method="d1",
        source_root=tmp_path,
        clean_rtl=clean,
        faulty_rtl=faulty,
    )
    parameter = ranking["ordering"][0]
    assert parameter["positive_authoritative_evidence"] is True
    assert parameter["differential_evidence_ids"] == [
        "differential-unique-delta:endpoint"
    ]


def test_selected_table_origin_is_limited_to_updated_rows(tmp_path):
    rtl = tmp_path / "table.sv"
    rtl.write_text(
        "casez (selector)\n"
        "  y = 2'h0; // Foo.scala:8:1\n"
        "  y = 2'h1; // Foo.scala:8:1\n"
        "  y = 2'h2; // Foo.scala:8:1\n"
        "  y = 2'h3; // Foo.scala:8:1\n"
        "endcase\n"
    )
    index = {
        "statements": [
            {
                "execution_phase": "elaboration",
                "entity_kind": "table_update",
                "exact_origin_spec": {
                    "selection_parameter": "variantIndex",
                    "selection_value": 1,
                    "row_width": 2,
                    "updates": [{"row_expression": "1", "value_expression": "row"}],
                },
            }
        ]
    }
    _attach_exact_origins(
        index,
        {
            "commands": {"elaborate_argv": ["emit variantIndex=1"]},
            "generated_files": [{"path": "table.sv"}],
        },
        tmp_path,
    )
    assert {origin["line"] for origin in index["statements"][0]["exact_origins"]} == {4, 5}
