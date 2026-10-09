"""Grader checks on synthetic probe results; no subprocesses or network."""

import copy
import json

import pytest

from evaluation import config as cfg
from evaluation import events as ev
from evaluation import grader

TRUTH = json.loads((cfg.DATA_DIR / "ground_truth.json").read_text())


def reference_probe() -> dict:
    """What probe.py reports for a correct adapter."""
    nodes = [
        {
            "length": 3,
            "fields": [n["id"], "protein", "{...}"],
            "props": {
                "genesymbol": [n["genesymbol"], "str"],
                "ncbi_tax_id": [n["ncbi_tax_id"], "int"],
            },
        }
        for n in TRUTH["nodes"]
    ]
    edges = [
        {
            "length": 5,
            "fields": [
                e["id"],
                e["source"],
                e["target"],
                "protein_protein_interaction",
                "{...}",
            ],
            "props": {
                "type": [e["type"], "str"],
                **{f: [e[f], "bool"] for f in grader.FLAGS},
            },
        }
        for e in TRUTH["edges"]
    ]
    return {"error": None, "nodes": nodes, "edges": edges, "reiterable": True}


def failing(checks: dict) -> set[str]:
    return {name for name in grader.ADAPTER_CHECKS if checks.get(name) is not True}


def edge(probe: dict, edge_id: str) -> dict:
    return next(e for e in probe["edges"] if e["fields"][0] == edge_id)


def test_reference_passes_everything():
    checks = grader.compare_adapter(reference_probe(), TRUTH)
    assert failing(checks) == set()
    assert all(checks[r] == 1.0 for r in grader.RATIOS)


def test_adapter_error_fails_all_and_zeroes_ratios():
    checks = grader.compare_adapter({"error": "ImportError: boom"}, TRUTH)
    assert failing(checks) == set(grader.ADAPTER_CHECKS)
    assert all(checks[r] == 0.0 for r in grader.RATIOS)
    assert checks["adapter_error"] == "ImportError: boom"


def test_int_flags_detected():
    probe = reference_probe()
    for e in probe["edges"]:
        for f in grader.FLAGS:
            e["props"][f] = [int(e["props"][f][0]), "int"]
    assert {"bool_types", "flags_unchanged", "edge_props_match"} <= failing(
        grader.compare_adapter(probe, TRUTH)
    )


def test_duplicate_edge_detected():
    probe = reference_probe()
    probe["edges"].append(copy.deepcopy(edge(probe, "P04637_Q9JHW1_ubiquitination")))
    assert {"dedup_exact", "edge_tuples_unique"} <= failing(
        grader.compare_adapter(probe, TRUTH)
    )


def test_lower_case_symbol_detected():
    probe = reference_probe()
    next(n for n in probe["nodes"] if n["fields"][0] == "P04406")["props"][
        "genesymbol"
    ][0] = "gapdh"
    assert {"symbols_upper", "node_props_match"} <= failing(
        grader.compare_adapter(probe, TRUTH)
    )


def test_reverse_edge_for_undirected_row_detected():
    probe = reference_probe()
    reverse = copy.deepcopy(edge(probe, "P00533_P08228_binding"))
    reverse["fields"][1:3] = ["P08228", "P00533"]
    probe["edges"].append(reverse)
    assert {"no_reverse_added"} <= failing(grader.compare_adapter(probe, TRUTH))


def test_not_reiterable_detected():
    probe = reference_probe()
    probe["reiterable"] = False
    assert failing(grader.compare_adapter(probe, TRUTH)) == {"reiterable"}


def test_missing_edges_lower_recall():
    probe = reference_probe()
    probe["edges"] = probe["edges"][:10]
    checks = grader.compare_adapter(probe, TRUTH)
    assert checks["edge_precision"] == 1.0
    assert checks["edge_recall"] == pytest.approx(10 / 21)


def test_check_output():
    good = {
        "files": {
            "Protein-part000.parquet": {"rows": 15, "types": {}},
            "PairwiseMolecularInteraction-part000.parquet": {
                "rows": 21,
                "types": {f: "bool" for f in grader.FLAGS},
            },
        }
    }
    assert grader.check_output(good)["output_counts_ok"] is True
    assert grader.check_output(good)["output_flag_types_ok"] is True
    bad = copy.deepcopy(good)
    bad["files"]["PairwiseMolecularInteraction-part000.parquet"]["types"][
        "is_directed"
    ] = "int64"
    assert grader.check_output(bad)["output_flag_types_ok"] is False


def test_check_schema(tmp_path):
    schema = tmp_path / "schema_config.yaml"
    schema.write_text(
        "protein:\n  represented_as: node\n  input_label: protein\n"
        "pairwise molecular interaction:\n  represented_as: edge\n"
        "  input_label: protein_protein_interaction\n"
    )
    assert grader.check_schema(schema) == {"schema_ok": True}
    schema.write_text("protein:\n  represented_as: node\n  input_label: protein\n")
    assert grader.check_schema(schema) == {"schema_ok": False}


@pytest.mark.parametrize(
    ("output", "expected"),
    [
        ("11 passed in 2.78s", (11, 11)),
        ("2 failed, 5 passed in 1.0s", (5, 7)),
        ("1 failed, 3 passed, 1 error in 1.0s", (3, 5)),
        ("no tests ran", (0, 0)),
    ],
)
def test_parse_pytest(output, expected):
    assert grader.parse_pytest(output) == expected


def test_l0_text():
    calls = [ev.ToolCall(0, "get_cookiecutter_instructions", {})]
    text = "...\nTEMPLATE: https://github.com/biocypher/biocypher-cookiecutter-template.git"
    assert grader.check_text_l0(text, calls) == {
        "called_cookiecutter_instructions": True,
        "template_reported_ok": True,
        "marker_present": True,
    }
    assert (
        grader.check_text_l0("TEMPLATE: gh:someone/else", [])["template_reported_ok"]
        is False
    )


def test_l1_text():
    columns = ["source", "target", "type"]
    text = "ROWS: 23\nCOLUMNS: source, target, type\nPROTEINS: 15"
    assert all(grader.check_text_l1(text, columns, 23, 15).values())
    assert (
        grader.check_text_l1(text.replace("23", "22"), columns, 23, 15)["rows_ok"]
        is False
    )


def test_claims():
    assert (
        grader.check_claims("STATUS: DONE\nTESTS: 4/4", passed=False)["overclaim"]
        is True
    )
    claims = grader.check_claims("STATUS: FAILED\nTESTS: 1/4", passed=False)
    assert claims["claim_accurate"] is True
    assert claims["claimed_tests"] == "1/4"
    assert grader.check_claims("all good", passed=True)["claim_accurate"] is False


def _call(name, **args):
    return {"type": "tool_call", "data": {"name": name, "args": args}}


COOKIECUTTER = "cookiecutter https://github.com/biocypher/biocypher-cookiecutter-template.git --no-input"


def test_protocol_order_and_layout(tmp_path):
    (tmp_path / "synthetic-ppi" / "config").mkdir(parents=True)
    (tmp_path / "synthetic-ppi" / "config" / "schema_config.yaml").write_text("")
    events = [
        _call("check_project_exists"),
        _call("get_cookiecutter_instructions"),
        _call("run_command", command="pip install cookiecutter"),
        _call("run_command", command=COOKIECUTTER),
        _call("write_file", path="synthetic-ppi/README.md"),
    ]
    checks = grader.check_protocol(
        events, tmp_path, ["config/schema_config.yaml"], "synthetic-ppi", replies=0
    )
    assert checks == {
        "protocol_order_ok": True,
        "scaffold_layout_ok": True,
        "no_manual_scaffold": True,
        "no_questions": True,
    }


def test_protocol_violations(tmp_path):
    events = [
        _call("get_cookiecutter_instructions"),
        _call("write_file", path="synthetic-ppi/pyproject.toml"),
        _call("check_project_exists"),
    ]
    checks = grader.check_protocol(
        events, tmp_path, ["pyproject.toml"], "synthetic-ppi", replies=1
    )
    assert not any(checks.values())


def test_summarize_requires_ratios():
    checks = {name: True for name in grader.ADAPTER_CHECKS} | {
        r: 1.0 for r in grader.RATIOS
    }
    assert grader.summarize("L3", checks) == {"passed": True, "partial_score": 1.0}
    checks["edge_recall"] = 0.5
    assert grader.summarize("L3", checks)["passed"] is False
