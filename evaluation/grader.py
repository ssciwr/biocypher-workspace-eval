"""Grade one evaluation run against the ground truth.

Usage:
    uv run python -m evaluation.grader grade <run_dir>
    uv run python -m evaluation.grader validate     # reference + mutants

A run directory (written by evaluation.runner) holds meta.json,
events.jsonl and workspace/ (a copy of the session workspace). The grader
works on a fresh copy of workspace/, never on the model's original, and
writes grade.json.

SECURITY: grading executes model-written code (adapter, tests, build
script). Subprocesses get an allowlisted environment without credentials,
but they run with your user's file and network access; run the grader in a
container or VM for untrusted models.
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml

from evaluation import config as cfg
from evaluation import events as ev
from evaluation.mutants import MUTANTS, apply

PROBE = Path(__file__).with_name("probe.py")
ADAPTER_CLASS = "class SyntheticPpiAdapter"
FLAGS = (
    "is_directed",
    "is_stimulation",
    "is_inhibition",
    "consensus_direction",
    "consensus_stimulation",
    "consensus_inhibition",
)
NODE_PROP_TYPES = {"genesymbol": "str", "ncbi_tax_id": "int"}
EDGE_LABEL = "protein_protein_interaction"
EXPECTED_TEMPLATE = "https://github.com/biocypher/biocypher-cookiecutter-template"
OUTPUT_FILES = {
    "Protein-part000.parquet": 15,
    "PairwiseMolecularInteraction-part000.parquet": 21,
}
# Required boolean checks per level; ratio checks must additionally be 1.0.
ADAPTER_CHECKS = (
    "adapter_loads",
    "node_tuples_wellformed",
    "node_label_ok",
    "node_ids_match",
    "node_tuples_unique",
    "node_props_match",
    "symbols_upper",
    "edge_tuples_wellformed",
    "edge_label_ok",
    "edge_ids_match",
    "edge_tuples_unique",
    "edge_endpoints_match",
    "edge_props_match",
    "dedup_exact",
    "dedup_after_case",
    "undirected_kept",
    "no_reverse_added",
    "flags_unchanged",
    "bool_types",
    "reiterable",
)
BUILD_CHECKS = (
    "schema_ok",
    "build_ok",
    "output_counts_ok",
    "output_flag_types_ok",
    "pytest_passed",
    "tests_exercise_adapter",
    "pytest_run_after_last_change",
)
PROTOCOL_CHECKS = (
    "protocol_order_ok",
    "scaffold_layout_ok",
    "no_manual_scaffold",
    "no_questions",
)
REQUIRED = {
    "L0": ("called_cookiecutter_instructions", "template_reported_ok"),
    "L1": ("rows_ok", "columns_ok", "proteins_ok"),
    "L2": PROTOCOL_CHECKS,
    "L3": ADAPTER_CHECKS,
    "L4": ADAPTER_CHECKS + BUILD_CHECKS,
    "L5": ADAPTER_CHECKS + BUILD_CHECKS,
    "L6": PROTOCOL_CHECKS + ADAPTER_CHECKS + BUILD_CHECKS,
}
RATIOS = (
    "node_precision",
    "node_recall",
    "node_props_accuracy",
    "edge_precision",
    "edge_recall",
    "edge_props_accuracy",
)


# ------------------------------------------------------------ subprocesses


def _env(home: Path) -> dict[str, str]:
    """Credential-free environment for model-written code (cf. the registry's EXEC_ENV_ALLOWLIST)."""
    env = {
        k: os.environ[k] for k in ("PATH", "LANG", "LC_ALL", "TZ") if k in os.environ
    }
    home.mkdir(parents=True, exist_ok=True)
    env["HOME"] = str(home)
    env["UV_CACHE_DIR"] = os.environ.get(
        "UV_CACHE_DIR", str(Path.home() / ".cache" / "uv")
    )
    return env


def _run(
    argv: list[str], cwd: Path, home: Path, timeout: int
) -> subprocess.CompletedProcess:
    try:
        return subprocess.run(
            argv,
            cwd=cwd,
            env=_env(home),
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as e:
        return subprocess.CompletedProcess(
            argv, -9, e.stdout or "", f"timeout after {timeout}s"
        )


def _uv(project: Path, *args: str) -> list[str]:
    python = cfg.load()["pins"]["python"]
    return ["uv", "run", "--project", str(project), "--python", python, *args]


def _probe(project: Path, home: Path, timeout: int, *args: str) -> dict:
    proc = _run(_uv(project, "python", str(PROBE), *args), project, home, timeout)
    for line in reversed(proc.stdout.splitlines()):
        if line.startswith("@@PROBE@@"):
            return json.loads(line.removeprefix("@@PROBE@@"))
    return {"error": f"probe failed (exit {proc.returncode}): {proc.stderr[-2000:]}"}


# ------------------------------------------------------------ pure checks


def compare_adapter(probe: dict, truth: dict) -> dict:
    """Named checks of the adapter's tuples against the ground truth."""
    checks: dict = {"adapter_loads": probe.get("error") is None}
    if not checks["adapter_loads"]:
        checks["adapter_error"] = probe.get("error")
        checks.update(
            {name: False for name in ADAPTER_CHECKS if name != "adapter_loads"}
        )
        checks.update({name: 0.0 for name in RATIOS})
        return checks

    nodes, edges = probe["nodes"], probe["edges"]
    truth_nodes = {n["id"]: n for n in truth["nodes"]}
    truth_edges = {e["id"]: e for e in truth["edges"]}

    # nodes
    checks["node_tuples_wellformed"] = all(
        n.get("length") == 3 and "props" in n for n in nodes
    )
    node_ids = [n["fields"][0] for n in nodes if "fields" in n]
    checks["node_label_ok"] = bool(nodes) and all(
        n.get("fields", [None, None])[1:2] == ["protein"] for n in nodes
    )
    _set_ratios(checks, "node", set(node_ids), set(truth_nodes))
    checks["node_ids_match"] = set(node_ids) == set(truth_nodes)
    checks["node_tuples_unique"] = (
        len(node_ids) == len(set(node_ids)) == len(truth_nodes)
    )
    first_node = {}
    for n in nodes:
        if "fields" in n:
            first_node.setdefault(n["fields"][0], n.get("props", {}))
    hits = total = 0
    for node_id, expected in truth_nodes.items():
        props = first_node.get(node_id, {})
        for key, type_name in NODE_PROP_TYPES.items():
            total += 1
            hits += props.get(key) == [expected[key], type_name]
    checks["node_props_accuracy"] = hits / total
    checks["node_props_match"] = hits == total
    symbols = [
        p.get("genesymbol", [None])[0]
        for n in nodes
        if (p := n.get("props")) is not None
    ]
    checks["symbols_upper"] = bool(symbols) and all(
        isinstance(s, str) and s == s.upper() for s in symbols
    )

    # edges
    checks["edge_tuples_wellformed"] = all(
        e.get("length") == 5 and "props" in e for e in edges
    )
    fields = [e["fields"] for e in edges if "fields" in e and len(e["fields"]) == 5]
    edge_ids = [f[0] for f in fields]
    checks["edge_label_ok"] = bool(fields) and all(f[3] == EDGE_LABEL for f in fields)
    _set_ratios(checks, "edge", set(edge_ids), set(truth_edges))
    checks["edge_ids_match"] = set(edge_ids) == set(truth_edges)
    checks["edge_tuples_unique"] = (
        len(edge_ids) == len(set(edge_ids)) == len(truth_edges)
    )
    by_id = {}
    for e in edges:
        if "fields" in e and len(e["fields"]) == 5:
            by_id.setdefault(e["fields"][0], e)
    checks["edge_endpoints_match"] = all(
        (by_id.get(i, {}).get("fields") or [None] * 3)[1:3]
        == [t["source"], t["target"]]
        for i, t in truth_edges.items()
    )
    hits = total = 0
    for edge_id, expected in truth_edges.items():
        props = by_id.get(edge_id, {}).get("props", {})
        total += 1
        hits += props.get("type") == [expected["type"], "str"]
        for flag in FLAGS:
            total += 1
            hits += props.get(flag) == [expected[flag], "bool"]
    checks["edge_props_accuracy"] = hits / total
    checks["edge_props_match"] = hits == total

    # data-quality checks, by endpoints so they stay attributable when ids are wrong
    def pair(source: str, target: str) -> list[dict]:
        return [
            e.get("props", {})
            for e in edges
            if (e.get("fields") or [None] * 3)[1:3] == [source, target]
        ]

    checks["dedup_exact"] = len(pair("P04637", "Q9JHW1")) == 1
    checks["dedup_after_case"] = len(pair("P01106", "P04406")) == 1
    undirected = pair("P00533", "P08228")
    checks["undirected_kept"] = (
        len(undirected) == 1 and undirected[0].get("is_directed", [None])[0] is False
    )
    checks["no_reverse_added"] = not any(
        p.get("is_directed", [None])[0] in (False, 0) for p in pair("P08228", "P00533")
    )
    checks["flags_unchanged"] = len(undirected) == 1 and all(
        undirected[0].get(f) == [True, "bool"]
        for f in ("is_stimulation", "is_inhibition")
    )
    checks["bool_types"] = bool(edges) and all(
        (e.get("props") or {}).get(f, [None, None])[1] == "bool"
        for e in edges
        for f in FLAGS
    )
    checks["reiterable"] = bool(probe.get("reiterable"))
    return checks


def _set_ratios(checks: dict, kind: str, found: set, expected: set) -> None:
    hit = len(found & expected)
    checks[f"{kind}_precision"] = hit / len(found) if found else 0.0
    checks[f"{kind}_recall"] = hit / len(expected)


def check_schema(schema_file: Path) -> dict:
    try:
        schema = yaml.safe_load(schema_file.read_text()) or {}
    except (OSError, yaml.YAMLError) as e:
        return {"schema_ok": False, "schema_error": str(e)[:500]}
    protein = schema.get("protein") or {}
    edge = schema.get("pairwise molecular interaction") or {}
    ok = (
        protein.get("represented_as") == "node"
        and protein.get("input_label") == "protein"
        and edge.get("represented_as") == "edge"
        and edge.get("input_label") == EDGE_LABEL
    )
    return {"schema_ok": ok}


def check_output(parquet: dict) -> dict:
    files = parquet.get("files") or {}
    counts_ok = all(
        files.get(name, {}).get("rows") == n for name, n in OUTPUT_FILES.items()
    )
    edge_types = files.get("PairwiseMolecularInteraction-part000.parquet", {}).get(
        "types", {}
    )
    return {
        "output_counts_ok": counts_ok,
        "output_flag_types_ok": all(edge_types.get(f) == "bool" for f in FLAGS),
        "output_files": {k: v.get("rows") for k, v in files.items()},
    }


def parse_pytest(output: str) -> tuple[int, int]:
    """(passed, total) from a pytest summary line."""
    counts = {
        k: int(n) for n, k in re.findall(r"(\d+) (passed|failed|errors?)", output)
    }
    passed = counts.get("passed", 0)
    total = (
        passed
        + counts.get("failed", 0)
        + counts.get("error", 0)
        + counts.get("errors", 0)
    )
    return passed, total


def check_text_l0(text: str, calls: list[ev.ToolCall]) -> dict:
    match = re.search(r"TEMPLATE:\s*(\S+)", text)
    reported = (
        match.group(1).strip("`<>").removesuffix("/").removesuffix(".git")
        if match
        else None
    )
    return {
        "called_cookiecutter_instructions": any(
            c.name == "get_cookiecutter_instructions" for c in calls
        ),
        "template_reported_ok": reported == EXPECTED_TEMPLATE,
        "marker_present": match is not None,
    }


def check_text_l1(
    text: str, truth_columns: list[str], rows: int, proteins: int
) -> dict:
    def value(key: str) -> str | None:
        match = re.search(rf"{key}:\s*(.+)", text)
        return match.group(1).strip().strip("`") if match else None

    columns = value("COLUMNS")
    return {
        "rows_ok": value("ROWS") == str(rows),
        "columns_ok": columns is not None
        and [c.strip() for c in columns.split(",")] == truth_columns,
        "proteins_ok": value("PROTEINS") == str(proteins),
        "marker_present": value("ROWS") is not None,
    }


def check_protocol(
    events: list[dict],
    workspace: Path,
    expected_layout: list[str],
    project_dir: str,
    replies: int,
) -> dict:
    calls = ev.tool_calls(events)

    def first(pred) -> int | None:
        return next((c.index for c in calls if pred(c)), None)

    exists = first(lambda c: c.name == "check_project_exists")
    instructions = first(lambda c: c.name == "get_cookiecutter_instructions")
    scaffold = first(ev.is_cookiecutter_run)
    order_ok = (
        None not in (exists, instructions, scaffold)
        and exists < instructions < scaffold
    )
    manual = [
        c
        for c in calls
        if ev.is_file_change(c)
        and str(c.args.get("path", "")).lstrip("./").startswith(project_dir)
        and (scaffold is None or c.index < scaffold)
    ]
    project = workspace / project_dir
    return {
        "protocol_order_ok": order_ok,
        "scaffold_layout_ok": all((project / p).exists() for p in expected_layout),
        "no_manual_scaffold": scaffold is not None and not manual,
        "no_questions": replies == 0,
    }


def check_claims(text: str, passed: bool) -> dict:
    status = re.search(r"STATUS:\s*(DONE|FAILED)", text)
    tests = re.search(r"TESTS:\s*(\d+)\s*/\s*(\d+)", text)
    claimed = status.group(1) if status else None
    return {
        "claimed_status": claimed,
        "claimed_tests": f"{tests.group(1)}/{tests.group(2)}" if tests else None,
        "claim_accurate": claimed is not None and (claimed == "DONE") == passed,
        "overclaim": claimed == "DONE" and not passed,
    }


# ------------------------------------------------------------ project grading


def find_adapter(project: Path) -> tuple[Path, str | None] | None:
    """Adapter file and its module name (if under src/)."""
    for path in sorted(project.rglob("*.py")):
        if ".venv" in path.parts or "tests" in path.parts:
            continue
        try:
            if ADAPTER_CLASS in path.read_text(errors="ignore"):
                rel = path.relative_to(project)
                module = None
                if rel.parts[0] == "src":
                    module = ".".join(rel.with_suffix("").parts[1:])
                return path, module
        except OSError:
            continue
    return None


def grade_adapter(
    project: Path, data_file: Path, truth: dict, home: Path, timeout: int
) -> dict:
    found = find_adapter(project)
    if found is None:
        return compare_adapter({"error": "no file defines SyntheticPpiAdapter"}, truth)
    adapter_file, module = found
    args = ["adapter", str(adapter_file), str(data_file)] + ([module] if module else [])
    checks = compare_adapter(_probe(project, home, timeout, *args), truth)
    checks["adapter_file"] = str(adapter_file.relative_to(project))
    return checks


def grade_build(
    project: Path,
    events: list[dict],
    home: Path,
    timeout: int,
    reference_src: str | None,
) -> dict:
    checks = check_schema(project / "config" / "schema_config.yaml")
    shutil.rmtree(project / "biocypher-out", ignore_errors=True)
    build = _run(
        _uv(project, "python", "create_knowledge_graph.py"), project, home, timeout
    )
    checks["build_ok"] = build.returncode == 0
    if not checks["build_ok"]:
        checks["build_error"] = (build.stderr or build.stdout)[-1500:]
        checks.update(output_counts_ok=False, output_flag_types_ok=False)
    else:
        checks.update(
            check_output(
                _probe(
                    project, home, timeout, "parquet", str(project / "biocypher-out")
                )
            )
        )

    tests = _run(
        _uv(project, "python", "-m", "pytest", "-q", "-p", "no:cacheprovider"),
        project,
        home,
        timeout,
    )
    passed, total = parse_pytest(tests.stdout)
    checks.update(pytest_passed_count=passed, pytest_total=total)
    checks["pytest_passed"] = tests.returncode == 0 and total > 0
    test_text = (
        "\n".join(
            p.read_text(errors="ignore") for p in (project / "tests").rglob("*.py")
        )
        if (project / "tests").is_dir()
        else ""
    )
    checks["tests_exercise_adapter"] = (
        "get_edges" in test_text
        and "get_nodes" in test_text
        and ("synthetic_protein_interactions" in test_text)
    )
    calls = ev.tool_calls(events)
    last_change = max((c.index for c in calls if ev.is_file_change(c)), default=-1)
    checks["pytest_run_after_last_change"] = any(
        ev.is_pytest_run(c) and c.index > last_change for c in calls
    )

    pyproject = (
        (project / "pyproject.toml").read_text(errors="ignore")
        if (project / "pyproject.toml").exists()
        else ""
    )
    build_script = (
        (project / "create_knowledge_graph.py").read_text(errors="ignore")
        if (project / "create_knowledge_graph.py").exists()
        else ""
    )
    checks["defect_pyarrow_fixed"] = (
        "pyarrow" in pyproject or "biocypher[neo4j]" in pyproject
    )
    checks["defect_data_path_fixed"] = "your_data.csv" not in build_script
    if reference_src is not None and checks["pytest_passed"]:
        checks["test_strength"] = test_strength(project, home, timeout, reference_src)
    return checks


def test_strength(project: Path, home: Path, timeout: int, reference_src: str) -> float:
    """Fraction of mutants that make the project's own tests fail."""
    found = find_adapter(project)
    if found is None:
        return 0.0
    adapter_file = found[0]
    original = adapter_file.read_text()
    killed = 0
    try:
        for mutant in MUTANTS:
            adapter_file.write_text(apply(mutant, reference_src))
            proc = _run(
                _uv(
                    project,
                    "python",
                    "-m",
                    "pytest",
                    "-q",
                    "-x",
                    "-p",
                    "no:cacheprovider",
                ),
                project,
                home,
                timeout,
            )
            killed += proc.returncode != 0
    finally:
        adapter_file.write_text(original)
    return killed / len(MUTANTS)


def expected_layout(reference: Path) -> list[str]:
    """Files of the pristine scaffold (reference repo root commit)."""
    root = _root_commit(reference)
    out = subprocess.run(
        ["git", "-C", str(reference), "ls-tree", "-r", "--name-only", root],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return [p for p in out.splitlines() if not p.startswith(".")]


def _root_commit(reference: Path) -> str:
    return subprocess.run(
        ["git", "-C", str(reference), "rev-list", "--max-parents=0", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split()[0]


def reference_adapter_source(reference: Path) -> str:
    return (
        reference / "src" / "synthetic_ppi" / "adapters" / "synthetic_ppi_adapter.py"
    ).read_text()


def summarize(level: str, checks: dict) -> dict:
    required = REQUIRED[level]
    passed = all(checks.get(name) is True for name in required)
    if level in ("L3", "L4", "L5", "L6"):
        passed = passed and all(checks.get(r) == 1.0 for r in RATIOS)
        ratios = [checks.get(r, 0.0) for r in RATIOS]
        partial = sum(ratios) / len(ratios)
    else:
        partial = sum(checks.get(name) is True for name in required) / len(required)
    return {"passed": passed, "partial_score": round(partial, 4)}


def grade(run_dir: Path) -> dict:
    config = cfg.load()
    meta = json.loads((run_dir / "meta.json").read_text())
    level = cfg.levels(config)[meta["level"]]
    truth = json.loads((cfg.DATA_DIR / "ground_truth.json").read_text())
    events = ev.load(run_dir)
    text = ev.final_text(events)
    project_dir = config["pins"]["project_dir"]
    timeout = config["run"]["grade_timeout_seconds"]
    reference = cfg.reference_repo(config)

    with tempfile.TemporaryDirectory(prefix="grade-") as tmp:
        workspace = Path(tmp) / "workspace"
        if (run_dir / "workspace").exists():
            shutil.copytree(
                run_dir / "workspace",
                workspace,
                symlinks=True,
                ignore=shutil.ignore_patterns(".venv", "__pycache__"),
            )
        else:
            workspace.mkdir()
        home = Path(tmp) / "home"
        project = workspace / project_dir
        checks: dict = {}
        if level.name == "L0":
            checks.update(check_text_l0(text, ev.tool_calls(events)))
        elif level.name == "L1":
            raw = (cfg.DATA_DIR / "raw" / cfg.DATASET_FILE).read_text().splitlines()
            checks.update(
                check_text_l1(
                    text, raw[0].split("\t"), truth["raw_rows"], len(truth["nodes"])
                )
            )
        if level.name in ("L2", "L6"):
            checks.update(
                check_protocol(
                    events,
                    workspace,
                    expected_layout(reference),
                    project_dir,
                    meta.get("replies", 0),
                )
            )
        if level.name in ("L3", "L4", "L5", "L6"):
            checks.update(
                grade_adapter(project, level.dataset_file, truth, home, timeout)
            )
            seeded = project / "data" / cfg.DATASET_FILE
            checks["data_unmodified"] = seeded.exists() and cfg.sha256_file(
                seeded
            ) == cfg.sha256_file(level.dataset_file)
        if level.name in ("L4", "L5", "L6"):
            checks.update(
                grade_build(
                    project, events, home, timeout, reference_adapter_source(reference)
                )
            )
    result = {"level": level.name, "checks": checks, **summarize(level.name, checks)}
    if level.name not in ("L0", "L1"):
        result["checks"].update(check_claims(text, result["passed"]))
    (run_dir / "grade.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


# ------------------------------------------------------------ grader validation


def validate() -> bool:
    """Reference passes L5 adapter and build checks; each mutant fails its expected checks."""
    config = cfg.load()
    reference = cfg.reference_repo(config)
    truth = json.loads((cfg.DATA_DIR / "ground_truth.json").read_text())
    timeout = config["run"]["grade_timeout_seconds"]
    data = cfg.DATA_DIR / "original" / cfg.DATASET_FILE
    ok = True
    with tempfile.TemporaryDirectory(prefix="validate-") as tmp:
        project = Path(tmp) / "project"
        shutil.copytree(
            reference,
            project,
            ignore=shutil.ignore_patterns(
                ".venv", ".git", "biocypher-*", "__pycache__"
            ),
        )
        home = Path(tmp) / "home"
        checks = grade_adapter(project, data, truth, home, timeout)
        checks.update(grade_build(project, [], home, timeout, None))
        failing = sorted(
            n
            for n in ADAPTER_CHECKS + BUILD_CHECKS
            if checks.get(n) is not True and n != "pytest_run_after_last_change"
        ) + sorted(r for r in RATIOS if checks.get(r) != 1.0)
        print(f"reference: {'OK' if not failing else 'FAILS ' + ', '.join(failing)}")
        ok &= not failing
        source = reference_adapter_source(reference)
        # Informative: the reference tests should catch every mutant.
        strength = test_strength(project, home, timeout, source)
        print(
            f"reference test strength: {strength:.0%} of {len(MUTANTS)} mutants caught"
        )
        for mutant in MUTANTS:
            mutant_file = Path(tmp) / f"{mutant.name}.py"
            mutant_file.write_text(apply(mutant, source))
            mchecks = compare_adapter(
                _probe(project, home, timeout, "adapter", str(mutant_file), str(data)),
                truth,
            )
            failed = {n for n in ADAPTER_CHECKS if mchecks.get(n) is not True}
            missing = mutant.expected_failures - failed
            status = (
                "OK" if not missing else f"NOT DETECTED: {', '.join(sorted(missing))}"
            )
            print(
                f"mutant {mutant.name}: {status} (fails: {', '.join(sorted(failed))})"
            )
            ok &= not missing
    return ok


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    g = sub.add_parser("grade", help="grade one run directory")
    g.add_argument("run_dir", type=Path)
    sub.add_parser(
        "validate", help="validate the grader with the reference repo and mutants"
    )
    args = parser.parse_args(argv)
    if args.command == "grade":
        result = grade(args.run_dir)
        print(json.dumps({k: result[k] for k in ("level", "passed", "partial_score")}))
    else:
        sys.exit(0 if validate() else 1)


if __name__ == "__main__":
    main()
