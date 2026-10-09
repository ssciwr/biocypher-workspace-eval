"""Compute per-run metrics and the model comparison report.

Usage:
    uv run python -m evaluation.metrics [runs_dir]    # default: evaluation/runs

Writes <runs_dir>/runs.csv (one row per run: metrics + grade) and
<runs_dir>/report.md (one row per model x level). Everything is recomputed
from meta.json, events.jsonl and grade.json, so metric definitions can
change without re-running models.
"""

import csv
import json
import math
import re
import statistics
import sys
from pathlib import Path

from evaluation import config as cfg
from evaluation import events as ev
from evaluation.grader import REQUIRED

RESULT_STATUS = re.compile(
    r"^\[(?:exit (\d+)|(Succeeded)|(Errored)|(Command usage error))\]"
)
LOOP_THRESHOLD = 3
RECOVERY_WINDOW = 3


def command_failed(result: dict) -> bool | None:
    """True/False for run_command results with an exit status, else None."""
    match = RESULT_STATUS.match(result.get("preview", ""))
    if not match:
        return None
    code, succeeded = match.group(1), match.group(2)
    return not (succeeded or code == "0")


def longest_repeat(calls: list[ev.ToolCall]) -> int:
    """Longest run of identical consecutive (tool, args) calls."""
    best = run = 0
    previous = None
    for call in calls:
        key = (call.name, json.dumps(call.args, sort_keys=True))
        run = run + 1 if key == previous else 1
        previous = key
        best = max(best, run)
    return best


def run_metrics(events: list[dict]) -> dict:
    calls = ev.tool_calls(events)
    results = ev.tool_results(events)
    usage = [e["data"] for e in events if e["type"] == "usage"]
    failures = [
        r.get("is_error")
        or (r.get("name") == "run_command" and command_failed(r) is True)
        for r in results
    ]
    # Pair results with calls in order (the service publishes them in the same order).
    recovered = attempts = 0
    for i, failed in enumerate(failures):
        if not failed or i >= len(calls):
            continue
        attempts += 1
        following = calls[i + 1 : i + 1 + RECOVERY_WINDOW]
        recovered += any(
            c.name == calls[i].name and c.args != calls[i].args for c in following
        )
    truncated = [
        i for i, r in enumerate(results) if "[truncated:" in r.get("preview", "")
    ]
    narrowed = sum(
        1
        for i in truncated
        if i + 1 < len(calls)
        and calls[i + 1].name == calls[i].name
        and calls[i + 1].args != calls[i].args
    )
    commands = [r for r in results if r.get("name") == "run_command"]
    return {
        "cycles": len(usage),
        "tool_calls": len(calls),
        "tool_error_rate": _ratio(
            sum(bool(r.get("is_error")) for r in results), len(results)
        ),
        "command_failure_rate": _ratio(
            sum(command_failed(r) is True for r in commands), len(commands)
        ),
        "longest_repeat": longest_repeat(calls),
        "loop": longest_repeat(calls) >= LOOP_THRESHOLD,
        "recovery_rate": _ratio(recovered, attempts),
        "truncation_narrowed_rate": _ratio(narrowed, len(truncated)),
        "input_tokens": sum(u.get("input") or 0 for u in usage),
        "output_tokens": sum(u.get("output") or 0 for u in usage),
        "peak_input_tokens": max((u.get("input") or 0 for u in usage), default=0),
        "calls_by_tool": json.dumps(_count(c.name for c in calls), sort_keys=True),
    }


def _ratio(num: int, den: int) -> float | None:
    return round(num / den, 4) if den else None


def _count(items) -> dict:
    out: dict = {}
    for item in items:
        out[item] = out.get(item, 0) + 1
    return out


def wilson(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion."""
    if n == 0:
        return (0.0, 0.0)
    p = successes / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return (max(0.0, centre - half), min(1.0, centre + half))


def collect(runs_dir: Path) -> list[dict]:
    rows = []
    for meta_file in sorted(runs_dir.glob("*/*/rep*/meta.json")):
        run_dir = meta_file.parent
        meta = json.loads(meta_file.read_text())
        grade_file = run_dir / "grade.json"
        grade = json.loads(grade_file.read_text()) if grade_file.exists() else {}
        checks = grade.get("checks", {})
        rows.append(
            {
                "model": meta["model"],
                "level": meta["level"],
                "repetition": meta["repetition"],
                "termination": meta["termination"],
                "replies": meta["replies"],
                "wall_seconds": meta["wall_seconds"],
                "graded": bool(grade),
                "passed": grade.get("passed"),
                "partial_score": grade.get("partial_score"),
                "overclaim": checks.get("overclaim"),
                "claim_accurate": checks.get("claim_accurate"),
                "test_strength": checks.get("test_strength"),
                **run_metrics(ev.load(run_dir)),
                "failed_checks": ";".join(
                    sorted(
                        k
                        for k in REQUIRED.get(meta["level"], ())
                        if checks and checks.get(k) is not True
                    )
                ),
            }
        )
    return rows


def _median(values: list) -> str:
    values = [v for v in values if v is not None]
    return f"{statistics.median(values):g}" if values else "–"


def report(rows: list[dict]) -> str:
    lines = [
        (
            "| Model | Level | n | success (95 % CI) | partial (median) | cycles, successful (median) "
            "| tool error rate (median) | loops | budget/time exhausted | overclaims | test strength (median) |"
        ),
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    groups: dict = {}
    for row in rows:
        if row["graded"]:
            groups.setdefault((row["model"], row["level"]), []).append(row)
    for (model, level), group in sorted(groups.items()):
        n = len(group)
        wins = sum(bool(r["passed"]) for r in group)
        low, high = wilson(wins, n)
        exhausted = sum(
            r["termination"] in ("budget_exhausted", "time_exhausted") for r in group
        )
        lines.append(
            f"| {model} | {level} | {n} | {wins}/{n} ({low:.0%}–{high:.0%}) "
            f"| {_median([r['partial_score'] for r in group])} "
            f"| {_median([r['cycles'] for r in group if r['passed']])} "
            f"| {_median([r['tool_error_rate'] for r in group])} "
            f"| {sum(bool(r['loop']) for r in group)} | {exhausted} "
            f"| {sum(bool(r['overclaim']) for r in group)} "
            f"| {_median([r['test_strength'] for r in group])} |"
        )
    failures: dict = {}
    for row in rows:
        for check in filter(None, row["failed_checks"].split(";")):
            failures.setdefault(row["model"], {}).setdefault(check, 0)
            failures[row["model"]][check] += 1
    lines += ["", "Failed checks per model (number of runs):", ""]
    for model, counts in sorted(failures.items()):
        top = ", ".join(
            f"{k} {v}" for k, v in sorted(counts.items(), key=lambda kv: -kv[1])
        )
        lines.append(f"- **{model}**: {top}")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    runs_dir = Path(argv[0]) if argv else cfg.EVAL_DIR / "runs"
    rows = collect(runs_dir)
    if not rows:
        raise SystemExit(f"no runs in {runs_dir}")
    with open(runs_dir / "runs.csv", "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (runs_dir / "report.md").write_text(report(rows))
    print(f"{len(rows)} runs -> {runs_dir / 'runs.csv'}, {runs_dir / 'report.md'}")


if __name__ == "__main__":
    main()
