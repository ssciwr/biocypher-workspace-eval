"""Run the evaluation: models x levels x repetitions, round robin.

Usage:
    uv run python -m evaluation.runner [--models NAME ...] [--levels L3 L4]
        [--repetitions N] [--out evaluation/runs] [--no-grade]

Each run drives a real workspace session in-process (SessionManager from
the registry checkout's src/core/workspace/service.py, see backend.py; no
HTTP, no GitHub auth): it seeds the
workspace, sends the level prompt, records every event to events.jsonl,
enforces the call and time budgets, answers questions with the scripted
reply, copies the workspace, and grades it. Existing runs (with grade.json)
are skipped (and graded if they were not), so an interrupted evaluation can
be resumed.

API keys come from the environment variable each model names in
evaluation/config.toml (api_key_env).
"""

import argparse
import asyncio
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

from mcp import ClientSession
from mcp.client.streamable_http import create_mcp_http_client, streamable_http_client

from evaluation import backend
from evaluation import config as cfg
from evaluation import events as ev
from evaluation.grader import _root_commit, grade

cl, svc = backend.load()

DRAIN_SECONDS = 60


def git_state(repo: Path) -> dict:
    def git(*args: str) -> str:
        return subprocess.run(
            ["git", "-C", str(repo), *args], capture_output=True, text=True, check=False
        ).stdout.strip()

    return {
        "commit": git("rev-parse", "HEAD"),
        "dirty": bool(git("status", "--porcelain")),
    }


async def mcp_server_version(url: str) -> str:
    async with (
        create_mcp_http_client(headers={}) as http_client,
        streamable_http_client(url, http_client=http_client) as (read, write),
        ClientSession(read, write) as session,
    ):
        info = (await session.initialize()).server_info
        return f"{info.name} {info.version}"


def run(argv: list[str], cwd: Path | None = None) -> str:
    return subprocess.run(
        argv, cwd=cwd, capture_output=True, text=True, check=True
    ).stdout


def prepare_workspace(
    workspace: Path, level: cfg.Level, config: dict, reference: Path
) -> list[str]:
    """Create the run_command venv and seed scaffold/data; return the venv's packages."""
    python = config["pins"]["python"]
    venv = workspace / ".venv"
    run(["uv", "venv", "--quiet", "--python", python, str(venv)])
    run(
        [
            "uv",
            "pip",
            "install",
            "--quiet",
            "--python",
            str(venv / "bin" / "python"),
            *config["workspace_env"]["packages"],
        ]
    )
    packages = run(
        ["uv", "pip", "freeze", "--python", str(venv / "bin" / "python")]
    ).splitlines()

    data_dir = workspace / "data"
    if level.seed_scaffold:
        project = workspace / config["pins"]["project_dir"]
        project.mkdir()
        archive = subprocess.run(
            ["git", "-C", str(reference), "archive", _root_commit(reference)],
            capture_output=True,
            check=True,
        ).stdout
        subprocess.run(["tar", "-x", "-C", str(project)], input=archive, check=True)
        data_dir = project / "data"
    if level.dataset_file is not None:
        data_dir.mkdir(exist_ok=True)
        shutil.copy2(level.dataset_file, data_dir / cfg.DATASET_FILE)
    return packages


def configure_provider(model: cfg.Model, config: dict) -> str:
    """Point the workspace modules at this model; return its API key."""
    key = os.environ.get(model.api_key_env)
    if not key:
        raise SystemExit(
            f"{model.api_key_env} is not set (needed for model {model.name})"
        )
    if model.base_url:
        os.environ["ANTHROPIC_BASE_URL"] = model.base_url
    else:
        os.environ.pop("ANTHROPIC_BASE_URL", None)
    cl.MODEL = model.name
    cl.MAX_TOKENS = config["run"]["max_tokens"]
    cl.RESULT_MAX_CHARS = config["run"]["mcp_result_max_chars"]
    return key


def classify_error(message: str) -> str:
    if message == "interrupted":
        return "interrupted"
    if "rejected" in message:
        return "auth_error"
    if "out of credit" in message:
        return "credit_error"
    return "provider_error"


async def drive(
    session, level: cfg.Level, prompt: str, reply: str, max_replies: int, log
) -> dict:
    """Send the prompt and consume events until the run terminates."""
    queue = session.subscribe()
    start = time.monotonic()
    deadline = start + level.max_minutes * 60
    seen: list[dict] = []
    calls = replies = 0
    termination = None

    def record(event: dict) -> None:
        event = {**event, "t": round(time.monotonic() - start, 3)}
        seen.append(event)
        log.write(json.dumps(event) + "\n")

    session.submit(prompt)
    while True:
        try:
            event = await asyncio.wait_for(
                queue.get(), timeout=max(deadline - time.monotonic(), 0.01)
            )
        except TimeoutError:
            if termination is None:
                termination = "time_exhausted"
                session.interrupt()
                deadline = time.monotonic() + DRAIN_SECONDS
                continue
            break  # no terminal event after interrupting
        record(event)
        if event["type"] == "usage":
            calls += 1
            if calls >= level.max_calls and termination is None and session.busy:
                termination = "budget_exhausted"
                session.interrupt()
        elif event["type"] == "turn_error":
            termination = termination or classify_error(
                event["data"].get("message", "")
            )
            break
        elif event["type"] == "turn_done":
            if level.final_marker not in ev.final_text(seen) and replies < max_replies:
                replies += 1
                session.submit(reply)
                continue
            termination = "done"
            break
    session.unsubscribe(queue)
    return {
        "termination": termination,
        "calls": calls,
        "replies": replies,
        "wall_seconds": round(time.monotonic() - start, 1),
    }


async def run_one(
    model: cfg.Model, level: cfg.Level, rep: int, out: Path, config: dict, context: dict
) -> Path | None:
    """Run once and write meta.json; None if this run completed earlier."""
    run_dir = out / model.name / level.name / f"rep{rep:02d}"
    if (run_dir / "meta.json").exists():
        return None  # completed earlier
    shutil.rmtree(run_dir, ignore_errors=True)  # leftovers of an aborted run
    run_dir.mkdir(parents=True)
    key = configure_provider(model, config)
    reference = context["reference"]
    manager = svc.SessionManager(
        workspaces_root=out / ".workspaces", mcp_url=config["pins"]["mcp_url"]
    )
    session = await manager.create(owner_github_user_id="evaluation")
    try:
        packages = prepare_workspace(session.workspace, level, config, reference)
        session.set_key(key, None)
        prompt = level.prompt_file.read_text()
        # Opened off the event loop; the per-event writes are small.
        log = await asyncio.to_thread(open, run_dir / "events.jsonl", "w")
        try:
            outcome = await drive(
                session,
                level,
                prompt,
                config["run"]["question_reply"],
                config["run"]["max_replies"],
                log,
            )
        finally:
            log.close()
        shutil.copytree(
            session.workspace,
            run_dir / "workspace",
            symlinks=True,
            ignore=shutil.ignore_patterns(".venv", "__pycache__", ".cache"),
        )
    finally:
        await manager.shutdown()
    meta = {
        "model": model.name,
        "base_url": model.base_url or "anthropic",
        "level": level.name,
        "repetition": rep,
        "started_at": datetime.now(UTC).isoformat(timespec="seconds"),
        **outcome,
        "budgets": {"max_calls": level.max_calls, "max_minutes": level.max_minutes},
        "prompt_sha256": cfg.sha256_file(level.prompt_file),
        "dataset_sha256": cfg.sha256_file(level.dataset_file)
        if level.dataset_file
        else None,
        "config_sha256": config["sha256"],
        "thinking": cl.thinking_config(),
        "prompt_cache": cl.cache_control_config(),
        "max_tokens": cl.MAX_TOKENS,
        "mcp_result_max_chars": cl.RESULT_MAX_CHARS,
        "mcp_server": context["mcp_server"],
        "registry": context["registry"],
        "evaluation_repo": context["evaluation"],
        "reference_repo": context["reference_state"],
        "workspace_env": packages,
    }
    (run_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    return run_dir


async def main_async(args: argparse.Namespace) -> None:
    config = cfg.load()
    levels = cfg.levels(config)
    models = [m for m in cfg.models(config) if not args.models or m.name in args.models]
    selected = [levels[name] for name in (args.levels or cfg.LEVELS)]
    reference = cfg.reference_repo(config)
    # Workspace sessions run unsandboxed here; give run_command the per-run
    # venv (Python 3.13 + workspace_env packages) instead of this process's.
    svc._exec_bin = lambda workspace: workspace / ".venv" / "bin"
    context = {
        "reference": reference,
        "reference_state": git_state(reference),
        "registry": git_state(backend.repo_path(config)),
        "evaluation": git_state(cfg.REPO_ROOT),
        "mcp_server": await mcp_server_version(config["pins"]["mcp_url"]),
    }
    print(
        f"MCP server: {context['mcp_server']}; registry: {context['registry']}; reference: {context['reference_state']}"
    )
    repetitions = args.repetitions or config["run"]["repetitions"]
    for rep in range(repetitions):
        for level in selected:
            for model in models:
                label = f"{model.name} {level.name} rep{rep:02d}"
                run_dir = await run_one(model, level, rep, args.out, config, context)
                if run_dir is None:
                    run_dir = args.out / model.name / level.name / f"rep{rep:02d}"
                    if args.no_grade or (run_dir / "grade.json").exists():
                        print(f"{label}: done already, skipped")
                        continue
                meta = json.loads((run_dir / "meta.json").read_text())
                line = f"{label}: {meta['termination']}, {meta['calls']} calls, {meta['wall_seconds']}s"
                if not args.no_grade:
                    result = grade(run_dir)
                    line += (
                        f", passed={result['passed']} partial={result['partial_score']}"
                    )
                print(line, flush=True)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--models", nargs="*", help="model names from config.toml (default: all)"
    )
    parser.add_argument("--levels", nargs="*", choices=cfg.LEVELS, help="default: all")
    parser.add_argument(
        "--repetitions", type=int, help="default: config [run] repetitions"
    )
    parser.add_argument("--out", type=Path, default=cfg.EVAL_DIR / "runs")
    parser.add_argument(
        "--no-grade",
        action="store_true",
        help="only run; grade later with evaluation.grader",
    )
    args = parser.parse_args(argv)
    try:
        asyncio.run(main_async(args))
    except KeyboardInterrupt:
        sys.exit(130)


if __name__ == "__main__":
    main()
