"""Read a run's event log (events.jsonl) written by the runner.

Each line is one workspace event as published by
the registry's src/core/workspace/service.py ({"seq", "type", "data"}) plus "t", seconds
since the run started. Turn boundaries are "turn_started"; within a turn,
each LLM call ends with a "usage" event, after which its tool calls follow.
"""

import json
from dataclasses import dataclass
from pathlib import Path

COOKIECUTTER_MARKER = "biocypher-cookiecutter-template"


@dataclass(frozen=True)
class ToolCall:
    index: int  # position in the event list
    name: str
    args: dict


def load(run_dir: Path) -> list[dict]:
    path = run_dir / "events.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def final_text(events: list[dict]) -> str:
    """Text of the last assistant message of the last turn."""
    message: list[str] = []
    last = ""
    for event in events:
        if event["type"] == "turn_started":
            message, last = [], ""
        elif event["type"] == "text_delta":
            message.append(event["data"].get("text", ""))
        elif event["type"] == "usage":
            last = "".join(message)
            message = []
    return "".join(message) or last


def tool_calls(events: list[dict]) -> list[ToolCall]:
    return [
        ToolCall(i, e["data"].get("name", ""), e["data"].get("args") or {})
        for i, e in enumerate(events)
        if e["type"] == "tool_call"
    ]


def tool_results(events: list[dict]) -> list[dict]:
    return [e["data"] for e in events if e["type"] == "tool_result"]


def is_cookiecutter_run(call: ToolCall) -> bool:
    command = str(call.args.get("command", ""))
    return (
        call.name == "run_command"
        and COOKIECUTTER_MARKER in command
        and "install" not in command.split(COOKIECUTTER_MARKER)[0].split("&&")[-1]
    )


def is_pytest_run(call: ToolCall) -> bool:
    return call.name == "run_command" and "pytest" in str(call.args.get("command", ""))


def is_file_change(call: ToolCall) -> bool:
    return call.name in ("write_file", "edit_file")
