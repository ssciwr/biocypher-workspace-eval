"""Run metrics from synthetic event logs."""

import pytest

from evaluation import metrics


def call(name, **args):
    return {"type": "tool_call", "data": {"name": name, "args": args}}


def result(name, preview="", is_error=False):
    return {
        "type": "tool_result",
        "data": {"name": name, "preview": preview, "is_error": is_error},
    }


def usage(inp, out):
    return {
        "type": "usage",
        "data": {"input": inp, "output": out, "cache_read": 0, "cache_write": 0},
    }


def test_run_metrics():
    events = [
        {"type": "turn_started", "data": {}},
        usage(100, 10),
        call("run_command", command="pytest"),
        result("run_command", "[Errored]\nFAILED"),
        usage(150, 20),
        call("run_command", command="pytest -x"),
        result("run_command", "[Succeeded]\n3 passed"),
        usage(200, 5),
        {"type": "turn_done", "data": {}},
    ]
    m = metrics.run_metrics(events)
    assert m["cycles"] == 3
    assert m["tool_calls"] == 2
    assert m["command_failure_rate"] == 0.5
    assert m["tool_error_rate"] == 0.0
    assert m["recovery_rate"] == 1.0
    assert m["input_tokens"] == 450
    assert m["peak_input_tokens"] == 200
    assert m["loop"] is False


def test_loop_detected():
    events = [call("read_file", path="a.py")] * 3
    assert metrics.run_metrics(events)["longest_repeat"] == 3
    assert metrics.run_metrics(events)["loop"] is True


@pytest.mark.parametrize(
    ("preview", "expected"),
    [
        ("[exit 0]\nok", False),
        ("[exit 3]\nboom", True),
        ("[Errored]", True),
        ("[Succeeded]", False),
        ("plain", None),
    ],
)
def test_command_failed(preview, expected):
    assert metrics.command_failed({"preview": preview}) is expected


def test_wilson():
    low, high = metrics.wilson(5, 10)
    assert (round(low, 3), round(high, 3)) == (0.237, 0.763)
    assert metrics.wilson(0, 0) == (0.0, 0.0)
