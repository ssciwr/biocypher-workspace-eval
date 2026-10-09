"""Runner budget and reply logic with a fake session; no model or MCP."""

import asyncio
import io

import pytest

from evaluation import config as cfg

runner = pytest.importorskip(
    "evaluation.runner", reason="registry clone not available", exc_type=ImportError
)


class FakeSession:
    """Plays one scripted list of events per submitted message."""

    def __init__(self, turns):
        self.turns = list(turns)
        self.queue = asyncio.Queue()
        self.submitted = []
        self.interrupted = False
        self.busy = False

    def subscribe(self):
        return self.queue

    def unsubscribe(self, queue):
        pass

    def submit(self, content):
        self.submitted.append(content)
        self.busy = True
        for event in self.turns.pop(0):
            self.queue.put_nowait(event)

    def interrupt(self):
        self.interrupted = True
        self.busy = False
        self.queue.put_nowait(
            {"type": "turn_error", "data": {"message": "interrupted"}}
        )
        return True


LEVEL = cfg.Level(
    "L3", "clean", True, max_calls=3, max_minutes=1, final_marker="STATUS:"
)


def turn(text, calls=1, end="turn_done"):
    events = [{"type": "turn_started", "data": {}}]
    for _ in range(calls):
        events += [
            {"type": "text_delta", "data": {"text": text}},
            {"type": "usage", "data": {"input": 1, "output": 1}},
        ]
    return events + [{"type": end, "data": {}}]


def drive(session, level=LEVEL):
    return asyncio.run(
        runner.drive(session, level, "prompt", "Proceed.", 2, io.StringIO())
    )


def test_done_with_marker():
    session = FakeSession([turn("STATUS: DONE")])
    outcome = drive(session)
    assert outcome["termination"] == "done"
    assert outcome["calls"] == 1
    assert session.submitted == ["prompt"]


def test_scripted_reply_when_marker_missing():
    session = FakeSession([turn("Which name should I use?"), turn("STATUS: DONE")])
    outcome = drive(session)
    assert outcome == {**outcome, "termination": "done", "replies": 1}
    assert session.submitted == ["prompt", "Proceed."]


def test_replies_are_capped():
    session = FakeSession([turn("?"), turn("?"), turn("?")])
    outcome = drive(session)
    assert outcome["replies"] == 2
    assert outcome["termination"] == "done"


def test_call_budget_interrupts():
    events = turn("working", calls=5)[:-1]  # no terminal event: the model keeps going
    session = FakeSession([events])
    outcome = drive(session)
    assert session.interrupted
    assert outcome["termination"] == "budget_exhausted"


def test_provider_error_classified():
    session = FakeSession(
        [
            [
                {"type": "turn_started", "data": {}},
                {
                    "type": "turn_error",
                    "data": {
                        "message": "Error from AI model stream. Please try again."
                    },
                },
            ]
        ]
    )
    assert drive(session)["termination"] == "provider_error"
