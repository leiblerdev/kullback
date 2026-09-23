"""A Real tool runs the shell batch for real in a container world, first of the D262 order."""

from __future__ import annotations

import pytest

from kullback.ai.provider import TestModel
from kullback.runner.loop import new_run_state, run
from kullback.runner.real_tools import ExportLimitError, RealTool, real_tools_from
from kullback.runner.route import Router
from kullback.runner.state import StateView


class FakeReceipt:
    """One scripted step answer in the shape a container world step returns."""

    def __init__(self, stdout=b"", stderr=b"", exit_code=0, timed_out=False, truncated=False):
        self.stdout = stdout
        self.stderr = stderr
        self.exit_code = exit_code
        self.timed_out = timed_out
        self.truncated = truncated


class UnresolvedError(Exception):
    """A world failure named like the container World's own family, without importing it."""


class FakeWorld:
    """A scripted stand-in for ContainerWorld: records commands, never touches docker."""

    def __init__(self, receipts=None, export=b"fake-tar-bytes"):
        self.script = list(receipts or [])
        self.commands: list = []
        self.reset_count = 0
        self.closed = False
        self.exported = export
        self.step_error = None
        self.reset_error = None
        self.close_error = None
        self.export_error = None
        self.export_limit = None

    def reset(self):
        self.reset_count += 1
        if self.reset_error is not None:
            raise self.reset_error
        return "fake-container-id"

    def step(self, command):
        self.commands.append(command)
        if self.step_error is not None:
            raise self.step_error
        if self.script:
            return self.script.pop(0)
        return FakeReceipt(stdout=b"ok")

    def export_workspace(self, limit_bytes):
        assert limit_bytes > 0
        self.export_limit = limit_bytes
        if self.export_error is not None:
            raise self.export_error
        if len(self.exported) > limit_bytes:
            raise ExportLimitError("workspace export exceeded its byte limit")
        return self.exported

    def close(self):
        if self.close_error is not None:
            raise self.close_error
        self.closed = True
        return True


def shell_batch(*keystrokes) -> dict:
    return {"commands": [{"keystrokes": text} for text in keystrokes]}


def test_shell_batch_runs_one_command_per_entry_and_resets_once():
    world = FakeWorld([FakeReceipt(stdout=b"a"), FakeReceipt(stdout=b"b")])
    tool = RealTool("shell", lambda: world)
    first = tool.call({"commands": [{"keystrokes": "ls"}, {"keystrokes": "pwd", "duration": 1.5}]})
    assert first.result == "ab"
    second = tool.call(shell_batch("whoami"))
    assert second.result == "ok"
    assert world.commands == ["ls", "pwd", "whoami"]
    assert world.reset_count == 1


def test_failing_entry_runs_the_rest_of_the_batch_and_answers_its_text():
    world = FakeWorld([FakeReceipt(stdout=b"one"),
                       FakeReceipt(stdout=b"half", stderr=b"nope", exit_code=1),
                       FakeReceipt(stdout=b"three")])
    tool = RealTool("shell", lambda: world)
    out = tool.call(shell_batch("one", "two", "three"))
    assert world.commands == ["one", "two", "three"]
    assert out.error_class is None
    assert out.result == "onehalfnopethree"
    assert isinstance(out.result, str)
    assert [receipt.exit_code for receipt in out.receipts] == [0, 1, 0]


def test_world_error_is_returned_not_raised_and_the_next_call_runs():
    world = FakeWorld()
    world.step_error = UnresolvedError("creation is unresolved")
    tool = RealTool("shell", lambda: world)
    out = tool.call(shell_batch("ls"))
    assert out.result is None
    assert out.error_class == "unknown"
    assert "UnresolvedError" in out.error_message
    world.step_error = None
    assert tool.call(shell_batch("ls")).result == "ok"


@pytest.mark.parametrize("receipts,step_error", [
    ([FakeReceipt(stdout=b"part", timed_out=True)], None),
    (None, TimeoutError("slow")),
])
def test_a_timeout_is_a_transient_error(receipts, step_error):
    world = FakeWorld(receipts)
    world.step_error = step_error
    tool = RealTool("shell", lambda: world)
    out = tool.call(shell_batch("sleep 99"))
    assert out.error_class == "transient"


def test_a_bad_batch_is_refused_before_the_world_and_an_empty_one_runs_nothing():
    world = FakeWorld()
    tool = RealTool("shell", lambda: world)
    out = tool.call({"commands": "ls"})
    assert out.error_class == "business_error"
    assert world.reset_count == 0
    assert tool.call({"commands": []}).result == ""
    assert world.commands == []
    assert world.reset_count == 1


def test_real_tools_from_builds_one_tool_per_declaration():
    seen = {}

    def make_world_for(declaration):
        def make():
            return FakeWorld()
        seen["declaration"] = declaration
        return make

    tools = real_tools_from({"shell": {"image": "img", "limits": {}}}, make_world_for)
    assert set(tools) == {"shell"}
    assert tools["shell"].name == "shell"
    assert seen["declaration"] == {"image": "img", "limits": {}}


def test_close_on_a_never_called_tool_opens_nothing():
    made = []
    tool = RealTool("shell", lambda: made.append(FakeWorld()) or made[-1])
    assert tool.close() is False
    assert made == []


def real_router(world) -> Router:
    tool = RealTool("shell", lambda: world)
    return Router(starting_state=StateView(shared={}), real_tools={"shell": tool})


def test_end_state_returns_the_export_bytes():
    world = FakeWorld(export=b"tar-bytes-here")
    tool = RealTool("shell", lambda: world)
    assert tool.end_state(64) == b""
    tool.call(shell_batch("ls"))
    assert tool.end_state(64) == b"tar-bytes-here"
    routed = FakeWorld([FakeReceipt(stdout=b"a")], export=b"tar-here")
    router = real_router(routed)
    router.route("shell", shell_batch("ls"))
    assert router.real_end_state("shell", 64) == b"tar-here"


def test_reset_failure_closes_the_created_world_and_only_an_honest_close_error_raises():
    world = FakeWorld()
    world.reset_error = UnresolvedError("creation is unresolved")
    tool = RealTool("shell", lambda: world)
    out = tool.call(shell_batch("ls"))
    assert out.error_class == "unknown"
    assert world.closed is True
    world.reset_error = None
    assert tool.call(shell_batch("ls")).result == "ok"

    failing = FakeWorld()
    failing.reset_error = UnresolvedError("creation is unresolved")
    failing.close_error = UnresolvedError("remove failed")
    assert RealTool("shell", lambda: failing).call(shell_batch("ls")).error_class == "unknown"

    honest = FakeWorld()
    honest.reset_error = UnresolvedError("creation is unresolved")
    honest.close_error = RuntimeError("cannot remove")
    with pytest.raises(RuntimeError):
        RealTool("shell", lambda: honest).call(shell_batch("ls"))


@pytest.mark.parametrize("ends", [True, False])
def test_the_loop_always_releases_the_world(workdir, ends):
    world = FakeWorld([FakeReceipt(stdout=b"hi")])
    state = new_run_state("r1", workdir=workdir)
    replies = [{"content": None,
                "tool_calls": [{"id": "c1", "name": "shell", "arguments": shell_batch("ls")}]}]
    if not ends:
        with pytest.raises(IndexError):
            run(state, TestModel(replies), tools=[{"name": "shell"}], router=real_router(world))
        assert world.commands == ["ls"]
        assert world.closed is True
        return
    run(state, TestModel([*replies, {"content": "done"}]), tools=[{"name": "shell"}], router=real_router(world))
    assert world.closed is True
    assert state.run.route_counts.get("real") == 1
    answered = [e for e in state.run.events if e.type == "tool_result"]
    assert [e.route for e in answered] == ["real"]


def done_after_call(batch) -> TestModel:
    return TestModel([
        {"content": None,
         "tool_calls": [{"id": "c1", "name": "shell", "arguments": batch}]},
        {"content": "done"},
    ])


@pytest.mark.parametrize("failing", ["close", "export"])
def test_run_returns_its_state_when_a_close_or_an_export_fails(workdir, failing):
    world = FakeWorld([FakeReceipt(stdout=b"hi")])
    if failing == "close":
        world.close_error = RuntimeError("cannot remove")
    else:
        world.export_error = UnresolvedError("export blew up")
    state = new_run_state("r1", workdir=workdir)
    out = run(state, done_after_call(shell_batch("ls")),
              tools=[{"name": "shell"}], router=real_router(world))
    assert out.stopped is True
    assert out.run.termination_reason == "agent_stop"
    assert out.run.route_counts.get("real") == 1
    answered = [e for e in out.run.events if e.type == "tool_result"]
    assert answered[0].payload["result"] == "hi"
    if failing == "close":
        failures = [e for e in out.run.events if e.type == "error"]
        assert [(e.payload["tool"], e.payload["message"]) for e in failures] == [
            ("shell", "RuntimeError: cannot remove")]
        assert failures[0].payload["class"] == "real_close_failure"
        from kullback.runner.verdict import _env_error
        assert _env_error(out.run) is False
        return
    # routed directly, the failed export is listed by close_real and leaves no end state
    routed = FakeWorld([FakeReceipt(stdout=b"out")], export=b"tar-here")
    router = real_router(routed)
    result = router.route("shell", shell_batch("ls"))
    routed.export_error = UnresolvedError("export blew up")
    assert router.close_real() == [("shell", "UnresolvedError: export blew up")]
    assert result.result == "out"
    assert router.real_end_state("shell", 64) == b""


def test_run_keeps_each_called_tools_export(workdir):
    world = FakeWorld([FakeReceipt(stdout=b"hi")], export=b"tar-bytes")
    router = Router(starting_state=StateView(shared={}), real_tools={
        "shell": RealTool("shell", lambda: world),
        "other": RealTool("other", lambda: FakeWorld())})
    state = new_run_state("r1", workdir=workdir)
    run(state, done_after_call(shell_batch("ls")),
        tools=[{"name": "shell"}], router=router)
    assert router.real_end_state("shell", 64) == b"tar-bytes"
    assert router.real_end_state("other", 64) == b""


def test_timed_out_world_marks_the_outcome_as_unanswered():
    world = FakeWorld([FakeReceipt(stdout=b"part", timed_out=True)])
    tool = RealTool("shell", lambda: world)
    out = tool.call(shell_batch("sleep 99"))
    assert out.world_answered is False
    assert out.error_class == "transient"


def test_nonzero_exit_marks_the_outcome_as_answered():
    world = FakeWorld([FakeReceipt(stdout=b"half", stderr=b"nope", exit_code=1)])
    tool = RealTool("shell", lambda: world)
    out = tool.call(shell_batch("two"))
    assert out.world_answered is True
    assert out.error_class is None
    assert out.result == "halfnope"
    routed = real_router(FakeWorld([FakeReceipt(stdout=b"half", stderr=b"nope", exit_code=1)]))
    through = routed.route("shell", shell_batch("two"))
    assert through.route == "real"
    assert through.error is None
    assert through.result == "halfnope"


def test_loop_stops_at_the_first_real_world_failure_with_an_env_error_verdict(workdir):
    """A timed out world stops the Run at once; the Verdict blames the Environment."""
    from kullback.runner.records import Verifier
    from kullback.runner.verdict import verdict
    world = FakeWorld([FakeReceipt(stdout=b"part", timed_out=True)])
    model = TestModel([
        {"tool_calls": [{"id": "c1", "name": "shell", "arguments": shell_batch("sleep 99")},
                        {"id": "c2", "name": "shell", "arguments": shell_batch("ls")}]},
    ])
    state = new_run_state("r1", workdir=workdir)
    run(state, model, tools=[{"name": "shell"}], router=real_router(world))
    assert state.stopped is True
    assert state.run.termination_reason == "environment_cannot_answer"
    assert state.run.route_counts == {"cannot_answer": 1}
    assert [event.type for event in state.run.events].count("tool_call") == 1
    done = verdict(state.run, Verifier(task_id="t", atoms=[]))
    assert done.class_ == "env_error"
