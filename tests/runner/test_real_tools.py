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


def test_world_error_is_returned_not_raised():
    world = FakeWorld()
    world.step_error = UnresolvedError("creation is unresolved")
    tool = RealTool("shell", lambda: world)
    out = tool.call(shell_batch("ls"))
    assert out.result is None
    assert out.error_class == "unknown"
    assert "UnresolvedError" in out.error_message


def test_timed_out_receipt_is_a_transient_error():
    world = FakeWorld([FakeReceipt(stdout=b"part", timed_out=True)])
    tool = RealTool("shell", lambda: world)
    out = tool.call(shell_batch("sleep 99"))
    assert out.error_class == "transient"


def test_timeout_exception_is_a_transient_error():
    world = FakeWorld()
    world.step_error = TimeoutError("slow")
    tool = RealTool("shell", lambda: world)
    out = tool.call(shell_batch("ls"))
    assert out.error_class == "transient"


def test_bad_batch_is_a_business_error_and_opens_nothing():
    world = FakeWorld()
    tool = RealTool("shell", lambda: world)
    out = tool.call({"commands": "ls"})
    assert out.error_class == "business_error"
    assert world.reset_count == 0


def test_empty_batch_runs_nothing_and_answers_empty_text():
    world = FakeWorld()
    tool = RealTool("shell", lambda: world)
    assert tool.call({"commands": []}).result == ""
    assert world.commands == []
    assert world.reset_count == 1


def test_run_continues_after_a_world_error():
    world = FakeWorld()
    world.step_error = UnresolvedError("gone")
    tool = RealTool("shell", lambda: world)
    assert tool.call(shell_batch("ls")).error_class == "unknown"
    world.step_error = None
    assert tool.call(shell_batch("ls")).result == "ok"


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


def test_end_state_returns_the_export_bytes():
    world = FakeWorld(export=b"tar-bytes-here")
    tool = RealTool("shell", lambda: world)
    assert tool.end_state(64) == b""
    tool.call(shell_batch("ls"))
    assert tool.end_state(64) == b"tar-bytes-here"


def real_router(world) -> Router:
    tool = RealTool("shell", lambda: world)
    return Router(starting_state=StateView(shared={}), real_tools={"shell": tool})


def test_reset_failure_closes_the_created_world():
    world = FakeWorld()
    world.reset_error = UnresolvedError("creation is unresolved")
    tool = RealTool("shell", lambda: world)
    out = tool.call(shell_batch("ls"))
    assert out.error_class == "unknown"
    assert world.closed is True
    world.reset_error = None
    assert tool.call(shell_batch("ls")).result == "ok"


def test_reset_failure_with_a_failing_close_still_answers_the_reset_error():
    world = FakeWorld()
    world.reset_error = UnresolvedError("creation is unresolved")
    world.close_error = UnresolvedError("remove failed")
    tool = RealTool("shell", lambda: world)
    assert tool.call(shell_batch("ls")).error_class == "unknown"


def test_reset_failure_with_an_honest_close_error_raises_it():
    world = FakeWorld()
    world.reset_error = UnresolvedError("creation is unresolved")
    world.close_error = RuntimeError("cannot remove")
    tool = RealTool("shell", lambda: world)
    with pytest.raises(RuntimeError):
        tool.call(shell_batch("ls"))


def test_loop_releases_the_world_when_the_run_raises(workdir):
    world = FakeWorld([FakeReceipt(stdout=b"hi")])
    state = new_run_state("r1", workdir=workdir)
    model = TestModel([
        {"content": None,
         "tool_calls": [{"id": "c1", "name": "shell", "arguments": shell_batch("ls")}]},
    ])
    with pytest.raises(IndexError):
        run(state, model, tools=[{"name": "shell"}], router=real_router(world))
    assert world.commands == ["ls"]
    assert world.closed is True


def test_loop_releases_the_world_when_the_run_ends(workdir):
    world = FakeWorld([FakeReceipt(stdout=b"hi")])
    state = new_run_state("r1", workdir=workdir)
    model = TestModel([
        {"content": None,
         "tool_calls": [{"id": "c1", "name": "shell", "arguments": shell_batch("ls")}]},
        {"content": "done"},
    ])
    run(state, model, tools=[{"name": "shell"}], router=real_router(world))
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


def test_run_returns_its_state_when_a_close_fails(workdir):
    world = FakeWorld([FakeReceipt(stdout=b"hi")])
    world.close_error = RuntimeError("cannot remove")
    state = new_run_state("r1", workdir=workdir)
    out = run(state, done_after_call(shell_batch("ls")),
              tools=[{"name": "shell"}], router=real_router(world))
    assert out.stopped is True
    assert out.run.termination_reason == "agent_stop"
    assert out.run.route_counts.get("real") == 1


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


def test_close_failure_is_recorded_and_leaves_the_runs_result(workdir):
    world = FakeWorld([FakeReceipt(stdout=b"hi")])
    world.close_error = RuntimeError("cannot remove")
    router = real_router(world)
    state = new_run_state("r1", workdir=workdir)
    out = run(state, done_after_call(shell_batch("ls")),
              tools=[{"name": "shell"}], router=router)
    assert out.run.termination_reason == "agent_stop"
    answered = [e for e in out.run.events if e.type == "tool_result"]
    assert answered[0].payload["result"] == "hi"
    failures = [e for e in out.run.events if e.type == "error"]
    assert [(e.payload["tool"], e.payload["message"]) for e in failures] == [
        ("shell", "RuntimeError: cannot remove")]
    assert failures[0].payload["class"] == "real_close_failure"
    from kullback.runner.verdict import _env_error
    assert _env_error(out.run) is False


def test_run_returns_its_state_when_the_export_fails(workdir):
    world = FakeWorld([FakeReceipt(stdout=b"hi")])
    world.export_error = UnresolvedError("export blew up")
    router = real_router(world)
    state = new_run_state("r1", workdir=workdir)
    out = run(state, done_after_call(shell_batch("ls")),
              tools=[{"name": "shell"}], router=router)
    assert out.stopped is True
    assert out.run.termination_reason == "agent_stop"
    answered = [e for e in out.run.events if e.type == "tool_result"]
    assert answered[0].payload["result"] == "hi"
