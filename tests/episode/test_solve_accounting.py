"""Solve-rate usage accounting and interruption behavior."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from kullback.runner import loop

pytestmark = pytest.mark.skipif(
    not (hasattr(loop, "ask") and hasattr(loop, "advance")),
    reason="needs the step-split patch",
)

from kullback.ai.provider import (  # noqa: E402
    ModelReply,  # noqa: E402
    TestModel,  # noqa: E402
)
from kullback.ai.usage import Usage  # noqa: E402
from kullback.episode import BuiltEnvironment, solve_rate  # noqa: E402
from tests.episode.invented import write_env  # noqa: E402


def _stub_priced(monkeypatch):
    from kullback.runner import budget

    price = {"input": 1000000.0, "output": 0.0, "cache_read": 0.0, "cache_write": 0.0}
    monkeypatch.setattr(budget, "price_for", lambda model_id: dict(price) if model_id == "test/priced" else None)
    monkeypatch.setattr(budget, "price_source", lambda model_id: "table" if model_id == "test/priced" else None)
    monkeypatch.setattr(budget, "window_for", lambda model_id: 10000)


def _priced_reply():
    return ModelReply(content="done", usage=Usage(input=2), model="priced")


def _cost_files(session):
    return list(Path(session).rglob("*.cost.json"))


def test_no_ceiling_records_cost_sidecar(tmp_path, monkeypatch):
    _stub_priced(monkeypatch)
    root = write_env(tmp_path / "env", with_rules=False)
    model = TestModel([_priced_reply()], name="test/priced")
    table = solve_rate(BuiltEnvironment(root), ["widget_task"], model, outdir=tmp_path / "out")
    assert table["spent_usd"] == 2.0
    assert table["calls"] == 1
    assert table["unpriced_calls"] == 0
    assert table["tasks"][0]["runs"] == 1
    session = table["session_dir"]
    costs = _cost_files(session)
    assert len(costs) == 1
    body = json.loads(costs[0].read_text(encoding="utf-8"))
    assert body["usd"] == 2.0
    assert body["calls"] == 1
    assert body["model"] == "test/priced"
    assert (tmp_path / "out" / "solve_rate.json").is_file()
    assert (Path(session) / "solve_rate.json").is_file()


def test_ceiling_overshoot_preserves_d86(tmp_path, monkeypatch):
    _stub_priced(monkeypatch)
    root = write_env(tmp_path / "env", with_rules=False)
    model = TestModel([_priced_reply()], name="test/priced", loop=True)
    table = solve_rate(
        BuiltEnvironment(root), ["widget_task"], model,
        runs_per_task=3, ceiling_usd=1.0, outdir=tmp_path / "out",
    )
    assert len(model.calls) == 1
    assert table["spent_usd"] == 2.0
    assert table["ceiling_stopped"] is True
    assert table["ceiling_mode"] == "D86_post_call"
    row = table["tasks"][0]
    assert row["runs"] == 0
    assert row["interrupted"] == 1
    assert row["solve_rate"] is None
    session = table["session_dir"]
    costs = _cost_files(session)
    assert len(costs) == 1
    assert json.loads(costs[0].read_text(encoding="utf-8"))["usd"] == 2.0
    run_files = [p for p in Path(session).rglob("*.jsonl") if p.name != "events.jsonl"]
    assert len(run_files) == 1
    footer = json.loads(run_files[0].read_text(encoding="utf-8").strip().splitlines()[-1])
    assert footer.get("termination_reason") == "budget_exceeded"


def test_unknown_model_and_bad_settings_refused(tmp_path, monkeypatch):
    from kullback.runner import budget

    _stub_priced(monkeypatch)
    root = write_env(tmp_path / "env", with_rules=False)
    env = BuiltEnvironment(root)
    unknown = TestModel([_priced_reply()], name="test/unknown")
    with pytest.raises(budget.UnpricedModel):
        solve_rate(env, ["widget_task"], unknown, ceiling_usd=5.0, outdir=tmp_path / "out")
    assert len(unknown.calls) == 0
    for bad in ({"runs_per_task": 0}, {"max_turns": 0}, {"ceiling_usd": 0},
                {"ceiling_usd": -1}, {"ceiling_usd": float("nan")}, {"ceiling_usd": float("inf")}):
        kwargs = {"runs_per_task": 1, "max_turns": 30, "ceiling_usd": None}
        kwargs.update(bad)
        model = TestModel([_priced_reply()], name="test/priced")
        with pytest.raises(ValueError):
            solve_rate(env, ["widget_task"], model, outdir=tmp_path / "out2", **kwargs)
        assert len(model.calls) == 0


def test_unpriced_without_ceiling_marks_incomplete(tmp_path, monkeypatch):
    from kullback.runner import budget

    monkeypatch.setattr(budget, "price_for", lambda model_id: None)
    monkeypatch.setattr(budget, "price_source", lambda model_id: None)
    monkeypatch.setattr(budget, "window_for", lambda model_id: 10000)
    root = write_env(tmp_path / "env", with_rules=False)
    model = TestModel([ModelReply(content="done", usage=Usage(input=2), model="unknown")], name="test/unknown")
    table = solve_rate(BuiltEnvironment(root), ["widget_task"], model, outdir=tmp_path / "out")
    assert table["calls"] == 1
    assert table["unpriced_calls"] == 1
    assert table["spent_usd"] == 0


def test_two_argument_policy_works(tmp_path, monkeypatch):
    _stub_priced(monkeypatch)
    root = write_env(tmp_path / "env", with_rules=False)

    class TwoArg:
        name = "test/priced"

        def __init__(self):
            self.calls = 0

        def query(self, messages, tools=None):
            self.calls += 1
            return _priced_reply()

    policy = TwoArg()
    table = solve_rate(BuiltEnvironment(root), ["widget_task"], policy, outdir=tmp_path / "out")
    assert policy.calls == 1
    assert table["spent_usd"] == 2.0


def test_shared_root_keeps_sessions_separate(tmp_path, monkeypatch):
    _stub_priced(monkeypatch)
    root = write_env(tmp_path / "env", with_rules=False)
    env = BuiltEnvironment(root)
    first = solve_rate(env, ["widget_task"], TestModel([_priced_reply()], name="test/priced"),
                       outdir=tmp_path / "out")
    first_session = Path(first["session_dir"])
    first_table = json.loads((first_session / "solve_rate.json").read_text(encoding="utf-8"))
    first_runs = {p.relative_to(first_session): p.read_bytes() for p in [q for q in first_session.rglob("*.jsonl") if q.name != "events.jsonl"]}
    assert first_runs
    second = solve_rate(env, ["widget_task"], TestModel([_priced_reply()], name="test/priced"),
                        outdir=tmp_path / "out")
    assert Path(second["session_dir"]) != first_session
    assert json.loads((first_session / "solve_rate.json").read_text(encoding="utf-8")) == first_table
    for rel, data in first_runs.items():
        assert (first_session / rel).read_bytes() == data
    root_table = json.loads((tmp_path / "out" / "solve_rate.json").read_text(encoding="utf-8"))
    assert root_table["session_dir"] == second["session_dir"]


def test_policy_error_aborts_without_reward(tmp_path, monkeypatch):
    from kullback.episode import Episode
    from kullback.episode.episode import EpisodeError

    _stub_priced(monkeypatch)
    root = write_env(tmp_path / "env", with_rules=False)
    env = BuiltEnvironment(root)

    class Boom:
        name = "test/priced"

        def query(self, messages, tools=None, config=None):
            raise RuntimeError("boom")

    with pytest.raises(RuntimeError):
        solve_rate(env, ["widget_task"], Boom(), outdir=tmp_path / "out")
    run_files = [p for p in (tmp_path / "out").rglob("*.jsonl") if p.name != "events.jsonl"]
    assert len(run_files) == 1
    footer = json.loads(run_files[0].read_text(encoding="utf-8").strip().splitlines()[-1])
    assert footer.get("termination_reason") == "policy_error"
    episode = Episode(env, outdir=tmp_path / "out2")
    episode.reset("widget_task", seed=7)
    episode.abort("policy_error")
    assert episode.reward().score is None
    episode.abort("policy_error")
    with pytest.raises(EpisodeError):
        Episode(env, outdir=tmp_path / "out3").abort("policy_error")
    with pytest.raises(EpisodeError):
        episode.abort("bogus")


def test_env_error_counts_as_no_signal(tmp_path, monkeypatch):
    from kullback.episode import Episode
    from kullback.episode.episode import EpisodeError
    from kullback.runner import loop

    _stub_priced(monkeypatch)
    root = write_env(tmp_path / "env", with_rules=False)

    def broken(self, message):
        loop._crashed(self._state, RuntimeError("invented world failure"), self._router)
        raise EpisodeError("broken")

    monkeypatch.setattr(Episode, "step", broken)
    table = solve_rate(
        BuiltEnvironment(root), ["widget_task"],
        TestModel([_priced_reply()], name="test/priced", loop=True),
        outdir=tmp_path / "out",
    )
    assert table["tasks"][0]["runs"] == 1
    assert table["tasks"][0]["no_signal"] == 1
    assert table["tasks"][0]["solve_rate"] is None
    assert table["spent_usd"] == 2.0


def test_memo_hits_remain_visible_through_policy_adapter(tmp_path, monkeypatch):
    from kullback.ai.provider import MemoModel
    from kullback.runner import budget

    _stub_priced(monkeypatch)
    root = write_env(tmp_path / "env", with_rules=False)
    inner = TestModel([_priced_reply()], name="test/priced")
    model = MemoModel(inner, tmp_path / "memo")
    table = solve_rate(BuiltEnvironment(root), ["widget_task"], model,
                       runs_per_task=2, outdir=tmp_path / "out")
    assert len(inner.calls) == 1
    assert table["calls"] == 2
    assert table["memo_hits"] == 1
    assert table["spent_usd"] == 2.0
    assert budget.load_totals(table["session_dir"])["total"]["memo_hits"] == 1
    costs = [json.loads(path.read_text()) for path in _cost_files(table["session_dir"])]
    assert sorted(row["memo_hits"] for row in costs) == [0, 1]
    assert sorted(row["usd"] for row in costs) == [0.0, 2.0]


def test_ceiling_accounts_across_attempts_without_resetting(tmp_path, monkeypatch):
    _stub_priced(monkeypatch)
    root = write_env(tmp_path / "env", with_rules=False)
    model = TestModel([_priced_reply()], name="test/priced", loop=True)
    table = solve_rate(BuiltEnvironment(root), ["widget_task"], model,
                       runs_per_task=3, ceiling_usd=3.0, outdir=tmp_path / "out")
    assert len(model.calls) == 2
    assert table["spent_usd"] == 4.0
    assert table["ceiling_stopped"] is True
    assert table["tasks"][0]["runs"] == 1
    assert table["tasks"][0]["interrupted"] == 1
    assert table["tasks"][0]["no_signal"] == 0
    assert table["tasks"][0]["solve_rate"] == 0.0
    assert len(_cost_files(table["session_dir"])) == 2


def _delete_run_files(outdir):
    for path in Path(outdir).rglob("*.jsonl"):
        if path.name != "events.jsonl":
            path.unlink()


def test_missing_record_file_still_reports_the_original_error(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from kullback.ai.usage import Usage
    from kullback.episode.episode import EpisodeError

    _stub_priced(monkeypatch)
    root = write_env(tmp_path / "env", with_rules=False)
    outdir = tmp_path / "out"

    class DeleteAndFail:
        name = "test/priced"

        def __init__(self):
            self.calls = 0

        def query(self, messages, tools=None, config=None):
            self.calls += 1
            if self.calls == 1:
                return SimpleNamespace(
                    content=None,
                    tool_calls=[{"id": "c1", "name": "describe_widget",
                                 "arguments": {"widget_id": "w1"}}],
                    model="test/priced", usage=Usage(input=2))
            _delete_run_files(outdir)
            raise EpisodeError("policy fell over")

    with pytest.raises(EpisodeError, match="policy fell over"):
        solve_rate(BuiltEnvironment(root), ["widget_task"], DeleteAndFail(), outdir=outdir)
    run_files = [p for p in outdir.rglob("*.jsonl") if p.name != "events.jsonl"]
    assert len(run_files) == 1
    footer = json.loads(run_files[0].read_text(encoding="utf-8").strip().splitlines()[-1])
    assert footer.get("termination_reason") == "policy_error"


def test_cost_sidecar_uses_null_when_the_record_cannot_be_read(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from kullback.episode import Episode
    from kullback.episode.solve import _attempt_cost
    from kullback.runner import budget

    _stub_priced(monkeypatch)
    root = write_env(tmp_path / "env", with_rules=False)
    episode = Episode(BuiltEnvironment(root), outdir=tmp_path / "out")
    episode.reset("widget_task", seed=7)
    episode.step({"content": None, "tool_calls": [
        {"id": "c1", "name": "describe_widget", "arguments": {"widget_id": "w1"}}]})
    _delete_run_files(tmp_path / "out")
    model = SimpleNamespace(workdir=tmp_path / "budgets", model_id="test/priced")
    before = budget.load_totals(model.workdir)["total"]
    _attempt_cost(episode, model, before)
    sidecars = list((tmp_path / "out").rglob("*.cost.json"))
    assert len(sidecars) == 1
    assert json.loads(sidecars[0].read_text(encoding="utf-8"))["termination_reason"] is None
