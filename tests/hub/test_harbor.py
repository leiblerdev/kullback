"""The Harbor form of an Environment (D363): the export, the tools in the image, the reward by code.

Every test runs on the invented widget world from tests/episode, so nothing here names a corpus.
The image's runtime is exercised in place: an exported task's environment/ directory is laid out
exactly as /app is inside the container, so `harbor_world` reads it the same way here.
"""

from __future__ import annotations

import io
import json
import shutil
import tomllib
from pathlib import Path

import pytest

from kullback.hub import harbor, harbor_world, package
from kullback.hub.client import LocalHub
from kullback.hub.package import HARBOR_DIR, verify_package
from kullback.runner.records import ItemResult, read_json, write_json
from kullback.runner.world.environment import BuiltEnvironment
from kullback.runner.world.episode import Episode
from kullback.spec.schema import FactSource, IntentFact, Spec, SpecIntent
from kullback.spec.trust import runs_on_disk
from tests.episode.invented import SCRIPTED_RENAME, write_env

TASK = "widget_task"
# The fixture Spec's ask: the volunteered fact, then the accepted one under the later lead; the echo never (D422).
ASK = (f"{package.FACTS_LEAD}\n- wants widget w1 to carry the label striped\n\n{package.LATER_LEAD}\n"
       "- agreed to a label the agent proposed")
EXPECTED = [{"cells": [{"table": "widgets", "row_id": "w1", "field": "label", "value": "striped"}]}]


def _fact(ident: str, text: str, stance: str = "volunteered", **flags) -> IntentFact:
    return IntentFact(id=ident, text=text, source=FactSource(recording="rec1", turn=0), stance=stance, **flags)


def _workdir(root: Path, *, atoms: list | None = None, trusted: bool = True) -> Path:
    """The invented world as a finished build: a trusted Task, its Spec, its Reference Run, a raw export."""
    write_env(root, verifier_atoms=atoms)
    verifier = json.loads((root / "verifiers" / f"{TASK}.json").read_text())
    write_json(root / "verifiers" / f"{TASK}.json", dict(verifier, expected=EXPECTED))
    write_json(root / "rounds.json", [{"round": 1, "counts": {"trusted_ids": [TASK] if trusted else []}}])
    spec = Spec(task_id=TASK, intent=SpecIntent(task_id=TASK, facts=[
        _fact("f1", "wants widget w1 to carry the label striped"),
        _fact("f2", "agreed to a label the agent proposed", stance="accepted"),
        _fact("f3", "repeats a value only the agent said", echo=True)]))
    write_json(root / "spec" / f"{TASK}.json", spec.model_dump())
    episode = Episode(BuiltEnvironment(root), outdir=root / "scratch", max_turns=10)
    episode.reset(TASK)
    for message in SCRIPTED_RENAME:
        episode.step(message)
    run = root / "runs" / TASK / "reference.jsonl"
    run.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(episode.run_file(), run)
    write_json(root / "references.json", {TASK: {"references": [{"kind": "recording", "run_id": "reference",
                                                                 "trace_id": "rec1"}]}})
    (root / "env" / "calls").mkdir()
    (root / "env" / "calls" / "rename_widget.jsonl").write_text('{"name": "rename_widget"}\n')
    write_json(root / "raw" / "export.json", [{"turn": "a recorded line nobody may ship"}])
    return root


@pytest.fixture
def dataset(tmp_path: Path) -> tuple[Path, dict]:
    workdir = _workdir(tmp_path / "work")
    wheel = tmp_path / "kullback-0.0.0-py3-none-any.whl"
    wheel.write_bytes(b"a wheel")
    out = tmp_path / "harbor"
    return out, harbor.export_harbor(workdir, out, name="widgets", wheel=wheel)


def _app(dataset_dir: Path, tmp_path: Path) -> Path:
    """A fresh container's /app: the exported environment/ directory, copied."""
    app = tmp_path / "app"
    shutil.copytree(dataset_dir / "widgets-widget_task" / "environment", app)
    return app


def _solve(dataset_dir: Path) -> list[tuple[str, dict]]:
    """The oracle's tool calls, read back off solve.sh."""
    import shlex

    calls = []
    for line in (dataset_dir / "widgets-widget_task" / "solution" / "solve.sh").read_text().splitlines():
        if line.startswith("tools call"):
            words = shlex.split(line)
            calls.append((words[2], json.loads(words[4])))
    return calls


def _reward(app: Path, dataset_dir: Path, tmp_path: Path) -> dict:
    return harbor_world.score(app, dataset_dir / "widgets-widget_task" / "tests" / "verifier.json",
                              tmp_path / "logs")


def test_one_harbor_task_directory_per_trusted_task_with_the_five_parts_and_a_manifest(dataset):
    out, manifest = dataset
    folder = out / "widgets-widget_task"
    for part in ("instruction.md", "task.toml", "environment/Dockerfile", "tests/test.sh", "solution/solve.sh"):
        assert (folder / part).is_file(), part
    assert manifest["tasks"][0]["harbor_name"] == "leibler/widgets-widget_task"
    assert manifest["counts"]["exported"] == 1 and manifest["leak_scan"]["leaks"] == 0
    toml = (folder / "task.toml").read_text()
    assert 'schema_version = "1.3"' in toml and 'network_mode = "no-network"' in toml
    assert 'transport = "stdio"' in toml and "needs_judge = false" in toml


def test_an_untrusted_task_is_left_out_by_default_and_exported_with_all(tmp_path):
    workdir = _workdir(tmp_path / "work", trusted=False)
    wheel = tmp_path / "k.whl"
    wheel.write_bytes(b"")
    assert harbor.export_harbor(workdir, tmp_path / "a", name="w", wheel=wheel)["counts"]["exported"] == 0
    assert harbor.export_harbor(workdir, tmp_path / "b", name="w", wheel=wheel, tasks="all")["counts"]["exported"] == 1


def test_the_instruction_says_what_the_user_volunteered_then_what_they_said_later_never_the_echo_then_policy_and_tools(
        dataset):
    out, _ = dataset
    text = (out / "widgets-widget_task" / "instruction.md").read_text()
    first, later = text.split(package.LATER_LEAD)
    assert "wants widget w1 to carry the label striped" in first and "proposed" not in first
    assert "- agreed to a label the agent proposed" in later and "only the agent said" not in text
    assert "Be helpful with widgets." in text and "tools call <name>" in text and "reply_to_user" in text


def test_the_image_holds_no_verifier_reference_recorded_call_or_rules(dataset):
    out, _ = dataset
    environment = out / "widgets-widget_task" / "environment"
    names = {path.relative_to(environment).as_posix() for path in environment.rglob("*") if path.is_file()}
    assert not any(name.startswith(("world/verifiers", "world/runs", "world/user_rules", "world/env/calls",
                                    "world/traces")) for name in names)
    assert "world/references.json" not in names


def test_tools_list_from_a_fresh_container_names_every_task_tool_and_reply_to_user(dataset, tmp_path, capsys):
    out, _ = dataset
    assert harbor_world.tools_main(["list"], app=_app(out, tmp_path)) == 0
    names = [spec["name"] for spec in json.loads(capsys.readouterr().out)]
    assert names == ["describe_widget", "rename_widget", "reply_to_user"]


def test_the_oracle_scores_one(dataset, tmp_path):
    out, _ = dataset
    app = _app(out, tmp_path)
    for name, arguments in _solve(out):
        assert harbor_world.call(app, name, arguments)["error"] is None
    body = _reward(app, out, tmp_path)
    assert body["reward"] == 1.0
    assert (tmp_path / "logs" / "reward.txt").read_text().strip() == "1.0"
    assert json.loads((tmp_path / "logs" / "reward.json").read_text()) == {"reward": 1.0}


def test_doing_nothing_scores_zero(dataset, tmp_path):
    out, _ = dataset
    assert _reward(_app(out, tmp_path), out, tmp_path)["reward"] == 0.0


def _record(workdir: Path, run_id: str, messages: list[dict]) -> None:
    """A Run of the Task played on the invented world and kept under runs/ as `run_id`."""
    episode = Episode(BuiltEnvironment(workdir), outdir=workdir / "scratch" / run_id, max_turns=10)
    episode.reset(TASK)
    for message in messages:
        episode.step(message)
    shutil.copyfile(episode.run_file(), workdir / "runs" / TASK / f"{run_id}.jsonl")


def test_the_oracle_comes_from_a_passing_run_when_the_reference_writes_nothing(tmp_path):
    workdir = _workdir(tmp_path / "work")
    shutil.move(workdir / "runs" / TASK / "reference.jsonl", workdir / "runs" / TASK / "passing.jsonl")
    _record(workdir, "reference", [SCRIPTED_RENAME[0], SCRIPTED_RENAME[-1]])
    wheel = tmp_path / "k.whl"
    wheel.write_bytes(b"")
    out = tmp_path / "out"
    manifest = harbor.export_harbor(workdir, out, name="widgets", wheel=wheel)
    assert manifest["counts"]["exported"] == 1
    assert ("rename_widget", {"widget_id": "w1", "label": "striped"}) in _solve(out)


def test_a_task_none_of_whose_runs_passes_on_replay_is_left_out_and_named_masked(tmp_path):
    workdir = _workdir(tmp_path / "work")
    wrong = {"content": None, "tool_calls": [{"id": "c2", "name": "rename_widget",
                                              "arguments": {"widget_id": "w1", "label": "dotted"}}]}
    _record(workdir, "reference", [wrong, SCRIPTED_RENAME[-1]])
    wheel = tmp_path / "k.whl"
    wheel.write_bytes(b"")
    manifest = harbor.export_harbor(workdir, tmp_path / "out", name="widgets", wheel=wheel)
    assert manifest["counts"]["exported"] == 0
    assert manifest["counts"]["skipped"] == {harbor.NO_ORACLE: 1}
    assert manifest["left_out"] == [{"task": TASK[-4:], "reason": harbor.NO_ORACLE}]
    assert not (tmp_path / "out" / "widgets-widget_task").exists()


def test_a_prose_length_argument_of_a_recorded_run_is_never_replayed_by_the_oracle(tmp_path):
    workdir = _workdir(tmp_path / "work")
    words = "a label long enough to be the recorded agent's own prose rather than a value of the world it reads"
    prose = {"content": None, "tool_calls": [{"id": "c2", "name": "rename_widget",
                                              "arguments": {"widget_id": "w1", "label": words}}]}
    _record(workdir, "reference", [prose, SCRIPTED_RENAME[-1]])
    runs = runs_on_disk(workdir)[TASK]
    calls = harbor.candidate_calls(workdir, TASK, BuiltEnvironment(workdir).write_tools(), runs)
    assert calls == [[{"name": "rename_widget", "arguments": {"widget_id": "w1", "label": harbor.PROSE_LINE}}]]


def test_a_wrong_value_written_through_the_tools_scores_zero(dataset, tmp_path):
    out, _ = dataset
    app = _app(out, tmp_path)
    harbor_world.call(app, "rename_widget", {"widget_id": "w1", "label": "dotted"})
    assert _reward(app, out, tmp_path)["reward"] == 0.0


def test_each_call_is_logged_and_the_world_written_back_so_the_next_process_sees_it(dataset, tmp_path):
    out, _ = dataset
    app = _app(out, tmp_path)
    harbor_world.call(app, "rename_widget", {"widget_id": "w1", "label": "striped"})
    assert json.loads((app / "db.json").read_text())["widgets"]["w1"]["label"] == "striped"
    assert harbor_world.call(app, "describe_widget", {"widget_id": "w1"})["result"]["label"] == "striped"
    logged = [json.loads(line) for line in (app / ".calls.jsonl").read_text().splitlines()]
    assert [row["name"] for row in logged] == ["rename_widget", "describe_widget"]


def test_a_message_to_the_user_is_answered_with_the_standing_consent(dataset, tmp_path):
    out, _ = dataset
    answer = harbor_world.call(_app(out, tmp_path), "reply_to_user", {"text": "Shall I go ahead?"})
    assert answer == {"result": harbor_world.CONSENT_LINE, "error": None}


def test_an_unknown_tool_is_refused_and_not_logged(dataset, tmp_path):
    out, _ = dataset
    app = _app(out, tmp_path)
    assert harbor_world.call(app, "delete_everything", {})["error"]
    assert not (app / ".calls.jsonl").exists()


def test_a_reply_under_an_unknown_parameter_name_is_refused_naming_the_expected_ones_and_leaves_no_turn(
        dataset, tmp_path, capsys):
    # D409: the shell path once logged {"message": ...} and the Run got an empty assistant turn
    out, _ = dataset
    app = _app(out, tmp_path)
    assert harbor_world.tools_main(["call", "reply_to_user", "--json", '{"message": "Done."}'], app=app) == 1
    error = json.loads(capsys.readouterr().out)["error"]
    assert "takes text" in error and "unknown message" in error
    assert harbor_world.read_log(app) == []


def test_a_task_tool_called_without_a_required_parameter_is_refused_the_same_way_over_mcp(dataset, tmp_path):
    out, _ = dataset
    app = _app(out, tmp_path)
    sigs_path = app / harbor_world.WORLD_DIR / "tool_sigs.json"
    sigs = json.loads(sigs_path.read_text())  # a mined signature lists what it requires
    for sig in sigs:
        sig["args_schema"]["required"] = sorted(sig["args_schema"]["properties"])
    sigs_path.write_text(json.dumps(sigs))
    request = {"jsonrpc": "2.0", "id": 1, "method": "tools/call",
               "params": {"name": "rename_widget", "arguments": {"widget_id": "w1"}}}
    written = io.StringIO()
    harbor_world.mcp_main(app, io.StringIO(json.dumps(request) + "\n"), written)
    result = json.loads(written.getvalue())["result"]
    assert result["isError"] is True and "missing label" in result["content"][0]["text"]
    assert "rename_widget takes label, widget_id" in result["content"][0]["text"]
    assert harbor_world.read_log(app) == []
    assert harbor_world.call(app, "rename_widget", {"widget_id": "w1", "label": "striped"})["error"] is None
    assert [entry["name"] for entry in harbor_world.read_log(app)] == ["rename_widget"]


def test_the_mcp_server_lists_and_calls_the_same_tools(dataset, tmp_path):
    out, _ = dataset
    app = _app(out, tmp_path)
    requests = [{"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}},
                {"jsonrpc": "2.0", "method": "notifications/initialized"},
                {"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
                {"jsonrpc": "2.0", "id": 3, "method": "tools/call",
                 "params": {"name": "rename_widget", "arguments": {"widget_id": "w1", "label": "striped"}}}]
    written = io.StringIO()
    harbor_world.mcp_main(app, io.StringIO("\n".join(json.dumps(r) for r in requests) + "\n"), written)
    answers = [json.loads(line) for line in written.getvalue().splitlines()]
    assert [answer["id"] for answer in answers] == [1, 2, 3]
    assert [tool["name"] for tool in answers[1]["result"]["tools"]][-1] == "reply_to_user"
    assert answers[2]["result"]["isError"] is False
    assert json.loads((app / "db.json").read_text())["widgets"]["w1"]["label"] == "striped"


def _item(ident: str, gate: bool, holds, weight: float = 1.0, kind: str = "state") -> ItemResult:
    return ItemResult(id=ident, kind=kind, gate=gate, holds=holds, weight=weight)


def test_an_unscored_gate_gives_zero_and_is_named_unscored_gate():
    body = harbor_world.reward_of([_item("g", True, True), _item("j", True, None, kind="judge")])
    assert body["reward"] == 0.0 and body["unscored_gates"] == ["j"]
    assert {row["id"]: row.get("status") for row in body["items"]}["j"] == "unscored_gate"


def test_an_unscored_weighted_item_leaves_the_denominator_and_is_named_unscored():
    body = harbor_world.reward_of([_item("g", True, True), _item("s", False, False),
                                   _item("j", False, None, kind="judge")])
    assert body["reward"] == pytest.approx(0.5) and body["unscored"] == ["j"]


def test_a_failed_gate_gives_zero_whatever_the_weighted_items_say():
    assert harbor_world.reward_of([_item("g", True, False), _item("s", False, True)])["reward"] == 0.0


def test_the_reward_is_the_verdict_score_when_every_item_is_scored():
    from kullback.runner.verdict import _Item, _score

    items = [_item("g", True, True, 2.0), _item("s", False, True, 1.0), _item("t", False, False, 3.0)]
    expected = _score([_Item(item, item.id) for item in items], True, True, [])
    assert harbor_world.reward_of(items)["reward"] == pytest.approx(expected)


def test_items_json_carries_every_item_with_id_kind_gate_weight_held_and_why(dataset, tmp_path):
    out, _ = dataset
    _reward(_app(out, tmp_path), out, tmp_path)
    rows = json.loads((tmp_path / "logs" / "items.json").read_text())["items"]
    assert rows and all({"id", "kind", "gate", "weight", "held", "why"} <= set(row) for row in rows)


JUDGE_GATE = [{"id": "j1", "kind": "communicate", "gate": True, "judge": True, "description": "says it is done",
               "target": {"question": "Does the agent say it is done?"}}]


def _judged(tmp_path: Path) -> Path:
    """The invented world exported with one gated judge item; answers the dataset directory."""
    workdir = _workdir(tmp_path / "work", atoms=JUDGE_GATE)
    wheel = tmp_path / "k.whl"
    wheel.write_bytes(b"")
    harbor.export_harbor(workdir, tmp_path / "out", name="widgets", wheel=wheel)
    return tmp_path / "out"


def test_the_oracle_reply_says_the_answer_each_judge_item_is_held_to(tmp_path):
    answered = [{**JUDGE_GATE[0], "target": {**JUDGE_GATE[0]["target"], "anchor": {"answer": "Declined, nothing changed."}}}]
    workdir = _workdir(tmp_path / "work", atoms=answered)
    wheel = tmp_path / "k.whl"
    wheel.write_bytes(b"")
    harbor.export_harbor(workdir, tmp_path / "out", name="widgets", wheel=wheel)
    oracle = read_json(tmp_path / "out" / "widgets-widget_task" / "solution" / harbor.ORACLE_NAME)
    assert oracle["reply"] == "Done. Declined, nothing changed."


def _oracle_score(out: Path, tmp_path: Path) -> dict:
    app = _app(out, tmp_path)
    for name, arguments in _solve(out):
        harbor_world.call(app, name, arguments)
    return _reward(app, out, tmp_path)


def _judge_says(monkeypatch, score: int) -> list:
    """The judge model as a scripted stand-in that answers every item with `score`; answers its calls."""
    from kullback.ai.provider import TestModel

    model = TestModel([json.dumps({"results": {"j1": {"score": score, "why": "read the reply"}}})], loop=True)
    monkeypatch.setattr(harbor_world, "live_model", lambda model_id: model)
    monkeypatch.setenv(harbor_world.JUDGE_VAR, "bedrock/anthropic.some-judge")
    return model.calls


def test_a_gated_judge_item_the_named_judge_meets_scores_one_and_judge_json_says_why(tmp_path, monkeypatch):
    out = _judged(tmp_path)
    calls = _judge_says(monkeypatch, 1)
    body = _oracle_score(out, tmp_path)
    assert body["reward"] == 1.0 and len(calls) == 1
    record = json.loads((tmp_path / "logs" / "judge.json").read_text())
    assert record["model"] == "bedrock/anthropic.some-judge"
    assert [(row["id"], row["score"]) for row in record["items"]] == [("j1", 1)]
    assert json.loads((tmp_path / "logs" / "reward.json").read_text()) == {"reward": 1.0}


def test_a_gated_judge_item_the_judge_rejects_scores_zero(tmp_path, monkeypatch):
    out = _judged(tmp_path)
    _judge_says(monkeypatch, 0)
    assert _oracle_score(out, tmp_path)["reward"] == 0.0


JUDGE_WEIGHTED = [{"id": "a1", "kind": "required", "gate": True, "predicate_src": "wrote('rename_widget')",
                   "target": {"kind": "write", "tool": "rename_widget"}},
                  {**JUDGE_GATE[0], "gate": False}]


def _judge_goes_quiet(monkeypatch) -> None:
    """The judge model answers every call with nothing, which the judge's own retry ends as an outage (D400)."""
    from kullback.ai.provider import TestModel

    monkeypatch.setattr(harbor_world, "live_model", lambda model_id: TestModel([""], loop=True))
    monkeypatch.setenv(harbor_world.JUDGE_VAR, "bedrock/anthropic.some-judge")


def _assert_unscored(body: dict, logs: Path, status: str) -> None:
    """No reward, no reward.txt, a reward.json Harbor refuses as a number, and the judge status in items.json."""
    assert body["reward"] is None and body["judge_status"] == {"j1": status}
    assert not (logs / "reward.txt").exists()
    assert json.loads((logs / "reward.json").read_text()) == {"reward": f"unscored: judge j1 {status}"}
    items = json.loads((logs / "items.json").read_text())
    assert {row["id"]: row.get("status") for row in items["items"]}["j1"] == status


def test_a_judge_outage_on_a_gated_item_leaves_the_trial_unscored_not_failed(tmp_path, monkeypatch):
    out = _judged(tmp_path)
    _judge_goes_quiet(monkeypatch)
    _assert_unscored(_oracle_score(out, tmp_path), tmp_path / "logs", "outage")


def test_a_judge_outage_on_a_weighted_item_leaves_the_trial_unscored_not_passed(tmp_path, monkeypatch):
    workdir = _workdir(tmp_path / "work", atoms=JUDGE_WEIGHTED)
    wheel = tmp_path / "k.whl"
    wheel.write_bytes(b"")
    harbor.export_harbor(workdir, tmp_path / "out", name="widgets", wheel=wheel)
    _judge_goes_quiet(monkeypatch)
    body = _oracle_score(tmp_path / "out", tmp_path)
    _assert_unscored(body, tmp_path / "logs", "outage")
    assert body["reward_had_it_been_scored"] == 1.0, "the masked score that used to be reported"


def test_a_weighted_judge_item_the_judge_answers_scores_as_before(tmp_path, monkeypatch):
    workdir = _workdir(tmp_path / "work", atoms=JUDGE_WEIGHTED)
    wheel = tmp_path / "k.whl"
    wheel.write_bytes(b"")
    harbor.export_harbor(workdir, tmp_path / "out", name="widgets", wheel=wheel)
    _judge_says(monkeypatch, 0)
    body = _oracle_score(tmp_path / "out", tmp_path)
    assert 0.0 < body["reward"] < 1.0 and "unscored_reason" not in body, "past the gate, the rejected item costs its weight"
    assert (tmp_path / "logs" / "reward.txt").read_text() == f"{body['reward']}\n"
    assert json.loads((tmp_path / "logs" / "reward.json").read_text()) == {"reward": body["reward"]}


def test_a_judge_item_on_a_run_with_no_assistant_turn_is_answered_zero_and_the_trial_gets_a_number(
        tmp_path, monkeypatch):
    # D408: the no-evidence shortcut is an answer; it once left status empty and harbor_world read it as not_started
    workdir = _workdir(tmp_path / "work", atoms=JUDGE_WEIGHTED)
    wheel = tmp_path / "k.whl"
    wheel.write_bytes(b"")
    harbor.export_harbor(workdir, tmp_path / "out", name="widgets", wheel=wheel)
    calls = _judge_says(monkeypatch, 1)
    app = _app(tmp_path / "out", tmp_path)
    for name, arguments in _solve(tmp_path / "out"):
        if name != harbor_world.REPLY_TOOL:
            harbor_world.call(app, name, arguments)
    body = _reward(app, tmp_path / "out", tmp_path)
    assert 0.0 < body["reward"] < 1.0 and "unscored_reason" not in body and calls == []
    record = json.loads((tmp_path / "logs" / "judge.json").read_text())
    assert [(row["id"], row["status"], row["score"], row["why"]) for row in record["items"]] == \
        [("j1", "answered", 0, "no evidence")]


def test_with_no_judge_named_a_judge_item_leaves_the_trial_unscored_and_judge_json_says_so(tmp_path, monkeypatch):
    out = _judged(tmp_path)
    monkeypatch.delenv(harbor_world.JUDGE_VAR, raising=False)
    _assert_unscored(_oracle_score(out, tmp_path), tmp_path / "logs", harbor_world.NOT_CONFIGURED)
    assert harbor_world.JUDGE_VAR in json.loads((tmp_path / "logs" / "judge.json").read_text())["error"]


def test_a_judge_that_cannot_start_leaves_the_trial_unscored(tmp_path, monkeypatch):
    out = _judged(tmp_path)

    def refused(model_id):
        raise RuntimeError("live model requests are off")
    monkeypatch.setattr(harbor_world, "live_model", refused)
    monkeypatch.setenv(harbor_world.JUDGE_VAR, "bedrock/anthropic.some-judge")
    _assert_unscored(_oracle_score(out, tmp_path), tmp_path / "logs", harbor_world.NOT_STARTED)
    assert "live model requests are off" in json.loads((tmp_path / "logs" / "judge.json").read_text())["error"]


def test_an_unscored_trial_exits_with_its_own_code_and_the_test_step_keeps_its_reward_json(tmp_path, monkeypatch):
    out = _judged(tmp_path)
    _judge_goes_quiet(monkeypatch)
    app = _app(out, tmp_path)
    for name, arguments in _solve(out):
        harbor_world.call(app, name, arguments)
    verifier = out / "widgets-widget_task" / "tests" / "verifier.json"
    code = harbor_world.verify_main([str(verifier), "--logs", str(tmp_path / "logs")], app=app)
    assert code == harbor_world.UNSCORED_EXIT != 0
    test_sh = (out / "widgets-widget_task" / "tests" / "test.sh").read_text()
    assert f'-ne {harbor_world.UNSCORED_EXIT} ]' in test_sh and test_sh.rstrip().endswith("exit $status")


def test_only_a_task_with_judge_items_passes_the_judge_through_and_opens_the_verifier_network(tmp_path, dataset):
    judged = tomllib.loads((_judged(tmp_path / "judged") / "widgets-widget_task" / "task.toml").read_text())
    plain = tomllib.loads((dataset[0] / "widgets-widget_task" / "task.toml").read_text())
    env = judged["verifier"]["env"]
    assert harbor_world.JUDGE_VAR in env and env == {name: f"${{{name}:-}}" for name in env}
    assert judged["verifier"]["network_mode"] == "public" and judged["environment"]["network_mode"] == "no-network"
    assert "env" not in plain["verifier"] and "network_mode" not in plain["verifier"]


def test_the_export_never_calls_a_judge_while_it_checks_its_oracle(tmp_path, monkeypatch):
    calls = _judge_says(monkeypatch, 1)
    _judged(tmp_path)
    assert calls == []


def test_a_task_with_a_judge_gate_is_marked_needs_judge(tmp_path):
    workdir = _workdir(tmp_path / "work", atoms=JUDGE_GATE)
    wheel = tmp_path / "k.whl"
    wheel.write_bytes(b"")
    manifest = harbor.export_harbor(workdir, tmp_path / "out", name="widgets", wheel=wheel)
    assert manifest["counts"]["needs_judge"] == 1 and manifest["tasks"][0]["needs_judge"] is True
    assert "needs_judge = true" in (tmp_path / "out" / "widgets-widget_task" / "task.toml").read_text()


def test_a_recorded_turn_in_what_a_task_adds_refuses_the_export(tmp_path):
    workdir = _workdir(tmp_path / "work")
    turn = "wants widget w1 to carry the label striped because the old plain label kept confusing the warehouse"
    spec = Spec(task_id=TASK, intent=SpecIntent(task_id=TASK, facts=[_fact("f1", turn)]))
    write_json(workdir / "spec" / f"{TASK}.json", spec.model_dump())
    write_json(workdir / "raw" / "export.json", [{"turn": turn}])
    wheel = tmp_path / "k.whl"
    wheel.write_bytes(b"")
    with pytest.raises(harbor.ExportError):
        harbor.export_harbor(workdir, tmp_path / "out", name="widgets", wheel=wheel)
    assert not (tmp_path / "out").exists()


def test_a_dry_run_publish_lays_harbor_beside_the_package_and_the_package_still_verifies(tmp_path, monkeypatch):
    from kullback.hub import publish

    workdir = _workdir(tmp_path / "work")
    wheel = tmp_path / "k.whl"
    wheel.write_bytes(b"")
    monkeypatch.setattr(harbor, "build_wheel", lambda out: wheel)
    write_json(workdir / "replays.json", {TASK: {"rec1": {"confirmed": True, "trace_id": "rec1"}}})
    host = LocalHub(tmp_path / "local")
    hosted, manifest = publish.publish(workdir, "someorg/widgets", client=host, preview=True, form="harbor")
    landed = tmp_path / "local" / "someorg" / "widgets"
    assert (landed / HARBOR_DIR / "widgets-widget_task" / "task.toml").is_file()
    assert 'name = "someorg/widgets-widget_task"' in (landed / HARBOR_DIR / "widgets-widget_task" / "task.toml").read_text()
    assert "`harbor/` holds 1 Tasks in the [Harbor]" in (landed / "README.md").read_text()
    assert manifest["harbor"]["counts"]["exported"] == 1
    assert read_json(landed / "manifest.json")["harbor"] == manifest["harbor"]
    readme = (landed / "README.md").read_text()
    assert "KULLBACK_JUDGE_MODEL" in readme and "HARNESS_ALLOW_MODEL_REQUESTS=1" in readme
    assert verify_package(landed) == []


# --- the verified Task set, tiers and labels (D399) ----------------------------------------------


def _wheel(tmp_path: Path) -> Path:
    wheel = tmp_path / "k.whl"
    wheel.write_bytes(b"")
    return wheel


def _toml(out: Path) -> dict:
    return tomllib.loads((out / "widgets-widget_task" / "task.toml").read_text())["metadata"]


def test_a_verifier_cell_the_scorer_cannot_read_on_its_start_row_sets_the_task_aside_and_is_counted(tmp_path):
    workdir = _workdir(tmp_path / "work")
    write_json(workdir / "db.json", {"widgets": {"w1": {"widget_id": "w1", "label": [{"tone": "plain"}]}}})
    verifier = read_json(workdir / "verifiers" / f"{TASK}.json", {})
    cells = [{"table": "widgets", "row_id": "w1", "field": "label.tone", "value": "striped"}]
    write_json(workdir / "verifiers" / f"{TASK}.json", dict(verifier, expected=[{"cells": cells}]))
    manifest = harbor.export_harbor(workdir, tmp_path / "out", name="widgets", wheel=_wheel(tmp_path), tasks="verified")
    assert manifest["counts"]["exported"] == 0
    assert manifest["counts"]["skipped"] == {harbor.UNREADABLE_CELL: 1}
    assert not (tmp_path / "out" / "widgets-widget_task").exists()


def test_verified_exports_an_untrusted_task_that_has_a_verifier(tmp_path):
    workdir = _workdir(tmp_path / "work", trusted=False)
    manifest = harbor.export_harbor(workdir, tmp_path / "out", name="widgets", wheel=_wheel(tmp_path), tasks="verified")
    assert manifest["tasks_chosen"] == "verified" and manifest["counts"]["exported"] == 1
    assert manifest["tasks"][0]["tier"] == "untrusted"


def test_verified_leaves_out_a_task_with_no_verifier_and_counts_the_reason(tmp_path):
    workdir = _workdir(tmp_path / "work", trusted=False)
    (workdir / "verifiers" / f"{TASK}.json").unlink()
    manifest = harbor.export_harbor(workdir, tmp_path / "out", name="widgets", wheel=_wheel(tmp_path), tasks="verified")
    assert manifest["counts"]["exported"] == 0 and manifest["counts"]["skipped"] == {"no Verifier": 1}


def test_verified_leaves_out_a_task_of_the_synthetic_tier(tmp_path):
    workdir = _workdir(tmp_path / "work", trusted=False)
    write_json(workdir / "tiers.json", {"rows": [{"task_id": TASK, "tier": "synthetic", "reason": None}]})
    manifest = harbor.export_harbor(workdir, tmp_path / "out", name="widgets", wheel=_wheel(tmp_path), tasks="verified")
    assert manifest["counts"]["exported"] == 0 and manifest["counts"]["skipped"] == {"synthetic": 1}


def test_each_task_carries_its_spec_tier_and_reason_off_the_workdir_tier_report(tmp_path):
    workdir = _workdir(tmp_path / "work", trusted=False)
    write_json(workdir / "tiers.json", {"tasks": 1, "rows": [{"task_id": TASK, "tier": "untrusted", "reason": "no_base"}]})
    out = tmp_path / "out"
    manifest = harbor.export_harbor(workdir, out, name="widgets", wheel=_wheel(tmp_path), tasks="verified")
    meta = _toml(out)
    assert meta["spec_tier"] == "untrusted" and meta["spec_tier_reason"] == "no_base"
    assert "label" not in meta and "labels" not in manifest
    assert manifest["counts"]["by_tier_and_label"] == {"no_base": {"": 1}}


def test_a_task_no_tier_row_names_is_marked_not_recorded(tmp_path):
    out = tmp_path / "out"
    harbor.export_harbor(_workdir(tmp_path / "work"), out, name="widgets", wheel=_wheel(tmp_path))
    assert _toml(out)["spec_tier"] == "not_recorded"


def test_a_tier_file_named_on_the_call_replaces_the_workdir_report_and_may_map_masked_ids(tmp_path):
    workdir = _workdir(tmp_path / "work")
    write_json(workdir / "tiers.json", {"rows": [{"task_id": TASK, "tier": "untrusted", "reason": "no_base"}]})
    named = tmp_path / "tiers-shard.json"
    write_json(named, {"tasks": {TASK[-4:]: {"tier": "trusted", "reason": None}}})
    out = tmp_path / "out"
    manifest = harbor.export_harbor(workdir, out, name="widgets", wheel=_wheel(tmp_path), tiers=[named])
    assert _toml(out)["spec_tier"] == "trusted" and "spec_tier_reason" not in _toml(out)
    assert manifest["tiers"] == {"files": ["tiers-shard.json"], "rows": 1, "unmatched": 0}


def test_a_label_row_keyed_by_the_masked_id_marks_the_task_and_a_later_file_wins(tmp_path):
    first, second = tmp_path / "labels-1.json", tmp_path / "labels-2.json"
    write_json(first, [{"task": TASK[-4:], "label": "wrong", "matches_gold": "no", "gold_diff": "first"}])
    write_json(second, [{"task": TASK[-4:], "label": "partial", "matches_gold": "yes", "gold_diff": "second"},
                        {"task": "zzzz", "label": "right"}])
    out = tmp_path / "out"
    manifest = harbor.export_harbor(_workdir(tmp_path / "work"), out, name="widgets", wheel=_wheel(tmp_path),
                                    labels=[first, second], labels_rubric="rubric r1")
    meta = _toml(out)
    assert (meta["label"], meta["label_matches_gold"], meta["label_gold_diff"]) == ("partial", "yes", "second")
    assert manifest["labels"] == {"files": ["labels-1.json", "labels-2.json"], "rows": 1, "unmatched": 1,
                                  "rubric": "rubric r1"}
    assert manifest["tasks"][0]["label"] == "partial"


def test_with_label_files_a_task_no_row_names_is_unlabelled(tmp_path):
    labels = tmp_path / "labels.json"
    write_json(labels, [{"task": "zzzz", "label": "right"}])
    out = tmp_path / "out"
    manifest = harbor.export_harbor(_workdir(tmp_path / "work"), out, name="widgets", wheel=_wheel(tmp_path),
                                    labels=[labels])
    assert _toml(out)["label"] == "unlabelled"
    assert manifest["counts"]["by_tier_and_label"] == {"not_recorded": {"unlabelled": 1}}


def test_a_key_two_task_ids_end_with_names_no_task_and_is_counted_unmatched():
    rows, unmatched = harbor.keyed_rows([("0001", {"n": 1}), ("a0001", {"n": 2}), ("full_id", {"n": 3})],
                                        ["xa0001", "yb0001", "full_id"])
    assert rows == {"xa0001": {"n": 2}, "full_id": {"n": 3}} and unmatched == 1


def test_a_label_file_that_is_not_a_list_refuses_the_export(tmp_path):
    labels = tmp_path / "labels.json"
    write_json(labels, {"task": TASK})
    with pytest.raises(harbor.ExportError):
        harbor.export_harbor(_workdir(tmp_path / "work"), tmp_path / "out", name="widgets", wheel=_wheel(tmp_path),
                             labels=[labels])


def test_a_leak_names_the_task_masked_and_the_part_and_the_count_never_the_string(tmp_path):
    workdir = _workdir(tmp_path / "work")
    turn = "wants widget w1 to carry the label striped because the old plain label kept confusing the warehouse"
    spec = Spec(task_id=TASK, intent=SpecIntent(task_id=TASK, facts=[_fact("f1", turn)]))
    write_json(workdir / "spec" / f"{TASK}.json", spec.model_dump())
    write_json(workdir / "raw" / "export.json", [{"turn": turn}])
    with pytest.raises(harbor.ExportError) as refused:
        harbor.export_harbor(workdir, tmp_path / "out", name="widgets", wheel=_wheel(tmp_path))
    message = str(refused.value)
    assert f"Task {TASK[-4:]} instruction 1" in message and turn not in message and TASK not in message


def test_a_rater_words_that_repeat_a_recorded_turn_refuse_the_export_as_a_label_leak(tmp_path):
    workdir = _workdir(tmp_path / "work")
    turn = "the customer said the old plain label kept confusing everyone in the north warehouse all week"
    write_json(workdir / "raw" / "export.json", [{"turn": turn}])
    labels = tmp_path / "labels.json"
    write_json(labels, [{"task": TASK, "label": "right", "matches_gold": "yes", "gold_diff": turn}])
    with pytest.raises(harbor.ExportError) as refused:
        harbor.export_harbor(workdir, tmp_path / "out", name="widgets", wheel=_wheel(tmp_path), labels=[labels])
    assert f"Task {TASK[-4:]} label 1" in str(refused.value)


def test_the_manifest_keeps_the_task_set_and_the_tier_and_label_counts_the_card_leaves_out(tmp_path, monkeypatch):
    from kullback.hub import publish

    workdir = _workdir(tmp_path / "work", trusted=False)
    monkeypatch.setattr(harbor, "build_wheel", lambda out: _wheel(tmp_path))
    write_json(workdir / "tiers.json", {"rows": [{"task_id": TASK, "tier": "untrusted", "reason": "open_ruling"}]})
    labels = tmp_path / "labels.json"
    write_json(labels, [{"task": TASK[-4:], "label": "partial", "matches_gold": "yes", "gold_diff": "same"}])
    _, manifest = publish.publish(workdir, "someorg/widgets", client=LocalHub(tmp_path / "local"), preview=True,
                                  form="harbor", harbor_options={"tasks": "verified", "labels": [labels],
                                                                 "labels_rubric": "rubric r1"})
    readme = (tmp_path / "local" / "someorg" / "widgets" / "README.md").read_text()
    assert "| Tier |" not in readme and "rubric r1" not in readme and "Task set" not in readme
    assert manifest["harbor"]["tasks_chosen"] == "verified" and manifest["harbor"]["labels"]["rows"] == 1
    assert manifest["harbor"]["counts"]["by_tier_and_label"] == {"open_ruling": {"partial": 1}}


ANSWER = "the widget now carries the striped label and every other widget is exactly as it was before"
ANCHORED = [{"id": "j1", "kind": "communicate", "gate": False, "judge": True, "weight": 1.0,
             "description": "says what changed",
             "target": {"question": "Does the agent say what changed?", "anchor": {"answer": ANSWER}}}]


def test_a_task_whose_only_leaking_part_is_its_oracle_is_left_out_and_listed_masked(tmp_path):
    workdir = _workdir(tmp_path / "work", atoms=ANCHORED)
    write_json(workdir / "raw" / "export.json", [{"turn": f"{harbor.DONE_LINE} {ANSWER}"}])
    out = tmp_path / "out"
    manifest = harbor.export_harbor(workdir, out, name="widgets", wheel=_wheel(tmp_path))
    assert manifest["counts"]["exported"] == 0 and manifest["counts"]["skipped"] == {"solution leaks": 1}
    assert {"task": TASK[-4:], "reason": "solution leaks"} in manifest["left_out"]
    assert manifest["leak_scan"]["leaks"] == 0
    assert manifest["leak_scan"]["solution_leaks"] == [{"task": TASK[-4:], "part": "oracle", "values": 1}]
    assert not (out / "widgets-widget_task").exists() and (out / "manifest.json").is_file()


def test_a_task_whose_oracle_and_another_part_leak_still_refuses_the_export():
    scan = {"leaking_files": [{"task": "0001", "part": "oracle", "values": 2},
                              {"task": "0001", "part": "Verifier", "values": 1},
                              {"task": "0002", "part": "oracle", "values": 1}]}
    scanned = [{"task": "a0001", "name": "x-a0001"}, {"task": "a0002", "name": "x-a0002"}]
    assert [row["task"] for row in harbor.solution_only(scan, scanned)] == ["a0002"]


def test_the_card_runs_harbor_under_use_it_without_the_tier_table_or_left_out_task_ids():
    from kullback.hub.card import card_markdown

    harbor_manifest = {"counts": {"exported": 2, "skipped": {"solution leaks": 1},
                                  "by_tier_and_label": {"trusted": {"right": 2}}},
                       "left_out": [{"task": "0001", "reason": "solution leaks"}],
                       "tasks": [{"path": "widgets-one"}], "labels": {"rubric": "r"}}
    card = card_markdown({"name": "widgets"}, "someorg/widgets", harbor=harbor_manifest)
    usage, inside = card.index("## Use it"), card.index("## What is inside")
    assert usage < card.index("`harbor/` holds 2 Tasks") < card.index("harbor run -p harbor/widgets-one -a oracle") < inside
    assert "0001" not in card and "| Tier |" not in card and "rubric" not in card
    assert "harbor/" not in card_markdown({"name": "widgets"}, "someorg/widgets")


def test_the_organisation_card_carries_no_news_section():
    from kullback.hub.card import organisation_card

    assert "## News" not in organisation_card([])


def test_verified_leaves_out_a_task_the_tier_report_rules_no_verifier_though_a_draft_is_on_disk(tmp_path):
    workdir = _workdir(tmp_path / "work", trusted=False)
    write_json(workdir / "tiers.json", {"rows": [{"task_id": TASK, "tier": "untrusted", "reason": "no_verifier"}]})
    manifest = harbor.export_harbor(workdir, tmp_path / "out", name="widgets", wheel=_wheel(tmp_path), tasks="verified")
    assert manifest["counts"]["exported"] == 0 and manifest["counts"]["skipped"] == {"no Verifier": 1}
    kept = harbor.export_harbor(workdir, tmp_path / "all", name="widgets", wheel=_wheel(tmp_path), tasks="all")
    assert kept["counts"]["exported"] == 1


def _no_passing_run(root: Path) -> Path:
    """The invented world where the one Run on disk writes the wrong label, so none passes on replay."""
    workdir = _workdir(root, trusted=False)
    verifier = json.loads((workdir / "verifiers" / f"{TASK}.json").read_text())
    expected = [{"cells": [{"table": "widgets", "row_id": "w1", "field": "label", "value": "dotted"}]}]
    write_json(workdir / "verifiers" / f"{TASK}.json", dict(verifier, expected=expected))
    return workdir


def test_verified_ships_a_task_no_run_passes_for_without_a_solution_marked_has_oracle_false(tmp_path):
    out = tmp_path / "out"
    manifest = harbor.export_harbor(_no_passing_run(tmp_path / "work"), out, name="widgets", wheel=_wheel(tmp_path),
                                    tasks="verified")
    folder = out / "widgets-widget_task"
    assert manifest["counts"]["exported"] == 1 and not (folder / "solution").exists()
    assert (folder / "tests" / "verifier.json").is_file()
    meta = _toml(out)
    assert meta["has_oracle"] is False and meta["no_oracle_reason"] == harbor.NO_ORACLE


def test_a_task_shipped_without_oracle_is_counted_and_listed_masked_under_no_oracle_not_left_out(tmp_path):
    manifest = harbor.export_harbor(_no_passing_run(tmp_path / "work"), tmp_path / "out", name="widgets",
                                    wheel=_wheel(tmp_path), tasks="all")
    assert manifest["counts"]["no_oracle"] == 1
    assert manifest["no_oracle"] == [{"task": TASK[-4:], "reason": harbor.NO_ORACLE}] and manifest["left_out"] == []


def test_under_trusted_a_task_no_run_passes_for_is_still_left_out(tmp_path):
    workdir = _no_passing_run(tmp_path / "work")
    write_json(workdir / "rounds.json", [{"round": 1, "counts": {"trusted_ids": [TASK]}}])
    manifest = harbor.export_harbor(workdir, tmp_path / "out", name="widgets", wheel=_wheel(tmp_path))
    assert manifest["counts"]["exported"] == 0 and manifest["counts"]["skipped"] == {harbor.NO_ORACLE: 1}


def test_a_task_with_an_oracle_is_marked_has_oracle_true(dataset):
    out, manifest = dataset
    assert _toml(out)["has_oracle"] is True and manifest["counts"]["no_oracle"] == 0


def test_the_leak_scan_has_no_oracle_part_to_scan_for_a_task_without_one(tmp_path):
    manifest = harbor.export_harbor(_no_passing_run(tmp_path / "work"), tmp_path / "out", name="widgets",
                                    wheel=_wheel(tmp_path), tasks="verified")
    assert manifest["leak_scan"]["files_scanned"] == 3


def test_the_known_issues_count_tasks_without_a_solution_and_say_the_oracle_agent_has_nothing_to_run():
    from kullback.hub.card import card_markdown

    card = card_markdown({"name": "widgets"}, "someorg/widgets", harbor={"counts": {"exported": 3, "no_oracle": 2}})
    issues = card.split("## Known issues")[1].split("## How it was built")[0]
    assert "2 Tasks have no `solution/`, so Harbor's oracle agent has nothing to run on them." in issues


def _staged(tmp_path: Path, monkeypatch) -> tuple[Path, Path]:
    """A dry run laid out by publish: the workdir and the staged package folder."""
    from kullback.hub import publish

    workdir = _workdir(tmp_path / "work")
    monkeypatch.setattr(harbor, "build_wheel", lambda out: _wheel(tmp_path))
    write_json(workdir / "replays.json", {TASK: {"rec1": {"confirmed": True, "trace_id": "rec1"}}})
    publish.publish(workdir, "someorg/widgets", client=LocalHub(tmp_path / "first"), preview=True, form="harbor")
    return workdir, tmp_path / "first" / "someorg" / "widgets"


def test_publishing_from_a_staged_package_gives_the_manifest_a_plain_stage_gives(tmp_path, monkeypatch):
    from kullback.hub import publish

    workdir, staged = _staged(tmp_path, monkeypatch)
    plain = publish.stage(workdir, tmp_path / "plain", "someorg/widgets", preview=True, harbor=True)
    _, reused = publish.publish(workdir, "someorg/widgets", client=LocalHub(tmp_path / "second"), preview=True,
                                form="harbor", staged=staged)
    landed = tmp_path / "second" / "someorg" / "widgets"
    assert {k: v for k, v in reused.items() if k != "created_at"} == {k: v for k, v in plain.items() if k != "created_at"}
    assert reused["created_at"] == read_json(staged / "manifest.json")["created_at"]
    assert verify_package(landed) == [] and (landed / HARBOR_DIR / "widgets-widget_task" / "task.toml").is_file()


def test_a_staged_package_with_a_tampered_file_is_refused_naming_the_file(tmp_path, monkeypatch):
    from kullback.hub import publish

    workdir, staged = _staged(tmp_path, monkeypatch)
    tampered = next(path for path in sorted((staged / "verifiers").glob("*.json")))
    tampered.write_text("{}")
    with pytest.raises(publish.PublishError, match=f"verifiers/{tampered.name}"):
        publish.publish(workdir, "someorg/widgets", client=LocalHub(tmp_path / "second"), preview=True,
                        form="harbor", staged=staged)


def test_a_card_published_from_a_staged_package_names_the_commit_its_harbor_tasks_were_laid_at(
        tmp_path, monkeypatch):
    from kullback.hub import publish

    workdir, staged = _staged(tmp_path, monkeypatch)
    manifest = read_json(staged / "manifest.json")
    write_json(staged / "manifest.json", dict(manifest, git_sha="0" * 40))
    publish.publish(workdir, "someorg/widgets", client=LocalHub(tmp_path / "second"), preview=True,
                    form="harbor", staged=staged)
    landed = tmp_path / "second" / "someorg" / "widgets"
    laid = read_json(landed / HARBOR_DIR / "manifest.json")["git_sha"]
    card = (landed / "README.md").read_text(encoding="utf-8")
    assert f"package built at git {'0' * 16}, Harbor Tasks laid at git {laid[:16]}" in card


# --- the pool (D401) ------------------------------------------------------------------------------

TASK_TWO = f"{TASK}_two"


def _two_tasks(root: Path) -> Path:
    """The finished build with a second trusted Task, a copy of the first under its own id."""
    workdir = _workdir(root)
    rename = (f'"{TASK}"', f'"{TASK_TWO}"')
    for folder in ("tasks", "overlays", "verifiers", "spec"):
        source = workdir / folder / f"{TASK}.json"
        (workdir / folder / f"{TASK_TWO}.json").write_text(source.read_text().replace(*rename))
    (workdir / "runs" / TASK_TWO).mkdir()
    run = (workdir / "runs" / TASK / "reference.jsonl").read_text().replace(*rename)
    (workdir / "runs" / TASK_TWO / "reference.jsonl").write_text(run)
    references = read_json(workdir / "references.json")
    write_json(workdir / "references.json", dict(references, **{TASK_TWO: references[TASK]}))
    write_json(workdir / "rounds.json", [{"round": 1, "counts": {"trusted_ids": [TASK, TASK_TWO]}}])
    return workdir


def _laid(out: Path) -> dict[str, bytes]:
    return {path.relative_to(out).as_posix(): path.read_bytes() for path in sorted(out.rglob("*"))
            if path.is_file() and path.name != "manifest.json"}


def test_the_dataset_laid_on_a_pool_is_byte_identical_to_the_one_laid_task_by_task(tmp_path):
    workdir = _two_tasks(tmp_path / "work")
    serial = harbor.export_harbor(workdir, tmp_path / "serial", name="widgets", wheel=_wheel(tmp_path), workers=1)
    pooled = harbor.export_harbor(workdir, tmp_path / "pooled", name="widgets", wheel=_wheel(tmp_path), workers=2)
    assert serial["counts"]["exported"] == 2
    assert pooled["content_hash"] == serial["content_hash"] and pooled["tasks"] == serial["tasks"]
    assert _laid(tmp_path / "pooled") == _laid(tmp_path / "serial")


def test_a_recorded_turn_still_refuses_the_export_when_the_tasks_are_laid_on_a_pool(tmp_path):
    workdir = _two_tasks(tmp_path / "work")
    turn = "wants widget w1 to carry the label striped because the old plain label kept confusing the warehouse"
    for task_id in (TASK, TASK_TWO):
        spec = Spec(task_id=task_id, intent=SpecIntent(task_id=task_id, facts=[_fact("f1", turn)]))
        write_json(workdir / "spec" / f"{task_id}.json", spec.model_dump())
    write_json(workdir / "raw" / "export.json", [{"turn": turn}])
    with pytest.raises(harbor.ExportError, match="instruction"):
        harbor.export_harbor(workdir, tmp_path / "out", name="widgets", wheel=_wheel(tmp_path), workers=2)
    assert not (tmp_path / "out").exists()


def test_the_workers_flag_reaches_the_pool_on_export_and_publish(tmp_path, monkeypatch):
    from typer.testing import CliRunner

    from kullback import cli, processes

    asked: list = []
    real = processes.each

    def counted(items, fn, workers=1):
        asked.append(workers)
        return real(items, fn, workers)

    monkeypatch.setattr(processes, "each", counted)
    monkeypatch.setattr(harbor, "build_wheel", lambda out: _wheel(tmp_path))
    workdir = _two_tasks(tmp_path / "work")
    result = CliRunner().invoke(cli.app, ["export", "--workdir", str(workdir), "--out", str(tmp_path / "out"),
                                          "--format", "harbor", "--workers", "1"])
    assert result.exit_code == 0, result.output
    assert asked == [1]
    write_json(workdir / "replays.json", {TASK: {"rec1": {"confirmed": True, "trace_id": "rec1"}}})
    result = CliRunner().invoke(cli.app, ["publish", "--workdir", str(workdir), "--repo", "someorg/widgets",
                                          "--preview", "--format", "harbor", "--workers", "3",
                                          "--dry-run", str(tmp_path / "local")])
    assert result.exit_code == 0, result.output
    assert asked[1:] == [3]


def test_the_package_ships_the_same_ask_as_the_harbor_instruction_for_a_task_with_a_spec_intent(tmp_path):
    from kullback.hub import package

    workdir = _workdir(tmp_path / "work")
    harbor.export_harbor(workdir, tmp_path / "harbor", name="widgets", wheel=_wheel(tmp_path))
    manifest = package.export(workdir, tmp_path / "package", name="widgets", preview=True)
    instruction = (tmp_path / "harbor" / "widgets-widget_task" / "instruction.md").read_text(encoding="utf-8")
    ask = read_json(tmp_path / "package" / "tasks" / f"{TASK}.json")["intent"]
    assert ask == ASK
    assert instruction.startswith(ask + "\n")
    row = json.loads((tmp_path / "package" / package.TASKS_ROWS_NAME).read_text(encoding="utf-8").splitlines()[0])
    assert row["instruction"] == ask and manifest["without_ask"] == 0


def test_end_facts_of_a_born_task_never_reach_the_package_ask_or_the_harbor_instruction(tmp_path):
    from kullback.hub import package
    from kullback.spec.schema import load_spec, save_spec
    from kullback.spec.writer import with_end

    workdir = _workdir(tmp_path / "work")
    spec = load_spec(workdir, TASK)
    end = "Widget w1 carries the label striped; the user is told it is striped."
    save_spec(workdir, spec.model_copy(update={"intent": with_end(spec.intent, end)}))
    harbor.export_harbor(workdir, tmp_path / "harbor", name="widgets", wheel=_wheel(tmp_path))
    package.export(workdir, tmp_path / "package", name="widgets", preview=True)
    ask = read_json(tmp_path / "package" / "tasks" / f"{TASK}.json")["intent"]
    assert ask == ASK
    instruction = (tmp_path / "harbor" / "widgets-widget_task" / "instruction.md").read_text(encoding="utf-8")
    assert "carries the label striped;" not in instruction and "is told it is striped" not in instruction


def test_a_task_with_neither_a_spec_intent_nor_intent_text_ships_an_empty_ask_and_the_manifest_counts_it(tmp_path):
    from kullback.hub import package

    workdir = _workdir(tmp_path / "work")
    (workdir / "spec" / f"{TASK}.json").unlink()
    record = read_json(workdir / "tasks" / f"{TASK}.json")
    write_json(workdir / "tasks" / f"{TASK}.json", dict(record, intent=None))
    manifest = package.export(workdir, tmp_path / "package", name="widgets", preview=True)
    assert read_json(tmp_path / "package" / "tasks" / f"{TASK}.json")["intent"] == ""
    row = json.loads((tmp_path / "package" / package.TASKS_ROWS_NAME).read_text(encoding="utf-8").splitlines()[0])
    assert row["instruction"] == "" and manifest["without_ask"] == 1


# --- what the user said later reaches the instruction; an item it cannot satisfy never ships (D422) ---


def _said(*facts: IntentFact) -> str:
    return package.opening(package.asked_facts(Spec(task_id=TASK, intent=SpecIntent(task_id=TASK, facts=list(facts)))),
                           None) or ""


def test_a_detail_the_user_gave_when_the_agent_asked_reaches_the_instruction():
    text = _said(_fact("f1", "wants widget w1 relabelled", kind="do"),
                 _fact("f2", "badge number is 00000", stance="fallback", condition="world_or_policy"),
                 _fact("f3", "account name is ann_0000", stance="accepted"))
    assert "- badge number is 00000" in text and "- account name is ann_0000" in text


def test_a_fallback_ask_reaches_the_instruction_under_the_later_lead_after_the_volunteered_ones():
    text = _said(_fact("f2", "wants widget w1 removed instead if the label cannot change", stance="fallback",
                       kind="do", condition="world_or_policy"),
                 _fact("f1", "wants widget w1 to carry the label striped", kind="do"))
    first, later = text.split(package.LATER_LEAD)
    assert "- wants widget w1 to carry the label striped" in first
    assert later.strip() == "- wants widget w1 removed instead if the label cannot change"


def test_an_instruction_with_only_volunteered_facts_has_no_later_lead():
    assert package.LATER_LEAD not in _said(_fact("f1", "wants widget w1 to carry the label striped"))


UNRENDERED = [{"id": "keep", "kind": "allowed", "gate": True, "description": "w1 holds the label",
               "target": {"kind": "row_is", "table": "widgets", "row": "w1", "expect": {"label": "striped"},
                          "from": {"fact_ids": ["f3"], "policy_line": None}}},
              {"id": "named", "kind": "allowed", "gate": True, "description": "rename_widget is called",
               "target": {"kind": "called", "tool": "rename_widget", "from": {"fact_ids": ["f1", "f3"],
                                                                              "policy_line": None}}},
              JUDGE_WEIGHTED[0]]


def test_an_item_resting_only_on_unrendered_facts_is_dropped_its_cell_freed_and_counted(tmp_path):
    workdir = _workdir(tmp_path / "work", atoms=UNRENDERED)
    manifest = harbor.export_harbor(workdir, tmp_path / "out", name="widgets", wheel=_wheel(tmp_path))
    shipped = read_json(tmp_path / "out" / "widgets-widget_task" / "tests" / harbor.VERIFIER_NAME)
    assert [atom["id"] for atom in shipped["atoms"]] == ["named", "a1"]
    state = shipped["expected"][0]
    assert not state["cells"]
    assert {"table": "widgets", "row_id": "w1", "field": "label", "why": harbor.UNRENDERABLE_WHY} in state["allowed"]
    assert manifest["counts"]["items_unrenderable"] == 1 and manifest["counts"]["tasks_with_items_unrenderable"] == 1
    assert manifest["items_unrenderable"] == [{"task": TASK[-4:], "items": ["keep"]}]
    assert manifest["tasks"][0]["items_unrenderable"] == ["keep"] and manifest["tasks"][0]["has_oracle"]


def test_a_verifier_with_nothing_unrendered_ships_byte_for_byte(dataset, tmp_path):
    out, manifest = dataset
    source = tmp_path / "work" / "verifiers" / f"{TASK}.json"
    assert (out / "widgets-widget_task" / "tests" / harbor.VERIFIER_NAME).read_bytes() == source.read_bytes()
    assert manifest["counts"]["items_unrenderable"] == 0 and manifest["items_unrenderable"] == []
