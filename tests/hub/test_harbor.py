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

from kullback.hub import harbor, harbor_world
from kullback.hub.client import LocalHub
from kullback.hub.package import HARBOR_DIR, verify_package
from kullback.runner.records import ItemResult, read_json, write_json
from kullback.runner.world.environment import BuiltEnvironment
from kullback.runner.world.episode import Episode
from kullback.spec.schema import FactSource, IntentFact, Spec, SpecIntent
from kullback.spec.trust import runs_on_disk
from tests.episode.invented import SCRIPTED_RENAME, write_env

TASK = "widget_task"
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


def test_the_instruction_says_only_what_the_user_volunteered_then_the_policy_and_the_tools(dataset):
    out, _ = dataset
    text = (out / "widgets-widget_task" / "instruction.md").read_text()
    assert "wants widget w1 to carry the label striped" in text
    assert "proposed" not in text and "only the agent said" not in text
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


def test_with_no_judge_named_a_gated_judge_item_fails_closed_and_judge_json_says_so(tmp_path, monkeypatch):
    out = _judged(tmp_path)
    monkeypatch.delenv(harbor_world.JUDGE_VAR, raising=False)
    body = _oracle_score(out, tmp_path)
    assert body["reward"] == 0.0 and body["unscored_gates"] == ["j1"]
    assert harbor_world.JUDGE_VAR in json.loads((tmp_path / "logs" / "judge.json").read_text())["error"]


def test_a_judge_that_cannot_start_fails_closed(tmp_path, monkeypatch):
    out = _judged(tmp_path)

    def refused(model_id):
        raise RuntimeError("live model requests are off")
    monkeypatch.setattr(harbor_world, "live_model", refused)
    monkeypatch.setenv(harbor_world.JUDGE_VAR, "bedrock/anthropic.some-judge")
    body = _oracle_score(out, tmp_path)
    assert body["reward"] == 0.0 and body["unscored_gates"] == ["j1"]
    assert "live model requests are off" in json.loads((tmp_path / "logs" / "judge.json").read_text())["error"]


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
    assert "## Run it with Harbor" in (landed / "README.md").read_text()
    assert manifest["harbor"]["counts"]["exported"] == 1
    assert read_json(landed / "manifest.json")["harbor"] == manifest["harbor"]
    readme = (landed / "README.md").read_text()
    assert "KULLBACK_JUDGE_MODEL" in readme and "HARNESS_ALLOW_MODEL_REQUESTS=1" in readme
    assert verify_package(landed) == []


# --- the verified Task set, tiers and labels (D392) ----------------------------------------------


def _wheel(tmp_path: Path) -> Path:
    wheel = tmp_path / "k.whl"
    wheel.write_bytes(b"")
    return wheel


def _toml(out: Path) -> dict:
    return tomllib.loads((out / "widgets-widget_task" / "task.toml").read_text())["metadata"]


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


def test_the_card_names_the_task_set_and_counts_tasks_by_tier_and_label_under_the_named_rubric(tmp_path, monkeypatch):
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
    assert "Task set: `verified`, every Task with a Verifier, trusted or not; 1 exported" in readme
    assert "| Tier | partial | Tasks |" in readme and "| open_ruling | 1 | 1 |" in readme
    assert "one blind model rater's judgement" in readme and "rubric r1" in readme
    assert manifest["harbor"]["tasks_chosen"] == "verified" and manifest["harbor"]["labels"]["rows"] == 1
