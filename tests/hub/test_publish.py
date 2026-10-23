"""Exporting an Environment, publishing it and fetching it back (D221)."""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import pytest
from nursery_domain import ELICITED_VALUE, RECORDED_TURN, build_unclosed_workdir
from typer.testing import CliRunner

from gates.verifier_fixtures import call, make_run, result
from kullback import cli
from kullback.gates import verifier_suite as S
from kullback.hub import card as card_mod
from kullback.hub import package as package_mod
from kullback.hub import publish as publish_mod
from kullback.runner.canon import CanonRules
from kullback.runner.records import RawPtr, Trace, Verifier, as_dict, read_json, write_json
from kullback.runner.tool import ReplayReport
from kullback.runner.world.environment import BuiltEnvironment


def _export(workdir: Path, out: Path, **kwargs):
    return package_mod.export(workdir, out, name="nursery", corpus="nursery traces",
                              corpus_license="MIT", **kwargs)


# --- the package ---------------------------------------------------------------------


def test_the_export_writes_the_world_the_tasks_and_the_verifiers_and_nothing_of_the_recordings(
        nursery, tmp_path, hub):
    out = tmp_path / "package"
    _export(nursery, out)
    assert (out / "env" / "db.json").is_file()
    assert (out / "env" / "policy.md").is_file()
    assert sorted(p.name for p in (out / "tasks").glob("*.json")) == \
        ["task_one.json", "task_three.json", "task_two.json"]
    assert sorted(p.name for p in (out / "verifiers").glob("*.json")) == ["task_one.json", "task_two.json"]
    written = {path.relative_to(out).as_posix().split("/", 1)[0] for path in out.rglob("*") if path.is_file()}
    assert not written & set(package_mod.NEVER_EXPORTED)
    task = read_json(out / "tasks" / "task_one.json")
    assert task["run_ids"] == [], "an exported Task names none of the recordings it was clustered from"
    assert task["intent"] == "water the plot that is due"

    # and what is fetched back holds no builder state either
    publish_mod.publish(nursery, "leibler/nursery", client=hub, preview=True)
    fetched = tmp_path / "fetched"
    publish_mod.fetch("leibler/nursery", fetched, client=hub)
    for name in ("raw", "runs", "task_status.json", "rounds.json", "replays.json"):
        assert not (fetched / name).exists()


def test_the_manifest_carries_the_numbers_the_artifacts_hold(nursery, tmp_path):
    manifest = _export(nursery, tmp_path / "package")
    assert manifest["round"] == 2
    assert manifest["tasks_total"] == 3
    assert manifest["replay_fidelity"] == {"tasks": 2, "tasks_total": 3, "tasks_rate": pytest.approx(0.6667, abs=1e-3),
                                           "runs": 2, "runs_total": 3, "runs_rate": pytest.approx(0.6667, abs=1e-3),
                                           "calls": 0, "calls_total": 0, "calls_rate": None}
    assert manifest["reference_confirmed"] == 2
    assert manifest["verifier_derived"] == 2
    assert manifest["trusted"] == 1
    assert manifest["runner_version"] == "runner-hash-1"
    assert manifest["env_id"] == "env-hash-1"
    assert manifest["source"] == {"corpus": "nursery traces", "license": "MIT", "url": None}
    assert manifest["content_hash"] and manifest["files"]


def test_a_build_with_no_closed_round_publishes_the_workdir_status_and_a_date_tag(tmp_path, hub):
    workdir = build_unclosed_workdir(tmp_path / "work")
    sessions = workdir / "sessions"
    sessions.mkdir()
    (sessions / "builder.jsonl").write_text("{}\n", encoding="utf-8")
    stamp = datetime(2026, 9, 23, 12, tzinfo=timezone.utc).timestamp()
    os.utime(sessions / "builder.jsonl", (stamp, stamp))
    hosted, manifest = publish_mod.publish(workdir, "leibler/nursery", client=hub, preview=True)
    assert manifest["round"] is None
    assert manifest["reference_confirmed"] == 3
    assert manifest["replay_fidelity"]["tasks"] == 3 and manifest["tasks_total"] == 4
    assert manifest["trusted"] == 2
    assert manifest["tag"] == hosted.tag == "build-20260923"
    assert manifest["counts_source"] == package_mod.NO_ROUND_SOURCE
    card = hub.repos["leibler/nursery"]["commits"][hosted.commit]["README.md"].decode()
    assert "Built: build-20260923, git " in card
    assert "round-unknown" not in card and "round-unknown" not in hub.messages[0]


def test_a_build_with_a_round_record_keeps_the_rounds_trusted_count_and_tag(nursery, tmp_path, hub):
    hosted, manifest = publish_mod.publish(nursery, "leibler/nursery", client=hub, preview=True)
    assert manifest["trusted"] == 1 and manifest["round"] == 2
    assert manifest["tag"] == hosted.tag == "round-2"
    assert manifest["counts_source"] == "the last round record"


def test_every_task_carries_its_funnel_stage_and_nothing_about_why_it_stopped(nursery, tmp_path):
    out = tmp_path / "package"
    manifest = _export(nursery, out)
    rows = {row["task_id"]: row for row in read_json(out / package_mod.TASKS_INDEX_NAME)["tasks"]}
    assert {task_id: row["stage"] for task_id, row in rows.items()} == {
        "task_one": "trusted", "task_two": "verifier_derived", "task_three": "clustered"}
    assert all("stopped_because" not in row for row in rows.values())
    assert "untrusted" not in manifest


def test_a_workdir_with_no_environment_is_refused(tmp_path):
    with pytest.raises(package_mod.ExportError, match="no env/"):
        package_mod.export(tmp_path / "empty", tmp_path / "package")


# --- the leak scan -------------------------------------------------------------------


def test_the_leak_scan_passes_a_package_whose_values_the_world_files_or_the_harness_words_hold(nursery, tmp_path):
    manifest = _export(nursery, tmp_path / "package")
    assert manifest["leak_scan"]["leaks"] == 0
    assert manifest["leak_scan"]["corpus_strings"] > 0
    assert manifest["leak_scan"]["files_scanned"] >= 4
    # a word of the harness's own vocabulary is not a leak however often the corpus says it
    kinds = {atom["kind"] for atom in read_json(nursery / "verifiers" / "task_one.json")["atoms"]}
    assert "question" in kinds  # a word the corpus and the harness both use

    # nor is a value the world files hold, however often the recordings say it
    verifier = read_json(nursery / "verifiers" / "task_one.json")
    verifier["atoms"][1]["target"]["value"] = "2026-03-01"  # a value db.json holds
    write_json(nursery / "verifiers" / "task_one.json", verifier)
    manifest = _export(nursery, tmp_path / "held")
    assert manifest["leak_scan"]["leaks"] == 0


def test_the_leak_scan_fails_an_export_that_copies_a_recorded_turn_into_a_grader(nursery, tmp_path):
    verifier = read_json(nursery / "verifiers" / "task_one.json")
    verifier["atoms"][1]["description"] = RECORDED_TURN
    write_json(nursery / "verifiers" / "task_one.json", verifier)
    out = tmp_path / "package"
    with pytest.raises(package_mod.ExportError, match="leak scan found 1"):
        _export(nursery, out)
    assert not out.exists()


def test_the_leak_scan_finds_a_recorded_turn_quoted_inside_a_description_or_a_predicate(nursery, tmp_path):
    verifier = read_json(nursery / "verifiers" / "task_one.json")
    verifier["atoms"][1]["description"] = f"the desk notes {RECORDED_TURN} and waters"
    verifier["atoms"][1]["predicate_src"] = f"wrote('water_plot', **{{'note': {RECORDED_TURN!r}}})"
    write_json(nursery / "verifiers" / "task_one.json", verifier)
    with pytest.raises(package_mod.ExportError, match="leak scan found 1"):
        _export(nursery, tmp_path / "package")


def test_an_old_verifier_pinning_agent_prose_exports_through_the_leak_scan_and_grades_its_reference_as_before(
        nursery, tmp_path):
    """D292: a Verifier derived before the fix is reduced as it is packaged, and the workdir is left alone."""
    source = nursery / "verifiers" / "task_one.json"
    verifier = read_json(source)
    fn = S.canon_fn(CanonRules())
    verifier["atoms"][0]["target"]["entity"] = fn("P-1")  # keyed the way a derivation keys it
    payload = {"kind": "write_value", "tool": "water_plot", "entity": fn("P-1"), "entity_raw": "P-1",
               "id_field": "plot_id", "field": "note", "value": S._key(fn, RECORDED_TURN), "raw": RECORDED_TURN}
    # the write the reference makes, and the old atom pinning the note the recorded agent wrote on it
    verifier["atoms"] = [verifier["atoms"][0], as_dict(S.make_atom(
        "w0.note", "allowed", payload, provenance="agent_chosen", description=f"water_plot note is {RECORDED_TURN}"))]
    write_json(source, verifier)
    before = source.read_bytes()

    manifest = _export(nursery, tmp_path / "package")
    assert manifest["leak_scan"]["leaks"] == 0
    assert source.read_bytes() == before, "the workdir's Verifier is never rewritten by an export"
    exported = (tmp_path / "package" / "verifiers" / "task_one.json").read_text(encoding="utf-8")
    assert RECORDED_TURN[:40] not in exported
    assert (tmp_path / "package" / "verifiers" / "task_two.json").read_bytes() == \
        (nursery / "verifiers" / "task_two.json").read_bytes(), "a Verifier with nothing to reduce ships as is"

    reference = make_run("reference", [call("water_plot", {"plot_id": "P-1", "note": RECORDED_TURN}),
                                       result({"status": "watered"})])
    old, new = Verifier.model_validate(verifier), Verifier.model_validate_json(exported)
    for run in (reference, make_run("empty", [])):
        assert S.check_run(new, run, CanonRules(), write_tools={"water_plot"}) == \
            S.check_run(old, run, CanonRules(), write_tools={"water_plot"})
    assert S.check_run(new, reference, CanonRules(), write_tools={"water_plot"}) == (True, None)


def test_a_required_atom_pinning_recorded_agent_prose_is_never_reduced_and_the_leak_scan_refuses_it(
        nursery, tmp_path):
    """Only an allowed prose atom is reduced; a required one ships as it is, so the scan fails loud."""
    source = nursery / "verifiers" / "task_one.json"
    verifier = read_json(source)
    fn = S.canon_fn(CanonRules())
    payload = {"kind": "write_value", "tool": "water_plot", "entity": fn("P-1"), "entity_raw": "P-1",
               "id_field": "plot_id", "field": "note", "value": S._key(fn, RECORDED_TURN), "raw": RECORDED_TURN}
    required = S.make_atom("w0.note", "required", payload, provenance="agent_chosen",
                           description=f"water_plot note is {RECORDED_TURN}")
    assert S.elide_agent_prose(required) == required
    verifier["atoms"] = [verifier["atoms"][0], as_dict(required)]
    write_json(source, verifier)
    assert S.elide_verifier_prose(Verifier.model_validate(verifier))[1] == 0

    with pytest.raises(package_mod.ExportError, match="leak scan found"):
        _export(nursery, tmp_path / "package")


def test_a_value_the_caller_gave_mid_conversation_is_an_echo_unless_the_task_states_it_and_never_stops_the_export(
        nursery, tmp_path):
    verifier = read_json(nursery / "verifiers" / "task_one.json")
    verifier["atoms"][0]["target"]["raw"] = ELICITED_VALUE
    write_json(nursery / "verifiers" / "task_one.json", verifier)
    manifest = _export(nursery, tmp_path / "package")
    assert manifest["leak_scan"]["leaks"] == 0
    assert manifest["leak_scan"]["value_echoes"] == 1

    task = read_json(nursery / "tasks" / "task_one.json")
    task["intent"] = f"water the plot and send the crew to {ELICITED_VALUE}"
    write_json(nursery / "tasks" / "task_one.json", task)
    manifest = _export(nursery, tmp_path / "stated")
    assert manifest["leak_scan"]["value_echoes"] == 0


# --- the release form ----------------------------------------------------------------


def test_a_release_is_refused_below_the_fidelity_bar_saying_the_number_and_passes_at_it(
        nursery, tmp_path, hub, monkeypatch):
    with pytest.raises(publish_mod.PublishError, match="66.7%"):
        publish_mod.stage(nursery, tmp_path / "package", "leibler/nursery")

    # the publish command refuses the same release and uploads nothing
    monkeypatch.setattr("kullback.hub.client.HuggingFaceHub", lambda *a, **k: hub)
    result = CliRunner().invoke(cli.app, ["publish", "--workdir", str(nursery), "--repo", "leibler/nursery"])
    assert result.exit_code != 0
    assert hub.repos == {}

    write_json(nursery / "replays.json", {
        "task_one": {"trace-task_one": {"confirmed": True}},
        "task_two": {"trace-task_two": {"confirmed": True}},
        "task_three": {"trace-task_three": {"confirmed": True}},
    })
    manifest = publish_mod.stage(nursery, tmp_path / "released", "leibler/nursery")
    assert manifest["status"] == "release"
    assert manifest["preview"] is False


def test_preview_passes_below_the_bar_and_the_card_says_it_is_a_preview_beside_its_fidelity(nursery, tmp_path):
    out = tmp_path / "package"
    manifest = publish_mod.stage(nursery, out, "leibler/nursery", preview=True,
                                 corpus="nursery traces", corpus_license="MIT")
    assert manifest["preview"] is True
    assert manifest["status"] == "preview"
    body = (out / package_mod.CARD_NAME).read_text(encoding="utf-8")
    assert "Preview: 66.7% (2 of 3) of its Tasks replay as recorded." in body
    assert "90%" not in body


def test_an_environment_whose_replays_were_never_scored_cannot_be_a_release():
    with pytest.raises(publish_mod.PublishError, match="not measured"):
        publish_mod.release_ruling({"replay_fidelity": {"tasks_rate": None}}, preview=False)


# --- the card ------------------------------------------------------------------------


def test_the_card_front_matter_parses_and_holds_the_licence_and_the_tags(nursery, tmp_path):
    out = tmp_path / "package"
    publish_mod.stage(nursery, out, "leibler/nursery", preview=True, corpus="nursery traces",
                      corpus_license="MIT")
    body = (out / package_mod.CARD_NAME).read_text(encoding="utf-8")
    assert body.startswith("---\n")
    front = body.split("---", 2)[1]
    fields = _parse_front_matter(front)
    assert fields["license"] == "mit"
    assert set(card_mod.BASE_TAGS) <= set(fields["tags"])
    assert "nursery" in fields["tags"]


def test_a_card_states_the_corpus_licence_the_task_count_and_the_content_hash(nursery, tmp_path):
    out = tmp_path / "package"
    manifest = publish_mod.stage(nursery, out, "leibler/nursery", preview=True,
                                 corpus="nursery traces", corpus_license="MIT")
    body = (out / package_mod.CARD_NAME).read_text(encoding="utf-8")
    assert "nursery traces" in body
    assert "MIT" in body
    assert str(manifest["content_hash"])[:16] in body
    assert "3 Tasks over" in body
    assert "uv run kullback fetch leibler/nursery" in body


def _parse_front_matter(block: str) -> dict:
    fields: dict = {}
    key = None
    for line in block.strip().splitlines():
        if line.startswith("  - ") and key:
            fields.setdefault(key, []).append(line[4:].strip())
        elif ":" in line:
            key, _, value = line.partition(":")
            key = key.strip()
            fields[key] = value.strip() if value.strip() else []
    return fields


# --- publish and fetch ---------------------------------------------------------------


def test_publishing_uploads_the_package_and_tags_the_commit_with_the_round(nursery, tmp_path, hub):
    hosted, manifest = publish_mod.publish(nursery, "leibler/nursery", client=hub, preview=True,
                                           corpus="nursery traces", corpus_license="MIT")
    assert hosted.tag == "round-2"
    assert hub.repos["leibler/nursery"]["tags"]["round-2"] == hosted.commit
    files = hub.repos["leibler/nursery"]["commits"][hosted.commit]
    assert package_mod.CARD_NAME in files
    assert "manifest.json" in files
    assert "round-2" in hub.messages[0]
    assert str(manifest["content_hash"])[:12] in hub.messages[0]


def test_publishing_again_writes_a_new_commit_and_a_new_tag_and_leaves_the_old_one(nursery, tmp_path, hub):
    first, _ = publish_mod.publish(nursery, "leibler/nursery", client=hub, preview=True)
    write_json(nursery / "rounds.json", [{"round": 3, "counts": {
        "round": 3, "fidelity": 3, "tasks": 3, "trusted": 2,
        "trusted_ids": ["task_one", "task_two"], "refused": {}}}])
    second, _ = publish_mod.publish(nursery, "leibler/nursery", client=hub, preview=True)
    tags = hub.repos["leibler/nursery"]["tags"]
    assert set(tags) == {"round-2", "round-3"}
    assert tags["round-2"] == first.commit != second.commit
    assert len(hub.repos["leibler/nursery"]["commits"]) == 2


def _recorded_under(workdir: Path, prompts: dict[str, str]) -> None:
    """Each Task's recording carries its system prompt, with replay rows the Runner's loader accepts."""
    replays = read_json(workdir / "replays.json", {})
    for task_id, prompt in prompts.items():
        write_json(workdir / "traces" / f"trace-{task_id}.json",
                   as_dict(Trace(trace_id=f"trace-{task_id}", raw_hash="raw-1", ingest_version="1", source="test",
                                 system_prompt=prompt, raw_ptr=RawPtr(file_hash="raw-1", sim_index=0))))
        replays[task_id][f"trace-{task_id}"]["trace_id"] = f"trace-{task_id}"
    write_json(workdir / "replays.json", replays)


def test_a_packaged_workdir_passes_the_workdir_check_when_the_build_wrote_no_policy_file_or_environment_json(
        nursery, tmp_path, hub):
    # D404: a build that ran under the recorded agent's prompt and wrote neither file still ships a runnable package
    (nursery / "env" / "policy.md").unlink()
    (nursery / "environment.json").unlink()
    prompt = "# Desk policy\nWater a plot only when it is due.\n"
    _recorded_under(nursery, {task_id: prompt for task_id in ("task_one", "task_two", "task_three")})
    publish_mod.publish(nursery, "leibler/nursery", client=hub, preview=True)
    out = tmp_path / "fetched"
    manifest = publish_mod.fetch("leibler/nursery", out, client=hub)
    assert manifest["missing_run_inputs"] == []
    assert (out / "env" / "policy.md").read_text(encoding="utf-8") == prompt
    assert read_json(out / "environment.json", {})["policy_version"] == hashlib.sha256(prompt.encode()).hexdigest()
    assert BuiltEnvironment(out).system_prompt(BuiltEnvironment(out).task("task_one")) == prompt


def test_tasks_recorded_under_different_prompts_ship_no_policy_and_the_fetch_names_it_missing(nursery, tmp_path, hub):
    (nursery / "env" / "policy.md").unlink()
    _recorded_under(nursery, {task_id: f"prompt {n}" for n, task_id in enumerate(("task_one", "task_two", "task_three"))})
    publish_mod.publish(nursery, "leibler/nursery", client=hub, preview=True)
    manifest = publish_mod.fetch("leibler/nursery", tmp_path / "fetched", client=hub)
    assert manifest["missing_run_inputs"] == ["env/policy.md"]


def test_a_tasks_run_files_never_include_a_sibling_task_whose_id_extends_it(tmp_path):
    for name in ("task_one/a.jsonl", "task_one-0.jsonl", "task_one-12.jsonl", "task_one-two-0.jsonl",
                 "task_one-two/b.jsonl", "task_one-x.jsonl"):
        (tmp_path / "runs" / name).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / "runs" / name).write_text("", encoding="utf-8")
    assert [p.relative_to(tmp_path / "runs").as_posix() for p in cli.task_run_files(tmp_path, "task_one")] == \
        ["task_one/a.jsonl", "task_one-0.jsonl", "task_one-12.jsonl"]


def test_a_fetch_verifies_the_hash_and_lays_out_a_directory_the_runner_loads(nursery, tmp_path, hub,
                                                                             toolkit_source):
    publish_mod.publish(nursery, "leibler/nursery", client=hub, preview=True)
    out = tmp_path / "fetched"
    manifest = publish_mod.fetch("leibler/nursery", out, client=hub)
    assert manifest["missing_run_inputs"] == []
    assert manifest["content_hash"] == package_mod.package_hash(out)
    toolkit = toolkit_source(out)
    assert toolkit.get_plot(plot_id="P-1").crop == "basil"
    toolkit.water_plot(plot_id="P-1", on_date="2026-03-09")
    assert toolkit.get_plot(plot_id="P-1").last_watered == "2026-03-09"


def test_a_fetch_refuses_a_tampered_package_and_leaves_nothing_behind(nursery, tmp_path, hub):
    hosted, _ = publish_mod.publish(nursery, "leibler/nursery", client=hub, preview=True)
    files = hub.repos["leibler/nursery"]["commits"][hosted.commit]
    body = json.loads(files["verifiers/task_one.json"])
    body["atoms"][0]["predicate_src"] = "True"
    files["verifiers/task_one.json"] = json.dumps(body).encode("utf-8")
    out = tmp_path / "fetched"
    with pytest.raises(publish_mod.PublishError, match="does not verify"):
        publish_mod.fetch("leibler/nursery", out, client=hub)
    assert not out.exists()


def test_a_fetch_can_ask_for_an_earlier_round_by_its_tag(nursery, tmp_path, hub):
    publish_mod.publish(nursery, "leibler/nursery", client=hub, preview=True)
    write_json(nursery / "rounds.json", [{"round": 3, "counts": {
        "round": 3, "fidelity": 3, "tasks": 3, "trusted": 2,
        "trusted_ids": ["task_one", "task_two"], "refused": {}}}])
    publish_mod.publish(nursery, "leibler/nursery", client=hub, preview=True)
    old = publish_mod.fetch("leibler/nursery", tmp_path / "old", client=hub, revision="round-2")
    new = publish_mod.fetch("leibler/nursery", tmp_path / "new", client=hub)
    assert old["round"] == 2 and old["trusted"] == 1
    assert new["round"] == 3 and new["trusted"] == 2


# --- the name ------------------------------------------------------------------------


def test_the_environment_name_comes_from_the_flag_and_not_from_the_environment_record(nursery, tmp_path):
    assert "name" not in read_json(nursery / "environment.json")
    named = package_mod.export(nursery, tmp_path / "a", name="hothouse", preview=True)
    assert named["name"] == "hothouse"
    defaulted = publish_mod.stage(nursery, tmp_path / "b", "leibler/nursery", preview=True)
    assert defaulted["name"] == "nursery"


# --- the organisation card ------------------------------------------------------------


def test_the_organisation_card_has_a_row_only_for_each_environment_with_its_status_tasks_and_harbor_tasks(
        nursery, tmp_path):
    manifest = package_mod.export(nursery, tmp_path / "package", name="nursery", preview=True)
    body = card_mod.organisation_card([{"name": "nursery", "repo_id": "leibler/nursery", "manifest": manifest}])
    row = "| [leibler/nursery](https://huggingface.co/datasets/leibler/nursery) | preview | 3 | none |"
    assert row in body
    assert [line for line in body.splitlines() if line.startswith("| [")] == [row]
    assert "kullback fetch leibler/<environment>" in body
    assert manifest["tag"] not in body


def test_an_organisation_card_is_published_where_the_host_takes_it_and_reported_where_it_will_not(
        hub, hub_with_organisation_card):
    took, note = publish_mod.publish_organisation_card("leibler", "# leibler\n", client=hub)
    assert took is False
    assert "organisation profile" in note

    host = hub_with_organisation_card
    took, where = publish_mod.publish_organisation_card("leibler", "# leibler\n", client=host)
    assert took is True and "leibler" in where
    assert host.organisation_cards["leibler"] == "# leibler\n"


# --- the three commands ---------------------------------------------------------------


def test_the_export_command_writes_a_package_and_prints_its_numbers(nursery, tmp_path):
    out = tmp_path / "package"
    result = CliRunner().invoke(cli.app, ["export", "--workdir", str(nursery), "--out", str(out),
                                          "--name", "nursery", "--corpus-license", "MIT"])
    assert result.exit_code == 0, result.output
    assert "replay fidelity 66.7% over Tasks" in result.output
    assert "leak scan: 0 recorded strings" in result.output
    assert (out / package_mod.MANIFEST_NAME).is_file()


def test_the_publish_and_fetch_commands_round_trip_an_environment(nursery, tmp_path, hub, monkeypatch):
    monkeypatch.setattr("kullback.hub.client.HuggingFaceHub", lambda *a, **k: hub)
    published = CliRunner().invoke(cli.app, ["publish", "--workdir", str(nursery),
                                             "--repo", "leibler/nursery", "--preview"])
    assert published.exit_code == 0, published.output
    assert "preview at https://example.invalid/datasets/leibler/nursery, tag round-2" in published.output
    out = tmp_path / "fetched"
    fetched = CliRunner().invoke(cli.app, ["fetch", "leibler/nursery", "--out", str(out)])
    assert fetched.exit_code == 0, fetched.output
    assert "cannot be run as a workdir" not in fetched.output
    assert (out / package_mod.MANIFEST_NAME).is_file()


def test_a_file_the_host_lays_down_beside_the_package_does_not_stop_it_verifying(nursery, tmp_path, hub):
    publish_mod.publish(nursery, "leibler/nursery", client=hub, preview=True)
    out = tmp_path / "fetched"
    publish_mod.fetch("leibler/nursery", out, client=hub)
    (out / ".gitattributes").write_text("*.json filter=lfs\n", encoding="utf-8")
    (out / ".cache" / "downloads").mkdir(parents=True)
    (out / ".cache" / "downloads" / "lock").write_text("", encoding="utf-8")
    assert package_mod.verify_package(out) == []


# --- the viewer rows -----------------------------------------------------------------


def test_the_viewer_rows_carry_one_flat_row_per_task_with_no_reason_it_stopped_and_the_front_matter_names_them(
        nursery, tmp_path):
    out = tmp_path / "package"
    manifest = package_mod.export(nursery, out, name="nursery", preview=True)
    lines = (out / package_mod.TASKS_ROWS_NAME).read_text(encoding="utf-8").splitlines()
    rows = [json.loads(line) for line in lines]
    assert [row["task_id"] for row in rows] == [row["task_id"] for row in read_json(out / package_mod.TASKS_INDEX_NAME)["tasks"]]
    for row in rows:
        task = read_json(out / "tasks" / f"{row['task_id']}.json")
        assert row["instruction"] == task.get("intent", "")
        assert row["stage"] and isinstance(row["trusted"], bool)
        assert set(row) == {"task_id", "instruction", "stage", "trusted", "refused", "bucket", "replay_confirmed",
                            "recordings"}
    assert package_mod.TASKS_ROWS_NAME in manifest["files"]
    body = card_mod.card_markdown(manifest, "leibler/nursery")
    front = body.split("---")[1]
    assert "configs:" in front and package_mod.TASKS_ROWS_NAME in front


# --- what the domain reading attests, on the card (D225) ------------------------------


def test_the_manifest_says_how_many_domain_archetypes_were_read_and_how_many_no_tool_realises_if_any_were(
        nursery, tmp_path):
    """The second number says what this Environment cannot be asked to do; the manifest carries it."""
    from kullback import domain as domain_mod
    from kullback.runner.records import write_json as write

    # None of the rows rather than zeros before the domain was read: never asked and nothing
    # found are not the same reading.
    unread = package_mod.export(nursery, tmp_path / "unread", name="nursery", preview=True)
    assert unread["domain"] == {}

    store = nursery / domain_mod.DIR
    write(store / domain_mod.ARCHETYPES, {"format": domain_mod.FORMAT, "counts": {},
                                          "archetypes": [{"goal": "a", "write_tools": ["water_bed"]},
                                                         {"goal": "b", "write_tools": []}]})
    write(store / domain_mod.GAPS, {"format": domain_mod.FORMAT, "gaps": [{"goal": "b"}]})
    manifest = package_mod.export(nursery, tmp_path / "package", name="nursery", preview=True)
    assert manifest["domain"] == {"archetypes_extracted": 2, "archetypes_mapped": 1,
                                  "archetype_gaps": 1}


# --- call fidelity, table cells and the words (PR #121) --------------------------------


def test_call_fidelity_counts_same_cosmetic_and_both_refused_as_agreeing(nursery, tmp_path):
    replays = read_json(nursery / "replays.json")
    replays["task_one"]["trace-task_one"]["checks"] = [
        {"verdict": "same"}, {"verdict": "cosmetic"}, {"verdict": "both_refused"}, {"verdict": "differs"}]
    replays["task_two"]["trace-task_two"]["checks"] = [{"verdict": "ours_refused"}]
    write_json(nursery / "replays.json", replays)
    manifest = package_mod.export(nursery, tmp_path / "package", name="nursery", preview=True)
    assert "call_fidelity" not in manifest
    fidelity = manifest["replay_fidelity"]
    assert (fidelity["calls"], fidelity["calls_total"], fidelity["calls_rate"]) == (3, 5, 0.6)


def test_a_recorded_call_the_replay_never_made_lowers_call_fidelity(nursery, tmp_path):
    replays = read_json(nursery / "replays.json")
    replays["task_one"]["trace-task_one"]["checks"] = [{"verdict": "same"}]
    replays["task_one"]["trace-task_one"]["counts"] = {"calls": 1, "unmade": 1}
    write_json(nursery / "replays.json", replays)
    fidelity = package_mod.export(nursery, tmp_path / "package", name="nursery", preview=True)["replay_fidelity"]
    assert (fidelity["calls"], fidelity["calls_total"], fidelity["calls_rate"]) == (1, 2, 0.5)


def test_a_call_the_recording_never_made_leaves_call_fidelity_unchanged(nursery, tmp_path):
    replays = read_json(nursery / "replays.json")
    replays["task_one"]["trace-task_one"]["checks"] = [{"verdict": "same"}, {"verdict": "unrecorded"}]
    replays["task_one"]["trace-task_one"]["counts"] = {"calls": 2, "unmade": 0}
    write_json(nursery / "replays.json", replays)
    fidelity = package_mod.export(nursery, tmp_path / "package", name="nursery", preview=True)["replay_fidelity"]
    assert (fidelity["calls"], fidelity["calls_total"], fidelity["calls_rate"]) == (1, 1, 1.0)


def test_a_replay_record_without_counts_is_counted_from_its_checks(nursery, tmp_path):
    replays = read_json(nursery / "replays.json")
    replays["task_one"]["trace-task_one"].pop("counts", None)
    replays["task_one"]["trace-task_one"]["checks"] = [{"verdict": "same"}, {"verdict": "differs"}]
    write_json(nursery / "replays.json", replays)
    manifest = package_mod.export(nursery, tmp_path / "package", name="nursery", preview=True)
    fidelity = manifest["replay_fidelity"]
    assert (fidelity["calls"], fidelity["calls_total"], fidelity["calls_rate"]) == (1, 2, 0.5)


def test_call_fidelity_counts_the_calls_a_replay_report_writes_to_replays_json(nursery, tmp_path):
    replays = read_json(nursery / "replays.json")
    report = ReplayReport(
        task_id="task_one", trace_id="trace-task_one", run_id="run-1", path="runs/run-1.json", confirmed=True,
        fidelity=1.0, counts={"calls": 3, "unmade": 1},
        calls=[{"call_id": "c1", "kind": "read", "tool": "get_user", "verdict": "same"},
               {"call_id": "c2", "kind": "write", "tool": "cancel", "verdict": "both_refused"},
               {"call_id": "c3", "kind": "read", "tool": "get_order", "verdict": "differs"},
               {"call_id": "c4", "kind": "read", "tool": "get_order", "verdict": "unrecorded"}])
    replays["task_one"]["trace-task_one"] = report.as_dict()
    write_json(nursery / "replays.json", replays)
    fidelity = package_mod.export(nursery, tmp_path / "package", name="nursery", preview=True)["replay_fidelity"]
    assert (fidelity["calls"], fidelity["calls_total"], fidelity["calls_rate"]) == (2, 4, 0.5)


def test_a_line_break_in_corpus_text_stays_on_its_card_line_and_a_pipe_inside_its_organisation_cell(
        nursery, tmp_path):
    manifest = package_mod.export(nursery, tmp_path / "package", name="nursery", preview=True,
                                  corpus="nursery | traces\nsecond line", corpus_license="MIT")
    assert "Source corpus: nursery | traces second line (MIT)." in card_mod.card_markdown(manifest, "leibler/nursery")
    org = card_mod.organisation_card([{"name": "n", "repo_id": "leibler/a|b", "manifest": manifest}])
    row = next(line for line in org.splitlines() if line.startswith("| ["))
    assert row.replace("\\|", "").count("|") == 5


def test_the_organisation_card_defines_four_plain_terms_and_names_no_gate_atom_or_fidelity():
    body = card_mod.organisation_card([])
    terms = body.split("## Terms\n\n", 1)[1].split("\n\n", 1)[0].splitlines()
    assert [term.split(":")[0] for term in terms] == ["- Environment", "- Task", "- Run", "- Verifier"]
    headings = [line for line in body.splitlines() if line.startswith("#")]
    assert headings[1:] == ["## Environments", "## Use it", "## Terms", "## Known issues", "## Links"]
    assert card_mod.DISCLAIMER in body.split("## Known issues")[1]
    for name in ("Atom", "Gates", "suite", "loosening", "Call fidelity", "Fidelity over", "Confirmed:", "90%"):
        assert name not in body


def test_the_known_issues_say_a_task_needing_a_mid_conversation_fact_cannot_be_finished_in_the_package(
        nursery, tmp_path):
    manifest = package_mod.export(nursery, tmp_path / "package", name="nursery", preview=True)
    note = next(line for line in card_mod.card_markdown(manifest, "leibler/nursery").splitlines()
                if "simulated user does not ship" in line)
    assert note.startswith(f"- {manifest['created_at'][:10]}: ")
    assert "A Task that needs a fact the user gives mid conversation cannot be finished in the package" in note


# --- a release replaces the repository ---------------------------------------------


def test_the_hub_adapter_asks_the_host_to_drop_remote_files_the_uploaded_folder_lacks(tmp_path):
    from kullback.hub.client import HuggingFaceHub

    class Host:
        def upload_folder(self, **asked):
            self.asked = asked
            return type("Commit", (), {"oid": "abc"})()

    host = Host()
    assert HuggingFaceHub(api=host).upload_folder("leibler/nursery", tmp_path, "release") == "abc"
    assert host.asked["delete_patterns"] == "*"


def test_a_dry_run_upload_leaves_no_file_an_earlier_upload_had_and_this_one_lacks(tmp_path):
    from kullback.hub.client import LocalHub

    older, newer = tmp_path / "older", tmp_path / "newer"
    for folder, name in ((older, "gone.json"), (newer, "kept.json")):
        folder.mkdir()
        (folder / name).write_text("{}", encoding="utf-8")
    local = LocalHub(tmp_path / "hub")
    local.upload_folder("leibler/nursery", older, "first")
    local.upload_folder("leibler/nursery", newer, "second")
    target = tmp_path / "hub" / "leibler" / "nursery"
    assert (target / "kept.json").is_file()
    assert not (target / "gone.json").exists()


UNTRUSTED_REASON = "a reason only the manifest holds"


def _card(**manifest) -> str:
    return card_mod.card_markdown(dict({"name": "widgets", "tasks_total": 4, "trusted": 1,
                                        "replay_fidelity": {"tasks": 4, "tasks_total": 4, "tasks_rate": 1.0},
                                        "untrusted": {"count": 3, "reasons": [{"reason": UNTRUSTED_REASON, "tasks": 3}]}},
                                       **manifest), "someorg/widgets")


def test_the_card_reads_summary_use_it_inside_known_issues_how_it_was_built_and_licence_in_that_order():
    body = _card()
    headings = [line for line in body.splitlines() if line.startswith("#")]
    assert headings == ["# widgets", "## Use it", "## What is inside", "## Known issues", "## How it was built",
                        "## Licence"]
    assert "Release: 100.0% (4 of 4) of its Tasks replay as recorded" in body.split("## Use it")[0]
    assert body.index("Built: ") < body.index("## Licence")


def test_the_card_says_once_that_some_tasks_can_be_wrong_and_that_unchecked_tasks_are_not_graded():
    body = _card()
    assert body.count(card_mod.DISCLAIMER) == 1 and card_mod.DISCLAIMER.startswith("Some Tasks can be wrong.")
    assert "Tasks that did not pass every check are not graded." in body.split("## Known issues")[1]


def test_the_card_names_no_gate_no_trust_word_no_reasons_no_audit_numbers_and_no_news():
    body = _card(leak_scan={"leaks": 0}, refused=0, reference_confirmed=4, verifier_derived=4).lower()
    for gone in ("trust", "suite", "loosening", "false rejection", "oracle replay", UNTRUSTED_REASON,
                 "| ", "leak scan", "refused", "reference confirmed", "## news"):
        assert gone not in body


def test_what_is_inside_counts_the_tools_and_quotes_one_tasks_first_ask(tmp_path):
    (tmp_path / "harbor" / "widgets-one").mkdir(parents=True)
    (tmp_path / "harbor" / "widgets-one" / "instruction.md").write_text(
        "A user writes in.\n- wants the widget relabelled\n- user ID is u1\n\n## Policy\n\n- rule\n")
    write_json(tmp_path / "tool_sigs.json", [{"name": "a"}, {"name": "b"}])
    inside = package_mod.inside(tmp_path, {"tasks": [{"path": "widgets-one"}]})
    assert inside == {"tools": 2, "example_ask": "wants the widget relabelled"}
    body = _card(inside=inside)
    assert "4 Tasks over 2 tools" in body and 'opens, for example: "wants the widget relabelled".' in body


def test_the_built_line_names_the_commit_that_laid_the_harbor_tasks_where_it_differs_from_the_package():
    package_sha, laid_sha = "a" * 40, "b" * 40
    built = card_mod.card_markdown({"git_sha": package_sha, "harbor": {"git_sha": laid_sha}}, "someorg/widgets")
    assert f"package built at git {'a' * 16}, Harbor Tasks laid at git {'b' * 16}" in built
    same = card_mod.card_markdown({"git_sha": package_sha, "harbor": {"git_sha": package_sha}}, "someorg/widgets")
    assert f"git {'a' * 16}," in same and "laid at" not in same


def _scripted_candidate(monkeypatch):
    from kullback.ai.provider import TestModel
    from tests.episode.invented import SCRIPTED_RENAME

    monkeypatch.setattr(cli, "_live_model", lambda model_id, base_url=None: TestModel(SCRIPTED_RENAME, loop=True))


def test_a_package_fetched_from_a_staged_publish_scores_a_candidate_run_and_counts_it_in_the_report(
        tmp_path, hub, monkeypatch):
    # D407: the build wrote no environment.json; the package derives one, so the regrade gate lets the Verdict through
    from tests.episode.invented import write_env

    workdir = write_env(tmp_path / "work")
    (workdir / "environment.json").unlink()
    (workdir / "db.json").rename(workdir / "env" / "db.json")  # where a build keeps its world, and what a package ships
    publish_mod.publish(workdir, "leibler/widgets", client=hub, preview=True)
    out = tmp_path / "fetched"
    manifest = publish_mod.fetch("leibler/widgets", out, client=hub)
    assert "environment.json" in manifest["files"] and manifest["missing_run_inputs"] == []
    _scripted_candidate(monkeypatch)
    runner = CliRunner()
    ran = runner.invoke(cli.app, ["run", "--workdir", str(out), "--task", "widget_task", "--model", "test/candidate"])
    assert ran.exit_code == 0, ran.output
    scored = runner.invoke(cli.app, ["verdict", "--workdir", str(out), "--task", "widget_task"])
    assert scored.exit_code == 0 and "refused" not in scored.output, scored.output
    stored = [read_json(path) for path in (out / "verdicts" / "widget_task").glob("widget_task-0.*.json")]
    assert [(v["pass"], v["env_id"]) for v in stored] == [(True, manifest["env_id"])]
    reported = runner.invoke(cli.app, ["report", "--workdir", str(out)])
    assert reported.exit_code == 0, reported.output
    assert "- Candidate Runs scored: 1 of 1 Tasks" in (out / "report.md").read_text(encoding="utf-8")


def test_a_verdict_the_regrade_gate_refuses_leaves_no_file_behind(tmp_path, monkeypatch):
    # D407: no environment.json and no policy to derive one from, so the D97 versions are missing
    from tests.episode.invented import write_env

    workdir = write_env(tmp_path / "work")
    _scripted_candidate(monkeypatch)
    runner = CliRunner()
    ran = runner.invoke(cli.app, ["run", "--workdir", str(workdir), "--task", "widget_task", "--model", "test/candidate"])
    assert ran.exit_code == 0, ran.output
    (workdir / "environment.json").unlink()
    (workdir / "tool_sigs.json").unlink()
    scored = runner.invoke(cli.app, ["verdict", "--workdir", str(workdir), "--task", "widget_task"])
    assert scored.exit_code != 0 and "schema_version is not on the Verdict" in scored.output, scored.output
    assert [path for path in (workdir / "verdicts").rglob("*") if path.is_file()] == []
