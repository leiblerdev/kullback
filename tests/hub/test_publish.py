"""Exporting an Environment, publishing it and fetching it back (D221)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from nursery_domain import ELICITED_VALUE, RECORDED_TURN
from typer.testing import CliRunner

from kullback import cli
from kullback.hub import card as card_mod
from kullback.hub import package as package_mod
from kullback.hub import publish as publish_mod
from kullback.runner.records import read_json, write_json


def _export(workdir: Path, out: Path, **kwargs):
    return package_mod.export(workdir, out, name="nursery", corpus="nursery traces",
                              corpus_license="MIT", **kwargs)


# --- the package ---------------------------------------------------------------------


def test_the_export_writes_the_world_the_tasks_and_the_verifiers_and_nothing_of_the_recordings(nursery, tmp_path):
    out = tmp_path / "package"
    _export(nursery, out)
    assert (out / "env" / "db.json").is_file()
    assert (out / "env" / "policy.md").is_file()
    assert sorted(p.name for p in (out / "tasks").glob("*.json")) == \
        ["task_one.json", "task_three.json", "task_two.json"]
    assert sorted(p.name for p in (out / "verifiers").glob("*.json")) == ["task_one.json", "task_two.json"]
    written = {path.relative_to(out).as_posix().split("/", 1)[0] for path in out.rglob("*") if path.is_file()}
    assert not written & set(package_mod.NEVER_EXPORTED)


def test_an_exported_task_names_none_of_the_recordings_it_was_clustered_from(nursery, tmp_path):
    out = tmp_path / "package"
    _export(nursery, out)
    task = read_json(out / "tasks" / "task_one.json")
    assert task["run_ids"] == []
    assert task["intent"] == "water the plot that is due"


def test_the_manifest_carries_the_numbers_the_artifacts_hold(nursery, tmp_path):
    manifest = _export(nursery, tmp_path / "package")
    assert manifest["round"] == 2
    assert manifest["tasks_total"] == 3
    assert manifest["replay_fidelity"] == {"tasks": 2, "tasks_total": 3, "tasks_rate": pytest.approx(0.6667, abs=1e-3),
                                           "runs": 2, "runs_total": 3, "runs_rate": pytest.approx(0.6667, abs=1e-3)}
    assert manifest["reference_confirmed"] == 2
    assert manifest["verifier_derived"] == 2
    assert manifest["trusted"] == 1
    assert manifest["runner_version"] == "runner-hash-1"
    assert manifest["gates_version"] == "gates-hash-1"
    assert manifest["env_id"] == "env-hash-1"
    assert manifest["source"] == {"corpus": "nursery traces", "license": "MIT", "url": None}
    assert manifest["content_hash"] and manifest["files"]


def test_every_task_carries_its_funnel_stage_and_why_it_stopped(nursery, tmp_path):
    out = tmp_path / "package"
    _export(nursery, out)
    rows = {row["task_id"]: row for row in read_json(out / package_mod.TASKS_INDEX_NAME)["tasks"]}
    assert rows["task_one"]["stage"] == "trusted"
    assert rows["task_one"]["stopped_because"] is None
    assert rows["task_two"]["stage"] == "verifier_derived"
    assert "mutation_flips" in rows["task_two"]["stopped_because"]
    assert "second_path_passes" in rows["task_two"]["stopped_because"]
    assert rows["task_three"]["stage"] == "clustered"
    assert rows["task_three"]["stopped_because"] == "no replay of the Task was confirmed"


def test_the_untrusted_reasons_are_the_harness_own_words_and_quote_no_record(nursery, tmp_path):
    manifest = _export(nursery, tmp_path / "package")
    assert manifest["untrusted"]["count"] == 2
    reasons = {entry["reason"] for entry in manifest["untrusted"]["reasons"]}
    assert "no replay of the Task was confirmed" in reasons
    assert all(entry["tasks"] == 1 for entry in manifest["untrusted"]["reasons"])


def test_a_workdir_with_no_environment_is_refused(tmp_path):
    with pytest.raises(package_mod.ExportError, match="no env/"):
        package_mod.export(tmp_path / "empty", tmp_path / "package")


# --- the leak scan -------------------------------------------------------------------


def test_the_leak_scan_passes_a_package_whose_values_the_world_files_hold(nursery, tmp_path):
    manifest = _export(nursery, tmp_path / "package")
    assert manifest["leak_scan"]["leaks"] == 0
    assert manifest["leak_scan"]["corpus_strings"] > 0
    assert manifest["leak_scan"]["files_scanned"] >= 4


def test_the_leak_scan_fails_an_export_that_copies_a_recorded_turn_into_a_grader(nursery, tmp_path):
    verifier = read_json(nursery / "verifiers" / "task_one.json")
    verifier["atoms"][1]["description"] = RECORDED_TURN
    write_json(nursery / "verifiers" / "task_one.json", verifier)
    out = tmp_path / "package"
    with pytest.raises(package_mod.ExportError, match="leak scan found 1"):
        _export(nursery, out)
    assert not out.exists()


def test_a_value_the_caller_gave_mid_conversation_is_counted_as_an_echo_and_does_not_stop_the_export(
        nursery, tmp_path):
    verifier = read_json(nursery / "verifiers" / "task_one.json")
    verifier["atoms"][0]["target"]["raw"] = ELICITED_VALUE
    write_json(nursery / "verifiers" / "task_one.json", verifier)
    manifest = _export(nursery, tmp_path / "package")
    assert manifest["leak_scan"]["leaks"] == 0
    assert manifest["leak_scan"]["value_echoes"] == 1


def test_a_value_the_task_instruction_states_is_not_an_echo_in_the_verifier(nursery, tmp_path):
    verifier = read_json(nursery / "verifiers" / "task_one.json")
    verifier["atoms"][0]["target"]["raw"] = ELICITED_VALUE
    write_json(nursery / "verifiers" / "task_one.json", verifier)
    task = read_json(nursery / "tasks" / "task_one.json")
    task["intent"] = f"water the plot and send the crew to {ELICITED_VALUE}"
    write_json(nursery / "tasks" / "task_one.json", task)
    manifest = _export(nursery, tmp_path / "package")
    assert manifest["leak_scan"]["value_echoes"] == 0


def test_a_word_of_the_harness_own_vocabulary_is_not_a_leak_however_often_the_corpus_says_it(
        nursery, tmp_path):
    manifest = _export(nursery, tmp_path / "package")
    kinds = {atom["kind"] for atom in read_json(nursery / "verifiers" / "task_one.json")["atoms"]}
    assert "question" in kinds  # a word the corpus and the harness both use
    assert manifest["leak_scan"]["leaks"] == 0


def test_a_value_the_world_files_hold_is_not_a_leak_however_often_the_recordings_say_it(nursery, tmp_path):
    verifier = read_json(nursery / "verifiers" / "task_one.json")
    verifier["atoms"][1]["target"]["value"] = "2026-03-01"  # a value db.json holds
    write_json(nursery / "verifiers" / "task_one.json", verifier)
    manifest = _export(nursery, tmp_path / "package")
    assert manifest["leak_scan"]["leaks"] == 0


# --- the release form ----------------------------------------------------------------


def test_a_release_is_refused_below_the_fidelity_bar_and_says_the_number(nursery, tmp_path):
    with pytest.raises(publish_mod.PublishError, match="66.7%"):
        publish_mod.stage(nursery, tmp_path / "package", "leibler/nursery")


def test_preview_passes_below_the_bar_and_the_card_opens_with_the_banner(nursery, tmp_path):
    out = tmp_path / "package"
    manifest = publish_mod.stage(nursery, out, "leibler/nursery", preview=True,
                                 corpus="nursery traces", corpus_license="MIT")
    assert manifest["preview"] is True
    assert manifest["status"] == "preview"
    body = (out / package_mod.CARD_NAME).read_text(encoding="utf-8")
    assert "> **Preview.**" in body
    assert "66.7%" in body


def test_a_release_passes_at_the_bar(nursery, tmp_path):
    write_json(nursery / "replays.json", {
        "task_one": {"trace-task_one": {"confirmed": True}},
        "task_two": {"trace-task_two": {"confirmed": True}},
        "task_three": {"trace-task_three": {"confirmed": True}},
    })
    manifest = publish_mod.stage(nursery, tmp_path / "package", "leibler/nursery")
    assert manifest["status"] == "release"
    assert manifest["preview"] is False


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


def test_a_card_states_the_corpus_licence_the_numbers_and_the_untrusted_count(nursery, tmp_path):
    out = tmp_path / "package"
    manifest = publish_mod.stage(nursery, out, "leibler/nursery", preview=True,
                                 corpus="nursery traces", corpus_license="MIT")
    body = (out / package_mod.CARD_NAME).read_text(encoding="utf-8")
    assert "nursery traces" in body
    assert "MIT" in body
    assert str(manifest["content_hash"])[:16] in body
    assert "2 of 3 Tasks are not trusted" in body
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


def test_a_fetched_directory_holds_no_builder_state(nursery, tmp_path, hub):
    publish_mod.publish(nursery, "leibler/nursery", client=hub, preview=True)
    out = tmp_path / "fetched"
    publish_mod.fetch("leibler/nursery", out, client=hub)
    for name in ("raw", "runs", "task_status.json", "rounds.json", "replays.json"):
        assert not (out / name).exists()


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


def test_the_organisation_card_lists_every_environment_with_its_numbers_and_status(nursery, tmp_path):
    manifest = package_mod.export(nursery, tmp_path / "package", name="nursery", preview=True)
    body = card_mod.organisation_card([{"name": "nursery", "repo_id": "leibler/nursery", "manifest": manifest}])
    assert "| [nursery](https://huggingface.co/datasets/leibler/nursery) | 66.7% | 1 of 3 | 2 | preview |" in body
    assert "kullback fetch leibler/<environment>" in body


def test_an_organisation_card_the_host_will_not_take_is_reported_rather_than_claimed(hub):
    took, note = publish_mod.publish_organisation_card("leibler", "# leibler\n", client=hub)
    assert took is False
    assert "organisation profile" in note


def test_an_organisation_card_the_host_takes_is_published(hub_with_organisation_card):
    host = hub_with_organisation_card
    took, where = publish_mod.publish_organisation_card("leibler", "# leibler\n", client=host)
    assert took is True and "leibler" in where
    assert host.organisation_cards["leibler"] == "# leibler\n"


def test_the_card_template_documents_every_manifest_field_a_card_shows():
    body = card_mod.environment_card_template()
    for field in ("replay_fidelity.tasks_rate", "trusted", "leak_scan", "content_hash", "preview"):
        assert field in body


# --- the three commands ---------------------------------------------------------------


def test_the_export_command_writes_a_package_and_prints_its_numbers(nursery, tmp_path):
    out = tmp_path / "package"
    result = CliRunner().invoke(cli.app, ["export", "--workdir", str(nursery), "--out", str(out),
                                          "--name", "nursery", "--corpus-license", "MIT"])
    assert result.exit_code == 0, result.output
    assert "replay fidelity 66.7% over Tasks" in result.output
    assert "leak scan: 0 recorded strings" in result.output
    assert (out / package_mod.MANIFEST_NAME).is_file()


def test_the_publish_command_refuses_a_release_below_the_bar(nursery, tmp_path, hub, monkeypatch):
    monkeypatch.setattr("kullback.hub.client.HuggingFaceHub", lambda *a, **k: hub)
    result = CliRunner().invoke(cli.app, ["publish", "--workdir", str(nursery), "--repo", "leibler/nursery"])
    assert result.exit_code != 0
    assert hub.repos == {}


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
