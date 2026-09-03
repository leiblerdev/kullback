"""The kullback screen: what it shows, and what it refuses to do without being told.

Nothing here calls a model. The Screen takes its runner by injection for exactly that reason:
the commands are tested against a stand-in that records what it was asked for, so the dispatch is
covered without a build and without a key.
"""

from __future__ import annotations

import io
import json

import pytest
from rich.console import Console

from kullback.tui import Board, Screen, _keys, banner


def _console():
    return Console(file=__import__("io").StringIO(), width=100, force_terminal=False, no_color=True)


def _text(console):
    return console.file.getvalue()


def _show(renderable, console=None):
    console = console or _console()
    console.print(renderable)
    return _text(console)


# --- the banner -------------------------------------------------------------

def test_the_banner_spells_kullback_in_five_rows():
    rows = banner().plain.rstrip("\n").split("\n")
    assert len(rows) == 5
    assert all(row.strip("█ ") == "" for row in rows)
    # eight letters, five columns of cell each: the word is there in width, not just in name.
    assert len(rows[0]) == len("kullback") * 5


def test_the_banner_only_knows_the_letters_it_has():
    with pytest.raises(KeyError):
        banner("kullbackx")


# --- the board --------------------------------------------------------------

def test_a_stage_that_starts_and_ends_keeps_its_order_and_its_last_word(tmp_path):
    board = Board(tmp_path)
    board.event({"kind": "stage", "stage": "mine", "state": "start", "attempt": 1})
    board.event({"kind": "stage", "stage": "mine", "state": "ran", "attempt": 1})
    board.event({"kind": "stage", "stage": "cluster", "state": "start", "attempt": 1})
    assert board.order == ["mine", "cluster"]
    assert board.status == {"mine": "ran", "cluster": "start"}
    assert board.seconds["mine"] >= 0


def test_the_board_reads_the_typed_stage_events_of_the_agent_core_too(tmp_path):
    """A subscriber on the Builder's harness can feed the board the StageStart and StageEnd events
    the pipeline emits (phase 4); anything else on the stream is not for the board."""
    from kullback.agent.events import AgentStart, StageEnd, StageStart

    board = Board(tmp_path)
    board.event(StageStart(name="mine"))
    board.event(AgentStart())
    board.event(StageEnd(name="mine", counts={"status": "cached", "attempts": 1, "rulings": ["mine"]}))
    board.event(StageStart(name="cluster"))
    board.event(StageEnd(name="cluster", counts={"status": "failed", "attempts": 3}))
    assert board.order == ["mine", "cluster"]
    assert board.status == {"mine": "cached", "cluster": "failed"} and board.attempts["cluster"] == 3
    assert board.seconds["mine"] >= 0


def test_the_screen_builds_through_the_round_driver_the_cli_uses(tmp_path, monkeypatch):
    """/build goes through rounds.run_rounds, the same entry the CLI uses, with the dict events wired.

    The one claim the `runner=` seam every other Screen test uses cannot make is which callable
    the Screen reaches for when nothing is injected, so this test alone replaces that default.
    The real run_rounds would drive whole rounds off an empty workdir.
    """
    from kullback import rounds

    seen = {}

    def fake(**kwargs):
        seen.update(kwargs)
        kwargs["on_event"]({"kind": "round", "state": "end", "round": 1, "counts": {}, "exit": "done"})
        kwargs["on_event"]({"kind": "pipeline", "state": "complete"})
        return {}

    monkeypatch.setattr(rounds, "run_rounds", fake)
    screen = Screen(tmp_path, console=_console())
    assert screen.command("/build --iterate") is True
    assert seen["iterate"] is True and seen["workdir"] == tmp_path and seen["model"] is None


def test_the_board_shows_the_round_and_the_beat_from_the_typed_events(tmp_path):
    """The BeatStart and BeatEnd events of the driver name the agent that holds the stream (D128);
    the board shows the round and the beat above the stages, and nothing before the first round."""
    from kullback.agent.events import BeatEnd, BeatStart, RoundEnd, RoundStart

    board = Board(tmp_path)
    assert board.beat().plain == ""
    board.event(RoundStart(round=1))
    board.event(BeatStart(agent="builder", round=1))
    assert board.beat().plain == "round 1, builder beat"
    assert "round 1, builder beat" in _show(board.render())
    board.event(BeatEnd(agent="builder", round=1))
    assert board.beat().plain == "round 1"
    board.event(BeatStart(agent="examiner", round=1))
    assert board.beat().plain == "round 1, examiner beat"
    board.event(BeatEnd(agent="examiner", round=1))
    board.event(RoundEnd(round=1, counts={"fidelity": 1, "tasks": 2}, exit=None))
    board.event({"kind": "beat", "state": "start", "agent": "builder", "round": 2})
    assert board.beat().plain == "round 2, builder beat"
    assert board.rounds == [{"round": 1, "counts": {"fidelity": 1, "tasks": 2}, "exit": None}]


def test_the_round_table_lists_each_finished_round_with_its_counts_and_the_exit(tmp_path):
    board = Board(tmp_path)
    board.event({"kind": "round", "state": "end", "round": 1, "exit": None, "counts": {
        "fidelity": 1, "tasks": 3, "trusted": 0, "refused_count": 0, "probes_passing": 2,
        "spend": {"builder": 0.0, "examiner": 0.0, "total": 0.0}}})
    board.event({"kind": "round", "state": "end", "round": 2, "exit": "done", "counts": {
        "fidelity": 3, "tasks": 3, "trusted": 2, "refused_count": 1, "probes_passing": 5,
        "spend": {"builder": 0.25, "examiner": 1.0, "total": 1.25}}})
    rows = [line.split() for line in _show(board.rounds_table()).splitlines() if line.strip()]
    assert rows[0] == ["round", "fidelity", "trusted", "refused", "probes", "passing", "spend", "exit"]
    assert rows[1] == ["1", "1/3", "0", "0", "2", "$0.0000"]
    assert rows[2] == ["2", "3/3", "2", "1", "5", "$1.2500", "done"]
    assert "3/3" in _show(board.render())


def test_status_prints_the_last_rounds_counts_beside_the_pipeline_state(tmp_path):
    (tmp_path / "pipeline").mkdir()
    (tmp_path / "pipeline" / "state.json").write_text(json.dumps({
        "status": "complete", "statuses": {"mine": "ran"}, "attempts": {"mine": 1}, "gates": []}))
    (tmp_path / "rounds.json").write_text(json.dumps([
        {"round": 1, "exit": None, "counts": {"fidelity": 1, "tasks": 2, "trusted": 0, "refused_count": 0,
                                               "probes_passing": 0, "spend": {"total": 0.0}}},
        {"round": 2, "exit": "stalled", "counts": {"fidelity": 1, "tasks": 2, "trusted": 1, "refused_count": 1,
                                                    "probes_passing": 3, "spend": {"total": 0.5}}},
    ]))
    screen = _screen(tmp_path)
    screen.command("/status")
    out = _text(screen.console)
    assert "mine" in out and "complete" in out
    assert "round 2" in out and "stalled" in out and "1/2" in out and "$0.5000" in out
    assert screen.runner.calls == []


def test_a_stage_is_only_timed_once_it_has_been_seen_to_start(tmp_path):
    board = Board(tmp_path)
    board.event({"kind": "stage", "stage": "mine", "state": "cached", "attempt": 1})
    assert "mine" not in board.seconds


def test_the_gate_line_names_the_last_failure_and_its_reason(tmp_path):
    board = Board(tmp_path)
    board.event({"kind": "gate", "stage": "mine", "passed": True, "failures": []})
    board.event({"kind": "gate", "stage": "compile_tools", "passed": False,
                 "failures": ["replay fidelity 0.82 below 0.9"]})
    out = board.verdict().plain
    assert "1 gates passed" in out and "1 failed" in out
    assert "compile_tools" in out and "replay fidelity 0.82 below 0.9" in out


def test_a_build_with_no_failing_gate_says_nothing_about_failures(tmp_path):
    board = Board(tmp_path)
    board.event({"kind": "gate", "stage": "mine", "passed": True, "failures": []})
    board.event({"kind": "pipeline", "state": "complete"})
    out = board.verdict().plain
    assert "failed" not in out and "complete" in out


def test_the_retry_count_is_shown_only_once_there_has_been_a_retry(tmp_path):
    board = Board(tmp_path)
    board.event({"kind": "stage", "stage": "mine", "state": "ran", "attempt": 1})
    assert "×" not in _show(board.stages())
    board.event({"kind": "stage", "stage": "compile_tools", "state": "rolled_back", "attempt": 2})
    assert "×2" in _show(board.stages())


def test_spend_is_read_from_the_file_the_build_writes_not_from_the_screen(tmp_path):
    (tmp_path / "budget.json").write_text(json.dumps(
        {"total": {"usd": 0.4213, "calls": 37, "input": 812000, "output": 41000, "unpriced_calls": 2}}))
    board = Board(tmp_path, ceiling=5.0)
    out = board.money().plain
    assert "$0.4213" in out and "of $5.00 ceiling" in out
    assert "37 calls" in out and "812,000 in / 41,000 out" in out
    assert "2 unpriced" in out


def test_spend_before_a_single_call_is_zero_and_says_no_ceiling(tmp_path):
    out = Board(tmp_path).money().plain
    assert "$0.0000" in out and "ceiling" not in out


def test_provenance_shows_the_content_hash_each_stage_wrote(tmp_path):
    (tmp_path / "cache").mkdir()
    (tmp_path / "cache" / "mine.abc123def456.json").write_text("{}")
    board = Board(tmp_path)
    board.event({"kind": "stage", "stage": "mine", "state": "ran", "attempt": 1})
    board.event({"kind": "stage", "stage": "cluster", "state": "ran", "attempt": 1})
    out = _show(board.provenance())
    assert "abc123def456" in out
    # cluster wrote nothing reusable, so it has no provenance row to claim one.
    assert "cluster" not in out


# --- the commands -----------------------------------------------------------

class _Runner:
    """The build the Screen was given, recording what it was asked for. What the board does with
    a build's events is pinned directly on Board above, so this one emits none."""

    def __init__(self):
        self.calls = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        return {}


def _screen(tmp_path, **kwargs):
    return Screen(tmp_path, console=_console(), runner=_Runner(), **kwargs)


def test_quit_closes_the_screen_and_everything_else_keeps_it_open(tmp_path):
    screen = _screen(tmp_path)
    assert screen.command("/quit") is False
    assert screen.command("/help") is True
    assert screen.command("") is True


def test_an_unknown_command_says_so_rather_than_doing_something(tmp_path):
    screen = _screen(tmp_path)
    screen.command("/deploy")
    assert "no command matches /deploy" in _text(screen.console)
    assert screen.runner.calls == []


@pytest.mark.parametrize("line,iterate,files", [
    ("/build --iterate --file traces.json", True, ["traces.json"]),
    ("/build", False, []),
])
def test_build_passes_the_flags_it_was_typed_and_no_model_by_default(tmp_path, line, iterate, files):
    screen = _screen(tmp_path)
    screen.command(line)
    call = screen.runner.calls[0]
    assert call["iterate"] is iterate
    assert [p.name for p in call["files"]] == files
    assert call["model"] is None


def test_the_ceiling_the_screen_was_opened_with_reaches_the_build(tmp_path):
    screen = _screen(tmp_path, ceiling_usd=2.5)
    screen.command("/build")
    assert screen.runner.calls[0]["ceiling_usd"] == 2.5


@pytest.mark.parametrize("line", ["/run", "/run --count 3"])
def test_run_needs_a_task_id_and_says_so_instead_of_guessing_one(tmp_path, line):
    screen = _screen(tmp_path)
    screen.command(line)
    assert "run needs a task id" in _text(screen.console)
    assert screen.runner.calls == []


def test_run_passes_the_task_and_the_count(tmp_path):
    screen = _screen(tmp_path)
    screen.command("/run task-7 --count 3")
    call = screen.runner.calls[0]
    assert call["task_id"] == "task-7" and call["count"] == 3


def test_a_build_that_raises_is_shown_as_an_outcome_not_thrown_at_the_terminal(tmp_path):
    def angry(**kwargs):
        raise RuntimeError("no key")
    screen = Screen(tmp_path, console=_console(), runner=angry)
    screen.command("/build")
    assert "RuntimeError: no key" in _text(screen.console)


def test_a_screen_with_a_model_named_refuses_to_call_it_with_live_calls_off(tmp_path, monkeypatch):
    monkeypatch.delenv("HARNESS_ALLOW_MODEL_REQUESTS", raising=False)
    monkeypatch.chdir(tmp_path)  # so load_dotenv finds no .env that could turn them on
    screen = _screen(tmp_path, model="openai/gpt-5.6-luna")
    screen.command("/build")
    assert "live model requests are off" in _text(screen.console)
    assert screen.runner.calls == []


def test_status_reads_the_last_build_off_disk_and_runs_no_stage(tmp_path):
    """The gates are written the way pipeline.py writes them, through as_dict, whose key for
    GateResult.passed is the alias "pass"; reading "passed" there counts every gate as failed."""
    from kullback.runner.records import GateResult, as_dict

    (tmp_path / "pipeline").mkdir()
    (tmp_path / "pipeline" / "state.json").write_text(json.dumps({
        "status": "failed", "statuses": {"mine": "ran", "compile_tools": "failed"},
        "attempts": {"mine": 1, "compile_tools": 3},
        "gates": [as_dict(GateResult(stage="mine", **{"pass": True})),
                  as_dict(GateResult(stage="compile_tools", **{"pass": False},
                                     failures=["bodies did not replay"]))]}))
    screen = _screen(tmp_path)
    screen.command("/status")
    out = _text(screen.console)
    assert "mine" in out and "compile_tools" in out and "failed" in out
    assert "1 gates passed" in out and "1 failed" in out
    assert "bodies did not replay" in out
    assert screen.runner.calls == []


def test_status_before_any_build_says_there_is_none(tmp_path):
    screen = _screen(tmp_path)
    screen.command("/status")
    assert "no build yet" in _text(screen.console)


def test_keys_says_which_are_set_and_never_what_they_are():
    out = _keys({"OPENAI_API_KEY": "sk-secret-value", "HARNESS_ALLOW_MODEL_REQUESTS": "1"}).plain
    assert "sk-secret-value" not in out
    assert "OPENAI_API_KEY" in out and "set" in out
    assert "ANTHROPIC_API_KEY" in out and "missing" in out


# --- a flag typed without its value is a usage message, not an IndexError (Greptile, PR 1) ---

@pytest.mark.parametrize("line", ["/build --file", "/build --file --iterate"])
def test_a_flag_with_no_value_is_told_in_words_and_starts_nothing(tmp_path, line):
    screen = _screen(tmp_path)
    screen.command(line)
    assert "--file needs a value after it" in _text(screen.console)
    assert screen.runner.calls == []


@pytest.mark.parametrize("line,message", [
    ("/run task-7 --count lots", "--count takes a whole number, not 'lots'"),
    ("/run task-7 --count 0", "--count takes a number of runs"),
])
def test_a_count_that_is_not_a_whole_number_of_runs_is_told_in_words(tmp_path, line, message):
    screen = _screen(tmp_path)
    screen.command(line)
    assert message in _text(screen.console)
    assert screen.runner.calls == []


# --- diagrammatic views -----------------------------------------------------

def test_map_draws_the_stages_in_order_with_states_and_hashes(tmp_path):
    from kullback.tui import diagrams

    out = diagrams.dag_text(["mine", "cluster"], {"mine": "ran", "cluster": "start"},
                            {"mine": 1, "cluster": 2}, {"mine": "3cbbe2f24725"})
    plain = out.plain
    assert plain.index("mine") < plain.index("cluster")
    assert "ran" in plain and "start ×2" in plain and "3cbbe2f2" in plain
    assert plain.count("▼") == 1


def test_map_before_any_build_says_there_is_none(tmp_path):
    from kullback.tui import diagrams

    assert "no stages yet" in diagrams.dag_text([], {}).plain


def test_loop_without_rounds_says_single_pass(tmp_path):
    from kullback.tui import diagrams

    assert "single-pass" in diagrams.loop_text(diagrams.read_rounds_file(tmp_path)).plain


def test_loop_draws_beats_counts_and_exit(tmp_path):
    from kullback.tui import diagrams

    (tmp_path / "rounds.json").write_text(json.dumps([
        {"round": 1, "counts": {"fidelity": 12, "tasks": 20, "trusted": 8, "refused_count": 1,
                                "probes_passing": 5, "spend": {"total": 0.4231}}, "exit": None},
        {"round": 2, "counts": {"fidelity": 20, "tasks": 20, "trusted": 15, "refused_count": 1,
                                "probes_passing": 9, "spend": {"total": 0.9}}, "exit": "done"},
    ]), encoding="utf-8")
    plain = diagrams.loop_text(diagrams.read_rounds_file(tmp_path)).plain
    assert "round 1" in plain and "round 2" in plain
    assert "12/20" in plain and "exit: done" in plain


def test_layers_names_every_package(tmp_path):
    from kullback.tui import diagrams

    plain = diagrams.layers_text().plain
    for name in ("builder", "examiner", "gates", "agent", "runner", "ai"):
        assert name in plain


def test_map_loop_layers_commands_print(tmp_path):
    (tmp_path / "pipeline").mkdir()
    (tmp_path / "pipeline" / "state.json").write_text(
        json.dumps({"statuses": {"mine": "ran"}, "attempts": {"mine": 1}}), encoding="utf-8")
    (tmp_path / "cache").mkdir()
    (tmp_path / "cache" / "mine.3cbbe2f2.json").write_text("{}", encoding="utf-8")
    console = _console()
    screen = Screen(tmp_path, console=console)
    assert screen.command("/map") is True
    assert screen.command("/layers") is True
    assert screen.command("/loop") is True
    out = _text(console)
    assert "mine" in out and "builder" in out and "single-pass" in out


# --- /login and /logout -----------------------------------------------------

LOGIN_CATALOG = {
    "opencode-go": {"id": "opencode-go", "name": "OpenCode Go", "npm": "@ai-sdk/openai-compatible",
                    "api": "https://opencode.ai/zen/go/v1", "env": ["OPENCODE_API_KEY"],
                    "models": {"muse-spark": {"limit": {"context": 1048576, "output": 131072},
                                              "cost": {"input": 3, "output": 15}}}},
}


def _snapshot(monkeypatch, tmp_path, catalog=LOGIN_CATALOG):
    import datetime

    from kullback.ai import provider as pv

    path = tmp_path / "models.dev.json"
    path.write_text(json.dumps({"fetched_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                                "catalog": catalog}), encoding="utf-8")
    monkeypatch.setattr(pv, "REGISTRY_SNAPSHOT_PATH", str(path))
    monkeypatch.delenv("OPENCODE_API_KEY", raising=False)
    return path


def test_login_with_no_args_inspects_the_current_model(tmp_path, monkeypatch):
    _snapshot(monkeypatch, tmp_path)
    console = _console()
    screen = Screen(tmp_path, model="opencode-go/muse-spark", console=console)
    assert screen.command("/login") is True
    out = _text(console)
    assert "opencode-go/muse-spark" in out and "OPENCODE_API_KEY" in out and "missing" in out


def test_login_with_no_model_says_so(tmp_path):
    console = _console()
    assert Screen(tmp_path, console=console).command("/login") is True
    assert "no model" in _text(console)


def test_login_to_an_id_without_a_slash_is_told_in_words(tmp_path):
    console = _console()
    screen = Screen(tmp_path, console=console)
    assert screen.command("/login kthumb") is True
    assert "provider/model" in _text(console) and screen.model is None


def test_login_to_an_unknown_provider_refuses_and_keeps_the_old_model(tmp_path, monkeypatch):
    _snapshot(monkeypatch, tmp_path, catalog={})
    console = _console()
    screen = Screen(tmp_path, model="openai/gpt-4", console=console)
    assert screen.command("/login nope/nothing") is True
    assert "no host" in _text(console) and screen.model == "openai/gpt-4"


def test_login_sets_the_model_the_build_will_use(tmp_path, monkeypatch):
    _snapshot(monkeypatch, tmp_path)
    screen = Screen(tmp_path, console=_console())
    assert screen.command("/login opencode-go/muse-spark") is True
    assert screen.model == "opencode-go/muse-spark"


def test_login_set_holds_the_key_in_memory_and_logout_forgets_it(tmp_path, monkeypatch):
    import os

    _snapshot(monkeypatch, tmp_path)
    console = _console()
    screen = Screen(tmp_path, console=console)
    assert screen.command("/login opencode-go/muse-spark --set OPENCODE_API_KEY=sk-zen-123") is True
    assert os.environ.get("OPENCODE_API_KEY") == "sk-zen-123"
    out = _text(console)
    assert "sk-zen-123" not in out and "set" in out
    assert screen.command("/logout") is True
    assert "OPENCODE_API_KEY" not in os.environ


def test_logout_restores_what_the_shell_held(tmp_path, monkeypatch):
    import os

    _snapshot(monkeypatch, tmp_path)
    monkeypatch.setenv("OPENCODE_API_KEY", "old-key")
    screen = Screen(tmp_path, console=_console())
    screen.command("/login opencode-go/muse-spark --set OPENCODE_API_KEY=new-key")
    assert os.environ["OPENCODE_API_KEY"] == "new-key"
    screen.command("/logout")
    assert os.environ["OPENCODE_API_KEY"] == "old-key"


def test_keys_marks_the_keys_this_session_set(tmp_path, monkeypatch):
    _snapshot(monkeypatch, tmp_path)
    console = _console()
    screen = Screen(tmp_path, console=console)
    screen.command("/login opencode-go/muse-spark --set OPENCODE_API_KEY=sk-zen-123")
    screen.command("/keys")
    out = _text(console)
    assert "OPENCODE_API_KEY" in out and "this session" in out and "sk-zen-123" not in out
    screen.command("/logout")


def test_a_login_secret_reaches_no_file_and_no_line(tmp_path, monkeypatch):
    _snapshot(monkeypatch, tmp_path)
    console = _console()
    screen = Screen(tmp_path, console=console)
    screen.command("/login opencode-go/muse-spark --set OPENCODE_API_KEY=topsecret-value-9")
    screen.command("/keys")
    screen.command("/status")
    screen.command("/map")
    screen.command("/loop")
    screen.command("/layers")
    assert "topsecret-value-9" not in _text(console)
    for path in tmp_path.rglob("*"):
        if path.is_file():
            assert "topsecret-value-9" not in path.read_text(encoding="utf-8", errors="replace")
    screen.command("/logout")


def test_loop_shows_pending_findings_and_exit_notes(tmp_path):
    from kullback.tui import diagrams

    (tmp_path / "rounds.json").write_text(json.dumps([
        {"round": 1, "counts": {}, "exit": None, "exit_note": "2 finding(s) owe the Builder a beat; the round continues",
         "pending_findings": [{"finding_id": "f1"}, {"finding_id": "f2"}]},
        {"round": 2, "counts": {}, "exit": "ceiling", "exit_note": "the ceiling ended the run first",
         "pending_findings": [{"finding_id": "f3"}]},
    ]), encoding="utf-8")
    plain = diagrams.loop_text(diagrams.read_rounds_file(tmp_path)).plain
    assert "2 finding(s) owed the Builder a beat" in plain
    assert "exit: ceiling" in plain and "the ceiling ended the run first" in plain


def test_the_banner_is_a_gradient_styles_vary_plain_does_not(tmp_path):
    from kullback.tui import banner

    text = banner()
    assert text.plain == banner("kullback").plain
    styles = {span.style for span in text.spans if "█" in text.plain[span.start:span.end]}
    assert len(styles) > 1, "one flat style is not a gradient"


def test_open_prints_status_segments(tmp_path, monkeypatch):
    # A wide console: open() cuts absurdly long workdirs with an ellipsis rather than folding them.
    console = Console(file=io.StringIO(), width=300, force_terminal=False, no_color=True)
    monkeypatch.setenv("HARNESS_ALLOW_MODEL_REQUESTS", "1")
    Screen(tmp_path, model="opencode-go/muse-spark", console=console).open()
    out = _text(console)
    assert "opencode-go/muse-spark" in out and "live on" in out and str(tmp_path) in out


def test_open_says_live_off_and_hides_zero_spend(tmp_path, monkeypatch):
    console = _console()
    monkeypatch.delenv("HARNESS_ALLOW_MODEL_REQUESTS", raising=False)
    Screen(tmp_path, console=console).open()
    out = _text(console)
    assert "live off" in out and "$" not in out


def test_loop_marks_a_failed_round(tmp_path):
    from kullback.tui import diagrams

    (tmp_path / "rounds.json").write_text(json.dumps([
        {"round": 1, "counts": {}, "exit": "stalled", "exit_note": "examiner failed: no derive",
         "failed": True, "pending_findings": []},
    ]), encoding="utf-8")
    plain = diagrams.loop_text(diagrams.read_rounds_file(tmp_path)).plain
    assert "exit: stalled (failed)" in plain and "examiner failed" in plain


# --- the entry screen, the / menu, sessions, the login menu, loop-aware status ---

def test_every_command_in_the_table_is_in_help(tmp_path):
    from kullback import tui as tui_mod

    HELP = tui_mod.HELP
    for name, _, _ in tui_mod.COMMANDS:
        assert f"/{name}" in HELP, name


def test_the_gradient_is_the_brand_white_into_gray():
    from kullback import tui as tui_mod

    assert tui_mod.GRADIENT[0] == (255, 255, 255)
    assert tui_mod.GRADIENT[-1] == (138, 138, 138)
    assert tui_mod.GRADIENT == sorted(tui_mod.GRADIENT, reverse=True)


def test_open_says_what_kullback_is_and_lists_commands_and_sessions(tmp_path, monkeypatch):
    monkeypatch.setenv("KULLBACK_SESSIONS_DIR", str(tmp_path / "sessions"))
    console = _console()
    Screen(tmp_path, console=console).open()
    out = _text(console)
    assert "Rebuilds your environment from traces" in out
    assert "01 commands" in out and "/status" in out
    assert "02 sessions" in out and "none yet" in out


def test_slash_alone_lists_every_command_and_waits_for_a_number(tmp_path):
    screen = _screen(tmp_path)
    assert screen.command("/") is True
    out = _text(screen.console)
    assert "pick a number" in out and "/status" in out and "/login" in out
    assert screen._pending is not None and screen._pending[0] == "commands"


def test_slash_fragment_with_one_match_runs_it(tmp_path):
    screen = _screen(tmp_path)
    assert screen.command("/layers") is True
    assert screen._pending is None


def test_slash_fragment_with_no_match_says_so(tmp_path):
    screen = _screen(tmp_path)
    assert screen.command("/zzz") is True
    assert "no command matches /zzz" in _text(screen.console)


def test_a_bare_number_picks_from_the_menu(tmp_path):
    screen = _screen(tmp_path)
    screen.command("/")
    out_before = _text(screen.console)
    assert screen.command("3") is True
    assert screen._pending is None
    assert _text(screen.console) != out_before


def test_a_bare_number_outside_the_list_says_the_range(tmp_path):
    screen = _screen(tmp_path)
    screen.command("/")
    assert screen.command("99") is True
    assert "pick 1-" in _text(screen.console)


def test_status_with_no_files_says_no_build_yet(tmp_path):
    screen = _screen(tmp_path)
    assert screen.command("/status") is True
    assert "no build yet" in _text(screen.console)


def test_status_reads_rounds_gates_and_spend_not_just_the_pipeline(tmp_path):
    (tmp_path / "rounds.json").write_text(json.dumps([
        {"round": 1, "exit": "stalled", "failed": True,
         "exit_note": "builder failed: boom",
         "counts": {"trusted_ids": ["t1"], "refused": {"t2": "r"}},
         "pending_findings": [{"task_id": "t3"}]}]), encoding="utf-8")
    (tmp_path / "gates.json").write_text(json.dumps([
        {"stage": "mine", "pass": True}, {"stage": "compile", "pass": False,
                                          "failures": ["tool x fails gate 2"]}]), encoding="utf-8")
    (tmp_path / "budget.json").write_text(json.dumps(
        {"total": {"usd": 0.42, "calls": 12, "input": 100, "output": 50}}), encoding="utf-8")
    screen = _screen(tmp_path)
    assert screen.command("/status") is True
    out = _text(screen.console)
    assert "round 1" in out and "stalled" in out and "builder failed: boom" in out
    assert "mine ✔" in out and "compile ✘" in out
    assert "1 passed, 1 failed" in out and "tool x fails gate 2" in out
    assert "trusted 1" in out and "refused 1" in out
    assert "1 finding(s) open" in out
    assert "$0.4200" in out and "12 calls" in out
    assert "no build yet" not in out


def test_status_with_spend_but_no_rounds_says_the_build_started(tmp_path):
    (tmp_path / "budget.json").write_text(json.dumps(
        {"total": {"usd": 0.01, "calls": 3, "input": 10, "output": 5}}), encoding="utf-8")
    screen = _screen(tmp_path)
    assert screen.command("/status") is True
    out = _text(screen.console)
    assert "no round closed yet" in out and "no build yet" not in out


def test_sessions_lists_a_running_build_and_watch_switches_to_it(tmp_path, monkeypatch):
    from kullback.runner import heartbeat

    monkeypatch.setenv("KULLBACK_SESSIONS_DIR", str(tmp_path / "sessions"))
    other = tmp_path / "other"
    other.mkdir()
    heartbeat.beat(other, "opencode-go/glm-5.3-flash", "running", spend_usd=0.5)
    from rich.console import Console

    wide = Console(file=io.StringIO(), width=300, force_terminal=False, no_color=True)
    screen = Screen(tmp_path, console=wide, runner=_screen(tmp_path).runner)
    assert screen.command("/sessions") is True
    out = _text(screen.console)
    assert "●" in out and str(other) in out and "glm-5.3-flash" in out
    assert screen.command("/watch 1") is True
    assert screen.workdir == other


def test_login_menu_walks_to_a_key_without_printing_it(tmp_path, monkeypatch):
    import getpass

    _snapshot(monkeypatch, tmp_path)
    monkeypatch.setenv("KULLBACK_SESSIONS_DIR", str(tmp_path / "sessions"))
    monkeypatch.setattr(getpass, "getpass", lambda prompt: "sk-menu-secret")
    console = _console()
    screen = Screen(tmp_path, console=console)
    answers = iter(["3", ""])
    screen._ask = lambda prompt: next(answers)
    screen.command("/login")
    out = _text(console)
    assert "OPENCODE_API_KEY" in out and "sk-menu-secret" not in out
    assert screen.model == "opencode-go/glm-5.3-flash"
    assert "keys held for this session: OPENCODE_API_KEY" in out
