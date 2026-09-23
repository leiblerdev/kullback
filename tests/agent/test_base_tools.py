"""The base tools: the root fence every one of them holds, the shell allowlist per pipeline
segment, the truncation, the exact-match edit, and the subset an extension is registered with."""

from __future__ import annotations

import asyncio
import json
import subprocess

import pytest

from kullback.agent.base_tools import (
    MAX_OUTPUT_BYTES,
    MAX_OUTPUT_LINES,
    Allowlist,
    base_tools,
    register_base_tools,
    truncate_head,
    truncate_tail,
)
from kullback.agent.extensions import ExtensionAPI
from kullback.agent.harness import AgentHarness
from kullback.agent.tools import ToolRegistry
from kullback.ai.provider import TestModel


def tools(root, **kwargs) -> dict:
    return {tool.name: tool for tool in base_tools(root, **kwargs)}


def run(tool, **arguments):
    return asyncio.run(tool.run(arguments))


@pytest.fixture
def root(tmp_path):
    (tmp_path / "notes.txt").write_text("alpha\nbeta\ngamma\n", encoding="utf-8")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "deep.py").write_text("def go():\n    return 1\n", encoding="utf-8")
    return tmp_path


# --- the root fence ---


@pytest.mark.parametrize(
    "name, arguments",
    [
        ("read", {"path": "PATH"}),
        ("write", {"path": "PATH", "content": "x"}),
        ("edit", {"path": "PATH", "old": "a", "new": "b"}),
        ("grep", {"pattern": "a", "path": "PATH"}),
        ("find", {"glob": "*", "path": "PATH"}),
        ("ls", {"path": "PATH"}),
        ("inspect", {"path": "PATH"}),
    ],
)
@pytest.mark.parametrize("outside", ["/etc/passwd", "../outside.txt", "sub/../../outside.txt"])
def test_every_path_tool_refuses_a_path_outside_the_root_and_names_the_rule(root, name, arguments, outside):
    tool = tools(root)[name]
    result = run(tool, **{key: (outside if value == "PATH" else value) for key, value in arguments.items()})
    assert result.is_error is True
    assert "rule root" in result.content and "root directory" in result.content


def test_a_link_that_leads_out_of_the_root_is_refused_like_a_path_that_steps_out(root, tmp_path_factory):
    outside = tmp_path_factory.mktemp("elsewhere") / "secret.txt"
    outside.write_text("secret", encoding="utf-8")
    (root / "link.txt").symlink_to(outside)
    result = run(tools(root)["read"], path="link.txt")
    assert result.is_error is True and "resolves outside" in result.content


@pytest.mark.parametrize("empty", ["", "   "])
def test_an_empty_path_lists_the_root(root, empty):
    result = run(tools(root)["ls"], path=empty)
    assert result.is_error is False
    assert "notes.txt" in result.content and "sub/" in result.content


# --- read, write, ls, find, grep ---


def test_read_answers_the_file_and_slices_it_with_offset_and_limit(root):
    read = tools(root)["read"]
    whole = run(read, path="notes.txt")
    assert whole.is_error is False and whole.details["text"] == "alpha\nbeta\ngamma\n"
    sliced = run(read, path="notes.txt", offset=2, limit=1)
    assert sliced.details["text"] == "beta"
    assert "read again with offset=3" in sliced.content


def test_read_past_the_last_line_answers_empty_with_the_line_count_and_is_no_error(root):
    result = run(tools(root)["read"], path="notes.txt", offset=70)
    assert result.is_error is False
    assert result.details["text"] == ""
    assert result.content == "\n[the file has 4 lines; nothing at offset 70]"


def test_read_of_a_missing_file_and_of_a_directory_are_both_refused(root):
    read = tools(root)["read"]
    assert "no file" in run(read, path="gone.txt").content
    assert "is a directory" in run(read, path="sub").content


def test_read_of_a_missing_file_names_the_file_beside_it_with_the_same_stem(root):
    result = run(tools(root)["read"], path="notes.json")
    assert result.is_error is True
    assert "no file 'notes.json'" in result.content and "the directory holds notes.txt" in result.content


def test_read_of_a_missing_file_whose_stem_nobody_has_keeps_the_plain_refusal(root):
    result = run(tools(root)["read"], path="sub/gone.json")
    assert result.is_error is True and "no file" in result.content
    assert "the directory holds" not in result.content


def test_output_over_the_limit_keeps_the_head_or_the_tail_and_read_says_how_to_continue(root):
    (root / "big.txt").write_text("\n".join(str(i) for i in range(MAX_OUTPUT_LINES + 500)), encoding="utf-8")
    result = run(tools(root)["read"], path="big.txt")
    assert result.details["truncation"]["truncated"] is True
    assert result.details["truncation"]["output_lines"] == MAX_OUTPUT_LINES
    assert f"showing lines 1 to {MAX_OUTPUT_LINES} of {MAX_OUTPUT_LINES + 500}" in result.content
    assert f"offset={MAX_OUTPUT_LINES + 1}" in result.content
    # truncate_head keeps the start, truncate_tail keeps the end
    text = "\n".join(str(i) for i in range(10))
    head, head_cut = truncate_head(text, max_lines=3)
    tail, tail_cut = truncate_tail(text, max_lines=3)
    assert head == "0\n1\n2" and head_cut.truncated is True and head_cut.total_lines == 10
    assert tail == "7\n8\n9" and tail_cut.truncated is True and tail_cut.output_lines == 3
    whole, untouched = truncate_head(text, max_lines=50)
    assert whole == text and untouched.truncated is False


def test_write_creates_a_file_with_its_parents_and_then_overwrites_it(root):
    write = tools(root)["write"]
    created = run(write, path="new/deeper/file.txt", content="one")
    assert created.details == {"path": "new/deeper/file.txt", "characters": 3, "created": True}
    assert (root / "new" / "deeper" / "file.txt").read_text(encoding="utf-8") == "one"
    again = run(write, path="new/deeper/file.txt", content="two")
    assert again.details["created"] is False
    assert (root / "new" / "deeper" / "file.txt").read_text(encoding="utf-8") == "two"


def test_ls_lists_a_directory_and_find_matches_a_glob_under_it(root):
    listing = run(tools(root)["ls"], path=".")
    assert {(e["name"], e["kind"]) for e in listing.details["entries"]} == {
        ("notes.txt", "file"),
        ("sub", "directory"),
    }
    found = run(tools(root)["find"], glob="**/*.py")
    assert found.details["paths"] == ["sub/deep.py"]
    assert "is not a directory" in run(tools(root)["find"], glob="*", path="notes.txt").content


def test_grep_answers_matching_lines_with_their_path_and_number_and_honours_the_glob(root):
    grep = tools(root)["grep"]
    result = run(grep, pattern="^def ")
    assert [(m["path"], m["line"]) for m in result.details["matches"]] == [("sub/deep.py", 1)]
    assert run(grep, pattern="alpha", glob="*.py").details["matches"] == []
    assert "is not a regular expression" in run(grep, pattern="[").content


# --- edit ---


def test_edit_replaces_the_one_exact_occurrence_and_says_where(root):
    result = run(tools(root)["edit"], path="notes.txt", old="beta", new="BETA")
    assert result.details == {"path": "notes.txt", "replacements": 1, "first_changed_line": 2}
    assert (root / "notes.txt").read_text(encoding="utf-8") == "alpha\nBETA\ngamma\n"


@pytest.mark.parametrize(
    "path, old, new, refusal",
    [
        ("notes.txt", "Beta", "x", "does not occur"),
        ("notes.txt", "", "x", "old must not be empty"),
        ("notes.txt", "beta", "beta", "change nothing"),
        ("gone.txt", "a", "b", "no file"),
    ],
)
def test_edit_refuses_text_that_does_not_occur_exactly_as_given_or_would_change_nothing(root, path, old, new, refusal):
    result = run(tools(root)["edit"], path=path, old=old, new=new)
    assert result.is_error is True and refusal in result.content
    assert (root / "notes.txt").read_text(encoding="utf-8") == "alpha\nbeta\ngamma\n"


def test_edit_refuses_text_that_occurs_more_than_once_unless_replace_all_is_set(root):
    (root / "twice.txt").write_text("same\nsame\n", encoding="utf-8")
    edit = tools(root)["edit"]
    refused = run(edit, path="twice.txt", old="same", new="other")
    assert refused.is_error is True and "occurs 2 times" in refused.content
    assert (root / "twice.txt").read_text(encoding="utf-8") == "same\nsame\n"
    replaced = run(edit, path="twice.txt", old="same", new="other", replace_all=True)
    assert replaced.details["replacements"] == 2
    assert (root / "twice.txt").read_text(encoding="utf-8") == "other\nother\n"


# --- the shell allowlist ---


def test_bash_runs_a_command_on_the_allowlist_in_the_root(root):
    result = run(tools(root)["bash"], command="ls")
    assert result.is_error is False
    assert sorted(result.details["output"].split()) == ["notes.txt", "sub"]
    assert result.details["exit_code"] == 0


def test_bash_refuses_a_command_off_the_allowlist_in_every_segment_of_a_pipeline_and_of_an_and_list(root):
    bash = tools(root)["bash"]
    piped = run(bash, command="cat notes.txt | python3 -")
    assert piped.is_error is True and "rule allowlist" in piped.content and "'python3'" in piped.content
    chained = run(bash, command="ls && python3 -c pass")
    assert chained.is_error is True and "'python3'" in chained.content
    # the allowed command of the refused pipeline never ran
    assert run(bash, command="ls | wc -l").is_error is False
    # a single command off the allowlist is refused with the rule named
    result = run(tools(root)["bash"], command="python3 -c pass")
    assert result.is_error is True and "rule allowlist" in result.content
    assert "'python3' is not one of the commands this agent may run" in result.content


@pytest.mark.parametrize(
    "command",
    ["rm notes.txt", "mv a b", "git status", "curl example", "wget example", "pip install x",
     "uv run x", "sudo ls", "chmod 777 notes.txt", "cat /etc/passwd", "cat ../outside.txt"],
)
def test_bash_refuses_a_segment_matching_a_refused_pattern_before_running_it(root, command):
    allowlist = Allowlist.of(("ls", "cat", "rm", "mv", "git", "curl", "wget", "pip", "uv", "sudo", "chmod"))
    result = run(tools(root, allowlist=allowlist)["bash"], command=command)
    assert result.is_error is True and "rule refused_pattern" in result.content
    assert (root / "notes.txt").exists()


def test_bash_refuses_an_empty_command_and_one_that_does_not_parse(root):
    bash = tools(root)["bash"]
    assert "the command is empty" in run(bash, command="   ").content
    assert "does not parse" in run(bash, command="cat 'unclosed").content


def test_bash_answers_the_last_lines_of_a_large_output_and_says_what_it_left_out(root):
    (root / "big.txt").write_text("\n".join(str(i) for i in range(MAX_OUTPUT_LINES + 500)), encoding="utf-8")
    result = run(tools(root)["bash"], command="cat big.txt")
    assert result.details["truncation"]["truncated"] is True
    assert result.details["truncation"]["output_lines"] == MAX_OUTPUT_LINES
    assert result.details["output"].splitlines()[-1] == str(MAX_OUTPUT_LINES + 499)
    assert "the last lines only" in result.content


def test_bash_kills_a_command_that_runs_past_its_timeout_and_says_so(root):
    allowlist = Allowlist.of(("sleep",))
    result = run(tools(root, allowlist=allowlist)["bash"], command="sleep 30", timeout=0.2)
    assert result.details["timed_out"] is True
    assert "killed after 0.2 seconds" in result.details["output"]


def test_bash_reports_a_failing_command_as_its_exit_code_not_as_a_tool_error(root):
    result = run(tools(root)["bash"], command="cat gone.txt")
    assert result.is_error is False and result.details["exit_code"] != 0
    assert "[exit code" in result.content


@pytest.mark.parametrize(
    "command, primitive",
    [
        ("awk 'BEGIN{system(\"id\")}'", "awk may not use system("),
        ("awk 'BEGIN{\"id\" | getline x; print x}'", "awk may not use getline"),
        ("find . -exec cat {} +", "find may not use -exec"),
        ("find . -name x -delete", "find may not use -delete"),
        ("sed -i s/alpha/omega/ notes.txt", "sed may not use -i"),
        ("sed -e 's/alpha/id/e' notes.txt", "sed may not use the e or w flag"),
        ("sed '1e id' notes.txt", "sed may not use the e command"),
        ("sed 'w copy.txt' notes.txt", "sed may not use the w command"),
        ("sort -o copy.txt notes.txt", "sort may not use -o"),
    ],
)
def test_bash_refuses_an_allowed_program_the_argument_that_would_start_or_write_through_another(
        root, command, primitive):
    result = run(tools(root)["bash"], command=command)
    assert result.is_error is True and "rule spawning" in result.content and primitive in result.content
    assert (root / "notes.txt").read_text(encoding="utf-8") == "alpha\nbeta\ngamma\n"
    assert not (root / "copy.txt").exists()


def test_bash_still_runs_the_harmless_forms_of_the_programs_that_can_spawn(root):
    bash = tools(root)["bash"]
    assert run(bash, command="sed -n 2p notes.txt").details["output"] == "beta\n"
    assert run(bash, command="awk '{print $1}' notes.txt").details["output"].split() == ["alpha", "beta", "gamma"]
    assert run(bash, command="sort -r notes.txt").details["output"].split() == ["gamma", "beta", "alpha"]


@pytest.mark.parametrize("command", ["cat notes.txt; cat notes.txt", "cat notes.txt && ls", "ls || ls",
                                     "echo x > out.txt", "cat < notes.txt", "ls &"])
def test_bash_refuses_every_operator_but_the_pipe(root, command):
    result = run(tools(root)["bash"], command=command)
    assert result.is_error is True and "rule shell: only pipes join commands here" in result.content
    assert not (root / "out.txt").exists()


def test_a_pipeline_without_a_shell_gives_the_output_the_shell_gave(root):
    for name in ("a", "b", "c"):
        (root / "sub" / name).mkdir()
        (root / "sub" / name / "x").write_text("", encoding="utf-8")
    command = "find . -name x | head -2"
    shell = subprocess.run(command, shell=True, cwd=root, capture_output=True, text=True, check=True).stdout
    result = run(tools(root)["bash"], command=command)
    assert result.details["output"] == shell
    assert len(result.details["output"].split()) == 2
    assert result.details["exit_code"] == 0


def test_a_stage_that_exits_non_zero_reports_its_exit_code(root):
    result = run(tools(root)["bash"], command="cat gone.txt | wc -l")
    assert result.details["exit_code"] == 1
    assert "gone.txt" in result.details["output"]


def test_a_stage_sees_only_path_and_lang_in_its_environment(root, monkeypatch):
    monkeypatch.setenv("KULLBACK_SECRET_FOR_TEST", "leaked")
    allowlist = Allowlist.of(("env",))
    result = run(tools(root, allowlist=allowlist)["bash"], command="env")
    names = {line.split("=", 1)[0] for line in result.details["output"].splitlines()}
    assert "KULLBACK_SECRET_FOR_TEST" not in names and "PATH" in names


# --- truncation on its own ---


def test_truncation_also_stops_at_the_byte_limit():
    text = "\n".join("x" * 100 for _ in range(10))
    output, cut = truncate_head(text, max_lines=50, max_bytes=250)
    assert cut.truncated is True and cut.truncated_by == "bytes" and len(output.encode()) <= 250


def test_a_line_longer_than_the_byte_limit_is_shown_cut_not_dropped():
    text = "a" * 300 + "\nnext"
    head, head_cut = truncate_head(text, max_lines=50, max_bytes=100)
    assert head == "a" * 100 and head_cut.output_lines == 1 and head_cut.truncated_by == "bytes"
    assert head_cut.line_cut is True
    tail, tail_cut = truncate_tail("first\n" + "b" * 300, max_lines=50, max_bytes=100)
    assert tail == "b" * 100 and tail_cut.line_cut is True


def test_read_of_a_line_over_the_limit_shows_its_head_and_names_the_cut(root):
    (root / "run.jsonl").write_text("\n".join(['{"row": 1}', "z" * (MAX_OUTPUT_BYTES + 500), '{"row": 3}']),
                                    encoding="utf-8")
    result = run(tools(root)["read"], path="run.jsonl", offset=2, limit=2)
    assert result.content.startswith("z" * 1000)
    assert "line 2 is longer than" in result.content and "was cut" in result.content
    assert "offset=3" in result.content
    assert "showing lines 2 to 1" not in result.content


# --- web_search ---


def test_web_search_says_it_is_not_configured_when_the_provider_layer_has_no_search(root):
    result = run(tools(root)["web_search"], query="anything")
    assert result.is_error is True and "not configured" in result.content


# --- inspect ---


@pytest.fixture
def world(root):
    rows = {f"r{n}": {"id": f"r{n}", "size": n, "tags": ["a"] if n % 2 else None} for n in range(12)}
    rows["r0"]["note"] = "x" * 400
    (root / "db.json").write_text(json.dumps({"things": rows, "meta": {"version": 3}, "names": ["p", "q"]}),
                                  encoding="utf-8")
    return root


def test_inspect_lists_each_top_level_key_with_its_type_and_size(world):
    result = run(tools(world)["inspect"], path="db.json")
    assert result.is_error is False
    assert "document: object with 3 keys" in result.content
    assert "things: object, table of 12 rows" in result.content
    assert "meta: object, 1 keys" in result.content
    assert "names: list, 2 items" in result.content


def test_inspect_of_a_table_names_its_columns_their_types_counts_and_compacted_samples(world):
    result = run(tools(world)["inspect"], path="db.json", key="things", rows=3)
    text = result.content
    assert "things: table: 12 rows" in text
    assert "id: string (12 of 12 rows)" in text
    assert "size: integer (12 of 12 rows)" in text
    assert "tags: null|list (12 of 12 rows)" in text or "tags: list|null (12 of 12 rows)" in text
    assert "note: string (1 of 12 rows)" in text
    assert "sample rows (3):" in text and "  r2: " in text and "  r3: " not in text
    sample = next(line for line in text.split("\n") if line.startswith("  r0: "))
    assert sample.endswith("...") and len(sample) < 220


def test_inspect_reads_an_object_of_differently_shaped_objects_as_keys_not_as_a_table(root):
    parts = {"left": {f"a{n}": {"v": n} for n in range(3)}, "right": {"b": {"w": 1}}, "top": {"c": 1, "d": 2}}
    (root / "mixed.json").write_text(json.dumps(parts), encoding="utf-8")
    text = run(tools(root)["inspect"], path="mixed.json").content
    assert "document: object with 3 keys" in text
    assert "left: object, table of 3 rows" in text and "top: object, 2 keys" in text


def test_inspect_follows_a_dotted_key_and_names_what_is_there_when_it_is_missing(world):
    assert "meta.version: integer: 3" in run(tools(world)["inspect"], path="db.json", key="meta.version").content
    missing = run(tools(world)["inspect"], path="db.json", key="meta.nope")
    assert missing.is_error is True and "no 'nope' under meta" in missing.content and "version" in missing.content


def test_inspect_of_jsonl_counts_lines_key_types_result_types_and_error_classes(root):
    records = [{"args": {"n": 1}, "result": {"ok": True, "id": "a"}},
               {"args": {"n": 2}, "result": {"ok": True, "id": "b"}},
               {"args": {"n": 3}, "result": [1, 2]},
               {"args": {"n": 4}, "error": "ValueError: bad n", "result": None},
               {"args": {"n": 5}, "error": {"type": "Missing", "message": "gone"}}]
    (root / "calls.jsonl").write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")
    text = run(tools(root)["inspect"], path="calls.jsonl", rows=1).content
    assert "jsonl: 5 lines" in text
    assert "args: object (5 of 5 rows)" in text
    assert "error: string|object (2 of 5 rows)" in text
    assert "object{id,ok}: 2" in text and "list[integer]: 1" in text and "null: 1" in text
    assert "ValueError: 1" in text and "Missing: 1" in text
    assert "sample lines (1):" in text and '"n":1' in text and '"n":2' not in text


def test_inspect_caps_its_output_and_says_how_much_more_there_is(root):
    wide = {f"column_{n}_" + "w" * 80: n for n in range(400)}
    (root / "wide.json").write_text(json.dumps(wide), encoding="utf-8")
    result = run(tools(root)["inspect"], path="wide.json")
    assert len(result.content) <= 4_100
    assert "more lines; pass key to look inside one value]" in result.content


def test_inspect_names_how_many_keys_it_left_out_of_a_wide_object(root):
    (root / "wide.json").write_text(json.dumps({f"c{n}": n for n in range(100)}), encoding="utf-8")
    text = run(tools(root)["inspect"], path="wide.json").content
    assert "  c59: integer" in text and "  c60: integer" not in text
    assert "[and 40 more keys]" in text


def test_inspect_refuses_more_than_five_sample_rows(world):
    result = run(tools(world)["inspect"], path="db.json", rows=6)
    assert result.is_error is True and "rows must be 0 to 5" in result.content


# --- registration ---


def test_an_extension_is_registered_with_the_subset_it_is_given(root):
    harness = AgentHarness(TestModel(["done"]))
    registered = register_base_tools(ExtensionAPI(harness), root, only=["read", "grep", "find", "ls"])
    assert [tool.name for tool in registered] == ["read", "grep", "find", "ls"]
    assert [schema["name"] for schema in harness.registry.schemas()] == ["read", "grep", "find", "ls"]
    # the tools an extension is not given are the boundary it cannot argue with
    assert "bash" not in harness.registry and "write" not in harness.registry


def test_a_subset_naming_a_tool_that_does_not_exist_is_a_mistake_at_registration(root):
    with pytest.raises(ValueError) as complaint:
        base_tools(root, only=["read", "compile"])
    assert "no base tool named compile" in str(complaint.value)


def test_two_extensions_get_tools_bound_to_their_own_root(tmp_path):
    first, second = tmp_path / "one", tmp_path / "two"
    first.mkdir()
    second.mkdir()
    (first / "mine.txt").write_text("mine", encoding="utf-8")
    registry = ToolRegistry(base_tools(second, only=["read"]))
    result = asyncio.run(registry.get("read").run({"path": "mine.txt"}))
    assert result.is_error is True and "no file" in result.content
    assert run(tools(first)["read"], path="mine.txt").details["text"] == "mine"
