"""Tests for the task registry: unpacking, both registry kinds, strength labels, matching.

Every id and every byte here is invented; the tars are built in the test from the
same five-piece layout the module reads, so no fixture file is ever loaded.
"""

from __future__ import annotations

import base64
import gzip
import io
import json
import tarfile

import pytest

from kullback.builder.registry import (
    DirectoryRegistry,
    TableRegistry,
    instruction_matches,
    read_task_dir,
    rows_from_file,
    unpack_task_binary,
    verifier_strength,
)


def make_tar_bytes(files: dict[str, bytes], links: dict[str, str] | None = None) -> bytes:
    """An invented tar of the given name to byte members, plus symlinks where asked."""
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as tar:
        for name, body in files.items():
            info = tarfile.TarInfo(name=name)
            info.size = len(body)
            tar.addfile(info, io.BytesIO(body))
        for name, target in (links or {}).items():
            info = tarfile.TarInfo(name=name)
            info.type = tarfile.SYMTYPE
            info.linkname = target
            tar.addfile(info)
    return buffer.getvalue()


def make_blob(files: dict[str, bytes], links: dict[str, str] | None = None) -> str:
    """The registry wire shape: base64 of the gzipped tar."""
    return base64.b64encode(gzip.compress(make_tar_bytes(files, links))).decode("ascii")


TASK_TOML = b"""[environment]
docker_image = "invented-image:1"

[verifier]
timeout_sec = 30

[metadata]
author_name = "invented"
"""

CONTENT_TESTS = {
    "tests/test.sh": b"cmp /output/actual.txt tests/expected_output.txt\n",
    "tests/expected_output.txt": b"invented golden\n",
}

EXISTENCE_TESTS = {
    "tests/test_state.py": b"from pathlib import Path\n\n\ndef test_answer_file_exists():\n"
                           b"    answer_path = Path(\"/app/answer.txt\")\n"
                           b"    assert answer_path.exists()\n",
}


def invented_files(task_id: str, instruction: bytes, tests: dict[str, bytes] | None = None) -> dict[str, bytes]:
    """The five-piece layout for one invented task id."""
    files = {
        "instruction.md": instruction,
        "task.toml": TASK_TOML,
        "environment/Dockerfile": b"FROM invented-base:1\nWORKDIR /app\n",
        "solution/solve.sh": b"#!/bin/sh\necho invented\n",
        "environment/seeds/seed.txt": b"invented seed\n",
    }
    if tests is not None:
        files.update(tests)
    return files


def test_unpack_valid_tar_gives_every_field():
    blob = make_blob(invented_files("invented-task-1", b"# invented\nSolve invented.\n", CONTENT_TESTS))
    definition = unpack_task_binary(blob, task_id="invented-task-1", source_ref="invented-row")
    assert definition.task_id == "invented-task-1"
    assert definition.instruction == "# invented\nSolve invented.\n"
    assert definition.dockerfile == "FROM invented-base:1\nWORKDIR /app\n"
    assert definition.docker_image == "invented-image:1"
    assert definition.task_toml["verifier"]["timeout_sec"] == 30
    assert definition.task_toml["metadata"]["author_name"] == "invented"
    assert set(definition.tests) == {"tests/test.sh", "tests/expected_output.txt"}
    assert set(definition.solution) == {"solution/solve.sh"}
    assert set(definition.seeds) == {"environment/seeds/seed.txt"}
    assert definition.source_ref == "invented-row"


def test_unpack_accepts_raw_bytes_blob():
    blob = make_blob(invented_files("invented-task-1", b"# invented\n", None))
    raw = base64.b64decode(blob.encode("ascii"))
    assert unpack_task_binary(raw).instruction == "# invented\n"


def test_unpack_tar_without_tests_leaves_empty_maps_and_no_image():
    files = {
        "instruction.md": b"# invented\n",
        "task.toml": b"[verifier]\ntimeout_sec = 10\n",
        "environment/Dockerfile": b"FROM invented-base:1\n",
    }
    definition = unpack_task_binary(make_blob(files))
    assert definition.tests == {}
    assert definition.solution == {}
    assert definition.seeds == {}
    assert definition.docker_image is None
    assert definition.dockerfile == "FROM invented-base:1\n"


@pytest.mark.parametrize("member", ["../evil.sh", "/absolute-evil.sh", "sub/../../evil.sh"])
def test_unpack_tar_escaping_its_directory_is_refused(member: str):
    blob = make_blob({**invented_files("invented-task-1", b"# invented\n"), member: b"evil"})
    with pytest.raises(ValueError, match="outside|absolute|\\.\\."):
        unpack_task_binary(blob)


def test_unpack_tar_with_a_link_is_refused():
    files = invented_files("invented-task-1", b"# invented\n")
    blob = make_blob(files, links={"tests/sneaky": "/etc/invented"})
    with pytest.raises(ValueError, match="[Ll]ink"):
        unpack_task_binary(blob)


def test_unpack_garbage_is_refused_with_a_reason():
    with pytest.raises(ValueError, match="[Bb]ase64|[Gg]zip|[Tt]ar"):
        unpack_task_binary("not base64 at all!!!")


def test_read_task_dir_reads_the_plain_layout(tmp_path):
    root = tmp_path / "invented-task-2"
    (root / "environment" / "seeds").mkdir(parents=True)
    (root / "tests").mkdir()
    (root / "solution").mkdir()
    for name, body in invented_files("invented-task-2", b"# invented two\n", EXISTENCE_TESTS).items():
        (root / name).write_bytes(body)
    definition = read_task_dir(root)
    assert definition.task_id == "invented-task-2"
    assert definition.instruction == "# invented two\n"
    assert definition.docker_image == "invented-image:1"
    assert set(definition.tests) == {"tests/test_state.py"}


def test_directory_and_table_registries_answer_the_same_definition(tmp_path):
    instruction = b"# invented same\nSolve invented.\n"
    root = tmp_path / "table"
    task_dir = root / "invented-task-3"
    (task_dir / "environment").mkdir(parents=True)
    for name, body in invented_files("invented-task-3", instruction, CONTENT_TESTS).items():
        target = task_dir / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(body)
    blob = make_blob(invented_files("invented-task-3", instruction, CONTENT_TESTS))
    table = TableRegistry([{"path": "invented-task-3", "task_binary": blob}])
    directory = DirectoryRegistry(root)
    left, right = table.lookup("invented-task-3"), directory.lookup("invented-task-3")
    assert left is not None and right is not None
    assert left.instruction == right.instruction == "# invented same\nSolve invented.\n"
    assert left.docker_image == right.docker_image == "invented-image:1"
    assert left.tests.keys() == right.tests.keys()
    assert table.lookup("invented-missing") is None
    assert directory.lookup("invented-missing") is None


def test_table_registry_reads_a_json_file_of_rows(tmp_path):
    blob = make_blob(invented_files("invented-task-4", b"# invented four\n", None))
    rows_path = tmp_path / "rows.json"
    rows_path.write_text(json.dumps([{"path": "invented-task-4", "task_binary": blob}]),
                         encoding="utf-8")
    table = TableRegistry(rows_from_file(rows_path))
    assert table.lookup("invented-task-4") is not None
    assert table.lookup("invented-missing") is None


def test_table_registry_refuses_parquet_without_the_optional_dependency(tmp_path):
    parquet_path = tmp_path / "rows.parquet"
    parquet_path.write_bytes(b"invented")
    with pytest.raises(ValueError, match="[Pp]yarrow|[Pp]arquet"):
        rows_from_file(parquet_path)


def test_verifier_strength_labels_content_existence_and_none():
    content = unpack_task_binary(make_blob(invented_files("i-1", b"# i\n", CONTENT_TESTS)))
    existence = unpack_task_binary(make_blob(invented_files("i-2", b"# i\n", EXISTENCE_TESTS)))
    bare = unpack_task_binary(make_blob(invented_files("i-3", b"# i\n", None)))
    assert verifier_strength(content) == "content"
    assert verifier_strength(existence) == "existence"
    assert verifier_strength(bare) == "none"


def test_instruction_match_ignores_whitespace_but_rejects_different_text():
    definition = unpack_task_binary(make_blob(invented_files("i-1", b"# invented\nSolve invented.\n", None)))
    assert instruction_matches(definition, "# invented\nSolve invented.\n") is True
    assert instruction_matches(definition, "  # invented   \n\n Solve   invented.  ") is True
    assert instruction_matches(definition, "# invented\nSolve something else.\n") is False
    assert instruction_matches(definition, None) is False
