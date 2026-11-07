"""Write the Harbor tasks from one environment and the task files: python export.py --out harbor

Standard library only. Every Harbor task repeats the same world, runtime and grading, so they are kept once under
environment/ and each file under tasks/ holds only what differs. The output has the same bytes each time;
harbor.sha256 holds its digest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
ENV = HERE / "environment"
WORLD_FILES = ("db.json", "toolkit.src", "tools.json", "world.json")
EXECUTABLE = 0o755


def dump(value) -> str:
    return json.dumps(value, indent=1, sort_keys=True, ensure_ascii=False) + "\n"


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="")
    path.chmod(0o644)


def copy(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)
    target.chmod(0o644)


def write_environment(root: Path) -> None:
    for name in ("Dockerfile", "docker-compose.yaml", "runtime.py"):
        copy(ENV / name, root / "environment" / name)
    for name in WORLD_FILES:
        copy(ENV / "world" / name, root / "environment" / "world" / name)
    copy(ENV / "runtime.py", root / "tests" / "runtime.py")
    shutil.copyfile(ENV / "test.sh", root / "tests" / "test.sh")
    (root / "tests" / "test.sh").chmod(EXECUTABLE)
    (root / "solution").mkdir(parents=True, exist_ok=True)
    shutil.copyfile(ENV / "solve.sh", root / "solution" / "solve.sh")
    (root / "solution" / "solve.sh").chmod(EXECUTABLE)


def write_task(task: dict, root: Path, shared: dict, head: str) -> None:
    write_environment(root)
    world = root / "environment" / "world"
    write(world / "overlay.json", task["overlay"])
    write(world / "places.json", dump(task["places"]))
    write(world / "secret", task["secret"])
    write(root / "instruction.md", head + task["instruction"])
    write(root / "task.toml", task["task_toml"])
    grade = dump({**shared, **task["grade"]})
    write(world / "grade.json", grade)
    write(root / "tests" / "grade.json", grade)
    write(root / "solution" / "walk.json", dump(task["walk"]))
    for name, walk in task["alts"].items():
        write(root / "solution" / "alts" / name, dump(walk))


def digest(out: Path) -> str:
    """The digest of every file's path, executable bit and bytes."""
    sha = hashlib.sha256()
    for path in sorted(p for p in out.rglob("*") if p.is_file()):
        executable = bool(path.stat().st_mode & 0o100)
        sha.update(f"{path.relative_to(out).as_posix()}\0{executable}\0".encode())
        sha.update(path.read_bytes())
    return sha.hexdigest()


def export(out: Path) -> str:
    shared = json.loads((ENV / "grading.json").read_text(encoding="utf-8"))
    head = (ENV / "instruction_head.md").read_text(encoding="utf-8")
    for source in sorted((HERE / "tasks").glob("*.json")):
        write_task(json.loads(source.read_text(encoding="utf-8")), out / source.stem, shared, head)
    copy(HERE / "card.md", out / "README.md")
    copy(HERE / "tasks.jsonl", out / "tasks.jsonl")
    return digest(out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=HERE / "harbor")
    print(export(parser.parse_args().out))


if __name__ == "__main__":
    main()
