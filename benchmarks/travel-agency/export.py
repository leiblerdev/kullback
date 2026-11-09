"""Write the Harbor tasks from one environment and the task files: python export.py --out harbor

Standard library only. Every Harbor task repeats the same world, runtime and grading, so they are kept once under
environment/ and each file under tasks/ holds only what differs. The output has the same bytes each time;
harbor.sha256 holds its digest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ENV = HERE / "environment"
WORLD_FILES = ("db.json", "toolkit.src", "tools.json", "world.json")
EXECUTABLE = 0o755
FILE = 0o644


def dump(value) -> str:
    return json.dumps(value, indent=1, sort_keys=True, ensure_ascii=False) + "\n"


def environment_files() -> dict[str, tuple[bytes, int]]:
    """The files every task repeats, by their path inside the task folder."""
    files = {f"environment/{name}": ENV / name for name in ("Dockerfile", "docker-compose.yaml", "runtime.py")}
    files.update({f"environment/world/{name}": ENV / "world" / name for name in WORLD_FILES})
    files["tests/runtime.py"] = ENV / "runtime.py"
    plan = {path: (source.read_bytes(), FILE) for path, source in files.items()}
    plan["tests/test.sh"] = ((ENV / "test.sh").read_bytes(), EXECUTABLE)
    plan["solution/solve.sh"] = ((ENV / "solve.sh").read_bytes(), EXECUTABLE)
    return plan


def task_files(task: dict, shared: dict, head: str) -> dict[str, tuple[bytes, int]]:
    """The files that differ between tasks, by their path inside the task folder."""
    grade = dump({**shared, **task["grade"]})
    texts = {
        "environment/world/overlay.json": task["overlay"],
        "environment/world/places.json": dump(task["places"]),
        "environment/world/secret": task["secret"],
        "instruction.md": head + task["instruction"],
        "task.toml": task["task_toml"],
        "environment/world/grade.json": grade,
        "tests/grade.json": grade,
        "solution/walk.json": dump(task["walk"]),
    }
    texts.update({f"solution/alts/{name}": dump(walk) for name, walk in task["alts"].items()})
    return {path: (text.encode("utf-8"), FILE) for path, text in texts.items()}


def plan() -> dict[str, tuple[bytes, int]]:
    """Every file the export writes, by its path inside the output folder."""
    shared = json.loads((ENV / "grading.json").read_text(encoding="utf-8"))
    head = (ENV / "instruction_head.md").read_text(encoding="utf-8")
    common = environment_files()
    files = {}
    for source in sorted((HERE / "tasks").glob("*.json")):
        own = task_files(json.loads(source.read_text(encoding="utf-8")), shared, head)
        files.update({f"{source.stem}/{path}": entry for path, entry in {**common, **own}.items()})
    files["README.md"] = ((HERE / "card.md").read_bytes(), FILE)
    files["tasks.jsonl"] = ((HERE / "tasks.jsonl").read_bytes(), FILE)
    return files


def owned(path: Path, out: Path) -> bool:
    """The export owns every entry except dot entries such as .git and what lies under them."""
    return not any(part.startswith(".") for part in path.relative_to(out).parts)


def remove_stale(out: Path, files: dict[str, tuple[bytes, int]]) -> None:
    """Remove files and folders an earlier export wrote that this export did not."""
    keep = {Path(path) for path in files}
    keep |= {parent for path in keep for parent in path.parents}
    for path in sorted(out.rglob("*"), key=lambda p: len(p.parts), reverse=True):
        if not owned(path, out) or path.relative_to(out) in keep:
            continue
        if path.is_dir() and not path.is_symlink():
            if not any(path.iterdir()):
                path.rmdir()
        else:
            path.unlink()


def digest(out: Path) -> str:
    """The digest of every owned file's path, executable bit and bytes."""
    sha = hashlib.sha256()
    for path in sorted(p for p in out.rglob("*") if p.is_file() and owned(p, out)):
        executable = bool(path.stat().st_mode & 0o100)
        sha.update(f"{path.relative_to(out).as_posix()}\0{executable}\0".encode())
        sha.update(path.read_bytes())
    return sha.hexdigest()


def export(out: Path) -> str:
    """Write every planned file, remove stale ones from an earlier export, and return the digest."""
    out = out.resolve()
    if out == HERE or out in HERE.parents:
        raise SystemExit(f"refusing to export into {out}: it holds the benchmark source")
    if out.is_dir() and not (out / "tasks.jsonl").is_file() and any(owned(p, out) for p in out.iterdir()):
        raise SystemExit(f"refusing to export into {out}: it holds files and no earlier export")
    files = plan()
    for name, (data, mode) in files.items():
        target = out / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        target.chmod(mode)
    remove_stale(out, files)
    return digest(out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=HERE / "harbor")
    print(export(parser.parse_args().out))


if __name__ == "__main__":
    main()
