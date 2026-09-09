"""One small workdir of an invented domain, and an in-memory stand-in for a dataset host.

The domain is a plant nursery: plots that hold a crop and a watering date, and two tools over them.
It is small enough to write out by hand and real enough that the Runner's own loader compiles it and
calls a tool, which is what makes "a fetched directory runs" a claim rather than a file count.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from kullback.runner.records import (
    Atom,
    Column,
    EntitySchema,
    FieldStat,
    RawPtr,
    Task,
    TaskOverlay,
    ToolSig,
    Verifier,
    as_dict,
    write_json,
)

# What the customer's export said. RECORDED_TURN is long enough to be prose rather than a field
# value: a package repeating it has a recording in it, and the export must refuse. ELICITED_VALUE is
# what a caller gave the desk mid conversation, so no world file holds it and a grader that checks
# the write has to carry it: the export counts that as an echo and lets it through.
GREETING = "good morning, this is the nursery desk"
RECORDED_TURN = ("Hello, this is Ada from the allotment on the far side. The thyme in plot P-2 was "
                 "never watered last night and I would like it done this morning before the sun is "
                 "up, if that is still possible.")
ELICITED_VALUE = "17 Fennel Row"

GET_PLOT_BODY = """
if plot_id not in self.db.plots:
    raise ValueError("Plot not found")
return self.db.plots[plot_id]
"""

WATER_PLOT_BODY = """
if plot_id not in self.db.plots:
    raise ValueError("Plot not found")
self.db.plots[plot_id].last_watered = on_date
return self.db.plots[plot_id]
"""

DB = {
    "plots": {
        "P-1": {"plot_id": "P-1", "crop": "basil", "last_watered": "2026-03-01"},
        "P-2": {"plot_id": "P-2", "crop": "thyme", "last_watered": "2026-03-02"},
    },
    "crews": {"C-9": {"crew_id": "C-9", "shift": "morning"}},
}

POLICY = """# Nursery desk policy

Water a plot only after reading it. Never water a plot twice on the same date.
"""

TASK_IDS = ("task_one", "task_two", "task_three")


def schema() -> EntitySchema:
    columns = [
        Column(table="plots", name="plot_id", **{"class": "hard"}),
        Column(table="plots", name="crop", **{"class": "hard"}),
        Column(table="plots", name="last_watered", **{"class": "hard"}),
        Column(table="crews", name="crew_id", **{"class": "hard"}),
        Column(table="crews", name="shift", **{"class": "hard"}),
    ]
    return EntitySchema(tables=["plots", "crews"], columns=columns,
                        id_patterns={"plots": "P-", "crews": "C-"})


def sigs() -> list[ToolSig]:
    return [
        ToolSig(name="get_plot", description="Read one plot.", kind="read", unclassified=False,
                args_fields=[FieldStat(name="plot_id", types=["str"], optional=False)]),
        ToolSig(name="water_plot", description="Record that a plot was watered.", kind="write",
                unclassified=False,
                args_fields=[FieldStat(name="plot_id", types=["str"], optional=False),
                             FieldStat(name="on_date", types=["str"], optional=False)]),
    ]


def verifier(task_id: str) -> Verifier:
    return Verifier(task_id=task_id, verifier_version="1", atoms=[
        Atom(id="w0", kind="required", description="water_plot writes P-1",
             predicate_src="wrote('water_plot', **{'plot_id': 'P-1'})",
             spans=[RawPtr(file_hash="reference-1", sim_index=0)],
             target={"kind": "write", "tool": "water_plot", "entity": "P-1", "id_field": "plot_id"}),
        Atom(id="q0", kind="question", description="the crop of the watered plot was told to the user",
             predicate_src="said('basil')", target={"kind": "question", "value": "basil"}),
    ])


def overlay(task_id: str) -> dict:
    return {"overlay": as_dict(TaskOverlay(task_id=task_id, rows=[])), "runs": {}, "values": {},
            "reference_run_id": "reference-1"}


def build_workdir(workdir: Path) -> Path:
    """A finished build's workdir: the world, and three Tasks stopped at three rungs of the funnel."""
    write_json(workdir / "env" / "db.json", DB)
    (workdir / "env" / "policy.md").write_text(POLICY, encoding="utf-8")
    (workdir / "env" / "tools.py").write_text("# rendered toolkit\n", encoding="utf-8")
    (workdir / "env" / "data_model.py").write_text("# rendered data model\n", encoding="utf-8")
    write_json(workdir / "env" / "tasks.json", [{"id": task_id} for task_id in TASK_IDS])
    write_json(workdir / "env" / "sidecar.json", {"env_id": "env-hash-1", "assisted_tools": []})
    write_json(workdir / "schema.json", as_dict(schema()))
    write_json(workdir / "tool_sigs.json", [as_dict(sig) for sig in sigs()])
    write_json(workdir / "bodies.json", {"get_plot": GET_PLOT_BODY, "water_plot": WATER_PLOT_BODY})
    write_json(workdir / "canon-rules.json", {"rules": []})
    write_json(workdir / "environment.json",
               {"env_id": "env-hash-1", "policy_version": "pol-1", "schema_version": "sch-1",
                "tools_version": "too-1", "assisted_tools": [], "version": 1})
    write_json(workdir / "runner_version.json",
               {"runner_version": "runner-hash-1", "gates_version": "gates-hash-1",
                "confirmed_by": "a person"})
    for task_id in TASK_IDS:
        write_json(workdir / "tasks" / f"{task_id}.json",
                   as_dict(Task(id=task_id, run_ids=[f"trace-{task_id}"],
                                intent="water the plot that is due")))
        write_json(workdir / "overlays" / f"{task_id}.json", overlay(task_id))
    for task_id in ("task_one", "task_two"):
        write_json(workdir / "verifiers" / f"{task_id}.json", as_dict(verifier(task_id)))
    write_json(workdir / "tasks_frozen.json", {"format": 2, "task_ids": list(TASK_IDS), "tasks": []})
    write_json(workdir / "replays.json", {
        "task_one": {"trace-task_one": {"confirmed": True}},
        "task_two": {"trace-task_two": {"confirmed": True}},
        "task_three": {"trace-task_three": {"confirmed": False}},
    })
    write_json(workdir / "task_status.json", {
        "task_one": {"reference_confirmed": True, "verifier_passed": True, "recordings": 1, "checks": {}},
        "task_two": {"reference_confirmed": True, "verifier_passed": False, "recordings": 1,
                     "checks": {"second_path_passes": False, "mutation_flips": False, "oracle_passes": True}},
        "task_three": {"reference_confirmed": False, "recordings": 1},
    })
    write_json(workdir / "rounds.json", [{"round": 2, "counts": {
        "round": 2, "fidelity": 2, "tasks": 3, "trusted": 1, "trusted_ids": ["task_one"], "refused": {}}}])
    write_json(workdir / "raw" / "export-hash.json",
               {"simulations": [{"id": "trace-task_one", "greeting": GREETING, "turn": RECORDED_TURN,
                                 "delivery_to": ELICITED_VALUE, "crop": "basil", "plot_id": "P-1",
                                 "watered_on": "2026-03-01"}]})
    return workdir


class FakeHub:
    """A dataset host in memory: repositories, their files, their commits and their tags.

    It answers the same five calls the real adapter does, so publish and fetch are exercised whole
    rather than around a boundary that only exists in tests.
    """

    def __init__(self, *, organisation_card_supported: bool = False):
        self.repos: dict[str, dict] = {}
        self.organisation_card_supported = organisation_card_supported
        self.organisation_cards: dict[str, str] = {}
        self.messages: list[str] = []

    def create_repo(self, repo_id: str) -> str:
        self.repos.setdefault(repo_id, {"commits": {}, "tags": {}, "head": None})
        return f"https://example.invalid/datasets/{repo_id}"

    def upload_folder(self, repo_id: str, folder: Path, message: str) -> str:
        repo = self.repos.setdefault(repo_id, {"commits": {}, "tags": {}, "head": None})
        files = {path.relative_to(folder).as_posix(): path.read_bytes()
                 for path in sorted(Path(folder).rglob("*")) if path.is_file()}
        commit = f"commit-{len(repo['commits']) + 1}"
        repo["commits"][commit] = files
        repo["head"] = commit
        self.messages.append(message)
        return commit

    def tag(self, repo_id: str, tag: str, revision=None) -> None:
        repo = self.repos[repo_id]
        repo["tags"][tag] = revision or repo["head"]

    def download(self, repo_id: str, out: Path, revision=None) -> Path:
        repo = self.repos[repo_id]
        commit = repo["tags"].get(revision, revision) if revision else repo["head"]
        files = repo["commits"][commit]
        out = Path(out)
        if out.exists():
            shutil.rmtree(out)
        for name, body in files.items():
            target = out / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(body)
        return out

    def set_organisation_card(self, organisation: str, markdown: str) -> str:
        from kullback.hub.client import HubError

        if not self.organisation_card_supported:
            raise HubError("this host has no API for an organisation profile")
        self.organisation_cards[organisation] = markdown
        return f"https://example.invalid/{organisation}"
