"""Phase 6 skills: the stage prompts the model may rewrite, versioned by content hash (D130).

Skills live as `skills/<name>/SKILL.md` in the workdir, are listed by name in the
system prompt, and load on demand. A skill edit is a node in the memory tree with
its content hash; every artifact compiled under a skill records the hash it was
compiled under. The skill gate is code over gate counts, never an LLM judgment: the
same artifacts compiled under both hashes, paired gate differences accumulated
across rounds, alpha 0.05, tentative until decisive, promoted or reverted when
decisive, the trunk parent advancing only on promotion, promoted skills re-checked
on later rounds and demoted the same way (D132).
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from kullback.runner.records import content_hash

SKILLS_DIRNAME = "skills"
SKILL_FILENAME = "SKILL.md"
EDITS_FILENAME = "edits.jsonl"
SKILL_GATE_ALPHA = 0.05
SKILL_GATE_MIN_ROUNDS = 3


# --- files --------------------------------------------------------------------


def skills_dir(workdir: Any) -> Path:
    return Path(workdir) / SKILLS_DIRNAME


def skill_path(workdir: Any, name: str) -> Path:
    return skills_dir(workdir) / name / SKILL_FILENAME


def list_skills(workdir: Any) -> list[str]:
    """Every skill with a SKILL.md in this workdir, sorted."""
    root = skills_dir(workdir)
    if not root.is_dir():
        return []
    return sorted(p.name for p in root.iterdir() if (p / SKILL_FILENAME).is_file())


def load_skill(workdir: Any, name: str) -> str:
    """The SKILL.md content for one skill; KeyError when the skill does not exist."""
    path = skill_path(workdir, name)
    if not path.is_file():
        raise KeyError(f"no skill {name!r} in {skills_dir(workdir)}")
    return path.read_text(encoding="utf-8")


def skill_prompt_section(workdir: Any) -> str:
    """The system-prompt lines listing this workdir's skills by name (loaded on demand)."""
    names = list_skills(workdir)
    if not names:
        return "Skills: none in this workdir yet."
    return "Skills: " + ", ".join(f"`{n}`" for n in names) + ". Load one on demand before its stage."


def write_skill(workdir: Any, name: str, content: str) -> dict:
    """Write one skill and record the edit as a memory-tree node: name, hash, previous hash, time.

    Returns {"name", "hash", "prev", "path"}. The caller (the rewrite_skill verb) attaches the
    hash to every artifact compiled under it, so a regrade can name the prompt version (D69).
    """
    path = skill_path(workdir, name)
    prev = path.read_text(encoding="utf-8") if path.is_file() else None
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    digest = content_hash({"skill": name, "content": content})
    prev_digest = content_hash({"skill": name, "content": prev}) if prev is not None else None
    edits = skills_dir(workdir) / EDITS_FILENAME
    with edits.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps({"name": name, "hash": digest, "prev": prev_digest, "at": time.time()},
                            sort_keys=True) + "\n")
    return {"name": name, "hash": digest, "prev": prev_digest, "path": str(path)}


def skill_edits(workdir: Any) -> list[dict]:
    """Every recorded skill edit in this workdir, oldest first."""
    edits = skills_dir(workdir) / EDITS_FILENAME
    if not edits.is_file():
        return []
    out = []
    for line in edits.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            out.append(json.loads(line))
    return out


# --- skill gate (D132, code over gate counts) -----------------------------------


def skill_gate_decision(paired_diffs: list[float], alpha: float = SKILL_GATE_ALPHA) -> dict:
    """Promote, revert, or hold a skill edit from paired per-artifact gate differences (new minus old).

    Each entry is +1 (new passes where old failed), -1 (new fails where old passed), or 0 (same).
    A normal approximation on the mean decides: |z| past the two-sided alpha threshold with at
    least SKILL_GATE_MIN_ROUNDS pairs promotes (mean > 0) or reverts (mean < 0); anything else is
    tentative. No model call; the caller accumulates diffs across rounds.
    """
    import math

    n = len(paired_diffs)
    if n == 0:
        return {"decision": "tentative", "n": 0, "mean": 0.0, "z": 0.0, "alpha": alpha,
                "reason": "no paired artifacts yet"}
    mean = sum(paired_diffs) / n
    if n < SKILL_GATE_MIN_ROUNDS:
        return {"decision": "tentative", "n": n, "mean": mean, "z": 0.0, "alpha": alpha,
                "reason": f"need {SKILL_GATE_MIN_ROUNDS} paired rounds, have {n}"}
    var = sum((d - mean) ** 2 for d in paired_diffs) / n
    sd = math.sqrt(var)
    if sd == 0:
        if mean > 0:
            return {"decision": "promote", "n": n, "mean": mean, "z": float("inf"), "alpha": alpha,
                    "reason": "new wins every paired artifact"}
        if mean < 0:
            return {"decision": "revert", "n": n, "mean": mean, "z": float("-inf"), "alpha": alpha,
                    "reason": "new loses every paired artifact"}
        return {"decision": "tentative", "n": n, "mean": mean, "z": 0.0, "alpha": alpha,
                "reason": "identical outcomes so far"}
    z = mean / (sd / math.sqrt(n))
    # Two-sided normal thresholds for common alphas; anything else falls back to 1.96.
    threshold = {0.10: 1.645, 0.05: 1.96, 0.01: 2.576}.get(alpha, 1.96)
    if z >= threshold:
        return {"decision": "promote", "n": n, "mean": mean, "z": z, "alpha": alpha,
                "reason": "paired gate wins are decisive"}
    if z <= -threshold:
        return {"decision": "revert", "n": n, "mean": mean, "z": z, "alpha": alpha,
                "reason": "paired gate losses are decisive"}
    return {"decision": "tentative", "n": n, "mean": mean, "z": z, "alpha": alpha,
            "reason": "not decisive yet"}


def recheck_promoted(paired_diffs: list[float], alpha: float = SKILL_GATE_ALPHA) -> dict:
    """Re-check a promoted skill on later rounds: a decisive negative reverts (demotes) it."""
    decision = skill_gate_decision(paired_diffs, alpha=alpha)
    if decision["decision"] == "revert":
        decision = dict(decision)
        decision["decision"] = "demote"
        decision["reason"] = "promoted skill fails re-check: " + decision["reason"]
    return decision
