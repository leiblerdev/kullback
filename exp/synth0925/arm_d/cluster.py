"""Step 3: cluster candidate weaknesses into named skills.

Grouping is code over the judge's skill names. One judge call merges
near-duplicate labels and writes one evidence sentence per cluster.
Writes clusters.json: skill, member Run ids, evidence.
"""
import collections
import json
import re

from exp.synth0925.arm_d.judge import cost_of


def normalize(skill: str) -> str:
    text = (skill or "").lower().strip()
    text = re.sub(r"[^a-z0-9 ]+", " ", text)
    return re.sub(r"\s+", " ", text).strip() or "unnamed weakness"


MERGE_SYSTEM = """You merge near-duplicate skill labels from failure analysis.
Each label names something one agent Run failed to do, with how many Runs
carried it and one sample rationale. Merge only labels that name the same
missing ability in different words. Keep distinct abilities apart even when
they appear in similar Runs. Answer as one JSON object:
{"clusters": [{"name": short generic name, "labels": [labels merged],
"evidence": one sentence on what the member Runs show}]}."""


def merge_with_judge(model, groups: dict) -> tuple[list[dict], float]:
    payload = [{"label": label, "runs": len(rows),
                "sample": (rows[0].get("rationale") or "")[:200]}
               for label, rows in sorted(groups.items(),
                                         key=lambda kv: -len(kv[1]))]
    messages = [{"role": "system", "content": MERGE_SYSTEM},
                {"role": "user", "content": json.dumps(payload)[:8000]}]
    try:
        reply = model.query(messages)
    except Exception:
        return [], 0.0
    spent = cost_of(reply)
    try:
        start, end = reply.content.index("{"), reply.content.rindex("}") + 1
        body = json.loads(reply.content[start:end])
        clusters = body.get("clusters") or []
    except (ValueError, AttributeError):
        clusters = []
    if not clusters:
        clusters = [{"name": label, "labels": [label], "evidence": ""}
                    for label in groups]
    return clusters, spent


def cluster(weaknesses: list[dict], model=None) -> tuple[list[dict], float]:
    groups: dict[str, list[dict]] = collections.defaultdict(list)
    for row in weaknesses:
        groups[normalize(row.get("skill"))].append(row)
    if model is None:
        merged = [{"name": label, "labels": [label], "evidence": ""}
                  for label in groups]
        return [{"skill": item["name"],
                 "member_run_ids": [row["run_id"] for label in item["labels"]
                                    for row in groups.get(label, [])],
                 "member_task_ids": sorted({row["task_id"] for label in item["labels"]
                                            for row in groups.get(label, [])}),
                 "evidence": item["evidence"]} for item in merged], 0.0
    merged, spent = merge_with_judge(model, groups)
    out = []
    for item in merged:
        members = [row for label in item.get("labels", [])
                   for row in groups.get(label, [])]
        if not members:
            continue
        out.append({"skill": str(item.get("name") or "unnamed weakness")[:120],
                    "member_run_ids": [row["run_id"] for row in members],
                    "member_task_ids": sorted({row["task_id"] for row in members}),
                    "evidence": str(item.get("evidence") or "")[:400]})
    return out, spent
