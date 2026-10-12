"""One Examiner session over named tasks, examiner model plus reroll model.

General: workdir, task ids, model ids, effort and caps arrive as arguments.
Nothing task, corpus or model specific lives in here.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

from kullback.ai.provider import Model, ModelConfig, live_model
from kullback.runner.budget import BudgetedModel, Ceiling, load_totals


class EffortModel(Model):
    """A model that sends a reasoning effort when the caller names none."""

    def __init__(self, inner: Model, effort: str):
        self.inner = inner
        self.effort = effort
        self.name = getattr(inner, "name", "model")

    def query(
        self,
        messages: list[dict],
        tools: Optional[list[dict]] = None,
        config: Optional[ModelConfig] = None,
    ):
        base = config.model_copy(deep=True) if config is not None else ModelConfig()
        if self.effort and base.reasoning_effort is None and base.effort is None:
            base.reasoning_effort = self.effort
        return self.inner.query(messages, tools=tools, config=base)


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("workdir")
    parser.add_argument("tasks_json")
    parser.add_argument("--examiner-model", required=True)
    parser.add_argument("--reroll-model", required=True)
    parser.add_argument("--reroll-effort", default="low")
    parser.add_argument("--examiner-cap-usd", type=float, default=2.2)
    parser.add_argument("--reroll-allowance-usd", type=float, default=0.8)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--user-model", default=None)
    args = parser.parse_args(argv)

    from kullback.examiner.session import examine

    workdir = Path(args.workdir)
    task_ids = (None if args.tasks_json == "NONE"
                else json.loads(Path(args.tasks_json).read_text()))
    prior = float(load_totals(workdir)["total"]["usd"])
    examiner = BudgetedModel(
        live_model(args.examiner_model), stage="examiner", workdir=workdir,
        model_id=args.examiner_model,
        ceiling=Ceiling(usd=prior + args.examiner_cap_usd, workdir=workdir),
    )
    rerolls = EffortModel(live_model(args.reroll_model), args.reroll_effort)
    user = (EffortModel(live_model(args.user_model), args.reroll_effort)
            if args.user_model else None)
    findings = examine(
        workdir, task_ids=task_ids, model=examiner,
        judge_model=examiner, probe_model=examiner, reroll_model=rerolls,
        allowance_usd=args.reroll_allowance_usd, workers=args.workers,
        user_model=user,
    )
    print(f"tasks={len(task_ids) if task_ids is not None else 'all'} findings={len(findings)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
