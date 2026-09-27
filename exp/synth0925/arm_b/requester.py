"""Arm B request writer: the synthesiser model writes the opening from the
units' goals and the user facts only. No tool, table, system or model names
in code or prompts: goals are change/refuse roles with counts, facts are read
off the workdir's askable fields. Spend is priced into the workdir budget with
a ceiling, and the running total is readable for summary.json.
"""
from __future__ import annotations

import json
from pathlib import Path

ENV_PATH = Path("/Users/krishuagarwal/Desktop/Programming/leibler/kullback/.env")
MODEL_ID = "opencode-go/muse-spark-1.3-contributor"
STAGE = "synthetic_request"
CEILING_USD = 5.0

PROMPT = (
    "A person is writing to a support agent about a problem they want solved. "
    "Write their first message.\n\n"
    "What they want:\n{goals}\n\n"
    "Write the symptom and the outcome they want, in their own words, in one "
    "or two sentences. Do not name any tool, any system, any table or any "
    "step, and do not say how the agent should do it. Do not include any "
    "value that is not in the list below.\n\n"
    "The only values this person knows and may state:\n{facts}\n\n"
    "Write the message and nothing else.")


def goals_for(n_changes: int, n_refusals: int) -> str:
    bits = []
    if n_changes == 1:
        bits.append("- something they want changed")
    elif n_changes > 1:
        bits.append(f"- {n_changes} things they want changed")
    if n_refusals == 1:
        bits.append("- something they want that the agent may have to refuse")
    elif n_refusals > 1:
        bits.append(f"- {n_refusals} things they want that the agent may have to refuse")
    return "\n".join(bits) or "- something they want checked"


class Writer:
    def __init__(self, budgetdir: Path, ceiling: float = CEILING_USD):
        """Spend is ledgered under budgetdir (the arm's own dir), never under
        the retail workdir: that ledger already holds the build's 71 USD, so
        a ceiling read from it would refuse the first call."""
        from kullback.ai import provider
        from kullback.runner import budget
        provider.load_dotenv(ENV_PATH)
        provider.enable_live_calls_from_env()
        inner = provider.live_model(MODEL_ID)
        if ceiling is not None:
            ceiling_obj = budget.Ceiling.from_totals(budgetdir, ceiling)
            ceiling_obj.require_priced(MODEL_ID)
        else:
            ceiling_obj = None
        memo = provider.MemoModel(inner, budgetdir)
        self.model = budget.BudgetedModel(
            memo, stage=STAGE, workdir=Path(budgetdir), model_id=MODEL_ID,
            ceiling=ceiling_obj,
            prompt_cache_key=f"kullback-{STAGE}")
        self.workdir = Path(budgetdir)

    def spent_usd(self) -> float:
        try:
            data = json.loads((self.workdir / "budget.json").read_text())
        except (OSError, ValueError):
            return 0.0
        stages = data.get("stages") or {}
        mine = stages.get(STAGE) or {}
        return float(mine.get("usd") or 0.0)

    def query(self, goals: str, facts: str) -> str:
        reply = self.model.query(
            [{"role": "user", "content": PROMPT.format(goals=goals, facts=facts)}])
        return str(getattr(reply, "content", "") or "").strip()
def write_request(model: Writer | None, goals: str, said: list[dict],
                  held: list[dict], fallback: str) -> tuple[str, str, float]:
    """The opening from goals and user facts only. A draft carrying a value
    only the world knows is refused, never edited. Returns text, reason, spend."""
    if model is None:
        return fallback, "no request writer was given", 0.0
    lines = "\n".join(f"- {r['field']}: {r['value']}" for r in said) or "- nothing"
    try:
        text = model.query(goals, lines)
    except Exception as error:  # noqa: BLE001 - provider failure keeps the fallback
        return fallback, f"the request writer failed: {type(error).__name__}", model.spent_usd()
    if not text:
        return fallback, "the request writer answered nothing", model.spent_usd()
    leaked = [r["field"] for r in held if str(r["value"]) and str(r["value"]) in text]
    if leaked:
        return fallback, "the draft stated a value only the world knows", model.spent_usd()
    return text, "", model.spent_usd()


def code_fallback(n_changes: int, n_refusals: int) -> str:
    if n_refusals and not n_changes:
        return "I need help with something in my account, even if it turns out it cannot be done."
    if n_refusals:
        return "I need a few things sorted out in my account, even if one of them turns out not to be possible."
    if n_changes > 1:
        return "I need a few things sorted out in my account."
    return "I need to check on something in my account."
