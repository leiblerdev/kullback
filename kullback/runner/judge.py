"""The agentic judge (D92): read-only tools over the Starting and End state, at least one check
before any verdict, two judges whose disagreement goes to a queue."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Optional

from pydantic import BaseModel, ConfigDict, Field

from kullback.ai.provider import Model, ModelConfig
from kullback.runner.records import as_dict, canonical_json, disagreement_stats, read_jsonl

JUDGE_VERSION = "0"
QUEUE_FILE = "disagreement_queue.jsonl"
PAIRS_FILE = "judge_pairs.jsonl"
ASIDE_FILE = "tasks_aside.jsonl"

# One row per judge use: the verdicts it may return, and which of them is its abstain.
# D57 Reference confirmation, D76 policy atoms, D84 semantic equality, D88 failure cause,
# and the dispute path. D88's abstain is spelled `undetermined`.
_USES: dict[str, tuple[tuple[str, ...], str]] = {
    "policy_atom": (("pass", "fail", "abstain"), "abstain"),
    "equivalence": (("equivalent", "not_equivalent", "abstain"), "abstain"),
    "reference": (("good_reference", "bad_reference", "abstain"), "abstain"),
    "cause": (("candidate", "environment", "simulated_user", "undetermined"), "undetermined"),
    "dispute": (("acceptable", "unacceptable", "abstain"), "abstain"),
}


def abstain_verdict(use: str) -> str:
    """The verdict that means "this judge did not decide" for one judge use."""
    return _USES[use][1] if use in _USES else "abstain"


class JudgeResult(BaseModel):
    """One judge's answer: the verdict, the spans it cited, and the tools it actually ran."""

    model_config = ConfigDict(populate_by_name=True)

    use: str
    verdict: str
    judge: str = ""
    cited_spans: list[str] = Field(default_factory=list)
    tools_run: list[str] = Field(default_factory=list)
    tool_results: list[dict] = Field(default_factory=list)
    sub_answers: list[dict] = Field(default_factory=list)
    reason: Optional[str] = None
    refused: bool = False
    judge_version: str = JUDGE_VERSION
    pair: list[dict] = Field(default_factory=list)

    @property
    def abstained(self) -> bool:
        return self.verdict == abstain_verdict(self.use)


class AgenticJudge:
    """One judge: a model, read-only tools over the Starting state and the End state, and the Verifier's output."""

    def __init__(
        self,
        model: Model,
        tools: Optional[dict[str, Callable[..., Any]]] = None,
        verifier_output: Any = None,
        name: Optional[str] = None,
        persona: Optional[str] = None,
        max_steps: int = 4,
        judge_version: str = JUDGE_VERSION,
    ) -> None:
        self.model = model
        self.tools = dict(tools or {})
        self.verifier_output = verifier_output
        self.name = name or getattr(model, "name", "judge")
        self.persona = persona
        self.max_steps = max_steps
        self.judge_version = judge_version

    # --- the five judge uses ---

    def judge_policy_atom(self, rule: Any, transcript: Any) -> JudgeResult:
        """D76: one policy rule against one transcript, as atomic yes or no sub-questions (R27 8b)."""
        prompt = (
            "Policy rule under check, and no other rule:\n"
            f"{_render(rule)}\n\n"
            "Break this rule into atomic yes or no sub-questions, answer each one from the transcript "
            "and from what the tools show, and return them in sub_answers as "
            '{"question": ..., "answer": "yes" or "no" or "abstain", "cited_span": ...}. '
            "Answer abstain on a sub-question the evidence does not settle.\n\n"
            f"Transcript up to and including the End state:\n{_render(transcript)}"
        )
        return self._ask("policy_atom", prompt)

    def judge_equivalence(self, column: Any, a: Any, b: Any, field_type: Optional[str] = None) -> JudgeResult:
        """D84: do two values mean the same thing for one semantic column (R27 8a, pairwise, no transcript)."""
        prompt = (
            f"Field: {_render(column)}\n"
            f"Semantic type: {field_type or 'unknown'}\n"
            f"Value A, from the Reference: {_render(a)}\n"
            f"Value B, from the Run being graded: {_render(b)}\n\n"
            "Do these two values mean the same thing for this field? Judge the two values themselves; "
            "do not reach for surrounding transcript context. Put any number, unit, date or negation "
            'mismatch in "flags"; a flagged pair is never equivalent.'
        )
        return self._ask("equivalence", prompt)

    def judge_reference(self, reference_run: Any, intent: Any, verifier_output: Any = None) -> JudgeResult:
        """D57: is this recorded Run a good Reference, in Trust or Escalate framing (R27 8c)."""
        checked = verifier_output if verifier_output is not None else self.verifier_output
        prompt = (
            "Trust or escalate. Decide whether this recorded Run is a good Reference for its Task.\n\n"
            f"Intent of the Task, the outcome the recorded user finally asked for:\n{_render(intent)}\n\n"
            f"Deterministic verifier output on this Run, already decided by code:\n{_render(checked)}\n\n"
            f"Candidate Reference Run:\n{_render(reference_run)}\n\n"
            "Grade the End state against the Intent, and only that. The transcript is not in evidence: "
            "whether the agent authenticated the user, asked for confirmation or followed a procedure "
            "cannot be seen through your tools and never decides the verdict. Judge the Intent, not the "
            "opening request; a user who changed their mind during the Run wanted what the Intent says. "
            "good_reference when the End state is what the Intent asks for, bad_reference when the End "
            "state contradicts it, abstain when the evidence does not decide it. Judge only what code "
            "cannot check. A bad Reference sets a wrong bar for every later Verdict on this Task, so "
            "escalate with abstain rather than guess."
        )
        return self._ask("reference", prompt)

    def judge_cause(self, failed_run: Any, reference_run: Any) -> JudgeResult:
        """D88: name the cause of a failed Run, with the Reference beside it."""
        prompt = (
            "Name the cause of this failure: candidate, environment, simulated_user, or undetermined.\n"
            "candidate means the graded model got it wrong. environment means the replica misled it. "
            "simulated_user means the user's side did. undetermined means the evidence does not say; "
            "use it rather than guessing.\n\n"
            f"Failed Run:\n{_render(failed_run)}\n\n"
            f"Reference Run:\n{_render(reference_run)}"
        )
        return self._ask("cause", prompt)

    def judge_dispute(self, end_state: Any, required: Any, allowed: Any) -> JudgeResult:
        """The dispute path (R27 8d): an End state outside the known required and allowed sets."""
        prompt = (
            "This End state falls outside the known required and allowed sets. Query the state with your "
            "tools and decide whether it is acceptable anyway.\n\n"
            f"End state under dispute:\n{_render(end_state)}\n\n"
            f"Required:\n{_render(required)}\n\n"
            f"Allowed:\n{_render(allowed)}\n\n"
            "Before you answer, state in reason what evidence would change your mind, then check with a "
            "tool whether that evidence is present."
        )
        return self._ask("dispute", prompt)

    # --- the tool loop ---

    def _ask(self, use: str, prompt: str) -> JudgeResult:
        verdicts, abstain = _USES[use]
        messages: list[dict] = [
            {"role": "system", "content": self._system(use, verdicts, abstain)},
            {"role": "user", "content": prompt},
        ]
        tools_run: list[str] = []
        results: list[dict] = []
        for _ in range(self.max_steps):
            reply = self.model.query(messages, tools=self._tool_specs(), config=ModelConfig(temperature=0))
            if not reply.tool_calls:
                return self._finish(use, reply.content, tools_run, results)
            messages.append({
                "role": "assistant", "content": reply.content or "",
                "tool_calls": [{"id": c.id, "name": c.name, "arguments": c.arguments} for c in reply.tool_calls],
            })
            for call in reply.tool_calls:
                ran, output = self._run_tool(call.name, call.arguments or {})
                if ran:
                    tools_run.append(call.name)
                    results.append({"name": call.name, "args": call.arguments or {}, "result": output})
                messages.append({"role": "tool", "tool_call_id": call.id, "name": call.name,
                                 "content": _render(output)})
        return self._result(use, abstain, tools_run, results, refused=True,
                            reason=f"no verdict within {self.max_steps} steps")

    def _run_tool(self, name: str, args: dict) -> tuple[bool, Any]:
        """Run one read-only tool. A missing tool or a raising tool is not a check."""
        tool = self.tools.get(name)
        if tool is None:
            return False, {"error": f"tool_not_found: {name}", "available": sorted(self.tools)}
        try:
            return True, tool(**args)
        except Exception as exc:  # a broken tool is evidence for the judge, not a crash
            return False, {"error": f"{type(exc).__name__}: {exc}"}

    def _tool_specs(self) -> list[dict]:
        return [
            {
                "name": name,
                "description": _first_line(tool.__doc__) or "read-only check over the state",
                "input_schema": {"type": "object", "additionalProperties": True},
            }
            for name, tool in self.tools.items()
        ]

    def _system(self, use: str, verdicts: tuple[str, ...], abstain: str) -> str:
        lines = [
            "You are one of two independent judges grading part of a recorded agent Run.",
            "Your tools are read-only views of the Task's Starting state and the Run's End state.",
            "Run at least one tool and check the state before you answer. A verdict with no tool check is refused.",
            "Answer with one JSON object and nothing else: "
            '{"verdict": one of ' + ", ".join(verdicts) + ', "cited_spans": [...], '
            '"sub_answers": [...], "flags": [...], "reason": "..."}.',
            "cited_spans must quote the exact substrings or tool results that decide the verdict.",
            f"Answer {abstain} when the evidence does not decide it; abstaining is better than guessing.",
        ]
        if self.persona:
            lines.append(f"Your persona for this judgement: {self.persona}")
        if self.verifier_output is not None:
            lines.append(
                "The deterministic verifier already ran; do not re-decide what it decided:\n"
                + _render(self.verifier_output)
            )
        return "\n".join(lines)

    # --- reading the answer ---

    def _finish(self, use: str, content: Optional[str], tools_run: list[str], results: list[dict]) -> JudgeResult:
        verdicts, abstain = _USES[use]
        if not tools_run:
            return self._result(
                use, abstain, tools_run, results,
                reason="refused: no tool check before the verdict (D92)", refused=True,
            )
        data = _parse_json(content)
        if data is None:
            return self._result(use, abstain, tools_run, results, reason="the reply was not a JSON verdict object")
        cited = {
            "cited_spans": [str(s) for s in _as_list(data.get("cited_spans"))],
            "sub_answers": [s for s in _as_list(data.get("sub_answers")) if isinstance(s, dict)],
        }
        verdict = str(data.get("verdict") or "").strip()
        reason = data.get("reason")
        if use == "policy_atom":
            # R27 8b: the overall verdict is a fixed combination of the atomic answers, not the model's word.
            verdict, reason = _verdict_from_sub_answers(cited["sub_answers"], reason)
        elif use == "equivalence" and verdict == "equivalent" and _as_list(data.get("flags")):
            verdict, reason = abstain, "the judge flagged a number, unit, date or negation mismatch"
        if verdict not in verdicts:
            verdict, reason = abstain, f"unknown verdict {verdict!r}"
        return self._result(use, verdict, tools_run, results, reason=reason, **cited)

    def _result(
        self, use: str, verdict: str, tools_run: list[str], results: list[dict],
        reason: Any = None, refused: bool = False, **cited: Any,
    ) -> JudgeResult:
        return JudgeResult(
            use=use, verdict=verdict, judge=self.name, tools_run=list(tools_run),
            tool_results=list(results), reason=None if reason is None else str(reason),
            refused=refused, judge_version=self.judge_version, **cited,
        )


# --- two judges (D92), with D97's third sample on a split ---

THIRD_PERSONA = ("a third reader brought in because the first two split; weigh the evidence yourself "
                 "and do not defer to either of them")


def third_judge(judge: AgenticJudge, persona: str = THIRD_PERSONA) -> AgenticJudge:
    """D97's third sample: one of the two models again, under a different persona, same tools."""
    return AgenticJudge(judge.model, judge.tools, judge.verifier_output, name=f"{judge.name}#3",
                        persona=persona, max_steps=judge.max_steps, judge_version=judge.judge_version)


def two_judges(
    judge_a: AgenticJudge,
    judge_b: AgenticJudge,
    fn: Any,
    *args: Any,
    workdir: Optional[Path] = None,
    item_id: Optional[str] = None,
    judge_c: Optional[AgenticJudge] = None,
    third_sample: bool = True,
    **kwargs: Any,
) -> tuple[JudgeResult, bool]:
    """Ask both judges the same question. Agreement is the answer; disagreement is the abstain (D92).

    D97's default takes one more sample when the two split on a non-Reference atom: `judge_c`, or a
    third sample from judge A's model under a different persona, and the majority decides. A
    three-way split still abstains to the queue. A Reference split goes to a person (D93), so no
    third sample is taken there. `third_sample=False` is the plain two-judge protocol.
    """
    ask = fn if callable(fn) else (lambda judge, *a, **k: getattr(judge, fn)(*a, **k))
    first = ask(judge_a, *args, **kwargs)
    second = ask(judge_b, *args, **kwargs)
    pair = [as_dict(first), as_dict(second)]
    disagreement = first.verdict != second.verdict
    third: Optional[JudgeResult] = None
    if disagreement and third_sample and first.use != "reference":
        extra = judge_c if judge_c is not None else third_judge(judge_a)
        try:
            third = ask(extra, *args, **kwargs)
        except Exception as error:
            # The default third sample reuses judge A's model, which may have nothing left to say.
            # A third sample that cannot answer is a refusal, and the split still reaches the queue.
            third = JudgeResult(use=first.use, verdict=abstain_verdict(first.use), judge=extra.name,
                                refused=True, judge_version=first.judge_version,
                                reason=f"the third sample did not answer: {type(error).__name__}: {error}")
        pair.append(as_dict(third))
        majority = next((r for r in (first, second) if r.verdict == third.verdict), None)
        if majority is not None:
            disagreement = False  # the third sample settled it, so nothing goes to the queue
            result = majority.model_copy(update={
                "pair": pair,
                "reason": (f"the two judges split, {third.judge} agreed with {majority.judge} "
                           f"on {majority.verdict}"),
            })
    if disagreement:
        result = JudgeResult(
            use=first.use,
            verdict=abstain_verdict(first.use),
            judge=f"{first.judge}+{second.judge}",
            cited_spans=first.cited_spans + second.cited_spans,
            tools_run=first.tools_run + second.tools_run,
            tool_results=first.tool_results + second.tool_results,
            sub_answers=first.sub_answers + second.sub_answers,
            reason=(
                f"judges disagreed: {first.judge} said {first.verdict}, "
                f"{second.judge} said {second.verdict}"
                + (f", {third.judge} said {third.verdict}" if third is not None else "")
            ),
            judge_version=first.judge_version,
            pair=pair,
        )
    elif third is None:
        result = first.model_copy(update={"pair": pair})
    reason = _queue_reason(result, first, second, third, disagreement)
    if workdir is not None:
        row = {
            "use": first.use,
            "item_id": item_id,
            "verdict_a": first.verdict,
            "verdict_b": second.verdict,
            "disagreement": disagreement,
            "abstain": bool(reason) and not disagreement,
            "reason": reason,
        }
        if third is not None:
            row["verdict_c"] = third.verdict
        _append(Path(workdir) / PAIRS_FILE, row)
        if reason:
            _append(Path(workdir) / QUEUE_FILE, dict(row, judge_a=pair[0], judge_b=pair[1]))
    return result, disagreement


def _queue_reason(result: JudgeResult, first: JudgeResult, second: JudgeResult,
                  third: Optional[JudgeResult], disagreement: bool) -> Optional[str]:
    """Why a person has to see this item, or None when the judges decided it (D92, D88).

    A split is not the only undecided outcome: two judges that both abstain, two that both refused
    for want of a tool check, and a third sample that makes abstain the majority all leave the item
    undecided, and D92 says every one of those goes to the queue.
    """
    if disagreement:
        return "split"
    if result.verdict != abstain_verdict(result.use):
        return None
    if first.refused and second.refused:
        return "refused"
    return "abstain_majority" if third is not None else "agreed_abstain"


def judge_atom_results(
    verifier: Any,
    transcript: Any,
    judge_a: AgenticJudge,
    judge_b: AgenticJudge,
    *,
    workdir: Optional[Path] = None,
    run_id: Optional[str] = None,
) -> dict:
    """Answer every judge atom of one Verifier for one Run: {atom_id: JudgeResult} (D76).

    This is the shape `verdict.py` takes as `judge_results`; the caller between the Run and the
    Verdict (the pipeline) runs it, because `verdict.py` never calls a model itself.
    """
    out: dict = {}
    for atom in getattr(verifier, "atoms", []) or []:
        if not getattr(atom, "judge", False):
            continue
        rule = getattr(atom, "description", None) or getattr(atom, "target", None) or atom.id
        result, _ = two_judges(
            judge_a, judge_b, "judge_policy_atom", rule, transcript,
            workdir=workdir, item_id=f"{run_id}:{atom.id}" if run_id else atom.id,
        )
        out[atom.id] = result
    return out


def judge_cause_result(
    failed_run: Any,
    reference_run: Any,
    judge_a: AgenticJudge,
    judge_b: AgenticJudge,
    *,
    workdir: Optional[Path] = None,
    run_id: Optional[str] = None,
) -> JudgeResult:
    """Name the cause of one failed Run (D88); `verdict.py` takes this as `cause_result`."""
    result, _ = two_judges(judge_a, judge_b, "judge_cause", failed_run, reference_run,
                           workdir=workdir, item_id=run_id)
    return result


def confirm_reference(
    judge_a: AgenticJudge,
    judge_b: AgenticJudge,
    reference_run: Any,
    intent: Any,
    verifier_output: Any = None,
    *,
    workdir: Optional[Path] = None,
    task_id: Optional[str] = None,
) -> tuple[bool, JudgeResult]:
    """D57 and D93: two judges confirm a Reference; if they split, the Task is set aside for a person."""
    result, disagreement = two_judges(
        judge_a, judge_b, "judge_reference", reference_run, intent, verifier_output,
        workdir=workdir, item_id=task_id,
    )
    if result.verdict == "good_reference" and not disagreement:
        return True, result
    reason = "reference_disputed" if disagreement else "reference_unconfirmed"
    if workdir is not None and task_id is not None:
        _append_aside(workdir, task_id, reason, result.pair or [as_dict(result)] * 2)
    return False, result


def set_task_aside(
    workdir: Path, task_id: str, reason: str, result_a: JudgeResult, result_b: JudgeResult
) -> None:
    """Record a Task as not gradeable until a person resolves it (D93)."""
    _append_aside(workdir, task_id, reason, [as_dict(result_a), as_dict(result_b)])


def read_disagreement_queue(workdir: Path) -> list[dict]:
    """Every judge disagreement, with both verdicts and both sets of cited spans."""
    return read_jsonl(Path(workdir) / QUEUE_FILE)


def tasks_set_aside(workdir: Path) -> list[dict]:
    """Every Task the report must list as not gradeable, Reference disputed (D93)."""
    return read_jsonl(Path(workdir) / ASIDE_FILE)


def disagreement_rate(workdir: Path, use: Optional[str] = None) -> dict:
    """The number that travels with every judge result until human labels exist (D92)."""
    rows = [r for r in read_jsonl(Path(workdir) / PAIRS_FILE) if use is None or r.get("use") == use]
    return disagreement_stats(rows)


# --- small helpers ---


def _verdict_from_sub_answers(subs: list[dict], reason: Any) -> tuple[str, Any]:
    if not subs:
        return "abstain", "no sub-answers to compute the verdict from"
    answers = [_first_word(s.get("answer", "")) for s in subs]
    if any(a not in ("yes", "no") for a in answers):
        return "abstain", "a sub-question was not answered yes or no"
    if all(a == "yes" for a in answers):
        return "pass", reason
    return "fail", reason


def _first_word(answer: Any) -> str:
    """The sub-answer's first word, without its punctuation: "Yes." and "yes, it did" are both yes."""
    text = str(answer or "").strip().lower()
    word = text.split()[0] if text.split() else ""
    return word.strip(".,;:!?\"'()[]")


def _parse_json(content: Optional[str]) -> Optional[dict]:
    """The judge's JSON object, whether it came bare or wrapped in prose."""
    if not content:
        return None
    text = content.strip()
    for candidate in (text, text[text.find("{") : text.rfind("}") + 1]):
        if not candidate:
            continue
        try:
            data = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict):
            return data
    return None


def _render(obj: Any) -> str:
    return obj if isinstance(obj, str) else canonical_json(obj)


def _as_list(value: Any) -> list:
    return value if isinstance(value, list) else []


def _first_line(text: Optional[str]) -> str:
    return (text or "").strip().splitlines()[0].strip() if (text or "").strip() else ""


def _append_aside(workdir: Path, task_id: str, reason: str, pair: list[dict]) -> None:
    _append(Path(workdir) / ASIDE_FILE,
            {"task_id": task_id, "reason": reason, "judge_a": pair[0], "judge_b": pair[1]})


def _append(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")



