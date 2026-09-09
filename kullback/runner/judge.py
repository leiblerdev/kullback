"""The agentic judge (D92): read-only views of the Starting and End state, the check the question
needs run by the harness before the model is asked, and two judges whose disagreement goes to a queue.

D92 asked the judge to look before it ruled and made a verdict with no tool call a refusal. That
made the verdict depend on a habit: a model that answers straight away is refused, and on one
contributor model every semantic-equivalence pair came back refused for that reason alone, which
D219 then read as unresolved rather than as agreement. So the look moved (D222). The harness runs
the reads the question needs itself, over the same material the judge would have read, and puts
them in the prompt as a checks section with the tool name, its arguments and its result. The model
may still call a tool for more, and its choosing to or not choosing to decides nothing. The refusal
is kept for the one case it now means something: a question the harness could prefill no check for,
which is a bug here and not a habit there.
"""

from __future__ import annotations

import inspect
import json
import re
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

from pydantic import BaseModel, ConfigDict, Field

from kullback.ai.provider import Model, ModelConfig, ProviderError
from kullback.runner.records import as_dict, canonical_json, read_jsonl
from kullback.runner.records import disagreement_stats as _pair_counts

# Bumped at D222: the prompt now carries the checks and an agreeing verdict has to cite one, so a
# ruling stored under the old version was decided by a different rule and is recomputed, not read.
JUDGE_VERSION = "1"
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


# One row per judge use: the sources that use hands its judge, in the words the judge is asked to
# name them by. Nothing else is in evidence. The Reference judge is the case this row exists for: it
# gets the Intent, the Verifier's output and the End state through its tools, and never a transcript,
# so authentication, a spoken confirmation and the opening request cannot be seen from where it sits.
_SOURCES: dict[str, tuple[str, ...]] = {
    "policy_atom": ("policy_rule", "transcript", "end_state", "state_tools"),
    "equivalence": ("value_a", "value_b", "state_tools"),
    "reference": ("intent", "verifier_output", "end_state", "state_tools"),
    "cause": ("failed_run", "reference_run", "end_state", "state_tools"),
    "dispute": ("end_state", "required_set", "allowed_set", "state_tools"),
}


# The answer of each judge use that means "these two are the same" (D222 rule 3). D186 already made
# a failure name a key the states differ on; this is the other half, so an agreement is a claim about
# a check that was run rather than a word with nothing behind it. Only the equivalence use has such
# an answer today; a use that grows one joins the row and takes the rule with it.
_AGREEMENT: dict[str, str] = {"equivalence": "equivalent"}

# The names of the counts a round reads off its judgements (D222).
JUDGE_COUNTS = ("judge_checks_prefilled", "judge_extra_tool_calls", "judge_refused_no_check",
                "judge_tool_choice_forced", "judge_tool_choice_rejected")

NO_CHECK = "refused: the harness prefilled no check for this question (D222)"
UNCITED = "uncited: the agreeing verdict named no check it rests on"

CHECK_RESULT_CHARS = 2000  # each prefilled result, clamped, so a big source is read once not twice


def abstain_verdict(use: str) -> str:
    """The verdict that means "this judge did not decide" for one judge use."""
    return _USES[use][1] if use in _USES else "abstain"


def sources_of(use: str) -> tuple[str, ...]:
    """The sources one judge use hands its judge; a ruling may rest on these and on nothing else."""
    return _SOURCES.get(use, ())


def sources_not_given(named: Any, available: Iterable[str]) -> list[str]:
    """Of the sources a ruling names, the ones its judge was not handed, in the order it named them.

    The rule is code over what the judge was given, not a keyword list over its prose: the judge is
    told which sources it has, it names the ones its verdict rests on, and this compares that list
    against them. A ruling that reaches for something absent by construction is an abstention (D93),
    which is what build 8 needed: eleven Tasks were failed "without evidence of the required
    authentication and explicit confirmation" by a judge that was never handed a transcript.
    """
    have = {_source_word(s) for s in available}
    out: list[str] = []
    for name in _names(named):
        word = _source_word(name)
        if word and word not in have and word not in out:
            out.append(word)
    return out


def _names(value: Any) -> list[str]:
    """A named-sources field, whether the judge wrote it as a list or as one string."""
    if isinstance(value, str):
        return [value] if value.strip() else []
    return [str(s) for s in value] if isinstance(value, (list, tuple)) else []


def _source_word(name: Any) -> str:
    """One source name as the rule compares it: lower case, underscored, and singular."""
    word = re.sub(r"[^a-z0-9]+", "_", str(name).strip().lower()).strip("_")
    return word[:-1] if word.endswith("s") and not word.endswith("ss") else word


class JudgeResult(BaseModel):
    """One judge's answer: the verdict, the spans it cited, and the tools it actually ran."""

    model_config = ConfigDict(populate_by_name=True)

    use: str
    verdict: str
    judge: str = ""
    cited_spans: list[str] = Field(default_factory=list)
    evidence: list[str] = Field(default_factory=list)  # the sources the judge said its verdict rests on
    tools_run: list[str] = Field(default_factory=list)
    tool_results: list[dict] = Field(default_factory=list)
    sub_answers: list[dict] = Field(default_factory=list)
    reason: Optional[str] = None
    refused: bool = False
    judge_version: str = JUDGE_VERSION
    pair: list[dict] = Field(default_factory=list)
    # D222: the reads the harness ran before the model was asked, each one a tool name, its
    # arguments and what it answered; the tool calls the model made on top of them; and whether the
    # first turn was made to choose a check and whether the provider took that instruction.
    checks: list[dict] = Field(default_factory=list)
    extra_tool_calls: int = 0
    tool_choice_forced: bool = False
    tool_choice_rejected: bool = False

    @property
    def abstained(self) -> bool:
        return self.verdict == abstain_verdict(self.use)


def check_sources(use: str) -> tuple[str, ...]:
    """The sources of one judge use the harness reads for the model, before it is asked (D222).

    They are the use's own sources, less `state_tools`, which names the tool bag rather than one
    read. A source the caller did not hand this judgement is simply not among its checks.
    """
    return tuple(name for name in sources_of(use) if name != "state_tools")


def count_judgement(result: JudgeResult, counts: Optional[dict] = None) -> dict:
    """Add one judgement to the round's judge counts (D222); every reader counts the same way."""
    counts = {name: int((counts or {}).get(name) or 0) for name in JUDGE_COUNTS}
    counts["judge_checks_prefilled"] += len(result.checks)
    counts["judge_extra_tool_calls"] += int(result.extra_tool_calls)
    counts["judge_refused_no_check"] += int(result.refused and result.reason == NO_CHECK)
    counts["judge_tool_choice_forced"] += int(result.tool_choice_forced)
    counts["judge_tool_choice_rejected"] += int(result.tool_choice_rejected)
    return counts


def checks_text(checks: list[dict]) -> str:
    """The checks section of the prompt: the tool, its arguments and what it answered (D222)."""
    if not checks:
        return "Checks already run for you: none. Say so and abstain."
    lines = ["Checks already run for you, before you were asked. These are the reads this question "
             "needs, and their results are what your verdict rests on:"]
    for check in checks:
        lines.append(f"- {check['tool']} {canonical_json(check.get('args') or {})} -> "
                     + _clamp(_render(check.get("result")), CHECK_RESULT_CHARS))
    return "\n".join(lines)


def _clamp(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[:limit] + " ..."


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
        tool_choice: Optional[str] = "required",
    ) -> None:
        self.model = model
        self.tools = dict(tools or {})
        self.verifier_output = verifier_output
        self.name = name or getattr(model, "name", "judge")
        self.persona = persona
        self.max_steps = max_steps
        self.judge_version = judge_version
        # What the first turn asks of the endpoint where a check is still the model's to choose
        # (D222 rule 2). None asks nothing, which is rule 1 on its own.
        self.tool_choice = tool_choice

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
        return self.ask("policy_atom", prompt, {"policy_rule": rule, "transcript": transcript})

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
        return self.ask("equivalence", prompt, {"value_a": a, "value_b": b})

    def judge_reference(self, reference_run: Any, intent: Any, verifier_output: Any = None) -> JudgeResult:
        """D57: is this recorded Run a good Reference, in Trust or Escalate framing (R27 8c)."""
        checked = verifier_output if verifier_output is not None else self.verifier_output
        prompt = (
            "Trust or escalate. Decide whether this recorded Run is a good Reference for its Task.\n\n"
            "Intent of the Task, the ground truth of what the user wanted by the end of the Run:\n"
            f"{_render(intent)}\n\n"
            f"Deterministic verifier output on this Run, already decided by code:\n{_render(checked)}\n\n"
            f"Candidate Reference Run:\n{_render(reference_run)}\n\n"
            "Grade the End state against the Intent, and only that. The transcript is not in evidence: "
            "whether the agent authenticated the user, asked for confirmation or followed a procedure "
            "cannot be seen through your tools and never decides the verdict. Judge the Intent, not the "
            "opening request; the user may have revised the opening request during the Run, and what the "
            "Intent says is what they wanted. "
            "good_reference when the End state is what the Intent asks for, bad_reference when the End "
            "state contradicts it, abstain when the evidence does not decide it. Judge only what code "
            "cannot check. A bad Reference sets a wrong bar for every later Verdict on this Task, so "
            "escalate with abstain rather than guess."
        )
        return self.ask("reference", prompt,
                         {"intent": intent, "verifier_output": checked, "end_state": reference_run})

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
        return self.ask("cause", prompt,
                         {"failed_run": failed_run, "reference_run": reference_run})

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
        return self.ask("dispute", prompt,
                         {"end_state": end_state, "required_set": required, "allowed_set": allowed})

    # --- the checks the harness runs (D222) ---

    def checks_for(self, use: str, material: dict) -> list[dict]:
        """The reads this question needs, run here, before the model is asked (D222).

        Two kinds, and both are reads the D92 rule wanted the model to make. The first is each
        source the use hands this judgement, read where a tool of that name exists and taken from
        the material itself where none does, since the material is what such a tool would answer
        with. The second is every tool this judge holds that needs no argument: a view that needs
        nothing chosen is one the harness can open on the model's behalf, and only a tool whose
        arguments still have to be chosen is left for rule 2 to force.
        """
        checks: list[dict] = []
        for name in check_sources(use):
            if name not in material:
                continue
            if name in self.tools:
                ran, result = self._run_tool(name, {})
                if not ran:
                    result = material[name]
            else:
                result = material[name]
            checks.append({"tool": name, "args": {}, "result": result})
        seen = {check["tool"] for check in checks}
        for name in self._tools_needing_no_argument():
            if name in seen:
                continue
            ran, result = self._run_tool(name, {})
            if ran:
                checks.append({"tool": name, "args": {}, "result": result})
        return checks

    def _tools_needing_no_argument(self) -> list[str]:
        """The judge's tools the harness can call for it: the ones with no argument left to choose."""
        out: list[str] = []
        for name, tool in self.tools.items():
            try:
                parameters = inspect.signature(tool).parameters.values()
            except (TypeError, ValueError):  # a builtin or a callable with no readable signature
                continue
            if all(p.default is not inspect.Parameter.empty
                   or p.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
                   for p in parameters):
                out.append(name)
        return out

    def _needs_a_choice(self) -> bool:
        """Whether a tool is left whose arguments only the model can choose (D222 rule 2)."""
        return bool(set(self.tools) - set(self._tools_needing_no_argument()))

    # --- the tool loop ---

    def ask(self, use: str, prompt: str, material: Optional[dict] = None) -> JudgeResult:
        """One judge use end to end: prefill the checks, ask the model, read the answer (D222).

        Public because it is the seam the five uses share and the one place the check-first rule
        lives; a caller with a question of its own asks here rather than reimplementing the loop.
        """
        verdicts, abstain = _USES[use]
        checks = self.checks_for(use, dict(material or {}))
        messages: list[dict] = [
            {"role": "system", "content": self._system(use, verdicts, abstain, checks)},
            {"role": "user", "content": prompt + "\n\n" + checks_text(checks)},
        ]
        tools_run: list[str] = []
        results: list[dict] = []
        forced = rejected = False
        for step in range(self.max_steps):
            want = self.tool_choice if step == 0 and self.tools and self._needs_a_choice() else None
            reply, refused_the_field = self._query(messages, want)
            forced = forced or (want is not None and not refused_the_field)
            rejected = rejected or refused_the_field
            if not reply.tool_calls:
                return self._finish(use, reply.content, tools_run, results, checks, forced, rejected)
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
        return self._result(use, abstain, tools_run, results, checks=checks, refused=True,
                            extra_tool_calls=len(tools_run), tool_choice_forced=forced,
                            tool_choice_rejected=rejected,
                            reason=f"no verdict within {self.max_steps} steps")

    def _query(self, messages: list[dict], want: Optional[str]) -> tuple[Any, bool]:
        """One model call, and whether the provider turned the tool_choice instruction down (D222).

        A provider that does not carry the field refuses the whole request rather than ignoring the
        field, so the refusal is caught here and the same question asked again without it. The
        checks are already in the prompt either way, so a provider without the parameter falls back
        to rule 1 alone and the fallback is recorded rather than passed over.
        """
        specs = self._tool_specs()
        plain = ModelConfig(temperature=0)
        if want is None:
            return self.model.query(messages, tools=specs, config=plain), False
        try:
            return self.model.query(messages, tools=specs,
                                    config=ModelConfig(temperature=0, tool_choice=want)), False
        except (ProviderError, TypeError, ValueError) as error:
            if isinstance(error, ProviderError) and (error.status or 400) >= 500:
                raise
            return self.model.query(messages, tools=specs, config=plain), True

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

    def _system(self, use: str, verdicts: tuple[str, ...], abstain: str,
                checks: Optional[list[dict]] = None) -> str:
        sources = sources_of(use)
        lines = [
            "You are one of two independent judges grading part of a recorded agent Run.",
            "Your tools are read-only views of the Task's Starting state and the Run's End state.",
            # D222: the check the question needs has already been run, so answering straight away is
            # right and is not a refusal. Whether you call a tool decides nothing about your verdict.
            f"The {len(checks or [])} check(s) this question needs have already been run for you and "
            "their results are in the message below. Read them and answer. You may call a tool for "
            "anything more you want, and answering with no further tool call is a complete answer.",
            "The sources you have for this question, and the only ones your verdict may rest on: "
            + ", ".join(sources) + ".",
        ]
        agreement = _AGREEMENT.get(use)
        if agreement:
            lines.append(
                f"A verdict of {agreement} names, in evidence, the check it rests on, by the name that "
                f"check is listed under. An {agreement} that names none is not a verdict: it is a "
                f"comparison nobody settled, and it is recorded as {abstain}."
            )
        if "transcript" not in sources:
            lines.append(
                "You do not have the transcript. What the agent said, whether it authenticated the user, "
                "whether it asked for and was given a confirmation, the order it called its tools in, and "
                "what the user asked for at the start of the Run are all outside your evidence."
            )
        lines += [
            "Answer with one JSON object and nothing else: "
            '{"verdict": one of ' + ", ".join(verdicts) + ', "cited_spans": [...], "evidence": [...], '
            '"sub_answers": [...], "flags": [...], "reason": "..."}.',
            "evidence names the sources your verdict rests on, each one taken from the list above.",
            f"A verdict that needs anything not on that list is not a verdict: answer {abstain}, and name "
            "what you would have needed in evidence.",
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

    def _finish(self, use: str, content: Optional[str], tools_run: list[str], results: list[dict],
                checks: Optional[list[dict]] = None, forced: bool = False,
                rejected: bool = False) -> JudgeResult:
        verdicts, abstain = _USES[use]
        checks = list(checks or [])
        extra = {"checks": checks, "extra_tool_calls": len(tools_run),
                 "tool_choice_forced": forced, "tool_choice_rejected": rejected}
        if not checks:
            # D222: the refusal is about this harness, not about the model. It fires where no check
            # could be prefilled for the question, which means the use hands its judge nothing to
            # read; a model that answers without calling a tool of its own is not refused any more.
            return self._result(use, abstain, tools_run, results, reason=NO_CHECK, refused=True, **extra)
        data = _parse_json(content)
        if data is None:
            return self._result(use, abstain, tools_run, results,
                                reason="the reply was not a JSON verdict object", **extra)
        cited = {
            "cited_spans": [str(s) for s in _as_list(data.get("cited_spans"))],
            "evidence": _names(data.get("evidence")),
            "sub_answers": [s for s in _as_list(data.get("sub_answers")) if isinstance(s, dict)],
        }
        missing = sources_not_given(cited["evidence"], sources_of(use))
        if missing:
            # D93: a ruling that rests on something this judge was not handed did not decide the
            # question, whatever verdict word came with it, so it abstains and a person sees it.
            return self._result(
                use, abstain, tools_run, results,
                reason="the verdict rests on " + ", ".join(missing) + ", which this judge was not given",
                **cited, **extra,
            )
        verdict = str(data.get("verdict") or "").strip()
        reason = data.get("reason")
        if use == "policy_atom":
            # R27 8b: the overall verdict is a fixed combination of the atomic answers, not the model's word.
            verdict, reason = _verdict_from_sub_answers(cited["sub_answers"], reason)
        elif use == "equivalence" and verdict == "equivalent" and _as_list(data.get("flags")):
            verdict, reason = abstain, "the judge flagged a number, unit, date or negation mismatch"
        if verdict not in verdicts:
            verdict, reason = abstain, f"unknown verdict {verdict!r}"
        elif verdict == _AGREEMENT.get(use) and not _cites_a_check(cited, checks):
            # D222 rule 3, the other half of D186: a failure already had to cite a key the states
            # differ on, and an agreement now has to cite the check it rests on. An agreement that
            # cites nothing is a comparison nobody settled, which D219 records as unresolved.
            verdict, reason = abstain, UNCITED
        return self._result(use, verdict, tools_run, results, reason=reason, **cited, **extra)

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


def judge_name(model_id: str, persona: str = "a") -> str:
    """`<provider/model>:<persona letter>`, which is how every ruling names the model it ran on (D160).

    Two judges may now be two different models, so a name that says only which persona spoke leaves
    the by-pair disagreement rate unable to say which two models parted.
    """
    return f"{model_id}:{persona}"


def third_judge(judge: AgenticJudge, persona: str = THIRD_PERSONA,
                name: Optional[str] = None) -> AgenticJudge:
    """D97's third sample: one of the two models again, under a different persona, same tools.

    `name` is for the caller that uses this to build the second judge out of one model (cli.py's
    default pair): it names that judge `<model>:b` rather than leaving it with the third sample's
    own name, which the tie-breaker would then collide with in the pair rows.
    """
    return AgenticJudge(judge.model, judge.tools, judge.verifier_output, name=name or f"{judge.name}#3",
                        persona=persona, max_steps=judge.max_steps, judge_version=judge.judge_version,
                        tool_choice=judge.tool_choice)


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
            # D222: both judges' checks and counters ride on the split, so a round counting the
            # pair counts what was actually run and not only what the judge that won ran.
            checks=first.checks + second.checks,
            extra_tool_calls=first.extra_tool_calls + second.extra_tool_calls,
            tool_choice_forced=first.tool_choice_forced or second.tool_choice_forced,
            tool_choice_rejected=first.tool_choice_rejected or second.tool_choice_rejected,
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
            # Which two judges these verdicts came from, by the name that carries their model id
            # (D160): without it no row on disk says which two models disagreed. `judges` is the
            # pair as the by-pair rate names it, written here so every reader of these rows counts
            # under the same name without formatting one of its own.
            "judge_a_name": first.judge,
            "judge_b_name": second.judge,
            "judges": pair_name(first.judge, second.judge),
            "disagreement": disagreement,
            "abstain": bool(reason) and not disagreement,
            "reason": reason,
        }
        if third is not None:
            row["verdict_c"] = third.verdict
            row["judge_c_name"] = third.judge
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


def pair_name(judge_a: str, judge_b: str) -> str:
    """`"<judge a> vs <judge b>"`: the name the by-pair disagreement rate counts this pair under."""
    return f"{judge_a} vs {judge_b}"


def by_pair(rows: Iterable[dict]) -> dict:
    """The pairs, disagreements and rate of each pair of judges over judge_pairs rows (D160).

    Rows written before D160 name no pair: they join none rather than being counted under an
    invented name, so an older build reads as having no by-pair numbers instead of wrong ones.
    """
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        name = str(row.get("judges") or "")
        if name:
            grouped.setdefault(name, []).append(row)
    return {name: {key: _pair_counts(group)[key] for key in ("pairs", "disagreements", "rate")}
            for name, group in sorted(grouped.items())}


def disagreement_stats(rows: Iterable[dict]) -> dict:
    """records.disagreement_stats over every row, plus `by_pair`: the same counts per pair of judges (D160).

    The two judges can now be two different models, and one rate over every pair cannot say which
    two of them parted; `by_pair` is that rate per pair, so two models disagreeing is a number a
    reader can point at.
    """
    rows = list(rows)
    return dict(_pair_counts(rows), by_pair=by_pair(rows))


def disagreement_rate(workdir: Path, use: Optional[str] = None) -> dict:
    """The number that travels with every judge result until human labels exist (D92)."""
    rows = [r for r in read_jsonl(Path(workdir) / PAIRS_FILE) if use is None or r.get("use") == use]
    return disagreement_stats(rows)


# --- small helpers ---


# --- the smoke pairs a relaunch picks its judge model by (D222 rule 4) ---

# Two invented pairs of an invented column, one pair the same and one pair not. They name no
# corpus and no customer, so the same two questions can be asked of any candidate judge model.
SMOKE_PAIRS: tuple[tuple[str, str, str, bool], ...] = (
    ("kiln.finish", "matte grey", "grey, matte", True),
    ("kiln.finish", "matte grey", "gloss white", False),
)


def smoke(judge: AgenticJudge) -> list[dict]:
    """Ask one judge the two smoke pairs: did it settle each, by which route, and was it right.

    The question a relaunch has to answer before it names a judge model is whether that model
    returns a verdict at all on a semantic pair, which until D222 depended on whether it happened
    to call a tool first. This asks it twice, in one command, and says which route each answer took.
    """
    return [_smoke_row(judge, column, a, b, same) for column, a, b, same in SMOKE_PAIRS]


def _smoke_row(judge: AgenticJudge, column: str, a: str, b: str, same: bool) -> dict:
    result = judge.judge_equivalence(column, a, b)
    settled = not result.refused and result.verdict in ("equivalent", "not_equivalent")
    return {
        "column": column, "a": a, "b": b,
        "expected": "equal" if same else "different",
        "verdict": result.verdict,
        "outcome": "resolved" if settled else "refused",
        "route": _smoke_route(result, settled),
        "checks": len(result.checks),
        "extra_tool_calls": int(result.extra_tool_calls),
        "agrees": bool(settled and (result.verdict == "equivalent") == same),
        "reason": result.reason,
    }


def _smoke_route(result: JudgeResult, settled: bool) -> str:
    """Why this pair ended where it did, in one word a relaunch can read."""
    if settled:
        return "judge"
    if result.reason == NO_CHECK:
        return "no_check"
    if result.reason == UNCITED:
        return "uncited"
    return "refused" if result.refused else "abstain"


def smoke_lines(rows: Iterable[dict]) -> list[str]:
    """One line per pair, and a last line saying whether this model can be a judge."""
    rows = list(rows)
    lines = [f"{row['column']} {row['a']!r} vs {row['b']!r}: {row['outcome']} ({row['route']}), "
             f"verdict {row['verdict']}, expected {row['expected']}, checks {row['checks']}, "
             f"extra tool calls {row['extra_tool_calls']}" for row in rows]
    resolved = sum(1 for row in rows if row["outcome"] == "resolved")
    agreed = sum(1 for row in rows if row["agrees"])
    lines.append(f"resolved {resolved} of {len(rows)}, right on {agreed} of {len(rows)}")
    return lines


def _cites_a_check(cited: dict, checks: list[dict]) -> bool:
    """Whether an agreeing ruling named one of the checks the harness ran for it (D222 rule 3).

    Named in evidence by the check's own name, or quoted in a cited span: both say which read the
    verdict rests on, and the judge is asked for the first. A ruling that names neither rests on
    nothing that was run, which is the state D219 calls unresolved rather than agreement.
    """
    names = {_source_word(check.get("tool")) for check in checks}
    if any(_source_word(name) in names for name in cited.get("evidence") or []):
        return True
    spans = " ".join(str(span) for span in cited.get("cited_spans") or []).lower()
    return any(name and name in spans for name in names)


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



