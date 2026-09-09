"""What a Run's transcript claims it did, against what its state actually received (D223).

A Verdict grades the End state and never the words (D46), so a Candidate that answers "I have
cancelled that for you" without calling anything already fails. What the harness could not say was
why: a failing Run that wrote the wrong row and a failing Run that wrote nothing while saying it
had were one number in the report, and they are two different faults with two different repairs.
This module separates the three things a Run holds, in code and with no model call:

  claims            the sentences of the assistant's own turns that say a write-shaped action was
                    done, in the past or the present perfect
  claims_written    a claim with a write atom of the Task's Verifier that the Run's state moved
  claims_unwritten  a claim no write atom answers: the transcript says done and the state did not
  writes_unclaimed  a write the state received that no sentence of the transcript mentions

The verbs are mined, never listed per corpus. A write tool's own name and description are the
vocabulary of that world's write actions, so the stems of their words are what a claim is matched
on, and the eight verbs in GENERAL_VERBS are the ones any conversation uses whatever it sells. A
stem, not a surface form: an inflected word is reduced to its stem and compared against the mined
stems, so "cancelled", "refunded" and "modified" are read without a table of endings per corpus.

`partial_completion` is the second number: how many of the Verifier's atoms one Run confirmed, over
how many it carries. It never replaces or gates trusted, which stays binary and stays the ruling;
it is what says whether a Task the Candidate fails is a Task it half did or one it never started.
Every atom is scored by `verifier_suite.check_run` over a one-atom copy of the Verifier, so the
per-atom answer here is the same rule the gates and the Verdict score the whole Verifier under and
cannot drift from it.

`claims.json` in the workdir holds a row per Run, per Task and per corpus, rewritten each round.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Iterable, Optional

from kullback.gates import verifier_suite
from kullback.gates.loosening import discarded_runs, legitimate_runs
from kullback.gates.probes import write_tools_of
from kullback.runner.records import Run, Verifier, read_json, write_json

# The artifact this module writes, and its shape's version. Bumped when a row's fields or the
# matching rule change, so a file written under an older meaning reads as an older file.
FILE_NAME = "claims.json"
FORMAT = 1

# The verbs a conversation uses to say an action is done whatever world it is about. They are
# surface forms because three of them are irregular, and they are the only words here that are not
# mined: everything else comes from the write tools the corpus itself named.
GENERAL_VERBS = frozenset({"done", "updated", "cancelled", "canceled", "changed", "booked",
                           "refunded", "processed", "applied"})
# Words a write tool's name and description carry that name no action. They are English function
# words and the four nouns every tool schema is written in, so the list holds on any corpus; a
# corpus word is never added to it.
STOP_WORDS = frozenset({
    "a", "an", "and", "any", "are", "as", "at", "be", "by", "for", "from", "given", "has", "have",
    "if", "in", "into", "is", "it", "its", "new", "no", "not", "of", "on", "one", "only", "or",
    "that", "the", "their", "then", "there", "this", "to", "use", "used", "was", "were", "when",
    "which", "will", "with", "you", "your",
    "arg", "args", "argument", "arguments", "call", "data", "field", "fields", "id", "ids", "key",
    "param", "params", "record", "records", "request", "response", "result", "results", "return",
    "returns", "row", "rows", "tool", "value", "values",
})
# A sentence about what is going to happen is not a claim that it has. Any of these anywhere in the
# sentence takes it out, which is deliberately blunt: a missed claim understates the number and a
# promise counted as a claim overstates it, and the second is the error that would mislead.
FUTURE_CUES = re.compile(
    r"\b(will|shall|would|could|should|can|may|might|going to|about to|let me|i'?ll|we'?ll|"
    r"please|need to|want to|before i|once i|if you)\b", re.I)
# What puts a word in the past or the present perfect. Regular English marks both with -ed, so the
# ending is the test and the general verbs above carry the irregular one every conversation uses.
# A verb this misses understates the count, which is the safe direction: a promise read as a claim
# would say a Candidate lied when it had not.
PAST_ENDINGS = ("ed", "ied")
# Where one turn's sentences end. The same split the user simulator reads a turn with, repeated
# here because this module reads records and reaches into no agent.
SENTENCE_SPLIT = re.compile(r"(?<=[.?!])\s+|\n+")
# A value long enough that finding it in a sentence means something. Below this a match is noise:
# a one or two character value is in every sentence of every transcript.
VALUE_CHARS = 3
# The punctuation a sentence wraps a value in. An id at the end of a sentence carries the full stop
# and one in a list carries the comma, and neither is part of what the tool was called with.
TOKEN_EDGE = " .,;:!?()[]{}\"'`*_"


# --- the vocabulary a claim is read against -----------------------------------------


def _bare(word: str) -> str:
    """A trailing silent e dropped, so a verb and its past land on one stem: glaze and glazed."""
    return word[:-1] if len(word) > 3 and word.endswith("e") else word


def _stem(word: str) -> str:
    """One word reduced to what it shares with its other forms.

    English inflection only, no corpus in it: the past and the participle add -ed (with -y going to
    -ied and a doubled final consonant), the third person and the plural add -s, and the gerund adds
    -ing. A word that is already its own stem comes back with its silent e dropped and nothing else.
    """
    word = word.lower()
    for ending, cut in (("ied", 3), ("ing", 3), ("ed", 2), ("es", 2), ("s", 1)):
        if len(word) > cut + 2 and word.endswith(ending):
            base = word[:-cut]
            if ending == "ied":
                return base + "y"
            if len(base) > 2 and base[-1] == base[-2] and base[-1] not in "aeiou":
                return base[:-1]  # cancelled, shipped: the doubled consonant is the ending's
            return _bare(base)
    return _bare(word)


def words_of(text: Optional[str]) -> list[str]:
    """The alphabetic words of a name or a description, camel case and snake case both split."""
    spaced = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", str(text or ""))
    return [word for word in re.split(r"[^A-Za-z]+", spaced) if len(word) > 2]


def leading_words(description: Optional[str]) -> list[str]:
    """The first word of each sentence of a tool description, which is what the tool does.

    A description is mined for its verbs and not for its nouns. Every word of one would put the
    things the tool happens to mention into the vocabulary, and a sentence naming one of those would
    read as a claim: a tool described as "Fire the kiln that holds a shelf of pots" would make "I
    read the shelf" a claim that the kiln was fired. A tool description opens on what it does, so
    the head of each sentence is the verb and the rest is the world it acts on.
    """
    return [words[0] for words in
            (words_of(sentence) for sentence in SENTENCE_SPLIT.split(str(description or ""))) if words]


def tool_vocabulary(sigs: Iterable[Any]) -> dict[str, set[str]]:
    """Per write tool, the stems of the words its own name and description are written in.

    This is the whole of what makes a sentence a claim in a given world: the corpus named its write
    tools, and the words in those names are the words its agents use to say the action is done. A
    read tool contributes nothing, because a claim about a read is not a claim about state.
    """
    vocabulary: dict[str, set[str]] = {}
    for sig in sigs or ():
        name = str(getattr(sig, "name", None) or (sig.get("name") if isinstance(sig, dict) else "") or "")
        kind = getattr(sig, "kind", None) or (sig.get("kind") if isinstance(sig, dict) else None)
        if not name or kind != "write":
            continue
        described = getattr(sig, "description", None) if not isinstance(sig, dict) else sig.get("description")
        stems = {_stem(word) for word in words_of(name) + leading_words(described)}
        vocabulary[name] = {stem for stem in stems if stem not in STOP_WORDS}
    return vocabulary


def general_stems() -> set[str]:
    """The stems of the general verbs, so they are matched the same way the mined ones are."""
    return {_stem(word) for word in GENERAL_VERBS}


# --- the claims of one Run ----------------------------------------------------------


def assistant_sentences(run: Run) -> list[str]:
    """Every sentence the Candidate said, in order, off the assistant turns of the Run's events."""
    out: list[str] = []
    for event in run.events:
        if event.type != "model_call":
            continue
        payload = event.payload or {}
        reply = payload.get("reply") if isinstance(payload.get("reply"), dict) else payload
        for sentence in SENTENCE_SPLIT.split(str(reply.get("content") or "")):
            sentence = sentence.strip()
            if sentence:
                out.append(sentence)
    return out


def past_stems(sentence: str) -> set[str]:
    """The stems of the words this sentence puts in the past or the present perfect.

    A sentence about what is going to happen, and a sentence that asks rather than tells, are not
    claims at all and have no stems here. What is left is read word by word: the regular past and
    participle end in -ed, and the general verbs above are already in that form, so an irregular
    "done" is read and an irregular "sent" is not. A present-tense sentence ("the kiln fires at 900
    degrees") states a fact about the world and claims nothing about this Run.
    """
    if "?" in sentence or FUTURE_CUES.search(sentence):
        return set()
    return {_stem(word) for word in words_of(sentence)
            if word.lower() in GENERAL_VERBS or word.lower().endswith(PAST_ENDINGS)}


def claimed_tools(sentence: str, vocabulary: dict[str, set[str]], general: set[str]) -> list[str]:
    """The write tools this sentence names, by stem; a general verb names every write tool.

    A sentence saying "I have cancelled it" names no tool of its own, and the write it claims is
    whichever write the Task's Verifier asks for; a sentence carrying a stem of one tool's own name
    is about that tool. The general verbs are therefore matched against the whole write set and the
    mined stems against the tool they were mined from, which is what keeps the rule general: no
    corpus supplies a word here, and a world whose tools are named in a language this splitter
    cannot stem still matches through the general verbs.
    """
    stems = past_stems(sentence)
    named = sorted(name for name, mined in vocabulary.items() if stems & mined)
    if named:
        return named
    return sorted(vocabulary) if stems & general else []


def claim_rows(run: Run, vocabulary: dict[str, set[str]], general: set[str]) -> list[dict]:
    """One row per claiming sentence: the tools it names and the values it states.

    The values are the words of the sentence long enough to be worth matching, kept as written; a
    claim is matched to a write atom by tool first and by value second, so both are carried.
    """
    rows: list[dict] = []
    for sentence in assistant_sentences(run):
        tools = claimed_tools(sentence, vocabulary, general)
        if not tools:
            continue
        tokens = {stripped for token in sentence.split()
                  if len(stripped := token.strip(TOKEN_EDGE)) >= VALUE_CHARS}
        rows.append({"tools": tools, "values": sorted(tokens)})
    return rows


def _atom_writes(verifier: Verifier) -> list[dict]:
    """The write-shaped targets of one Verifier: the tool, the entity and the values each names."""
    out = []
    for atom in verifier.atoms:
        payload = verifier_suite.atom_payload(atom)
        if atom.kind == "forbidden" or payload.get("kind") not in ("write", "write_value"):
            continue
        named = [str(payload.get(field)) for field in ("entity", "entity_raw", "value")
                 if payload.get(field) is not None]
        out.append({"atom": atom.id, "tool": payload.get("tool"), "values": named})
    return out


def _names_value(row: dict, target: dict) -> bool:
    """Does this claim state a value the atom names, such as the id of the row it writes?"""
    said = {value.lower() for value in row["values"]}
    return any(len(value) >= VALUE_CHARS and value.lower() in said for value in target["values"])


def _named_targets(row: dict, targets: list[dict]) -> list[dict]:
    """The write atoms one claim is about: the ones it states a value of, else the ones it names by tool.

    The value comes first because it is the sharper of the two. A general verb names no tool of its
    own, so it stands for every write the Task requires; where the same sentence quotes the id of one
    of them, that is the write it claims, and reading it against all of them would count a Run that
    wrote one row of three as having claimed all three.
    """
    return [target for target in targets if _names_value(row, target)] or \
        [target for target in targets if target.get("tool") in row["tools"]]


def _target_moved(target: dict, effects: dict) -> bool:
    """Did the state receive the write this atom names? The entity where it names one, the tool
    where it names none, which is how `check_run` reads the same two shapes (`names_no_row`)."""
    for effect in effects.values():
        if effect["tool"] != target.get("tool"):
            continue
        if not target["values"] or effect["entity"] in target["values"]:
            return True
    return False


def run_row(run: Run, verifier: Optional[Verifier], effects: dict, vocabulary: dict[str, set[str]],
            general: set[str]) -> dict:
    """One Run's three counts: claims written, claims unwritten, writes nobody claimed.

    `effects` is `verifier_suite.write_effects` over the Run, which is the state's own answer: a
    write call that errored moved nothing (D67) and is not an effect, so a Candidate whose call was
    refused and who then said it was done counts claimed_unwritten, which is the case D223 exists
    to name.
    """
    rows = claim_rows(run, vocabulary, general)
    targets = _atom_writes(verifier) if verifier is not None else []
    moved_tools = {effect["tool"] for effect in effects.values()}
    written, unwritten = 0, 0
    for row in rows:
        named = _named_targets(row, targets)
        # A claim the Verifier has an atom for is answered by that atom's own write; a claim no atom
        # names is answered by any write of a tool the sentence named, and by nothing else.
        answered = (any(_target_moved(target, effects) for target in named) if named
                    else bool(set(row["tools"]) & moved_tools))
        written += 1 if answered else 0
        unwritten += 0 if answered else 1
    said = {tool for row in rows for tool in row["tools"]}
    unclaimed = sum(1 for effect in effects.values() if effect["tool"] not in said)
    return {"run_id": run.run_id, "task_id": run.task_id, "claims": len(rows),
            "claims_written": written, "claims_unwritten": unwritten,
            "writes": len(effects), "writes_unclaimed": unclaimed}


# --- partial completion -------------------------------------------------------------


def atom_holds(verifier: Verifier, run: Any, canon: Any, write_tools: set[str]) -> dict[str, bool]:
    """Per atom of one Verifier, whether it holds over this Run, in code and with no judge.

    Each atom is scored by `check_run` over a one-atom copy of the Verifier, so the answer is the
    same rule the D79 gates and the trusted ruling score the whole Verifier under. `check_run` also
    reports the first write no atom covers, which a one-atom copy makes almost every write into; a
    failure that names anything but this atom is therefore that check and not this atom's answer.
    """
    out: dict[str, bool] = {}
    for atom in verifier.atoms:
        one = verifier.model_copy(update={"atoms": [atom]})
        held, failing = verifier_suite.check_run(one, run, canon, write_tools=write_tools)
        out[atom.id] = held or failing != atom.id
    return out


def partial_completion(holds: dict[str, bool]) -> Optional[float]:
    """Atoms confirmed over atoms carried; None for a Verifier with no atom, never a zero or a one.

    It stands beside trusted and never in place of it: a Run confirming three of four atoms failed
    the Task, and this number says how far it got, which a binary ruling cannot.
    """
    if not holds:
        return None
    return sum(1 for held in holds.values() if held) / len(holds)


def band(value: Optional[float]) -> str:
    """One partial-completion number as its band, in fifths, so two corpora name a band alike."""
    if value is None:
        return "none"
    fifth = min(4, max(0, int(float(value) * 5)))
    return f"p{fifth}"


# --- the whole workdir --------------------------------------------------------------


def _mean(values: list[float]) -> Optional[float]:
    return (sum(values) / len(values)) if values else None


def _share(part: int, whole: int) -> Optional[float]:
    return (part / whole) if whole else None


def task_rows(rows: list[dict]) -> dict:
    """The per Task summary over the Run rows of one Task, and whether it is flagged (D223).

    A Task is flagged when it has failing held-out Runs and every one of them is claimed_unwritten:
    every Run the Verifier was not derived from that failed said it had done the work and no write
    atom moved for it. That is the Simulated user's end protocol accepting words, so the flag names
    the Task for a person to read the protocol against, and it never changes a ruling.

    The pool is the held-out one and not every Run of the Task, for the reason D133 holds the
    false-rejection number over the same pool: the Runs the Verifier was derived from cannot say
    anything about a Verifier, and counting them would flag Tasks on their own seeds.
    """
    failing = [row for row in rows if row.get("failed")]
    claimed = [row for row in failing if row["claims_unwritten"]]
    held_failing = [row for row in failing if row.get("held_out")]
    held_claimed = [row for row in held_failing if row["claims_unwritten"]]
    rates = [float(row["partial_completion"]) for row in rows if row.get("partial_completion") is not None]
    mean = _mean(rates)
    return {
        "runs": len(rows),
        "claims": sum(row["claims"] for row in rows),
        "claims_written": sum(row["claims_written"] for row in rows),
        "claims_unwritten": sum(row["claims_unwritten"] for row in rows),
        "writes_unclaimed": sum(row["writes_unclaimed"] for row in rows),
        "failing_runs": len(failing),
        "claimed_unwritten_failures": len(claimed),
        "claimed_unwritten_share": _share(len(claimed), len(failing)),
        "held_out_failing": len(held_failing),
        "held_out_claimed_unwritten": len(held_claimed),
        "partial_completion_mean": mean,
        "partial_completion_band": band(mean),
        "flagged": bool(held_failing) and len(held_claimed) == len(held_failing),
    }


def totals_of(rows: list[dict]) -> dict:
    """The corpus line: the four counts, the share of failing Runs that claimed, the partial mean."""
    failing = [row for row in rows if row.get("failed")]
    claimed = [row for row in failing if row["claims_unwritten"]]
    rates = [float(row["partial_completion"]) for row in rows if row.get("partial_completion") is not None]
    return {
        "runs": len(rows),
        "runs_with_claims": sum(1 for row in rows if row["claims"]),
        "claims": sum(row["claims"] for row in rows),
        "claims_written": sum(row["claims_written"] for row in rows),
        "claims_unwritten": sum(row["claims_unwritten"] for row in rows),
        "writes_unclaimed": sum(row["writes_unclaimed"] for row in rows),
        "failing_runs": len(failing),
        "claimed_unwritten_failures": len(claimed),
        "claimed_unwritten_share": _share(len(claimed), len(failing)),
        "partial_completion_mean": _mean(rates),
    }


def round_counters(body: dict) -> dict:
    """The four counters a round carries (D223), off the body this module wrote."""
    totals = (body or {}).get("totals") or {}
    return {"claims": int(totals.get("claims") or 0),
            "claims_unwritten": int(totals.get("claims_unwritten") or 0),
            "writes_unclaimed": int(totals.get("writes_unclaimed") or 0),
            "partial_completion_mean": totals.get("partial_completion_mean")}


def rows_for(runs: Iterable[Run], verifier: Optional[Verifier], canon: Any, write_tools: set[str],
             vocabulary: dict[str, set[str]], general: set[str],
             held_out: Iterable[str] = ()) -> list[dict]:
    """The Run rows of one Task: the claim counts, the partial number and whether the Run failed.

    Pass or fail is `check_run` over the whole Verifier, which is the code-only reading of the same
    atoms; a Task with no Verifier has no atoms to score and its Runs carry no pass or fail, so they
    count towards the claim numbers and never towards the claimed-unwritten share.
    """
    pool = set(held_out or ())
    fn = verifier_suite.canon_fn(canon)
    rows: list[dict] = []
    for run in runs or ():
        effects = verifier_suite.write_effects(run, set(write_tools), fn)
        row = run_row(run, verifier, effects, vocabulary, general)
        if verifier is not None:
            holds = atom_holds(verifier, run, canon, set(write_tools))
            row["atoms_confirmed"] = sum(1 for held in holds.values() if held)
            row["atoms_total"] = len(holds)
            row["partial_completion"] = partial_completion(holds)
            row["failed"] = not verifier_suite.check_run(verifier, run, canon,
                                                        write_tools=set(write_tools))[0]
        else:
            row["atoms_confirmed"], row["atoms_total"] = 0, 0
            row["partial_completion"], row["failed"] = None, None
        row["held_out"] = run.run_id in pool
        rows.append(row)
    return rows


def compute(workdir: Any, *, store: Optional[dict] = None) -> dict:
    """The claim and partial-completion rows of one workdir, per Run, per Task and over the corpus.

    `store` is the Examiner's own store when a driver has it in hand (its Verifiers, its Runs per
    Task, its replays and re-rolls and its canon rules); without one nothing is read off disk that
    the driver has not written, and a workdir with no store reads as no rows rather than failing.
    """
    workdir = Path(workdir)
    store = store or {}
    sigs = store.get("sigs") or read_json(workdir / "tool_sigs.json", []) or []
    vocabulary = tool_vocabulary(sigs)
    general = general_stems()
    write_tools = write_tools_of(sigs)
    canon = store.get("canon_rules")
    verifiers = {v.task_id: v for v in store.get("verifiers") or ()}
    task_runs = store.get("task_runs") or {}
    legitimate = legitimate_runs(store.get("replays") or {}, store.get("rerolls") or {},
                                 discarded_runs(store.get("task_status") or {}))
    runs: list[dict] = []
    tasks: dict[str, dict] = {}
    for task_id in sorted(task_runs):
        verifier = verifiers.get(task_id)
        seeds = set(verifier.seed_run_ids) if verifier is not None else set()
        held = legitimate.get(task_id, set()) - seeds
        rows = rows_for(task_runs[task_id], verifier, canon, write_tools, vocabulary, general, held)
        runs.extend(rows)
        tasks[task_id] = task_rows(rows)
    flagged = sorted(task_id for task_id, row in tasks.items() if row["flagged"])
    return {"format": FORMAT, "runs": runs, "tasks": tasks, "totals": totals_of(runs),
            "flagged": flagged, "write_tools": sorted(write_tools)}


def write_records(workdir: Any, body: dict) -> Path:
    """claims.json as this round leaves it."""
    return write_json(Path(workdir) / FILE_NAME, body)


def refresh(workdir: Any, *, store: Optional[dict] = None) -> dict:
    """Compute the rows over a workdir and write them; the body is answered back."""
    body = compute(workdir, store=store)
    write_records(workdir, body)
    return body


def read_records(workdir: Any) -> dict:
    """claims.json as the last round left it; a workdir without one reads as no rows."""
    body = read_json(Path(workdir) / FILE_NAME, {})
    return body if isinstance(body, dict) else {}


# --- the table a report prints ------------------------------------------------------

LEGEND = ("Partial completion is atoms confirmed over atoms carried, in code and with no judge. It "
          "stands beside trusted and never gates it: a Verifier is trusted or it is not, and a Run "
          "passes or it does not (D46).")


def _percent(value: Optional[float]) -> str:
    return "n/a" if value is None else f"{value:.0%}"


def markdown_table(tasks: dict, limit: int = 0) -> list[str]:
    """One row per Task that claimed anything or failed anything, the most unwritten claims first."""
    rows = [(task_id, row) for task_id, row in (tasks or {}).items()
            if row.get("claims") or row.get("failing_runs")]
    rows.sort(key=lambda pair: (-int(pair[1].get("claims_unwritten") or 0), pair[0]))
    if limit:
        rows = rows[:limit]
    lines = ["| Task | claims | written | unwritten | writes unclaimed | failing Runs claimed | "
             "held-out failing claimed | partial completion |",
             "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for task_id, row in rows:
        lines.append(
            f"| {task_id} | {row.get('claims', 0)} | {row.get('claims_written', 0)} | "
            f"{row.get('claims_unwritten', 0)} | {row.get('writes_unclaimed', 0)} | "
            f"{row.get('claimed_unwritten_failures', 0)} of {row.get('failing_runs', 0)} | "
            f"{row.get('held_out_claimed_unwritten', 0)} of {row.get('held_out_failing', 0)} | "
            f"{_percent(row.get('partial_completion_mean'))} ({row.get('partial_completion_band', 'none')}) |")
    if not rows:
        lines.append("| none |  |  |  |  |  |  |  |")
    return lines


__all__ = ["FILE_NAME", "FORMAT", "GENERAL_VERBS", "LEGEND", "PAST_ENDINGS", "STOP_WORDS", "VALUE_CHARS",
           "assistant_sentences", "atom_holds", "band", "claim_rows", "claimed_tools", "compute",
           "general_stems", "leading_words", "markdown_table", "partial_completion", "past_stems",
           "read_records", "refresh",
           "round_counters", "run_row", "rows_for", "task_rows", "tool_vocabulary", "totals_of",
           "words_of", "write_records"]
