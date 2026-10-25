# Decisions

The decisions that govern the harness now, one entry each, by area; retired numbers are indexed at the end so citations resolve.
A decision changes by a new D number: the old entry moves to Retired with "superseded by D<n>"; numbers are never reused; next free: D443.
The full history is docs/decision-log.md in git, last held at commit 8c6a2d00cf7a407f65b770a0fed1fcfb4eb8b646.

## World and builder

D6 (2026-08-24). No Environment, no Verdict: the Environment is built from traces, tools are code over explicit state. Where: docs only.

D13 (2026-08-24). The Starting state is reconstructed by inverse replay: recorded reads first, then the recorded writes applied. Where: docs only.

D14 (2026-08-24). The system prompt and the policy compile into Hard constraints, checked on the Reference first. Where: docs only.

D16 (2026-08-24). Intake filtering is keep and tag: failed, retried, truncated, orphaned and chit-chat Runs stay, tagged; retries take the last
attempt; duplicates collapse with a weight. Where: docs only.

D29 (2026-08-26). Every hole in a traces-only Environment is listed with its fill; the assisted share per tool decides which fills are needed.
Where: docs only.

D33 (2026-08-26). The Environment is built from the whole trace corpus, not one Run; every entity any Run read is a row. Where:
kullback/builder/compile_env.py.

D35 (2026-08-26). Environment data follows customer trust: traces first, then schema and policy, a database snapshot last (ADR-0006). Where: docs
only.

D40 (2026-08-26). Rows the traces did not show are synthesised from the observed rows' shape, tagged, and a Run that reads one is assisted. Where:
kullback/builder/compile_env.py.

D66 (2026-08-26). Raw traces are the source of truth, stored byte for byte and content-hashed; the normalised Trace is derived, re-derivable, and
points back to its raw location; part superseded by D365 (the trace is one input among others). Where: kullback/runner/records.py.

D67 (2026-08-26). Tool errors are normalised at ingest into a small fixed taxonomy plus the verbatim payload, with classified_by recorded. Where:
kullback/builder/ingest.py.

D68 (2026-08-26). A tool's read or write class is proposed by a model from the observed context; computed effect evidence overrides the model.
Where: kullback/builder/mine.py.

D70 (2026-08-27). A tool with no effect evidence and low confidence defaults to read and carries unclassified: true. Where:
kullback/builder/mine.py.

D71 (2026-08-27). A user-side write enters the End state only when the traces record it (provisional; widened by D176). Where: kullback/derive.py.

D72 (2026-08-27). A tool's result and argument schemas are the union of every observed field, each with counts and a declared flag. Where:
kullback/builder/mine.py.

D73 (2026-08-27). Column class: code proposes by rule with its evidence, a model confirms or overrides per column, re-run evidence overrides both;
part superseded by D99. Where: kullback/builder/mine.py.

D74 (2026-08-27). One world per corpus with each Task's starting state as an overlay of the rows its own Runs saw; the overlay lookup lives in
route, not in tool bodies; an overlay conflict fails the export only. Where: kullback/builder/compile_env.py.

D95 (2026-08-27). A truncated tool result is tagged, the rest is asked for, and a reconstruction is served meanwhile; a Run that read one is
assisted. Where: kullback/builder/ingest.py.

D98 (2026-08-28). A tool is generic by a name rule (calculate, think, hand-off to a human and the like) at medium confidence, and a generic tool is
never credited with an observed effect. Where: docs only.

D99 (2026-08-28). Only system timestamp and counter names are exempt at high confidence; other date-like or version-like names are hard at low
confidence. Where: docs only.

D100 (2026-08-28). Clustering similarity is idf-weighted token Jaccard with argument keys dropped, complete linkage, default threshold 0.4. Where:
docs only.

D101 (2026-08-29). A tool's kind comes from what its calls show (a later read shows the change; the answer echoes what was sent; message answers
naming a row), never from a verb list. Where: docs only.

D102 (2026-08-29). An id is a column some call passes as an argument and whose value is distinct across the rows of one result. Where: docs only.

D103 (2026-08-29). A tool is about the noun before the first preposition in its name; ties break by entity id, then distinct ids, then the only id.
Where: docs only.

D104 (2026-08-29). Mined id patterns are read under either the table key or the table.column key, and the guard rejects an argument that is not an
id of that table. Where: docs only.

D106 (2026-08-29). A miss with signal in the traces is mined, not prompted around: nested rows are homed in their parent, the shared error prefix is
read from the corpus. Where: scripts/env_fidelity.py.

D107 (2026-08-29). The Starting state can grow to a chosen size by structural rules mined from observed rows (templates, identities, foreign keys,
back references, homes, sums), no model; observed rows are never edited, grown rows are checked (uniqueness, key closure, distributions) and tagged
synthetic; off by default. Where: kullback/builder/compile_env.py.

D114 (2026-08-29). The Environment refuses what the real system refuses: a write body handed an unknown reference value must raise
(gate_refuses_unknown); a body constant where the recording was constant passes (recorded_constant); a call after a write on the same value is not
gate evidence (after_write_calls); MISCOMPILED_SHARE is 0.02; a scorecard covering no Task fails. Where: kullback/examiner/reference.py.

D147 (2026-09-06). The compiler prompt says a row is read by attribute and what a column holds is a plain list or dict read by key; an attempt that
raises what the last one raised is told so. Where: docs only.

D148 (2026-09-06). Arithmetic in a tool body goes through the code-owned evaluate_arithmetic (stdlib ast, capped, fixed decimal context); eval,
exec, compile and ast stay refused to bodies. Where: docs only.

D155 (2026-09-06). When several recorded calls agree with each other and part from the tool's description, the recording is the standard: the body
reproduces it and the part is recorded as a finding for the customer. Where: kullback/builder/domain_tools.py.

D162 (2026-09-07). A tool body may not memorise the recordings: memorised_values refuses a literal with a mined id shape, a Starting-state row id,
or a recorded argument value the description does not name. Where: kullback/gates/tool_runs.py.

D164 (2026-09-07). A ToolSig carries callers and refused_callers; a name no caller was answered on, or that is not a Python identifier, gets no
ToolSig and is listed in unknown_tools; the Router refuses a non-caller with tool_not_found. Where: kullback/builder/mine.py.

D165 (2026-09-07). The non_trivial gate passes a tool whose recorded answers differ under identical arguments as state_driven, and replay fidelity
judges it in sequence. Where: kullback/gates/tool_runs.py.

D167 (2026-09-07). An id shape is a pattern no ordinary word matches: mixed positions get a character class, and a pattern accepting a plain or
capitalised word of length 1 to 24 is shapeless. Where: kullback/builder/mine.py.

D176 (2026-09-07). A requestor's prose results are read into rows of the one world by model-written readers (read(result) to dict or None) under a
free gate; such rows carry revealed_by and are keyed per Task by requestor. Where: kullback/gates/tool_runs.py.

D179 (2026-09-07). A row whose identity is more than one column is keyed by composite_keys: hard columns that are also arguments of the call and
differ between conflicting sightings, intersected over the corpus; a partial sighting folds into matching keyed rows and never invents a key part.
Where: tests/builder/test_compile_env.py.

D180 (2026-09-07). A row is homed in the entity whose id the call passed (row_homes.json); revealed columns take the D73 classes; a numeric column
that never repeats a value is exempt at low confidence; only hard columns split a Task's world. Where: kullback/builder/compile_env.py.

D183 (2026-09-07). A prose result has a model-written render, the inverse of its reader, held to every recorded result by round trip; a no-argument
tool whose results never change is a world constant; ids nested at any depth of the arguments are referenced. Where: kullback/builder/templates.py.

D184 (2026-09-08). On a recompile the previous body is replayed under the current world and gates and competes on equal terms; a tie keeps it; a
body that ran nothing is no candidate; the gates recorded are the released body's. Where: kullback/builder/lesson.py.

D188 (2026-09-08). The pinner reads a result as a tree (every dict at any depth tested against the schema, eight levels at most), a row stated on
its own is described by those sightings alone, and a Task's overlay is pinned column by column from its earliest sighting (overlay_pins.json).
Where: kullback/builder/compile_env.py.

D191 (2026-09-08). A narrowed repair ruling carries the score pair and the released body; calls the last Reference replay failed on are evidence for
that tool whatever the filters say; recompiles that score no higher count toward "stalled". Where: kullback/builder/replay_evidence.py.

D195 (2026-09-08). The sensitivity ruling: where two Task worlds hold a column differently and the recordings part in it, the body must part too;
pairs on unread prose are reported as unread_pairs. Where: kullback/builder/sandbox.py.

D200 (2026-09-08). A Task id is content-addressed over the sorted Run ids it groups; once a workdir holds a frozen Task list that list is
authoritative, new Runs may form new Tasks, and a frozen Task the rebuild would drop is kept as frozen_only. Where: kullback/builder/cluster.py.

D201 (2026-09-08). Every acting repair is a transaction: the lights of the Tasks it can touch (replay, Reference, suite, trusted) are read before
and after, and it lands only if its target moved up and no Task lost a light; otherwise its files are restored byte for byte and the ruling reads
reverted or no effect. Where: kullback/examiner/exam_files.py.

D202 (2026-09-08). A column a Task only ever writes is pinned by running the released write body backwards over the corpus's values for that column
(at most sixteen candidates), keeping the first that reproduces the recorded result; the pin is recorded as an assumption. Where:
kullback/builder/compile_env.py.

D203 (2026-09-08). A tool whose homed prose results no reader answers gets one: results align into templates whose slots bind to row columns, kept
only if the round trip reproduces every result; one model call per tool where a slot binds to nothing; unread results are counted and filed as a
finding. Where: kullback/examiner/findings.py.

D211 (2026-09-09). A stalled body changes strategy: the lesson carries every differing leaf, a relation found across the failing calls (broadcast,
shared value, refusal predicate, arithmetic, order) and the lines no recorded call reached; past six unbeaten recompiles the ask becomes a rewrite.
Where: kullback/builder/compile_env.py.

D213 (2026-09-09). When a Task's own Runs disagree on a column, the column leaves the Task overlay and each Run replays against a layer holding its
own earliest value; a Run with no sighting, and any Candidate Run, gets the Reference Run's layer; a disagreement on a written row flags a split
candidate. Where: kullback/builder/compile_env.py.

D216 (2026-09-09). The split of Runs into Tasks is a function of the recordings alone: a row sighting's version is the canonical recorded result
projected to that row under the tool that answered; sightings through different tools are never compared; the clustering records a fingerprint and
raises when it moves with no input moved. Where: kullback/builder/cluster.py.

D220 (2026-09-09). A Builder stage is handed ctx.evidence (seed Runs, Tasks without held-out members); raw Traces raise AnchorLeak; stages needing
every Run are declared in ALL_RUNS_STAGES. A value only held-out Runs witnessed stays in the world, is masked for the body writer, refused as a
literal (holdout_value) and counted (answered_from_holdout). Where: kullback/builder/compile_env.py.

D233 (2026-09-10). Caller-side state that states columns of the table mined for its requestor is homed to that table, one row per requestor,
earliest wins; the split compares it per hard column, semantic and exempt columns left out; prose sightings and requestors with no mined table keep
the opaque sentence key, and the cluster reason says which case applied. Where: kullback/builder/compile_env.py.

D234 (2026-09-10). An ambiguous effect column is credited to a write only where its arguments name the row or the same tool was seen moving that
column alone; otherwise it is filed once, unattributed and unchecked, and never shown to a writer's prompt; a single-write span is credited as
before. Where: kullback/builder/effects.py.

D250 (2026-09-11). Compile hold-out is by argument shape (sorted key paths, nested keys, list lengths, values dropped); where every call of a tool
shares one shape the split falls back to every third call and records no_shape_holdout; part superseded by D334 (test_body removed). Where:
kullback/builder/compile_env.py.

D257 (2026-09-18). A transition gate at build time compares the state change a body made for each recorded write with the change later reads in the
same recording witness; where none does it rules unwitnessed, published per tool; D215 stays the replay safety net. Where:
kullback/builder/sandbox.py.

D260 (2026-09-18). Recordings are grouped by what the customer asked for and the starting world only; the write set is a label; a path outlier stays
in the Task and its held-out pool but never becomes the Reference or feeds the Intent; the grouping tightness is one fixed constant. Where:
kullback/builder/cluster.py.

D261 (2026-09-18). One intake seam: an adapter detects on positive evidence or returns unknown, iterates and emits a versioned internal record with
a source pointer; an adapter is labelled proven on a build where a real public corpus exists, else conformant. Where: kullback/builder/sources.

D262 (2026-09-18). A tool that can be run for real is run for real in a sandbox with no outside network (a Real tool), otherwise imitated from
recordings; preference order real, compiled code, recording, model stand-in; for a terminal corpus the World is a container; the grader runs in a
fresh container with tests mounted only there; every Task must defeat a code cheat suite and the do-nothing Run must fail. Where:
kullback/runner/real_tools.py.

D263 (2026-09-18). Intake rules on each recording alone: admitted, set aside with a reason (no result, no starting state, not a recording,
duplicate, suspected cheat) or rejected; the build stops only under a declared floor; a rescue step may repair what was set aside; Task-eligible
versus evidence-only standing; a protected-path hash change marks a Run tampered and fails the Verdict. Where: kullback/builder/ingest.py.

D271 (2026-09-18). datetime, time, random and uuid leave a compiled body's allowed imports; every body receives one context carrying now, a random
source and new-id-for-table, fed from the recording in replay and the Run's seed off it. Where: kullback/gates/confinement.py.

D277 (2026-09-18). Every mined fact carries its supporting calls; a name guess is a starting assumption evidence can overrule, a source-declared
kind outranks both, and facts resting on a name alone are counted per corpus. Where: docs only.

D282 (2026-09-24). A non-write tool that answers with a scalar, is called at least three times and ends at least nine in ten of its Traces is an
ending tool: record_ending_actions adds an actions table and makes it a write, so a write atom can demand the hand-off; part narrowed by D288.
Where: kullback/builder/compile_env.py.

D283 (2026-09-24). Every Run has one world clock read off the recording (runner/world/clock.py), set on the tool context at every reset and served
by self.ctx.now(); a body that imports a clock module or reads another clock is refused with its line. Where: kullback/builder/compile_env.py.

D288 (2026-09-24). A tool ends the Run only when on every counted call no argument equals a value of the Starting state or an earlier result of the
same Trace, and never when the miner classed it a read. Where: kullback/builder/compile_env.py.

D290 (2026-09-24). A row a Trace's own write created is absent at that Trace's start and from the shared db (compile_env.created_rows); a Trace that
only sighted it gets it in its own Task's overlay; replay feeds each recorded call's context (recorded_call_contexts in runner/world/recorded.py) so
new_id returns the recorded id. Where: kullback/builder/compile_env.py.

D313 (2026-10-05). Tasks are cut by the user's request alone (intent clustering over user turns, threshold 0.4, then the starting-world split); each
Run's write set rides as write_labels, two labels mean dissenting; GROUPING_FORMAT is 3. Where: kullback/builder/cluster.py.

D318 (2026-10-06). The otel_genai and claude_code_jsonl adapters map: OTLP GenAI spans give turns and calls per traceId (execute_tool spans answer
calls), Claude Code transcripts one recording per sessionId ordered by the parentUuid chain, other lines counted as non_turn_lines; pointers cite
the span or line read from. Where: kullback/builder/sources.

D425 (2026-10-19). derive_world writes dangling into world_report.json: per list column of text ids, the table they key and how many name no row
(dangling_ids); nothing is pruned. Where: kullback/builder/world_tools.py.

D442 (2026-10-25). Six modules no command, module or script reaches are deleted with their tests: container_world.py 50441fb6, container_cheats.py
10c2aae8, container_grader.py 10c2aae8, builder/skills.py c9ff6ebf, store.py 50441fb6, store_specs.py 10c2aae8; this supersedes D267 and narrows
D334's kept list; ADR-0010's real-tool route keeps its factory seam in runner/real_tools.py, and the backend returns from history when a real-tool
build starts. Where: docs only.

## Spec, Intent and Verifier

D11 (2026-08-24). The Verifier is anchored to the Intent (did the user get what they wanted), not to the Reference End state. Where: docs only.

D15 (2026-08-24). An Intent is written from the trace with a grounding pass; every clause cites a trace span and an ungrounded Intent is rejected.
Where: docs only.

D30 (2026-08-26). A Verifier says what a Task expects the agent to do, and that expectation comes from the traces, not from hand-written actions.
Where: docs only.

D42 (2026-08-26). Every value in a write effect carries a Provenance (user-stated, system-derived, user-elicited, agent-chosen); the first two are
required, the last two allowed. Where: kullback/derive.py.

D47 (2026-08-26). The Intent is shown to the customer as the Task name; a correction applies to the Verifier only if it grounds to a span. Where:
kullback/runner/records.py.

D76 (2026-08-27). A policy rule that does not compile is rewritten into a checkable form first; what still cannot be checked by code becomes a judge
item; a rejected rule is a residual reported as not checked. Part superseded by D255 and D328. Where: kullback/builder/policy.py.

D113 (2026-08-29). A compiled policy rule that the confirmed recordings break often is demoted (MISCOMPILED_SHARE over at least MIN_RUNS_TO_DEMOTE);
an overlay conflict is recorded on the tau2_export gate and the Task stays gradeable. Where: docs only.

D143 (2026-09-06). Intents ground over one tokeniser that keeps money, decimals and ids whole, with a plural normaliser; a clause grounds as a whole
phrase first, then by its content tokens. Where: docs only.

D157 (2026-09-07). The leak check reads user turns across the Reference and every seed Run; a span that is not a user utterance grounds only what
some user also said. Where: kullback/user/value_strip.py.

D182 (2026-09-07). A stated fact is required only when the Intent or a user turn asked for it, else it is reported (communicate_reported, allowed);
the judge sees what each state told the user; list order is not an End state. Where: kullback/examiner/stage.py.

D190 (2026-09-08). A required value no user said becomes a shape over its column (a value the Starting state holds there) instead of a literal; a
Verifier with nothing failable gets one claim that the touched tables end as they started. Where: kullback/derive.py.

D196 (2026-09-08). Intents are written symptom-only: before grading, every value only the recording's tools held is stripped by class (a code keeps
its last four characters, a date its month, the rest nothing); the record names the column, never the value, and the leak check names what the strip
missed. Where: kullback/examiner/findings.py.

D206 (2026-09-09). A shape atom is satisfied by a value the Run itself read under that column before the write, off any row, not only by the row it
writes; the derivation stores the narrowest rung that accepts its own Reference; a shape that rejects its own Reference is dropped
(shape_self_reject). Where: kullback/derive.py.

D285 (2026-09-24). A stated fact is keyed by its source field plus canonical value; a fact is generic when it is no field's whole value (FRAGMENT)
or more than GENERIC_SHARE of at least GENERIC_MIN_ROWS world rows hold it; generic facts are never demanded, and the suite check verifier_specific
refuses a Verifier whose demands are all generic. Where: kullback/gates/verifier_suite.py.

D292 (2026-09-24). A Verifier never pins prose the agent chose: an agent_chosen write_value at least PROSE_MIN_LENGTH characters or PROSE_MIN_WORDS
words loses value and raw and becomes filled(tool, field); short values stay pinned; derivation, Examiner rows and package.export pass through
elide_agent_prose. Where: kullback/gates/verifier_suite.py.

D296 (2026-09-25). A question atom is derived only when some seed could not read the value before its write; a confirmation is a user reply after
the agent's question holding an affirmation and no denial; part superseded by D337 (the volunteered-value and stated-fact changes reverted). Where:
kullback/runner/target.py.

D301 (2026-10-05). A Spec carries a writer dict with the writer session's counts (demands, kept, refusals, gaps, parse error, truncation), spend,
read rounds, attempts and sent-back counts, with writer.first_attempt kept after a retry; write_from_intent holds the Intent fixed. Where:
kullback/spec/stage.py.

D302 (2026-10-05). An empty Spec is a writer failure: retried once in a fresh session, then saved with set_aside = empty_spec, task.set_aside
published and no Verifier written; a Spec keeping a no_write or cap check is not empty. Where: kullback/spec/stage.py.

D303 (2026-10-05). A write demand's because must quote at least 12 characters of a witnessed Intent fact or of the policy text
(ground.valid_write_because); refused demands go back to the writer once in the same conversation and corrections that ground are kept under their
ids. Where: kullback/spec/ground.py.

D308 (2026-10-05). A tool the world records as an action (evidence.actions_of off the actions table) is scored as a call, never a write: it leaves
the write set, and a demand on it compiles to an action atom (required is a Hard rule that a successful call is in the Run, forbidden the no-write
rule over that tool). Where: kullback/runner/expected.py.

D309 (2026-10-05). A no-write or a cap naming a tool covers that tool only (the cap a Hard rule of at most count successful calls); with no tool
both are unchanged. Where: kullback/spec/atoms.py.

D310 (2026-10-05). Compile names unsatisfiable Specs (compile.unsatisfiable: no-write beside a required write on the same tool, forbidden equal to
required, required write with no id_field); they go back to the writer once, then are set aside as unsatisfiable_spec with no Verifier. Where:
kullback/spec/compile.py.

D315 (2026-10-06). A Task's expected end states are data (ExpectedCell, EndState with allowed, Forbidden, Conduct), each value with a ValueSource
(user turn, policy clause, tool result) or None, derived by code in runner/expected.py; part superseded by D330 (used for fidelity measurement only,
never as the Verifier source). Where: kullback/runner/expected.py.

D323 (2026-10-07). expected_from_run gives cells only for what someone said or wrote: an actions table gives no cells; a value no source holds on a
row the last sourced call named, showed or made is the world's answer (_world_source); an affirmed agent-stated value is sourced by the user turn;
part superseded by D330 (fidelity only). Where: kullback/runner/expected.py.

D330 (2026-10-07). The Spec writer is blind to the Reference: it writes typed items (state row_is, row_new, row_keeps; event called, not_called,
before; text said; judge {question, anchor, evidence}) with gate, weight and provenance, searching the world with read_schema, find_rows, read_row,
read_policy; a value no world row, policy line or user turn holds is refused (find, do not invent); a judge anchor cites its row; code adds
nothing_else to every Task; free fields may move unchecked; alt items make several end states; spec/ may not import runner.expected or read
references.json. Where: kullback/spec/items.py.

D335 (2026-10-07). The fold: the gate flag decides, never the kind; state items are end-state cells, event items are decided in the Verdict, judge
items by score; empty judge evidence scores 0; judge evidence defaults to every agent turn up to the first hand-off; a refusal judge item always
gates; free is {field: reason}; echo facts (accepted or agent-said values) cannot alone carry a gate value; a computed judge anchor is refused; a
Spec that writes with no confirm before item gets one fixed question. Where: kullback/spec/items.py.

D337 (2026-09-27). Of D296's question-atom loosenings only the world-readable question drop and the anywhere-confirmation are kept; the
volunteered-value and stated-fact changes are reverted. Where: docs only.

D341 (2026-10-08). Learned tool effects (builder/tool_effects.py, spec/effects.py, exam/review/effects.json from Sandbox.run_diff) free each
always-moved column of an expected tool on its declared rows; the writer refuses a called item whose argument type, object shape or never-equal pair
no recorded call shows. Where: kullback/spec/effects.py.

D343 (2026-10-08). A writer repair carries every field the review loop and Spec stage own (REVIEW_FIELDS: rulings_open, truth_open, round,
examined_rounds, pending_repair, set_aside) from the Spec as it stood. Where: kullback/spec/review.py.

D348 (2026-10-09). The writer prompt gates the ask, not the recording's path: one item per ask; what the user only accepted after a proposal or
refusal is weighted; an open choice accepts any allowed option; a forbid gates only with the policy line that forbids it; a repair drops a ruled
item only when unasked, and a rebuttal starts with rebut: and quotes the user or the policy, else the writer applies the ruling. Where:
kullback/spec/writer.py.

D350 (2026-10-09). A writer round answers every ruling still open, including one the Examiner kept open after an answer (rulings.unanswered), shown
with its last answer and the Examiner's why; the set-aside rule counts writer rounds (router.writer_rounds); trust holds any Task with an open
blocking ruling, cap or no cap; part superseded by D359 (the cap). Where: kullback/spec/router.py.

D351 (2026-10-09). Every writer session (write, repair, confirm question) appends one line to exam/writer/<task>.jsonl with kind, round, ruling,
spend, counts, final text and every conversation in full; the writer is never shown these files. Where: kullback/spec/writer.py.

D352 (2026-10-09). A called check with a set-valued argument compiles to a Hard rule matching any member of the set, scoped to its row for a legacy
write value; state cells already match sets. Where: kullback/spec/compile.py.

D353 (2026-10-09). IntentFact carries after_pushback, marked by the miner only; a gate resting only on after-pushback or accepted facts with no
policy line beside it is weighed by code, with the kept line saying why; mixed support still gates. Where: kullback/spec/items.py.

D354 (2026-10-09). Volunteered facts (not after pushback, not unfounded) with no gated cover go back to the writer once more (asked_missing); a told
value is a gated judge item anchored on the world row holding it, a named variant or method a gated check on that field. Where:
kullback/spec/writer.py.

D356 (2026-10-10). spec/mutants.py makes code mutants of a base end state (value, undone, untouched, extra_row), at most 40 per Task with a fixed
seed, scored killed, survived or open; scripts/mutants.py writes exam/mutants/<task>.json; part superseded by D362, D368 and D375 (rulings, ask
keys, the built base). Where: kullback/spec/mutants.py.

D359 (2026-10-10). The round cap is a setting: router.rounds_cap() reads KULLBACK_ROUNDS_CAP (a whole number of at least one, else refused), else
ROUNDS_CAP 5; writer_rounds counts the union of every round an answer was given in (rulings.answer_rounds). Where: kullback/spec/router.py.

D361 (2026-10-11). A side fact is scored, never gated: the miner reads each fact's kind (do, keep, told, detail) and code sets side
(intent_tools.mark_side) for a told fact beside a do or keep fact, unless said in the opening ask; a fallback stance needs the quoted agent line
(in_reply_to) else fallback_unquoted; gates citing side or fallback facts are weighed by code. Where: kullback/spec/intent_tools.py.

D362 (2026-10-11). Surviving mutants become Examiner rulings by code (spec/mutant_rulings.py): unchecked_field on a check resting on a volunteered
fact, unchecked_write where every undone mutant on a row survived, unchecked_row per table, else a note; a ruling names table and field, never a
value; a review that never called verify gets it by code; part superseded by D368 (ask keys, no Reference base). Where:
kullback/spec/mutant_rulings.py.

D366 (2026-10-16). Each Intent fact names its condition: none, asked, world_or_policy, or agent (reactions); a late fact must quote the agent line
it answers (in_reply_to) or is sent back once then kept as agent; a reaction another recording volunteered first is promoted to none; a gate citing
a reaction is weighed; a world_or_policy gate holds only when the item confirms the claim (a policy_line, or confirms naming the row). Where:
kullback/spec/intent_tools.py.

D368 (2026-10-16). Mutants key to the ask: ask_keys reads gated items whose facts are asks (do or keep, not a reaction); the Reference is never a
base for mutants or the stray write; part superseded by D375 (the base is built from the Verifier). Where: kullback/spec/mutants.py.

D369 (2026-10-16). A gated told-kind item (judge, said) that cites no told fact and quotes no policy_line is weighed with reason untold
(items.TELL_WEIGHED); weighings run from one ordered table (items._WEIGHINGS); a refusal judge item is untouched. Where: kullback/spec/items.py.

D370 (2026-10-16). Every do or keep fact that is not a reaction must be cited by a gate item (schema.unchecked_asks); at verify code files each
uncited ask as a blocking unchecked_ask ruling, one per fact; trust holds the Task untrusted with reason unchecked_ask, after open_ruling; part
superseded by D376 (only a rebuttal excuses an ask). Where: kullback/spec/mutant_rulings.py.

D375 (2026-10-16). The mutant base is built by code from the Verifier (constructed_base: the start state with every gated cell set and gated new row
made, checked to hold); where a gated cell is not one literal value the base is the newest passing fresh Run; with neither the status is no_base; a
mutant is killed when a gate that held on the base fails; solvable stays a flag. Where: kullback/spec/mutants.py.

D376 (2026-10-16). An ask is excused only by the writer's rebuttal: rebut_fact(fact_id, why) is kept only for a fact sent back in that session and a
why quoting at least 12 characters of the user's words or a policy line; Spec.rebuttals records kept ones; coverage gaps never excuse; the Examiner
may rule unchecked_ask on a wrong rebuttal. Where: kullback/spec/writer_tools.py.

D384 (2026-10-17). kullback/spec/coherence.py asks one judge question per Task between miner and writer, from the Intent and recordings only, with a
quote kept only when it sits word for word in a line of the named recording; a user ending sets the Task aside as intent_strayed (no Verifier,
reason no_verifier); a failed call sets nothing aside; part superseded by D385 (the ender). Where: kullback/spec/coherence.py.

D385 (2026-10-17). Each opening ask is answered carried_out, refused_or_withdrawn, or ended_by with an ender (user, agent, world); only ended_by
user sets the Task aside as intent_strayed, kept only when the quote is in a user line and carries words beyond control markers; part superseded by
D388 (no separate call). Where: kullback/spec/coherence.py.

D387 (2026-10-17). A gated literal value is kept only when reachable: a write tool's learned writes for that table and field (literal values, at
most 12, or from_args), a write result of a recording of the Task, the user's own words (not echo), or a start row under that field name; a policy
line alone never reaches a value; add_items refuses the rest with up to ten reachable values; a cell's source is where it is reachable from; the
constructed base never builds a sourceless cell. Where: kullback/spec/items.py.

D388 (2026-10-17). The miner answers each opening ask's outcome (answer, ender, recording, quote, why) in add_facts, with no separate judge call; a
user dropping an ask after a price, condition or refusal is refused_or_withdrawn; an ask with no outcome sets nothing aside and is counted. Where:
kullback/spec/coherence.py.

D392 (2026-10-17). Two gated called items naming the same tool with differing args are a contradiction, sent back through the existing turn; part
superseded by D396 (equal args are a repeat). Where: kullback/spec/items.py.

D396 (2026-10-17). A gated not_called whose args are none or a subset of a gated called item's args on the same tool, and a gated called or
not_called repeating an earlier one, are contradictions (_gated_call_clashes); a called atom's description names its args. Where:
kullback/spec/items.py.

D414 (2026-10-18). A born Task carries its scenario's end in born/<task>.json; load_inputs adds it as end facts (end1, end2, kind do, source
END_SOURCE, never a user turn); END_RULE: every change an end fact names is gated, everything it rules out gated absent, told values gated, other
gates weighed, picked writes are alternative end states; an end fact uncited after the follow-up sets the Spec aside as unchecked_end; a gated
called item stands only where its policy line names the tool (names_tool); a said number compiles to said_shape; end facts never render as the
user's words. Where: kullback/spec/writer.py.

D417 (2026-10-18). When the learned effects file is absent or holds no learned writes and the workdir has write tools, the Spec records effects:
none learned on Spec.writer and the stage counts effects_none_learned; nothing is refused. Where: kullback/spec/writer.py.

D419 (2026-10-19). For a born Task: each end write needs a gated row_is differing from the start cell or a gated row_new (ungated_writes); every
value the clause names of the changed row must appear in a gated state item (ungated_values); findings go back once as an ungated_write ruling, then
are counted; a Verifier whose code side holds on the do-nothing Run is untrusted with reason held_by_idle. Where: kullback/spec/ungated.py.

D421 (2026-10-19). A failed Spec write records the error class and the first 200 characters of its message under failures, first_failure once in the
counts, and restores the Task's Spec file as it was. Where: kullback/spec/stage.py.

D423 (2026-10-19). A row_is, row_keeps or judge anchor path that meets a list at a non-index part on the start row is refused
(items.unreadable_path) naming the path, list and indexed form; the Harbor export sets such a Task aside as unreadable_cell, counted under skipped.
Where: kullback/spec/items.py.

D426 (2026-10-19). In the reach check (D387) an end fact counts as the user's words only for a field no tool reach knows (no learned write of that
table and field, no recorded write result under that name); else a gated value must come from a tool write, a start row, a real user turn or a
copied argument, or is refused listing the reachable values. Where: kullback/spec/reach.py.

D427 (2026-10-19). A said value that is one amount (currency sign, thousands comma, decimals or a unit of up to three letters) compiles to its
digits past the last thousands group, decimals kept without trailing zeros (said_shape, in atom_of); a bare integer, several numbers or other text
stay as written; the description keeps the writer's values. Where: kullback/spec/items.py.

D428 (2026-10-19). At compile each check is said once: two atoms of one kind on one target are one atom, first id kept, gated where either was
(merged_atoms); two cells on one field of one end state fold where one absorbs the other, the stricter kept only when it is a gate wherever the
other is (merged_cells), each fold a merged line; cells whose values neither implies and subset row_new pairs are contradictions (cell_clashes,
new_row_overlaps) sent back to the writer. Where: kullback/spec/compile.py.

D435 (2026-10-19). A told item (judge or said) whose cited facts are all end facts, with no user turn fact and no policy line, is weighed with the
note "told gate weighed: only the end names it"; it stays gated with a user turn fact or a policy line requiring the telling; D369 stands beside it;
an end fact that only names a telling goes to rebut_fact. Where: kullback/spec/items.py.

D436 (2026-10-19). items.contradictions names a row_keeps item when a row_is or row_new of the same end state, table and resolved row names one of
its fields; it goes back to the writer once and, unresolved, makes the Spec unsatisfiable; a row_new names a kept row only where it carries that
row. Where: kullback/spec/items.py.

D438 (2026-10-19). World.licence licenses a value a rendered fact or policy line names, a value on a world row a rendered fact names by key or one
picking field value of three characters or more (World.named_rows), or a value equal to a fact's words after canon, hyphens as spaces and case
folded (items.folded); no abbreviation table; World._unfound still refuses an unlicensed value naming LICENCE_WAYS; an echo-only gated row_is or
row_new value stays a gate only where a licence holds, else it is weighed with reason echo (ECHO_WEIGHED); a choice licenses nothing. Where:
kullback/spec/items.py.

D439 (2026-10-19). After compile and the Examiner, spec/stage.py end_unpassable ends a Task with no Verifier as precondition_unrendered (a gated
write's binding policy line asks a user-given value of a schema field the world holds and no rendered fact gives one; a line speaks of a tool by
name or by a heading sharing a word with it) or no_reference_holds (faithful References exist and none holds every gate); a Task with no Reference
keeps its path; both join router.NO_VERIFIER_WHYS and the tier report's no_verifier_whys; amends D367 (the Reference vetoes, never sources). Where:
kullback/spec/stage.py.

D440 (2026-10-19). Two gated called items on one tool contradict only when both write one existing row with an argument set two ways
(items.contradictions(items, world), World.target_rows); with no World every pair clashes (D392); two row_new items on one table, end state, count
and gate whose where clauses set no field two ways merge into one (merged_new_rows), a weighed one never folding into a gated one; the hand-off rule
is unchanged (handoff_optional held back). Where: kullback/spec/compile.py.

D441 (2026-10-19). At the Spec stage the compiled Verifier is scored on the idle Run; if no code gate fails it must carry a gated judge item or a
gated said item naming a value, else idle_send_back sends it back once (counted idle_sent_back) and, still so after the Examiner, end_unpassable
ends it as idle_run_passes (counted set_aside_idle_run_passes, a NO_VERIFIER_WHYS reason); a Spec already set aside is not sent back. Where:
kullback/spec/stage.py.

## Examiner and trust

D8 (2026-08-24). Trust gates are pass or fail, never blended into a score. Where: docs only.

D10 (2026-08-24). The frontier can be wrong: a recorded Run becomes a Reference only after confirmation; frontier-failed Runs go to a separate
bucket. Where: docs only.

D12 (2026-08-24). A tool-equipped judge may rule on a novel End state but can never award a pass. Where: kullback/examiner/judge.py.

D48 (2026-08-26). Two human checks exist (setup review before Verdicts, blind audit after) but neither is required for a build or a Verdict (D92).
Where: kullback/gates/artifacts.py.

D49 (2026-08-26). A Task that cannot be replayed is "not gradeable" and names the blocking tool or gap; assisted Verdicts are shown beneath it and
never counted. Where: kullback/gates/scorecard.py.

D57 (2026-08-26). Confirming a Reference passes several gates, including a judge that marks pass or fail and abstains below a real threshold; part
superseded by D92 (no human hand-off). Where: kullback/gates/scorecard.py.

D79 (2026-08-27). Every Verifier passes an automated suite before use (empty Run fails, wrong path fails, second valid path passes, loophole probe,
leak check); part superseded by D333 (the suite no longer gates trust). Where: kullback/examiner/findings.py.

D81 (2026-08-27). The held-out anchor is a fixed share of every Task's Runs (20 percent, at least one, fixed seed); a Task with fewer than three
Runs is unguarded and cannot alone justify an edit. Where: kullback/examiner/stage.py.

D92 (2026-08-27). Every judge is agentic (read tools, at least one check before answering); disagreement between two judges replaces the human
label, and the pipeline completes without a person. Where: kullback/runner/judge.py.

D111 (2026-08-29). A recorded Run is a Reference when its replayed End state is what the Intent plus the policy say should happen; References of a
Task must agree; the benchmark reward is never a build input and is read only by the scorecard. Where: kullback/examiner/stage.py.

D112 (2026-08-29). Each Task gets three frontier re-rolls in the built Environment by default; the benchmark agreement lines are scaffolding to be
deleted once the rule holds. Where: kullback/examiner/reference.py.

D119 (2026-08-29). The unsolved-state check: a Reference cut one step short must score no pass; a Reference with nothing to cut leaves the check not
run, which counts as a fail. Where: kullback/gates/verifier_suite.py.

D122 (2026-09-02). Gates are a package no agent can write; a gate is code with no model call, knows no corpus, and rules on both agents' work; a
tool_call hook blocks writes into gates and runner. Where: kullback/builder/sandbox.py.

D123 (2026-09-02). Two agents on one core: the Builder makes the Environment and never touches a Verifier; the Examiner reads what the Runner
produced and never touches the Environment; neither sees the other's reasoning. Where: kullback/examiner/exam_files.py.

D127 (2026-09-02). A Verifier may be tightened freely and loosened only toward what the frontier did: every probe stays in the Task's pool and must
keep scoring no pass, and a new version may newly pass only the Reference or a re-roll. Where: kullback/runner/records.py.

D133 (2026-09-02). The loosening pool grows only from the solver (frontier re-rolls from any round, live production Runs); the Examiner may spend
allowance on more re-rolls; the per-Task false-rejection rate is reported beside the trusted count. Where: kullback/examiner/stage.py.

D156 (2026-09-07). A Run that wrote nothing has the End state ANSWERED when its answer states facts read from the world, else the empty state; the
two group apart. Where: kullback/gates/verifier_suite.py.

D160 (2026-09-07). The judge models are named (--judge-model, --second-judge-model; each ruling reads provider/model:persona) and disagreement is
reported per model pair. Where: kullback/runner/judge.py.

D170 (2026-09-07). Findings are filed by rule off the records before any model choice (blocking tool, failed or not-run check, over-strict atom,
unsettled End state), ranked by the Tasks they cost, deduplicated by key, at most ten per round. Where: kullback/examiner/findings.py.

D172 (2026-09-07). A Task with no Reference is unfinished until it is trusted or refused with a reason; an exit never reads done while such Tasks
remain. Where: kullback/gates/counts.py.

D173 (2026-09-07). D79 fixes for Verifiers that never write: an atom with nothing to mutate is no failed flip, the unfinished cut for a no-write
Verifier falls before the first answer stating a required fact, the false-rejection pool drops failed_recordings, a single-Reference Task reports
second path as not run. Where: kullback/gates/verifier_suite.py.

D185 (2026-09-08). The held-out pool holds only Runs whose settled End state equals the Reference's; the agent judge ships off by default (opt-in).
Where: kullback/examiner/stage.py.

D186 (2026-09-08). A judge failure names the state, a key it differs on, the value there and what it should be; a failure that does not check out in
code drops the judgement to an abstention. Where: kullback/examiner/reference.py.

D189 (2026-09-08). A lone Reference is re-rolled for a second path, three Runs a batch and three batches at most, stopping at the first Run reaching
the Reference's settled End state; these Runs corroborate and never reopen the Reference. Where: kullback/examiner/stage.py.

D193 (2026-09-08). A judgement leaving two or more End states gets one second pass over the survivors or abstains with the keys they differ on;
failures agreeing on one required value record no_correct_recording; part superseded by D198. Where: kullback/examiner/reference.py.

D194 (2026-09-08). A Task is trusted only when its false rejection over the held-out pool is under FALSE_REJECTION_THRESHOLD, or the pool is empty
(no_pool); every ruling carries the fraction and the pool size. Where: kullback/examiner/stage.py.

D198 (2026-09-08). A judgement that leaves several End states hands them to the derivation: one Verifier per survivor (at most four) through the
suite, the best becomes the Reference (survivor_chosen, survivors_equivalent, survivors_all_fail); a suite failure names every failing and not-run
check. Where: kullback/examiner/stage.py.

D199 (2026-09-08). When re-rolls find no second path, one is written from the Reference Run (swap independent adjacent calls, repeat a read before a
write, drop an unused read) and kept only if it replays to the Reference End state; a Run offering no rewrite is single by structure and stops
blocking only when every other check passes and the pool holds a reaching Run. Where: kullback/examiner/stage.py.

D205 (2026-09-08). An over-strict Verifier is loosened by the harness, code only, by atom kind (drop a question atom, a row-naming write becomes
wrote-any, a value becomes a shape, an answer becomes reported), at most two relaxations a proposal, gated by the suite, the probe pool and
one-directional loosening. Where: kullback/examiner/plan.py.

D208 (2026-09-09). A Verifier carries the identity of the Reference and Run ids it was derived from and is live only while its Task confirms that
Reference; otherwise it is retired (reference_withdrawn, reference_rederived), kept in history unaccepted, and every consumer reads the live set
through one accessor. Where: kullback/difficulty.py.

D222 (2026-09-09). The harness runs the reads a judge needs before asking it (a checks section of prefilled sources and argument-free tools);
tool_choice required only where a tool's arguments remain; an agreeing verdict must cite a check, else unresolved; counters judge_checks_prefilled,
judge_refused_no_check and others; kullback judge-smoke. Where: kullback/runner/judge.py.

D230 (2026-09-10). A tool asked about a Task it cannot serve answers with the Task's state (turned-down Runs with reasons, whether a Reference is
unreadable or absent, the verbs that help) as a result, not an error; a session that read an error back closes on what the beat left; only a session
with no output stalls; a round whose derive errored with no clean derive may not exit done. Where: kullback/examiner/session.py.

D266 (2026-09-18). The harness's own agents read recordings and Runs through one set of tools (outline, paged payloads with a next offset, search
returning locations, neighbours) beside a read-only query over the workdir store; accepted on measured tokens, not on reputation. Where: docs only.

D280 (2026-09-24). The Builder writes one note per Task through note_task, with a reason off a fixed list (outcome not in state, Intent contradicts
Reference, fact unavailable to the user, needs an action record) and one sentence, never atom or Verifier text; the Examiner rules on each with
note_ruling, shown in the Builder's next opening message; an open note blocks only that Task's refusal. The edit_verifier repair tool is gone.
Where: kullback/examiner/domain_tools.py.

D281 (2026-09-24). A Run file is one Run of one Task, named by prefix, Task and seed, opening with its run_id and task_id, created once; the loader
refuses a file of two Runs or of another Task; the trusted gate resolves every seed of the accepted version and fails when one is not a Run of the
same Task; one ruling, workdir_trusted_ruling, gives the one trusted number. Where: kullback/gates/trust.py.

D305 (2026-10-05). The intent tier report scores every runs/**/*.jsonl that loads as a Run, grouped by Task. Where: kullback/spec/trust.py.

D311 (2026-10-05). A replay is known by its recorded model, never by its run id: trust.run_models reads run.model and reply models; fresh only when
it names a model that is not the recorded one; a Run naming none is neither. Where: kullback/spec/trust.py.

D317 (2026-10-06). A Finding is a diff with its why and rows: a body edit names the Environment file, the recorded call, the first differing column
and both values, for the Builder only; FindingVerb, suggested and hint are removed; old findings files still load; part superseded by D331. Where:
kullback/examiner/findings.py.

D320 (2026-10-06). The Spec is the one writer of Verifiers and the Examiner writes no Verifier and runs nothing; derive.py and variants.py left the
Examiner; loosen.py, the edit, probe and re-roll tools and --verifier-from are removed; part superseded by D330 (Reference-free writer) and D331
(rulings). Where: kullback/spec/review.py.

D331 (2026-10-07). The Examiner gives feedback and never edits: rulings under exam/rulings/<task> (item, kind and code, reason, fix, blocking or
note); rule, close and verify in examiner/rule_tool.py; the writer applies or rebuts each (answer_rulings); run_rounds alternates Examiner and
writer; an open blocking ruling after the last round holds the Task untrusted; a ruling that the Reference is wrong is a note; the cap, part
superseded by D350 and D359. Where: kullback/spec/rounds.py.

D333 (2026-10-07). A Task is trusted when the do-nothing Run and a code-built stray-write Run both fail its Verifier and no ruling is open;
untrusted reasons in order no_intent, open_ruling, constructed_run_passed; the stray write counts only when it fails at the sanity item on a base
the Verifier passes; reference_passes and solvable ride as flags; a failing kept Reference files a blocking fails_reference ruling; can_fail,
reference_replay and independent_pass gates and the old tiers are retired. Where: kullback/spec/trust.py.

D336 (2026-09-25). A Task whose every Reference the Examiner rejected waits in exam/reference_pool.json, untrusted and not refused; a later Run
reaching a non-excluded End state takes it out by the D111 rule; a Run reaching an excluded End state fails. Where: kullback/gates/trust.py.

D339 (2026-10-08). review_specs runs one Examiner session per Task (REVIEW_TURNS_PER_TASK) over one round bus; examined rounds are recorded on the
Spec; a Task with no examined round is untrusted with reason not_examined, read after no_intent and before open_ruling. Where:
kullback/examiner/session.py.

D340 (2026-10-08). Each Task's view shows every recording's user turns and each fact's source turn, stance, echo and unfounded flags; three blocking
scope codes: intent_unfounded (the writer may not gate on unfounded facts alone), gated_against_ask and not_enough_truth (only rebuttable by citing
a user turn, holds the Task untrusted); a Task session gets 20 turns. Where: kullback/examiner/rule_tool.py.

D342 (2026-10-08). A review session's system prompt carries the full exam/review/policy.md as the section examiner_policy; the search tool
find_tools returns at most 8 matching tools; policy-paragraph search is gone. Where: kullback/examiner/session.py.

D344 (2026-10-08). A review session's bus subscribers (pricer, watchers) are unsubscribed in a finally when the session ends, so each call is booked
once. Where: kullback/examiner/session.py.

D346 (2026-10-09). Judge answers for a fresh Run are asked once and kept under exam/judged/<task>/<run>.<verifier hash>.json, the reason never
reaching the writer; part superseded by D371 (verify no longer judges the Reference). Where: kullback/spec/judged.py.

D347 (2026-10-09). Trust reads saved judge answers (spec/judged.py), reused only when verdicts/<task>/scored_with.jsonl names the current Verifier's
hash; when the do-nothing Run fails and no base exists the reason is stray_untested; part superseded by D371 (no Reference answers). Where:
kullback/spec/judged.py.

D349 (2026-10-09). The Examiner's choice rule opens with the goal step: from the user's own requests made before any refusal, failure or proposal,
and the policy, say what a correct agent leaves behind, hold the Intent against it, then check the items; intent_unfounded only where a gate or
verdict-deciding item rests on the accepted fact. Where: kullback/examiner/prompt.py.

D357 (2026-10-10). A not_enough_truth ruling kept open ends the Task with no Verifier (set_aside intent_unclear, trust reason no_verifier); the
writer replies rebut: cannot verify: <why> rather than loosen a check; part superseded by D367 (fails_reference and reference_wrong gone). Where:
kullback/spec/rulings.py.

D358 (2026-10-10). examiner/goal.py derives the expected end state per Task from the users' own words, the policy, the Starting state (four read
tools) and the signatures, never the Intent, items, Verifier, Run or Reference (refuse_run_inputs); writes exam/goal/<task>.json (required, choice,
untouched, told, refusals, unknowns); a tool-set value is choice or unknown unless the policy states it; lists are whole or appended, never indexed
paths; report only. Where: kullback/examiner/goal.py.

D367 (2026-10-16). The Reference never repairs a Verifier: no fails_reference ruling, no reference_wrong set-aside; a failing kept Reference is a
Builder finding (reference_fails_rows, key reference_fails) filed after the rounds; reference_passes stays a flag; no Verifier has one reason left,
intent_unclear. Where: kullback/examiner/findings.py.

D371 (2026-10-16). The Examiner never sees a Run: verify lays the do-nothing Run and the stray write on the mutant base, files unchecked asks and
ask-keyed mutant rulings; the Reference's steps and judge rows are gone; review file tools refuse reads under Run copies and whole-root grep or
find; the fidelity examine session still reads the Reference. Where: kullback/examiner/rule_tool.py.

## Runner, user and Run protocol

D9 (2026-08-24). An off-path call (a real tool, valid arguments, no recorded response) is not a hallucination; an off-path Run is reported assisted,
never failed for being off-path. Where: docs only.

D26 (2026-08-26). The Environment's write log is the transaction log; assertions and fingerprints are computed from the state it produces. Where:
docs only.

D44 (2026-08-26). The Simulated user repeats this user's recorded facts exactly and is representative on everything else; an improvised answer is
allowed, never required. Where: kullback/user/rules.py.

D45 (2026-08-26). A call to a nonexistent tool or with invalid arguments is counted, never failed; a fabricated result fails the Run; errors come
back in the customer's own encoding. Where: kullback/runner/route.py.

D64 (2026-08-26). A Run is one flat JSONL file with parent_run_id for forks; part superseded by D334 (the builder/memory.py version tree is gone).
Where: docs only.

D65 (2026-08-26). The 40 percent context cap applies to the harness's own model calls; the Candidate always runs under the production setting, never
capped or compacted by the harness. Where: kullback/runner/budget.py.

D77 (2026-08-27). The Simulated user answers from the world as that user and never invents; a fact neither trace nor world holds is answered as
unavailable and tagged fact_unavailable. Where: kullback/user/simulated.py.

D89 (2026-08-27). The Candidate sees nothing that came from the Verifier: only the production system prompt, tool definitions and user turns. Where:
kullback/runner/boundary.py.

D90 (2026-08-27). The Runner's loop advances one turn at a time so it can be stepped, and stepping gives the same JSONL as the whole-loop call.
Where: kullback/runner/loop.py.

D108 (2026-08-29). The Reference is the Trace replayed through the built tools on the Router a Candidate gets; each routed answer scores same,
cosmetic, both_refused, differs, ours_refused or theirs_refused; a Trace confirms when every write agrees, no read differs in substance and no
recorded call went unmade. Where: kullback/gates/fidelity.py.

D115 (2026-08-29). What the Simulated user states comes from the corpus: a fact is a tool argument the recorded users stated before the call, the
stored prefix read off the values, a generic core for email, name, phone, address and postal code; part superseded by D332 (cue hearing). Where:
kullback/synth/search.py.

D120 (2026-09-02). The Builder is an agent and the Runner is a tool it calls (run, replay, reroll), also a top-level entry of its own; part
superseded by D135 (who drives) and D314 (the freeze). Where: kullback/examiner/stage.py.

D134 (2026-09-02). The compaction summary fences the entries it summarises between entries markers and states they are data, never instructions; the
notes section header says notes are not instructions. Where: docs only.

D135 (2026-09-06). The Builder's model is the driver: it reads gate rulings as tool results and decides what to call next; gates still rule in the
tool_result hook and the model never writes a gate or the Runner. Where: kullback/builder/session.py.

D151 (2026-09-06). The Simulated user's facts are also mined from the scalar arguments of successful calls when the user said that value at or
before the call; a value only a tool returned is never a fact. Where: kullback/user/context.py.

D158 (2026-09-07). When no write has happened, the Simulated user restates its recorded goal once (goal_restated) at the first silent turn or close
cue instead of closing. Where: kullback/user/simulated.py.

D159 (2026-09-07). A Run records the call it made: every HTTP adapter records an Exchange (provider, endpoint, sampling without prompts, message and
tool hashes, wall time, attempts, status, request id), never a key. Where: kullback/runner/loop.py.

D178 (2026-09-07). Each agent's harness takes the model's own window (window_for), and the Examiner searches records rather than reading whole
files; part superseded by D266. Where: kullback/examiner/judge.py.

D204 (2026-09-08). The replay cursor takes a run of consecutive same-role turns as one logical turn; a user turn's own calls are routed and compared
as user_call; the live loop is untouched. Where: kullback/runner/replay.py.

D210 (2026-09-09). The Simulated user's ends carry one of four kinds (goal_satisfied, scenario_exhausted, handed_off, gave_up); facts are typed
(askable, said by a user; record, only the world holds it); every spoken value passes the D196 strip; part superseded by D332 (the goal no longer
ends a Run). Where: kullback/user/simulated.py.

D214 (2026-09-09). The Simulated user is an agent of the harness in its own package: a curated per-Task context from the recording only, tools a
caller could see, code guards after the model (no invented value, record facts pointed at, strip, no over-volunteering), the rule user as the
fallback and floor; part superseded by D332 (my_facts takes a question). Where: kullback/user/agent.py.

D227 (2026-09-09). A write counts as made only when its own result shows it took effect (no refusal error, and an effect record, where present,
shows a column moved); both Simulated users read this through one function in kullback/user/rules.py; a refused write leaves the goal open; the
round counts user_ended_on_refused_write; FindingKind carries runs_disagree. Where: kullback/user/ends.py.

D229 (2026-09-10). A provider is a registry row, not an adapter: built-in rows in pricing.py plus an optional providers.local.json are laid over the
models.dev catalog key by key; non-finite rates read as absent; a base_url endpoint takes PROVIDER_API_KEY; reasoning fields pass the wire
whitelist; a reply is priced under the id it was asked for, provider included; a row may name a live price list, written rates the fallback. Where:
docs only.

D232 (2026-09-10). kullback build takes --user-model <provider/model>, landing in models["user_agent"]; without it the rule user drives; the id is
recorded in report_config.json and cleared by a rebuild without the flag. Where: docs only.

D243 (2026-09-10). The posting client's read timeout is 300 s by default, overridable by KULLBACK_MODEL_TIMEOUT_S (an unparseable value is an
error), connect 10 s; the Exchange carries read_timeout_s and RetryExhausted on a timeout names the budget. Where: docs only.

D245 (2026-09-10). Single-message stages send a stable system head: the invariant instruction lines as the system message, the per-Task evidence as
the user message; payload dumps use sort_keys=True so equal payloads hash and prefix alike. Where: kullback/examiner/reference.py.

D253 (2026-09-18). The loop's step is split into ask (a policy gives the next assistant message) and advance (tools through the router, then the
Simulated user, then the stop rules); step stays a thin wrapper, landed on its own re-freeze. Where: kullback/runner/loop.py.

D256 (2026-09-18). One factory builds the Simulated user and the round scorer for the CLI and every Run; each user turn leaves one typed record
(driver, facts given, agreement, guard fired, end kind and reason), stored whole. Where: kullback/user/factory.py.

D269 (2026-09-18). The Simulated user's system prompt is stable (static skill text, then a per-Task block byte-identical across turns and Runs) and
the conversation is appended as spoken turns only; every request must be a byte-prefix extension of the one before, checked by code; cache share is
reported, never gated. Where: kullback/agent/prefix_check.py.

D272 (2026-09-18). Every tool call is a transaction: a crash rolls the state back and is recorded as its own outcome, a body fault, never shown to
the Candidate as the system's answer. Where: kullback/runner/transaction.py.

D273 (2026-09-18). A training Run never uses the model stand-in: where no code can answer, the Run ends as Environment cannot answer and carries no
reward signal. Where: kullback/runner/route.py.

D275 (2026-09-18). One usage record everywhere (fresh input, cached input, output, reasoning) and cost computed in one place. Where:
kullback/ai/usage.py.

D289 (2026-09-24). A price or window is looked up under the exact wire id with the provider's catalog name (PROVIDER_CATALOG_NAMES), then without
its profile, then the vendor row, then the hand table, and price_source records which; the ledger records the id sent; kullback budget reprice;
bedrock openai ids use BedrockOpenAIModel on the Chat Completions path. Where: kullback/ai/model_limits.py.

D291 (2026-09-24). budget.reprice prices the cumulative counters in budget.json, feed lines priced per call and the remainder under --model or the
stage's last model; the ledger keeps cache_write_1h per stage. Where: kullback/runner/budget.py.

D293 (2026-09-24). The budget ledger keeps each stage's calls and tokens per model and reprice prices each model's share at its own row; Ceiling.add
charges land in direct_usd and direct_calls, carried over unchanged; only an allowed agent-chosen prose atom is elided, never a required one. Where:
kullback/runner/budget.py.

D294 (2026-09-25). One predicate, ends.goal_done: a goal naming writes is done when all are made; one naming none is done only after at least one
tool call and a closing latest turn; a close with no tool call is handed_off. Where: kullback/user/simulated.py.

D296 (2026-09-27). The agent user is the Simulated user by default in every Run, on the run model over the rule floor; --rule-user keeps the rule
user alone. Where: kullback/cli.py.

D297 (2026-09-27). The catalogue row decides the Responses wire shape: a model row whose provider.npm override is @ai-sdk/openai
(pricing.RESPONSES_NPM) resolves to OpenAIResponsesModel; everything else stays on chat; providers.local.json may set the override. Where:
kullback/ai/pricing.py.

D298 (2026-09-27). Every model request takes a slot from its provider key's bucket (kullback/ai/limits.py), with rates from an optional rate object
on a provider or model row; a retry counts against the cap, a backoff holds no slot, and a 429 sets a cooldown every waiter honours. Where:
kullback/ai/limits.py.

D299 (2026-09-27). A Run file carries what its Simulated user recorded about each turn (tool log even when empty, what it weighed, driver, drop
reason, end marks) through runner/loop._user_payload. Where: kullback/runner/loop.py.

D300 (2026-09-27). The user turn message states the stop rule as sequencing: use the reading tools the rules name, then answer with the turn itself,
never with a tool call as the turn. Where: kullback/user/agent.py.

D306 (2026-10-05). A yes to what the Candidate asked to do is an answered turn: guards.confirms reads CONFIRM_REQUEST in the Candidate's turn and a
sentence opening on AFFIRM_CUE with none on NEGATIVE_CUE, which never counts toward UNANSWERABLE_LIMIT. Where: kullback/runner/atom_context.py.

D307 (2026-10-05). The guard strip keeps the sentence and removes only the words the strip changed (_unchanged_words); a sentence flagged without
change is still dropped whole. Where: kullback/user/guards.py.

D326 (2026-10-07). The rule user hears id asks (vocabulary_of falls back to the signatures' id arguments), answers held facts under a confirm cue,
closes on a repeated line with user_repeat, defers the transfer token one turn, ends only when every requested write is made (goal_counts); the
Runner prompt ends with a line to list every asked item and whether it was done. Where: kullback/user/simulated.py.

D332 (2026-10-07). The user looks facts up (user/lookup.py) by field words, alias and question-word overlap, with missing for a field held nowhere;
the goal never ends a Run (the Candidate's close, the scenario running out and the turn limit do; goal_satisfied is a label read off the Spec's
write items; user_goal_met on every turn); no scorer reads the goal; Task.facts_in_instruction runs a single-turn Task with no Simulated user.
Where: kullback/user/lookup.py.

D338 (2026-09-25). The agent user's prompt names my_choices and my_account with one example each; an unmatched kind answers with the kinds held; the
account lookup takes the ids the rules hold outside the identity as row-id keys. Where: kullback/user/tools.py.

D397 (2026-10-17). In the agent loop a turn cut at the output cap with no tool call is not a stop: CUT_TURN_ASK is asked once, a second cut ends the
run; the unfinished hook (at most UNFINISHED_ASKS 2) asks only when the run would end; generation sessions name the item numbers not written. Where:
kullback/agent/loop.py.

D413 (2026-10-18). A retry session gets at least the turns per item its first pass spent (turns_for); at the turn cap the unfinished hook gives one
last turn opened by LAST_TURN; on a cut turn that called tools the last call is answered CUT_CALL and not run while complete calls run. Where:
kullback/agent/loop.py.

## Gates, scoring and reward

D1 (2026-08-24). A Step is only screened against the recorded action; a Run is re-executed and gets the Verdict on its End state. Where: docs only.

D3 (2026-08-24). The Verdict is on the End state, never the path or the reasoning; path differences are reason codes, reasoning is read for
fabrication only. Where: docs only.

D20 (2026-08-24). The tool-selection (ACTION) match is computed but never counts toward the Verdict; part superseded by D254. Where: docs only.

D22 (2026-08-26). Exact means exact after canonicalisation (ids, timestamps, float formatting, key order); the canonicaliser has its own tests.
Where: docs only.

D25 (2026-08-26). Model judges stay only as grounded rubrics (semantic columns, non-literal facts, reason codes and disputes) and are never the sole
basis of a Verdict. Where: docs only.

D31 (2026-08-26). The reward basis borrows the best of each benchmark: state diff, assertions, communicated facts, an exact diff-key forbidden rule,
protected fields as Hard constraints, a null-agent gate. Where: docs only.

D39 (2026-08-26). Write effects are exact after canonicalisation; a read mismatch is cosmetic or semantic, unknown types count as semantic, semantic
must be zero; a 100 percent strict read fidelity is a warning sign. Where: kullback/runner/canon.py.

D43 (2026-08-26). A Verdict grades effects: world writes plus what the user was told or asked; reads, their order and reasoning are the path and
never change a Verdict. Where: kullback/derive.py.

D46 (2026-08-26). A failure is explained by its first failing atom, computed by code, plus same-path or different-path; a free-text explanation is
opt-in. Where: kullback/runner/state.py.

D84 (2026-08-27). A semantic column goes to the judge only when the canonical strings differ, and the ruling is cached in an equivalence table a
person can overturn. Where: kullback/runner/canon.py.

D88 (2026-08-27). Failure attribution is code first (environment_suspected when the Run touched an assisted or flagged path), a judge on the rest.
Where: kullback/runner/judge.py.

D94 (2026-08-27). End state only, reaffirmed: "a required write tool was called" is a path check and is not an atom. Where:
kullback/runner/verdict.py.

D110 (2026-08-29). Code verifies what code can verify; a judge decides only what code cannot and never awards a pass; rubric-scoring judges are out.
Where: kullback/examiner/reference.py.

D154 (2026-09-06). A fidelity ruling names the first leaf where two answers part, with both values cut at 60 characters (first_difference), walking
past noise the canon rules absorb. Where: kullback/examiner/findings.py.

D171 (2026-09-07). Tool fidelity is kept per call and attributed per Task: a Task is blocked by a tool only when one of its own recorded calls
differs (blocking_tools); the corpus gate still asks 100 percent. Where: kullback/examiner/stage.py.

D187 (2026-09-08). The scoring path honours column classes per column (exempt equal, hard by canonical string, semantic through canon.compare),
prose results compare by their read columns, keyed lists pair by key; memorised values also refuses a datum a recorded result carried. Where:
kullback/gates/tool_runs.py.

D215 (2026-09-09). A write is judged on every row it changed: effects between sightings are read off the recording, and the replay checks each
credited column after the last credited write, failing with kind effect and marking the stale read downstream_of it. Where:
kullback/runner/replay.py.

D217 (2026-09-09). Two differences are never cosmetic: presence (a value one side holds and the other does not fails, unless both are canonically
empty) and domain tokens (differing token sets give differs with reason token_set, the judge not asked). Every verdict records its route; the stage
key carries a verdict format. Where: kullback/gates/tool_runs.py.

D219 (2026-09-09). A comparison answers equal, different or unresolved, and unresolved is never agreement: replay records it as its own route and
fidelity counts it a failure; an unsettled atom is unevaluable; a rate with no denominator is not measured; the judge and a persistent equivalence
table are wired where a build makes the comparer. Where: kullback/gates/tool_runs.py.

D228 (2026-09-09). The claim gate is measurement only: one hard claim atom over all write tools, scored and reported, never failing a Verdict,
because the round 2 bar was missed. Where: kullback/spec/must_not.py.

D254 (2026-09-18). A Verifier has one scorer: the structured atom target is the only meaning of an atom, read by one interpreter (check_run) for the
Verdict and every gate; Hard rules stay compiled policy source; the customer's canon rules reach that interpreter. Where: kullback/runner/target.py.

D255 (2026-09-18). Reward that reads text is grounded in facts: a communicate atom passes only on a value a tool returned in this Run, a confirm
atom on the Simulated user's typed agreement record, and a model judge is reported beside the Verdict and never decides it. Where:
tests/test_cli.py.

D284 (2026-09-24). One canonicaliser per Environment: CanonRules load once (canon.load_rules, canon.rules_of) and go to every scorer;
target.canon_fn accepts only CanonRules or a bound callable, anything else raises CanonMissing naming the caller. Where: kullback/runner/target.py.

D286 (2026-09-24). verifier_wrong_run always runs a swap probe: each demanded value is replaced in a copy of the Reference by another value the
world holds under the same field (swapped_runs), and the Verifier must fail every swap. Where: kullback/gates/verifier_suite.py.

D287 (2026-09-24). The leak check compares canonical values, searches only the content of the user rules, and treats a value a row the Reference
read holds as world knowledge, not a secret; only a value no read row holds can leak. Where: kullback/gates/verifier_suite.py.

D316 (2026-10-06). The verdict decides gates first (one expected end state matches with no collateral move, no forbidden write or call, each conduct
rule holds), failing with gate:expected, gate:forbidden or gate:conduct; None leaves the Run not verdicted; then the atoms weigh; a forbidden atom
that cannot be checked leaves the Run not verdicted; predicates get deep copies. Where: kullback/runner/verdict.py.

D328 (2026-10-07). A judge item ({question, anchor, evidence}) is one model call per item at temperature 0, met 1 or not met 0, nothing guessed, cut
evidence marked; evidence refs are assistant_turns, final_answer, assistant:<k>, after_call:<tool>, clamped at 4000 and 24000 chars; a failed call
is None, never 0; empty evidence scores 0 by code; a gate item scored 0 fails the Run; JudgeScore records it on Verdict.items; narrows D255 for gate
items. Where: kullback/judge/items.py.

D329 (2026-10-07). Every item gates or scores: Atom, ExpectedCell, Forbidden, Conduct and Check carry gate and weight; pass is every gate item
holding; score is 0 on a failed gate, else the weighted mean with gate items counted held and at least half the weight, None when open; a code
sanity item for every Task (nothing outside the matched end state and allowed moved); check_run reads gate atoms only; VERDICT_VERSION 4. Where:
kullback/runner/verdict.py.

D400 (2026-10-17). The item judge asks at JUDGE_TOKENS 4000 and once more at RETRY_TOKENS 16000 on an empty or cut reply, JSON first; JudgeScore
carries status (answered, unreadable, outage), stop_reason and retries; an outage leaves the item unscored; ITEM_JUDGE_VERSION 2. Where:
kullback/judge/items.py.

D408 (2026-10-18). A judge item with no evidence is scored 0 by code with status answered and why no evidence, so a Harbor trial gets a number.
Where: kullback/judge/items.py.

D424 (2026-10-19). A judge reply with exactly one result row is read as the asked item whatever its key, noted in why; with several rows the key
must be the item id; the score stays 0 or 1. Where: kullback/judge/items.py.

## Hub, export and Harbor

D221 (2026-09-09). An Environment is published from an allow list (env package, Starting states, toolkit records, frozen Task list, Verifiers) with
a manifest of versions and hashes, verified on fetch; a leak scan stops on recorded prose and counts short values as echoes; a release needs replay
fidelity 0.90, else a preview. Where: kullback/cli.py.

D363 (2026-10-11). kullback export --format harbor writes one Harbor task per Task (instruction.md, task.toml schema 1.3, a digest-pinned
environment with the one Task's world, tests with the Verifier, solution/solve.sh from a replaying Run, Tasks without one left out and counted);
harbor_world serves the tools as a CLI and MCP server with a consent user; reward.json holds the D329 reward, items.json every item; judge items use
KULLBACK_JUDGE_MODEL at test time and fail closed without it. Where: kullback/hub/harbor.py.

D399 (2026-10-17). publish and export --format harbor take --tasks verified (every recorded Task with a Verifier; no Verifier, no_verifier drafts
and synthetic left out and counted); task.toml carries spec_tier and spec_tier_reason from the tier report and, with --labels, label,
label_matches_gold and label_gold_diff; leaks are reported masked by Task, part and count; an oracle-only leak leaves the Task out; a Task with no
replaying Run ships without solution/ (has_oracle false); --from-staged reuses a dry run; a release replaces the repository (delete_patterns *);
part superseded by D402 (cards). Where: kullback/hub/harbor.py.

D401 (2026-10-17). kullback/processes.py each(items, fn, workers) runs a spawn process pool in item order; intent_tiers, package.export,
export_harbor, stage and publish pass workers (--workers); stage rules the Task index once; output is byte-identical whatever the count; the default
is the smaller of cores minus 2 and 4. Where: kullback/processes.py.

D402 (2026-10-18). An Environment card is a short page: summary with release or preview status, Use it, What is inside, Known issues (a dated list
saying some Tasks can be wrong), How it was built, Licence; no trust numbers, tiers, gates or leak scans on a card; the organisation card holds a
mission sentence, the Environments table, Use it, five Terms, Known issues and Links; no News. Where: kullback/hub/card.py.

D403 (2026-10-18). package.task_ask is the one source of a Task's ask for both exports (the Spec Intent's volunteered facts, else the Task's Intent
text, else an older mined Intent); a Task with no ask ships intent "" and is counted without_ask. Where: kullback/hub/package.py.

D404 (2026-10-18). A package ships env/policy.md (the workdir's own, else the recorded system prompt when every Task shares one; none and named on
fetch otherwise); environment.json is not a required run input. Where: kullback/hub/package.py.

D405 (2026-10-18). A judge item whose status is not answered (outage, unreadable, not_configured, not_started) leaves a Harbor trial unscored: no
reward.txt, a non-numeric reward.json reason, items.json with statuses, exit UNSCORED_EXIT 3. Where: kullback/hub/harbor_world.py.

D406 (2026-10-18). cli.task_run_files lists a Task's Run files from runs/<task>/*.jsonl and runs/<task>-<seed>.jsonl (digit seeds only), so verdict
scores what run wrote. Where: kullback/cli.py.

D407 (2026-10-18). environment_record derives environment.json from the shipped files (schema, tool and policy versions as sha256 of schema.json,
tool_sigs.json and env/policy.md, env_id hashing the three); verdict scores first, runs the D97 gate, and writes Verdicts and scored_with.jsonl only
when it passes; the report adds Candidate Runs scored. Where: kullback/hub/package.py.

D409 (2026-10-18). harbor_world.argument_error checks every call against its tool's spec: a missing required or unknown parameter is refused naming
the parameters, not logged and not answered; no aliases. Where: kullback/hub/harbor_world.py.

D418 (2026-10-19). The published package carries no stopped_because column and no untrusted reasons table; stage, trusted, refused and the D333
fields stay. Where: kullback/hub/package.py.

D422 (2026-10-19). The Harbor instruction renders every Intent fact the user said whatever its stance (asked_facts), keeping out the agent's echo,
facts ruled unfounded, end facts and empty text: volunteered facts under the opening lead, then the facts said on request or after a refusal under
one later lead; viewer rows and instruction.md both come from opening(asked_facts(spec)). The export drops every item citing only facts the
instruction does not render, freeing its cells; an item citing no fact, an end fact or a policy line stays; the Verifier ships byte for byte when
nothing drops; the manifest carries items_unrenderable. Where: kullback/hub/harbor.py.

D429 (2026-10-19). At export only, a list column of text ids read as keying one table (every member naming a row names that table and at least half
name one) drops the members naming no row from the shipped world, shared and overlay rows alike; a column naming two tables or too few rows is
ambiguous and named, nothing dropped; a Task whose kept Reference passes a dropped id is set aside as reads_dropped_id; the manifest carries
ids_dropped. Where: kullback/hub/listed_ids.py.

D430 (2026-10-19). Every Harbor user reply after the opening is CONSENT_LINE: yes to what was asked, no to anything different, nothing to add.
Where: kullback/hub/harbor_world.py.

D431 (2026-10-19). Before the D423 gate the export names a cell on a one-member start list by index 0 (resolve_cells), reading the start row only
and never changing a value, gate or weight; a list of several or no members is left to the D423 gate; the manifest carries cells_resolved; part
superseded by D437 (now its one-member case). Where: kullback/hub/repair.py.

D432 (2026-10-19). The export allows, per end state, the columns a required tool's learned effects always move (allow_effects, why "free: learned
effect: <tool>"); required are a called item's tool, the one tool whose effects alone answer a declared row, and a kept Reference's tool on a
declared table; a checked or already freed column gains nothing; made rows are never freed; a missing or empty effects file is learned again offline
(effects_from). Where: kullback/hub/repair.py.

D433 (2026-10-19). Harbor reward.txt and reward.json carry the pass alone: 1.0 when every gate holds, else 0.0, failing closed; items.json carries
shaped_reward (the gated then weighted Verdict score) beside the items; the card and manifest say what reward.txt means. Where:
kullback/hub/harbor.py.

D437 (2026-10-21). The export's repair (resolve_cell) compiles a list value with one entry per member of the start list (plus members a required
tool's effects append, a <list>.N pattern) to one indexed cell per member, same gate, weight split evenly, members matched by their own leaf first;
a mismatched length, one value over several members, or a nested list stays for the D423 gate; a new row where path through a list sets the Task
aside as UNREADABLE_CELL; a non-prohibition called set tests every list member (list_sets); the Reference is never read. Where:
kullback/hub/repair.py.

## Synthetic scenarios and born Tasks

D7 (2026-08-24). Only Replicas count toward the bar; a synthesised Scenario never does. Where: docs only.

D37 (2026-08-26). Tasks beyond Replicas (Scenarios from traces or a snapshot, customer cases) serve coverage, hardening and training, never the bar.
Where: docs only.

D224 (2026-09-09). Synthetic Tasks are walked over a tool-call graph mined from the Runs (edges only where values chained), bound from evidence, run
through the Builder's bodies (a refused walk is discarded and counted), and kept under synthetic/ apart from tasks.json; synthetic never counts as
trusted. Where: kullback/synthesise.py.

D225 (2026-09-09). Task archetypes are read off the domain's public material as structured records (goal, situation, effects, rule, source); a
source under the corpus's own repository or paper is refused, operator-side and copied records dropped; unmapped archetypes become coverage gaps;
nothing shaped enters trusted. Where: kullback/cli.py.

D259 (2026-09-18). A second synthetic Task system sits beside the generator: an explorer proposes kinds of Task, code builds each from a walk, code
gates at birth (the scripted walk passes and a do-nothing Run fails, every argument traces to the Intent, a user fact or an earlier result, a
Decline Task trips a policy predicate), a controller picks Tasks; solve rate is a label, never the gate; Frontier, Decline and Mixed Tasks. Where:
docs only.

D372 (2026-10-16). A scenario is one record under workdir/scenarios (kullback/synth/records.py) with id, statement, source, also_from, policy lines,
frequency, review rows, evidence and status (born, fixing, passed, unsure, infeasible, struck); the generator is a role on the shared agent harness
with tools over the store; write tools refuse tool names, id-shaped values, unknown policy lines, spans a page does not hold and copied typed lines;
exact repeats fold by code; part superseded by D378, D381 and D383. Where: kullback/synth/records.py.

D373 (2026-10-16). The scenario review is blind (views drop source, folds, frequency and fix count), one session per batch under synth_review;
outcomes by code: repeat_of strikes, occurs no strikes, any unsure goes to scenarios/human.json, feasible no marks infeasible and files an
Environment finding, else passed; one send back rewrites once; part superseded by D382 (cites and checks). Where: kullback/synth/review.py.

D374 (2026-10-16). The scenario layer searches through the D115 reader (kullback/synth/search.py onto builder/search.py): TinyFish when
TINYFISH_API_KEY is set, then Firecrawl keyless, memoized under workdir/web_cache, nothing live unless HARNESS_ALLOW_MODEL_REQUESTS; search.Crawled
reads pages through the D225 crawl. Where: kullback/synth/search.py.

D378 (2026-10-17). The scenario step is named generate: CLI verb kullback synth scenarios, stage synth_generation, summary generated.json. Where:
kullback/synth/generate.py.

D380 (2026-10-17). kullback synth scenarios plans first: plan sessions (PLAN_BATCH 150 openings) put every opening in exactly one situation (si_
plus a content hash, one-word kind, variants) or skip it with a reason, through write_situation, attach_openings and skip_opening; hedges and held
openings are refused; the plan lives in scenarios/plan.json; leftovers count as openings_unplanned. Where: kullback/synth/plan.py.

D381 (2026-10-17). Scenarios are written per situation and variant of the plan: write_scenario takes situation and variant, code fills source Task
ids and frequency from the plan; one scenario per situation and variant, a repeat refused naming the holder; two to four third-person sentences
about one customer, values said, never an id or tool name; part superseded by D420 (rows picked by code, binding computed, requires refused). Where:
kullback/synth/generate_tools.py.

D382 (2026-10-17). The reviewer cites a page it read: occurs yes needs occurs_url read this session and occurs_span found in its cached text; a
search result or policy line is no cite; feasible yes needs checked (tool:<name>, row:<table>:<key>, verified) and empty requires; feasible no files
a world edit holding requires and binding; send backs quote the statement, with why and change; part superseded by D412 (any row the world holds).
Where: kullback/synth/review.py.

D383 (2026-10-17). A statement whose ROUGE-L against any stored or session statement is at least NEAR_REPEAT 0.7 is refused as near_repeat naming
the twin; exact twins fold and same-situation variants are exempt; generated.json counts spread and cells. Where: kullback/synth/generate_tools.py.

D389 (2026-10-17). kullback synth tasks born_tasks makes one Task and one recording per passed, bound recordings-source scenario with empty requires
(task_ plus 12 hex of the scenario id's sha256; recording sc-, source scenario, one user turn); born ids join the build sample; spec/trust.py tiers
a Task whose recordings are a scenario's as synthetic, never trusted, kept out of TIERS. Where: kullback/synth/tasks.py.

D390 (2026-10-17). A born Task gets an empty overlay (overlays/<task id>.json), so its Starting state is the shared world; a rerun writes a missing
overlay and counts overlays_written. Where: kullback/synth/tasks.py.

D393 (2026-10-17). A scenario carries ask (the customer's first-person opening) and end (what a correct agent leaves, third person, quoting the
policy) apart; generate_tools.pair_refusal refuses a shared sentence (ask_shares_end), a tool name in either, or one without the other; both empty
stays allowed. Where: kullback/synth/generate_tools.py.

D394 (2026-10-17). Items a generation session left unwritten are retried once in batches of half the size, at least RETRY_MIN 4; BATCH stays 25;
generated.json counts retried items and written. Where: kullback/synth/generate.py.

D395 (2026-10-17). A born Task's recording carries the prompt most recordings of the workdir share, and its user turn is the scenario's ask, else
its statement; a rerun repairs a missing prompt and counts prompts_written. Where: kullback/synth/tasks.py.

D398 (2026-10-17). generate.from_gaps runs last: one item per unhit (situation kind, table) cell and one per policy line no kept scenario quotes
(lines of at least LINE_WORDS 5 words), GAP_PER 1; source kind gaps needs ask and end; a kind outside the plan is refused kind_not_in_plan;
generated.json records gaps before and after; --source selects sources. Where: kullback/synth/generate.py.

D410 (2026-10-18). plan_search sessions write situations from a page span read (no openings, kind from the plan, near_repeat refused, at most
PLAN_SEARCH_BATCH 25, naming the THIN 3 thinnest kinds); search scenarios take a plan kind; --target N runs rounds (plan_search, recordings over
unwritten situations, gaps) until the store keeps N, a round adds under STOP_SHARE 5 percent, or the ceiling; generated.json records rounds, stop
and diversity. Where: kullback/synth/generate.py.

D411 (2026-10-18). Generator fixes from live observation: reuse of earlier customers allowed (list_scenarios marks this_session); row search counts
matches and a miss shows the key shape and two samples; requires only when no row fits; the id rule spares a bound row's value and names the matched
word; a kind sharing a stem with a plan kind is refused; refusals quote the hedge word, nearest page line or policy line; plan_search writes only
where a tool or a policy refusal exists. Where: kullback/synth/generate_tools.py.

D412 (2026-10-18). feasible yes takes row:<table>:<key> for any row the world's Starting state holds, bound or not; a row or tool the world lacks is
refused naming it. Where: kullback/synth/review.py.

D415 (2026-10-18). A scenario bears a Task (bears_task) when passed, bound, with an ask and an end and nothing required, whatever its source kind; a
search scenario carries its page URL and byte span as provenance. Where: kullback/synth/tasks.py.

D416 (2026-10-18). A born recording's user rules carry the scalar cells and nested leaves of the owner row only (the bound row the others refer to,
else the one holding recorded user fields), filtered to fields recorded users give, on request, its key dropped where recorded fields exist; the
writer treats them as facts the user says (RECORDS_SOURCE); run_user gives a born Task the agent user from its rules on disk. Where:
kullback/synth/tasks.py.

D420 (2026-10-19). synth/slice.py picks by code the slice each scenario is written onto (owner row, rows naming the owner key, one level of referred
rows, under a size limit), deterministic by seed and spread over owners; generation sessions get each item with its slice, kind and one policy line
and lose list and lookup tools; the binding (dict of table to keys) is computed from the rows named; every end value must be in the slice, a quoted
policy line, or a sum or difference of two slice amounts; requires is refused; kinds and policy lines are assigned by plan.by_distance;
TURNS_PER_ITEM 4, at most five slices per session; the review's end_outcome choice sends back. Where: kullback/synth/slice.py.

## Measurement and process

D4 (2026-08-24). Customer metrics are few: Run pass rate with a paired non-inferiority interval, hard-constraint violations, tool selection P/R/F1
with the hallucinated-tool rate, cost per Run, pass^k as a conditional gate, Environment fidelity. Where: docs only.

D5 (2026-08-24). A Candidate clears the bar when it is not worse than the frontier by more than a pre-registered margin (3 to 5 points), paired per
Run, clustered SEs, one-sided 95 percent interval. Where: docs only.

D17 (2026-08-24). Simple by default; complexity only where it is worth it. Where: docs only.

D19 (2026-08-24). There is no gold: the stand-in is the confirmed Reference plus agreement, and its confidence is published per Task. Where: docs
only.

D21 (2026-08-24). A good Task is one where two domain experts would independently reach the same verdict; diagnosis is labelled less reliable.
Where: docs only.

D23 (2026-08-26). Unbiased reporting: every recommendation says where sources disagree, where an earlier answer was wrong, and where coverage is
thin. Where: docs only.

D24 (2026-08-26). Build for the hardest case: long, multi-call, multi-turn, off-path Runs drive the design. Where: docs only.

D34 (2026-08-26). The founder's words are kept verbatim in docs/founder-words.md. Where: docs only.

D38 (2026-08-26). Every assumption is registered in docs/assumptions.md with its evidence grade, its test and what breaks if it is wrong. Where:
docs only.

D41 (2026-08-26). Representativeness is a hard constraint on everything the harness generates: an artifact that cannot be tied to observed data is
rejected. Where: kullback/builder/compile_env.py.

D51 (2026-08-26). Generalisability over overfitting: every stage is tested on Runs it did not see; a perfect seed score with a held-out collapse is
reported as overfitting. Where: kullback/builder/compile_env.py.

D53 (2026-08-26). The harness is layered like tau: a provider-neutral model layer, a small core loop, faces that only consume events; the core knows
no corpus. Where: docs only.

D58 (2026-08-26). Customer trace intake questions live in docs/trace-intake.md; traces are never committed. Where: docs only.

D59 (2026-08-26). The Harness is the whole system; the Builder makes the Environment and the Runner re-executes Runs in it. Where: docs only.

D61 (2026-08-26). The harness is a workflow for RL environment generation and takes its principles from environment-generation pipelines; the
Builder improves until it builds high-quality Environments; part superseded by D314 (the freeze). Where: docs only.

D63 (2026-08-26). The four building principles: a hard 40 percent context cap, minimal and compact design, load things when needed, the model
studies its own failures. Where: docs only.

D78 (2026-08-27). The re-run count k is a config value chosen by experiment, not by design. Where: docs only.

D80 (2026-08-27). The target is 100 percent with every miss explained; a miss without a recorded reason is a gate failure. Where:
kullback/gates/fidelity.py.

D85 (2026-08-27). The report opens with whether the Environment was built and which gates passed, then the numbers, then a suggestion; the person
decides. Where: kullback/cli.py.

D86 (2026-08-27). When the spend ceiling is hit the build stops, reports as is and asks before continuing; nothing is switched to assisted to get
past it. Where: kullback/cli.py.

D96 (2026-08-27). Coverage is counted in Tasks against the Task list fixed at the start of the build, also weighted by Run count. Where:
kullback/gates/scorecard.py.

D116 (2026-08-29). Model prices come from a models.dev snapshot refreshed at most every 7 days when live calls are allowed; the hand table is the
offline fallback, and each call records which source priced it. Where: kullback/ai/pricing.py.

D121 (2026-09-02). Packages depend one way only (ai at the bottom, agent and runner above it, the agents on top), enforced by import-linter. Where:
kullback/runner/boundary.py.

D125 (2026-09-02). Skills are the prompts a model may rewrite: SKILL.md files listed by name and loaded on demand; an edit is versioned and accepted
only by a gate; part superseded by D132. Where: kullback/agent/context.py.

D129 (2026-09-02). The package is kullback at the repo root with subpackages and two frontends (cli.py, tui/) that only consume events and call
entrypoints. Where: docs only.

D132 (2026-09-02). A skill edit is tentative until a paired sequential test (alpha 0.05) over rounds is decisive; promoted skills are re-checked and
demoted the same way; the skill gate is code, never a model judgment. Where: kullback/builder/skills.py.

D136 (2026-09-06). The mechanic repairs only what a model wrote (tool bodies, predicates, Verifiers, probes); everything that grades or simulates is
fixed by people before a build, never by the loop. Where: tests/builder/test_user_sim.py.

D138 (2026-09-06). No agent session has a turn cap; the spend ceiling is the hard wall. Where: docs only.

D144 (2026-09-06). Every agent system prompt follows the GEPA order in tagged sections (task, tools with one example call each, skills, general
examples, rules, targets, stop last); examples are invented and general; feedback is text. Where: kullback/examiner/reference.py.

D152 (2026-09-06). Every spend bucket carries cache_saved_usd beside the dollars paid; memo hits stay a count. Where: kullback/report/render.py.

D209 (2026-09-09). Every Task with a live Verifier carries an objective difficulty record (write atoms, tools called, paths, question atoms,
held-out acceptance, judge agreement) in difficulty.json, bucketed by fixed bands, and trusted coverage is reported per bucket. Where:
kullback/cli.py.

D212 (2026-09-09). Every sample the harness draws per Task or Run comes from sample_key(kind, id, build salt); the salt is written into the workdir
on first draw and read back; budgets go out in key order. Where: kullback/examiner/stage.py.

D223 (2026-09-09). kullback/claims.py writes claims.json per round: past-tense sentences on the write vocabulary are matched to write atoms
(written, unwritten, writes_unclaimed); partial completion is atoms confirmed over atoms carried; it never gates trusted. Where:
kullback/difficulty.py.

D252 (2026-09-18). Replay fidelity stays the public headline; beside it a thin interface lets anyone drive a Run in a built Environment (reset,
step, code reward, seed, no Builder import), and a solve-rate stage is a measuring instrument, not a trainer. Where: kullback/laws.py.

D258 (2026-09-18). Environment consistency: code-generated call sequences check laws for any tool (a read changes nothing, writes are deterministic,
an error changes nothing, a write reads back, minted ids unique, no crash or hang); a broken law is shrunk and handed to repair; published beside
fidelity, never called fidelity; Traces that break a law on purpose turn it off for that tool. Where: kullback/cli.py.

D276 (2026-09-18). The round loop's stop rule reads the fidelity rate and the trusted share and records a reason every time it stops. Where: docs
only.

D278 (2026-09-18). The grill's removals: the second scorer's meaning (D254), the affirmation list and substring communicate check and the judge in
the pass (D255), the three user construction sites (D256), the all-or-nothing intake gate (D263), hand-set stage versions, and tau2 and data_model
in the allowed imports with the any-dict-with-messages guess (D261); the synthetic generator and D215's replay effect check stay. Where: docs only.

D279 (2026-09-23). Smoke 7 set the next wave of harness changes (write gate replays the trace prefix, static gates before replay, every failing row
shown, not run is not a failure, unscored atoms refused, every Task derived in parallel, run only Tasks with a Verifier); no web access for the
agents until needed. Where: docs only.

D314 (2026-10-05). One content hash of every .py under runner/, gates/ and spec/ (runner/code_hash.py) is the runner_version on every Run, Replay
and Verdict and the code_hash on every tier row; mixed rows warn; freeze-runner records it for export only; agents still may not write runner/ or
gates/ (gates.PROTECTED). Where: kullback/runner/code_hash.py.

D334 (2026-10-07). Dead code no command, module or script reaches is deleted (vulture plus reference counts): builder/memory.py, builder/triage.py,
agent/reading.py, builder/body_skill.py, the model-written body path and the Intent writer in builder/intent.py; builder/skills.py, the container
world and cheat suite, runner/world/birth.py, the tau-mirroring ai and agent surfaces, the hub and documented CLI commands are kept on purpose.
Where: docs only.

D364 (2026-10-12). The heldout 100 (unseen Tasks, blind label per Verifier against benchmark gold as a ruler) passed every pre-registered bar, and
trusted is not right: the wrong Verifiers that reached trusted leaned on the Reference, which D365 answers. Where: docs only.

D365 (2026-10-15). The product is an agent that asks for the behaviour wanted and builds the evals and training data for it, a human reviews a
sample, every model is benchmarked, the chosen model trained. The source of truth is the ask, the policy and the world; the Reference is for
fidelity only and never enters the Verifier, repair or the mutants; reactions to the agent's conduct never gate; benchmark gold is a ruler, never an
input; mined held-out Tasks are the eval, synthetic Tasks training data. Where: docs/vision.md.

D377 (2026-10-16). Web search and page reading are one module, kullback/agent/web_tools.py, registered by any role through register_web_tools(api,
search, refuse, on_call); search returns at most ten rows; read_page asks refuse first and returns numbered lines with a from_line continuation;
with search None both say the search is off. Where: kullback/agent/web_tools.py.

D386 (2026-10-17). The 100 held-out rerun on the clean-ask head failed all five pre-registered bars; the Reference had been catching unreachable
gate values and recording-only transfer gates, which D387 replaces. Where: docs only.

D391 (2026-10-17). The 100 again and the 253 on the gate and stray head held every bar but the stray filter; nearly every no_base Task left by D387
sits on gated whole-field lists. Where: docs only.

## Retired

D2: judge for disputes beside Match and adequacy, superseded by D12.
D18: reference environments first, then the generator, dropped from the log on 2026-09-22 (founder's cut).
D27: Simulated users conditioned on production traces, deferred, superseded by D44.
D28: deferred items, with reasons, to todo.md, dropped from the log on 2026-09-22 (founder's cut).
D32: monitor the actual DB transactions, dropped from the log on 2026-09-22 (founder's cut).
D36: runs become Tasks by write-tool partition, then Intent clustering, dropped from the log on 2026-09-22 (founder's cut).
D50: the first build is tested against tau2 and against real production traces, dropped from the log on 2026-09-22 (founder's cut).
D52: real-world tool-using traces, not coding traces, are the first build's real input, dropped from the log on 2026-09-22 (founder's cut).
D54: build shape follows TauForge's four stages, dropped from the log on 2026-09-22 (founder's cut).
D55: the smallest slice runs through the Verdict, dropped from the log on 2026-09-22 (founder's cut).
D56: the builder emits tau2's shape first, dropped from the log on 2026-09-22 (founder's cut).
D60: self-generated traces are built in parallel with the tau2 slice, and must, dropped from the log on 2026-09-22 (founder's cut).
D62: the Builder's stop condition is customer Verdict agreement, scored against tau2's own, dropped from the log on 2026-09-22 (founder's cut).
D69: which missing components enter the first build, dropped from the log on 2026-09-22 (founder's cut).
D75: tool body repair, dropped from the log on 2026-09-22 (founder's cut).
D82: builder self-improvement, dropped from the log on 2026-09-22 (founder's cut).
D83: tasks have a hierarchy, dropped from the log on 2026-09-22 (founder's cut).
D87: Cross-corpus lessons with a relevance condition, component removed (kullback/builder/memory.py, D334).
D91: verifier.py lives in the Builder and calls the Runner through records only, dropped from the log on 2026-09-22 (founder's cut).
D93: a disputed Reference sets the Task aside until a person resolves it, dropped from the log on 2026-09-22 (founder's cut).
D97: remaining build-level details closed by defaults, to be revised by the tau2, dropped from the log on 2026-09-22 (founder's cut).
D105: the screen is the pipeline, not a chat, dropped from the log on 2026-09-22 (founder's cut).
D109: every stage hashes the modules it delegates to, dropped from the log on 2026-09-22 (founder's cut).
D117: the body-writing model gets a row lookup and a self-test, both inside, dropped from the log on 2026-09-22 (founder's cut).
D118: the Builder's independent model calls run on a few threads, dropped from the log on 2026-09-22 (founder's cut).
D124: agentic context management with a code floor at 40%, dropped from the log on 2026-09-22 (founder's cut).
D126: a round is four beats ending at gates, dropped from the log on 2026-09-22 (founder's cut).
D128: the two agents take turns on one event stream, dropped from the log on 2026-09-22 (founder's cut).
D130: seven build phases, each small, each leaving the tests green and the, dropped from the log on 2026-09-22 (founder's cut).
D131: research on D124, dropped from the log on 2026-09-22 (founder's cut).
D137: build 9 measures the mechanic on the Tasks that write, dropped from the log on 2026-09-22 (founder's cut).
D139: one table per build, printed by one script, with the detail underneath, dropped from the log on 2026-09-22 (founder's cut).
D140: the status report shows every red light, grouped, with a zoom, dropped from the log on 2026-09-22 (founder's cut).
D141: the Examiner's finding names the Builder verb and carries the hint, dropped from the log on 2026-09-22 (founder's cut).
D142: a round has moved when a gate count, an artifact or a, dropped from the log on 2026-09-22 (founder's cut).
D145: a repair result opens with its own target's ruling, and a request, dropped from the log on 2026-09-22 (founder's cut).
D146: the Intent stage ratchets, dropped from the log on 2026-09-22 (founder's cut).
D149: a Task the re-rolls stage skips loses the re-rolls an earlier build, dropped from the log on 2026-09-22 (founder's cut).
D150: the Builder loads a triage skill written from where it kept failing, dropped from the log on 2026-09-22 (founder's cut).
D153: the driver builds the target when the model repaired and never called, dropped from the log on 2026-09-22 (founder's cut).
D161: the driver builds the target again after a narrowed repair, dropped from the log on 2026-09-22 (founder's cut).
D163: the Examiner derives per Task, in parallel, from a cache, dropped from the log on 2026-09-22 (founder's cut).
D166: a round is not done until the target was built, and status, dropped from the log on 2026-09-22 (founder's cut).
D168: the body skill module for body-writing calls, component removed (kullback/builder/body_skill.py).
D169: two more exits, dropped from the log on 2026-09-22 (founder's cut).
D174: a recompile replaces the kept body only by beating it, dropped from the log on 2026-09-22 (founder's cut).
D175: Examiner read index and READ_CHARS clamp, superseded by D266.
D177: a Task's re-rolls are keyed on the Task's own inputs, inside the, dropped from the log on 2026-09-22 (founder's cut).
D181: nine repairs to the Builder's and the Examiner's loop, read off what, dropped from the log on 2026-09-22 (founder's cut).
D192: five loop repairs read off the second watch of the Builder and, dropped from the log on 2026-09-22 (founder's cut).
D197: a row read more than once in a Task is served its, dropped from the log on 2026-09-22 (founder's cut).
D207: a nested row is named by its own scope, and the class, dropped from the log on 2026-09-22 (founder's cut).
D218: a round's Task rows are written once, at round close, and every, dropped from the log on 2026-09-22 (founder's cut).
D231: round accounting survives a raised beat, dropped from the log on 2026-09-22 (founder's cut).
D240: a Task's re-roll Runs run side by side in Run number order, dropped from the log on 2026-09-22 (founder's cut).
D246: the compile heads are built once per build and the helper specs, dropped from the log on 2026-09-22 (founder's cut).
D251: number held for the write-end branch, no row written, never built.
D264: User population per Task drawn by key from good recordings, component removed (kullback/user/population.py).
D265: One typed user-signal record with one owner per field, component removed (kullback/user/signals.py).
D267: one workdir store and an append-only journal, superseded by D442.
D268: Merges proven by code, embeddings proposer not adopted, never built (grepped merge proof and embedding in kullback/).
D270: User prompt examples mined per corpus by situation, component removed (kullback/user/examples.py).
D274: a stage's cache key is computed from the code it imports, dropped from the log on 2026-09-22 (founder's cut).
D304: A ruling re-roll starts at the Task's next free seed, superseded by D320.
D312: reference_replay gate on kept Reference replays, superseded by D333.
D319: Code trust ruling on sourced expected end states, tiers, superseded by D333.
D321: Consistency checks contradicts_user and unasked_writes as trust gates, superseded by D333.
D322: One trust ruling with the sourced gate inside Spec tiers, superseded by D333.
D324: Refusals and no-writes as forbidden writes from the Reference, superseded by D330.
D325: Examiner edit kinds reaching the Spec (cell, conduct, reference), superseded by D331.
D327: Not run apart from failed, writer disagreement flag, superseded by D333.
D345: verify reads the Reference three ways, superseded by D371.
D434: literal pick check for echo values, never built (held back, see D438).
