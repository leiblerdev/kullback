# The learnings ledger of the 2026-09-22 overhaul

> Record of 2026-09-22: the index the overhaul was judged against. On 2026-09-24 two module paths were corrected to where the code now lives.

Every decision (D01 to D278), every learning in `docs/learnings.md` without a decision of its own (L<section>.<n>) and every standing founder word (F<n>) has a row here, and each row says where that learning lives in the shape of ADR-0011 or why it is dropped. Two counts, so the next reader does not re-derive them: the table has 264 decision rows, one for every decision the log ever carried, and after the prune of 2026-09-22 `docs/decision-log.md` holds 212 of them, 155 as headed entries from D69 on and 57 as earlier bold entries in the same file. The 52 that were removed are the ones the log's own "Removed on 2026-09-22" list names, and every one of their rows says so in its reason column: 27 are dropped and the other 25 keep the place that carries what they taught.
How to use it: a brief cites the rows its stream owns, and it may not carry a learning that no row points at its package.
A stream's acceptance re-checks its own rows: for each one, the module named is present at the named path, the gate named rules, the prompt section named exists, or the drop is still true.
A row that says "dropped" carries no work, but the paragraph it points at in `docs/learnings.md` still holds and the next reader must be able to find it.
A row whose location turns out wrong is corrected here in the same commit as the code, never silently: this file is the index the overhaul is judged against.
Locations use the vocabulary of the brief (carried module, gate, hook, prompt section, runner tool, base tool, dropped) plus one the brief did not have, "standing rule", for a decision that is a rule for people rather than code and lives only in a document.

| id | the learning | where it lives | reason |
|---|---|---|---|
| **kullback/ai** | | | |
| D116 | Model prices come from models.dev, the hand table is the offline fallback | carried module kullback/ai/pricing.py | |
| D152 | Spend is reported with the cache's effect beside it | carried module kullback/ai/usage.py | |
| D229 | Providers are reachable by name through the registry, not through adapters of their own | carried module kullback/ai/provider.py | |
| D243 | A model client's read timeout fits the call it is making | carried module kullback/ai/provider.py | |
| D275 | One usage record everywhere and one place that computes cost | carried module kullback/ai/usage.py | |
| **kullback/agent (core, base tools, hooks, bus)** | | | |
| D63 | Four principles: a hard context cap, minimal and compact design, general mechanisms, measured behaviour | carried module kullback/agent/ (compaction at 40 percent) | the cap is code; the other three are house rules |
| D64 | Builder history is a tree, Runner Runs are flat files | carried module kullback/agent/session/ | the session tree stands; the outside evaluator half goes with the skills gate (D132) |
| D65 | The context cap applies to the harness's own model calls, never to the Candidate | carried module kullback/agent/ (compaction) | the Candidate always runs under the production setting |
| D117 | The model writing a body gets a row lookup and a self-test, both inside the sandbox | base tool (read, bash over env/) | the one-shot writer's two affordances become the Builder's own base tools; the decision entry was removed from the log on 2026-09-22, learning in learnings.md section 2 |
| D134 | The compaction floor fences the entries it summarises, and notes are not instructions | carried module kullback/agent/ (compaction) | the notes sentence is a prompt section of every extension |
| D175 | A read answers an index for a whole file and is never longer than the read limit | base tool (read) | registered for the Examiner over exam/ |
| D178 | The floor cuts a protected result that is alone over the line, and the harness takes the model's window | carried module kullback/agent/ (compaction) | the Examiner's search half becomes the grep and find base tools |
| D245 | Single-message stages send a stable system head | prompt section of every extension | a stable head is what the prompt cache keys on |
| D266 | The harness's own agents read recordings and Runs through one set of tools | base tool (read, grep, find, ls) | ADR-0011 makes this the base tool set of the core |
| D267 | One workdir store, and an append-only journal | carried module kullback/store.py | the journal becomes the one bus, append-only and sequence-numbered |
| D269 | Prompts grow by appending, and the head stays stable | prompt section of every extension | |
| D272 | Every tool call is a transaction, and a crash is its own outcome | carried module kullback/agent/ (the tool call path) | the wrapper is in the core, the rollback is the runner's world |
| L1.4 | One snapshot, written once, and every search that can end without a ruling writes a reason | carried module kullback/report/ (one bus subscriber) | the round goes; the snapshot rule survives as a subscriber, an empty record is still a bug |
| L7.1 | Feedback that is delivered is not feedback that is usable: a ruling carries the rows that differ | hook (the tool_result hook appends the ruling) | learnings.md section 7, airline rounds 6 and 7 |
| L7.2 | A repair verb that calls a fresh one-shot writer is not an edit | base tool (edit) | the Builder edits the body itself and keeps what it tried in its own transcript |
| F2 | "we don't let the context to increase beyond 40% hard constraint" | carried module kullback/agent/ (compaction) | founder, 2026-08-29 |
| **kullback/builder (ingest, derive_world, examine, grow, prompts, skills)** | | | |
| D13 | Reconstruct the starting state from the recorded reads, then apply the recorded writes | carried module kullback/builder/readers.py | part of derive_world |
| D14 | The system prompt and the policy compile into hard constraints, checked on the Reference first | carried module kullback/builder/policy.py | |
| D15 | The Intent is written from the trace with a grounding pass, and every clause cites a span | carried module kullback/builder/intent.py | |
| D16 | Filtering is keep and tag, two gates only | carried module kullback/builder/ingest.py | |
| D33 | The Environment is built from the whole trace corpus, not from one Run | carried module kullback/builder/mine.py | |
| D36 | Runs become Tasks by write-tool partition, then Intent clustering | carried module kullback/builder/cluster.py | The decision entry was removed from the log on 2026-09-22, learning in learnings.md section 3 |
| D37 | Sources of Tasks beyond Replicas: Scenarios are generated, never counted | carried module kullback/builder/synth.py | the grow tool |
| D40 | Synthetic rows for the reads are generated by observing the real data in the traces | carried module kullback/builder/synth.py | never in a trusted Task |
| D42 | Every value in a write effect carries a provenance, and provenance decides required or allowed | carried module kullback/builder/effects.py | the atom class is read in kullback/examiner/derive.py |
| D66 | Raw traces are the source of truth, stored byte for byte and content-hashed | carried module kullback/builder/ingest.py | |
| D67 | Tool errors are normalised at ingest into a small fixed taxonomy plus the verbatim payload | carried module kullback/builder/ingest.py | |
| D68 | A tool's read or write class is proposed by a model and overridden by computed effect evidence | carried module kullback/builder/effects.py | |
| D70 | The default class for a tool with no effect evidence and low confidence | carried module kullback/builder/effects.py | |
| D71 | User-side writes enter the End state only when the traces record them | carried module kullback/builder/effects.py | provisional; widened by D176 and by the witness rule of learnings.md section 7 |
| D72 | A tool result schema is the union of everything observed | carried module kullback/builder/mine.py | |
| D73 | Column class: code proposes by rule, a model verifies per column | carried module kullback/builder/mine.py | |
| D76 | A policy rule that does not compile is rewritten for approval first and becomes a judge atom second | carried module kullback/builder/policy.py | open under D255: which rules stay diagnostic |
| D83 | Tasks have a hierarchy, a Category above a Task | carried module kullback/builder/cluster.py | The decision entry was removed from the log on 2026-09-22, learning in learnings.md section 3 |
| D87 | The Builder carries lessons between customers and must question each one's relevance | prompt section of builder | |
| D95 | A truncated tool result is asked for again, reconstructed meanwhile, tagged and assisted | carried module kullback/builder/ingest.py | |
| D96 | Coverage is counted in Tasks, against the Task list fixed at the start | carried module kullback/builder/cluster.py | |
| D98 | The name rule for a generic tool, and what a generic tool is never credited with | carried module kullback/builder/mine.py | |
| D99 | Only system time and counter names are exempt, at high confidence | carried module kullback/builder/mine.py | correction to D73 |
| D100 | Cluster similarity is idf-weighted token Jaccard, not a raised threshold | carried module kullback/builder/cluster.py | |
| D101 | A tool's kind comes from what its calls show, never from a verb list | carried module kullback/builder/mine.py | |
| D102 | An id is a column the calls address and that is distinct per row | carried module kullback/builder/mine.py | |
| D103 | A tool is about the noun before the preposition | carried module kullback/builder/mine.py | |
| D104 | A mined pattern nothing reads is a bug, not a feature | carried module kullback/builder/mine.py | the miss that found it is the reason mine is measured |
| D106 | A miss with signal in the traces is mined, not prompted around | standing rule (docs/learnings.md section 3) | the design rule behind every mining module |
| D107 | The starting state grows from the rows the traces showed, by structural rules only | carried module kullback/builder/readers.py | part of derive_world |
| D114 | The Environment refuses what the real system refuses, and a rule the recordings break is a residual | carried module kullback/builder/policy.py | |
| D115 | What a user states comes from the corpus; the web adds only the words agents ask with | carried module kullback/user/vocabulary.py | mined by the Builder, held by the user extension |
| D118 | The Builder's independent model calls run on a few threads, and nothing about an artifact moves | carried module kullback/builder/parallel.py | The decision entry was removed from the log on 2026-09-22, no learning |
| D143 | Intents ground in two steps over a splitter that keeps numbers, money and ids whole | carried module kullback/builder/intent.py | |
| D147 | The compiler names the shape of what a column holds | carried module kullback/builder/mine.py | "you raised this last time" is now the Builder's own transcript |
| D150 | The Builder loads a triage skill written from where it kept failing | prompt section of builder (skills) | The decision entry was removed from the log on 2026-09-22, no learning |
| D157 | The leak check reads every seed recording's user, and a tool argument grounds only what a user said | gate artifacts (the Intent leak check) | |
| D164 | A tool has callers, and a name the recording refused on every call is not a tool | carried module kullback/builder/mine.py | |
| D167 | An id shape is a pattern an ordinary word cannot match | carried module kullback/builder/mine.py | |
| D168 | The body skill: what a body does, read off where bodies kept failing | prompt section of builder (skills) | |
| D176 | A requestor's prose results are read into rows of the one world, marked by the requestor | carried module kullback/builder/readers.py | |
| D179 | A row whose identity is more than one column is keyed by the columns the call was asked for | carried module kullback/builder/mine.py | |
| D180 | A row is homed by the id the call asked for, and only hard columns split a Task | carried module kullback/builder/readers.py | |
| D188 | Nested rows and homed partial reads are pinned per Task | carried module kullback/builder/readers.py | |
| D196 | Intents are written symptom-only, and the leak check audits the strip | carried module kullback/builder/intent.py | |
| D200 | The frozen Task list is authoritative across the whole build | carried module kullback/builder/cluster.py | |
| D202 | A column a Task only ever writes is pinned by running the write's own body backwards | carried module kullback/builder/readers.py | |
| D203 | A tool whose prose nobody reads gets a reader of its own | carried module kullback/builder/readers.py | |
| D207 | A nested row is named by its own scope | carried module kullback/builder/readers.py | The decision entry was removed from the log on 2026-09-22, learning in learnings.md section 1 |
| D209 | Every Task carries an objective difficulty record, and coverage is reported per bucket | carried module kullback/difficulty.py | moves under the builder extension |
| D212 | Every sample the harness draws is a function of the id and a build salt | carried module kullback/sampling.py | |
| D216 | The split of Runs into Tasks is a function of the recordings alone | carried module kullback/builder/cluster.py | the identity learning of learnings.md section 3 |
| D220 | A build stage draws from the seed Runs, and the world keeps what only a held-out Run witnessed | carried module kullback/builder/mine.py | the held-out filter is on evidence, never on value |
| D224 | Synthetic Tasks are walked over the mined tool-call graph and reported apart | carried module kullback/synthesise.py | the grow tool |
| D259 | A second, controllable synthetic Task system, checked by code at birth and judged at run time | carried module kullback/synthesise.py | the grow tool; difficulty is D209 and the run-time half is open |
| D225 | Task archetypes are read off the domain's own public material, and walks are shaped to them | base tool (web_search) | the grow tool shapes its walks to what the search found |
| D233 | Caller-side state is homed to a row like every other sighting | carried module kullback/builder/readers.py | |
| D234 | An ambiguous effect is credited to the write that owes it, not to the last call | carried module kullback/builder/effects.py | |
| D260 | A Task is what the customer wanted and the world it started in, never what the agent did | carried module kullback/builder/cluster.py | |
| D261 | One intake seam with many telemetry formats behind it | carried module kullback/builder/sources/ | |
| D263 | Intake rules on each recording on its own, rescues what it can, keeps the rest as evidence | carried module kullback/builder/ingest.py | |
| D268 | A merge of two recordings is a proposal, and code proves it before it is a Task | carried module kullback/builder/cluster.py | |
| D277 | A mined fact carries the calls that support it, and a name is only a starting assumption | carried module kullback/builder/mine.py | |
| L3.5 | Evidence shown to a writer is filtered by held-out membership, never by value | carried module kullback/builder/mine.py | holds for effect evidence too |
| L7.3 | Findings that land one round late pay for a round in which nothing can move | prompt section of builder (the Builder calls examine when it judges the Environment ready) | |
| L7.5 | A thinner sighting of a row is not a new version of it | carried module kullback/builder/mine.py | 502 subset rows, 53 export conflicts, a gate tied for sixteen recompiles |
| L7.6 | Every requestor's call through a listed tool is a sighting of the world | carried module kullback/builder/mine.py | with kullback/builder/readers.py; 42 of 51 calls dropped in one Task before the fix |
| F7 | "there has to be a loop which runs so that the harness can steer the creation of the environment" | prompt section of builder | founder, 2026-08-29; the autonomous Builder is this loop |
| F8 | "loop over its environment run it figure out where the problem is and then solve it" | prompt section of builder | founder, 2026-08-29; with the base tools this is now literal |
| F12 | "keep the good recordings and try to learn about the structure of the environment from the broken recordings" | carried module kullback/builder/ingest.py | founder, 2026-09-18 |
| F20 | "we need to control the difficulty based on what the model under training is able to solve" | carried module kullback/difficulty.py | founder, 2026-09-18; the second half is still open |
| **kullback/examiner** | | | |
| D10 | The frontier can be wrong, so a recorded Run becomes a Reference only after confirmation | carried module kullback/examiner/reference.py | |
| D11 | The Verifier is anchored to the Intent, not to the Reference End state | carried module kullback/examiner/derive.py | |
| D19 | We do not have gold: the approximation is the confirmed Reference plus re-roll agreement, published with its confidence | carried module kullback/examiner/derive.py | |
| D30 | The Verifier says what actions a Task expects, and that expectation comes from production traces | carried module kullback/examiner/derive.py | |
| D57 | Confirming a Reference passes several gates, with a judge that marks pass or fail and hands unsure to a person | carried module kullback/examiner/reference.py | |
| D93 | A disputed Reference sets the Task aside until a person resolves it | carried module kullback/examiner/reference.py | The decision entry was removed from the log on 2026-09-22, learning in learnings.md section 2 |
| D111 | A Run is a Reference when its End state is what the Intent plus the policy say should have happened | carried module kullback/examiner/reference.py | the benchmark reward is read only by the report |
| D123 | Two agents on one core: one builds the Environment, one writes the Verifiers and the probes | carried module kullback/examiner/session.py | the separation stands (ADR-0007); the queue messaging is replaced by the Builder's examine tool over one bus |
| D156 | A no-write recording has an outcome: whether its answer stated facts it read | carried module kullback/examiner/reference.py | |
| D163 | The Examiner derives per Task, in parallel, from a cache | carried module kullback/examiner/derive.py | The decision entry was removed from the log on 2026-09-22, no learning |
| D170 | Findings are filed by rule off the records, ranked by the Tasks they cost | carried module kullback/examiner/findings.py | the finding now carries per-call rows |
| D172 | A Task with no Reference is unfinished until it is refused with a reason | carried module kullback/examiner/lifecycle.py | |
| D182 | A stated fact is required only when it was asked for, and list order is not an End state | carried module kullback/examiner/derive.py | |
| D190 | A required atom's value is one a user said, and a read-only Task can be falsified | carried module kullback/examiner/derive.py | |
| D198 | A residue is settled by deriving a Verifier per surviving End state | carried module kullback/examiner/derive.py | a fail-only judge abstained 13 of 13 |
| D199 | When re-rolls find no second path, one is written from the Reference Run | carried module kullback/examiner/variants.py | |
| D206 | A shape atom is checked against what the Run read, not against the row it writes | carried module kullback/examiner/derive.py | |
| D208 | A derived artefact is retired when its source is withdrawn | carried module kullback/examiner/lifecycle.py | |
| **kullback/user** | | | |
| D27 | Simulated users come from production traces, and the error rate is published per Task | carried module kullback/user/rules.py | |
| D44 | The simulated user is exact on this user's facts and representative on everything else | carried module kullback/user/rules.py | |
| D77 | The Simulated user answers from the world and never invents | carried module kullback/user/guards.py | |
| D151 | The user's facts are mined from the arguments the recorded agent used, not only from its words | carried module kullback/user/rules.py | |
| D158 | The Simulated user restates its goal once before it closes | carried module kullback/user/rules.py | |
| D210 | The Simulated user ends by protocol and answers from typed facts | carried module kullback/user/rules.py | |
| D214 | The Simulated user is an agent of the harness, with its own context, guards, score and lessons | carried module kullback/user/ (the extension) | it keeps my_facts, my_goal, what_i_said and end_run, and no base tools |
| D232 | A build names the model that drives the Simulated user | carried module kullback/user/factory.py | |
| D256 | One factory builds the Simulated user, and every user turn leaves one typed record | carried module kullback/user/factory.py | |
| D264 | A user population per Task, then new users by controlled variation | carried module kullback/user/population.py | |
| D265 | User signals live in one typed record with one owner per field | carried module kullback/user/signals.py | |
| D270 | Examples in the Simulated user's prompt are mined per corpus by situation | carried module kullback/user/examples.py | |
| F15 | "unfinished recordings will give us the user signals, but for running we shall use the runs which are complete" | carried module kullback/user/signals.py | founder, 2026-09-18 |
| F16 | "both, but with one source of truth. Harness and the code reads the signals" | carried module kullback/user/signals.py | founder, 2026-09-18 |
| **kullback/runner (the tool: world, route, Verdict, version hash)** | | | |
| D01 | The Step is screened, the Run is verdicted | runner tool | |
| D02 | Match against the Reference and adequacy against the task, with a judge only where they disagree | runner tool | |
| D03 | The Verdict is on the End state, not the path and not the reasoning | runner tool | |
| D06 | A high-fidelity Environment is the centre: no Environment, no Verdict | runner tool (the world) | with gate fidelity |
| D09 | An off-path call is not a hallucination, and an off-path Run is assisted, never failed | runner tool (route) | |
| D12 | A tool-equipped judge rules on novel End states and can never award a pass | carried module kullback/runner/judge.py | |
| D20 | The reward basis is fixed for every Replica: state fingerprint, atom list, communicate matches, hard constraints | runner tool (Verdict) | |
| D22 | Exact means exact after canonicalisation | carried module kullback/runner/canon.py | |
| D25 | Judges stay, as grounded rubrics, in three places only | carried module kullback/runner/judge.py | |
| D26 | The Environment's write log is the transaction log, and fingerprints are computed from it | carried module kullback/runner/state.py | |
| D31 | Learn from every reward basis and assemble the best version | runner tool (Verdict) | |
| D32 | The Environment keeps an append-only write log per Run | carried module kullback/runner/state.py | The decision entry was removed from the log on 2026-09-22, no learning |
| D43 | A Verdict grades effects, and the End state includes what the user was told or asked | runner tool (Verdict) | |
| D45 | Hallucinated tool calls are counted, not failed; a fabricated result always fails | runner tool (route and Verdict) | |
| D46 | A failure is explained by its failing atom, computed by code | runner tool (Verdict) | |
| D61 | The Runner is frozen and the Builder improves | runner tool (version hash) | the hash now covers the runner's code and the core loop it ran on |
| D74 | One world per customer, with the starting state per Task as an overlay | runner tool (the world, from kullback/runner/world/loading.py) | |
| D78 | The re-run count is decided by experiment, not by design | runner tool (reroll) | |
| D81 | The held-out anchor is a share of every Task's Runs | runner tool (holdout, in the workdir outside every agent's root) | |
| D84 | A semantic column is settled by the judge only when the canonical strings differ, with a cached equivalence table | carried module kullback/runner/judge.py | |
| D88 | Failure attribution is code first, a judge on the rest | carried module kullback/runner/verdict.py | |
| D89 | The Candidate sees nothing that came from the Verifier | carried module kullback/runner/boundary.py | |
| D90 | The loop is written so it can be stepped | runner tool (run, replay, reroll) | the tau2 shape and the OpenEnv wrapper were first-build scope |
| D92 | Agentic judges by default, and the pipeline runs without human support | carried module kullback/runner/judge.py | |
| D94 | End state only, reaffirmed | runner tool (Verdict) | |
| D108 | The Reference is the Trace replayed through the built tools, inside the build | runner tool (replay) | |
| D110 | Code verifies what code can verify, a judge only where it cannot, and it never awards a pass | runner tool (Verdict) | |
| D112 | Three extra Runs per Task by default | runner tool (reroll) | the scorecard beside it was scaffolding |
| D119 | A Reference stopped one step short must score no pass | gate verifier_suite | the unsolved-state check is one of the D79 negatives |
| D120 | The Runner is a tool the agent calls, not an agent | runner tool | reaffirmed by ADR-0011: run, replay, reroll |
| D148 | Arithmetic in a body goes through a code-owned evaluator | carried module kullback/runner/arith.py | |
| D159 | A Run records the call it made, not only the answer it got | carried module kullback/runner/records.py | |
| D160 | The judge model is chosen, named, and may be two different models | carried module kullback/runner/judge.py | |
| D177 | A Task's re-rolls are keyed on the Task's own inputs | runner tool (reroll) | The decision entry was removed from the log on 2026-09-22, no learning |
| D186 | A judge failure cites a value the states differ on, in code, or it abstains | carried module kullback/runner/judge.py | |
| D187 | The scoring path honours column classes, and keyed lists are unordered | runner tool (Verdict) | |
| D189 | A lone Reference is re-rolled for a second path, bounded | runner tool (reroll) | |
| D191 | One scorer, and the calls the Reference replay fails on are evidence | runner tool (Verdict) | |
| D193 | The judge's residue gets a second pass or an abstention | carried module kullback/runner/judge.py | |
| D197 | A row read more than once in a Task is served its recorded values in call order | runner tool (the world) | The decision entry was removed from the log on 2026-09-22, learning in learnings.md section 1 |
| D204 | The replay cursor takes a run of consecutive same-role turns as one logical turn | runner tool (replay) | |
| D213 | When a Task's own Runs disagree on a row, each Run replays against its own sighting | runner tool (the world) | |
| D217 | A cosmetic verdict needs the same values on both sides | runner tool (Verdict and canon) | |
| D219 | A comparison nobody settled is not agreement | runner tool (Verdict) | |
| D222 | The check a judge needs is run by the harness before the model is asked | carried module kullback/runner/atom_context.py | |
| D223 | A Run's transcript claims are checked against its state | carried module kullback/claims.py | moves under the runner |
| D230 | A tool asked about a Task it cannot serve answers with that Task's state | runner tool (the world) | the stall half goes with the round |
| D240 | A Task's re-roll Runs run side by side in Run number order | runner tool (reroll) | The decision entry was removed from the log on 2026-09-22, no learning |
| D252 | The architecture answers to replay fidelity and to a Run anyone can drive | runner tool (run, replay, reroll) | the Run interface is the runner tool's face |
| D253 | The step is split into asking a policy and advancing the world | runner tool | the policy side is now the core loop, the world side is the runner |
| D254 | A Verifier has one scorer, and the structured atom target is the only meaning of an atom | carried module kullback/runner/target.py | |
| D255 | Anything in the reward that reads text is grounded in facts | carried module kullback/runner/target.py | open: founder decision 1 on atom classes, carried unchanged |
| D258 | Environment consistency: laws that hold for any tool, checked off the recorded path | carried module kullback/laws.py | moves under the runner as part of the world |
| D262 | A tool that can be run for real is run for real, and the grader is out of the Candidate's reach | runner tool (route: real, code, recording, stand-in) | |
| D271 | Time, randomness and new ids reach a body through one context | runner tool (the world's context) | |
| D273 | A training Run never uses the model stand-in | runner tool (route) | |
| L1.1 | Check what code a build ran on before attributing a failure to an open class | runner tool (version hash beside every Verdict) | six of twelve classes in one reading were already closed |
| L6.2 | The unresolved semantic route forgives calls and needs its own route and count | runner tool (route) | open, 387 telecom calls |
| L7.4 | A scorer that fails every Run on an atom no code can evaluate zeroes trust everywhere | carried module kullback/runner/target.py | open: founder decision 1; 54 of 54, 194 of 194, 26 of 26 |
| F9 | "Run it for real, have the bash tool with it. We should always run it for real." | runner tool (the real route) | founder, 2026-09-18; the bash base tool is the other half |
| F10 | "end state, be as close as possible to the truth" | runner tool (Verdict) | founder, 2026-09-18 |
| F11 | "The customer is not going to say which tools to run or imitate" | runner tool (route) | founder, 2026-09-18; the harness rules per tool from the recordings |
| **kullback/gates (bound to paths)** | | | |
| D07 | Only Replicas count toward the bar, and a synthesised Scenario never does | gate trust | |
| D08 | Trust gates are pass or fail, no blends | gate trust | |
| D29 | The holes in a traces-only Environment are enumerated with a fill for each | gate fidelity | the assisted share per tool decides which fills are needed |
| D39 | Writes are exact, reads are classified cosmetic or semantic | gate fidelity | |
| D41 | Representativeness is a hard constraint on everything the harness generates | gate trust | |
| D49 | A Task that cannot be replayed is not gradeable, and it says why | gate trust | |
| D79 | Verifier validation is automated checks first, humans only on hits | gate verifier_suite | |
| D113 | A miscompiled constraint is demoted by the recordings, and an overlay conflict fails the export | gate artifacts | |
| D122 | Gates are a package no agent can write, and a gate is code, generic, and rules on both agents' work | gate (the registry) | ADR-0007; in the new shape a gate is bound to the path that is written |
| D127 | A Verifier may be tightened freely and loosened only toward what the frontier did | gate loosening | probes are monotone |
| D132 | A skill edit is tentative until a paired sequential test is decisive, and accepted skills are re-checked | gate skills | |
| D133 | The legitimate pool is expandable, k is adaptive, and false rejection is measured per Task | gate loosening | |
| D154 | A fidelity ruling names the leaf where two answers part, not only the column | gate fidelity | |
| D155 | The recording is the standard: the body reproduces it, and a finding tells the customer | gate fidelity | |
| D162 | A tool body may not memorise the recordings | gate body gates (non_trivial) | |
| D165 | A tool whose answer changed under identical arguments is state-driven | gate body gates (non_trivial) | |
| D171 | Tool fidelity is attributed per Task as well as per corpus | gate fidelity | |
| D173 | Three of the D79 checks were wrong about a Verifier that never writes | gate verifier_suite | and the false-rejection pool must not count Runs the Reference rule threw out |
| D174 | A recompile replaces the kept body only by beating it | gate body gates | The decision entry was removed from the log on 2026-09-22, no learning |
| D183 | A prose result is rendered back out of the row and the round trip is the gate | gate body gates | |
| D184 | The previous body competes on equal terms, and a tie keeps it | gate body gates | |
| D185 | The held-out pool holds only the Runs that reached the Reference | gate trust | |
| D194 | False rejection over the held-out pool gates trusted | gate trust | |
| D195 | A body that reads state answers differently when that state differs | gate body gates (sensitivity) | |
| D201 | A repair lands only if it fixed what it targeted and cost nothing elsewhere | gate body gates (transactional landing) | |
| D205 | An over-strict Verifier is loosened by the harness itself, gated as any repair | gate loosening | |
| D211 | A stalled body is given every differing leaf, the relation across failing calls, and the lines nothing witnessed | gate body gates (the ruling carries the rows that still differ) | the stall limit goes with the recompile loop; the dense ruling is the point |
| D215 | A write is judged on every row it changed, not only on the answer it returned | gate fidelity (effect check) | |
| D227 | A write counts as made only when its result shows it took effect | gate fidelity | |
| D250 | Compile hold-out is by argument shape, and the body test faces that split | gate body gates | |
| D257 | A tool body is accepted on its answers and on its effects | gate body gates | |
| L2.1 | The funnel is fidelity, Reference confirmed, Verifier derived, the D79 suite, false rejection, trusted | gate trust | |
| L2.3 | Every stage of the funnel loses Tasks for reasons unrelated to the Task's quality, so each one records why | gate trust | with the report |
| L3.3 | Strictness must hold at every layer independently | gate (every layer keeps its own) | a seeding gap, an unwitnessed read and a lenient comparer are three gates, not one |
| L7.7 | A corpus that builds on an empty world with no gate failing is worse than one that refuses to build | gate (the empty-world gate, new) | one tool, zero tables, zero columns, 64 Tasks with no intent |
| F6 | "the environment which is created should be 100 exact match from the tool names to everything" | gate fidelity | founder, 2026-08-26; exact after canonicalisation per D22 |
| F14 | "check for reward hacking as well" | gate (the cheat suite) | founder, 2026-09-18; the Run-side half of D263 |
| **kullback/cli, tui, report (bus subscribers)** | | | |
| D04 | The customer metrics are few and each one earns its place | carried module kullback/report/numbers.py | |
| D05 | The bar is non-inferiority, not superiority | carried module kullback/report/numbers.py | |
| D47 | The Intent is shown to the customer as the Task name, and they can correct it | carried module kullback/report/render.py | |
| D85 | The report shows whether the Environment was built and the numbers; it suggests, the person decides | carried module kullback/report/render.py | |
| D105 | The screen is the pipeline, not a chat | carried module kullback/tui/ | the pipeline view becomes a view over the bus; the decision entry was removed from the log on 2026-09-22, learning in learnings.md section 5 |
| D139 | One table per build, printed by one script, with the detail underneath | carried module kullback/report/ | The decision entry was removed from the log on 2026-09-22, learning in learnings.md section 5 |
| D140 | The status report shows every red light, grouped, and nothing is cut at a line count | carried module kullback/report/render.py | The decision entry was removed from the log on 2026-09-22, learning in learnings.md section 5 |
| D221 | An Environment is published as a package of the rebuilt world and its graders, and fetched back verified | carried module kullback/hub/ | |
| L5.4 | A report carries task ids, tool names and column names only | carried module kullback/report/ | customer data reached reports twice and was scrubbed |
| **standing rules (people, not code)** | | | |
| D17 | Simplicity by default, and complexity where it is worth it | standing rule (docs/design-philosophy.md) | |
| D18 | Study real environments first, so the generator has a reference set | standing rule (docs/design-philosophy.md) | build order of the first build, kept as history; the decision entry was removed from the log on 2026-09-22, no learning |
| D21 | A good Task is one where two domain experts would independently reach the same verdict | standing rule (docs/decision-log.md principles) | |
| D23 | Unbiased reporting: say where the sources disagree and where coverage is thin | standing rule | |
| D24 | Build for the hardest case | standing rule | |
| D28 | Deferrals are written down with their reason | standing rule (docs/todo.md) | the list was cleared on 2026-09-22 and is re-earned against the rebuilt harness; the decision entry was removed from the log on 2026-09-22, no learning |
| D34 | The founder's words are kept verbatim | standing rule (docs/founder-words.md) | |
| D35 | Environment data follows customer trust | standing rule (docs/adr/0006) | |
| D38 | Assumptions are registered, graded and tested | standing rule (docs/assumptions.md) | |
| D48 | Two human checks: setup review before any Verdict, blind audit after | standing rule | |
| D50 | The first build must prove faithful replay and agreement with the customer's own outcome | standing rule | The decision entry was removed from the log on 2026-09-22, no learning |
| D51 | Generalizability over overfitting | standing rule (house rule: every fix is general) | |
| D52 | Real-world tool-using traces, not coding traces, are the first input | standing rule | the coding domain reopened as D262 and AgentTrove; the decision entry was removed from the log on 2026-09-22, no learning |
| D53 | The harness is layered like tau: a provider layer, a core, and the faces | standing rule (docs/architecture.md) | the layering is now the shape of ADR-0011 |
| D55 | The smallest slice runs all the way through the Verdict | standing rule | The decision entry was removed from the log on 2026-09-22, no learning |
| D56 | Emit a known shape first, then mould it into our own | standing rule | The decision entry was removed from the log on 2026-09-22, no learning |
| D58 | The customer intake questions live in one place | standing rule (docs/trace-intake.md) | |
| D59 | The harness is the whole system, the Builder makes the Environment, the Runner re-executes | standing rule (the glossary) | |
| D60 | Self-generated traces must not be biased toward the Builder | standing rule | The decision entry was removed from the log on 2026-09-22, no learning |
| D97 | Remaining details are closed by defaults and revised by measurement, not by more grilling | standing rule | The decision entry was removed from the log on 2026-09-22, no learning |
| D121 | Packages depend one way only, and the frozen code is hashed | standing rule (docs/architecture.md) | the package set grew; the one-way rule and the hash stand |
| D129 | The package is kullback at the repo root, and harness retires as a code name | standing rule (docs/architecture.md) | |
| D278 | What the grill removes, and what stays | standing rule (docs/decision-log.md) | |
| L1.2 | Readers overstate, builders measure: a reader's count is a premise to be checked | standing rule | |
| L1.3 | Absolutes come from live builds only, deltas may come from copies, and each says which | standing rule | |
| L3.1 | Every rule keys on a class the harness mines, never on a corpus, tool, table or column name | standing rule (house rule) | |
| L3.6 | Counters over claims: a mechanism whose counter reads zero is shipped and unexercised, not working | standing rule | |
| L4.1 | Outside work is read, named and taken as decisions, never adopted wholesale | standing rule (docs/papers.md) | |
| L5.1 | Builds never run from a temporary directory | standing rule | |
| L5.2 | Long suites run in chunks, in the background, and are polled | standing rule | |
| L5.3 | One merge agent lands branches serially, and review findings are answered, never ignored | standing rule | |
| L5.5 | A model change makes the next round a new baseline, not a harness delta | standing rule | |
| L5.6 | After a re-freeze the first reading is of the counters the frozen decisions promised | standing rule | |
| L6.1 | Effect checks make fidelity stricter before they make it higher | standing rule (open) | |
| L6.3 | The open items of section 6 are re-earned against the rebuilt harness | standing rule (docs/todo.md) | |
| L7.8 | A claim stays a claim until it is reproduced on the code that was actually merged | standing rule | where the readings of 2026-09-22 disagreed |
| F1 | "make it the tool as close to reality as possible" | standing rule | founder, 2026-08-26 |
| F3 | "keep it simple doesn't mean that where we need complexity we don't add that" | standing rule | founder, 2026-08-26 |
| F4 | "we need to care much more about generalizability than the overfitting please" | standing rule (house rule) | founder, 2026-08-27 |
| F5 | "Keep everything of my words please, those are the real decisions as well" | standing rule (docs/founder-words.md) | founder, 2026-08-27 |
| F13 | "We need to be very very honest and strict with ourselves to design this much much better" | standing rule | founder, 2026-09-18 |
| F17 | "as the architecture improves the fidelity will definitely improve" | standing rule | founder, 2026-09-18 |
| F18 | "if the experiments come well they go into the harness" | standing rule | founder, 2026-09-18 |
| F19 | "all the improvements we discussed, and then fidelity checks, and then we test on AgentTrove" | standing rule | founder, 2026-09-18; the order of work |
| F21 | "the overhaul carves, never rewrites; every brief names the decisions and learnings it carries" | standing rule (DESIGN.md, this ledger) | founder, 2026-09-22; founder-words.md has no session for that day |
| F22 | Boundaries have two levels only, cannot and should not, and the shell is the one tool that is many | standing rule (DESIGN.md) | founder, 2026-09-22; enforced by registration and by the bash allowlist |
| **hooks and prompt sections** | | | |
| D62 | The Builder's stop condition is a measured number, not a turn count | prompt section of builder (the stop rule) | The decision entry was removed from the log on 2026-09-22, no learning |
| D86 | When the spend ceiling is hit, stop, report as is, and ask before continuing | hook (the spend hook) | |
| D125 | Skills are the prompts the model may rewrite, versioned and accepted by a gate | prompt section of builder (skills) | with gate skills |
| D138 | There is no turn cap; the dollar ceiling is the wall | hook (the spend hook) | the mechanic it was written for is gone |
| D144 | Agent prompts follow the GEPA order and carry general examples; feedback is text, not a score | prompt section of every extension | |
| D276 | The stop rule reads the numbers the loop is for, and records a reason every time | prompt section of builder (the stop rule) | |
| **dropped** | | | |
| D54 | The build follows four fixed stages | dropped | removed from the decision log on 2026-09-22: no learning |
| D69 | Which missing components enter the first build, and the Builder may modify itself | dropped | removed from the decision log on 2026-09-22: no learning |
| D75 | Tool body repair is bounded retries with growing evidence | dropped | removed from the decision log on 2026-09-22: learning kept in learnings.md section 7 |
| D80 | Scorecard thresholds measured on tau2, with every miss explained | standing rule (the target is 100 percent, every miss explained) | the tau2 scorecard is scaffolding (D112); the rule that a miss without a recorded reason is a gate failure stands |
| D82 | The Builder changes one thing per round, batches later | dropped | removed from the decision log on 2026-09-22: no learning |
| D91 | The Verifier code lives in the Builder and reaches the Runner through records only | dropped | removed from the decision log on 2026-09-22: no learning |
| D109 | Every stage hashes the modules it delegates to | dropped | removed from the decision log on 2026-09-22: learning kept in learnings.md section 1 |
| D124 | The model manages its own context with forget, recall, load and unload, over a code floor | dropped | removed from the decision log on 2026-09-22: learning kept in learnings.md section 4 |
| D126 | A round is four beats ending at gates | dropped | removed from the decision log on 2026-09-22: learning kept in learnings.md section 5 |
| D128 | The two agents take turns on one event stream | dropped | removed from the decision log on 2026-09-22: no learning |
| D130 | Seven phases, each small, each leaving the artifacts byte-identical until the phase that changes them | dropped | removed from the decision log on 2026-09-22: no learning |
| D131 | Research on the context manager: dependency guards, protected tool output, capped loaded tools | dropped | removed from the decision log on 2026-09-22: learning kept in learnings.md section 4 |
| D135 | The model is the driver and the graph refreshes stale inputs | standing rule (ADR-0011: the model is the driver) | the Builder reads the rulings and decides what to call next; there is no scheduler and no stage graph |
| D136 | The mechanic repairs what the model wrote | standing rule (ADR-0011: the boundary) | the agent repairs only what a model wrote; gates, the runner, the Verifiers and the user rules live outside every agent root and are fixed by people |
| D137 | The 2026-09-06 build measures the mechanic on the Tasks that write | dropped | removed from the decision log on 2026-09-22: no learning |
| D141 | The Examiner's finding names the Builder verb and carries a hint | dropped | removed from the decision log on 2026-09-22: learning kept in learnings.md section 7 |
| D142 | A round has moved when a gate count, an artifact or a ruling changed | dropped | removed from the decision log on 2026-09-22: learning kept in learnings.md section 5 |
| D145 | A repair result opens with its own target's ruling | dropped | removed from the decision log on 2026-09-22: learning kept in learnings.md section 7 |
| D146 | The Intent stage ratchets, and a repaired file moves the cache key | dropped | removed from the decision log on 2026-09-22: no learning |
| D149 | A Task the re-rolls stage skips loses the re-rolls an earlier build left it | dropped | removed from the decision log on 2026-09-22: no learning |
| D153 | The driver builds the target when the model repaired and never called build | dropped | removed from the decision log on 2026-09-22: no learning |
| D161 | The driver builds the target again after a narrowed repair | dropped | removed from the decision log on 2026-09-22: no learning |
| D166 | A round is not done until the target was built | dropped | removed from the decision log on 2026-09-22: learning kept in learnings.md section 5 |
| D169 | Two more exits: fidelity flat for k rounds, and a round cap | dropped | removed from the decision log on 2026-09-22: learning kept in learnings.md section 5 |
| D181 | Nine repairs to the Builder's and the Examiner's loop | dropped | removed from the decision log on 2026-09-22: learning kept in learnings.md section 2 |
| D192 | Five loop repairs read off the second watch | dropped | removed from the decision log on 2026-09-22: learning kept in learnings.md section 7 |
| D218 | A round's Task rows are written once, at round close | dropped | removed from the decision log on 2026-09-22: learning kept in learnings.md section 1 |
| D231 | Round accounting survives a raised beat | dropped | removed from the decision log on 2026-09-22: learning kept in learnings.md section 5 |
| D246 | The compile heads are built once per build and the helper specs ride one object | dropped | removed from the decision log on 2026-09-22: learning kept in learnings.md section 1 |
| D274 | A stage's cache key is computed from the code it imports | dropped | removed from the decision log on 2026-09-22: learning kept in learnings.md section 1 |

## Counts

314 rows: 264 decisions (D01 to D278, every entry of `docs/decision-log.md`), 28 learnings of `docs/learnings.md` that have no decision of their own, 22 standing founder words.

| location | rows |
|---|---|
| carried module | 137 |
| standing rule | 50 |
| runner tool | 41 |
| gate | 39 |
| dropped | 27 |
| prompt section | 12 |
| base tool | 5 |
| hook | 3 |

Twenty-seven rows are dropped, all of them to ADR-0011, and every one of them names the section of `docs/learnings.md` that keeps what it taught or says it carried no learning. Fifty-two decisions were removed from `docs/decision-log.md` on 2026-09-22: those 27 are the ones the new shape has no place for, and the other 25 keep the location they had, because a module, a gate, a tool, a prompt section or a standing rule still carries what the entry taught although the entry itself is gone, and their reason column says so. Nothing else is dropped: the other 287 rows name a module, a gate, a hook, a prompt section, a tool or a document that a later stream has to produce.
