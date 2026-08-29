# Decision log

Chronological record of every design decision for the monitoring tool, with the reason, the alternative that was rejected, and where the decision now lives (glossary term in `/CONTEXT.md`, ADR in `/docs/adr/`, section of `eval-design.md`). Short quotes are my own words; the full, unedited messages are in `founder-words.md`, which is the primary source. This log is the index over them. Append one entry per decision, never rewrite history; if a decision is reversed, add a new entry that points back.

Read this first if you want to know why the tool is the way it is. The principles at the top are distilled from the entries below.

## Principles (distilled)

1. **Reality over proxies.** Re-execute real Runs in an Environment rebuilt from the customer's own traces. "Make the tool as close to reality as possible."
2. **No Environment, no Verdict.** A Run that cannot be re-executed gets a Screen result, never a routing Verdict.
3. **End state is the Verdict.** Not the path, not the reasoning. Path and reasoning are diagnostics.
4. **Trust is measured, not asserted.** Replay fidelity, Verifier validity, judge agreement and human audit rates are numbers the customer sees.
5. **The frontier can be wrong.** A recorded Run is a Reference only once its success is confirmed. Candidates may take other valid paths.
6. **Verifiers are anchored to what the user wanted (Intent), grounded in the trace.** "The end goal is to satisfy the user."
7. **Keep everything, tag everything.** Failed, retried, truncated and chit-chat Runs stay; they are diagnosis, not noise.
8. **Simple by default, complexity where it pays.** "Where we need complexity we add that, if the complexity is worth it."
9. **Unbiased reporting.** Every recommendation says where the research disagrees with it and where coverage is thin.
10. **A good Task is one where two domain experts would independently reach the same pass/fail.**

## Entries

### 2026-08-24 to 2026-08-25

**D01. Unit of grading: Step is screened, Run is verdicted.** Steps (one model call) are compared to the frontier's recorded action as a screen only; Runs are re-executed and verdicted on End state. Reason: static Step matching is unsound after the first divergent action (R06). Rejected: aggregating Step matches into a Run score. Lives in: CONTEXT.md (Step, Run, Screen, Verdict), ADR-0003.

**D02. Both adequacy and agreement, plus a judge for disputes.** "actually both why not right": Match against the Reference and adequacy against the task, with a judge only where the two disagree. Later narrowed (D12) so the judge cannot award a pass. Lives in: CONTEXT.md (Match, Appeal, Clears the bar).

**D03. Verdict on End state, not path, not reasoning.** "We need to be very clear with what we are not doing." Reasoning is monitored for fabrication only (CoT faithfulness 20 to 40%); path differences are reason codes. Counter-evidence (EnvFactory 0.5/0.5, BFCL subset, AgentScaler, ClawTrack) recorded in ADR-0004 on 2026-08-26; it concerns training signal, not pass/fail. Lives in: ADR-0004.

**D04. Minimal customer metrics.** Run pass rate with paired non-inferiority CI; hard-constraint violations; tool selection P/R/F1 plus hallucinated-tool rate; cost per Run and net savings at equal success; pass^k as a conditional gate; Environment fidelity. Everything else internal. Lives in: eval-design.md "Customer-facing metrics".

**D05. Non-inferiority, not superiority.** A Candidate clears the bar when it is not worse than the frontier by more than a pre-registered margin (3 to 5 points), paired per Run, clustered SEs, one-sided 95% CI. Lives in: CONTEXT.md (Clears the bar, The bar), eval-design.md step 12.

**D06. High-fidelity Environment is the center of the product.** "No Environment, no Verdict." Built from traces, tools as code over explicit state, per-tool strategy (R11). Rejected: LLM-simulated tools as the default. Lives in: CONTEXT.md (Environment), eval-design.md step 4.

**D07. Replicas only count toward the bar; Scenarios deferred.** "Let's keep our focus on the replica right now." Synthesized Scenarios never count; they come after Replicas are trusted, for hardening and post-training. Lives in: ADR-0005, CONTEXT.md (Replica, Scenario).

**D08. Trust gates are pass/fail, no blends.** Gate A Environment (replay fidelity), Gate B off-path (assisted Runs get no counted Verdict), Gate C Verifier validity, Gate D frontier sanity. "The environment which is created should be 100 exact match from the tool names to everything." Refined by D22. Lives in: CONTEXT.md (Replay fidelity), eval-design.md "Trust gates".

**D09. Off-path is not hallucination.** An off-path call is a real tool with valid arguments and no recorded response; a hallucinated tool is a nonexistent tool or argument. "Off fidelity tool calls, they are totally valid as well." Off-path Runs are reported "assisted", never failed for being off-path. Lives in: eval-design.md Gate B.

**D10. The frontier can be wrong: Reference confirmation.** "There is always a chance that the frontier is wrong as well." A recorded Run becomes a Reference only after confirmation (customer outcome signal, agreeing frontier re-rolls, or human label). Frontier-failed Runs go to a separate bucket. Lives in: CONTEXT.md (Reference), eval-design.md step 5.

**D11. Verifier anchored to Intent, not to the Reference End state.** "The verifier should check whether the model did a good job, good job meaning did it satisfy the user." Atoms derived from k=4 to 5 frontier re-rolls: required (in every successful re-roll and implied by Intent), allowed (in some), forbidden (write effects in none, unimplied). Five cases handled: different path same End state; frontier wrong; different satisfying End state; novel End state; ambiguous task (re-rolls disagree, Run excluded, ambiguity rate reported). Lives in: CONTEXT.md (Verifier, Intent), eval-design.md step 7.

**D12. Tool-equipped judge for novel End states; it can never award a pass.** "Equipping a model with tools is really required." The judge reads replica state, re-runs the Verifier, replays writes, reads policy; outcomes: Verifier too narrow (widened for all Candidates, re-gated), fail stands, cannot tell (Run leaves the bar). Human audits 10% of rulings after the fact (assumption, not yet confirmed as after-the-fact). Queue capped at ~5% of Runs per Task. Lives in: eval-design.md step 11.

**D13. Inverse replay of S0 is mandatory.** Reconstruct the starting state from the recorded reads, apply the recorded writes. "Not optional, it is the only way to get one. Agreed." Lives in: eval-design.md step 4.

**D14. System prompt and policy compile into Hard constraints, checked on the Reference first.** "They become a real checker and system prompt is a good way to define the constraints." Lives in: CONTEXT.md (Hard constraint).

**D15. Intent is written by the frontier from the trace with a grounding pass.** Every clause cites a trace span; a cross-family model checks grounding; ungrounded Intents are rejected and the Run gets no Verdict. "The rubric needs to be really grounded." Shown to the customer as the Task label (assumption pending). Lives in: CONTEXT.md (Intent), eval-design.md step 6.

**D16. Filtering: keep and tag, two gates only.** I overruled the proposal to reject truncated and orphaned Runs and exclude chit-chat: failed Runs help diagnosis; retries take the last attempt; duplicates collapse with a weight; compaction is handled, not filtered; orphaned and truncated are kept to explain why; chit-chat stays if that is the user's end goal. Research (R18) later agreed. Lives in: eval-design.md step 2.

**D17. Simplicity principle.** "Everything should be grounded in simplicity." Refined on 2026-08-26: "keep it simple doesn't mean that where we need complexity we don't add that." Lives in: eval-design.md "Principle", memory.

**D18. Reference environments first, then the generator.** Study how real environments are built (tau2 mock domain printed in full, R17) so the generator has a reference set; emit list: `data_model.py`, `tools.py`, `db.json`, `policy.md`, `tasks.json`; harness imported from tau2. "Once this synthetic environment generator works we can create 1000s of environments." Lives in: eval-design.md open questions (architecture).

**D19. "We don't have gold."** My gold is an approximation: the confirmed Reference's write calls executed on S0, atoms from re-roll agreement, with confidence (re-roll agreement, audit rate) published per Task. Lives in: eval-design.md step 7.

**D20. Reward basis fixed for every Replica.** `DB` (fingerprint with exempt/hard/semantic columns) + `ENV_ASSERTION` (atom list) + `COMMUNICATE` (plain matches) + Hard constraints. `ACTION` is computed but never counts (it is the tool-selection metric). Refined by D25. Lives in: eval-design.md steps 7 and 10.

**D21. A good Task is one where two domain experts would independently reach the same verdict.** Holds for verdicts; diagnosis (reason codes) is less reliable and is labeled as such. Lives in: eval-design.md step 12 and reporting.

### 2026-08-26

**D22. Gate A "100%" means exact after canonicalization.** "100% exact will reject good environments as well." The canonicalizer (ids, timestamps, float formatting, key order) is part of the Environment and has its own tests. Lives in: eval-design.md Gate A.

**D23. Unbiased reporting is a standing rule.** Every recommendation states where sources disagree, where an earlier recommendation was wrong, and where coverage is thin. Lives in: memory, and the "Counter-evidence" section of ADR-0004.

**D24. Build for the hardest case.** Long multi-call, multi-turn, off-path Runs drive the design; single-call Runs are the easy case of the same machinery. "Build for the hardest case so that it can handle these requests well." Lives in: eval-design.md "Principle".

**D25. LLM judges stay, as grounded rubrics, in three places only.** "We keep the LLM judges as rubrics then." Semantic column class, non-literal communicate facts, reason codes and the dispute path. Never the sole basis of a Verdict. Correction recorded: tau2 retail does use `NL_ASSERTION` in reward on 112 of 114 tasks; the earlier claim that tau2's real domains do not was wrong. Lives in: eval-design.md "Where an LLM judge is allowed".

**D26. DB transactions are the signal.** The Environment's write log is the transaction log; assertion helpers query the state it produces; fingerprints are computed from it. If a customer can supply real DB audit or CDC logs they are the strongest Reference-confirmation signal, optional under ADR-0002. Lives in: eval-design.md step 7.

**D27. Simulated users must come from production traces.** Hand-written personas are a stopgap; the simulator is conditioned on the customer's real user turns, with its error rate published per Task. Deferred to after Gate A. Superseded on timing by D44: the fact-consistent part is built with the Environment. Lives in: todo.md.

**D28. Deferred, with reasons, to `todo.md`:** output perturbation (slightly modified tool outputs), process reward model for the path (End state only for now), post-training on generated Environments to test hill-climbing on public benchmarks, human audit rate validation, schema and tool-definition ingestion.

**D29. Holes in a traces-only Environment are enumerated with a fill for each**, and Gate B's "assisted" share per tool decides which fills are needed. Lives in: eval-design.md "Holes in a traces-only Environment".

**D30. The Verifier is "given a task, what actions do we expect the agent to take", and that expectation comes from production traces.** "This is exactly what comes to my mind, given a task what are the actions we expect the agent should take (and this is my friends we get this from actual production traces)." tau2 writes that expectation by hand as `actions`; I derive it from the confirmed Reference and the re-rolls. Lives in: eval-design.md step 7, D19.

**D31. Learn from every reward basis, assemble the best version.** "I really like tau2 reward basis we need to learn from all of these and make the best for us." Mine: tau2's `DB` and `ENV_ASSERTION` and `COMMUNICATE`, AgentDojo's exact diff-key set as the forbidden rule, WorkArena's protected-field rule as a Hard constraint, AppWorld's no-op labeling as the null-agent gate, CRMArena's answer match for answer-only Tasks. Lives in: eval-design.md steps 7, 8, 10.

**D32. Monitor the actual DB transactions.** "This confirms that we need to monitor the actual db transactions as well." The Environment keeps an append-only write log per Run; if the customer can provide real DB audit or CDC logs they become a Reference-confirmation signal. Confirms D26.

**D33. Build the Environment from the whole trace corpus, not one Run.** "Yes agreed we need to build the environment from the whole traces." S0 for a Run is the corpus state at that Run's start; every entity any Run read is a row. Lives in: eval-design.md "Holes", fill 1.

**D34. My words are kept verbatim.** "Keep everything of my words please, those are the real decisions as well." `founder-words.md` holds every message unedited; this log indexes them.

**D35. Environment data follows customer trust (Q10 decided).** "Lets start with traces only and then as the trust of the customer increases we can ask for more information (traces plus a db snapshot is the end goal we get from the customer to build the environment for evaluation and training)." Traces first, tool definitions from the traces on day one, then schema, policy, and finally a DB snapshot inside the customer's boundary. Lives in: ADR-0006, eval-design.md "What the customer supplies".

**D36. Runs become Tasks by write-tool partition, then Intent clustering, with a Verifier-template merge/split test (Q11 decided).** "Agreed ... 1. 2. 3. 4. (just need to figure out empirically)." Sample-size thresholds are not fixed numbers: a Task is verdicted when its paired CI half-width is below the margin; 30 and 150 Runs are only the prior expectation of where that happens and will be replaced by observed per-Task variance. Tasks carry a priority (traffic share, cost, frontier failure rate) so the report leads with the prominent ones. Lives in: eval-design.md step 3.

**D37. Sources of Tasks beyond Replicas.** "Once we have the snapshots then we can generate tasks ourselves or viewing the traces also think about other relevant tasks." Replicas are the bar; Scenarios from traces (variations, policy-rule tasks, uncalled-tool tasks, error-path tasks), Scenarios from a snapshot (sample a real row, generate the goal from it, cover every write tool and precondition), and customer-supplied cases are for coverage, hardening and training, never the bar (ADR-0005). Lives in: eval-design.md "Sources of Tasks".

**D38. Assumptions are registered, graded and tested.** "In assumptions.md add the 9 and other assumptions." `assumptions.md` lists every assumption the design rests on with its evidence grade, how it gets tested, and what breaks if it is wrong; tested entries move to the bottom with the result. TauForge's non-existence in public is known and not an assumption.

**D39. Gate A split: writes exact, reads classified (Q12 decided).** "Yes I agree on the read." Write-effect fidelity 100% after canonicalization, hard gate. Read mismatches classified cosmetic or semantic; semantic must be zero, hard gate; cosmetic rate reported per Task with an empirical floor; unknown mismatch types default to semantic. "One shouldn't get to 100% strict because then that would be a problem": a read fidelity of exactly 100% strict is itself a warning sign (the Environment is serving recordings, not a world). Lives in: eval-design.md Gate A, CONTEXT.md (Replay fidelity).

**D40. Synthetic data for reads, generated by observing the real data in the traces.** "We need the whole database in some sense ... synthetic dataset generation for the reads is very necessary by observing the actual dataset from the traces." The Environment fills what the traces did not show with synthetic rows whose shape and distribution are learned from the rows the traces did show; synthetic rows are tagged and a Run that reads one is reported "assisted" until the customer's snapshot rung (ADR-0006) replaces them. Every published environment does this in some form. Lives in: eval-design.md "Holes", fill 1b; todo.md.

**D41. Representativeness is a Hard constraint on everything I generate.** "The tasks we generate should exactly / similarly represent the actual tasks but one thing which should never fail is that they should be a representative (have to be hard constraint) of the actual tasks." Synthetic rows and generated Scenarios must be shown to represent the observed traces (same fields, value ranges, entity relationships, task mix); a generated artifact that cannot be tied to observed data is rejected, the same rule as an ungrounded Intent. Lives in: eval-design.md "Sources of Tasks" and "Holes".

**D42. Every value in a write effect carries a Provenance, and Provenance decides whether the atom is required or allowed.** I said: "2. because the main goal was to cancel the order and get the reason." Four Provenances, each grounded to a trace span like an Intent clause: user-stated (in the opening request) = required; system-derived (read from a tool) = required; user-elicited (the agent asked, the user answered) = allowed, any plausible user answer passes; agent-chosen (no source) = allowed. Rejected: literal match of every value (tau2's DB hash; fails good Runs when the simulated user answers differently, R21) and checking only the opening request (a wrong refund amount from a bad read would pass). The Provenance classifier is an LLM call and is audited like the Intent. Lives in: CONTEXT.md (Provenance), eval-design.md step 7.

**D43. A Verdict grades effects; the End state includes what the user was told or asked.** I said: "if we want the model to ask why + cancel then we have that did it ask why and record the why because that is a tool call ... totally depends on the design and from the traces because 1. asking why is also a tool call 2. cancelling the order is also a tool call." A Run's effects fall on the world (writes) and on the user (messages and questions); both are End state. Effect-free actions (reads), their order, and reasoning text are the path and never change a Verdict. Whether a message or question is required comes from the traces by the same agreement rule as writes: present in every successful frontier re-run = required, in some = allowed, in none = not an atom; a question whose answer landed in a write binds that write's value to the user's reply in the Candidate's own Run. Reverses my earlier recommendation that a skipped question passes by default. Lives in: CONTEXT.md (End state, Verdict, Hard constraint), ADR-0004 amendment, eval-design.md step 7.

**D44. The simulated user is exact on this user's facts and representative on everything else.** I said: "facts stays consistent and everything is a representative is a good setup." Facts the recorded user gave (which order, refund method, address, confirmations) are repeated verbatim; wording, tone, patience, how much is volunteered are sampled from the customer's real users (D41 applies). A question the recorded user was never asked gets a representative improvised answer, which becomes an allowed value, never a required one. Measured per Task: fact consistency (near 100%, anything less is a simulator bug) and style representativeness (paired Turing test, A23). Replaces D27's "deferred to after Gate A": the simulator is part of the Environment before any Verdict, because D43 depends on it (A26). Lives in: CONTEXT.md (Simulated user), eval-design.md step 9 and Holes table, todo.md.

**D45. Hallucinated tool calls are counted, not failed; fabricated results always fail.** I said: "fabricated result are definitely not going to be accepted, tools almost everyone has a database interaction in it. i agree on hallucinated tool call." Three cases: a call to a nonexistent tool and a real tool with invalid arguments have no effect and never change a Verdict by default (D43); they are reported as hallucinated-tool calls per Run (a repeated call counts each time) and a customer can promote "rate must be 0" to a Hard constraint for their account. A fabricated result (the model claims a tool outcome it never obtained, in its answer or reasoning) is a Hard constraint and fails the Run. The Environment answers a bad call with the customer's own error encoding as seen in their traces; when their traces contain no errors, the encoding is a guess and is listed as an assumption on that Environment (R21 point 4). Lives in: eval-design.md step 4 and metrics, assumptions.md A27.

**D46. A failure is explained by its failing atom, computed by code.** I said: "the failing atom is the best option saying okay we failed this because of this." The line under every failed Run is the first Verifier atom that failed, in the atom's own terms (value was X, required Y; write missing; extra write; fact not stated; fabricated result; Hard constraint violated), plus a second field, same path as the Reference or different path, taken from the tool-selection metric and never from the Verdict. R21: 7 of 10 real failures were same-path, one value wrong, which this covers exactly. A fuzzy atom decided by the judge is labeled "judge:" with its audit agreement rate (D25). A free-text LLM explanation is opt-in only, labeled as opinion, never in the report by default (diagnosis agreement between humans is low, R18). Lives in: eval-design.md "Customer-facing metrics".

**D47. The Intent is shown to the customer as the Task name, and they can correct it.** I said: "show them and let them fix it, because the checklist comes from the tasks." The name shown is the cluster's Intent with its trace spans one click away. A correction is applied to the Verifier only if it can be grounded to a trace span (the same rule Intent clauses already follow); otherwise it is stored as a customer-supplied policy line (ADR-0006 rung 4) and compiled to a Hard constraint. Correction rate is published per customer as a signal on the Intent writer. Confirms A14. Lives in: eval-design.md step 6, "What the customer supplies".

**D48. Two human checks: setup review before any Verdict on a Task, and blind verdict audit after.** I said: "before the tasks are out we grade and after the tasks are out we also check if there is agreement with the human grader or not." Check 1, before: a person reviews the Task's checklist (atoms, Hard constraints) and the Runs chosen as References and confirms the setup is what the customer's agent is supposed to do; no Verdict on that Task until this is done. Check 2, after: a person grades a sample of Runs blind to the Verdict, and the agreement rate is published per Task; low agreement pulls the Task's Verdicts back to "under review". Replaces A13 (audit after the fact only). Check 1 is done by my reviewer first, then the customer's domain expert ("agreed on the second pair of eyes"), and covers the prominent Tasks, a sample chosen by the D36 priority (traffic share, cost per Run, frontier failure rate); other Tasks go out marked "setup not reviewed". Lives in: eval-design.md step 8, step 11, metrics.

**D49. A Task that cannot be replayed is "not gradeable, here is why"; assisted Verdicts are shown underneath, never counted.** I said: "i agree with the first one but I think we can still fake it right maybe till we get the actual stuff from them and till try to fake it as much as possible." Official status names the blocking tool or gap and the ladder rung that would unblock it (ADR-0006). Below it, the assisted Verdicts (Runs where an LLM stood in for a tool with no code and no recording) with the assisted share per tool; they never enter the routing plan. A guessed write to the outside world is simulated, never executed; an assisted Run never counts as evidence for Gate A. Step agreement (the Screen) stays internal and is not shown as a tier. Todo: measure assisted-vs-real Verdict agreement on the first customer with a snapshot; if high, revisit counting. Lives in: eval-design.md Gate B, metrics; CONTEXT.md (Assisted).

**D50. The first build is tested against tau2 and against real production traces, and must prove two things: the rebuilt world replays the frontier faithfully, and the Verdicts agree with the customer's.** I said: "i would like it against tau2 + as close to reality as possible (like the actual production traces) ... given only the traces it rebuilds the world in which ... but here we also need to add about the customer's verdict that it agrees with the world of our customer as well as close to real world as much as possible." Inputs: tau2's recorded trajectories (retail first; truth known, so the rebuild is checked line by line) and real production traces (Claude Code JSONL, R13, until a design partner's traces exist). Proof, three lines: (1) Gate A from traces alone, on both inputs (D39); (2) on tau2, per-Run Verdict agreement between the Verifier in the rebuild and tau2's own reward on the official result files, plus the aggregate pass rate within the non-inferiority margin of the official number; (3) on real traces, Verdict agreement with a human-labeled sample of 20 to 50 Runs (the number already in "What the customer supplies"). Lives in: eval-design.md "First build".

**D51. Generalizability over overfitting.** I said: "we need to care much more about generalizability than the overfitting please." A rebuilt Environment, Verifier or Simulated user that only works for the exact recorded Runs it was built from is a failure even if every fidelity number is perfect. Every stage is tested on Runs it did not see (held-out Runs of the same Task, alternative-path re-rolls, a second customer or domain), and a 100% score on the seed Runs with a collapse on held-out Runs is reported as overfitting, not success (same spirit as D39's "100% strict reads is a flag"). Lives in: eval-design.md Principles; assumptions.md A28.

**D52. Real-world tool-using traces, not coding traces, are the first build's real input.** I said: "we are not really focusing on coding traces we are focusing on real world tasks." Amends D50: Claude Code JSONL is dropped as the real-trace input; the real-world source is chosen from R23 (production-style support, ticketing, CRM, banking, travel traces, or traces I generate myself against a real sandboxed helpdesk with real tool schemas). tau2 stays the known-truth input; tau3 and other 2026 environments are surveyed in R22 to sharpen what a good Environment looks like. Lives in: eval-design.md "First build".

**D53. The harness is as simple and efficient as possible, layered like Tau (twotimespi.dev): brain, environment, face.** I said: "we need to make the harness as efficient and simple as possible sir." and "https://twotimespi.dev/ and to build the whole harness we can learn from this." Tau's portable core is about 1,500 lines: a provider-neutral model interface, a tool list (name, schema, execute), a loop that streams events, before/after tool-call hooks, JSONL sessions; UI only consumes events. The re-execution harness follows the same cut: the Candidate model behind a provider interface; the Environment as the tool list with the Hard-constraint check and the assisted/recorded/code routing in the tool-call hooks; the Simulated user as a message source; every Run persisted as JSONL; the report consumes events. Nothing in the core knows about tau2 or any customer. Lives in: architecture doc (to be written), todo.md.

**D54. Build shape follows TauForge's four stages; hardening is a later todo.** I said: "tau forge uses 1. build environment 2. augment seeds 3. generate tasks 4. harden tasks (which is for the later todo) ... so the environment should be grounded (very much in the customer's tasks a lot a lot and a lot)." Stage 1 (build Environment from traces) and the Replica path are the first build; stage 2 (augment seeds: the customer's own Tasks are the seeds, never abstract scenarios first) and stage 3 (generate Tasks, D37 sources) come after Gate A; stage 4 (harden) is todo. TauForge's supporting assets (logs, skills, past builds, utility scripts, knowledge graph, abstract scenarios, personas, hardening taps) are the checklist for what the builder accumulates across customers, with grounding in the customer's traces as the constraint on every one. Built iteratively: "smallest slice to validate the way we are building is correct or not." Lives in: eval-design.md "First build", todo.md.

**D55. The smallest slice runs through the Verdict: one tau2 retail Task, one cheaper model, the rebuilt world against real tau2, compared Run by Run. tau2's Environment quality is the first bar; improving on it comes after.** I said: "i would say 2 but we are using tau2 as our reference first is to give (first we need to hit the environment created by tau2 quality level) and then find ways to improve on it." Order: (a) rebuild retail from tau2's recorded trajectories alone and pass Gate A; (b) derive the Verifier for one Task and check it passes the frontier's own Runs and fails a null agent; (c) re-run that Task with one cheaper model (official tau2 result files already hold gpt-4.1-mini and o4-mini Runs, R21) in the world and compare the Verdict per Run with tau2's reward; (d) only then look for where the Verifier or world is better than tau2's (the 18.5% / 9.8% human misalignment in 2607.02577 is the target to beat). Lives in: eval-design.md "First build".

**D56. The builder emits tau2's shape first; its own shape comes after, molded from it.** I said: "i would really like the flexibility but first lets get it in tau2 shape and then we mold it to our shape (i will start uploading the real traces very soon this should help a lot)." First build writes `data_model.py`, `tools.py`, `db.json`, `policy.md`, `tasks.json` so tau2's own harness loads the rebuilt domain and the comparison in D55 is direct. Verifier detail that does not fit (Provenance, allowed atoms, questions asked, trace spans) is kept in a sidecar file next to `tasks.json` until the harness's own shape exists, so nothing decided is lost while the shape is tau2's. The customer's real traces, arriving soon, are the input that drives the molding. Lives in: eval-design.md "First build".

**D57. Confirming a recorded Run as a Reference passes through several gates, including an LLM judge that marks pass or fail and hands "unsure" to a human.** I said: "we have several gates including llms as a judge to mark pass fail and where judge is unsure we can have the humans." This is for labeling recorded Runs (step 5), not for Candidate Verdicts, which stay code-only (D25). The judge sits beside the existing gates (Hard constraints on the Reference, customer outcome signals, k frontier re-rolls reaching the same End state) and must abstain at a real confidence threshold; its rulings on the sample are part of check 2 (D48) so its agreement with humans is published. Lives in: eval-design.md step 5.

**D58. Customer trace intake questions live in trace-intake.md.** I said: "a vendor export, maybe in a week or two we have them ... also add the questions for the customer traces." Thirteen questions (source, format, truncation, tool definitions, grouping, errors, system prompt, volume, outcome signals, labels, policy docs, storage location, retention). Answered when the export arrives; traces are never committed to git. Lives in: trace-intake.md, todo.md ingestion entry.

**D59. Harness is the whole system; the environment creator is the Builder and the re-execution loop is the Runner.** I said: "yes environment creator and the whole system." Harness keeps its glossary meaning (replay, grade, post-train, deploy, route); Builder and Runner are its first two parts and the two modules of the first build. "Harness design" means the design of both, Builder first. Lives in: CONTEXT.md (Harness, Builder, Runner).

**D60. Self-generated traces are built in parallel with the tau2 slice, and must not be biased toward the Builder.** I said: "in parallel with the tau2 slice (just make sure that it is not biased please)." A real helpdesk in Docker (Chatwoot or Zammad), an existing MCP server, an LLM agent and a simulated user produce traces with real schemas and real errors (R23 rank 3). Bias controls: the agent side is run with at least two harnesses and models I did not write (not the Runner); the simulated user is tau2's, not mine; agent prompts are written by someone other than the Builder's author or taken from public projects; the Builder is never tuned on these traces alone, and every number from them is labeled "self-generated" and not used to claim representativeness (D41). Their purpose is the format and error-path rehearsal until the vendor export arrives. Lives in: harness-design.md section 11, todo.md.

**D61. The Harness is a workflow for RL environment generation; its principles come from environment-generation pipelines, not from coding-agent harnesses. The Runner is frozen; the Builder improves until it generates high-quality Environments.** I said: "we are building the harness (which is kinda a workflow for RL environment generation, so all the decisions and principles of harness design should be coming from those). 1. i agree on runner must be frozen (that is the essential context around it) 2. builder should be improving till we hit that the harness is able to generate high quality environments." Consequences: the principles table in harness-design.md is re-derived from EnvFactory, Envs-FORGE, Agent-World, AgentSynth, TauForge, Westworld, OpenEnv and the learn-harness-engineering writeups (R25), with coding-agent sources kept only where they agree; the Builder's improvement loop has an explicit stop condition ("generates high-quality Environments", measured by Gate A on held-out Runs and Verdict agreement with the customer, D50), and the evaluator stays outside that loop. Lives in: harness-design.md sections 2 and 12.

**D62. The Builder's stop condition is customer Verdict agreement; until a customer exists, it is Environment quality measured against tau2's own Environments, in a scorecard I can read.** I said: "ideally it should be the customer agreement but for the time being we need to compare our environments with the tau2 environments in terms of quality so that initial environment creator can be built and also I can evaluate the quality of the environments as well." The scorecard, per rebuilt domain, side by side with tau2's real one: (1) per-tool replay fidelity on held-out recorded calls, success and error separately; (2) state coverage: entities, fields and ids in the `db.json` versus theirs, synthetic share; (3) policy coverage: constraints compiled versus rules in `policy.md`, residual count; (4) Simulated user fact consistency on re-runs; (5) per-Run Verdict agreement with tau2's reward, by-decision disagreements listed separately; (6) defects per Task under the tau3 fix taxonomy (R22 1.5), found by my own review. Gate A on held-out Runs is the entry condition; the evaluator stays outside the Builder's loop. Lives in: harness-design.md, eval-design.md "First build".

**D63. My four principles for developing the Harness.** Verbatim: "1. we don't let the context to increase beyond 40% hard constraint 2. it should be minimal in design and very compact 3. We need to get stuff when needed, get tool calls get system prompt and stuff 4. we need the model to analyze its mistake and improve the prompts, tools, middle ware and memory after studying past failures, we need sessions trees (the model should decide that what it needs to improve on what it doesn't), we can debate on this." Recorded as principles of the Builder; (1) and (4) have open sub-questions: whether the 40% cap applies to the Builder's own model calls only (the Candidate under test must see what the customer's production agent saw, D45), and the shape of session trees and the model's authority over its own improvements (debate open). The D62 quality scorecard is confirmed: "quality numbers looks good." Lives in: harness-design.md section 12.

**D64. Builder history is a tree; Runner Runs are flat files. The model proposes edits; an evaluator outside the loop decides whether they stay.** I said: "tree for builder and flat files for the runner ... agreed we need an evaluator who decides what improves or not." Each Builder version is a node (prompts, gates, seed corpus, tool bodies, scorecard, anchor results) with its parent; a Run stays one JSONL with `parent_run_id` for forks. The improvement agent reads failures, picks the component (prompts, tools, middleware, memory), writes the edit and a prediction; acceptance is by the gates only: held-in improves, the anchor does not regress, the D62 scorecard does not drop. Lives in: harness-design.md sections 12 and 13.

**D65. The 40% context cap applies to the Builder's own model calls, including the evaluator's; the Candidate is always tested under the production setting.** I said: "builder gets the 40% cap (because beyond that it is not worth it). tools prompts are loaded when needed and delete when don't and the evaluator also gets the call here. we test candidate under production setting always." Every Builder call is fresh and bounded; tools, prompts and records are loaded when needed and dropped after; a call that would exceed 40% of the window is refused, not compacted; the evaluator's calls obey the same cap. The Candidate's context is exactly what the customer's production agent gave it (system prompt, tool definitions, conversation, and production's own compaction if the traces show one); the Harness never caps or compacts it. Closes D63's open parts. Lives in: harness-design.md sections 3 and 12.

**D66. Raw traces are the source of truth, stored byte for byte and content-hashed; the normalized Trace record is derived from them and can be re-derived; ingest extracts as much as the raw allows.** I said: "we should keep the raw but normalized should be derived from that (we should always have the source of truth stored) and derive as much information as much possible from it." Each derived field carries a pointer to its raw location (file, offset), which is what lets Intent clauses and Provenance cite a span. Re-deriving when the schema grows is a pipeline stage keyed on (raw hash, ingest code hash), so nothing is ever lost at the door. Raw files are never committed to git (D58). Lives in: harness-design.md section 4 (`ingest.py`), section 5 (`Trace`).

**D67. Tool errors are normalized at ingest into a small fixed taxonomy plus the verbatim payload.** I said: "i am going towards small fixed taxonomy plus the raw text." Classes: `tool_not_found`, `invalid_arguments`, `permission_denied`, `business_error` (the tool ran and refused), `not_found_entity`, `transient` (rate limit, 5xx, timeout), `cancelled`, `unknown`. Sources with a flag or typed output are classified by code; string-only sources (OpenAI, HF STS) by rules first and an LLM pass second, with `classified_by: code | rule | llm` on the record and the raw pointer kept. The Environment reproduces each class in the customer's own encoding (D45); the Builder reports observed error classes per tool (A27); `unknown` above a small share on any tool is a flag on the Environment. Lives in: harness-design.md section 4 (`ingest.py`), section 5 (`Trace.tool_calls.error`).

**D68. A tool's read/write class is decided by an LLM given full observed context; effect evidence is computed by code first and overrides the LLM when they disagree.** I said: "we observe what the tool is intended to do right? and from that we already know, then we use the LLM to classify (they are very strong on classification with enough context)." Code gathers the context: name, description, argument and result schemas, sample calls and results, MCP annotations if present, and observed effects across the corpus (a later read showing a changed value after the call). The LLM classifies read, write, or generic with a stated confidence and reason. A tool with observed effects is a write whatever the LLM says. Low-confidence classifications are listed in the setup review (D48 check 1) as a table the reviewer confirms. Open: the default for a tool with no effect evidence and low LLM confidence. Lives in: harness-design.md section 4 (`mine.py`), `ToolSig.kind` with `kind_confidence` and `kind_reason`.

### D69. Which missing components enter the first build; the Builder may modify itself (2026-08-27)

I was asked which components the design had named or implied but the harness module list did not carry (23 items in four groups: decided in eval-design but not in the module list; unnamed infrastructure; TauForge, AHE and Weng assets; later product stages). My choice, verbatim: "we need the following, 1. clusters 2. intent writer 3. canonicalizer 4. report 5. cost and budget accounting 14. provider adapter ( see how open code does it ? ) 15. maybe later 19. we need this and also the ability for the builder to modify itself -> rest are todos".

Decided:
- Into the first build as modules: `cluster.py` (Runs into Tasks), `intent.py` (grounded Task name, D47), `canon.py` (canonicalizer D39 as its own module shared by `route.py` and `verdict.py`), `report.py` (the face), `budget.py` (cost and token accounting, the 40% cap D65, spend ceiling), `provider.py` (`Model.query` with Anthropic, OpenAI and OpenAI-compatible adapters; research OpenCode's provider layer first, R26), `memory.py` (the Builder's version tree with scorecards and failure notes, D64).
- The Builder can modify itself: the improvement agent may edit files under the Builder's own directory (prompts, mining, compile, policy, user-sim, intent, cluster, canonicalizer rules) and each edit is a node in the memory tree accepted or rejected by the evaluator outside the loop (D64) on the anchor sample and held-out Runs. Read-only for it: `loop.py`, `route.py`, `verdict.py`, `validate.py`, `budget.py`, the runs directory, the raw store, the evaluator.
- Review UI: "maybe later" (todo).
- To `todo.md`: Filter and Screen, Reference confirmation as a module, dispute path, statistics module, sandbox for model-written tool code, anonymization, secrets, cache store, scenario generation and seed augmentation, dataset export, deploy and route, production monitoring.

Where this disagrees with earlier decisions, noted here so it is not missed: Reference confirmation (D57, step 5) and the statistics module (step 12) were decided design steps; deferring them is fine for the tau2 slice (tau2 supplies references and one Task needs no CI) but both must land before the first customer Verdict, and the todo entries say so. The sandbox is infrastructure, not a design choice, and `compile_env.py` should not run model-written code outside one from the first slice on.

Supersedes the "a provider abstraction beyond `Model.query`" item of the deliberately-absent list in `harness-design.md` section 4; `Model.query` stays the only interface, the adapters behind it are now a named module.

### D70. Default class for a tool with no effect evidence and low LLM confidence (2026-08-27)

Q35. I said: "i would say read and then unclassified." Read as: the row defaults to `kind: read` so Runs stay gradeable, and it also carries `unclassified: true` (with `kind_confidence: low`) so the setup review (D48) sees it as an open row, not a settled one. Runs are not blocked. If I meant a two-step rule instead (read first, unclassified after some second signal), the log is wrong and should be corrected.

Consequences:
- `ToolSig` gets an `unclassified` flag beside `kind`, `kind_confidence`, `kind_reason`, `effects_observed`.
- Observed effects still override (D68): the moment a later read in any trace shows a change explained only by this tool, it becomes `write` and the flag clears.
- The report (D69 `report.py`) lists every Verdict whose Run called a flagged tool, with the count, so a customer sees how much of a pass rate rests on tools nobody has confirmed.
- The setup review cannot be closed with a flagged tool still open; a reviewer sets `read` or `write` and the flag clears with `classified_by: human`.

Where research pulls the other way: tau3's own fix list is mostly grading that was too narrow, not too wide (R22, 75+ fixes), and a misjudged write is exactly a Verifier that is too narrow, silent until an audit disagreement. That is the cost accepted here; the report line and the review block are what bound it.

### D71. User-side writes in the End state: only when the traces record them (2026-08-27, provisional)

Q29, asked four times. I said: "i would say this as well. we will need to discuss this more as well." Provisional: a write performed by the user during a Run is an End-state atom only when the customer's logs record it as an event. Then it carries `requestor: user` (tau3 banking shape), Provenance to its span (D42), and the Simulated user (D44) performs it at the recorded point. Writes the logs do not show do not exist for the Verifier, and the report says "user-side actions not in your logs, not graded". Grounds: D66 (raw trace is the source of truth), D42, D41.

Where research disagrees: tau3 banking grades user-side writes in every Task because its Simulated user has tools and the benchmark owns the world; a customer whose logs omit user actions gets a narrower Verifier from me. Trace-intake question 9 (outcome signals) is where I learn whether the logs carry them.

Open for the next session: whether the Simulated user gets tools of its own (a user-side write means the Simulated user calls a tool), how a user-side write interacts with Hard constraints of the form "never X without a prior user action", and whether a user-side write can be a required atom or only allowed.

### D72. Tool result schema is the union of everything observed, fixed in review (2026-08-27)

Q36. I said: "i would say union of everything which is fixed eventually."

- `ToolSig.result_schema` is the union of every field seen in any observed result; each field carries `count`, `first_seen`, `last_seen` (trace ids) and `declared: bool` (present in the `tools` list sent to the model, when the traces have one). Fields seen in fewer than all calls are optional.
- The declared schema is not the contract; it is one more source, recorded as `declared` so the review can see where the customer's own definition and their logs disagree.
- "Fixed eventually": the setup review (D48) can drop a field (an old code path, a leak) or mark it as a variant; the change is recorded with `classified_by: human` and the Environment version bumps. Until then the Environment serves the widest shape it saw.
- Same rule for `args_schema`.

Where research pulls the other way: tau3's fix list includes tasks broken by tool outputs that were wider than the task author expected (R22); the union puts every rare field in front of the Candidate, so a Candidate can rely on a field the customer's production tool returns once a year. The report's per-field counts are the guard; nothing else.

### D73. Column class: code proposes by rule, LLM verifies per column (2026-08-27)

Q37. I said: "i would say llm per column + code by rule ( which is verified by the LLM )".

- Code runs first over every column and proposes a class from rules (timestamps and counters `exempt`, ids and enums `hard`, long strings `semantic`) together with the evidence it used (distinct-value count, value shapes, monotonicity, length distribution, foreign-key hits).
- The LLM sees the column, the proposed class, the evidence and sample values, and either confirms or overrides with a reason; `EntitySchema.columns[i]` records `class`, `class_rule` (what code proposed), `class_confidence`, `class_reason`, `classified_by: rule | llm | human`.
- Low-confidence rows and every override of a rule go to the setup review (D48).
- Re-run evidence overrides both (option 3 as correction): a column that varies across successful re-runs with the same outcome becomes `exempt`, recorded with `classified_by: observed`.

Same mechanism as D68 for tools and D70 for unknowns: code gathers, LLM classifies, observation overrides, human closes.

Where research disagrees: tau2 has no column classes (whole-database hash); tau3 added canonicalization on top (R22). Column classes are my deviation; the cost is one review list per customer.

### D74. One world per customer, starting state per Task as an overlay (2026-08-27)

Q38 and Q39, settled over four turns. I first said "one s0 per task not per customer", then clarified: "we should be able to generate the world from the whole traces and then we come up with tasks in that world right ? and evaluate per task right ?" and accepted option 3 ("yes!").

- One shared world per customer: `db.json`, `EntitySchema` and every `ToolSig` are mined once from all traces (D72, D73 apply to the whole corpus). Inverse replay over the whole corpus (D33) builds the shared rows; where two traces show the same row in different states, the shared world keeps the latest observation.
- Each Task carries an overlay: the rows its own Runs touched, in the version those Runs saw. A tool reads the overlay first and the shared world second; that lookup lives in `route.py`, not in generated tool bodies. So the March cancel Task sees order 123 pending and the June delivery Task sees it delivered, and both replay.
- Nothing is invented: the overlay pins observed rows only; no `valid_from` is guessed (D41).
- The report states per Task how many of its rows are overlay rows, and the setup review sees a Task whose overlay is large relative to its reads.
- Emitting for tau2's harness (D56) merges overlays into one `db.json`; a conflict between two Tasks' overlays is a Gate failure for the tau2 export and the Task stays gradeable in the Runner. Expected not to occur on tau2 retail.

Where research pulls the other way: tau2 and tau3 have one database per domain and tasks written to fit it; a shared world that is a patchwork of moments is my deviation, and a Candidate crossing Tasks can meet the same entity in two states. Rejected: latest-observation-only (loses the March Task), time-versioned rows (a clock in every generated tool body, and `valid_from` would be invented).

Consequences for Tasks: a Task does not make the order pending or delivered; those are where Tasks start. The cancel Task and the delivery Task are two Tasks on one row in two starting states.

### D75. Tool body repair: bounded retries with growing evidence, every rewrite checked against the traces (2026-08-27)

Q40. I said: "i would say 1 + 2. but the rewriting should be check to be honest that is that the correct rewrite according to the traces we got."

- A tool body that fails a compile gate is rewritten by the LLM at most three times. Each attempt is a node in the Builder tree (D64) with the gate result attached.
- Evidence grows per attempt: attempt 1 gets the failing call and its recorded result; attempt 2 gets every failing call for that tool; attempt 3 gets the full table of recorded calls for that tool (args, result, error class). An attempt whose evidence would exceed the 40% cap (D65) is refused, not truncated, and the tool goes to the assisted path at that point.
- Every rewrite is checked the same way the first version was, against all recorded calls for the tool, not only the ones that failed. The check is split: the calls the LLM was shown, and a held-out set it was not shown (D51, A28). Fidelity is reported on both; a rewrite that passes the shown calls and fails the held-out ones is a failed attempt. Nothing the LLM writes is accepted on its own say-so.
- After three misses the tool is marked assisted (D49): recorded calls are answered from the recording, anything else by the LLM stand-in, every Run that calls it is Assisted and the report says so per tool.

Where research pulls the other way: EnvHarness and EnvRigger (R25) keep the verifier frozen and let the proposer loop far longer than three tries; three is a budget choice, and the number can move once the first customer shows how often tools need a fourth try. The number is in config, not code.

### D76. Rules that do not compile: rewrite for approval first, LLM judge atom second (2026-08-27)

Q41. I said: "2 + 3".

- Order: a rule that does not compile is first rewritten by the Builder into a checkable form, grounded to a span in the traces or the policy, and shown in the setup review (D48) with the original text beside it. The customer's expert accepts, edits or rejects. An accepted rewrite that compiles becomes a normal `Constraint`.
- A rule that still cannot be checked by code after the rewrite (or that the reviewer keeps in its vague form) becomes a judge atom: the LLM judge evaluates it at Verdict time, with D57's abstain-to-human when unsure, and it can fail a Run. Judge atoms carry their audit agreement rate (D48) on every report line.
- A rule the reviewer rejects goes to `residual` and is reported as not checked.
- `verdict.py` stays code-only. Judge atoms are evaluated by a separate `judge.py` in the Runner that returns per-atom results into the Verdict; `verdict.py` combines code atoms and judge results and sets `judge_used`. This keeps the "never calls a model" property on the deterministic part and makes the judge's share of every Verdict visible.
- Regrade (D-regrade, `regrade.py`) re-runs code atoms for free and re-runs judge atoms only when the judge version changed.

Where research pulls the other way: tau2's `nl_assertions` (R22) are judge-only and tau3 measured 18.5% evaluator-human misalignment on them; that number is the reason judge atoms carry an audit rate and cannot be hidden inside a pass rate. R00 and earlier decisions (D46) put no model in the Verdict; this is the first exception and it is fenced to judge atoms.

### D77. The Simulated user answers from the world, never invents (2026-08-27)

Q42. I said: "the answers should always be grouned in the real world to be honest ( which are the traces ) -> should always be consistent with the world".

- When a Candidate asks for a fact the real user was never asked, the Simulated user answers from the world: the Task's Starting state (overlay then shared `db.json`, D74) read as that user. If the world holds the user's card's last four, the Simulated user knows it, because the real user would.
- When neither the trace nor the world holds the fact, the Simulated user says it does not have it. Nothing is invented (D41). The event is tagged `fact_unavailable` with the field name.
- A Run that fails after a `fact_unavailable` event gets that as its failure explanation (D46 class `blocked_on_unavailable_fact`), so the report separates "the Candidate needed something the world does not contain" from "the Candidate did the task wrong". These Runs count; the Candidate chose a path the world could not support, and the recorded agent did not need it.
- Disclosure rules (D44) still apply on top: a fact the world has but the real user only gave on request is given on request; a fact the real user refused stays refused.
- Consequence for `user_sim.py`: it is a reader of the Starting state, with the same off-path holes as any tool; the assisted share per Task includes Simulated user reads that hit a synthetic row (D40).

Where research pulls the other way: tau2's user simulator is given a scenario text and instructed to invent nothing outside it, which is option 1 for everything not in the scenario; tau3's task-fix list (R22) includes user simulators that stonewalled agents on facts a real user would know. Reading from the world is my deviation and it is what closes that class of false failures.

### D78. The re-run count k is decided by experiment, not by design (2026-08-27)

Q43. I said: "i mean we need to do experiments to do decide the number here right ?"

- k is a config value, not a design constant. No option from Q43 is chosen; the experiment chooses.
- The experiment, on the tau2 retail slice (D55) first, then on the first real traces: for each Task run the frontier model n = 10 times, compute the required and allowed sets (D43) from the first k re-runs for k = 2 to 10, and record at which k the sets stop changing. Report the distribution of that k across Tasks, and the per-Run Verdict agreement with tau2's reward at each k. The default k is the smallest value at which the required set is stable on 90% of Tasks; the stopping rule from Q43 option 2 is tested against the fixed k on cost and agreement.
- Registered as assumption A30: the required set stabilizes within 10 re-runs on most Tasks. If it does not, D43's "present in every successful re-run" definition is the thing to revisit, not k.
- Recorded as a todo experiment with its own output table.

### D79. Verifier validation: automated checks first, humans only on hits (2026-08-27)

Q44. I said: "i mean you should check it and then if there are something then the human checks right ? what does the research suggests ? what other are doign ?"

- Every Verifier passes an automated suite before it enters the pool, and a human sees only what the suite flags. The suite (R25 principle 6, R12 step 9, tau3 practice):
  1. Provenance span check by code: a user-stated value must appear in a user turn of the Reference Run; a system-derived value must appear in a tool result before the write; no span, no Verifier.
  2. Oracle passes: the Reference Run itself scores pass.
  3. Empty Run fails: a Run that does nothing scores below the oracle (CUA-Gym `r(s_init) < r(s_gold)`).
  4. A plausible-but-wrong path fails: one deliberately wrong End state (wrong entity, missing question atom) scores fail.
  5. A second valid path passes: a frontier re-run that reaches the End state by a different route (D46 different-path) still passes; otherwise the Verifier is too narrow.
  6. Loophole probe: an agent told to reach the End state while skipping the policy step must score fail (tau3 loophole fixes, 2606.08960 hacker loop).
  7. Leak check: the Task's Intent and the Simulated user rules are grepped for constants that only the Verifier should know.
- Any hit sends that Verifier to the setup review (D48) with the failing check named. No hit, no human before the Verdict; the blind audit after (D48) still samples.
- Checks 1 to 5 and 7 are code or one re-run; check 6 is one adversarial LLM run per Task and is the only cost that scales with the pool.

What others do, from the research: Sierra's tau3 had two reviewers manually simulate at least one valid trajectory per task and re-audit every trajectory after the large experiments for shortcuts and loopholes, then still shipped 75 to 100 task fixes in 1.0.0 and 1.0.1 (R22); Envs-FORGE automates the whole suite; SWE-smith spent about 20 human hours; CUA-Gym forces the empty-run inequality; Agent-World uses Pass@5 tolerance; LOGIGEN's spoiler-free instruction rule is check 7; 2606.08960 found 16% of 1,968 tasks hackable from the description alone and a hacker/fixer loop took attack success to 0% (R22 section 3.11). No documented pipeline disagrees with validating the verifier; they differ only in how much is automated versus human. Humans-first (tau3) still missed enough to need two fix releases, which is the argument for the suite running before any reviewer.

### D80. Scorecard thresholds: measured on tau2 first, then set at or above; the target is 100% with every miss explained (2026-08-27)

Q45. I said: "we cannot fix that, we ideally need 100% 3. what do you suggest ? the world needs be to as close as to the real world sir." Also: "CUA-Gym forces an empty run to score below the gold run. The hacker-fixer paper ... i like these approaches."

- Option 3: run the pipeline on tau2's own files first and measure every scorecard number on the real tau2 environment; the written thresholds start at or above those numbers; every later change to a threshold is a logged decision.
- The target is 100%, not a percentage chosen to be reachable. Concretely:
  - Replay of recorded calls: 100%. A recorded call whose rebuilt result differs from the recording is a bug in the tool body, not a statistic; it goes to the D75 repair loop, and after three misses the tool is assisted and shown as such.
  - Held-out calls, Verdict agreement, user fact consistency: also 100%, measured as "no unexplained miss". Every miss gets one reason recorded by code or the Builder: my bug (fixed), tau2's grader bug (tau3 shows 18.5% judge-human disagreement, R22), or ambiguous (goes to the setup review). A miss without a reason is a gate failure. The scorecard shows the raw number and the explained number side by side.
- The empty-run rule (CUA-Gym) and the hacker-fixer loop (2606.08960) are confirmed as parts of D79 (checks 3 and 6) and my preferred approach to Verifier hardening.
- The wider intent, in my own words: the world must be as close to the real world as the traces allow; a number below 100% is a list of things to fix, not a score to accept.

Where research pulls the other way: no pipeline in R22 or R25 reaches 100% Verdict agreement against a human panel, and tau2's own grader does not; "100% with every miss explained" is how the target stays honest without lowering it.

### D81. Held-out anchor: a share of every Task's Runs (2026-08-27)

Q46. I said: "i would say we do it per task and then a proportion per task ?"

- The anchor (held-out Runs, D51, A28) is drawn per Task as a fixed share of that Task's Runs, 20% by default, at least one Run, picked once with a fixed seed and never touched by the Builder.
- A Task with too few Runs to give one up (fewer than three) is built from all its Runs and marked "unguarded" in the report; the Builder's improvement loop is not allowed to accept an edit on evidence from unguarded Tasks alone.
- The share and the floor are config; the tau2 slice and the first customer decide whether 20% and three are right.
- Anchor Runs are used for the D75 held-out split, the D79 different-path check where they suffice, and the D62 scorecard; they are never Verifier seeds (D43 uses the seed Runs).

### D82. Builder self-improvement: one change per round first, batches later (2026-08-27)

Q47. I said: "several changes tested together what is the problem in that ? start with one change per round and then accept or reject and then several changes ?"

- First build and first customer: one change per round. The Builder proposes one edit to itself, the evaluator runs the anchor (D81), accepts or rejects, and the round is one node in the memory tree (D69 `memory.py`) with the prediction the Builder made and what actually happened.
- Batches later, once the tree holds enough single-change rounds to know which kinds of edits are safe to combine. A batch is accepted or rejected as a whole; when it is rejected, the evaluator splits it and re-tests halves (bisection), so the culprit is found in a few anchor runs rather than one per change.
- The problem with batches from day one, in plain words: when five changes go in together and the anchor gets worse, nothing says which one did it, and when it gets better, nothing says which four were dead weight. The tree then records "batch helped" instead of "this kind of edit helps", and the Builder learns nothing it can reuse. Single rounds are slow, and that is the price of knowing.
- Switch rule, in config: batches are allowed once at least N single-change rounds (default 20) are in the tree and the evaluator's per-edit-kind acceptance rates are stable.

From research: AHE's decision observability (each edit paired with a prediction checked next round) only works one edit at a time; the monthly ablation in learn-harness-engineering lecture 12 is the batch-era tool.

### D83. Tasks have a hierarchy: Category above Task (2026-08-27)

Q48. I said: "two differnet tasks but the category could be same, tasks also have hierarchy right ?"

- Two levels. A Category is the set of Runs whose confirmed References write through the same tool set (the hard partition eval-design step 3 already makes). A Task is a cluster inside a Category that shares one user intent. "Cancel because late" and "cancel by mistake" are two Tasks in the Category "cancel order".
- Verdicts, Verifiers, Starting states (D74 overlays), Simulated user rules and the anchor (D81) are per Task. The report rolls Tasks up to their Category, so it can say both "the cheap model handles cancellations" and "except when the delivery was late".
- `cluster.py` emits both levels: Category id from the write-tool signature by code; Task membership by Intent similarity with an LLM naming the cluster (D69). A Task that would be unguarded (D81) is still its own Task; the Category-level number is what the customer sees when a Task is too small to stand alone.
- Nothing deeper than two levels until a customer's traces show a need; a third level would be a sub-intent and belongs in the Task name for now.

Glossary: Category added to CONTEXT.md; Task entry now says it sits inside a Category.

### D84. Semantic columns: the judge decides, only when canonical strings differ, with a cached equivalence table (2026-08-27)

Q49. I said: "the judge 3."

- A `semantic` column (D73) is compared by `canon.py` first; equal canonical strings are equal, no model call.
- When they differ, `judge.py` (D76) decides whether the two values mean the same thing for that column, with the column's observed values as context. The result is a judge atom with the judge's audit rate (D48).
- The pair (column, canonical a, canonical b, verdict, judge version) is cached in an equivalence table per customer; later Runs reuse it. The table is a file the setup review and the blind audit can open, and a human can overturn an entry (`classified_by: human`), which invalidates every Verdict that used it and queues a regrade (`regrade.py`).
- The audit sample (D48) draws from cached entries in proportion to how many Verdicts rest on them, so a wrong entry that many Runs depend on is the likeliest to be caught.
- Cost: one judge call per new pair per column; the table grows toward the number of distinct wordings, not the number of Runs.

Where research pulls the other way: tau3 reads are never hashed and free-text fields are excluded from the DB hash altogether (R22), which is option 2; I grade them because a wrong recorded reason is a wrong End state (D43, my cancel-and-ask-why example).

### D85. The report shows whether the Environment was built and the numbers; it suggests, the person decides (2026-08-27)

Q50. I said: "report should also have that whether we were able to create the environment or not as well including the numbers and the verdict is decided by the the persons ( just show the numbers and suggest ) decision is the person."

- The report opens with the Environment, before any Task: built or not built, which Gates passed and failed (D79 suite, compile gates, D80 scorecard with raw and explained numbers side by side), assisted tools, unguarded Tasks (D81), overlay counts (D74). A customer sees first whether the world is trustworthy and how much of it is real code.
- Per Task the report shows the numbers (Runs graded, assisted and not counted, judge atoms with audit rate, frontier and Candidate pass rates, the margin, failing atoms by class) and then a suggestion, worded as a suggestion: "the numbers support routing this Task to Candidate X" or "they do not". The decision belongs to the person. The routing plan is written only from decisions a person has made in the report, never from the suggestion alone.
- "Clears the bar" stays a computed status; it is the input to the suggestion, not the decision. Glossary updated.
- None of the three Q50 orderings was chosen as written; my order is Environment status first, numbers second, suggestion last.

### D86. Spend ceiling hit: stop, report as is, ask before continuing (2026-08-27)

Q51. I said: "we need to be as close as to the reality sir, report as it is is. stop and seek permission to continue further."

- When `budget.py` reaches the ceiling, the Builder stops where it is. Nothing is finished cheaply and nothing half-done is used: no tool is switched to assisted to get past the ceiling (that would trade reality for completion, the opposite of D80).
- The report is written as is: Environment not built, the stage and the tool or Verifier it stopped on, what was completed, what remains, the cost so far, and the estimated cost to finish (from the per-stage numbers already recorded).
- Continuing requires a person's permission, given as a new ceiling; `build --iterate` resumes from the content-addressed state (pipeline is idempotent, harness-design section 8), so nothing already paid for is redone.
- The 40% context cap (D65) is unaffected: it is per call and refused per call, this is the per-build ceiling.
- Rejected: finishing with assisted tools (hides the gap), stopping at stage boundaries (overrun by one stage's cost without asking).

### D87. The Builder carries lessons between customers, and must question each one's relevance (2026-08-27)

Q52. I said: "lessons that it learnt from the previous customers sir ( so that it doesn't repeat ) but also ask it to question the relevance of the lessons as well."

- What travels: a lessons file written by the improvement agent, one entry per lesson with the failure pattern it came from, the edit that fixed it, the anchor result that confirmed it, and a relevance condition ("applies when a tool returns a list", "applies when policy text contains 'unless'"). No customer data, no customer tool names, no artifacts (option 3 rejected; the trust ladder ADR-0006 stays intact).
- Anonymization is a gate on the lessons file: an entry that names a customer's tool, field, entity or value fails to save; the improvement agent must rewrite it in terms of the pattern.
- At the start of a build the Builder reads the lessons and, for each, answers whether its relevance condition holds for this customer's traces, with the evidence (a `ToolSig`, a policy span). Lessons judged not relevant are set aside for that build and the report lists them with the reason, so a wrongly discarded lesson is visible. Lessons judged relevant are applied and their outcome on this customer's anchor is appended to the entry, so a lesson that stops paying off loses standing.
- A lesson with N applications and no confirmed benefit is retired by the evaluator (D64), not by the Builder.
- Loaded when needed (D65): the lessons file is small and read whole at build start; the tree of past builds is grepped, never loaded.

Where research pulls the other way: AHE and Self-Harness carry the harness itself (prompts, code) forward and measure on the next task, which is option 3 in spirit; carrying only lessons is slower on the second customer and is the price of never letting one customer's code shape another's.

### D88. Failure attribution: code first, judge on the rest (2026-08-27)

Q53. I said: "3".

- Every failed Run gets a cause beside its failing atom (D46): `candidate`, `environment`, `simulated_user`, or `undetermined`.
- Code first. A failed Run is marked `environment_suspected` when any of its events was assisted (D49), hit a `fact_unavailable` (D77), touched a flagged tool (D70) or an overlay miss (D74), or when the failing atom's value came through a tool whose held-out fidelity is below 100% explained (D80). Such a Run does not count against the Candidate until the setup review or the audit resolves it; the report shows the count.
- Judge second. A failed Run with no such mark goes to `judge.py` with the transcript and the Reference Run side by side; the judge names the cause and cites the span, with its audit rate (D48). `undetermined` is the judge's abstain (D57) and goes to a human.
- An `environment` or `simulated_user` cause from the judge is a Builder failure pattern (D69 self-modification input) and a regrade trigger once fixed (`regrade.py`); a `candidate` cause is what the pass rate counts.
- Cost: one judge call per failed Run without a code mark.

Where research pulls the other way: TRAIL (R-earlier, in todo "Process reward for the path") reports 11% localization accuracy and kappa 0.55 for LLM error localization on agent traces; naming the cause is easier than naming the step, but the audit rate on this judge is expected to be the lowest of the four judge uses, and the report must show it.

### D89. The Candidate sees nothing that came from the Verifier (2026-08-27)

Q54. I said: "no! what do you recommend ?" Recommendation given after the answer: option 1, with the production-setting reading of option 3 folded in.

- The Candidate receives exactly what the recorded agent received: the customer's system prompt, the tool definitions as sent, and the user turns (now from the Simulated user). Nothing produced by the Builder for grading (Verifier atoms, Intent text, the Simulated user's fact list, Category or Task names) reaches it.
- If a customer's production system passes a label or context to their agent, it is in the traces as part of the prompt and the Candidate gets it the same way, because it is production input (D65), not Verifier output. The Intent is never substituted for it.
- The Builder never reads grader fields from benchmark inputs (D66 strips them at ingest); the Runner never reads Verifier files (`loop.py` and `route.py` have no import path to them, enforced by a test in `validate.py`).
- D79 check 7 (leak grep of the Intent and the user rules) stays, because the Simulated user's utterances are the one channel through which Verifier knowledge could still reach the Candidate.

### D90. tau2 shape first, the loop written so it can be stepped, the OpenEnv wrapper after (2026-08-27)

Q55. I said: "but our environment will envolve ( tau2 is the baseline right now, we will move further into more high fidelity better environment than tau2 ) so thinly wrap ? but lets first build the tau2 shape and then we build the openENV wrapper".

- Order: the tau2 file shape (D56) is built first. The OpenEnv wrapper (`reset() / step() / state()`) is built after the tau2 slice passes its gates, as a thin layer over `route.py` and the Task's Starting state, about a hundred lines.
- So that the wrap is mechanical later, the Runner's loop is written now as a function that advances one turn and returns (option 3 of Q55), and `validate.py` has a test that drives a Run step by step and gets the same JSONL as the whole-loop call. No OpenEnv code in the first slice.
- My framing, recorded here: tau2 is the baseline, not the destination; the Environment is expected to move past tau2 in fidelity, and the wrapper is what lets training frameworks consume it when it does.
- Todo entry updated: OpenEnv wrapper, gated on the tau2 slice.

### D91. `verifier.py` lives in the Builder and calls the Runner through records only (2026-08-27)

Q56. I said: "1. what do you suggest ?" Recommendation given after the answer: 1, with the one discipline from option 3.

- `verifier.py` is a Builder module. It asks the Runner for k re-runs (D78) the way anyone does, through `cli.py run` or the same function, and reads the resulting `Run` JSONL files back from disk. It never imports Runner internals and the Runner never imports it (D89 test covers both directions).
- The Runner therefore never knows what a Verifier is: it executes Runs and writes records. All grading rules, and everything the Builder learns about them (D69 self-modification), stay on the Builder side, so "Runner frozen" (D61) holds.
- The re-runs count against the Builder's spend ceiling (D86) and each model call against the 40% cap (D65), which is where I wanted the cost seen.
- Option 3's separate `derive.py` is not needed: the discipline it buys (no execution inside derivation) is the same as "talk to the Runner through records", and one file is fewer to keep straight.

### Correction from R27 (2026-08-27)

D76 and D80 cite "tau3 measured 18.5% evaluator-human misalignment" on nl_assertions. R27 traced the figure: it is a four-benchmark aggregate validity-audit rate, not a tau2 or tau3 specific judge-accuracy number. The direction (judge atoms disagree with humans often enough to need an audit rate) stands; the number should not be quoted as tau3's. R27 also could not verify "RubricForge" or "AgenticAI-Supervisor" (cited in R25) as independently citable artifacts under those names; treat those two citations as unconfirmed until R25's sources are rechecked. Better anchor from R27: the best LLM judges on agent trajectories reach about 69 to 70% precision on AgentRewardBench, so a transcript-reading judge should be assumed wrong on 20 to 30% of plausible cases.

### D92. Agentic judges by default; disagreement between judges replaces the human label; the pipeline must run without human support (2026-08-27)

Q57 and a reframe. I said: "agent as a judge should be the go to thingy because model + verifier is when we see the biggest gains ( we need agentic judges as well ). we don't have human labelled set and our end goal is to just have a synthetic environmetn creator withotu the human support ( agentic judges and then we see the disagreement )".

- Every judge use (D57 Reference confirmation, D76 policy atoms, D84 semantic equality, D88 failure cause, the dispute path) is an agentic judge: it has read tools over the Task's Starting state and the Run's End state, the deterministic Verifier's output, and must run at least one check before answering (R27: tool access raised agreement from under 42% to about 72%; Agent-as-a-Judge plus 4 to 30 points at 97.6% lower cost than a plain judge).
- Abstention signal (Q57): disagreement between two different agentic judges (different models, or the same model under different personas when only one model is available), each with tools, plus the Verifier's own output as the tie-breaker where it applies. No stated-confidence gate on its own (R27: self-consistency at 0.8 still wrong 48% of the time).
- No human-labeled set is assumed. The pipeline must complete without a person: where earlier decisions send an item "to a human" (D57, D76, D79, D88 `undetermined`), it now goes to a disagreement queue that is listed in the report with the two judges' verdicts and cited spans. A person may resolve the queue; the build does not wait for them.
- The report line "audit agreement rate" (D48) becomes "judge disagreement rate" until human labels exist; when a person does resolve queue items, those resolutions become the labeled set and the audit rate appears beside the disagreement rate.
- Labels we do get for free: on the tau2 slice, tau2's own reward per Run (D50 proof 2); at a customer, their verdicts (D62 stop condition). Both are used to calibrate the judges' disagreement threshold, and neither is a human panel.
- Where this touches earlier decisions: D48's two human checks stay available and are no longer required for a build or a Verdict to exist; D85 is unchanged (the customer decides on routing; that is the customer's decision, not pipeline support).

Where research pulls the other way: R27's abstention section rests on human agreement targets (Trust or Escalate, SCOPE's conformal guarantee) that need a labeled set; without one, my disagreement rate has no proven error bound and must be reported as such. tau3 fixed 75 to 100 tasks with human review (R22); my bet is that two agentic judges with tools plus the Verifier catch most of that class, and the tau2 slice is where the bet is measured (judge disagreement against tau2 reward).

### D93. A disputed Reference sets the Task aside until a person resolves it (2026-08-27)

Q58. I said: "human resolves the dispute then sir."

- When the two agentic judges (D92) disagree on whether a Reference is good, the Task is set aside: no Verifier, no Verdicts, listed in the report as "not gradeable, Reference disputed" with both judges' verdicts and cited spans. The rest of the build continues; the pipeline does not wait.
- A person resolves the dispute. Their resolution is recorded with `classified_by: human`, becomes a labeled item (the start of the audit rate, D92), and the Task re-enters the build on the next `build --iterate`.
- No third judge and no majority vote for References: a Reference sets the bar for every Verdict on its Task, and I want that call made by a person when the machines split. Judge disagreement on other atoms (D76, D84, D88) still goes to the queue without stalling the Verdict, per D92.
- Consequence for D92's "without human support": the pipeline completes without a person, and Tasks with a disputed Reference are the part of the output that a person unlocks. The report shows how much of the corpus sits there.

### D94. End state only, reaffirmed against R28 (2026-08-27)

Q59. I said: "i mean we did the research right and that said that we should only care about the end state also we are not grading the process right now so."

- ADR-0004 and D43 stand: the Verdict grades the End state (world writes plus what the user was told or asked) and nothing about the path. R28's suggested "required write tools were called" atom is not adopted; it is a path check.
- The case where two paths reach the same End state by different means (cancelled row versus deleted row): a deleted row and a cancelled row are different End states already, because the state diff records the row's presence and its columns; where the customer cares about the means beyond that, it is a Hard constraint they write ("never call delete_order"), checked on the transcript as D43 case 3 allows. No widening of the End state definition is needed for this case.
- Process grading remains a todo (process reward for the path, deferred 2026-08-26) and never a Verdict input.
- Recorded as a reaffirmation so R28's disagreement is visibly answered rather than left open.

### D95. Truncated tool results: ask for the rest, reconstruct meanwhile, tagged and Assisted (2026-08-27)

Q60. I said: "ask for the rest and reconstruct also i don't think any schema would be 9000 character long as well."

- Ingest marks the call `truncated` with the visible length and the cut marker it found. The report lists, per tool, the share of calls truncated, and the trace-intake sheet (question 3) asks the customer for the full results from their own store or a raised log limit.
- Meanwhile the Builder reconstructs the missing part from the tool's result schema (D72 union) and from complete calls to the same tool, with the same tag as D40 synthetic rows. The reconstructed part is never Verdict evidence: a Run whose Candidate read a reconstructed result is Assisted (D49) until the real result replaces it, at which point `regrade.py` re-scores it.
- D41 boundary kept: reconstruction serves plausible output so the Environment keeps running; it does not claim to be what the agent saw.
- Clarification recorded: it is results, not schemas, that get long (a search returning fifty rows, a full ticket history); the schema of such a result is short and is what makes reconstruction possible. The 9,000 figure was illustrative.

### D96. Coverage is counted in Tasks, against the Task list fixed at the start of the build (2026-08-27)

Q61. I said: "1. i won't trust touching the rows as the selection 2. lets define what does coverage means? i would love to hear it in the form of total tasks covered ( which were defined early )."

- The Task list comes from `cluster.py` at the start of the build (D83) and is frozen for that build. Coverage is measured against that list, so Tasks discovered or split later do not inflate it.
- A Task is covered when every one of its Runs replays with no assisted event (D49), no `fact_unavailable` (D77), no overlay miss (D74), no truncated result still reconstructed (D95), its Reference is confirmed (D57, D93), and its Verifier passed the D79 suite. Anything less and the Task is uncovered, with the first failing reason attached.
- Two numbers on the scorecard: covered Tasks over total Tasks, and the same weighted by Run count, so a customer sees both "31 of 40 Tasks" and "92% of traffic". Categories (D83) roll up the same way.
- Rows, columns and paths (Q61 options 1 to 3) are not coverage; they are the reasons listed under an uncovered Task ("3 Runs hit an overlay miss on `orders`", "tool `search_products` assisted on 12 calls"), which is what the Builder's improvement loop (D69) reads.
- Replaces "state coverage" and "policy coverage" in the D62 scorecard with Task coverage; policy coverage stays as its own report line per R22 item 10 ("your traces exercise 6 of 40 policy items"), because it is about the customer's policy, not the Environment.

Where research pulls the other way: R29 found no formal coverage metric in the literature and pipelines use proxy counts (tools, domains, novelty); counting Tasks is my definition, and its weakness is that a Task with one Run is covered as easily as one with forty. The Run-weighted number and the unguarded flag (D81) are what show that.

### D97. Remaining build-level details closed by defaults, to be revised by the tau2 slice (2026-08-27)

I said: "also how many questions are left ? feel that these are too many questions." The items below are implementation choices the slice can test more cheaply than a discussion can settle. Each is a config default with the assumption it rests on; the slice's numbers, not a further question, decide whether it changes.

- Judge samples per verdict: 2 agentic judges, 1 sample each at temperature 0; a third sample only when the two disagree on a non-Reference atom (References go to a person, D93). Assumption: R27's finding that majority voting helps applies less once each judge has tools.
- Disagreement threshold: none to calibrate at first, since disagreement is binary between two judges; on the tau2 slice, judge-vs-tau2-reward agreement is reported per judge use so a weak judge use is visible.
- Judge prompt shape: R27 section 8 as written, per use (pairwise with field type for D84; atomic yes/no sub-questions for D76; Trust-or-Escalate framing with the Verifier's output shown for D57; tool-using with read-only queries and at least one query required for D88 and the dispute path); every output carries verdict, cited spans, and the tools it ran.
- `provider.py`: as recorded from R26 (section 4 item 20); revisit only if a provider's quirk breaks a Run.
- Sub-versions on Verdicts (R29 rule 10): `Environment` carries `schema_version`, `tools_version`, `policy_version` beside `env_id`, and `Verdict` copies them; cost is three fields.
- Intent for a Task with one Run: written from that Run alone, the cross-family grounding check is skipped and the Task is marked unguarded (D81), which the report already shows.
- Policy coverage wording: "your traces exercise N of M policy items; the rest are not tested" as one line under the Environment section (D85), with the untested items listed.

Left open on purpose, because I still need to decide them: D71's three parts (Simulated user tools, user-side writes against sequence Hard constraints, required vs allowed for user-side writes). Everything else in the grill is closed; the next input is the vendor export and the tau2 slice.

### D98. The name rule for a generic tool, and what a generic tool is never credited with (2026-08-28)

D68 lists `generic` as a kind and D70 leaves the default open, but no decision named which tools are generic, so `propose_kind` had no rule for the kind it could return. The rule now in code, recorded here so the code is not the only place it lives:

- Name test, after the read and write prefixes have had their turn: `^(calculate|compute|think|reflect|transfer_to_human|escalate_to_human|hand_?off_to_human)`.
- Confidence `medium`, and the row is classified rather than unclassified: the name is evidence, so the setup review sees a settled row, not an open one. A name nothing matches is still `read` at low confidence with `unclassified: true` (D70), which is unchanged.
- A generic tool is never credited with an observed effect. Inverse replay may show the world changing across a generic call, but a tool that reads nothing and writes nothing did not do it, so the effect stays unattributed rather than turning the tool into a write.
- `compile_env.py` emits `@is_tool(ToolType.GENERIC)` for these, which is tau2's own toolkit shape.

Evidence: tau2's `src/tau2/domains/retail/tools.py` marks `calculate` and `transfer_to_human_agents` GENERIC, which is the same set this rule picks out of the retail corpus.

Where research pulls the other way: a name test is a guess about intent, and a customer whose write tool is called `compute_refund` gets a generic row that no Verifier atom will cover. The medium confidence and the setup review are what bound that; observed effects do not, because this is the one kind they cannot override.

### D99. Correction to D73: only system time and counter names are exempt at high confidence (2026-08-28)

D73 says "timestamps and counters `exempt`" as a name rule. Taken literally that exempts any column whose name reads like a time, and an exempt column is never compared, so a Candidate that corrupts a birth date, a delivery date or a version number would pass. That is the failure D73 exists to prevent, so the rule is narrowed:

- A name that reads as a system timestamp or a system counter (`created_at`, `updated_at`, `_seq`, and the like) stays `exempt` at high confidence.
- A name that reads like a date, a time or a version and is not one of those is `hard` at low confidence, so it is compared and the setup review sees it. It becomes `exempt` at medium confidence only when every value in the column has a timestamp shape, and then only for a reviewer to confirm.
- Whole numbers that only increase are still `exempt`, at low confidence, for the same reason: the review sees them.

The re-run override of D73 is unchanged and still outranks both: a column that varies across successful re-runs with the same outcome is `exempt` with `classified_by: observed`.

Where research pulls the other way: this makes the exempt set smaller than tau2's whole-database hash would suggest and adds rows to the setup review. The trade is deliberate; a wrongly exempt column is silent, a wrongly hard one is a review line.

### D100. Cluster similarity: idf-weighted token Jaccard, not a higher threshold (2026-08-28)

The first build recorded that the D97 cluster threshold of 0.3 should be 0.6 (pair F1 0.276 versus 0.720). The slice was rerun against the alternative in the same todo item, replacing the similarity rather than moving its threshold, and that is what is now in `cluster.py`:

- Token Jaccard weighted by corpus idf, argument keys dropped, complete linkage, default threshold 0.4.
- Pair F1 0.717 at the default, and 0.685 to 0.719 across thresholds 0.3 to 0.6, against 0.276 to 0.720 for the unweighted measure.

The number at the best threshold is the same either way; what changes is that the result no longer depends on picking it. This replaces the "raise the default to 0.6" line in the Pending block below and in `todo.md`.

Where research pulls the other way: the unweighted measure at 0.6 scores marginally higher on this one corpus (0.720 versus 0.717), so this is a trade of a third of a point of F1 for a flat curve. On a corpus whose traces share more boilerplate than tau2 retail's, the flat curve is worth more than the point.

### D101. Tool kind comes from what the calls show, not from a verb list (2026-08-29)

`propose_kind` decided read or write from `READ_PREFIXES` and `WRITE_PREFIXES`, which are retail's own
verb vocabulary. The cross-domain check showed what that costs off retail: airline's `book_reservation`
and `send_certificate` and telecom's `send_payment_request` are real writes mined as reads, so no
Category write signature ever contained a booking, a certificate or a payment request.

The lists are not widened. Three signals read off the recorded calls now decide, and the name rule is
only what is left when none of them speaks:

1. A later read shows a field this call changed, and no other call explains it. Unchanged, D68.
2. Most of what came back is what the call sent. A read answers with the world; a create answers with
   what you handed it. Measured across all three domains the two do not overlap: `book_reservation`
   sits at 0.81 and the highest read anywhere sits at 0.20.
3. Every call answered with a message rather than data, the tool was handed something that names a row,
   and no read of that row ever showed it unmoved. This is the weakest of the three and the only one
   held to `MIN_OBSERVED_CALLS`, the floor the mine gate already uses: a tool seen once or twice stays
   unclassified and flagged rather than being called a write on its answer shape alone.

Alongside them, `quiet_tools` reads the same evidence the other way: two identical reads that bracket a
call with nothing changed are evidence against a write. A read only counts as evidence about a call
when the read asked about what the call named, which mirrors how `_explains` credits a change.

Result, with nothing tuned per domain: kind is exact on all three, 15/15 retail, 14/14 airline, 9/9
telecom, against 15/15, 12/14 and 8/9 before.

Where this pulls the other way: airline's cluster pair F1 falls from 0.788 to 0.756, because two real
writes now enter the Category signature and split Runs that a missing write had held together. The
signature is more truthful and the clustering is slightly worse, and the signature is the thing the
Verifier rests on, so the trade is taken.

### D102. An id is a column the calls address and that is distinct per row (2026-08-29)

`_is_id` was `name == "id" or name.endswith("_id")`, which is retail's id convention. Airline's
`flight_number` defeats it completely: the `flights` table was never proposed despite 338 calls
showing its rows, and every Gate A loss on airline traced back to it.

`id_columns` now reads the corpus for two things at once. The column is passed as an argument by some
call, which is what makes it an id rather than a value; and wherever a result came back with several
rows carrying it, every one of those rows had a different value for it, which is what an id does and
what a status, a price or a filter does not. No threshold and no vocabulary, so nothing here is fitted
to a domain: being addressed alone would make `origin` an id of every flight search.

`id_field` reads the columns the miner recorded a pattern for after its three name candidates, which
is how a table whose id the name rule cannot see stops being proposed and left empty.

### D103. A tool is about the noun before the preposition (2026-08-29)

`_table_of` walked a row's `_id` keys and took the first whose singular appeared anywhere in the tool
name. Telecom filed every `Bill` under `customers`, because `customer` is a token of
`get_bills_for_customer` and `bill` is not, and the `bills` table was lost entirely.

A tool name says what it is about before any preposition and how it is addressed after one. The noun
is now the head of the run before the first preposition, and the tie-break is: the id whose entity is
that noun; then the id whose values are distinct across the rows this one came back with, which a
foreign key's are not; then the only id there is.

Result: airline recovers `flights` (3/3 tables) and telecom recovers `bills` (5/5), retail unchanged
at 3/3, and every field our rows carry that the real rows also carry is exact on all three
(1559/1559, 2393/2393, 83/84).

### D104. The mined id patterns were never actually read (2026-08-29)

Found while threading D102 through, and not a cross-domain issue at all. `mine_schema` records a
pattern per column keyed `table.column`; `match_table` and `referenced_ids` looked it up by the table
alone. On every mined schema the lookup missed, the pattern came back `None`, and both guards let
every candidate through whatever its id looked like. `id_pattern_for` reads either key, and a test
holds that the guard rejects an argument that is not an id of that table.

### D105. The screen is the pipeline, not a chat (2026-08-29)

Asked for a TUI "inspired from pi", with feynman.is named as the reference because it is built on
top of Pi and the difference between them is supposed to tell us what to build. The difference is
not a better chat. Feynman keeps Pi's loop and adds two things: named workflows over one domain
(`/deepresearch`, `/lit`, `/replicate`, `/audit`) and provenance for what each one produced. Both
of those already exist here under other names. The stage graph is the named workflow and the
content-addressed cache is the provenance, so the screen invents neither: it lists the stages
pipeline.py is about to run, marks the one running, and shows the cache hash each stage wrote.

What it adds that no file gave us before is timing: `pipeline/state.json` is written once, when
the pipeline is done, so until now a build in progress was invisible. `Pipeline` now takes an
`on_event` callback and emits a stage start, a stage end, every gate result and the final status.
The callback is a screen, not a stage, so anything it raises is swallowed: a view that dies has no
business failing a build that was going fine, and there is a test for that.

The two numbers that decide a live build are on it and nowhere else together: what the gates said,
and what has been spent against the D86 ceiling. Both are read back from the files the build
writes, never from state the screen keeps, so closing the screen loses nothing.

Left out on purpose: a chat pane, a tool picker, an approval prompt, a diff view. The pipeline is
the conversation. `harness tui` opens it; `/build`, `/run`, `/status`, `/keys` are the commands.

## Pending (asked, not yet answered)

- D71 provisional (user-side writes); I want more discussion: Simulated user tools, interaction with sequence Hard constraints, required vs allowed. Three questions, to take up when I'm ready.
- Defaults set by D97 (judge samples, judge prompts, provider, sub-versions, single-Run Intent, policy coverage line) are revised by the tau2 slice's numbers, not by further grilling.
- R29 open note: expert curation vs no-human default (D92); revisit when the first customer's disagreement queue shows its size.
- Grill of 2026-08-27 complete: D69 to D97.
- First build done 2026-08-28 (756 tests, offline slice in the README's measured section). Slice evidence that revises D97 defaults: cluster threshold 0.3 should be 0.6 (F1 0.276 versus 0.720); `propose_kind` needs a `generic` kind or a rule for it; error payloads need a per-source prefix rule (`Error: ` in tau2). The first two are now closed, by D100 (the similarity was replaced rather than the threshold raised) and D98 (the generic name rule). The error prefix rule is still open and still listed in `todo.md`.
- Verification pass of 2026-08-28 (1,153 tests). Two Runner sandbox holes closed by static gates before `exec`: an atom predicate in `verdict.py` and a generated tool body in `compile_env.load_toolkit`, both of which had reached `os` through `().__class__.__base__.__subclasses__()` under a restricted `__builtins__` alone. A real sandbox for model-written code is still deferred (design section 4) and the checks say so; `todo.md` carries it.
- Size bands in design section 10 no longer describe the code. The overage is recorded in that section, not fixed: whether the bands move or the modules split is the design owner's call.
