# Design philosophy

This is why Kullback looks the way it does, in one place. What I built, why, and what I left out on purpose. The decision log (`decision-log.md`, D01 to D100) has every single choice next to the alternative it beat. The ADRs in `adr/` hold the four I can't easily undo. This document is the shape of the whole thing.

## Where it starts

I gave myself three rules and everything else follows from them. It should be simple enough and get the job done in high quality. It should be as efficient and simple as possible. And I care much more about generalizability than about overfitting.

Simple means I can explain every stage in one sentence, and the first version of each stage is the plainest one that keeps the trust gates honest. It doesn't mean small at any cost. Where complexity buys something I can name (per-Task overlays, the repair loop, two judges) it goes in, and I write the reason next to it.

Generalizability means every stage gets measured on runs it never saw. The number on the seed runs is printed next to the number on the held-out runs, every time. A gap between them is reported as overfitting, not averaged away.

High quality means I build for the hardest runs first: long, multi-turn, many tool calls, off the recorded path. The common single-call run is then just the easy case of the same machinery, not a separate path with its own bugs.

## What I did, and why

### The trace is the only truth

Every customer file is stored byte for byte and hashed before anything reads it. Every derived value, from a tool signature to a column class to a line of an intent, points back to the byte range it came from. When a number in the report is wrong, the chain from that number to the raw bytes is short and mechanical.

I did this for trust. The claim is "on your runs, this model does the job". A customer can only believe that if they can follow any figure back to their own data. It also kills a whole class of argument: if something can't be pointed back to a trace, it isn't in the environment. Full stop.

### Verdicts come from the end state after re-execution

A candidate model runs through the conversation in the rebuilt environment. The verdict is computed from what changed in the world and what the user was told, compared with what the frontier model's successful runs changed. The route the model took, the order of its calls and the text of its reasoning never move a verdict. (ADR-0004.)

Replaying a logged run statically with a different model stops making sense at the first divergent action: a 2026 CMU study found only 3 to 8 percent of later recorded states stay valid once an early action changes. Path matching measures imitation, not competence, and punishes every valid alternative route. Reasoning text can't be graded from text alone: chain-of-thought faithfulness sits at 20 to 40 percent and drops as models get stronger, and editing only the reasoning inflates judge false positives by up to 90 percent. Every agent benchmark that has lasted grades the world. So do I.

"End state" means effects, and effects land on two things: the world (writes) and the user (what they were told or asked). A question the frontier asked in every successful re-run is a required atom. A write whose value came from the user's answer has to hold the answer given in the candidate's own run. Whatever is left over is "path", and that is exactly the part with no effect.

### Code decides, models propose

Code computes the verdict from the run's event log and the Task's verifier. Models show up at defined points in defined roles: writing a tool body that then has to pass five gates, proposing a column class that a rule verifies or a re-run overrides, naming a cluster whose membership code already decided, and answering a question that code couldn't compile into a predicate.

Where a model judges, two agentic judges do it independently. Each has read tools over the starting and end state, each has to run at least one tool check and cite a span. If they disagree, the run goes to a queue for a person. A judge can take a run off the bar or widen a verifier for every candidate. It can never award a pass. (ADR-0003, D92.)

The reason is that the two mistakes don't cost the same. A conservative error keeps a Task on the frontier model and costs the customer some savings. A liberal error moves a Task that then regresses in production and costs the customer's trust in the whole report. I built the machinery to make only the first kind of mistake. That also means the savings in the report are a floor, not an estimate.

### Only replicas count toward the bar

The bar is the frontier model's score on the customer's own inputs. Only re-executions of recorded runs feed it. Synthesized scenarios, however well grounded, are a later phase for coverage, hardening and post-training, and get reported separately if at all. (ADR-0005.)

A synthesized task is not the customer's input. Letting it into the verdict changes the claim from "safe on your traffic" to "safe on tasks we invented", in either direction. Weighting it lower doesn't fix that. Any weight makes the bar a blend the customer can't audit.

### Gates, never blends

Every stage ends in a code gate that passes or fails. The failure is attached to the artifact and handed back to the stage that produced it, for a bounded number of retries. There is no score that mixes a good environment with a weak verifier into one number. Gate A wants 100 percent of recorded writes reproduced after canonicalization and zero unexplained read mismatches, on the held-out runs as well as the seed. A verifier enters the pool only after the recorded run passes it, an empty run fails it, a plausible wrong run fails it, and a grep finds no leaked reference.

A run that needed a model to stand in for any tool call is reported as assisted and gets no counted verdict. A Task with too few runs to hold anything out is marked unguarded. A tool nobody could classify is flagged and blocks the review. None of these get averaged in. The report opens with whether the environment was built at all, before it shows a single per-Task number.

A blended score hides exactly the failure it should expose. The one number a customer would look at would be a mix of environment fidelity, verifier validity and model quality, and nobody could say which one had moved.

### Data follows trust, one rung at a time

The environment is built from traces alone first. Every entity a run read becomes a row. The shared world is reconstructed by replaying the corpus backwards, and each Task carries an overlay of the rows its runs actually saw, in the version they saw. Unseen entities return "not found" and the run is reported assisted. Tool definitions that already travel in the traces are parsed from day one. Schema, policy documents and finally a database snapshot are asked for later, and each rung has to be justified by the assisted share the previous rung left behind. (ADR-0006, D74.)

The first conversation with a customer must not start with a data-sharing agreement. Their traces are already leaving their system to a tracing vendor. A snapshot is a different conversation, and the assisted share is the evidence that makes it a reasonable one.

### Two programs, one set of records, and a wall between them

The Builder writes the environment. The Runner executes runs and computes verdicts. They share the record definitions and nothing else. The Runner has no import path to the Builder, and the candidate model sees nothing from the verifier: not its atoms, not the grader fields of a benchmark trace, not the reference. A test scans the imports and fails the suite if the wall is breached. (D89, D91.)

The loop that advances one turn is a single function, written so that a `reset` and `step` wrapper for any environment standard is packaging, not redesign. The tau2-bench file shape came first because it is the one place I hold ground truth for the reconstruction. The wrappers come after the slice passes its gates. (D90.)

### The report shows, the person decides

The report opens with whether the environment was built (gates, scorecard, assisted tools, unguarded Tasks, overlays), then gives each Task's numbers and a suggestion. The routing plan is written from what the person decides, not from the suggestion. When the spend ceiling is hit, the build stops where it is and reports as it is. Nothing gets quietly switched to a cheaper path to finish. (D85, D86.)

### The Builder improves itself, slowly and under watch

The Builder may edit its own prompts, rules and code, one change per round. Each edit is a node in a memory tree, and an evaluator outside the loop accepts or rejects it on the held-out runs. Batches of changes are allowed only after twenty accepted single rounds, with bisection on a reject. Lessons from one customer travel to the next as a file, and the Builder has to argue the relevance of each lesson, with evidence, before applying it. Lessons it sets aside are listed in the report. (D82, D87.)

## What I didn't do, and why

I don't grade the path. A different tool sequence that reaches a satisfying end state passes. Path differences are reason codes on the report.

I don't grade reasoning text. It is watched for fabricated observations, nothing more.

I don't add up step-level matches into a run score. Agents recover from most local errors, only about 16 percent of steps are outcome critical, and all-or-nothing aggregation is systematically pessimistic.

I don't let a judge overrule an end-state check or award a pass, and I don't bring in a third judge to break a tie on a reference. A disputed reference sets the Task aside as not gradable until a person resolves it. (D93.)

I don't count synthesized scenarios toward the bar, and I don't synthesize rows for entities the traces never showed and then treat them as real. Synthetic rows are tagged and their runs are assisted.

I don't cap or compact the candidate's context. The candidate sees the production setting: the system prompt, the tool definitions and the conversation the customer's agent gave it, with production's own compaction if the traces show one. (D65.)

I didn't build an event bus, a plugin system, MCP inside the core, subagents, a UI framework, or session trees for runs. A fork is a new run with a parent id. Each of those would be a second way of doing something the records already do.

I didn't train a user simulator. The simulated user is rules from the recorded user: facts exact, style representative, and when a fact is neither in the trace nor in the world it says so and the run is marked. Prompt-only simulators err 40 to 47 percent of the time in the published measurements, tool-constrained ones about 16 percent. Rules grounded in the recorded facts are the version of that I can audit. (D44, D77.)

I didn't keep the benchmark secret to make it last. The counter-argument in the research is that expert curation, not secrecy, is what extends a benchmark's life. My default is no human in the loop until the disagreement queue shows how big it is. This one is still open. (R29 against D92.)

I didn't write the sandbox yet. Model-written code (compiled tools, verifier atoms, constraint predicates) passes a static check before it runs: no imports, no dunder attributes, restricted builtins, and a refused atom becomes `not_verdicted` rather than a pass. Compiled tools also run in a subprocess for the compile gates. That reduces the blast radius, it is not a security boundary, and a real sandbox is the first item on the list before any customer trace gets compiled.

## Where the evidence disagrees with me

I keep these in the open because a design that hides its counter-evidence is asking to be trusted instead of checked.

EnvFactory's ablation found that half trajectory match plus half final-state equivalence beat either one alone. BFCL multi-turn requires the ground-truth call sequence to be a subset of the executed calls. AgentScaler requires an exact call match for read-only tasks. I hold to end state only, and the dissent is written into ADR-0004.

The synthetic-data research wanted a light path check on top of the end state. I said no (D94), for the same reason.

The re-run count k that fixes the required and allowed atom sets is an assumption (A30, "stabilizes within ten re-runs") until the experiment on tau2 retail decides it.

The first clustering default was wrong on real data: threshold 0.3 gave F1 0.28 against tau2's task groups where 0.6 gave 0.72. The design said the slice would revise the defaults, and it did. The similarity itself got replaced (D100), and F1 now sits between 0.685 and 0.719 across thresholds 0.3 to 0.6. That default was fit on retail, so airline and telecom are the next check.

The design budgeted 2,700 to 3,700 lines. The code is about 13,800 after the first build and a verification pass, and `harness-design.md` section 10 lists every module against its band. Whether the bands move or the code shrinks is an open call, and it's mine to make.

## How to argue with any of this

Find the D number, read the alternative it beat, and open an issue that names it. A changed decision is a new entry in the decision log. An ADR is only for a decision that is hard to reverse, surprising without context, and the result of a real trade-off. Numbers that contradict a decision are the strongest argument there is, and the README's measured section is where they go, with the script that produced them.
