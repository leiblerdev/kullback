# Kullback

Kullback rebuilds the world your agent works in from the traces it already produced, checks the rebuild by replaying those traces, and runs other models through it, graded on what they changed.

The one claim I built this to earn, and haven't earned yet: a 2B parameter model post-trained in an environment Kullback built from traces performs on the real tasks. Everything below is the reconstruction work that has to hold before that experiment means anything.

It is the open-source Builder and Runner behind [Leibler](https://leibler.dev). Apache-2.0. Python 3.11.

## 1. What this is and what pain it solves

If you run an agent in production, you have a log of everything it did: the prompt, the tools it called, what they answered, what it wrote back. You probably pay a frontier model for every run, and suspect a cheaper model could handle a good share of them. Proving that is the hard part.

The usual way is to build an eval: someone writes tasks by hand, someone mocks the tools, a judge model scores transcripts, and weeks later you have numbers nobody trusts, because the mocks drift from the real system and the judge grades the wording, not the outcome. The tasks that matter most, long multi-turn runs with many tool calls, are the ones a hand-built eval covers worst.

Kullback starts from the other end. Your traces already contain the tasks, the tool signatures, the data the tools returned, and the effect every write had. That's enough to rebuild an executable copy of your system: a database with the rows your runs touched, one function per tool that behaves the way the real tool was observed to behave, the policy rules your agent was told to follow compiled into checks, and a simulated user who knows what the real user knew. Once that copy exists, any model can run through the same conversations, graded on what it changed in the world, not on how it talked about it.

Two things follow from grading the end state instead of the transcript. The grader is code, so it's cheap and has no opinions. And a model that reaches the right outcome by a different route isn't punished for the route, which is the whole point of asking whether a different model can do the job.

## 2. How it solves it

Every harness runs the same loop. Traces go in, an Environment comes out, candidates run in it, code grades what they changed, you read the report and decide, and what you decide, plus the next week of traces, feeds the next build.

```mermaid
flowchart LR
    T[Your traces] --> B[Build the Environment]
    B --> R[Run candidates in it]
    R --> V[Verdict per Run, code only]
    V --> P[Report per Task, you decide]
    P -. new traces, fixes, disputes .-> B
```

Inside the loop, Kullback is two programs over one set of data records, with every model behind one interface.

```mermaid
flowchart TB
    T[(Your agent traces)] --> I

    subgraph B[Builder]
        direction LR
        I[ingest] --> M[mine and cluster] --> C[compile tools, policy, verifiers]
    end

    C --> E

    subgraph E[Environment]
        direction LR
        W[(db and overlays)] ~~~ K[tools and predicates] ~~~ V[tasks and verifiers]
    end

    E --> L

    subgraph R[Runner]
        direction LR
        L[loop] --> D[verdict]
    end

    CA[candidate model] -.-> L
    J[two judge models] -.-> D
    D --> P[Report per Task]
```

The Builder reads traces and writes an Environment. The Runner takes an Environment, a recorded run and a candidate model, replays the conversation and computes a Verdict. They share nothing but the record definitions; the Runner cannot import the Builder, and a test checks that.

### The Builder, stage by stage

Every stage is a function over records on disk, content-addressed, ending in a code gate. A stage that fails its gate hands the artifact back to the stage that produced it, with the failure attached, for a bounded number of retries. A share of every Task's runs is held out from the start and never used for building, so the numbers you see at the end are measured on runs the Builder never saw.

```mermaid
flowchart LR
    raw[(Raw traces)]
    subgraph S1[1 ingest]
        direction TB
        i[Trace records] --> ig{{every call parsed<br/>grader fields stripped}}
    end
    subgraph S2[2 mine]
        direction TB
        m[Tool signatures<br/>tables and columns] --> mg{{3 calls per tool<br/>or flagged}}
    end
    subgraph S3[3 cluster]
        direction TB
        c[Categories and Tasks<br/>one line intent each] --> cg{{intent grounded<br/>in more than one Run}}
    end
    subgraph S4[4 compile]
        direction TB
        e[db.json<br/>one tool body per tool] --> eg{{Gate A: replay recorded calls<br/>writes 100%, reads 0 mismatch}}
        eg -. miss, rewrite, max 3 .-> e
    end
    subgraph S5[5 policy and user]
        direction TB
        p[predicate per rule<br/>simulated user rules] --> pg{{passing and failing<br/>test per rule}}
    end
    subgraph S6[6 verifier]
        direction TB
        v[atoms from k frontier re runs] --> vg{{oracle passes<br/>empty run fails<br/>plausible wrong fails}}
        vg -. fail .-> v
    end
    env[(Environment)]
    raw --> S1 --> S2 --> S3 --> S4 --> S5 --> S6 --> env
```

Ingest stores your files unchanged and hashed, then derives trace records from them. Every derived field carries a pointer back to the byte range it came from, so a wrong value can always be traced to its source. Benchmark grader fields, if your traces come from a benchmark, are stripped into a sidecar that only the final comparison code reads.

Mine works out, for each tool, the union of every argument and result field ever observed, with counts and first and last sighting, and whether the tool reads or writes. Reads and writes are decided by rule from the name, then confirmed by diffing state before and after the call; a tool nobody can classify defaults to read and gets a flag that blocks the setup review until a person answers it. The same pass recovers the tables and columns behind the tools and marks each column exempt (ids, timestamps), hard (must match exactly), or semantic (may differ in form).

Cluster groups runs into Categories by the set of tools they write through, then into Tasks by similarity of intent inside each Category. Membership is decided by code; a model only names the cluster. Intent writes the one-line task description, and every noun phrase in it has to point at a span in a member run. An intent grounded in a single run is refused.

Compile builds the shared database by replaying the corpus backwards (latest observation wins), with a per-Task overlay of the rows each Task's runs actually saw, and writes one tool body per signature. A tool body is written by a model but must pass five gates in order: parses, executes on the starting state, deterministic, not a constant, and reproduces held-out recorded calls. A body that fails gets at most three rewrites with growing evidence, then the tool is marked assisted and every run that touches it is reported separately.

Policy turns the rules in the system prompt into predicates that run before every write, each shipped with one passing and one failing test. A rule that can't be compiled is rewritten into a checkable form for review; if that fails too it becomes a judge question, and a judge's rejection lists it as residual; it never enters a Verdict.

User sim writes the rules for a simulated user from the recorded one. Facts are exact; style is representative. If the candidate asks for something the trace never contained, the simulated user answers from the starting state, and if the world doesn't have it either, says so, and the run is marked.

Verifier derives the pass condition for each Task from what the frontier model did in k re-runs: the set of writes present in every successful re-run is required, the writes present in some are allowed, and anything else is forbidden. Each written value carries its provenance: a user utterance, a tool result or a policy rule. A Verifier enters the pool only after the recorded run passes it, an empty run fails it, a plausible wrong run fails it, and a grep finds no leaked reference in it.

### The Runner

```mermaid
sequenceDiagram
    participant C as Candidate
    participant L as loop
    participant W as World
    participant U as Simulated user
    loop one turn, one JSONL line per event
        C->>L: message with tool calls
        L->>W: route each call: code, else recording, else stand in (Run marked Assisted)
        W-->>L: result in your tools' own error encoding
        L->>U: turn ends
        U-->>C: next message from the recorded user's facts
    end
    L->>L: verdict: end state against the Task's Verifier, code only
```

The loop is one function that advances a single turn, so the wrapper for any environment standard is packaging rather than redesign. Route tries code first, then an exact recording, then a model stand-in; the route taken is on the event, and a run served by a stand-in anywhere gets no counted Verdict. Verdict is a separate pass over the run's JSONL and the Task's Verifier, code only: required atoms present, forbidden writes absent, policy predicates never fired, the user's questions answered. It reports pass or fail with the failing atom and whether the candidate took the same path as the recording or a different one.

Where a judgment call is unavoidable (a semantic column whose strings differ, a rule that could not be compiled, the cause of a failure), two agentic judges with read access to the starting and end state each have to cite a span and run at least one tool check. If they disagree, the run goes to a queue for a person. A judge can remove a run from the bar or widen a Verifier for every candidate; it can never award a pass.

The report opens with whether the Environment was built at all (which gates passed, how many tools are assisted, how many Tasks have too few runs to be guarded), then gives per-Task numbers and a suggestion. The decision is yours; the routing plan is written from what you decide, not from the suggestion.

## 3. What has been measured

Everything below comes from the offline part of the first slice: Sierra's public tau2-bench retail run (Claude 3.7 Sonnet as the agent, GPT-4.1 as the simulated user, 456 runs over 114 tasks), with no model calls anywhere. tau2-bench is the one place I hold the ground truth (the real `tools.py`, the real `db.json`), so it's where the reconstruction can be checked exactly.

Ingest read 456 runs, 3,591 tool calls, 109 tool errors and 0 truncated results. A hand count over the raw JSON, importing nothing from Kullback, gives the same four numbers. Three runs checked call by call match exactly. Hashes are identical across two ingests into separate directories.

Mining recovered 15 tool signatures from the calls alone. Against tau2's own `tools.py`, argument names match 15 of 15 in both directions, with nothing missing and nothing extra. The mined set of write tools is identical to tau2's seven. Read or write class matches on 13 of 15; the two misses are tau2's `calculate` and `transfer_to_human_agents`, which have no side effects and fell to the read-and-flagged default. The recovered schema has tau2's three tables with identical column sets, including the seven optional order fields that only appear in post-write results.

The reconstructed starting state matches tau2's `db.json` on every one of the 252 rows the runs touched: orders 161 of 161, products 38 of 38, users 53 of 53.

Gate A, replaying the recorded calls through tau2's own tools against my reconstructed database, on 20 seed runs and 10 held-out runs (257 calls): writes 37 of 37 and 20 of 20 after canonicalization, reads 125 of 125 and 65 of 65, recorded errors reproduced 6 of 6 and 4 of 4, end state over written rows 33 of 33 and 17 of 17. Replaying against tau2's original database instead of mine gives the same numbers in every cell.

Clustering was the one place the first default was wrong: at the original threshold of 0.3 the 456 runs fell into 74 Tasks with purity 0.53 against tau2's task ids, one of them holding 58 runs. The similarity was replaced (idf-weighted token Jaccard, complete linkage, default 0.4, decision D100), and F1 against tau2's task ids is now 0.685 to 0.719 across thresholds 0.3 to 0.6. The ceiling for any clustering by write tools is about 80%, because only 91 of tau2's 114 tasks have all four trials writing through the same tools. This default was fit on retail. Run unchanged on tau2 airline it gives F1 0.788; on telecom 0.207.

The same code was then run on tau2 airline and telecom with nothing retuned (`docs/cross-domain-check.md`). Ingest matched a hand count on both. Airline: 14 of 14 tool signatures, 12 of 14 kinds, starting state 147 of 148 rows, but the `flights` table was never recovered because the id detector only knows `_id` suffixes, and Gate A against my database fell to 10 of 27 writes on seed and 4 of 14 held out while the control against tau2's own database stayed at 100%. Telecom is where it breaks: the traces interleave the simulated user's own phone tools with the agent's, nothing in the miner reads `requestor`, so 38 tools were mined against 13 real ones, a user action landed in the task signature, cluster F1 dropped to 0.207 and 240 of 356 replayed calls had no tool to hit even on tau2's real database. Five retail-shaped assumptions are named in that file with the fix for each. The gates themselves went red on telecom and green elsewhere, which is what they are for.

The code is 23 modules, about 13,800 lines, with 1,167 tests that run in about sixteen seconds and never call a model. It is larger than its design said it should be; `docs/harness-design.md` section 10 records every module against its band.

What hasn't been measured yet, because it needs a live model: the compiled tool bodies and policy predicates, the Verifier on a real Task, candidate runs, and the agreement between my Verdicts and tau2's own reward. Also not measured: any trace format other than Sierra's tau2 export. Those are the next numbers, and they get published whether they're good or not.

## 4. The claim

I'm not claiming Kullback finds cheaper models for your runs, or that its environments are faithful, or that its verdicts agree with people. None of that has been measured yet, on anything but a reconstruction check.

The claim I intend to make is narrower and checkable: take a 2B parameter open model, post-train it on trajectories that passed the Verifier inside an environment Kullback built from traces, and it performs on the real held-out tasks, measured against the same model untrained and against the frontier model that produced the traces. tau2 airline and telecom first, because their real tools and databases exist to check the environment against; a customer domain after. The numbers get published either way, with the environment gates beside them, so a good training number can't hide a bad environment.

If you build evals or environments, the design choices are worth reading even before that number exists: the trace is the only source of truth and every derived value points back into it; the grader is code, and a judge can only narrow the bar, never award a pass; every stage has a held-out set and reports the seed number beside the held-out number; a run that needed a model stand-in for any tool is reported, never counted. `docs/design-philosophy.md` says what I built, why, and what I left out; the design document, the decision log with every choice and the alternative it beat, and 29 research reports are in `docs/` and `research/`. Disagree with the design if you want; the decision log exists so you can see what was already argued.

## 5. Future work

In rough order.

The end-to-end build with a live model: the `build` and `run` commands are wired to stages that exist, but the orchestration that needs a model (tool bodies, policy predicates, tool classification) is not written yet, and steps 5 to 8 of the slice (Verifier on the cancel-pending-order Task, candidate runs with a cheaper model, Verdict against tau2's reward, regrade after an environment fix) run only once it is.

A sandbox for model-written tool code. Today compiled tools run in a subprocess for the compile gates but in-process for the Runner. Nothing from a customer's traces should be compiled before that changes.

The re-run count k, decided by experiment rather than by me: the atom sets stop changing at some k on tau2 retail, and that's the default.

The post-training experiment above, once the live build holds: a 2B model, trajectories filtered by the Verifier, measured on real held-out tasks.

The overfitting check: the same code on tau2 airline and telecom with no retuning, then a domain that is not tau2 at all. Every constant that has to move is recorded in the decision log as a constant that was fit to retail.

More trace formats on ingest, each a reader that yields the same records with a pointer into the original file: Langfuse exports, OpenTelemetry GenAI in both dialects, OpenInference (Arize Phoenix), LangSmith runs, plain OpenAI and Anthropic message logs, Claude Code JSONL, MCP logs, and public benchmark formats (tau-bench v1, BFCL multi-turn, AgentBench, WebArena and SWE-bench trajectories, AppWorld, ToolSandbox). tau2 native is the only one wired today.

Synthetic data from the environment: new Tasks in the clusters the traces cover and the gaps beside them, runs of a strong model filtered by the Verifier, and a report of what was kept and why the rest was dropped. This is the training set for the experiment above.

An OpenEnv wrapper over the loop, so the Environments run anywhere that standard is accepted.

The parts of the design that are named and deferred: a filter that chooses which runs are worth re-executing when there are more than the budget; reference confirmation as its own module (the slice uses tau2's references); the dispute path for end states outside the required and allowed sets; a statistics module with paired non-inferiority intervals and pass^k; mining user behaviour alongside tool use, so the simulated user is calibrated on how often real users volunteer facts or walk away; anonymization at the customer boundary; hardening Tasks with traps and perturbations, only after the base Environment holds.

The Builder improving itself: it can already edit its own prompts and rules, commit the change as a node in a memory tree and have an evaluator outside the loop accept or reject it on the held-out runs, one change per round. What it learns on one customer travels to the next as a lessons file it is required to argue the relevance of before applying.

## Getting started

```
uv sync
uv run pytest
uv run harness ingest path/to/traces.json --workdir work
```

`CONTRIBUTING.md` says how to get a change in and which rules the code keeps. `DEVELOPING.md` covers the layout, the records, the offline test models and the fixtures. `docs/harness-design.md` is the spec; `docs/decision-log.md` is why.
