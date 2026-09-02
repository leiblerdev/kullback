# Architecture

Kullback is two agents, one runner and one set of gates on a layering inspired by huggingface/tau: a provider layer, an agent core that knows nothing about the application, and the applications as extensions on that core. This page is the map; the decisions are in `decision-log.md` and `adr/`, and each phase of the rebuild left a note in `tech/`.

## Packages

Each package imports only what sits below it. An import-linter contract in the pre-commit hook enforces the direction.

```
kullback/
  ai/         models behind one interface, the provider stream, pricing, usage
  agent/      the agent core: messages, tools, events, the stateless loop, the harness,
              the session tree, context management with a code floor
  builder/    the Environment agent
  examiner/   the Verifier and probe agent (phase 5)
  runner/     records, the frozen loop, route, the Verdict, judges, replay, regrade
  gates/      every accept-or-reject check and the registry that names them; no model call
  cli.py      the command line
  tui/        the terminal screen
  report.py   the customer-facing report
```

Builder and Examiner import agent, gates and runner. Gates import runner and never the agent core. Runner and agent import ai. Nothing imports the frontends.

## The agent core

The loop is a function: it takes a state, a model and a tool registry and returns when the model stops. Hooks run before each tool call (a raising hook blocks the call) and after each tool result (a hook may rewrite the result, which is where gate rulings are appended). The harness owns the transcript, the tool registry, two queues (steer, delivered after the current tool batch; follow-up, delivered when the run would stop) and the subscribers; one run at a time. Every message is recorded in a session tree, and the model manages its own context (forget, recall, load, unload) with a code floor at 40% of the window that compacts when the model has not.

## The Builder

The Builder turns traces into an Environment: the rows the runs touched, one function per tool, the policy compiled into predicates, a simulated user, and the References that confirm a trace replays. Its stages are tools over a declared graph: a stage starts when everything it reads is complete, independent stages run side by side, and every artifact is content-addressed so a resumed build redoes only what changed. A code driver walks the graph by default; with `--agent` the model drives the session.

## The Examiner

The Examiner derives each Task's Verifier from the Intent and the frontier's re-rolls, writes probes against it, repairs it when a gate rejects it, refuses a Task no frontier Run finishes, and sends findings about the Environment to the Builder as follow-ups. It never reads a tool body and never writes the Environment (ADR-0007).

## The gates

A gate is a pure function over an artifact that returns a ruling. The registry names every gate and what it rules over. Gates run inside the tool-result hook, so a model sees the ruling with the artifact and cannot skip it. A second hook refuses any tool call that would write under `gates/` or `runner/`. The gates hash and the runner hash are stored beside every Verdict.

## The Runner

The Runner takes an Environment, a recorded Run and a candidate model and advances one turn at a time: tool calls go to code first, then to an exact recording, then to a model stand-in, and the route taken is on the event. The Verdict is a code-only pass over what changed. A Run served by a stand-in is reported and never counted. The Runner is frozen once a person confirms it (`kullback freeze-runner`), and stored Runs can be re-scored without re-running the model.

## The round

A round is four beats on one event stream: the Builder builds and the Environment gates rule; the Examiner derives and the Verifier gates rule; the Examiner repairs, probes, refuses or sends a finding; the Builder repairs. One agent has the model's attention at a time. The round ends when both queues are empty and no stage runs, with counts that come from gates and never from a model. The loop stops when done (every Task has a trusted Verifier and clears fidelity or is refused with a reason, and no probe passes), stalled (a round moved no count), or at the ceiling.

## On disk

A workdir holds the raw files unchanged and hashed, the derived traces, the schema and database, the tool bodies and the sandbox outputs, the policy and its predicates, the Intents, the References and replays, the re-rolls and every Run as JSONL, the Verifiers and probes, `gates.json` with every ruling, `budget.json` with every call's cost, the pipeline state with the graph, the memo cache of model replies, and `runner_version.json`. Everything derived is content-addressed and points back to the bytes it came from.
