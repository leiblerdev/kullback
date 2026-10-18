# Glossary

The words Kullback uses, defined once. Other pages link here instead of redefining them.

## Plain words

- Environment: a simulated world with tools and rules that an agent acts in.
- Task: one goal the agent is asked to reach in an Environment.
- Run: one attempt by an agent at a Task, recorded step by step.
- Verifier: the check that decides from the end state whether a Run succeeded.
- Release and preview: a release is a published Environment that replays at least 90 percent of its Tasks; a
  preview is one we have not yet checked as thoroughly.

## Contributor words

- Trace: a recorded Run, as the customer exported it.
- Reference: a recorded Run of a Task chosen to replay it; it measures fidelity and never feeds a Verifier.
- Replay fidelity: the share of the recording the rebuilt world reproduces, over Tasks, Runs or calls.
- Atom: one Verifier check: required, allowed, forbidden, question, communicate or hard.
- Gate: an accept-or-reject check in code (`kullback/gates/`); inside a Verifier, an item that must hold.
- Trusted: a Task whose Verifier fails the constructed Runs (doing nothing, one stray write) with no open ruling.
- Open: a Task neither trusted nor refused.
- Refused: a Task the harness showed nobody can finish.
- Builder: the agent that turns Traces into an Environment; it never writes a Verifier.
- Spec: the role that mines a Task's Intent (what the user asked for) and writes its Verifier from it.
- Examiner: the agent that reviews each Spec and files rulings; it writes no Verifier.
- Runner: the code, not an agent, that executes a Run and decides its Verdict (pass or fail).
- User: the Simulated user a Run meets, rule-driven from the recording or model-driven over those rules.
