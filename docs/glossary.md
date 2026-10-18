# Glossary

The words Kullback uses, defined once. Other pages link here instead of redefining them.

## Plain words

- Environment: a simulated world with tools and rules that an agent acts in.
- Task: one goal the agent is asked to reach in an Environment.
- Run: one attempt by an agent at a Task, recorded step by step.
- Verifier: the check that decides from the end state whether a Run succeeded.

## Contributor words

- Trace: a recorded Run, as the customer exported it.
- Reference: a recorded Run of a Task chosen to replay it; it measures fidelity and never feeds a Verifier.
- Replay fidelity: the share of the recording the rebuilt world reproduces, over Tasks, Runs or calls.
- Atom: one Verifier check: required, allowed, forbidden, question, communicate or hard.
- Gate: an accept-or-reject check in code (`kullback/gates/`); inside a Verifier, an item that must hold.
- Trusted: a Task with an Intent and a Verifier, examined by an Examiner, with no ruling open, every ask of the
  user checked, not refused or synthetic, and a Verifier that fails both constructed Runs (doing nothing, and one stray write
  outside its declared rows, with every untouched mutant killed).
- Open: a Task neither trusted nor refused.
- Refused: a Task the harness showed nobody can finish.
- Builder: the agent that turns Traces into an Environment; it never writes a Verifier.
- Spec: the role that mines a Task's Intent (what the user asked for) and writes its Verifier from it.
- Examiner: the agent that reviews each Spec and files rulings; it writes no Verifier.
- Runner: the code, not an agent, that executes a Run and decides its Verdict (pass or fail).
- User: the Simulated user a Run meets, rule-driven from the recording or model-driven over those rules.
