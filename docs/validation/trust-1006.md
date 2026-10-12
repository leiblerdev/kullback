# Trust by code on the expected end states: the wrong-Reference rate (D319)

Pre-registered on 2026-10-06, before any number below was computed.

## What is measured

For every Task of a workdir that has a benchmark sidecar (the grader folder set aside beside the
build, read only by this measurement, never by the build), the Reference is classed by the reward
of its kept recordings: right (every reward 1), wrong (every reward 0), mixed, or unknown (no
reward found). Unknown Tasks are left out of the rate and counted.

The rate: of the trusted Tasks whose Reference is right, wrong or mixed, the share whose Reference
is wrong. It is computed twice on the same files:

- old gate: the pre-D319 rule (`_legacy_trusted`: suite row, probe pool, history, loosening,
  held-out false rejection, refusals, pool and seeds);
- new gate: `trusted_gate` with every Verifier carrying the end states `expected_from_run` derives
  from its faithful replay (user turns and the policy text of that replay as sources, the forbidden
  list from user negations and recorded refusals, alternatives from the user turns, no conduct
  derived). A Verifier with no faithful replay gets one empty end state, so the code ruling, not the
  legacy rule, decides it (it is pending: no faithful replay).

Each rate carries a Wilson 95 percent interval, and beside it the trusted, unconfirmed, refused and
pending counts of each gate.

## The rule, written before computing

The new rate must be under 10 percent. If it is not, the sourcing gate is not doing the work and the
design is wrong, not the corpus. Whatever comes out is reported here unchanged.

Workdirs: the smoke9 retail copy; the airline workdir if one sits beside it.

## What it cannot show

One corpus, one build, 222 Tasks, all with a sidecar class; no airline workdir sits beside the smoke9
retail copy, so airline is not measured. The end states are derived by code from each Task's own
faithful replay, not written by the Spec, so this measures the gates on derived end states; a Spec that
writes them from the Intent could do better or worse. No conduct rule was derived, so the empty-Run
gate only bites on the end state. The legacy gate is run on the stored Verifiers and the new gate on the
same Verifiers with end states added; both read the same suite rows.

## Outcome (smoke9 retail copy, no model call)

| gate | trusted | unconfirmed | refused | pending | trusted, wrong Reference | rate | Wilson 95 percent |
|---|---|---|---|---|---|---|---|
| old (pre-D319) | 193 | 0 | 0 | 29 | 63 of 193 | 32.6 percent | 26.4 to 39.5 percent |
| new (D319) | 191 | 28 | 0 | 3 | 64 of 191 | 33.5 percent | 27.2 to 40.5 percent |

Reference classes: right 146, wrong 71, mixed 5, unknown 0. Valid other solutions failing (D133,
unchanged): 16 of 54 held-out Runs.

The rule fails: 33.5 percent is not under 10 percent. By the rule written above, the sourcing gate is
not doing the work and the design is wrong, not the corpus.

Where the new gate withheld trust: 28 unconfirmed by an unsupported cell (columns `users.payment_methods`
17, `actions.tool` 6, `orders.address` 3, `orders.items` 1, `orders.exchange_items` 1), 2 pending because
the wrong Run passes, 1 pending because the empty Run passes. By Reference class the unconfirmed Tasks
are 23 right and 5 wrong: the gate withholds trust from right References more often than from wrong ones.
Of the old gate's 193 trusted, 107 right and 61 wrong stay trusted, 19 right and 2 wrong drop to
unconfirmed, and 4 mixed stay trusted.

Why, read off the counts: a wrong Reference is a recorded agent that did a plausible wrong thing, and
the values it wrote were in the user's words or in a lookup it made (another item, another payment
method, an address on file). Literal sourcing finds them there and calls them supported. A source says
the value was available, not that the user asked for it, so it cannot separate a right choice from a
wrong one among values the conversation showed.

## Round 2: consistency with the request (pre-registered 2026-10-06, before computing)

Two checks read the other way from sourcing, both by code, in `kullback/runner/expected.py`:

- contradiction (`contradicts_user`): for each changed scalar value or changed nested leaf, the
  column's domain is every value that column (and leaf path) holds anywhere in the start state. If the
  user's own turns name one or more values of that domain and the cell's value is not among them, the
  cell contradicts the user. A column the user never named a value of is not contradicted.
- unasked write (`unasked_writes`): every successful write call in the Reference must follow a user
  turn naming one of its argument values (a row id or a scalar argument, canonical match) or a user
  yes (D306's reading). A write with neither is unasked.

Either one leaves the Task unconfirmed, naming the cell or the tool. Both run in `code_ruling` after
the sourcing check and before the replay match.

The rule: on the same 222 Tasks, trusted-on-wrong-Reference under 10 percent with both checks. Reported
beside it: trusted, unconfirmed, pending, and the right References newly withheld, with the wrong cases
split by which check caught them and which neither did. If the rule fails, that is said first and no
third check follows.

### Round 2 outcome (smoke9 retail copy, no model call)

The rule fails: 55 of 132 trusted rest on a wrong Reference (41.7 percent, Wilson 95 percent 33.6 to
50.2), worse than round 1. By the rule above, no third check follows.

| gate | trusted | unconfirmed | refused | pending | trusted, wrong Reference | rate | Wilson 95 percent |
|---|---|---|---|---|---|---|---|
| old (pre-D319) | 193 | 0 | 0 | 29 | 63 of 193 | 32.6 percent | 26.4 to 39.5 percent |
| round 1 (sourcing) | 191 | 28 | 0 | 3 | 64 of 191 | 33.5 percent | 27.2 to 40.5 percent |
| round 2 (sourcing, contradiction, unasked write) | 132 | 88 | 0 | 2 | 55 of 132 | 41.7 percent | 33.6 to 50.2 percent |

The 64 wrong References round 1 trusted: contradiction caught 7, unasked write 2, both 0, neither 55.
The 126 right References round 1 trusted: contradiction withheld 42, unasked write 5, both 1, neither 74;
48 right References newly withheld against 9 wrong ones. Mixed: 2 of 5 withheld by contradiction.

Withheld by column (right, wrong): `orders.status` 21, 7; `users.address` leaves 9, 0; `orders.address`
leaves 3, 0; `orders.items` leaves (price, options) 8, 0; `orders.payment_history` leaves 2, 0. Unasked
writes by tool (right, wrong): the hand-off tool 4, 1; the return tool 1, 1; the exchange tool 0, 1.

Why, read off the columns: the user names a value of a column to describe where the row stands ("my
delivered order", "I live in" a state), not what it should become, so a status or address the right
Reference wrote reads as contradicting the value the user used as a description. A domain match cannot
tell a description from a request. The unasked-write check is near silent: the user almost always names
the order or says yes before a write, right or wrong. 55 wrong References pass all three checks: they
wrote values the user named or never constrained, on rows the user named, after a yes.
