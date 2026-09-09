# A no-argument tool whose results never change is a constant of the world (pattern 15)

Failure pattern 15: a catalog listing on the first corpus answered only the categories the world's
rows hold, so 19 of 19 recorded calls differed and 15 of the recorded categories were on the
recording's side only. The listing takes no arguments, so nothing about the call explains the
answer; the answer is a fact about the world, and the world did not hold it.

## The rule

`mine.world_constants`, run at the end of `mine_schema` over the assistant's own calls:

1. the tool was called at least twice (one call is no evidence that a result never changes),
2. every call carried the same arguments, canonically compared (so a no-argument tool qualifies,
   and so does one whose arguments never vary),
3. no call errored and none answered nothing,
4. every call answered the same result, canonically compared,
5. the result is not rows of a table, asked of the miner's own homing rule
   (`mine.is_rows_of_a_table`, the rule `extract_rows` uses): if any row of the result homes into
   an entity, the result is rows and the world already holds it,
6. the tool is not a write tool.

`mine.apply_constants` then names the schema's own `constants` table, one column per surviving
tool, each column `hard`, `classified_by` `observed`, its `evidence` carrying `constant_of` (the
tool) and `constant_value` (the value). `compile_env.build_starting_state` puts one row into that
table, `_schema_block` tells the body writer to read the row with
`next(iter(self.db.<table>.values()))` and never by key, and the mine stage writes
`world_constants.json` so a reader of the build can see what was mined.

### Why one row of columns rather than one row per value

A body that reached a per-value row would have to name a row key, and a literal id in a body is
refused by the memorised-values gate (D162). One row whose columns are the tools means the body
writes `next(iter(self.db.<table>.values()))["<its own tool name>"]`, which names nothing the
corpus taught it. It is the same shape the readers' one-row-per-requestor table already uses.

## Measured

One round on a copy of the first corpus's workdir
(`--target replay_reference --max-rounds 1 --workers 8 --model openai/gpt-5.6-luna`).

| | before | after |
|---|---|---|
| constants mined | 0 | 1 |
| entries in the constant | n/a | 50 |
| rows of the table the listing was assembled from | 35 | 35 |
| the listing tool's recorded calls matched | 0 of 19 | 19 of 19 |
| Tasks confirmed at replay fidelity | 179 of 205 | 165 of 205 |
| recorded calls matched, all tools | 3,134 of 3,220 (97.3%) | 3,109 of 3,220 (96.6%) |
| bodies kept assisted | 3 of 16 | 2 of 16 |
| wall time, this round | not recorded | 10m 04s |
| spend, this round over the copy | not recorded | $0.393 |

The 50 against 35 is the pattern stated as a number: the constant holds fifteen more entries than
the world holds rows, which is exactly the fifteen categories that were on the recorded side only.
With the constant in the Starting state the listing tool matches every recorded call.

### Where the corpus disagrees with the change

The change is worth 19 calls and costs 100 elsewhere, all of it in one write tool whose recorded
calls fell from 83 of 139 (59.7%) to 39 of 139 (28.1%). That tool's body was rewritten in this
round; nothing about the constants table is in its inputs, and section 8 and section 9 of the
readers report measured the same body-writing variance on the other corpus in both directions.
The round exited `max_rounds`, no stage failed. Of the $0.393 this round spent, compile_policy took
$0.162 over 116 calls, a loophole probe $0.138 over 399, compile_tools $0.082 over 86 and the
reference judge $0.011 over 17.

So the honest reading is that one round of body writing on this corpus moves single tools by tens
of calls either way, the constants rule moves one tool from 0% to 100%, and the Task count moved
inside that noise rather than because of the rule.

The other corpora on disk were checked with the same rule. The second mines one, a no-argument
listing whose result is a list of 20 values that home into no table and whose 3 recorded calls
matched 0 of 3 before. The third mines none. So the rule fires on both corpora where the pattern
was found and stays silent on the one where it was not.

## Reproduce

    cp -R .work-b14 .work-b14-const
    uv run kullback build -w .work-b14-const --target replay_reference \
        --model openai/gpt-5.6-luna --workers 8 --ceiling-usd 25.76 --max-rounds 1

The ceiling is the copied workdir's own spend plus ten: the budget is cumulative over a workdir and
that copy had already spent $15.76, so `--ceiling-usd 10` refuses before the first stage.

Numbers are read from `replays.json` (per call verdicts and Task confirmation), `schema.json` and
`db.json` (the constants table and its row), `world_constants.json` (what the mine stage recorded),
`tool_fidelity.json` and `pipeline/state.json` (assisted bodies) and `budget.json` (spend).
