# Validation

What we're trying to learn, and how.

## The north-star question

**Is LLM cost optimization a recurring problem, or a one-time fix?**

- One-time → audit/consulting business.
- Recurring → subscription; the full vision (routing → eval → optimization → hosting) holds.

## Hypotheses (in order of cost to test)

- **H1 (technical):** traffic is heavy-tailed: a few task types dominate spend. Tested by the traffic-shape report (from traces, not interviews).
- **H2 (pain):** the target feels the problem (bill anxiety / "are we using the right model?"). Tested by whether anyone replies + the triggering moment.
- **H3 (trust):** they'll share traces / drop in the wrapper. Tested by the actual ask.
- **H4 (money):** they'll pay, and pay repeatedly. Tested last, via commitments.

## The 4 goals each call must answer

1. **Recurring vs one-time** (business model)
2. **Triggering moment** (pain exists)
3. **Trust** (they'll share traces)
4. **Pay repeatedly** (money)

## Map to interview questions

| Goal | Questions (interview-guide.md) |
| --- | --- |
| Recurring vs one-time | 5–8 |
| Triggering moment | 1–3 |
| Trust | reveal + commitment ask |
| Pay repeatedly | reveal + commitment ask |

## Files

- `hypotheses.md`: the ledger (what we believe, and the evidence)
- `loop.md`: the learning loop (cycle + stop criteria + who does what)
- `interview-guide.md`: the question script (Mom Test)
- `progress.md`: call log
