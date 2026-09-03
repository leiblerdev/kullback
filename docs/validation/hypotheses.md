# Hypothesis Ledger

Living list, updated after every interview. Status: `untested` → `testing` → `confirmed` / `killed`.

## Thesis (sharpened 2026-08-24)

Token-heavy AI-native teams hit a wall: their LLM bill is real ($60k+/mo and growing) and it hits margins, but what stops them from optimizing is **not** cost — it's that they **can't evaluate** whether a cheaper model is "good enough" per task (they're "in a bad place in terms of evaluations"). Fine-tuning is the instinct and is *attempted* at scale, but it's a 1000× maintenance burden (retraining, drift detection, surface area).

So the wedge is **Shadow Evaluation**: replay a customer's own traces through candidate models and grade the outputs, giving a per-task answer to "is this cheaper model good enough?" — offline, as a plan, never in the live path. **Cost savings (30–50%) are the proof; the eval is the product.** No fine-tuning, nothing in the live path.

- **Buyer:** founding engineer / CTO (H7 confirmed).
- **Trigger:** bill hits margins — profitability push, credit-caps exhausted, $25k+/mo growing (H9).
- **Gate:** trust — EU residency + per-customer data + NDA (H10 confirmed).

## Active

### H2: Felt pain (triggering moment)
- **Claim:** founders feel bill anxiety / "are we even using the right model?": there's a real triggering moment.
- **Test:** do they reply + describe a *past* trigger? (interview Q1–3)
- **Status:** testing
- **Evidence:** Sitefire.ai CTO (Vincent Jeltsch, YC W26, 2–10 people): "API costs don't matter yet, we're too small." → pain is stage-gated; no pain below ~$1k/mo spend. (Not a kill: he's outside the target profile.) Clera COO (Alex Farr, $3M pre-seed, warm): "not a priority, the bills aren't huge yet." Manex founding engineer (Valentin Golz, €8M seed): "spending is not a concern" — big startup credits, no felt pain. Spherecast engineer (YC S24): "not looking at spends — we have credits." → **Pattern: the #1 pain-suppressor is now credits, not company size.** Seed/pre-seed teams on YC/cloud credits don't feel the bill regardless of scale. Manex's *actual* felt pain is elsewhere — self-hosting + EU data residency + "agents not doing what we want" (context), not token cost. **Zauber founding engineer (Sebastian Lettner, ~$60k/mo tokens, growing 10k→60k):** first felt-pain confirmation with specifics — explicitly confirmed "this is a big challenge," spend "heavily impacts margins," and they've started optimization efforts but "haven't found effective measures." Caveat: philosophy is "create value first, then optimize" → pain is arriving with scale, not yet a top-3 priority. **Comena (Jiehua Wu, 2026-08-28, email):** "we're using the latest frontier models and are fine with their cost" — confirmed high token spend (first-hand at the event) but zero pain. → Spend alone does not create pain; frontier satisfaction suppresses it even at real volume.

### H5: Recurring vs one-time (north-star)
- **Claim:** the problem recurs: model releases + traffic changes decay any routing plan, so savings drift back.
- **Test:** interview Q7–8 ("did the savings stick or drift back?")
- **Status:** testing
- **Evidence:** Manex (Valentin): they run Langfuse but "don't do evaluations — they're outdated so fast, new models come in." → first direct evidence that model releases decay any eval/plan faster than a team can maintain it. Routing/evals must be self-refreshing to stick. **Zauber (Sebastian):** "in a bad place in terms of evaluations" — can't quickly assess dataset quality / model quality per task. Stronger than Manex: evals aren't just decaying, they barely exist. → Evaluation is the bottleneck, not just cost.

### H3: Trust (they'll share traces)
- **Claim:** they'll hand over traces / drop in the wrapper.
- **Test:** the actual ask (commitment).
- **Status:** testing
- **Evidence:** gated by H10: regulated segments (healthcare, construction) won't share traces without an EU-residency answer. ADR-0002's "your data → your model" story may be necessary but not sufficient for them. **Zauber (Sebastian, 2026-08-24):** conditionally agreed to send "the NDA along with the traces" if things work out → first real commitment on the ladder (traces). BUT conditional (gated on the Nazib call landing), and the NDA is the trust mechanism — confirms H10 (they'll only share under NDA). Action: have an NDA- and EU-residency-ready data story before Nazib.

### H4: Money (recurring payment)
- **Claim:** they'll pay, and pay repeatedly.
- **Test:** commitments stronger than "sounds cool" (intro / traces / pre-order).
- **Status:** untested
- **Evidence:** -

### H1: Heavy-tail traffic (technical)
- **Claim:** a few task types dominate spend.
- **Test:** traffic-shape report from traces (not interviews).
- **Status:** testing
- **Evidence:** Zauber founding engineer (Sebastian Lettner): "a lot of repetitive tasks" for customers — first interview-level confirmation that workload is heavy-tailed, ahead of the traffic-shape report.

## Candidate (test as they emerge)

### H6: Default-to-frontier
- **Claim:** founders pick GPT-4o/Sonnet out of risk aversion, not benchmarking.
- **Test:** "how did you pick your model?" → "it was safe/default" confirms.
- **Status:** confirmed
- **Evidence:** Manex (Valentin): "right context + the model will make the right decisions" — betting on frontier + context engineering, skeptical of fine-tuning ("fine-tuned models… costs were like a couple hundred %"; "only consider if current models lack the knowledge with the right context"). Default-to-frontier is a *deliberate* bet, not just risk aversion.

### H7: Buyer is the founding engineer / CTO
- **Claim:** the person who feels the pain is the technical founder, not the CEO.
- **Test:** who replies and who shows the emotion.
- **Status:** confirmed
- **Evidence:** Both productive calls were with founding engineers (Manex's Valentin, Spherecast's engineer). Valentin explicitly advised: "target CTOs, founding engineers, direct contact — not CEOs." Two engaged technical contacts + explicit expert confirmation.

### H8: Burned once, distrust
- **Claim:** they already tried a cheaper model and got burned, creating distrust.
- **Test:** interview Q6 ("what stopped you?").
- **Status:** candidate

### H10: Data residency is a gate for regulated segments
- **Claim:** the companies that feel cost pain (healthcare, construction, finance, legal) also carry data-sovereignty requirements (EU residency, GDPR, ISO 27001). They will ask "where do my traces live?" and refuse unless we can answer EU-resident + certified. Trace capture is data sharing, so the trust bar is higher than ADR-0002's "never pooled".
- **Test:** note whether the founder raises data residency unprompted; ask "would your traces need to stay on EU servers?"
- **Status:** confirmed
- **Evidence:** Emidat markets "private, stored on European servers", GDPR-compliant, ISO 27001 (construction EPD under EU regs). Avi Medical is healthcare: special-category GDPR. **Manex (Valentin, unprompted):** enterprise manufacturing = "everything needs to be self-hosted… because Europe" — customer systems are air-gapped on-prem, "huge servers but for the whole company." Self-hosting + residency "increases complexity by a lot" and is their *real* pain, ahead of token cost. Residency isn't just a sales objection — it's the primary problem for enterprise EU buyers. **Zauber (Sebastian):** open-weight models are ALSO blocked by compliance — where the model is hosted + DPA; "there could be customers who can't." So "self-host open weights" is not a clean EU escape either: hosting location + DPA gates it the same way. → EU-resident, certified execution is required for ANY path (SaaS API, self-hosted open weights, or our offline re-run).

### H9: Spend threshold (pain is stage-gated)
- **Claim:** felt pain becomes a buying trigger only above roughly $25k/month LLM API spend. Below that, founders genuinely don't care ("too small").
- **Test:** correlate "felt pain" (H2) with reported monthly spend; "too small" replies should cluster well under $25k/month.
- **Status:** testing
- **Evidence:** Sitefire.ai CTO: "API costs don't matter yet, we're too small" (2–10 people, ~$500K pre-seed). Clera COO (Alex Farr, warm): "not a priority, the bills aren't huge yet." Netlight (consultancy): engineers "run out of credits every month" = hitting usage caps = proven spend + felt constraint. Benchmarks: first pain ~$100–500/mo at 10–100 users; seed AI budget $1.5–4k/mo; LLM infra 40–60% of revenue for LLM-native startups at scale.
- **Refinement (Vincent's note):** the companies that feel it are those where cost hits the P&L: **Series B+ steering to profitability** and **bootstrapped / PE-owned** companies. Vincent's suggested floor: >$25k/month LLM API spend, plus a decision-maker contact (founder/CTO/founding engineer). Adopted as hard GTM requirements in `gtm/docs/criteria.md`.
- **Refinement (Manex + Spherecast + Emdash):** pain is gated by spend **AND** by credits. Three more independent "no pain" replies citing startup credits (Manex $8M seed, Spherecast YC S24, Emdash YC W26). Rule: even real-scale spenders don't feel pain while on YC/cloud credits; screen "credits exhausted?" before investing time. Raban (Emdash): "this is more of an enterprise need" — independently confirming the enterprise/late-stage target. Spherecast: "small companies focus too much on selling, not optimizing costs." **Zauber (Sebastian):** $60k/mo tokens, grew 10k→60k → first lead ABOVE the floor that ALSO feels pain. Confirms the >$25k floor: at $60k/mo and growing, the bill "heavily impacts margins."
- **Refinement (Dataleap, Jan-Hendrik Rüttinger, 2026-08-27, WhatsApp):** at **$50k/mo** ($600k/yr) he still says "our spend is too small for it to make sense. I guess your token spend needs to be in the XX Mio per year for it to make sense. Maybe X Mio is sufficient already." → For a **post-train-and-host** offering the threshold is materially higher than the $25k/mo routing/report floor; roughly €1M+/yr (~$85k+/mo). Two tiers: reports/evals can sell lower; hosted custom models need ~€1M/yr accounts.

### H11: Own-model / fine-tuning is deferred or rejected near-term (frontier + context wins)
- **Claim:** AI-native startups default to frontier APIs + context engineering; training/fine-tuning their own models is a distant, capital- and talent-gated ambition, not a current alternative to routing to cheaper models.
- **Test:** "are you training/fine-tuning any of your own models today? what would make you?"
- **Status:** confirmed (early-stage) / refined (scale-gated)
- **Evidence:** Manex (Valentin): won't fine-tune — "only if current models don't have the knowledge with the right context"; fine-tuned models "scariest part… costs were like a couple hundred %". Spherecast (engineer): "we'd like to make our own models, but not a priority — future, needs engineering resources + talent." Emdash (Raban, YC W26): "everyone wants frontier abilities, so fine-tuning/smaller models aren't really of interest afaict." → Three independent confirmations: no one is fine-tuning now, and founders explicitly don't want smaller/fine-tuned models. **Do not pitch "we fine-tune a smaller model" — the market has now rejected it three times.**
- **Refinement (Zauber, $60k/mo, repetitive + lots of training data):** fine-tuning IS attempted at real scale — a colleague (Nazib) is actively fine-tuning on repetitive tasks. But it's found to be a huge burden: "effort to get there is very high, 1000x — retraining and drift detection and surface area." So fine-tuning is **scale-gated** (rejected below ~$25k/mo, attempted above) but **painful to maintain at any scale**, and gated by evals ("bad place in terms of evaluations"). → Positioning: routing + Shadow Eval is the cheaper alternative to fine-tuning for the "good enough" tasks. Don't sell fine-tuning; sell "you may not need to fine-tune these tasks at all."
- **Dataleap (Jan, 2026-08-27):** "we're using the latest frontier models" — and the user picks the model themselves (no smart routing). Comena (Jiehua, 2026-08-28): "using the latest frontier models and are fine with their cost." → Frontier-default now confirmed 5×, including at $50k+/mo spend.

### H13: Raw traces are not enough — product-outcome data is the real training signal
- **Claim:** post-training from captured traces alone won't produce a meaningfully better model; you need the customer's product-side outcome data (did the user accept, edit, abandon, retry?) to label what actually worked. The company running the product sees what the model provider never does.
- **Test:** ask founders "would you share outcome signals (accepted/reverted/edited) alongside traces?"; check whether trace-only post-training produces a model that clears the bar.
- **Status:** candidate
- **Evidence:** Dataleap (Jan, 2026-08-27, WhatsApp): "Imagine what data Cursor gets vs Anthropic when a user uses Opus through Cursor. Anthropic only gets the raw trace. Cursor however knows if the code was abandoned or merged, manually fixed, used and then thrown away, search operations etc. That's the data that is importantly for training." Also: "a friend tried the same (inference.net) — they were too early, but I think now the market for post training is big enough." → Traces capture the request; the product captures whether it worked. Without the latter, post-training optimizes toward guesswork. (Leibler's answer is Environments + End-state Verdicts — replayable ground truth instead of product telemetry — but Jan's point stands as a requirement to validate.)

### H14: Routing/gateway is a commodity — nobody pays for a router
- **Claim:** every team already has an LLM gateway in some form (open source); smart routing is feared ("a difficult task gets routed to a smaller model" frustrates users), so standalone routing or gateway products have no willingness to pay.
- **Test:** ask prospects what they'd pay for routing vs. for proven per-task quality; collect competitor outcomes (Pump.co et al.).
- **Status:** confirmed
- **Evidence:** Dataleap (Jan, 2026-08-27): "I guess almost everybody has a llm gateway already in their code in some form or shape. The gateway is a commodity and open source. We are not willing to pay for a gateway. The smart routing is tricky to get right and often frustrating for the user when a difficult task gets routed to a smaller model. So we don't do that. The user has to pick the model." Competitor data point: **Pump.co** pitched him the same (1: intelligent routing to smaller/open models, 2: group credit buying for volume discounts) — best offer at his $50k/mo spend was **6–10% discount**, "not that great and not worth it for us to route through them instead." → Routing-only positioning is dead on arrival; the value must be hosted models at better price-at-same-latency (his explicit ask) or proven per-task quality, never the router itself. (The glossary already says live routing is out of scope; this confirms it from the market.)

### H12: Evaluations are the bottleneck / entry point
- **Claim:** for token-heavy repetitive companies, the real blocker to optimization (fine-tune OR route-to-cheaper) is *evaluations* — they can't quickly assess per-task model quality, so they can't move anything.
- **Test:** "how do you know a cheaper model / a fine-tune is 'good enough' for a task today?" → "we don't / we eyeball it" confirms.
- **Status:** testing → **PROMOTED: this is the wedge (thesis, top).**
- **Evidence:** Zauber (Sebastian): "they are in a bad place in terms of evaluations" — dataset quality + speed of assessment are the gaps. Manex (Valentin): runs Langfuse but "don't do evaluations." → Two independent signals that evals are the missing piece, not cost. Shadow Evaluation is the product; cost is the proof.

## Update log

- 2026-08-19: ledger created with H1–H5 (active) and H6–H8 (candidate).
- 2026-08-19: Sitefire.ai signal logged → H2 to `testing`; added H9 (spend threshold).
- 2026-08-19: Vincent's note (profitability-driven / bootstrapped segments) → refined H9; added H10 (data residency gate).
- 2026-08-19: Alex Farr (Clera) "too small" signal logged; now 2 of 3 independent "too small" responses.
- 2026-08-19: Netlight feedback + Vincent's >$25k/decision-maker suggestion → adopted strict GTM criteria (`gtm/docs/criteria.md`); H9 threshold raised to >$25k/month.
- 2026-08-20: Emdash (Raban von Spiegel, YC W26) logged — "no pain, credits" + "more of an enterprise need" + "frontier-only, fine-tuning not of interest." H11 → confirmed (3rd signal). H9 refined (enterprise-need now 3×). Added insight ins_market_enterprise + Emdash lead (status no).
- 2026-08-24: Zauber (Sebastian Lettner, founding engineer) logged — first QUALIFIED felt-pain lead ($60k/mo, 10k→60k). H2 first confirmation w/ specifics; H1 first interview-level heavy-tail confirmation; H9 floor validated; H10 extended (open-weight models also DPA-gated); H11 refined (scale-gated + maintenance burden); added H12 (evals = bottleneck).
- 2026-08-24 (follow-up): Zauber — Sebastian conditionally agreed to send NDA + traces ("if everything works out"). H3 → testing (first trace commitment); trust mechanism = NDA (confirms H10).
- 2026-08-24 (sharpening): added Thesis block; H12 promoted candidate→testing (eval = wedge, cost = proof). Pitches + website sharpened to eval-first, no-fine-tuning framing.
- 2026-08-28: Dataleap (Jan Rüttinger, WhatsApp 2026-08-19→27) logged — Pump.co competitor data (routing + group credits, only 6–10% at $50k/mo, rejected), gateway = commodity (never sell the router), his requirement = hosted open-source models at better price, same latency; raw traces insufficient without product-outcome data (Cursor vs Anthropic example); spend threshold for post-train-and-host raised to ~€1M/yr. Added H13 (candidate) + H14 (confirmed). Jiehua Wu (Comena, email 2026-08-28) declined: "using the latest frontier models and are fine with their cost" → H2/H9/H11 evidence; lead status no, kept warm.
