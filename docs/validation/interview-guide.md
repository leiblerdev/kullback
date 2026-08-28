# User Interview Guide: LLM Cost-Reduction Validation

Built from the Mom Test framework and our validation hypotheses (H1–H4).

## Goal of each conversation

Learn four things:

1. **Recurring vs one-time**: after they've cut costs once, does the problem come back? (decides SaaS vs consulting)
2. **The triggering moment**: what actually made them feel the LLM-spend pain (H2).
3. **How they currently choose models, and whether they've tried to fix it** (H2 → H3).
4. **Whether they'd act**: via commitment, never "would you buy?" (H3 → H4).

## Screen first (5 seconds)

Target: runs LLM calls in production, roughly $1k–$50k/month spend.
If they have no production LLM traffic, thank them and ask "who else should I talk to?", then move on.

**Homework first:** read their website, LinkedIn, GitHub, and public product before the call. Learn what they build and which models they already use. Never ask what's publicly answerable.

## Rules

1. Don't mention the idea until the reveal.
2. Talk about their life, not the idea.
3. Past actions only, never "would you…", "do you like…".
4. Talk less, listen more.
5. Dig into the motivation behind every answer.



## Opening (casual, ~1 min, shows you did your homework)

> "I saw you're building {X} and running {models} in production. I'm exploring how teams like yours handle LLM cost and model choice. Walk me through how you landed on {current model}."

**Alternative opener for technical founders (Valentin's suggestion):** open broad before narrowing — "What's your biggest problem right now? / How did you solve it? / Who else should I talk to?" Lets them name cost (or not) unprompted, so a "no" on cost is a cleaner signal than a leading question.

## The questions



### The triggering moment (the most important question)

1. "Tell me about the last time you actually looked at your LLM spend. What made you look?"
2. "When's the last time your LLM bill surprised you? What happened?"
3. "Have you ever dug into your traces to see where the money actually goes? What did you find?"



### Current solution & past attempts

1. "How do you keep an eye on spend today?" (spreadsheet / dashboard / nothing)
2. "Have you ever tried to cut LLM costs? What did you try? What happened?"
3. "Have you tried swapping a cheaper model in for some tasks? What stopped you?" ← surfaces the constraints



### Recurrence (one-time fix, or does it come back?)

1. "After the last time you changed models or cut costs, did the savings stick, or did it drift back? What made it drift?"
2. "How often do new model releases or changes in your traffic make you reconsider which model you use?"



### The cost & urgency

1. "Ballpark: what does a month of LLM calls cost you, and is it going up or down?"
2. "If you had to figure out where that spend goes today, how long would it take you?"
3. "Is this a top-3 problem for you right now, or a 'we'll get to it' thing?"



### Constraints (reduction-under-constraints)

1. "What's the scariest part about moving a task to a cheaper model?" (quality / latency / compliance)
2. "How would you know a cheaper model is 'good enough'? What would convince you?"



### Network & close

1. "Who else do you know dealing with this? Who should I talk to?"
2. "Is there anything else I should have asked?"

## Deflect bad data

- Compliment ("cool idea!") → "Thanks. What's the last time this actually bit you?"
- Future opinion ("I'd use that") → "What are you doing for it today?"
- Feature idea → "What problem does that solve? How are you coping without it now?"



## The reveal + commitment (only at the end)

> "Here's what I'm actually working on: you're probably overpaying 30–50% on LLM calls and can't see it. Send me a week of traces and in 48 hours I'll show you exactly where.
>
> Does this sound relevant to what you just described? What would it take for you to try it?"

Then ask for a concrete commitment (in order of strength):

- Send a week of traces / drop in the 5-line wrapper
- A follow-up meeting with a specific goal
- An intro to someone else

If they say no, ask why. That's the real signal.

## Record after each call

- The triggering moment, verbatim.
- Monthly spend ballpark.
- Have they tried to solve it? (yes/no + what)
- Recurrence: did past savings stick or drift back?
- Their biggest constraint fear.
- Commitment given: none / time / intro / data.



## Good vs bad signals

- Bad: "cool, let me know when it launches."
- Good: "what are the next steps", "can I see the prototype", an intro, or traces actually sent.

