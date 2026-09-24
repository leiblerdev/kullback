# Real-world trace candidates

Four tiers of "where can we get real-world trace data from," as the orchestrator laid them out in
`.claude/herdr/overhaul-0922/reports/ov-sources/context.md` on 2026-09-24. AppWorld is covered separately
in docs/trace-sources/appworld.md and skipped here. Facts below were checked against each source's own
canonical page on 2026-09-24 unless marked unverified.

## Tier 1: customer exports

Support and operations platforms hold a conversation plus the actions taken on it; no public dataset exists
for any of these, they need a design partner's redacted export.

- **Zendesk**: no public dataset. Public API at https://developer.zendesk.com/api-reference/, ticket
  export via the Tickets API. Licence: none, this is a live SaaS product's API, not a dataset.
- **Intercom**: no public dataset. Public API at https://developers.intercom.com/. Same as above.
- **Freshdesk**: no public dataset. Public API at https://developers.freshdesk.com/api/. Same as above.
- **Salesforce Service Cloud**: no public dataset. Public API (REST/SOAP) documented at
  https://developer.salesforce.com/. Same as above.

## Tier 2: public systems with real full change history and a live API

- **GH Archive**: https://www.gharchive.org/, every public GitHub event since 2011 (issues opened,
  labelled, assigned, PRs reviewed and merged) by real people, hourly JSON archives back to 2011-02-12.
  Licence: not stated on gharchive.org itself (no licence or terms text found on the homepage in this
  pass); the underlying data is GitHub's public event stream, governed by GitHub's own API Terms of
  Service, not a separate open licence asserted by the archive. Size: unverified in this pass (the archive
  grows continuously; no fixed total was checked).
- **Apache Jira public projects**: https://issues.apache.org/jira/, a live JIRA instance, not a static
  dataset: real tickets with workflow transitions, assignments, comments. Licence: unverified; Apache
  Software Foundation's own terms apply to the site, no separate dataset licence exists because this isn't
  a packaged dataset. Size: not applicable, live and growing.
- **Wikipedia edit history**: https://dumps.wikimedia.org/, real edits, reverts, page moves, talk pages,
  and the live MediaWiki API. Licence: GFDL confirmed on the dumps legal page
  (https://dumps.wikimedia.org/legal.html, "GNU Free Documentation License" text present); Wikipedia's
  current text licence is dual GFDL/CC BY-SA 4.0 per Wikipedia's own terms, though the CC BY-SA 4.0 text
  specifically was not found verbatim on the dumps legal page checked here, so that half is unverified in
  this pass. Size: the full English Wikipedia edit-history dump runs into hundreds of GB compressed;
  exact current figure not checked.
- **OpenStreetMap changesets**: https://wiki.openstreetmap.org/wiki/Changeset, real map edits with
  discussion, via the OSM API (https://wiki.openstreetmap.org/wiki/API). Licence: ODbL (Open Database
  License), confirmed on the OSM copyright page (https://wiki.openstreetmap.org/wiki/Copyright, "Open
  Database License" text present). Size: the full changesets planet dump
  (`planet.openstreetmap.org/planet/changesets-latest.osm.bz2`) did not return a Content-Length in a HEAD
  request in this pass; size unverified.

## Tier 3: human demonstrations on real apps

- **Mind2Web**: https://huggingface.co/datasets/osunlp/Mind2Web (project page
  https://osu-nlp-group.github.io/Mind2Web/). Licence: cc-by-4.0 (Hub API `cardData.license`, confirmed
  2026-09-24). A row is one action step within one task episode on a real website (over 2,000 open-ended
  tasks collected from over 100 real websites across 31 domains, per the dataset's own summary; exact
  action-level row count not re-counted here). What it captures: clicks and typing on real DOM elements,
  not API calls, so it needs lifting to tool level before it fits a tool-call trace shape.
- **Android in the Wild (AitW)**: https://github.com/google-research/google-research/tree/master/android_in_the_wild.
  Licence: repository is Apache-2.0 (GitHub API license endpoint, confirmed 2026-09-24). 715,142 episodes,
  5,689,993 examples, 30,378 unique prompts across five sub-datasets (GoogleApps, Install, WebShopping,
  General, Single), per the repo's own statistics table. Instructions come from "a variety of sources
  including humans, large language models, and technical documentation" (repo README), so not purely human
  written; actions are UI taps/swipes on real Android devices, not API calls, same lifting problem as
  Mind2Web.

## Tier 4: look real and are not

- **ABCD (Action-Based Conversations Dataset)**: https://github.com/asappresearch/abcd. Licence: MIT
  (GitHub API license endpoint). Crowdworkers role-playing customer-agent conversations with actions from a
  fixed action space, not real customer interactions.
- **MultiWOZ**: https://github.com/budzianowski/multiwoz. Licence: MIT (GitHub API license endpoint).
  Crowdworker-collected multi-domain task-oriented dialogues, not real bookings.
- **Schema-Guided Dialogue (SGD)**: https://github.com/google-research-datasets/dstc8-schema-guided-dialogue.
  Licence: CC BY-SA 4.0 (GitHub API license endpoint). Crowdworker role-play over a schema-defined API
  surface, not real service calls.
- **ToolBench**: two projects share this name; the one matching "real APIs, model-written tasks and calls"
  from context.md is OpenBMB's ToolBench, https://github.com/OpenBMB/ToolBench (ICLR'24 spotlight,
  "An open platform for training, serving, and evaluating large language model for tool learning").
  Licence: Apache-2.0 (GitHub API license endpoint). Real RapidAPI endpoints, but tasks and API-call
  sequences are LLM-generated (ChatGPT-synthesized instructions and solution paths), not human or
  real-user-driven.
- **AgentTrove**: canonical dataset is `open-thoughts/AgentTrove`,
  https://huggingface.co/datasets/open-thoughts/AgentTrove, released by the OpenThoughts-Agent team.
  Licence: apache-2.0 (Hub API `cardData.license`). 1,696,847 rows drawn from 219 source datasets spanning
  code repair, shell scripting, math, competitive programming and general computer-use tasks
  (`datasets-server/size`: 1,696,847 rows, 19,552,345,444 bytes on disk, about 19.5 GB; matches the earlier
  memory note of "1.7M-row terminus-2 terminal traces"). The dataset card's own framing: "the largest
  open-source collection of agentic interaction traces to date," aggregated from many existing sources,
  not a single coherent real-world corpus; several related repos under the same and adjacent orgs
  (`mlfoundations-dev`, `DCAgent`, `DCAgent2`) hold narrower slices of the same terminus-2 trace family.
- **tau-bench**: https://github.com/sierra-research/tau-bench. Licence: MIT (GitHub API license endpoint).
  Simulated users (an LLM playing the customer role) against simulated retail/airline domains with seeded
  databases; no real customers, no real APIs.
- **Twitter customer-support corpora**: canonical source is the Kaggle dataset "Customer Support on
  Twitter" by user/org Thought Vector, https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter.
  Licence: CC BY-NC-SA 4.0 (Kaggle's own dataset metadata, confirmed 2026-09-24). "Over 3 million tweets and
  replies from the biggest brands on Twitter," zip download 176,772,673 bytes (~177 MB). Real customers,
  real companies, but only the public tweet text, no actions taken, so it has no tool-call content at all.
  Several unlicensed Hugging Face mirrors exist (for example `gorkemsevinc/Customer_Support_on_Twitter`,
  no licence stated on the Hub); the Kaggle original is the canonical, licensed source.

## Summary

Real people plus real actions plus a live, queryable API (tier 2) is the strongest match for Kullback's
harness assumptions among everything surveyed here, at the cost of needing to recover the customer-side
instruction from issue text or discussion rather than having it stated directly (as context.md's own note
puts it, "the instruction must be recovered from issue text and discussion"). Tier 3 (Mind2Web, AitW) is
real humans and real apps but click/tap-level, not tool-call-level, so it needs a lifting step before it fits
a Trace at all. Tier 4 sources are useful as contrast cases (crowdworker role-play, LLM-generated tasks and
calls, or simulated users over simulated domains) but should not be mistaken for real-world data; AgentTrove
in particular is large and diverse but is itself an aggregation of many existing (mostly non-real-world)
trace sources, not a single real-world corpus.
