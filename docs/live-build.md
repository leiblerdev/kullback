The live build narrative moved to docs/builds/README.md on 2026-09-10, where it now serves as the builds index. This pointer stays so older links keep working.

Latest build there: smoke 8, retail and airline on `openai/gpt-6-sol` at commit f913cd6, both finished on 2026-09-23.
- retail: 205 of 205 References confirmed, call fidelity 1.0000 over 3220 calls, 205 Verifiers, 133 trusted, 72 open,
  80 turns, 27.28 USD. The Builder stopped itself.
- airline: 95 of 119 References confirmed (79.8% over Tasks), call fidelity 0.9603 over 1513 calls, 95 Verifiers,
  53 trusted, 66 open, 69 turns, 13.23 USD. Stopped by the stop rule.
- Workdirs: `.work-retail` and `.work-airline` in the smoke 8 worktree. Launcher `.claude/scripts/smoke8.sh`, logs
  `.claude/logs/smoke-<corpus>-0923-sol8.log`.
- Published under the tag build-20260923: retail as a release (2026-09-23), https://huggingface.co/datasets/leibler/retail;
  airline as a preview (2026-09-24), https://huggingface.co/datasets/leibler/airline. Telecom is still round 5 from 2026-09-09.
- The trusted counts are under review, because 49 retail and 27 airline of them rest on a second-path seed read
  from a file that several Tasks wrote at once; they will be re-derived before anything is built on them.
- An investigation into the open Tasks, the spend and the Builder's failures is running.
