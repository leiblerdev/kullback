# Mutation testing, first full run (2026-08-28)

I ran mutmut 3.7 over every module before the review fixes landed, so these numbers describe the code as it was on the morning of 2026-08-28, not the code in this commit. 24,564 mutants: 17,372 killed, 7,031 survived, 81 timed out, 77 were never tested (mutmut could not reach them), 3 suspicious. Kill rate 70.7 percent.

Survivors by module, most first. A survivor is a mutation no test noticed; it is a hint about where the tests check that code runs rather than what it produces, not a bug count.

| Module | Survived | Timed out |
|---|---|---|
| builder/compile_env | 722 | 2 |
| shared/report | 706 | 0 |
| builder/build | 610 | 10 |
| runner/validate | 595 | 18 |
| shared/provider | 512 | 8 |
| builder/verifier | 486 | 1 |
| builder/mine | 426 | 1 |
| builder/memory | 380 | 1 |
| builder/policy | 292 | 4 |
| builder/ingest | 280 | 0 |
| runner/pipeline | 267 | 2 |
| runner/verdict | 265 | 13 |
| runner/judge | 265 | 0 |
| shared/canon | 220 | 9 |
| builder/user_sim | 205 | 0 |
| cli | 168 | 0 |
| shared/budget | 132 | 7 |
| runner/route | 130 | 0 |
| runner/loop | 128 | 5 |
| runner/regrade | 86 | 0 |
| builder/intent | 78 | 0 |
| builder/cluster | 43 | 0 |
| shared/records | 35 | 0 |

What I read from it. report.py's survivors are mostly wording: a mutated sentence still passes a test that checks a section exists. That is fine for prose and I will not chase it. compile_env, build and validate are the ones that matter: a survivor there can be a gate that stopped checking something. The policy survivors cluster in the sentence-splitting helpers (research/36 saw the same 292 and named the four functions). Next pass: rerun on the fixed code, then go through compile_env, validate and verifier survivors by hand, and write the test that kills each one that guards a gate. The raw per-mutant list is kept outside the repository.
