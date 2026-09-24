The live build narrative moved to docs/builds/README.md on 2026-09-10, where it now serves as the builds index. This pointer stays so older links keep working.

Latest build there: smoke 9, retail and airline on Opus 5.5 on Amazon Bedrock, both finished on 2026-09-24.
- retail: 223 of 223 References confirmed, 456 of 456 Runs, 222 Verifiers, 193 trusted, 0 refused, 71.01 USD.
- airline: 130 of 130 References confirmed, 199 of 200 Runs, 128 Verifiers, 82 trusted, 0 refused, 74.67 USD.
- Published on 2026-09-24: both Environments are on the Hub as releases under the tag build-20260924. The first
  export's leak scan refused 13 Verifiers whose atoms quoted a whole recorded agent message; D292 fixed that.
