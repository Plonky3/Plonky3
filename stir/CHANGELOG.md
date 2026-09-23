# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]
## [0.8.0] - 2026-09-23
### Merged PRs
- Perf(stir): sub-coset virtual oracle, cheaper multiplies, barycentric Ans check (#2013)
- Ci: fail on unused dependencies, and drop the ones already there (#2029)
- Wire STIR into the full proving pipeline (#2037)
- Perf(stir): committed round-0 oracle with single-row input openings (#2041)
- Refactor(stir): share the commitment absorption between challenger backends (#2040)
- Fix(security,fri): accounting and grinding hygiene fixes (#2048)
- Perf(stir): reduce prover work and add opt-in smaller proofs (#2056)
- Fix(stir): reject the compact answer encoding in the default representation (#2065)
- Refactor(commit): separate univariate STARK capabilities from PCS (#2059)
- Test: share PCS opening contracts and batch fixtures (#2061)
- Feat(stir)!: drive the STIR transcript through the typed Fiat-Shamir layer (#2088)
- Fix(security)!: enforce PCS budgets and compose multi-STARK soundness (#2100)
- Fix(security): reject non-canonical grinding witnesses at zero difficulty (#2106)
- Feat(pcs)!: add batching grinding to STIR and Circle (#2112)
- Perf(ci): speed up slow test-suite tests (#2118)
- Fix(transcript): address the review follow-ups from the typed-transcript stack (#2117)
- Fix(stir)!: seed the batching phase at every difficulty (#2121)
- Review: couple tweaks here and there (#2120)
- Feat(sumcheck)!: drive the ring-switching transcript through the typed Fiat-Shamir layer (#2122)
- Test(transcript): tie each unpriced grinding site to the config it is credited from (#2143)
- Feat(errors)!: make diagnostics actionable (#2202)
- Fix(stir): use the rigorous field size on the standalone config path (#2259)
- Fix(uni-stark,circle): reject degree_bits below the PCS minimum trace height (#2257)
- Refactor!(uni-stark,batch-stark,commit): group the preprocessed openings and name the trace-height bounds as a pair (#2281)
- Perf(maybe-rayon): size parallel tasks from a cost model (#2039)
- Feat(security): use the DKT26 Johnson MCA bound (#2282)
- Refactor!: give sumcheck, STIR, WHIR and binary PCS planning, transcript steps and configuration a single owner (#2297)

## [0.7.0] - 2026-09-04
### Merged PRs
- Chore: fix latest stable clippy (#1994)
- Perf(stir): allow round 0's folding arity to differ from later rounds (#1989)
- Feat: adapt log levels of some inner functions (#1999)
- Perf(stir): commit height classes on a shared domain, merge via Combine (§7) (#1990)
- Perf(stir): batch the OOD-sampling and degree-correction hot loops (#1992)
- Fix(stir): fallible config derivation & dedup eta-parameterized security formulas (#1998)
- Perf(stir): bucket committed heights into several bounded-spread shared domains (#2005)
- Refactor(stir)!: replace the catch-all InvalidProofShape with typed variants (#2012)

