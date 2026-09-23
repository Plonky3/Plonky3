# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]
## [0.8.0] - 2026-09-23
### Merged PRs
- Ci: fail on unused dependencies, and drop the ones already there (#2029)
- Fix(security,fri): accounting and grinding hygiene fixes (#2048)
- Feat(security): integrate legacy conjectured regime (#2064)
- Fix(security)!: enforce PCS budgets and compose multi-STARK soundness (#2100)
- Feat(security): check recorded grinding difficulties against the security model (#2109)
- Fix(binary-pcs)!: enforce composed opening security budgets (#2114)
- Fix(stir)!: seed the batching phase at every difficulty (#2121)
- Review: couple tweaks here and there (#2120)
- Fix: address the minor follow-ups from the #2120 review (#2127)
- Fix(security): charge the budget's DEEP batching round over the LDE domain (#2131)
- Feat(pcs)!: bind commitments and hiding claims through the typed layer (#2129)
- Feat(binary-pcs)!: drive the binary-tower PCS through the typed transcript (#2140)
- Feat(security): charge a univariate-skip round in the multilinear budget (#2136)
- Feat(security,multi-stark)!: charge and make room for the indexed reduction (#2149)
- Feat(binary-pcs)!: build the small-field commitment path (#2166)
- Feat(errors)!: make diagnostics actionable (#2202)
- Feat(security): account for binary bus soundness (#2214)
- Perf: batch Boolean trace openings and keep Blake3 traces packed (#2227)
- Perf: prove Keccak-f through the Boolean commitment with successor openings (#2228)
- Feat!: add WHIR over binary additive domains (#2198)
- Feat(word): reduce shifted operands to one Boolean opening (#2226)
- Feat(security): use the DKT26 Johnson MCA bound (#2282)
- Feat(word): prove the zero and bitwise relations through the Boolean PCS (#2275)
- Feat(sumcheck,binary-pcs,security)!: batch bit ring-switch claims at several points into one sumcheck (#2303)
- Feat(word-backend,security)!: prove full-width unsigned multiplication through an exponent lift (#2304)
- Refactor(security,sumcheck,multi-stark): keep one implementation of the candidate-set charge (#2291)


### Added

- `CandidateSet`, the checked candidate count a commitment leaves open. A charge forwards
  the set rather than consuming it, and a count that names no set is refused rather than
  subtracted — a negative one used to add bits to the term it was charged over.

### Changed

- **Breaking:** `SecurityTerm::over_candidates` takes a `CandidateSet` instead of an `f64`.
  It is now the only implementation of the candidate-set charge in the workspace.

## [0.7.0] - 2026-09-04
### Merged PRs
- Fix(stir): fallible config derivation & dedup eta-parameterized security formulas (#1998)
- Fix(security): account for out-of-domain point count in budget's OOD round (#2007)
- Feat(security): add legacy conjectured FRI soundness bound (#2018)

