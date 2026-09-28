# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]
## [0.8.0] - 2026-09-23
### Merged PRs
- Feat(binary-pcs): multilinear commitment over binary tower fields (#2032)
- Fix(binary-pcs): query at pair granularity, fold without copying (#2034)
- Feat(multi-stark): add binary-field end-to-end example (#2047)
- Perf(binary-pcs): fold the codeword packed in the polynomial basis (#2053)
- Perf(binary-pcs): expose proving phases and speed ARM basis conversion (#2094)
- Perf(binary-pcs): add opt-in grouped Merkle leaves (#2095)
- Perf(binary-pcs): batch folding commitments over full cosets (#2097)
- Feat(sumcheck)!: drive the quadratic sumcheck transcript through the typed Fiat-Shamir layer (#2081)
- Fix(binary-pcs): reject a non-canonical grinding witness at zero difficulty (#2105)
- Fix(binary-pcs)!: enforce composed opening security budgets (#2114)
- Perf(sumcheck)!: fuse bind and measure across single-round calls and under suffix order (#2055)
- Review: couple tweaks here and there (#2120)
- Feat(sumcheck)!: drive the ring-switching transcript through the typed Fiat-Shamir layer (#2122)
- Feat(pcs)!: bind commitments and hiding claims through the typed layer (#2129)
- Feat(binary-pcs)!: drive the binary-tower PCS through the typed transcript (#2140)
- Feat(sumcheck): close the binary zerocheck against a real commitment (#2141)
- Perf(binary-pcs): change the fold's basis a block at a time (#2158)
- Perf(sumcheck,multi-stark,binary-field): faster binary PCS openings, zerocheck kernels and tower serialization (#2165)
- Perf(sumcheck,binary-pcs): run binary PCS sumcheck rounds in Ghash128 and bind long suffix rounds in place (#2174)
- Perf(sumcheck,binary-pcs): build the residual weights in the representation field (#2177)
- Feat(examples): merge the binary-hash examples and expose NTT/representation choice (#2185)
- Feat(binary-pcs)!: build the small-field commitment path (#2166)
- Feat(errors)!: make diagnostics actionable (#2202)
- Perf: batch Boolean trace openings and keep Blake3 traces packed (#2227)
- Perf: prove Keccak-f through the Boolean commitment with successor openings (#2228)
- Perf(binary): speed up the Boolean-committed prover (#2245)
- Feat!: add WHIR over binary additive domains (#2198)
- Perf(binary): reuse the reduction's evaluation, and make the sliced-round count a choice (#2246)
- Perf: run the bit ring-switch reduction in the polynomial basis (#2253)
- Feat(examples): let the binary harness choose its commitment hash and leaf geometry (#2250)
- Feat(binary-pcs): open Boolean and small-field traces through additive-domain WHIR (#2273)
- Feat(sumcheck,binary-pcs,lookup,multilinear-util)!: enforce caller soundness obligations instead of documenting them (#2293)
- Refactor!: give sumcheck, STIR, WHIR and binary PCS planning, transcript steps and configuration a single owner (#2297)
- Feat(sumcheck): authenticate jagged claims and ingest live traces (#2285)
- Perf(binary)!: speed up the sliced zerocheck, ring switch, PCS commit and opening, and BLAKE3 witness (#2307)
- Feat(sumcheck,binary-pcs,security)!: batch bit ring-switch claims at several points into one sumcheck (#2303)
- Refactor(security,sumcheck,multi-stark): keep one implementation of the candidate-set charge (#2291)

