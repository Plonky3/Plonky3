# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]
## [0.8.0] - 2026-09-23
### Merged PRs
- Perf(sumcheck): bind and measure in one pass (#2036)
- Perf(sumcheck): bind and measure in one pass in the hiding driver (#2038)
- Perf(binary-dft): incremental twiddles, tiled stages, and fused kernels (#2043)
- Feat(sumcheck)!: drive the generic-degree transcript through the typed Fiat-Shamir layer (#2024)
- Feat(multi-stark): add binary-field end-to-end example (#2047)
- Perf(field): route the self-algebra mixed dot product to the type's own dot product (#2050)
- Fix(security)!: enforce PCS budgets and compose multi-STARK soundness (#2100)
- Feat(sumcheck)!: drive the quadratic sumcheck transcript through the typed Fiat-Shamir layer (#2081)
- Feat(sumcheck)!: drive the hiding sumcheck transcript through the typed Fiat-Shamir layer (#2103)
- Perf(ci): speed up slow test-suite tests (#2118)
- Fix(transcript): address the review follow-ups from the typed-transcript stack (#2117)
- Perf(sumcheck)!: fuse bind and measure across single-round calls and under suffix order (#2055)
- Review: couple tweaks here and there (#2120)
- Feat(sumcheck)!: drive the ring-switching transcript through the typed Fiat-Shamir layer (#2122)
- Feat(pcs)!: bind commitments and hiding claims through the typed layer (#2129)
- Feat(sumcheck): add a univariate-skip round for binary zerochecks (#2134)
- Feat(sumcheck): reduce a skip round's opening to an evaluation point (#2135)
- Feat(sumcheck): generalise the skip constraint and stream the prover path (#2137)
- Feat(sumcheck): own the binary zerocheck from witness to opening point (#2139)
- Feat(sumcheck): close the binary zerocheck against a real commitment (#2141)
- Test(transcript): tie each unpriced grinding site to the config it is credited from (#2143)
- Perf(sumcheck): fold pinned zerocheck blocks without multiplying (#2159)
- Perf(sumcheck,multi-stark,binary-field): faster binary PCS openings, zerocheck kernels and tower serialization (#2165)
- Perf(sumcheck,binary-pcs): run binary PCS sumcheck rounds in Ghash128 and bind long suffix rounds in place (#2174)
- Perf(sumcheck,binary-pcs): build the residual weights in the representation field (#2177)
- Perf(monty-31): fuse x86 dot products of length 5 to 8 into one reduction (#2157)
- Perf(binary-dft): exploit subfield structure in the additive transform and the encoder (#2173)
- Perf(sumcheck)!: bind the stacked rows over one aggregated column (#2188)
- Feat(binary-pcs)!: build the small-field commitment path (#2166)
- Perf(sumcheck,multilinear-util): fold suffix tables in place without the parallel feature (#2195)
- Fix(sumcheck): hand off a prefix residual narrower than one packed element unpacked (#2193)
- Feat(sumcheck): fill suffix witnesses in place (#2213)
- Perf: batch Boolean trace openings and keep Blake3 traces packed (#2227)
- Perf: prove Keccak-f through the Boolean commitment with successor openings (#2228)
- Perf(binary): speed up the Boolean-committed prover (#2245)
- Feat!: add WHIR over binary additive domains (#2198)
- Fix(sumcheck): require a cryptographic RNG for HVZK mask sampling (#2260)
- Perf(binary): reuse the reduction's evaluation, and make the sliced-round count a choice (#2246)
- Perf: run the bit ring-switch reduction in the polynomial basis (#2253)
- Perf(sumcheck): factor the ring-switch equality weights into scaled blocks (#2251)
- Perf(sumcheck): commit the suffix layout straight from the source tables (#2268)
- Feat(sumcheck): add the basic jagged reduction (#2225)
- Feat!(multi-stark): authenticate binary bus claims (#2244)
- Feat(binary-pcs): open Boolean and small-field traces through additive-domain WHIR (#2273)
- Perf(binary): reduce early-round work and representation passes (#2286)
- Feat(sumcheck,binary-pcs,lookup,multilinear-util)!: enforce caller soundness obligations instead of documenting them (#2293)
- Refactor!: give sumcheck, STIR, WHIR and binary PCS planning, transcript steps and configuration a single owner (#2297)
- Feat(sumcheck): authenticate jagged claims and ingest live traces (#2285)
- Perf(binary)!: speed up the sliced zerocheck, ring switch, PCS commit and opening, and BLAKE3 witness (#2307)
- Perf(multilinear-util,bus,multi-stark,word-backend,sumcheck)!: borrowed points and one packed equality-table kernel (#2305)
- Feat(sumcheck,binary-pcs,security)!: batch bit ring-switch claims at several points into one sumcheck (#2303)
- Refactor(security,sumcheck,multi-stark): keep one implementation of the candidate-set charge (#2291)


### Added

- `PrescribedOpeningSecurity::candidates`, the checked view of `log2_max_candidates`.

### Changed

- **Breaking:** `BitRingSwitch::num_variables` and `SvoPoint::num_variables_svo` are no longer `const`.
- `PrescribedOpeningSecurity::charge_reduction` delegates to `SecurityTerm::over_candidates`
  and documents that the candidate count is forwarded to the layer above, not consumed. A
  count that names no set now leaves the term with no bound instead of raising it.

## [0.7.0] - 2026-09-04
### Merged PRs
- Chore: fix latest stable clippy (#1994)
- Feat(sumcheck): subtraction-free projective (monomial-basis) sum-check on the prover (eprint 2026/762) (#1900)
- Feat(field): characteristic-agnostic groundwork for binary fields (#2000)
- Feat: adapt log levels of some inner functions (#1999)
- Feat(binary-dft): additive NTT and the Encoder abstraction (#2003)
- Feat(sumcheck): ring switching over an arbitrary field extension (#2006)
- Fix(sumcheck): unpack suffix-bound product polynomials (#2008)

## [0.6.0] - 2026-06-11
### Merged PRs
- Refactor: move sumcheck to an independent crate (#1672)
- Perf(whir): use collect_n for select challenge powers (#1681)
- Perf: use shifted_powers + collect_n for combine (#1682)
- Perf: extend collect_n and shifted_powers to remaining call sites (#1683)
- Ci: tighten doc/release/TOML checks (#1689)
- Feat: add a `HornerIter` supertrait on `DoubleEndedIterator` (#1692)
- Fix: sample HZVK masks from EF (#1726)
- Fix: require ell_zk >= 3 in HVZK sumcheck (#1727)
- Feat(whir): HVZK sumcheck suffix-binding prover (#1665)
- Perf: reduce some allocations on WHIR (#1729)
- Sumcheck: type zk handoff and residual claim producer (#1732)
- Perf(whir): avoid extra allocation in prefix-order commit encoding (#1743)
- Whir: add HVZK-WHIR hiding polynomial commitment scheme (#1767)
- Refactor: remove unwraps in verifier path (#1788)

