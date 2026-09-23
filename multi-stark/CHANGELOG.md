# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]
## [0.8.0] - 2026-09-23
### Merged PRs
- Perf(multi-stark): precomputed alpha powers, coefficient-wise windows, packed accumulators (#2014)
- Fix(whir): draw independent STIR query samples (#2031)
- Feat(sumcheck)!: drive the generic-degree transcript through the typed Fiat-Shamir layer (#2024)
- Feat(multi-stark): add multilinear lookups (#1968)
- Feat(multi-stark): add binary-field end-to-end example (#2047)
- Refactor: rename fixtures according to versioning (#2058)
- Perf(binary-pcs): expose proving phases and speed ARM basis conversion (#2094)
- Feat(whir)!: seed the WHIR transcript through the shared Fiat-Shamir layer (#2089)
- Feat(multi-stark)!: drive the multi-STARK transcript through the typed Fiat-Shamir layer (#2093)
- Perf(binary-pcs): add opt-in grouped Merkle leaves (#2095)
- Perf(binary-pcs): batch folding commitments over full cosets (#2097)
- Fix(security)!: enforce PCS budgets and compose multi-STARK soundness (#2100)
- Feat(sumcheck)!: drive the quadratic sumcheck transcript through the typed Fiat-Shamir layer (#2081)
- Feat(multi-stark)!: type the statement-level transcript (#2102)
- Feat(whir)!: drive the plain WHIR transcript through the pattern player (#2104)
- Fix(binary-pcs)!: enforce composed opening security budgets (#2114)
- Perf(ci): speed up slow test-suite tests (#2118)
- Fix(transcript): address the review follow-ups from the typed-transcript stack (#2117)
- Review: couple tweaks here and there (#2120)
- Fix: address the minor follow-ups from the #2120 review (#2127)
- Feat(sumcheck)!: drive the ring-switching transcript through the typed Fiat-Shamir layer (#2122)
- Feat(air,multi-stark)!: public inputs bound by trace position (#1947)
- Feat(pcs)!: bind commitments and hiding claims through the typed layer (#2129)
- Fix(multi-stark): draw fractional-GKR round polynomials at the field's own nodes (#2132)
- Feat(binary-pcs)!: drive the binary-tower PCS through the typed transcript (#2140)
- Feat(security): charge a univariate-skip round in the multilinear budget (#2136)
- Feat(multi-stark)!: add the logUp* indexed-lookup reduction (#2145)
- Fix(multi-stark): build opening points and batches from one description (#2144)
- Feat(lookup,multi-stark)!: declare and plan indexed lookups (#2146)
- Feat(security,multi-stark)!: charge and make room for the indexed reduction (#2149)
- Feat(multi-stark): run the indexed reduction inside the proof (#2150)
- Perf(sumcheck,multi-stark,binary-field): faster binary PCS openings, zerocheck kernels and tower serialization (#2165)
- Fix(multi-stark): keep the residual-row AIR evaluation out of Rayon's recursive split (#2183)
- Feat(binary-pcs)!: build the small-field commitment path (#2166)
- Perf(multi-stark)!: run the binary zerocheck's first rounds sixty-four rows at a time (#2203)
- Perf(multi-stark)!: cache AIR profiles at setup (#2215)
- Perf: batch Boolean trace openings and keep Blake3 traces packed (#2227)
- Perf: prove Keccak-f through the Boolean commitment with successor openings (#2228)
- Perf(binary): speed up the Boolean-committed prover (#2245)
- Feat!: add WHIR over binary additive domains (#2198)
- Perf(binary): reuse the reduction's evaluation, and make the sliced-round count a choice (#2246)
- Perf(multi-stark): read the boundary zerocheck round and fold off the planes (#2249)
- Perf(multi-stark): read a later zerocheck round's lane group in one load (#2255)
- Perf(multi-stark): share one zeroed successor buffer across a stage's workers (#2266)
- Perf(multi-stark): weight a batch of AIR constraints with one dot product (#2267)
- Test(multi-stark): cover a proof and an AIR set that disagree on an optional section (#2278)
- Fix(multi-stark): state the lane-group bound on the sparse packed fold (#2279)
- Feat!(multi-stark): authenticate binary bus claims (#2244)
- Perf(binary): reduce early-round work and representation passes (#2286)
- Perf: defer eligible AIR residual materialization by one round (#2290)
- Feat(multi-stark): publish the machine-facing backend contract and proof envelope (#2272)
- Feat(bus)!: type the channel name and make boundary flushes first class (#2294)
- Fix(multi-stark): encode the end a boundary flush names (#2298)
- Feat(sumcheck,binary-pcs,lookup,multilinear-util)!: enforce caller soundness obligations instead of documenting them (#2293)
- Perf(binary)!: speed up the sliced zerocheck, ring switch, PCS commit and opening, and BLAKE3 witness (#2307)
- Perf(multilinear-util,bus,multi-stark,word-backend,sumcheck)!: borrowed points and one packed equality-table kernel (#2305)
- Feat(multi-stark)!: prove binary-bus shares inside the zerocheck sumcheck (#2302)
- Refactor(security,sumcheck,multi-stark): keep one implementation of the candidate-set charge (#2291)
- Feat(multi-stark): segment claims, cost reports, and a tiny chained machine (#2301)


### Changed

- **Breaking:** Multi-STARK proofs carry an optional binary-bus section.
- **Breaking:** `MultiStarkShape::new` takes a `has_bus` flag and `MultiStarkShape` gains a `has_bus` field.
- **Breaking:** `VerificationError` and `ProvingError` gain binary-bus variants, and `BusBindingError` is re-exported.
- Regenerated the pinned WHIR proof fixture for the new proof shape.
- **Breaking:** Binary-bus shares join the AIR zerocheck in one back-loaded sumcheck.
  `MultiStarkProof::bus` holds the product-tree proof alone, bus tables open only at the zerocheck point,
  `ZerocheckShape`, `ZerocheckChallenges` and `ZerocheckReduction` gain the bus batching fields,
  `ZerocheckError` gains `BusClaim` and `BusBinding`.
  `MultiStarkProof::bus` is now a `p3_bus::BusProof`.
  `BusBindingError::CompositionPointDimension` now accepts any point at least as wide as the tallest bus table.
  The security report charges one `binary-bus-batching` term in place of
  `binary-bus-direction-batching` and `binary-bus-composition-sumcheck`.
- The candidate-set charge on every outer reduction term is applied once, by
  `p3_security::SecurityTerm::over_candidates`, instead of being open-coded over a
  term slice. Reported security levels are unchanged.

### Removed

- **Breaking:** `p3_multi_stark::proof::BusProof`, the wrapper around the product-tree proof and the separate composition sumcheck.
- **Breaking:** `VerificationError::BusSumcheck`.
- **Breaking:** `BusBindingError::InitialClaimMismatch` and `BusBindingError::TerminalMismatch`.

## [0.7.0] - 2026-09-04
### Merged PRs
- Test(multi-stark): add WHIR proof-serialization compat fixture (#1996)
- Feat(field): characteristic-agnostic groundwork for binary fields (#2000)
- Feat: adapt log levels of some inner functions (#1999)
