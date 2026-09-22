# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
