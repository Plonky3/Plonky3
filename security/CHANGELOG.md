# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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

