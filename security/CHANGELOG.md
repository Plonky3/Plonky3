# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `CandidateSet`, the checked candidate count a commitment leaves open, and `ChargedTerm`,
  a term that has already paid for one. A charge forwards the set rather than consuming it,
  and a charged term cannot be charged again.

### Changed

- **Breaking:** `SecurityTerm::over_candidates` takes a `CandidateSet` instead of an `f64`
  and returns a `ChargedTerm`. It is now the only implementation of the candidate-set charge
  in the workspace.

## [0.7.0] - 2026-09-04
### Merged PRs
- Fix(stir): fallible config derivation & dedup eta-parameterized security formulas (#1998)
- Fix(security): account for out-of-domain point count in budget's OOD round (#2007)
- Feat(security): add legacy conjectured FRI soundness bound (#2018)

