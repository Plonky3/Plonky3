# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]
## [0.6.0] - 2026-06-11
### Merged PRs
- Air: some debugging improvements (#1431)
- Field: rm unused packed_linear_combination (#1460)
- Perf: abstract away Copy vs Clone (#1463)
- Feat: add support for Periodic Columns at runtime (#1462)
- Feat: add `assert_zeros_ext()` method to `ExtensionBuilder` (#1493)
- Feat: add `assert_eq_arrays` to `AirBuilder` (#1507)
- Air: document virtual pair column composition patterns (#1419)
- Feat: add Quintic extension impl for SymbolicExpression (#1522)
- Refactor(air): split air.rs into dedicated modules (#1466)
- Add debug shape checks (#1568)
- Air,batch-stark: show labels in constraint panic output (#1570)
- Feat: add support for extensions of degree 3 (#1497)
- Feat: bus-based cross-AIR interactions and lookup crate redesign (#1566)
- Refactor(air): merge PeriodicAirBuilder into AirBuilder (#1611)
- Chore: use T::zero_vec(n) instead of vec![T::ZERO; n] (#1633)
- Ci: tighten doc/release/TOML checks (#1689)
- Refactor: remove duplication around AirBuilder `_named` methods (#1724)
- Chore: update CHANGELOGs (#1785)

## [0.5.3] - 2026-05-15
### Merged PRs
- Air: document virtual pair column composition patterns (#1419)

## [0.5.2] - 2026-03-27
### Merged PRs
- Perf: abstract away Copy vs Clone (#1463)

## [0.5.1] - 2026-03-16
## [0.5.0] - 2026-03-10
### Merged PRs
- Refactor: integrate `Lookup` logic into the `Air` trait (#1239)
- Refactor: remove `PairBuilder` (#1250)
- Refactor(air): remove unused code and dependencies (#1268)
- Perf(air): add identity folding for SymbolicExpression arithmetic (#1292)
- Perf: only open at zeta_next for preprocessed columns that need it (#1317)
- Air: add `num_constraints` and `AirBuilderWithContext` (#1327)
- Air: unify `DebugConstraintBuilder` (#1330)
- Refactor(air): move SymbolicAirBuilder from uni-stark to air crate (#1334)
- Air: add max_constraint_degree in BaseAir (#1331)
- Air: add flag for next row of the main trace access (#1336)
- Air: merge `AirBuilderWithPublicValues` into `AirBuilder` (#1337)
- Air: rm num_public_values parameter when useless (#1339)
- Air: more granularity for next row (#1340)
- Feat: switch AirBuilder::Var back to Copy (#1368)
- Perf: change `ExprEF` bound to `Algebra` in `ExtensionBuilder` (#1342)
- Refactor(air): split symbolic expressions into base and extension types (#1369)
- Feat(air): add PeriodicAirBuilder extension trait and BaseEntry::Periodic (#1380)
- Feat(air): implement PeriodicAirBuilder for FilteredAirBuilder (#1383)
- Air: more complete symbolic test suite and adjustments (#1375)
- Air: change return type of preprocessed in air builder (#1387)
- Perf: vectorize constraint evaluations (#1388)
- Introduce AirLayout struct to bundle symbolic builder parameters (#1390)
- Feat(air): add permutation_values to PermutationAirBuilder (#1391)
- Air: rm `is_transition_window` and add `RowWindow` (#1357)
- Refactor(lookup): decouple lookup concerns from Air trait (#1392)
- Air: some touchups for `ConstraintLayout` and tests (#1393)
- Docs: fix documentation for selectors (#1407)
- Refactor: split `AirBuilder::M` into `MainWindow` and `PreprocessedWindow` (#1405)

## [0.4.2] - 2026-01-05
### Authors

## [0.4.1] - 2025-12-18
### Authors

## [0.4.0] - 2025-12-12
### Merged PRs
- Chore: add descriptions to all sub-crate manifests (#906) (Himess)
- Air: more unit tests for air utils (#936) (Thomas Coratger)
- Replace `Copy` with `Clone` in `AirBuilder`'s `Var` (#930) (Linda Guiga)
- Air: better doc for traits (#935) (Thomas Coratger)
- Chore: various small changes (#944) (Thomas Coratger)
- Weaken the trait bound of AirBuilder to allow `F` to be merely a Ring. (#977) (AngusG)
- Doc: add better doc in air and fix TODO (#1061) (Thomas Coratger)
- Clippy: small step (#1102) (Thomas Coratger)
- Clippy: add semicolon_if_nothing_returned (#1107) (Thomas Coratger)
- Add preprocessed/transparent columns to uni-stark (#1114) (o-k-d)
- Integrate lookups to prover and verifier (#1165) (Linda Guiga)

### Authors
- AngusG
- Himess
- Linda Guiga
- Thomas Coratger
- o-k-d

