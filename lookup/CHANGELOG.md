# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]
## [0.5.0] - 2026-03-10
### Merged PRs
- Refactor: integrate `Lookup` logic into the `Air` trait (#1239)
- Refactor: remove `PairBuilder` (#1250)
- Update Cargo.toml (#1297)
- Perf: parallelize logUp (#1295)
- Chore: revert #1296 (#1306)
- Deps: update rand and rand_xoshiro (#1314)
- Feat: add debugging tool for lookups (#1310)
- Perf: optimize lookups further (#1315)
- Refactor(air): move SymbolicAirBuilder from uni-stark to air crate (#1334)
- Air: merge `AirBuilderWithPublicValues` into `AirBuilder` (#1337)
- Feat: include lookups in max_degree hint (#1338)
- Lookup: optimize logup (#1335)
- Update Cargo.toml (#1362)
- Feat: switch AirBuilder::Var back to Copy (#1368)
- Refactor(air): split symbolic expressions into base and extension types (#1369)
- Feat(air): add PeriodicAirBuilder extension trait and BaseEntry::Periodic (#1380)
- Air: change return type of preprocessed in air builder (#1387)
- Perf: vectorize constraint evaluations (#1388)
- Introduce AirLayout struct to bundle symbolic builder parameters (#1390)
- Feat(air): add permutation_values to PermutationAirBuilder (#1391)
- Air: rm `is_transition_window` and add `RowWindow` (#1357)
- Refactor(lookup): decouple lookup concerns from Air trait (#1392)
- Feat: add high-arity support in `MerkleTree` and `MMCS`  (#1373)
- Refactor: split `AirBuilder::M` into `MainWindow` and `PreprocessedWindow` (#1405)

## [0.4.2] - 2026-01-05
### Merged PRs
- Small changes for recursive lookups (#1229) (Linda Guiga)

### Authors
- Linda Guiga

## [0.4.1] - 2025-12-18
### Authors

## [0.4.0] - 2025-12-12
### Merged PRs
- Add modular lookups (local and global) with logup implementation (#1090) (Linda Guiga)
- Clippy: small step (#1102) (Thomas Coratger)
- Clippy: add nursery (#1103) (Thomas Coratger)
- Clippy: add `needless_pass_by_value` (#1112) (Thomas Coratger)
- Update lookup traits and add folders with lookups (#1160) (Linda Guiga)
- ExtensionBuilder for SymbolicAirBuilder (#1161) (Linda Guiga)
- Core: add error messages to error enums via thiserror (#1168) (Thomas Coratger)
- Doc: add intra-doc links (#1174) (Robin Salen)
- Integrate lookups to prover and verifier (#1165) (Linda Guiga)
- Core: small touchups (#1186) (Thomas Coratger)

### Authors
- Linda Guiga
- Robin Salen
- Thomas Coratger

