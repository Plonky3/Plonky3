# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]
## [0.5.0] - 2026-03-10
### Merged PRs
- Refactor: integrate `Lookup` logic into the `Air` trait (#1239)
- Refactor: remove `PairBuilder` (#1250)
- Tests: add backward-compat proof fixtures for uni/batch verifiers (#1249)
- Fix: reduce logging noise in batch-stark with multiple AIRs (#1258)
- Update check_constraints.rs (#1296)
- [BREAKING] feat: Implement high-arity folding (#1277)
- Chore: revert #1296 (#1306)
- Deps: update rand and rand_xoshiro (#1314)
- Feat: add Merkle Caps (#1321)
- Feat: add Clone to StarkConfig and StarkGenericConfig (#1328)
- Fix: dead code removal and twiddle table race condition (#1318)
- Air: add `num_constraints` and `AirBuilderWithContext` (#1327)
- Air: unify `DebugConstraintBuilder` (#1330)
- Refactor(air): move SymbolicAirBuilder from uni-stark to air crate (#1334)
- Air: add max_constraint_degree in BaseAir (#1331)
- Air: add flag for next row of the main trace access (#1336)
- Air: merge `AirBuilderWithPublicValues` into `AirBuilder` (#1337)
- Air: rm num_public_values parameter when useless (#1339)
- Air: more granularity for next row (#1340)
- Feat: switch AirBuilder::Var back to Copy (#1368)
- Perf(uni-stark): only open preprocessed at zeta_next when needed (#1354)
- Refactor(air): split symbolic expressions into base and extension types (#1369)
- Chore: remove unused dependencies (#1374)
- Feat(air): add PeriodicAirBuilder extension trait and BaseEntry::Periodic (#1380)
- Air: change return type of preprocessed in air builder (#1387)
- Perf: vectorize constraint evaluations (#1388)
- Introduce AirLayout struct to bundle symbolic builder parameters (#1390)
- Air: rm `is_transition_window` and add `RowWindow` (#1357)
- Refactor(lookup): decouple lookup concerns from Air trait (#1392)
- Feat: add high-arity support in `MerkleTree` and `MMCS`  (#1373)
- Refactor: split `AirBuilder::M` into `MainWindow` and `PreprocessedWindow` (#1405)

## [0.4.2] - 2026-01-05
### Merged PRs
- Refactor(field): Add packed field extraction helpers and FieldArray utilities (#1211) (Adrian Hamelink)
- Enable ZK for preprocessing and in batch-stark (#1178) (Linda Guiga)
- Avoid change of Pcs's `open` method signature (#1230) (Linda Guiga)

### Authors
- Adrian Hamelink
- Linda Guiga

## [0.4.1] - 2025-12-18
### Authors

## [0.4.0] - 2025-12-12
### Merged PRs
- Field.rs: `Powers::packed_collect_n` (#888) (Adrian Hamelink)
- Uni stark: small touchups on the verifier (#910) (Thomas Coratger)
- Clippy wants us to put things inside of fmt now instead of just extra arguments... (#916) (AngusG)
- Chore: add descriptions to all sub-crate manifests (#906) (Himess)
- Fixed "attempt to subtract with overflow" issue in uni-stark (#934) (Gabriel Barreto)
- Replace `Copy` with `Clone` in `AirBuilder`'s `Var` (#930) (Linda Guiga)
- Docs: Add comprehensive documentation to constraint folder implementation (#856) (Ragnar)
- Shrink some test sizes (#524) (Daniel Lubarov)
- Fixing error on main (#939) (AngusG)
- Chore: various small changes (#944) (Thomas Coratger)
- Remove Nightly Features (#932) (AngusG)
- Small visibility changes for recursion (#1046) (Linda Guiga)
- Refactor: remove redundant clones in crypto modules (#1086) (Skylar Ray)
- Add modular lookups (local and global) with logup implementation (#1090) (Linda Guiga)
- Add multi-STARK prover and verifier (#1088) (Sai)
- Clippy: small step (#1102) (Thomas Coratger)
- Clippy: add nursery (#1103) (Thomas Coratger)
- Update symbolic_builder.rs (#1106) (AJoX)
- Clippy: add semicolon_if_nothing_returned (#1107) (Thomas Coratger)
- Clippy: add `needless_pass_by_value` (#1112) (Thomas Coratger)
- Refactor: Replace &Vec<T> with &[T] in function parameters (#1111) (Merkel Tranjes)
- Add preprocessed/transparent columns to uni-stark (#1114) (o-k-d)
- Add Preprocessed trace setup and VKs (#1150) (Sai)
- Update lookup traits and add folders with lookups (#1160) (Linda Guiga)
- ExtensionBuilder for SymbolicAirBuilder (#1161) (Linda Guiga)
- Uni-stark: add unit tests for SymbolicExpression (#1169) (Thomas Coratger)
- Uni stark: small touchups (#1163) (Thomas Coratger)
- Clarify quotient degree vs quotient chunks naming (#1156) (Sai)
- Core: add error messages to error enums via thiserror (#1168) (Thomas Coratger)
- Feat: add `SubAirBuilder` module (#1172) (Robin Salen)
- Doc: add intra-doc links (#1174) (Robin Salen)
- Integrate lookups to prover and verifier (#1165) (Linda Guiga)
- Core: small touchups (#1186) (Thomas Coratger)
- Feat: add PoW phase for batching in FRI commit phase (#1164) (Zach Langley)

### Authors
- AJoX
- Adrian Hamelink
- AngusG
- Daniel Lubarov
- Gabriel Barreto
- Himess
- Linda Guiga
- Merkel Tranjes
- Ragnar
- Robin Salen
- Sai
- Skylar Ray
- Thomas Coratger
- Zach Langley
- o-k-d

