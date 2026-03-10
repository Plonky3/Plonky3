# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]
## [0.5.0] - 2026-03-10
### Merged PRs
- Refactor: integrate `Lookup` logic into the `Air` trait (#1239)
- Chore: minor fixes (#1246)
- Refactor: remove `PairBuilder` (#1250)
- Tests: add backward-compat proof fixtures for uni/batch verifiers (#1249)
- Fix: reduce logging noise in batch-stark with multiple AIRs (#1258)
- Feat: split ProverData (#1278)
- [BREAKING] feat: Implement high-arity folding (#1277)
- Chore: revert #1296 (#1306)
- Perf: only open at zeta_next for preprocessed columns that need it (#1317)
- Feat: add debugging tool for lookups (#1310)
- Feat: add Merkle Caps (#1321)
- Air: add `num_constraints` and `AirBuilderWithContext` (#1327)
- Air: unify `DebugConstraintBuilder` (#1330)
- Refactor(air): move SymbolicAirBuilder from uni-stark to air crate (#1334)
- Air: add max_constraint_degree in BaseAir (#1331)
- Update prover.rs (#1276)
- Air: add flag for next row of the main trace access (#1336)
- Air: merge `AirBuilderWithPublicValues` into `AirBuilder` (#1337)
- Feat: include lookups in max_degree hint (#1338)
- Air: rm num_public_values parameter when useless (#1339)
- Fix: avoid unnecessary clone of quotient chunk matrices in batch-stark prover (#1341)
- Air: more granularity for next row (#1340)
- Feat: switch AirBuilder::Var back to Copy (#1368)
- Perf: change `ExprEF` bound to `Algebra` in `ExtensionBuilder` (#1342)
- Refactor(air): split symbolic expressions into base and extension types (#1369)
- Chore: remove unused dependencies (#1374)
- Feat(air): add PeriodicAirBuilder extension trait and BaseEntry::Periodic (#1380)
- Fix: add lookup expected_cumulated value in FS transcript (#1385)
- Air: change return type of preprocessed in air builder (#1387)
- Perf: vectorize constraint evaluations (#1388)
- Fix(batch-stark): handle preprocessed_uses_next_row() == false in prover parsing (#1356)
- Introduce AirLayout struct to bundle symbolic builder parameters (#1390)
- Feat(air): add permutation_values to PermutationAirBuilder (#1391)
- Air: rm `is_transition_window` and add `RowWindow` (#1357)
- Refactor(lookup): decouple lookup concerns from Air trait (#1392)
- Feat: add high-arity support in `MerkleTree` and `MMCS`  (#1373)
- Perf: borrow trace in StarkInstance instead of cloning (#1402)

## [0.4.2] - 2026-01-05
### Merged PRs
- Refactor(field): Add packed field extraction helpers and FieldArray utilities (#1211) (Adrian Hamelink)
- Enable ZK for preprocessing and in batch-stark (#1178) (Linda Guiga)
- Small changes for recursive lookups (#1229) (Linda Guiga)
- Avoid change of Pcs's `open` method signature (#1230) (Linda Guiga)

### Authors
- Adrian Hamelink
- Linda Guiga

## [0.4.1] - 2025-12-18
### Authors

## [0.4.0] - 2025-12-12
### Merged PRs
- Rename multi-stark crate to batch-stark (#1122) (Sai)
- Add preprocessed/transparent columns to uni-stark (#1114) (o-k-d)
- Challenger: add `observe_base_as_algebra_element ` to `FieldChallenger` trait (#1152) (Thomas Coratger)
- Add preprocessed column support in batch-STARK  (#1151) (Sai)
- Update lookup traits and add folders with lookups (#1160) (Linda Guiga)
- Derive Clone for PreprocessedInstanceMeta (#1166) (Linda Guiga)
- Clarify quotient degree vs quotient chunks naming (#1156) (Sai)
- Doc: add intra-doc links (#1174) (Robin Salen)
- Integrate lookups to prover and verifier (#1165) (Linda Guiga)
- Core: small touchups (#1186) (Thomas Coratger)
- Feat: add PoW phase for batching in FRI commit phase (#1164) (Zach Langley)
- Implement uniform sampling of bits from field elements (#1050) (Sebastian)

### Authors
- Linda Guiga
- Robin Salen
- Sai
- Sebastian
- Thomas Coratger
- Zach Langley
- o-k-d

