# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]
## [0.6.0] - 2026-06-11
### Merged PRs
- Feat: expose arity schedule (#1433)
- Add broadcast, pack_columns, pack_columns_fn, and unpack_iter to PackedValue (#1450)
- Merkle-tree: unify arity step scheduling rules (#1465)
- Feat(merkle-tree): add Merkle path pruning for compact multi-opening proofs (#1544)
- Chore: remove needless_range_loop allows across the workspace (#1632)
- Fix(merkle-tree): make MerkleTreeHidingMmcs Sync (#1559)
- Perf(merkle-tree): amortize verifier cost in verify_batch_pruned (#1648)
- Fix: local refs for dev-deps (#1663)
- Verifier: add a couple strengthening checks (#1666)
- Ci: tighten doc/release/TOML checks (#1689)
- Refactor(merkle-tree): remove unused error variants (#1705)
- Fix(merkle-tree): reject unreachable MMCS dimensions (#1768)
- Fix(merkle-tree): enforce opened row widths in batch verification (#1757)
- Chore: update CHANGELOGs (#1785)
- Doc: add basic READMEs in main crates (#1786)
- Fix(merkle-tree): bound pruned-proof scratch early and enrich the error set (#1794)
- Refactor(merkle-tree): split mmcs.rs into a feature-organized mmcs/ module (#1801)

## [0.5.3] - 2026-05-15
### Merged PRs
- Add broadcast, pack_columns, pack_columns_fn, and unpack_iter to PackedValue (#1450)

## [0.5.2] - 2026-03-27
## [0.5.1] - 2026-03-16
## [0.5.0] - 2026-03-10
### Merged PRs
- Deps: update rand and rand_xoshiro (#1314)
- Feat: add Merkle Caps (#1321)
- Feat: add high-arity support in `MerkleTree` and `MMCS`  (#1373)

## [0.4.2] - 2026-01-05
### Merged PRs
- Refactor(field): Add packed field extraction helpers and FieldArray utilities (#1211) (Adrian Hamelink)

### Authors
- Adrian Hamelink

## [0.4.1] - 2025-12-18
### Authors

## [0.4.0] - 2025-12-12
### Merged PRs
- Merkle tree: add documentation for MerkleTreeMmcs and errors (#908) (Thomas Coratger)
- Clippy wants us to put things inside of fmt now instead of just extra arguments... (#916) (AngusG)
- Merkle tree: full documentation for first_digest_layer (#924) (Thomas Coratger)
- Merkle tree: very small doc touchup (#928) (Thomas Coratger)
- Merkle tree: add const assert (#1040) (Thomas Coratger)
- Doc: add better doc in air and fix TODO (#1061) (Thomas Coratger)
- Clippy: small step (#1102) (Thomas Coratger)
- Clippy: add semicolon_if_nothing_returned (#1107) (Thomas Coratger)
- Clippy: add `needless_pass_by_value` (#1112) (Thomas Coratger)
- Add input size checks in MMCS (#1119) (Sai)
- Core: add error messages to error enums via thiserror (#1168) (Thomas Coratger)

### Authors
- AngusG
- Sai
- Thomas Coratger

