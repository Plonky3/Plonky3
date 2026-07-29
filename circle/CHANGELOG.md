# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]
## [0.6.0] - 2026-06-11
### Merged PRs
- Feat: add support for Periodic Columns at runtime (#1462)
- Perf(circle): drop vp_denoms after batch inversion (#1502)
- Fix(circle): add debug_assert for usize underflow in Point methods (#1445)
- Fix: make test suite pass in release mode (#1513)
- Refactor(fri): move FRI parameter constructors to associated methods (#1530)
- Type Circle FRI Shape Errors (#1541)
- Chore: use T::zero_vec(n) instead of vec![T::ZERO; n] (#1633)
- Chore: remove needless_range_loop allows across the workspace (#1632)
- Enforce Positive max_log_arity (#1652)
- Verifier: add a couple strengthening checks (#1666)
- Harden log_arity validation in FRI and Circle (#1676)
- Add fail-fast checks to periodic LDE builders (#1614)
- Perf(circle): pack the v_n_prod chain in compute_lagrange_den_batched (#1708)
- Fix: add PoW check in Circle STARK's commit phase (#1723)
- Minor (#1739)
- Fix(merkle-tree): enforce opened row widths in batch verification (#1757)
- Fix(circle): reject zero-query configurations (#1772)
- Fix(circle): reject oversized query-index widths (#1774)
- Perf(circle): parallelize selectors_on_coset via fused passes (#1776)
- Perf(circle): fill the LDE extension in parallel (#1778)
- Chore: update CHANGELOGs (#1785)
- Doc: add basic READMEs in main crates (#1786)
- Fix(circle): reject proofs that under-report commit rounds (#1792)
- Perf: collapse allocations in cfft path (#1795)
- Perf(circle): fuse CFFT butterfly layers into cache-resident parallel passes (#1796)
- Perf(circle): batch and parallelize the PCS open phase (#1797)

## [0.5.3] - 2026-05-15
### Merged PRs
- Perf(circle): drop vp_denoms after batch inversion (#1502)

## [0.5.2] - 2026-03-27
## [0.5.1] - 2026-03-16
## [0.5.0] - 2026-03-10
### Merged PRs
- [BREAKING] feat: Implement high-arity folding (#1277)
- Deps: update rand and rand_xoshiro (#1314)
- Feat: add Merkle Caps (#1321)
- Feat: add Clone to StarkConfig and StarkGenericConfig (#1328)
- Feat: add high-arity support in `MerkleTree` and `MMCS`  (#1373)

## [0.4.2] - 2026-01-05
### Merged PRs
- Enable ZK for preprocessing and in batch-stark (#1178) (Linda Guiga)
- Avoid change of Pcs's `open` method signature (#1230) (Linda Guiga)

### Authors
- Linda Guiga

## [0.4.1] - 2025-12-18
### Authors

## [0.4.0] - 2025-12-12
### Merged PRs
- Chore: add descriptions to all sub-crate manifests (#906) (Himess)
- Remove Nightly Features (#932) (AngusG)
- Docs: improve documentation for Circle STARKs deep quotient algorithms (#1079) (Adrian)
- Clippy: small step (#1102) (Thomas Coratger)
- Clippy: add nursery (#1103) (Thomas Coratger)
- Clippy: add semicolon_if_nothing_returned (#1107) (Thomas Coratger)
- Clippy: add `needless_pass_by_value` (#1112) (Thomas Coratger)
- Circle: batch inverses in selectors_on_coset (#1068) (Forostovec)
- Core: add error messages to error enums via thiserror (#1168) (Thomas Coratger)
- Challenger: use `observe_algebra_slice` when possible (#1187) (Thomas Coratger)
- Feat: add PoW phase for batching in FRI commit phase (#1164) (Zach Langley)

### Authors
- Adrian
- AngusG
- Forostovec
- Himess
- Thomas Coratger
- Zach Langley

