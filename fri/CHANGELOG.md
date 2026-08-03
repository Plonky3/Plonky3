# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]
## [0.6.0] - 2026-06-11
### Merged PRs
- Feat: add support for Periodic Columns at runtime (#1462)
- Refactor(verifier): remove unused MissingInput variant (#1484)
- Fix(fri): split InvalidProofShape into typed verifier errors (#1519)
- Refactor(fri): move FRI parameter constructors to associated methods (#1530)
- Refactor(matrix): move column-appending helpers to `DenseMatrix` methods (#1533)
- Refactor(interpolation): replace free functions with Interpolate extension trait (#1540)
- Fri: full test coverage for verifier errors (#1529)
- Perf(interpolation): optimize barycentric weights via algebraic identity (#1553)
- Chore: use T::zero_vec(n) instead of vec![T::ZERO; n] (#1633)
- Chore: remove needless_range_loop allows across the workspace (#1632)
- Enforce Positive max_log_arity (#1652)
- Fix(fri): enforce num_chunks > 1 in HidingFriPcs::get_quotient_ldes (#1595)
- Verifier: add a couple strengthening checks (#1666)
- Harden log_arity validation in FRI and Circle (#1676)
- Ci: tighten doc/release/TOML checks (#1689)
- Feat: add a `HornerIter` supertrait on `DoubleEndedIterator` (#1692)
- Chore: HornerIter follow-ups from #1692 review (#1693)
- Fri: reject global max height mismatches (#1698)
- Add fail-fast checks to periodic LDE builders (#1614)
- Fix(fri): type hiding PCS random opening errors (#1704)
- Couple miscellaneous tweaks (#1731)
- Fix(params): error instead of panic on infeasible WHIR/FRI params (#1737)
- Fix(fri): reject zero-query instances on both sides (#1735)
- Refactor(fri): hoist height-1 check and unify max-height in prover (#1756)
- Fix(merkle-tree): enforce opened row widths in batch verification (#1757)
- Fix(fri): reject opening point equal to the query point instead of panicking (#1762)
- Chore: update CHANGELOGs (#1785)
- Doc: add basic READMEs in main crates (#1786)
- Fix(fri): reject input matrices opened at zero points (#1793)

## [0.5.3] - 2026-05-15
## [0.5.2] - 2026-03-27
## [0.5.1] - 2026-03-16
## [0.5.0] - 2026-03-10
### Merged PRs
- Chore: minor fixes (#1246)
- Fix: reduce logging noise in batch-stark with multiple AIRs (#1258)
- [BREAKING] feat: Implement high-arity folding (#1277)
- Deps: update rand and rand_xoshiro (#1314)
- Feat: add Merkle Caps (#1321)
- Perf: reduce number of batch inversions in general FRI arity case (#1325)
- Feat: add Clone to StarkConfig and StarkGenericConfig (#1328)
- Fix: remove unnecessary clone of opening points (#1344)
- Feat(fri): handle extrapolation for LDEs (#1352)
- Feat: make HidingFriPcs Sync (#1395)
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
- Field.rs: `Powers::packed_collect_n` (#888) (Adrian Hamelink)
- Chore: add descriptions to all sub-crate manifests (#906) (Himess)
- More Clippy Complaints (#931) (AngusG)
- Shrink some test sizes (#524) (Daniel Lubarov)
- Update doc comment and some other comment fixes. (#959) (AngusG)
- Minor FRI refactor - make open input its own function (#961) (AngusG)
- Remove duplicated definition (#1031) (AngusG)
- Refactor: remove redundant clones in crypto modules (#1086) (Skylar Ray)
- Clippy: small step (#1102) (Thomas Coratger)
- Clippy: add nursery (#1103) (Thomas Coratger)
- Clippy: add semicolon_if_nothing_returned (#1107) (Thomas Coratger)
- Clippy: add `needless_pass_by_value` (#1112) (Thomas Coratger)
- Core: add error messages to error enums via thiserror (#1168) (Thomas Coratger)
- Challenger: use `observe_algebra_slice` when possible (#1187) (Thomas Coratger)
- Feat: add PoW phase for batching in FRI commit phase (#1164) (Zach Langley)

### Authors
- Adrian Hamelink
- AngusG
- Daniel Lubarov
- Himess
- Skylar Ray
- Thomas Coratger
- Zach Langley

