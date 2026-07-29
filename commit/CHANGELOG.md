# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]
## [0.6.0] - 2026-06-11
### Merged PRs
- Feat: add support for Periodic Columns at runtime (#1462)
- Core: clippy fix (#1526)
- Refactor(interpolation): replace free functions with Interpolate extension trait (#1540)
- Whir: full integration from whir-p3 (#1523)
- Whir: wire stacked layouts to pcs (#1612)
- Chore: use T::zero_vec(n) instead of vec![T::ZERO; n] (#1633)
- Verifier: add a couple strengthening checks (#1666)
- Ci: tighten doc/release/TOML checks (#1689)
- Add fail-fast checks to periodic LDE builders (#1614)
- Perf: borrow MMCS to remove deep copy (#1721)
- Fix(merkle-tree): enforce opened row widths in batch verification (#1757)
- Chore: update CHANGELOGs (#1785)
- Doc: add basic READMEs in main crates (#1786)

## [0.5.3] - 2026-05-15
## [0.5.2] - 2026-03-27
## [0.5.1] - 2026-03-16
## [0.5.0] - 2026-03-10
### Merged PRs
- Feat: add Clone to StarkConfig and StarkGenericConfig (#1328)

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
- Mmcs: better doc for `ExtensionMmcs` (#947) (Thomas Coratger)
- Optimize split_evals to reduce copying (#1043) (sashass1315)
- Clippy: small step (#1102) (Thomas Coratger)
- Clippy: add nursery (#1103) (Thomas Coratger)
- Clippy: add semicolon_if_nothing_returned (#1107) (Thomas Coratger)
- Add preprocessed/transparent columns to uni-stark (#1114) (o-k-d)

### Authors
- Adrian Hamelink
- Himess
- Thomas Coratger
- o-k-d
- sashass1315

