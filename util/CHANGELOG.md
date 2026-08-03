# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]
## [0.6.0] - 2026-06-11
### Merged PRs
- Utils: add `log3_strict_usize` (#1444)
- Guard verifier degree_bits (#1535)
- Chore: use T::zero_vec(n) instead of vec![T::ZERO; n] (#1633)
- Chore: remove needless_range_loop allows across the workspace (#1632)
- Fix(merkle-tree): make MerkleTreeHidingMmcs Sync (#1559)
- Fix: local refs for dev-deps (#1663)
- Feat: add security estimation (#1329)
- Couple fixes (#1688)
- Ci: tighten doc/release/TOML checks (#1689)
- Isolate bit_reverse benchmark from clone overhead (#1703)
- Refactor: move DisjointMutPtr to p3-util (#1720)
- Couple miscellaneous tweaks (#1731)
- Fix(util): guard reverse_bits_len against oversized bit_len (#1752)
- Chore: update CHANGELOGs (#1785)

## [0.5.3] - 2026-05-15
## [0.5.2] - 2026-03-27
## [0.5.1] - 2026-03-16
## [0.5.0] - 2026-03-10
### Merged PRs
- Util: better rect transpose with NEON for 32 bits fields (#1192)
- Update square.rs (#1261)
- Fix: deduplicate key lookup logic in LinearMap (#1291)
- Deps: update rand and rand_xoshiro (#1314)
- Util: faster rectangular transposition 64-bit fields (#1332)

## [0.4.2] - 2026-01-05
### Merged PRs
- Refactor: add public const `new` and `new_array` for all fields (#1222) (Adrian Hamelink)

### Authors
- Adrian Hamelink

## [0.4.1] - 2025-12-18
### Authors

## [0.4.0] - 2025-12-12
### Merged PRs
- Clippy wants us to put things inside of fmt now instead of just extra arguments... (#916) (AngusG)
- Chore: add descriptions to all sub-crate manifests (#906) (Himess)
- GCD based inversion for 31 bit fields (#921) (AngusG)
- Fast GCD Inverse for Goldilocks  (#925) (AngusG)
- More Clippy Complaints (#931) (AngusG)
- Chore: remove useless bench_reverse_bits benchmark (#933) (Galoretka)
- Packed Goldilocks Small Refactor (#946) (AngusG)
- Make Assume unsafe and add a doc comment (#1005) (AngusG)
- Compile Time asserts  (#1015) (AngusG)
- Clippy: small step (#1102) (Thomas Coratger)
- Clippy: add nursery (#1103) (Thomas Coratger)
- Clippy: add semicolon_if_nothing_returned (#1107) (Thomas Coratger)
- Clippy: add match_bool (#1126) (Thomas Coratger)

### Authors
- AngusG
- Galoretka
- Himess
- Thomas Coratger

