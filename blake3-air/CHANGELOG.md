# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]
## [0.8.0] - 2026-09-23
### Merged PRs
- Feat(examples): prove Keccak-f and BLAKE3 over GF(2^128) with the binary PCS (#2164)
- Perf: batch Boolean trace openings and keep Blake3 traces packed (#2227)
- Perf(binary): reuse the reduction's evaluation, and make the sliced-round count a choice (#2246)
- Perf(multi-stark): weight a batch of AIR constraints with one dot product (#2267)
- Perf(binary)!: leave Keccak-f booleanity to the Boolean commitment in `prove_hash_binary` (#2287)
- Perf(binary)!: speed up the sliced zerocheck, ring switch, PCS commit and opening, and BLAKE3 witness (#2307)

## [0.7.0] - 2026-09-04
## [0.6.0] - 2026-06-11
### Merged PRs
- Ci: tighten doc/release/TOML checks (#1689)
- Chore: update CHANGELOGs (#1785)

## [0.5.3] - 2026-05-15
## [0.5.2] - 2026-03-27
## [0.5.1] - 2026-03-16
## [0.5.0] - 2026-03-10
### Merged PRs
- Refactor: minor code cleanups across blake3-air, bn254, challenger (#1287)
- Deps: update rand and rand_xoshiro (#1314)
- Air: add flag for next row of the main trace access (#1336)
- Air: more granularity for next row (#1340)
- Feat: switch AirBuilder::Var back to Copy (#1368)
- Air: rm `is_transition_window` and add `RowWindow` (#1357)

## [0.4.2] - 2026-01-05
### Authors

## [0.4.1] - 2025-12-18
### Authors

## [0.4.0] - 2025-12-12
### Merged PRs
- Chore: add descriptions to all sub-crate manifests (#906) (Himess)
- Replace `Copy` with `Clone` in `AirBuilder`'s `Var` (#930) (Linda Guiga)
- Clippy: small step (#1102) (Thomas Coratger)
- Clippy: add semicolon_if_nothing_returned (#1107) (Thomas Coratger)

### Authors
- Himess
- Linda Guiga
- Thomas Coratger

