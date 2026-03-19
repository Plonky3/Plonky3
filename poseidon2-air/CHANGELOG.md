# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]
## [0.5.1] - 2026-03-16
## [0.5.0] - 2026-03-10
### Merged PRs
- Deps: update rand and rand_xoshiro (#1314)
- Feat: add Merkle Caps (#1321)
- Fix: dead code removal and twiddle table race condition (#1318)
- Air: add max_constraint_degree in BaseAir (#1331)
- Air: add flag for next row of the main trace access (#1336)
- Fix(poseidon2-air): remove unused export column (#1358)
- Air: more granularity for next row (#1340)
- Feat: switch AirBuilder::Var back to Copy (#1368)
- Air: rm `is_transition_window` and add `RowWindow` (#1357)
- Feat: add high-arity support in `MerkleTree` and `MMCS`  (#1373)
- Poseidon1: add arithmetization crate (#1384)
- Poseidon: add Rust constants for rounds (#1416)

## [0.4.2] - 2026-01-05
### Merged PRs
- Small changes for recursive lookups (#1229) (Linda Guiga)

### Authors
- Linda Guiga

## [0.4.1] - 2025-12-18
### Authors

## [0.4.0] - 2025-12-12
### Merged PRs
- Chore: add descriptions to all sub-crate manifests (#906) (Himess)
- More Clippy Complaints (#931) (AngusG)
- Replace `Copy` with `Clone` in `AirBuilder`'s `Var` (#930) (Linda Guiga)
- Weaken the trait bound of AirBuilder to allow `F` to be merely a Ring. (#977) (AngusG)
- Generic Poseidon2 Simplifications (#987) (AngusG)
- Clippy: small step (#1102) (Thomas Coratger)
- Refactor: Replace &Vec<T> with &[T] in function parameters (#1111) (Merkel Tranjes)
- Make generate_trace_rows_for_perm public (#1159) (Alonso González)

### Authors
- Alonso González
- AngusG
- Himess
- Linda Guiga
- Merkel Tranjes
- Thomas Coratger

