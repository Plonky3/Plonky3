# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]
## [0.5.2] - 2026-03-27
### Merged PRs
- Make batched_linear_combination chunk size per-impl tunable (#1451)
- Perf: abstract away Copy vs Clone (#1463)
- Fix: use `u64` arithmetic in 64-bit challenger to avoid truncation (#1482)

## [0.5.1] - 2026-03-16
### Merged PRs
- Fix overflow dot_product_5 neon (#1429)

## [0.5.0] - 2026-03-10
### Merged PRs
- Field: implement quintic extension for KoalaBear (#1293)
- Field: faster `quintic_mul` and `quintic_square` (#1301)
- Deps: update rand and rand_xoshiro (#1314)
- Feat: support for 4-to-1 Poseidon2 instantiations for 32-bit fields (#1359)
- Chore: add default Poseidon2 regression vectors for koala-bear and baby-bear (#1361)
- Poseidon1: apply packing for koalabear (#1397)
- Poseidon1: fix round constants for KoalaBear and BabyBear (#1398)
- Naming: agree on convention for original Poseidon permutation (#1417)
- Poseidon: add Rust constants for rounds (#1416)

## [0.4.2] - 2026-01-05
### Authors

## [0.4.1] - 2025-12-18
### Authors

## [0.4.0] - 2025-12-12
### Merged PRs
- Chore: add descriptions to all sub-crate manifests (#906) (Himess)
- GCD based inversion for 31 bit fields (#921) (AngusG)
- Adding Degree 8 extensions for KoalaBear and BabyBear. (#954) (AngusG)
- Fast Octic inverse (#955) (AngusG)
- Packing Trick for Field Extensions (#958) (AngusG)
- Refactor to packed add methods (#972) (AngusG)
- Remove Nightly Features (#932) (AngusG)
- Move div_2_exp_u64 to ring (#970) (AngusG)
- Generic Poseidon2 Simplifications (#987) (AngusG)
- Koalabear: add default poseidon constants (#1008) (Thomas Coratger)
- Poseidon2: add Neon implementation for Monty31 (#1023) (Thomas Coratger)
- Fix: remove unused alloc::format imports (#1066) (Skylar Ray)
- Refactor: remove redundant clones in crypto modules (#1080) (Skylar Ray)
- Clippy: small step (#1102) (Thomas Coratger)
- Clippy: add semicolon_if_nothing_returned (#1107) (Thomas Coratger)
- Refactor: deduplicate field JSON serialization tests (#1162) (andrewshab)
- Implement uniform sampling of bits from field elements (#1050) (Sebastian)

### Authors
- AngusG
- Himess
- Sebastian
- Skylar Ray
- Thomas Coratger
- andrewshab

