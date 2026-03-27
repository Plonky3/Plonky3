# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]
## [0.5.2] - 2026-03-27
### Merged PRs
- Make batched_linear_combination chunk size per-impl tunable (#1451)
- Perf: abstract away Copy vs Clone (#1463)

## [0.5.1] - 2026-03-16
### Merged PRs
- Fix div2_asm goldilocks neon (#1430)

## [0.5.0] - 2026-03-10
### Merged PRs
- Feat: implement neon vectorization for Poseidon2-Goldilocks (#1303)
- Deps: update rand and rand_xoshiro (#1314)
- Rand: small import fix (#1316)
- Feat: impl `GenericPoseidon2LinearLayers<WIDTH>` for `Goldilocks` field (#1343)
- Perf: optimize Poseidon2 instances for Goldilocks (#1367)
- Goldilocks: complete proptest coverage for Poseidon2 asm (#1372)
- Poseidon1: implementation based on HorizenLab (#1333)
- Poseidon1: packed form for monty31 (#1378)
- Poseidon1: add aarch64 neon packing strategy for Goldilocks (#1401)
- Chore: pacify clippy for Neon (#1413)
- Poseidon1: fix round constants for KoalaBear and BabyBear (#1398)
- Naming: agree on convention for original Poseidon permutation (#1417)
- Poseidon: add Rust constants for rounds (#1416)

## [0.4.2] - 2026-01-05
### Merged PRs
- Refactor: add public const `new` and `new_array` for all fields (#1222) (Adrian Hamelink)

### Authors
- Adrian Hamelink

## [0.4.1] - 2025-12-18
### Authors

## [0.4.0] - 2025-12-12
### Merged PRs
- Chore: add descriptions to all sub-crate manifests (#906) (Himess)
- Adding custom mul/div_exp_2_u64 for the Goldilocks field. (#923) (AngusG)
- Fast GCD Inverse for Goldilocks  (#925) (AngusG)
- Packing: small touchups (#937) (Thomas Coratger)
- Use `#[derive(...)]` for Debug and Default for packed fields. (#945) (AngusG)
- Adding Macros to remove boilerplate impls (#943) (AngusG)
- Packed Goldilocks Small Refactor (#946) (AngusG)
- Combining Interleave Code (#950) (AngusG)
- Add a macro for implying PackedValue for PackedFields (#949) (AngusG)
- Packing Trick for Field Extensions (#958) (AngusG)
- Remove Nightly Features (#932) (AngusG)
- Move halve to ring (#969) (AngusG)
- Move div_2_exp_u64 to ring (#970) (AngusG)
- Must Use (#996) (AngusG)
- Make Assume unsafe and add a doc comment (#1005) (AngusG)
- Clippy: small step (#1102) (Thomas Coratger)
- Clippy: add nursery (#1103) (Thomas Coratger)
- Clippy: add semicolon_if_nothing_returned (#1107) (Thomas Coratger)
- Clippy: add `needless_pass_by_value` (#1112) (Thomas Coratger)
- Allow users to impl either permute or permute_mut (#1175) (AngusG)
- Implement uniform sampling of bits from field elements (#1050) (Sebastian)

### Authors
- AngusG
- Himess
- Sebastian
- Thomas Coratger

