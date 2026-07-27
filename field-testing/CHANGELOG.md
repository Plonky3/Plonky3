# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]
## [0.6.0] - 2026-06-11
### Merged PRs
- Field: reinforcement of tests with edge cases and proptests (#1435)
- Add broadcast, pack_columns, pack_columns_fn, and unpack_iter to PackedValue (#1450)
- Make batched_linear_combination chunk size per-impl tunable (#1451)
- Field: rm unused packed_linear_combination (#1460)
- Field: implement packed / packed division (#1457)
- Perf: faster `GoldilocksPackedNeon` field operations (#1515)
- Field: add division ops for packed binomial extension (#1512)
- Perf: halve goldilocks optimizations (#1606)
- Add carry-critical NEON dot product regression tests (#1600)
- Perf(field): preserve packed ILP in batch_multiplicative_inverse for non-WIDTH-aligned inputs (#1608)
- Field: expand packed-extension API; fix unsound PackedValue impls (#1620)
- Test(field): cover packed-extension Div with per-lane consistency check (#1624)
- Feat: add WASM32 SIMD128 support for Goldilocks (#1644)
- Ci: tighten doc/release/TOML checks (#1689)
- Fix(field): make from_int total for iN::MIN (#1758)
- Fix(mersenne-31): make serde encoding canonical (#1773)
- Chore: update CHANGELOGs (#1785)

## [0.5.3] - 2026-05-15
### Merged PRs
- Perf: faster `GoldilocksPackedNeon` field operations (#1515)
- Perf: halve goldilocks optimizations (#1606)
- Add broadcast, pack_columns, pack_columns_fn, and unpack_iter to PackedValue (#1450)
- Add carry-critical NEON dot product regression tests (#1600)

## [0.5.2] - 2026-03-27
### Merged PRs
- Field: reinforcement of tests with edge cases and proptests (#1435)
- Make batched_linear_combination chunk size per-impl tunable (#1451)

## [0.5.1] - 2026-03-16
### Merged PRs
- Fix overflow dot_product_5 neon (#1429)

## [0.5.0] - 2026-03-10
### Merged PRs
- Fix(field): return ONE for empty Product iterator (#1272)
- Field: implement quintic extension for KoalaBear (#1293)
- Deps: update rand and rand_xoshiro (#1314)
- Field: add packing strategy for `mixed_dot_product` (#1404)
- Add KoalaBear Deserialize Boundary Tests (#1406)

## [0.4.2] - 2026-01-05
### Authors

## [0.4.1] - 2025-12-18
### Authors

## [0.4.0] - 2025-12-12
### Merged PRs
- Field.rs: `Powers::packed_collect_n` (#888) (Adrian Hamelink)
- Clippy wants us to put things inside of fmt now instead of just extra arguments... (#916) (AngusG)
- Chore: add descriptions to all sub-crate manifests (#906) (Himess)
- Custom halve impl for Bn254 (#919) (AngusG)
- Adding custom mul/div_exp_2_u64 for the Goldilocks field. (#923) (AngusG)
- More Clippy Complaints (#931) (AngusG)
- Chore: use `collect_n` with powers when possible (#963) (Thomas Coratger)
- Move halve to ring (#969) (AngusG)
- Move div_2_exp_u64 to ring (#970) (AngusG)
- Speed Up Base-Extension Multiplication (#998) (AngusG)
- Monty31: add aarch64 neon custom `exp_5` and `exp_7` (#1033) (Thomas Coratger)
- Clippy: small step (#1102) (Thomas Coratger)
- Feat: add thread safety to dft implementations (#999) (Jeremi Do Dinh)
- Clippy: add semicolon_if_nothing_returned (#1107) (Thomas Coratger)
- Refactor: deduplicate field JSON serialization tests (#1162) (andrewshab)

### Authors
- Adrian Hamelink
- AngusG
- Himess
- Jeremi Do Dinh
- Thomas Coratger
- andrewshab

