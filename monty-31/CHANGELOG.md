# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]
## [0.5.2] - 2026-03-27
### Merged PRs
- Poseidon1: speedup monty-31 packed version on avx (#1420)
- Monty31: use transmute in poseidon1 packing (#1446)
- Make batched_linear_combination chunk size per-impl tunable (#1451)
- Perf: abstract away Copy vs Clone (#1463)

## [0.5.1] - 2026-03-16
### Merged PRs
- Monty31: faster aarch64 neon `quintic_mul_packed` (#1422)
- Monty31: faster aarch64 neon `octic_mul_packed` (#1423)
- Fix overflow dot_product_5 neon (#1429)

## [0.5.0] - 2026-03-10
### Merged PRs
- Transpose: use p3-util rectangular transposition everywhere (#1256)
- Fix: reduce logging noise in batch-stark with multiple AIRs (#1258)
- Field: implement quintic extension for KoalaBear (#1293)
- Deps: update rand and rand_xoshiro (#1314)
- SIMD for koala-bear extension 5 (#1305)
- Fix: dead code removal and twiddle table race condition (#1318)
- Feat: support for 4-to-1 Poseidon2 instantiations for 32-bit fields (#1359)
- Poseidon1: implementation based on HorizenLab (#1333)
- Poseidon1: packed form for monty31 (#1378)
- Fix: add range check in MontyField31 deserialization (#1399)
- Field: add packing strategy for `mixed_dot_product` (#1404)
- Poseidon1: speedup monty-31 packed version (#1414)
- Naming: agree on convention for original Poseidon permutation (#1417)
- Monty 31: faster forward dft (#1418)

## [0.4.2] - 2026-01-05
### Merged PRs
- Refactor: add public const `new` and `new_array` for all fields (#1222) (Adrian Hamelink)

### Authors
- Adrian Hamelink

## [0.4.1] - 2025-12-18
### Authors

## [0.4.0] - 2025-12-12
### Merged PRs
- Field.rs: `Powers::packed_collect_n` (#888) (Adrian Hamelink)
- Chore: add descriptions to all sub-crate manifests (#906) (Himess)
- From_biguint method for Bn254 (#914) (AngusG)
- GCD based inversion for 31 bit fields (#921) (AngusG)
- Fixing a pair of clippy complaints in AVX512 (#926) (AngusG)
- Monty31: small touchups for packing (#927) (Thomas Coratger)
- Packing: small touchups (#937) (Thomas Coratger)
- Use `#[derive(...)]` for Debug and Default for packed fields. (#945) (AngusG)
- Adding Macros to remove boilerplate impls (#943) (AngusG)
- Combining Interleave Code (#950) (AngusG)
- Add a macro for implying PackedValue for PackedFields (#949) (AngusG)
- Packing Trick for Field Extensions (#958) (AngusG)
- Chore: small touchups and poseidon external unit tests (#971) (Thomas Coratger)
- Refactor to packed add methods (#972) (AngusG)
- Speed up Extension Field Addition (#980) (AngusG)
- Remove Nightly Features (#932) (AngusG)
- Move halve to ring (#969) (AngusG)
- Packed Sub Refactor (#979) (AngusG)
- Move div_2_exp_u64 to ring (#970) (AngusG)
- Speed Up Extension Field Subtraction (#988) (AngusG)
- Must Use (#996) (AngusG)
- Speed Up Base-Extension Multiplication (#998) (AngusG)
- Generic Poseidon2 Simplifications (#987) (AngusG)
- Compile Time asserts  (#1015) (AngusG)
- Monty31: add halve for aarch64 neon (#1020) (Thomas Coratger)
- More Const Assert fixes (#1024) (AngusG)
- Poseidon2: add Neon implementation for Monty31 (#1023) (Thomas Coratger)
- Monty31: add aarch64 neon custom `exp_5` and `exp_7` (#1033) (Thomas Coratger)
- Small Neon Refactor (#1037) (AngusG)
- Monty31: better Poseidon2 for aarch64 neon using `exp_small` (#1035) (Thomas Coratger)
- Monty 31: more efficient aarch64 neon `quartic_mul_packed` (#1060) (Thomas Coratger)
- Poseidon2 doc comment fixes (#1071) (AngusG)
- Monty-31: implement more efficient `dot_product_2` for neon (#1070) (Thomas Coratger)
- Clippy: small step (#1102) (Thomas Coratger)
- Clippy: add nursery (#1103) (Thomas Coratger)
- Feat: add thread safety to dft implementations (#999) (Jeremi Do Dinh)
- Clippy: add semicolon_if_nothing_returned (#1107) (Thomas Coratger)
- Fixing a few clippy lints (#1115) (AngusG)
- Monty31: const assert in dot product (#1154) (Thomas Coratger)
- Allow users to impl either permute or permute_mut (#1175) (AngusG)
- Core: small touchups (#1186) (Thomas Coratger)

### Authors
- Adrian Hamelink
- AngusG
- Himess
- Jeremi Do Dinh
- Thomas Coratger

