# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]
## [0.6.0] - 2026-06-11
### Merged PRs
- Add broadcast, pack_columns, pack_columns_fn, and unpack_iter to PackedValue (#1450)
- Make batched_linear_combination chunk size per-impl tunable (#1451)
- Field: rm unused packed_linear_combination (#1460)
- Perf: abstract away Copy vs Clone (#1463)
- Field: implement packed / packed division (#1457)
- Batch stark: separate Fiat Shamir transcript (#1475)
- Field: add missing extension degree for `packed_mod_add` and `packed_mod_sub` (#1421)
- Field: batch invert packed quintic divisors (#1505)
- Field: add division ops for packed binomial extension (#1512)
- Core: couple more compile time assertions (#1525)
- Field: specialize packed mixed_dot_product by chunk strategy (#1573)
- Feat: add support for extensions of degree 3 (#1497)
- Perf(whir): delayed Montgomery reduction in sumcheck round-coefficient kernel (#1597)
- Perf(field): preserve packed ILP in batch_multiplicative_inverse for non-WIDTH-aligned inputs (#1608)
- Field: expand packed-extension API; fix unsound PackedValue impls (#1620)
- Chore: use T::zero_vec(n) instead of vec![T::ZERO; n] (#1633)
- Chore: remove needless_range_loop allows across the workspace (#1632)
- Fix: leverage packing for Powers (#1639)
- Fix: local refs for dev-deps (#1663)
- Perf: extend collect_n and shifted_powers to remaining call sites (#1683)
- Ci: tighten doc/release/TOML checks (#1689)
- Feat: add a `HornerIter` supertrait on `DoubleEndedIterator` (#1692)
- Chore: HornerIter follow-ups from #1692 review (#1693)
- Refactor(field): unify extension-field logic under common abstraction (#1696)
- Refactor(field): default ext_square to a general multiply (#1697)
- Fix(field): make from_int total for iN::MIN (#1758)
- Chore: update CHANGELOGs (#1785)
- Doc: add basic READMEs in main crates (#1786)

## [0.5.3] - 2026-05-15
### Merged PRs
- Field: add missing extension degree for `packed_mod_add` and `packed_mod_sub` (#1421)
- Core: couple more compile time assertions (#1525)
- Field: specialize packed mixed_dot_product by chunk strategy (#1573)
- Add broadcast, pack_columns, pack_columns_fn, and unpack_iter to PackedValue (#1450)
- Add carry-critical NEON dot product regression tests (#1600)

## [0.5.2] - 2026-03-27
### Merged PRs
- Make batched_linear_combination chunk size per-impl tunable (#1451)
- Perf: abstract away Copy vs Clone (#1463)

## [0.5.1] - 2026-03-16
## [0.5.0] - 2026-03-10
### Merged PRs
- Field: relax redundant bounds in From<[A; D]> for BinomialExtensionField (#1231)
- Chore(field): Remove deprecated scale_slice_in_place wrapper (#1264)
- Fix(field): return ONE for empty Product iterator (#1272)
- Field: small touchup for binomial extension (#1300)
- Field: implement quintic extension for KoalaBear (#1293)
- Field: faster `quintic_mul` and `quintic_square` (#1301)
- Deps: update rand and rand_xoshiro (#1314)
- SIMD for koala-bear extension 5 (#1305)
- Fix(poseidon2-air): remove unused export column (#1358)
- Poseidon1: implementation based on HorizenLab (#1333)
- Poseidon1: add arithmetization crate (#1384)

## [0.4.2] - 2026-01-05
### Merged PRs
- Chore(field): Make `BinomialExtensionField::new` public (#1209) (Adrian Hamelink)
- Chore(field): revert making `BinomialExtensionField::new` public and replace with `From<[A; D]>` (#1210) (Adrian Hamelink)
- Refactor(field): Add packed field extraction helpers and FieldArray utilities (#1211) (Adrian Hamelink)
- Refactor: add public const `new` and `new_array` for all fields (#1222) (Adrian Hamelink)
- Feat: use compile-time asserts for const generic parameters (#1232) (Himess)

### Authors
- Adrian Hamelink
- Himess

## [0.4.1] - 2025-12-18
### Merged PRs
- fix: remove undefined WIDTH const in interleave module (#1199) (Robin Salen)

## [0.4.0] - 2025-12-12
### Merged PRs
- Field.rs: `Powers::packed_collect_n` (#888) (Adrian Hamelink)
- Clippy wants us to put things inside of fmt now instead of just extra arguments... (#916) (AngusG)
- Chore: add descriptions to all sub-crate manifests (#906) (Himess)
- From_biguint method for Bn254 (#914) (AngusG)
- More Clippy Complaints (#931) (AngusG)
- Adding Macros to remove boilerplate impls (#943) (AngusG)
- Combining Interleave Code (#950) (AngusG)
- Add a macro for implying PackedValue for PackedFields (#949) (AngusG)
- Fast Octic inverse (#955) (AngusG)
- Fast Optic Square (#957) (AngusG)
- Fast Octic Multiplication (#956) (AngusG)
- Packing Trick for Field Extensions (#958) (AngusG)
- Refactor to packed add methods (#972) (AngusG)
- Speed up Extension Field Addition (#980) (AngusG)
- Remove Nightly Features (#932) (AngusG)
- Move halve to ring (#969) (AngusG)
- Packed Sub Refactor (#979) (AngusG)
- Move div_2_exp_u64 to ring (#970) (AngusG)
- Speed Up Extension Field Subtraction (#988) (AngusG)
- Must Use (#996) (AngusG)
- Move Interleave into the Packed submodule (#997) (AngusG)
- Speed Up Base-Extension Multiplication (#998) (AngusG)
- Compile Time asserts  (#1015) (AngusG)
- Rename frobenius_inv -> pseudo_inv and fix doc (#1049) (Tom Wambsgans)
- Fix: Clarify NEON transmute conversions (#1073) (Skylar Ray)
- Refactor: remove redundant clones in crypto modules (#1080) (Skylar Ray)
- Clippy: small step (#1102) (Thomas Coratger)
- Clippy: add nursery (#1103) (Thomas Coratger)
- Clippy: add semicolon_if_nothing_returned (#1107) (Thomas Coratger)
- Challenger: add `observe_base_as_algebra_element ` to `FieldChallenger` trait (#1152) (Thomas Coratger)
- Feat: revert `builder.assert_bool` to previous impl (#1191) (Zach Langley)

### Authors
- Adrian Hamelink
- AngusG
- Himess
- Robin Salen
- Skylar Ray
- Thomas Coratger
- Tom Wambsgans
- Zach Langley

