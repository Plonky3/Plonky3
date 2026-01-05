# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]
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

