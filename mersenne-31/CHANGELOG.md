# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]
## [0.4.2] - 2026-01-05
### Merged PRs
- Refactor: add public const `new` and `new_array` for all fields (#1222) (Adrian Hamelink)

### Authors
- Adrian Hamelink

## [0.4.1] - 2025-12-18
### Merged PRs
- mersenne 31: optimize Poseidon2 for aarch64 Neon (#1196) (Thomas Coratger)

## [0.4.0] - 2025-12-12
### Merged PRs
- Chore: add descriptions to all sub-crate manifests (#906) (Himess)
- GCD based inversion for 31 bit fields (#921) (AngusG)
- Fixing a pair of clippy complaints in AVX512 (#926) (AngusG)
- More Clippy Complaints (#931) (AngusG)
- Packing: small touchups (#937) (Thomas Coratger)
- Use `#[derive(...)]` for Debug and Default for packed fields. (#945) (AngusG)
- Adding Macros to remove boilerplate impls (#943) (AngusG)
- Combining Interleave Code (#950) (AngusG)
- Add a macro for implying PackedValue for PackedFields (#949) (AngusG)
- Chore: use `collect_n` with powers when possible (#963) (Thomas Coratger)
- Packing Trick for Field Extensions (#958) (AngusG)
- Refactor to packed add methods (#972) (AngusG)
- Remove Nightly Features (#932) (AngusG)
- Move halve to ring (#969) (AngusG)
- Packed Sub Refactor (#979) (AngusG)
- Move div_2_exp_u64 to ring (#970) (AngusG)
- Must Use (#996) (AngusG)
- Generic Poseidon2 Simplifications (#987) (AngusG)
- More Const Assert fixes (#1024) (AngusG)
- Perf: optimize ext_two_adic_generator with precomputed table (#1038) (Avory)
- Mersenne-31: Implement NEON-optimized halve for PackedMersenne31Neon (#1054) (VolodymyrBg)
- Clippy: small step (#1102) (Thomas Coratger)
- Clippy: add nursery (#1103) (Thomas Coratger)
- Clippy: add semicolon_if_nothing_returned (#1107) (Thomas Coratger)
- Clippy: add `needless_pass_by_value` (#1112) (Thomas Coratger)
- Fixing a few clippy lints (#1115) (AngusG)
- Fix: Add bounds check to circle_two_adic_generator to prevent underflow (#1130) (Fibonacci747)
- Allow users to impl either permute or permute_mut (#1175) (AngusG)
- Implement uniform sampling of bits from field elements (#1050) (Sebastian)

### Authors
- AngusG
- Avory
- Fibonacci747
- Himess
- Sebastian
- Thomas Coratger
- VolodymyrBg

