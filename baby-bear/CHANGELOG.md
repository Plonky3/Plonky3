# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]
## [0.4.0] - 2025-12-12
### Merged PRs
- Chore: add descriptions to all sub-crate manifests (#906) (Himess)
- GCD based inversion for 31 bit fields (#921) (AngusG)
- Adding Degree 8 extensions for KoalaBear and BabyBear. (#954) (AngusG)
- Packing Trick for Field Extensions (#958) (AngusG)
- Refactor to packed add methods (#972) (AngusG)
- Remove Nightly Features (#932) (AngusG)
- Move div_2_exp_u64 to ring (#970) (AngusG)
- Speed Up Base-Extension Multiplication (#998) (AngusG)
- Generic Poseidon2 Simplifications (#987) (AngusG)
- Poseidon2: add Neon implementation for Monty31 (#1023) (Thomas Coratger)
- Monty31: add aarch64 neon custom `exp_5` and `exp_7` (#1033) (Thomas Coratger)
- Fix: remove unused alloc::format imports (#1066) (Skylar Ray)
- Monty 31: more efficient aarch64 neon `quartic_mul_packed` (#1060) (Thomas Coratger)
- Refactor: remove redundant clones in crypto modules (#1080) (Skylar Ray)
- Refactor: remove redundant clones in crypto modules (#1086) (Skylar Ray)
- Clippy: small step (#1102) (Thomas Coratger)
- Feat: add thread safety to dft implementations (#999) (Jeremi Do Dinh)
- Clippy: add semicolon_if_nothing_returned (#1107) (Thomas Coratger)
- Refactor: deduplicate field JSON serialization tests (#1162) (andrewshab)
- Implement uniform sampling of bits from field elements (#1050) (Sebastian)

### Authors
- AngusG
- Himess
- Jeremi Do Dinh
- Sebastian
- Skylar Ray
- Thomas Coratger
- andrewshab

