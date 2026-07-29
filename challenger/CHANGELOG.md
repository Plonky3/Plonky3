# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]
## [0.6.0] - 2026-06-11
### Merged PRs
- Fix: use `u64` arithmetic in 64-bit challenger to avoid truncation (#1482)
- Core: couple more compile time assertions (#1525)
- Fix(whir): use unbiased rejection sampling for STIR query indices (#1598)
- Refactor(challenger): align bits==0 short-circuit across grind impls (#1602)
- Fix: local refs for dev-deps (#1663)
- Couple fixes (#1688)
- Couple miscellaneous tweaks (#1731)
- Fix(challenger): reject absorb length tags that overflow u8 (#1738)
- Fix(challenger): evaluate PoW bit-bound in u64 to close release-mode bypass (#1747)
- Fix(challenger): make duplex sponge absorbs length-binding (#1769)
- Chore: update CHANGELOGs (#1785)
- Doc: add basic READMEs in main crates (#1786)
- Perf(challenger): precompute the grind state for MultiField32Challenger (#1782)

## [0.5.3] - 2026-05-15
### Merged PRs
- Core: couple more compile time assertions (#1525)

## [0.5.2] - 2026-03-27
### Merged PRs
- Fix: use `u64` arithmetic in 64-bit challenger to avoid truncation (#1482)

## [0.5.1] - 2026-03-16
## [0.5.0] - 2026-03-10
### Merged PRs
- Perf: optimize HashChallenger buffering (#1266)
- Refactor: minor code cleanups across blake3-air, bn254, challenger (#1287)
- Feat: add Merkle Caps (#1321)
- Fix: restrict MultiField32Challenger output to rate portion only (#1299)
- Feat: add `CanFinalizeDigest` trait for challenger transcript commitment (#1409)

## [0.4.2] - 2026-01-05
### Merged PRs
- SIMD optimization for proof-of-work grinding in DuplexChallenger (#1208) (Utsav Sharma)

### Authors
- Utsav Sharma

## [0.4.1] - 2025-12-18
### Authors

## [0.4.0] - 2025-12-12
### Merged PRs
- Chore: add descriptions to all sub-crate manifests (#906) (Himess)
- Add a comment about non-uniformity in `CanSampleBits` (#1026) (Tom Wambsgans)
- Clippy: small step (#1102) (Thomas Coratger)
- Clippy: add semicolon_if_nothing_returned (#1107) (Thomas Coratger)
- Challenger: add `observe_base_as_algebra_element ` to `FieldChallenger` trait (#1152) (Thomas Coratger)
- Challenger: add unit tests for `observe_base_as_algebra_element` (#1155) (Thomas Coratger)
- Challenger: add `observe_algebra_elements` method (#1176) (Thomas Coratger)
- Feat: add PoW phase for batching in FRI commit phase (#1164) (Zach Langley)
- Implement uniform sampling of bits from field elements (#1050) (Sebastian)

### Authors
- Himess
- Sebastian
- Thomas Coratger
- Tom Wambsgans
- Zach Langley

