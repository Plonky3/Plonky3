# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]
## [0.6.0] - 2026-06-11
### Merged PRs
- Chore: more const assertions (#1441)
- Perf: abstract away Copy vs Clone (#1463)
- Feat: expose Poseidon2 large instances (#1508)
- Core: couple more compile time assertions (#1525)
- Harden(poseidon1, poseidon2): add paper-derived parameter assertions to constructors (#1551)
- Bench: route Goldilocks packed benches via HashPackedGoldilocks (#1564)
- Fix(bn254): add Poseidon2 round-number constants and fix benchmark (#1557)
- Perf(goldilocks): scalar add/sub for PackedGoldilocksNeon (#1619)
- Chore: update CHANGELOGs (#1785)

## [0.5.3] - 2026-05-15
### Merged PRs
- Core: couple more compile time assertions (#1525)

## [0.5.2] - 2026-03-27
### Merged PRs
- Perf: abstract away Copy vs Clone (#1463)

## [0.5.1] - 2026-03-16
## [0.5.0] - 2026-03-10
### Merged PRs
- Deps: update rand and rand_xoshiro (#1314)
- Feat: support for 4-to-1 Poseidon2 instantiations for 32-bit fields (#1359)
- Update external.rs (#1371)
- Poseidon1: fix round constants for KoalaBear and BabyBear (#1398)
- Naming: agree on convention for original Poseidon permutation (#1417)
- Poseidon: add Rust constants for rounds (#1416)

## [0.4.2] - 2026-01-05
### Authors

## [0.4.1] - 2025-12-18
### Authors

## [0.4.0] - 2025-12-12
### Merged PRs
- Porting BN254 to our own code base (#913) (AngusG)
- Chore: add descriptions to all sub-crate manifests (#906) (Himess)
- Poseidon: make ExternalLayerConstants new const (#968) (Thomas Coratger)
- Chore: small touchups and poseidon external unit tests (#971) (Thomas Coratger)
- Remove Nightly Features (#932) (AngusG)
- Packed Sub Refactor (#979) (AngusG)
- Generic Poseidon2 Simplifications (#987) (AngusG)
- Clippy: small step (#1102) (Thomas Coratger)
- Clippy: add semicolon_if_nothing_returned (#1107) (Thomas Coratger)
- Clippy: add `needless_pass_by_value` (#1112) (Thomas Coratger)

### Authors
- AngusG
- Himess
- Thomas Coratger

