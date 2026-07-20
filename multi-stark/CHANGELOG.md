# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]
## [0.6.2] - 2026-07-20
### Merged PRs
- Reuse LDE from `LdtBasedPcs` to compute quotient (#120)
- Split Keccak example into a few variants (#154)
- Remove some unnecessary PhantomData (#164)
- Remove separate `Domain` field types (#176)
- Remove unused dependencies (#180)
- Add a `SymbolicAirBuilder` in uni-stark (#185)
- Rename StarkConfig (#237)
- Tentative pcs rework (#253)
- Add multi-STARK prover and verifier (#1088)
- Clippy: small step (#1102)
- Clippy: add nursery (#1103)
- Clippy: add `needless_pass_by_value` (#1112)
- Rename multi-stark crate to batch-stark (#1122)
- Feat(multi-stark): add crate skeleton with boundary selectors and AIR folder (#1700)
- Feat(multi-stark): add config trait and trace commitment (#1872)
- Feat(multi-stark): add constraint metadata module (#1871)
- Feat(multi-stark): AIR zerocheck via generic-degree sumcheck (#1879)
- Perf(multi-stark): tighten AIR zerocheck round-poly and degree (#1885)
- Revert "perf(multi-stark): tighten AIR zerocheck round-poly and degree (#1885)"
- Revert "feat(multi-stark): AIR zerocheck via generic-degree sumcheck (#1879)"

