# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]
## [0.3.1] - 2025-12-08
### Merged PRs
- More Customizable Benchmarks (#576) (AngusG)
- Better error management (#602) (François Garillot)
- Adding m31 to customizable examples (#590) (AngusG)
- Correct `num_hashes` computation for changing `P2_VECTOR_LEN` (#623) (Georg Wiese)
- Add extra_capacity_bits argument to all trace generation. (#634) (AngusG)
- Update rand crate dependency (#653) (AngusG)
- Add proof size report in examples (#660) (Sai)
- Core: some improvements (#669) (Thomas Coratger)
- Update Rust Version to 2024 (#685) (AngusG)
- Updates to Bincode crate for the 2.0.0 release. (#726) (AngusG)
- Replace `std::` with `core::` where possible. (#741) (Hamish Ivey-Law)
- Remove `thread_rng` dependency (#740) (Hamish Ivey-Law)
- Rm unused deps (#760) (Thomas Coratger)
- Clean unused deps (#774) (taikoon)
- Remove unused p3-mds dependency from all Cargo.toml (#784) (taikoon)
- Adding some end to end tests (#795) (AngusG)
- Initialize Challenger through Stark Config (#793) (AngusG)
- Support Hashing Vector Space Elements #2 (#775) (AngusG)
- Bump and release automation (#546) (BGluth)
- [CI] Introduce `cargo-sort` and remove unused dependencies (#894) (Adrian Hamelink)
- Leftover automated build cleanup (#895) (BGluth)
- Rm uneeded twoadics (#892) (AngusG)
- Rename FriConfig/FriGenericConfig (#891) (AngusG)
- Added placeholder descriptions to all plonky3 sub-crates (#903) (BGluth)
- Clippy wants us to put things inside of fmt now instead of just extra arguments... (#916) (AngusG)
- Air example: trait bound touchups (#941) (Thomas Coratger)
- Weaken the trait bound of AirBuilder to allow `F` to be merely a Ring. (#977) (AngusG)
- Remove Nightly Features (#932) (AngusG)
- Generic Poseidon2 Simplifications (#987) (AngusG)
- Fixing RecursiveDFT initlialisation (#1022) (AngusG)
- Doc: add better doc in air and fix TODO (#1061) (Thomas Coratger)
- Add SmallBatchDft to example options. (#1074) (AngusG)
- Clippy: small step (#1102) (Thomas Coratger)
- Clippy: add nursery (#1103) (Thomas Coratger)
- Clippy: add semicolon_if_nothing_returned (#1107) (Thomas Coratger)
- Clippy: add `needless_pass_by_value` (#1112) (Thomas Coratger)
- Refactor: Replace &Vec<T> with &[T] in function parameters (#1111) (Merkel Tranjes)

### Authors
- Adrian Hamelink
- AngusG
- BGluth
- François Garillot
- Georg Wiese
- Hamish Ivey-Law
- Merkel Tranjes
- Sai
- Thomas Coratger
- taikoon

