# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]
## [0.4.0] - 2025-12-12
### Merged PRs
- Field.rs: `Powers::packed_collect_n` (#888) (Adrian Hamelink)
- Uni stark: small touchups on the verifier (#910) (Thomas Coratger)
- Clippy wants us to put things inside of fmt now instead of just extra arguments... (#916) (AngusG)
- Chore: add descriptions to all sub-crate manifests (#906) (Himess)
- Fixed "attempt to subtract with overflow" issue in uni-stark (#934) (Gabriel Barreto)
- Replace `Copy` with `Clone` in `AirBuilder`'s `Var` (#930) (Linda Guiga)
- Docs: Add comprehensive documentation to constraint folder implementation (#856) (Ragnar)
- Shrink some test sizes (#524) (Daniel Lubarov)
- Fixing error on main (#939) (AngusG)
- Chore: various small changes (#944) (Thomas Coratger)
- Remove Nightly Features (#932) (AngusG)
- Small visibility changes for recursion (#1046) (Linda Guiga)
- Refactor: remove redundant clones in crypto modules (#1086) (Skylar Ray)
- Add modular lookups (local and global) with logup implementation (#1090) (Linda Guiga)
- Add multi-STARK prover and verifier (#1088) (Sai)
- Clippy: small step (#1102) (Thomas Coratger)
- Clippy: add nursery (#1103) (Thomas Coratger)
- Update symbolic_builder.rs (#1106) (AJoX)
- Clippy: add semicolon_if_nothing_returned (#1107) (Thomas Coratger)
- Clippy: add `needless_pass_by_value` (#1112) (Thomas Coratger)
- Refactor: Replace &Vec<T> with &[T] in function parameters (#1111) (Merkel Tranjes)
- Add preprocessed/transparent columns to uni-stark (#1114) (o-k-d)
- Add Preprocessed trace setup and VKs (#1150) (Sai)
- Update lookup traits and add folders with lookups (#1160) (Linda Guiga)
- ExtensionBuilder for SymbolicAirBuilder (#1161) (Linda Guiga)
- Uni-stark: add unit tests for SymbolicExpression (#1169) (Thomas Coratger)
- Uni stark: small touchups (#1163) (Thomas Coratger)
- Clarify quotient degree vs quotient chunks naming (#1156) (Sai)
- Core: add error messages to error enums via thiserror (#1168) (Thomas Coratger)
- Feat: add `SubAirBuilder` module (#1172) (Robin Salen)
- Doc: add intra-doc links (#1174) (Robin Salen)
- Integrate lookups to prover and verifier (#1165) (Linda Guiga)
- Core: small touchups (#1186) (Thomas Coratger)
- Feat: add PoW phase for batching in FRI commit phase (#1164) (Zach Langley)

### Authors
- AJoX
- Adrian Hamelink
- AngusG
- Daniel Lubarov
- Gabriel Barreto
- Himess
- Linda Guiga
- Merkel Tranjes
- Ragnar
- Robin Salen
- Sai
- Skylar Ray
- Thomas Coratger
- Zach Langley
- o-k-d

