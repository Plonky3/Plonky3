# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]
## [0.4.2] - 2026-01-05
### Merged PRs
- Refactor(field): Add packed field extraction helpers and FieldArray utilities (#1211) (Adrian Hamelink)
- Enable ZK for preprocessing and in batch-stark (#1178) (Linda Guiga)
- Small changes for recursive lookups (#1229) (Linda Guiga)
- Avoid change of Pcs's `open` method signature (#1230) (Linda Guiga)

### Authors
- Adrian Hamelink
- Linda Guiga

## [0.4.1] - 2025-12-18
### Authors

## [0.4.0] - 2025-12-12
### Merged PRs
- Rename multi-stark crate to batch-stark (#1122) (Sai)
- Add preprocessed/transparent columns to uni-stark (#1114) (o-k-d)
- Challenger: add `observe_base_as_algebra_element ` to `FieldChallenger` trait (#1152) (Thomas Coratger)
- Add preprocessed column support in batch-STARK  (#1151) (Sai)
- Update lookup traits and add folders with lookups (#1160) (Linda Guiga)
- Derive Clone for PreprocessedInstanceMeta (#1166) (Linda Guiga)
- Clarify quotient degree vs quotient chunks naming (#1156) (Sai)
- Doc: add intra-doc links (#1174) (Robin Salen)
- Integrate lookups to prover and verifier (#1165) (Linda Guiga)
- Core: small touchups (#1186) (Thomas Coratger)
- Feat: add PoW phase for batching in FRI commit phase (#1164) (Zach Langley)
- Implement uniform sampling of bits from field elements (#1050) (Sebastian)

### Authors
- Linda Guiga
- Robin Salen
- Sai
- Sebastian
- Thomas Coratger
- Zach Langley
- o-k-d

