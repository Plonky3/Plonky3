# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]
## [0.4.3] - 2026-03-10### Merged PRs- Feat(matrix): add `columnwise_dot_product_batched` for multi-point evaluation (#1225)- Transpose: use p3-util rectangular transposition everywhere (#1256)- Perf: optimize rowwise dot-product for `Matrix` (#1284)- Fix(matrix): correct idx initialization in FlatMatrixView::row_subseq_unchecked (#1286)- Deps: update rand and rand_xoshiro (#1314)- Matrix: remove spurious extra element from `padded_horizontally_packed_row` (#1309)- Air: change return type of preprocessed in air builder (#1387)- Poseidon1: packed form for monty31 (#1378)## [0.4.2] - 2026-01-05
### Merged PRs
- Refactor(field): Add packed field extraction helpers and FieldArray utilities (#1211) (Adrian Hamelink)

### Authors
- Adrian Hamelink

## [0.4.1] - 2025-12-18
### Authors

## [0.4.0] - 2025-12-12
### Merged PRs
- Clippy wants us to put things inside of fmt now instead of just extra arguments... (#916) (AngusG)
- From_biguint method for Bn254 (#914) (AngusG)
- More Clippy Complaints (#931) (AngusG)
- Chore: various small changes (#944) (Thomas Coratger)
- Doc: add better doc in air and fix TODO (#1061) (Thomas Coratger)
- Eq poly: implement batched eval_eq (#1051) (Thomas Coratger)
- Clippy: small step (#1102) (Thomas Coratger)
- Clippy: add nursery (#1103) (Thomas Coratger)
- Matrix: make `HorizontallyTruncated` more generic (#1170) (Thomas Coratger)
- Matrix: add `pad_to_power_of_two_height` (#1185) (Thomas Coratger)

### Authors
- AngusG
- Thomas Coratger

