# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]
## [0.6.0] - 2026-06-11
### Merged PRs
- Matrix: add pad_to_min_power_of_two_height (#1448)
- Matrix: add `from_flat_padded` and `widen_right` (#1449)
- Matrix: implement BitReversibleMatrix for wrapper types (#1456)
- Refactor(matrix): move column-appending helpers to `DenseMatrix` methods (#1533)
- Refactor(interpolation): replace free functions with Interpolate extension trait (#1540)
- Perf(interpolation): optimize barycentric weights via algebraic identity (#1553)
- Feat(interpolation): add arbitrary-point Lagrange interpolation trait (#1552)
- `fix(matrix): interpolate_arbitrary_point honors documented duplicate-domain contract` (#1627)
- Chore: use T::zero_vec(n) instead of vec![T::ZERO; n] (#1633)
- Fix: local refs for dev-deps (#1663)
- Ci: tighten doc/release/TOML checks (#1689)
- Feat: add a `HornerIter` supertrait on `DoubleEndedIterator` (#1692)
- Fix(matrix): handle on-domain point in interpolate_coset (#1702)
- Perf(matrix): specialize dense vertical packing (#1712)
- Fix(matrix): reject inverted column ranges in HorizontallyTruncated (#1749)
- Fix(matrix): correct strided view height for offset >= stride (#1750)
- Chore: update CHANGELOGs (#1785)
- Doc: add basic READMEs in main crates (#1786)

## [0.5.3] - 2026-05-15
## [0.5.2] - 2026-03-27
## [0.5.1] - 2026-03-16
## [0.5.0] - 2026-03-10
### Merged PRs
- Feat(matrix): add `columnwise_dot_product_batched` for multi-point evaluation (#1225)
- Transpose: use p3-util rectangular transposition everywhere (#1256)
- Perf: optimize rowwise dot-product for `Matrix` (#1284)
- Fix(matrix): correct idx initialization in FlatMatrixView::row_subseq_unchecked (#1286)
- Deps: update rand and rand_xoshiro (#1314)
- Matrix: remove spurious extra element from `padded_horizontally_packed_row` (#1309)
- Air: change return type of preprocessed in air builder (#1387)
- Poseidon1: packed form for monty31 (#1378)

## [0.4.2] - 2026-01-05
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

