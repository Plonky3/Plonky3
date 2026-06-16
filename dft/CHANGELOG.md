# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]
## [0.6.0] - 2026-06-11
### Merged PRs
- Dft: testing for internals (#1494)
- Perf(dft): butterfly micro-optimizations for Radix2DitParallel (~3% on coset_lde_batch) (#1492)
- Fix(dft): merge twiddle and inv_twiddle locks in Radix2DFTSmallBatch (#1534)
- Bench(dft): use iter_batched to exclude clone cost from measurement (#1575)
- Chore: use T::zero_vec(n) instead of vec![T::ZERO; n] (#1633)
- Perf(dft): generalise coset LDE to support fused quotient commitment (#1621)
- Perf(dft): unroll DIT butterfly inner loops by 4 (#1555)
- Perf: extend collect_n and shifted_powers to remaining call sites (#1683)
- Perf(dft): optimize coset_shift_cols with early return and vectorized scaling (#1468)
- Ci: tighten doc/release/TOML checks (#1689)
- Chore: update CHANGELOGs (#1785)
- Doc: add basic READMEs in main crates (#1786)

## [0.5.3] - 2026-05-15
### Merged PRs
- Perf(dft): butterfly micro-optimizations for Radix2DitParallel (~3% on coset_lde_batch) (#1492)
- Fix(dft): merge twiddle and inv_twiddle locks in Radix2DFTSmallBatch (#1534)

## [0.5.2] - 2026-03-27
## [0.5.1] - 2026-03-16
## [0.5.0] - 2026-03-10
### Merged PRs
- Fix: reduce logging noise in batch-stark with multiple AIRs (#1258)
- Fix(dft): reduce twiddle factor allocation in Radix2Dit (#1285)

## [0.4.2] - 2026-01-05
### Authors

## [0.4.1] - 2025-12-18
### Authors

## [0.4.0] - 2025-12-12
### Merged PRs
- Field.rs: `Powers::packed_collect_n` (#888) (Adrian Hamelink)
- Chore: add descriptions to all sub-crate manifests (#906) (Himess)
- More Clippy Complaints (#931) (AngusG)
- Small refactor trying to clean up #897 (#900) (AngusG)
- Chore: use `collect_n` with powers when possible (#963) (Thomas Coratger)
- Remove Nightly Features (#932) (AngusG)
- Clippy: small step (#1102) (Thomas Coratger)
- Feat: add thread safety to dft implementations (#999) (Jeremi Do Dinh)
- Clippy: add semicolon_if_nothing_returned (#1107) (Thomas Coratger)
- Clippy: add `needless_pass_by_value` (#1112) (Thomas Coratger)
- Implement uniform sampling of bits from field elements (#1050) (Sebastian)

### Authors
- Adrian Hamelink
- AngusG
- Himess
- Jeremi Do Dinh
- Sebastian
- Thomas Coratger

