# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]
## [0.6.0] - 2026-06-11
### Merged PRs
- Multilinear util: integrate poly utils (#1379)
- Multilinear utils: fix some visibility stuffs (#1442)
- Multilinear utils: fix some visibility stuffs (#1443)
- Perf: abstract away Copy vs Clone (#1463)
- Whir: integration of all the required primitives (#1477)
- Whir: stacked sumcheck (#1554)
- Perf simd compress hi dot (#1574)
- Perf(split_eq): SIMD-delayed compress_prefix_to_packed for the Packed eq1 path (#1592)
- Whir: wire stacked layouts to pcs (#1612)
- Chore: use T::zero_vec(n) instead of vec![T::ZERO; n] (#1633)
- Fix: local refs for dev-deps (#1663)
- Perf(multilinear-util): field-aware MLE recursion threshold (#1664)
- Ci: tighten doc/release/TOML checks (#1689)
- Perf(whir): drop redundant leaf clones in fold-query evaluation (#1744)
- Chore(multilinear-util): remove dead compress_suffix_into wrapper (#1753)
- Fix(multilinear-util): correct packed-batch workspace slicing and crossover (#1751)
- Chore: update CHANGELOGs (#1785)

## [0.5.3] - 2026-05-15
## [0.5.2] - 2026-03-27
### Merged PRs
- Perf: abstract away Copy vs Clone (#1463)

## [0.5.1] - 2026-03-16
## [0.5.0] - 2026-03-10
### Merged PRs
- Deps: update rand and rand_xoshiro (#1314)

## [0.4.2] - 2026-01-05
### Authors

## [0.4.1] - 2025-12-18
### Authors

## [0.4.0] - 2025-12-12
### Merged PRs
- Possible different take on PR #973 (#978) (AngusG)
- Multilinear utils: add multilinear point (#1011) (Thomas Coratger)
- Rm Multilinear Point (#1018) (AngusG)
- Fix(multilinear-util): use core::marker::PhantomData in no_std (#1063) (Skylar Ray)
- Eq poly: implement batched eval_eq (#1051) (Thomas Coratger)
- Multilinear utils: rm `eval_eq` (#1087) (Thomas Coratger)
- Clippy: small step (#1102) (Thomas Coratger)
- Clippy: add semicolon_if_nothing_returned (#1107) (Thomas Coratger)

### Authors
- AngusG
- Skylar Ray
- Thomas Coratger

