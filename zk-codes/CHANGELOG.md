# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]
## [0.8.0] - 2026-09-23
### Merged PRs
- Ci: fail on unused dependencies, and drop the ones already there (#2029)
- Fix(sumcheck): require a cryptographic RNG for HVZK mask sampling (#2260)

## [0.7.0] - 2026-09-04
### Merged PRs
- Feat(binary-dft): additive NTT and the Encoder abstraction (#2003)

## [0.6.0] - 2026-06-11
### Merged PRs
- Feat: extract ZK encoding traits to p3-zk-codes (#1601)
- Chore: use T::zero_vec(n) instead of vec![T::ZERO; n] (#1633)
- Fix: local refs for dev-deps (#1663)
- Fix awkward specialization (#1675)
- Feat(whir): HVZK sumcheck suffix-binding prover (#1665)
- Whir: add HVZK-WHIR hiding polynomial commitment scheme (#1767)

