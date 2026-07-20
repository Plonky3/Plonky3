# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]
## [0.6.2] - 2026-07-20
### Merged PRs
- Bench(sumcheck): add generic Criterion benchmark suite (#1835)
- Docs(sumcheck): document Fiat-Shamir binding contract on Constraint::new (#1838)
- Perf(sumcheck): reuse Poly::unpack for packed-to-scalar conversion (#1844)
- Docs(sumcheck): fix inverted batch_pows ordering and complete SelectStatement panics doc (#1859)
- Perf(sumcheck): hoist invariant alpha exponentiation out of the SVO round loop (#1861)
- Fix(sumcheck): assert accumulator length in scalar combine paths (#1860)
- Perf(whir): commit folded rounds straight from the live sumcheck buffer (#1804)
- Fix: doc and light tweaks (#1920)

## [0.6.1] - 2026-06-13
## [0.6.0] - 2026-06-11
### Merged PRs
- Refactor: move sumcheck to an independent crate (#1672)
- Perf(whir): use collect_n for select challenge powers (#1681)
- Perf: use shifted_powers + collect_n for combine (#1682)
- Perf: extend collect_n and shifted_powers to remaining call sites (#1683)
- Ci: tighten doc/release/TOML checks (#1689)
- Feat: add a `HornerIter` supertrait on `DoubleEndedIterator` (#1692)
- Fix: sample HZVK masks from EF (#1726)
- Fix: require ell_zk >= 3 in HVZK sumcheck (#1727)
- Feat(whir): HVZK sumcheck suffix-binding prover (#1665)
- Perf: reduce some allocations on WHIR (#1729)
- Sumcheck: type zk handoff and residual claim producer (#1732)
- Perf(whir): avoid extra allocation in prefix-order commit encoding (#1743)
- Whir: add HVZK-WHIR hiding polynomial commitment scheme (#1767)
- Refactor: remove unwraps in verifier path (#1788)

