# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed

- **Breaking:** Multi-STARK proofs carry an optional binary-bus section.
- **Breaking:** `MultiStarkShape::new` takes a `has_bus` flag and `MultiStarkShape` gains a `has_bus` field.
- **Breaking:** `VerificationError` and `ProvingError` gain binary-bus variants, and `BusBindingError` is re-exported.
- Regenerated the pinned WHIR proof fixture for the new proof shape.

## [0.7.0] - 2026-09-04
### Merged PRs
- Test(multi-stark): add WHIR proof-serialization compat fixture (#1996)
- Feat(field): characteristic-agnostic groundwork for binary fields (#2000)
- Feat: adapt log levels of some inner functions (#1999)
