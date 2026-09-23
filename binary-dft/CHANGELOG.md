# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]
## [0.8.0] - 2026-09-23
### Merged PRs
- Perf(binary-dft): chain butterfly twiddles instead of rebuilding each one (#2017)
- Refactor(binary-dft): hoist the twiddle increments out of the stage loop (#2027)
- Feat(binary-field): Ghash128, GF(2^128) in the polynomial basis, with SIMD packings (#2030)
- Perf(binary-dft): incremental twiddles, tiled stages, and fused kernels (#2043)
- Perf(binary-dft): fuse long-stride stages through a staging tile (#2052)
- Ci: reduce exhaustive test runtime (#2099)
- Perf(ci): speed up slow test-suite tests (#2118)
- Feat(pcs)!: bind commitments and hiding claims through the typed layer (#2129)
- Perf(binary-field): convert whole blocks of GF(2^128) through GFNI (#2153)
- Perf(binary-dft): give the generic additive transform a tiled schedule (#2156)
- Perf(binary-dft)!: scale by the twiddle's own subfield (#2160)
- Perf(sumcheck,multi-stark,binary-field): faster binary PCS openings, zerocheck kernels and tower serialization (#2165)
- Perf(binary-dft): gather runs of adjacent rows into the staging tile (#2155)
- Perf(binary-dft): exploit subfield structure in the additive transform and the encoder (#2173)
- Feat(examples): merge the binary-hash examples and expose NTT/representation choice (#2185)
- Feat(binary-pcs)!: build the small-field commitment path (#2166)
- Feat!: add WHIR over binary additive domains (#2198)
- Perf(binary-dft): one-byte subfield butterfly on NEON (#2222)
- Perf(binary-dft): deepen the polynomial-basis tiles past the shared cache (#2252)
- Perf(sumcheck): commit the suffix layout straight from the source tables (#2268)
- Refactor!: give sumcheck, STIR, WHIR and binary PCS planning, transcript steps and configuration a single owner (#2297)
- Perf(binary)!: speed up the sliced zerocheck, ring switch, PCS commit and opening, and BLAKE3 witness (#2307)

## [0.7.0] - 2026-09-04
### Merged PRs
- Feat(binary-dft): additive NTT and the Encoder abstraction (#2003)

