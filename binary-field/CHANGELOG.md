# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]
## [0.8.0] - 2026-09-23
### Merged PRs
- Perf(binary-field): Karatsuba fallback, vector-native GF(2^128) mul, Frobenius conjugation, table squaring (#2016)
- Ci: fail on unused dependencies, and drop the ones already there (#2029)
- Feat(binary-field): Ghash128, GF(2^128) in the polynomial basis, with SIMD packings (#2030)
- Test(binary-field): check the vector kernel's algebra on every target (#2028)
- Perf(binary-field): squaring, Frobenius, dot products, and buffer reuse (#2042)
- Feat(multi-stark): add binary-field end-to-end example (#2047)
- Chore: expose the parallel feature on every crate that uses rayon (#2049)
- Perf(binary-pcs): expose proving phases and speed ARM basis conversion (#2094)
- Perf(binary-field): vectorize the polynomial-basis butterflies with a split twiddle (#2054)
- Test(binary-field): pin the packed lane backend to the scalar model (#2101)
- Feat(sumcheck)!: drive the quadratic sumcheck transcript through the typed Fiat-Shamir layer (#2081)
- Test(challenger): assert no two protocols share a transcript seed (#2108)
- Review: couple tweaks here and there (#2120)
- Perf(binary-field): convert whole blocks of GF(2^128) through GFNI (#2153)
- Perf(sumcheck): fold pinned zerocheck blocks without multiplying (#2159)
- Feat(binary-field): add bit-sliced GF(2) packings and the square bit transpose (#2161)
- Feat(binary-field)!: add the AES field, GF(2^64) and the Frobenius engine (#2162)
- Perf(sumcheck,multi-stark,binary-field): faster binary PCS openings, zerocheck kernels and tower serialization (#2165)
- Fix(binary-field): run the GHASH inversion chain before deciding on zero (#2179)
- Perf(sumcheck,binary-pcs): run binary PCS sumcheck rounds in Ghash128 and bind long suffix rounds in place (#2174)
- Perf(binary-field): allocate zero vectors of the polynomial-basis fields lazily (#2181)
- Perf(binary-dft): exploit subfield structure in the additive transform and the encoder (#2173)
- Feat(binary-pcs)!: build the small-field commitment path (#2166)
- Perf(binary): speed up the Boolean-committed prover (#2245)
- Feat!: add WHIR over binary additive domains (#2198)
- Perf(binary): reuse the reduction's evaluation, and make the sliced-round count a choice (#2246)
- Perf(binary): reduce early-round work and representation passes (#2286)
- Refactor!: give sumcheck, STIR, WHIR and binary PCS planning, transcript steps and configuration a single owner (#2297)
- Perf(binary)!: speed up the sliced zerocheck, ring switch, PCS commit and opening, and BLAKE3 witness (#2307)

## [0.7.0] - 2026-09-04
### Merged PRs
- Feat(binary-field): GF(2) through GF(2^128) tower fields (#2001)
- Feat(binary-dft): additive NTT and the Encoder abstraction (#2003)
- Feat(sumcheck): ring switching over an arbitrary field extension (#2006)

