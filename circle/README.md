# p3-circle

A STARK framework over the unit circle of a finite field, following the
[Circle STARKs paper](https://eprint.iacr.org/2024/278) by Haböck, Levit
and Papini. This enables Mersenne-31, which has no large two-adic
multiplicative subgroup, to be used as a STARK field.

Key items:

- `CirclePcs` — the `p3_commit::Pcs` instantiation over circle domains
- `CircleDomain`, `CircleEvaluations` — circle-group evaluation domains and the circle FFT (`cfft`)
- DEEP quotients and circle-specific FRI folding for the opening argument

Part of [Plonky3](https://github.com/Plonky3/Plonky3), dual-licensed under MIT and Apache 2.0.
