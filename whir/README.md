# p3-whir

WHIR: Reed–Solomon proximity testing with super-fast verification.

An IOP of proximity for constrained Reed–Solomon codes that serves as a
multilinear polynomial commitment scheme, implementing the
`p3_commit::MultilinearPcs` interface.

A hiding variant lives in the zero-knowledge PCS module: masked sumcheck
batches, HVZK code-switching rounds, and a masked base case compose into a
commitment that reveals only the requested evaluations.

References:

- <https://eprint.iacr.org/2024/1586> (WHIR)
- <https://eprint.iacr.org/2026/391> (HVZK-WHIR)

Part of [Plonky3](https://github.com/Plonky3/Plonky3), dual-licensed under MIT and Apache 2.0.
