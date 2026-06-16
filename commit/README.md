# p3-commit

A framework for cryptographic commitment schemes, including non-hiding
variants. This crate defines the traits that connect proof systems to
their commitment backends.

Key items:

- `Pcs` / `MultilinearPcs` — polynomial commitment scheme interfaces used by the STARK provers and verifiers
- `Mmcs` — "Mixed Matrix Commitment Scheme", a vector-commitment abstraction over batches of matrices of differing heights
- `PolynomialSpace` and `TwoAdicMultiplicativeCoset` — evaluation-domain abstractions
- `periodic` — periodic-column evaluation helpers
- `testing` — mock instantiations for downstream tests

Implementations live in `p3-merkle-tree` (Mmcs), `p3-fri`, `p3-circle` and
`p3-whir` (Pcs).

Part of [Plonky3](https://github.com/Plonky3/Plonky3), dual-licensed under MIT and Apache 2.0.
