# p3-dft

Discrete Fourier transform implementations over finite fields, used for
low-degree extensions of trace matrices.

Key items:

- `TwoAdicSubgroupDft` — the core trait: (inverse) DFTs and coset LDEs over two-adic subgroups, batched column-wise over matrices
- `Radix2Dit`, `Radix2DitParallel`, `Radix2Bowers`, `Radix2DFTSmallBatch` — radix-2 variants with different parallelism and cache trade-offs
- `NaiveDft` — a quadratic reference implementation for testing

Part of [Plonky3](https://github.com/Plonky3/Plonky3), dual-licensed under MIT and Apache 2.0.
