# p3-matrix

A matrix library for finite-field elements, centered on the `Matrix` trait
and the dense row-major implementation `RowMajorMatrix` used to store
execution traces and LDEs.

Key items:

- `Matrix` — the core trait, with packed-row access for SIMD-friendly iteration
- `dense` — `RowMajorMatrix` and borrowed/mutable views
- `bitrev`, `row_index_mapped`, `stack`, `strided`, `horizontally_truncated` — lazy index-remapping and composition wrappers
- `extension`, `interpolation`, `util` — extension-field flattening and evaluation helpers

Part of [Plonky3](https://github.com/Plonky3/Plonky3), dual-licensed under MIT and Apache 2.0.
