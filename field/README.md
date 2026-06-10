# p3-field

A modular framework for finite fields: core field traits, generic binomial
extension fields, and SIMD-packed field arithmetic.

Key items:

- `PrimeCharacteristicRing`, `Algebra`, `Field`, `PrimeField`, `TwoAdicField` — the core algebraic trait hierarchy
- `ExtensionField`, `BasedVectorSpace` and the `extension` module — generic binomial extension fields
- `PackedField` / `PackedValue` — SIMD-packed arithmetic abstractions
- `coset`, `exponentiation`, batch inversion and dot-product helpers

Concrete field implementations live in their own crates (`p3-baby-bear`,
`p3-koala-bear`, `p3-goldilocks`, `p3-mersenne-31`, `p3-bn254`).

Part of [Plonky3](https://github.com/Plonky3/Plonky3), dual-licensed under MIT and Apache 2.0.
