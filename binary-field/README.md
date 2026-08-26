# p3-binary-field

The Wiedemann binary tower `GF(2) ⊂ GF(4) ⊂ … ⊂ GF(2^128)`, providing characteristic-2
field arithmetic as a building block toward binary-field SNARKs.

Key items:

- `Gf2` — the base field `GF(2)`
- `BinaryField2`, `BinaryField4`, `BinaryField8`, `BinaryField16`, `BinaryField32`, `BinaryField64`, `BinaryField128` — the tower levels, each a quadratic extension of the one below
- `BasedVectorSpace` / `ExtensionField` between every pair of byte-aligned levels, in the tower basis
- `BinaryChallenger` — Fiat–Shamir over a byte challenger; every bit pattern is a field element, so no rejection sampling is needed
- Carryless-multiply fast paths for `BinaryField64` and `BinaryField128` on x86-64 (`pclmulqdq`) and AArch64 (`aes`)

Little-endian targets only. The tower-basis coefficients of an element are borrowed in place
as a slice of the level below, which is sound only where the byte layout coincides with the
numeric representation; a compile-time assertion rejects big-endian targets.

`from_u64` and the other `PrimeCharacteristicRing` integer constructors go through the prime
subfield `GF(2)`, so they carry the parity of their argument rather than its bit pattern:
`from_u64(2)` is zero. `from_le_bytes` and `interpolation_node` are the bit-pattern
constructors.

Two deliberate scope boundaries:

- `TowerLevel` — the trait carrying `LOG_BITS`, `from_repr`, `to_repr` and `mul_alpha` — is
  `pub(crate)`, keeping the public surface minimal. Downstream crates work through the
  `p3-field` traits instead; exporting it later is an additive change.
- Arithmetic is unpacked: `Packing` is `Self` at every level, so elements are never bit-packed
  into SIMD lanes. A packed representation is a separate design.

Part of [Plonky3](https://github.com/Plonky3/Plonky3), dual-licensed under MIT and Apache 2.0.
