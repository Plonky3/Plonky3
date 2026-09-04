# p3-binary-field

Characteristic-2 field arithmetic as a building block toward binary-field SNARKs.

The crate carries two representations of `GF(2^128)`, which are the same field seen in two
bases:

- the Wiedemann tower `GF(2) ⊂ GF(4) ⊂ … ⊂ GF(2^128)`, whose levels are subfields of one
  another;
- the polynomial basis of `x^128 + x^7 + x^2 + x + 1`, the GHASH modulus.

Key items:

- `Gf2` — the base field `GF(2)`
- `BinaryField2`, `BinaryField4`, `BinaryField8`, `BinaryField16`, `BinaryField32`, `BinaryField64`, `BinaryField128` — the tower levels, each a quadratic extension of the one below
- `Ghash128` — `GF(2^128)` in the GHASH polynomial basis, with `From` conversions to and from the widest tower level
- `PackedGhash128` — the SIMD packing, holding two elements per register under `AVX2` and four under `AVX-512`
- `BasedVectorSpace` / `ExtensionField` between every pair of byte-aligned tower levels, in the tower basis
- `BinaryChallenger` — Fiat–Shamir over a byte challenger; every bit pattern is a field element, so no rejection sampling is needed
- Carryless-multiply fast paths on x86-64 (`pclmulqdq`, `vpclmulqdq`) and AArch64 (`aes`), with a software backend everywhere else

## Which representation to use

With hardware carryless multiplication, a 128-bit tower product converts both operands into
polynomial coordinates and converts the result back.
Each conversion reads sixteen operand-indexed table entries.

`Ghash128` works directly in polynomial coordinates and avoids those conversions.
Its SIMD packings process two or four independent elements per register.
Dot products accumulate unreduced polynomials and reduce the sum once.

What it gives up is the subfield structure: `GF(2^8)`, `GF(2^16)`, `GF(2^32)` and `GF(2^64)`
are byte-aligned inside the tower representation and are not inside this one.

Code that needs the subfield structure — ring switching, small-field witnesses — wants the
tower. Code that only needs a large binary field — challenges, folding, transforms — wants
`Ghash128`.

## Conventions

Little-endian targets only. The tower-basis coefficients of an element are borrowed in place
as a slice of the level below, which is sound only where the byte layout coincides with the
numeric representation; a compile-time assertion rejects big-endian targets.

`from_u64` and the other `PrimeCharacteristicRing` integer constructors go through the prime
subfield `GF(2)`, so they carry the parity of their argument rather than its bit pattern:
`from_u64(2)` is zero. `from_le_bytes` and `interpolation_node` are the bit-pattern
constructors.

The tower levels are unpacked: `Packing` is `Self` at every one of them, because a tower
product is table lookups that no vector unit widens. Only `Ghash128` has a packing.

GHASH coordinates assign the coefficient of `x^i` to bit `i` of the backing integer.
NIST GCM blocks assign `x^0` to the leftmost bit instead.
For a block written as a big-endian integer, reverse all 128 bits before constructing an element.
The field arithmetic alone is not an AES-GCM implementation.

## Timing and backend selection

GHASH multiplication, squaring, square roots, and dot products use no operand-indexed tables.
Their software multiplication assumes constant-time integer multiplication on the target CPU.

GHASH inversion, tower arithmetic, and conversions between the two bases use operand-indexed tables.
These operations are not constant-time and should not process secrets when cache or timing leakage matters.
Hardware GHASH inversion uses five precomputed maps, totaling 320 KiB of read-only tables.
Software GHASH inversion uses the tower norm instead, and does not compile those tables.

Backends are selected at compile time.
For a binary intended for the build machine, enable its instructions with:

```sh
RUSTFLAGS="-C target-cpu=native" cargo test -p p3-binary-field -p p3-binary-dft
```

A baseline build uses the software backend.

Part of [Plonky3](https://github.com/Plonky3/Plonky3), dual-licensed under MIT and Apache 2.0.
