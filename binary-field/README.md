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
- `HasSubfield<BinaryField2>` for `BinaryField128` — `GF(4)` as a subfield of `GF(2^128)`, with a slice membership test and `GF(4)` scalars applied without a product
- `Ghash128` — `GF(2^128)` in the GHASH polynomial basis, with `From` conversions to and from the widest tower level, and an `Algebra<BinaryField128>` action whose results stay in the polynomial basis
- `PackedGhash128` — the SIMD packing, two elements per register on `avx2` and four on `avx512f`,
  in both cases only when `vpclmulqdq` is also enabled, so it is absent from the rendered docs
- `Poly64` — `GF(2^64)` in the polynomial basis of `x^64 + x^4 + x^3 + x + 1`
- `Poly192` — `GF(2^192)` as the cubic extension `y^3 + y + 1` of `Poly64`
- `PackedPoly64` — the packing of `Poly64`, four elements per 256-bit register, under the same
  `vpclmulqdq` condition
- `PackedPoly192` — the extension packing of `Poly192` over `PackedPoly64`, one register per
  coordinate, under the same condition
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

`TowerLevel` exposes the tower structure and typed generator multiplication.

The tower levels are unpacked: `Packing` is `Self` at every one of them, because a tower
product is table lookups that no vector unit widens.

The polynomial-basis fields pack wherever `vpclmulqdq` widens the multiply:

- `Ghash128` packs one element per 128-bit lane.
- `Poly64` packs one element per quadword, so a product is two carryless multiplies per register.
- `Poly192` packs over `Poly64` with one register per coordinate, so twelve carryless multiplies
  make four products.

`GF(2^64)` stays at 256 bits even with `avx512f`.
A prover keeps one packed value per trace column, and a 512-bit `GF(2^192)` value is 192 bytes.
Measured end to end on Zen 5, the smaller footprint beats twice the products per instruction.

Reductions modulo the `GF(2^64)` polynomial use shifts and one byte shuffle, never a multiply.
Dot products at every level accumulate unreduced products and reduce the sum once.
Without a wide carryless multiply every one of these fields is its own packing.

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

`Poly64` and `Poly192` multiplication, squaring and inversion use no operand-indexed tables.
With `gfni`, `avx512f`, `avx512bw` and `avx512vbmi`, each run of squarings in the `GF(2^64)`
inversion chain is one bit-matrix product on the byte-affine instruction.
Its matrices are compile-time constants read whole, so the chain stays constant-time.

Hardware dispatch is selected at compile time, so this `no_std` crate performs no CPU checks
inside scalar arithmetic. `poly_basis::HAS_HARDWARE_CLMUL` describes that compiled choice.
The default `aarch64-apple-darwin` target enables `aes` (including PMULL); generic AArch64 Linux
and baseline x86-64 use the portable tower fallback.

For a binary intended for the build machine, enable its instructions with:

```sh
RUSTFLAGS="-C target-cpu=native" cargo test -p p3-binary-field -p p3-binary-dft
```

For a binary deployed only to CPUs supporting the named feature:

```sh
RUSTFLAGS="-C target-feature=+aes" cargo build --release --target aarch64-unknown-linux-gnu
RUSTFLAGS="-C target-feature=+pclmulqdq" cargo build --release --target x86_64-unknown-linux-gnu
```

Use the baseline target when distributing to heterogeneous CPUs. To exercise the portable
AArch64 path explicitly:

```sh
CARGO_TARGET_DIR=target/portable RUSTFLAGS="-C target-feature=-aes" cargo test -p p3-binary-field --release
```

Part of [Plonky3](https://github.com/Plonky3/Plonky3), dual-licensed under MIT and Apache 2.0.
