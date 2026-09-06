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

`TowerLevel` exposes the tower structure and typed generator multiplication.

Arithmetic is unpacked: `Packing` is `Self` at every level, so elements are never bit-packed
into SIMD lanes. The polynomial slice kernels process independent SIMD products while
preserving the scalar field layout.

## Hardware builds

Hardware dispatch is selected at compile time, so this `no_std` crate performs no CPU checks
inside scalar arithmetic. `poly_basis::HAS_HARDWARE_CLMUL` describes that compiled choice.
The default `aarch64-apple-darwin` target enables `aes` (including PMULL); generic AArch64 Linux
and baseline x86-64 use the portable tower fallback.

For a binary deployed only to CPUs supporting the named feature:

```sh
RUSTFLAGS="-C target-feature=+aes" cargo build --release --target aarch64-unknown-linux-gnu
RUSTFLAGS="-C target-feature=+pclmulqdq" cargo build --release --target x86_64-unknown-linux-gnu
```

For a local build, `RUSTFLAGS="-C target-cpu=native" cargo build --release` enables the host's
features. Use the baseline target when distributing to heterogeneous CPUs. To exercise the
portable AArch64 path explicitly:

```sh
CARGO_TARGET_DIR=target/portable RUSTFLAGS="-C target-feature=-aes" cargo test -p p3-binary-field --release
```

Part of [Plonky3](https://github.com/Plonky3/Plonky3), dual-licensed under MIT and Apache 2.0.
