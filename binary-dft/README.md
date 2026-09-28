# p3-binary-dft

The additive NTT over the Plonky3 binary tower fields, and the Reed-Solomon encoder built on it.

The evaluation domain `S_l` is the span of the first `l` Cantor basis vectors.

It plays the role a multiplicative coset plays for a prime-field DFT.

## Transforms

- `NaiveAdditiveNtt` evaluates the definition directly, as the oracle the fast transforms are tested against.
- `LchNtt` is the Lin-Chung-Han transform, for every tower level and representation.
- `PolyBasisNtt` runs that transform on `GF(2^128)` in its polynomial basis, where a carryless multiply is cheap.

## Kernels

- Twiddles in a small tower subfield scale byte by byte, with GFNI on x86 and nibble tables on NEON.
- `Poly64` multiplies a whole register per carryless multiply, with AVX-512 or AVX2 and VPCLMULQDQ.
- Every kernel keeps a portable fallback, and tests pin each one to it.

Part of [Plonky3](https://github.com/Plonky3/Plonky3), dual-licensed under MIT and Apache 2.0.
