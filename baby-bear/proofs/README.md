# p3-baby-bear proofs

This directory houses the formal verification of this crate, in Lean 4.
[`lean/`](lean/) is a Lake package: it extracts `p3-baby-bear` to Lean with
hax's `lean` backend (charon + aeneas, hax 0.4.1) and proves theorems about the
result.

The extraction is scalar-only (`thumbv7em-none-eabi`): AVX2, AVX-512 and NEON
are out of scope. Production on x86-64 and aarch64 runs those backends, not the
portable `no_packing` path modelled here.

What is proved:

- BabyBear's constants (`PRIME`, `TWO_ADICITY`, `MONTY_BITS`, `MONTY_MU`) match
  CompPoly's specification of the field; the modulus is prime, `MONTY_MU` is
  the Montgomery inverse, `x ↦ x^7` is a permutation, and `TWO_ADICITY` is the
  largest power of two dividing `p - 1`.
- `MontyField31::new` never panics on BabyBear and returns `x · 2^32 mod p`;
  the table constructors `new_array` and `new_2d_array` never panic.
- Every Poseidon round-constant table has the length its `const` assertion
  requires.

No field arithmetic (add, mul, Montgomery reduction, inversion) and no
permutation (Poseidon1/2, MDS) is proved anything about.

The generated Lean is not committed. To build, run `./lean/extract.sh`; see
[`lean/README.md`](lean/README.md). What is trusted is in
[`lean/TCB.md`](lean/TCB.md).
