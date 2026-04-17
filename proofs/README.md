# Formal proofs

Machine-checked proofs of selected Plonky3 analytical claims.  Each file is a standalone Lean 4 module verifiable against a current Mathlib checkout.

## Contents

- `sample_bits_uniformity.lean`: uniformity and rejection bound for the `sample_bits` rejection sampler landed in [#1050], resolving the quantitative half of [#613].  Exact conditional uniformity on acceptance and a sharp bound on the rejection count.  Zero `sorry`, zero `admit`.

[#1050]: https://github.com/Plonky3/Plonky3/pull/1050
[#613]: https://github.com/Plonky3/Plonky3/issues/613

## Verifying

One way, from the repo root:

```bash
lake new plonky3_proof math
cp proofs/sample_bits_uniformity.lean plonky3_proof/Plonky3Proof.lean
cd plonky3_proof && lake build
```

Or paste the file into <https://live.lean-lang.org> with Mathlib loaded.

Toolchain used: Lean 4.30.0-rc1 with Mathlib commit c290b55.  Newer versions should work provided Mathlib tracks them.

Leaving the directory flat keeps the diff small.  Happy to wire it up as a proper Lake project with pinned Mathlib and a CI workflow if that is preferred.
