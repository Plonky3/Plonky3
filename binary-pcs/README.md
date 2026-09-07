# p3-binary-pcs

A multilinear polynomial commitment scheme over `BinaryField128`, committing via an
additive-domain Reed–Solomon code (the Cantor domain from `p3-binary-dft`) and proving
proximity BaseFold-style, folding the codeword in lockstep with a multilinear sumcheck. See
Diamond, Posen, *Succinct Arguments over Towers of Binary Fields* (Binius),
<https://eprint.iacr.org/2023/1784>, and Diamond, Posen, *Polylogarithmic Proofs for
Multilinears over Binary Towers* (FRI-Binius, ring switching),
<https://eprint.iacr.org/2024/504>. Parameters are derived only in the unique-decoding regime.
The capacity bound is refuted over characteristic 2 with `F_2`-subspace domains, and the Cantor
domain is one. The Johnson bound is not refuted — it is an unconditional theorem whose radius
those same counterexamples show to be tight — but `p3-security` documents it as resting on a
correlated-agreement conjecture, and it is excluded here by choice, not by mathematics.

Part of [Plonky3](https://github.com/Plonky3/Plonky3), dual-licensed under MIT and Apache 2.0.
