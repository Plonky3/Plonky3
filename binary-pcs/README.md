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

Two obligations the types do not carry:

- **Binding, not hiding.** The final codeword travels in the clear and every query opening is a
  raw codeword symbol, so this must not be used where zero-knowledge is required.
- **Collision resistance is the caller's.** The derived schedule prices the field's width and
  the query count, never the Merkle tree it is paired with.

`BinaryPcsConfig` delegates its algebraic budget to `p3_security::binary::BinaryPcsRegime`.
The target covers the sum of opening-claim batching, every fold and sumcheck round, and
query error. Query grinding applies only to queries; it runs after the alpha challenge.
When every base coset is queried, query error is zero and its reserved budget is released.
`PrescribedPointPcs::prescribed_security` supplies this composed bound and the unique-decoding
candidate bound to security-checked multi-STARK callers, which also need collision evidence.

This tightens previously accepted configurations and can change the derived query count
and proof transcript. Opening protocols must also fit `BinaryPcsConfig::max_opening_claims()`;
each current or successor-column evaluation consumes one claim. The limit is conservative,
not a tight attack threshold. Use `BinaryPcs::validate_opening_protocol` to preflight a
protocol, or `try_open` / `try_open_at` to receive typed errors without changing the challenger
on rejection. The existing infallible `open` / `open_at` traits panic on invalid or
over-budget protocols. Existing callers must handle this new rejection or choose a feasible
target and protocol before opening.

Part of [Plonky3](https://github.com/Plonky3/Plonky3), dual-licensed under MIT and Apache 2.0.
