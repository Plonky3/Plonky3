# p3-binary-pcs

A multilinear polynomial commitment scheme over the binary tower, committing via an
additive-domain Reed–Solomon code (the Cantor domain from `p3-binary-dft`) and proving
proximity BaseFold-style, folding the codeword in lockstep with a multilinear sumcheck. See
Diamond, Posen, *Succinct Arguments over Towers of Binary Fields* (Binius),
<https://eprint.iacr.org/2023/1784>, and Diamond, Posen, *Polylogarithmic Proofs for
Multilinears over Binary Towers* (FRI-Binius, ring switching),
<https://eprint.iacr.org/2024/504>. Parameters are derived only in the unique-decoding regime.
The capacity bound is refuted over characteristic 2 with `F_2`-subspace domains, and the Cantor
domain is one. The classical Johnson list-decoding radius remains unconditional; using it in a
folding argument additionally needs the proven Reed--Solomon mutual-correlated-agreement bound.
This BaseFold implementation nevertheless stays in unique decoding by design; the WHIR adapter
below also accepts the Johnson regime.

The committed alphabet and the challenge field are separate choices. Columns and the base
codeword live in the alphabet; every challenge, every folded codeword and every claimed value
live in the challenge field. A narrower alphabet halves the bytes of the largest Merkle tree in
the proof without moving a single soundness term, because every error is charged against the
challenge field's width. Two constraints follow from the alphabet alone: the grinding witness is
one of its elements, so its width caps the difficulty, and the base codeword lives on its
additive domain, so `num_variables + log_inv_rate` must not exceed its bit width.

Two obligations the types do not carry:

- **Binding, not hiding.** The final codeword travels in the clear and every query opening is a
  raw codeword symbol, so this must not be used where zero-knowledge is required.
- **Collision resistance is the caller's.** The derived schedule prices the field's width and
  the query count, never the Merkle tree it is paired with.

`BinaryPcsConfig` delegates its algebraic budget to `p3_security::binary::BinaryPcsRegime`.
The target covers the sum of opening-claim batching, every fold and sumcheck round, and
query error. Query grinding applies only to queries; it runs after the alpha challenge,
every fold, and absorption of the entire final codeword. This final-codeword binding changes
the Fiat–Shamir transcript: proofs generated without it are not guaranteed to verify, although
the proof's serialization layout is unchanged.
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

## WHIR over the additive domain

The `whir` module connects WHIR to the Cantor additive code. Commitments use
the 64-bit polynomial-basis field. Challenges use its 192-bit cubic extension.

Queries follow a transcript-bound stratified schedule. Use
`recommended_cap_height` to stop Merkle paths at its deepest stratum.

Unique decoding and the Johnson regime use proven bounds. The capacity regime
is unsupported because its assumption is refuted for this domain.

This adapter is binding, not hiding.

Part of [Plonky3](https://github.com/Plonky3/Plonky3), dual-licensed under MIT and Apache 2.0.
