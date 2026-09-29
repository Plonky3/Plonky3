/- Hand-written specifications and their proofs, one file under `Proofs/`
per Rust item. These are properties stated directly in Lean, independent of
any `hax_lib::requires` / `hax_lib::ensures` contracts in the Rust source
(those are proved in `ProofObligations.lean`).

Conventions:
- Name theorems `<item>.<property>`, after the Rust item they are about
  (`BabyBearParameters::PRIME` is `baby_bear.BabyBearParameters.PRIME`).
  Avoid `<fn>.spec`, `.pre` and `.post`, which hax reserves for the
  generated contracts, and `<def>.eq_<n>`, which Lean reserves for equation
  lemmas.
- Each theorem is its own specification: the statement is the claim, and
  the proof follows it.
- `ProofObligations.lean` imports this file; nothing here depends on the
  generated contracts. -/
import P3BabyBear.Verification.Proofs.BabyBearParameters
import P3BabyBear.Verification.Proofs.BabyBear
import P3BabyBear.Verification.Proofs.Poseidon1
import P3BabyBear.Verification.Proofs.Poseidon2
