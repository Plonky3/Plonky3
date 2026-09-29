/- Everything hand-verified about the crate, in two parts that are kept apart:

- `Proofs`: hand-written specifications and their proofs.
- `ProofObligations`: proofs of the contracts hax extracts from
  `hax_lib::requires` / `hax_lib::ensures` attributes in the Rust source, a
  1:1 answer to the template hax regenerates in
  `Extraction/ProofObligations.lean`. There are none yet. -/
import P3BabyBear.Verification.Proofs
import P3BabyBear.Verification.ProofObligations
