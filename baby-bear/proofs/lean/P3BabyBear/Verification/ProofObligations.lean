/- Proofs of the contracts hax extracts from the Rust source.

`p3-baby-bear` carries no `hax_lib::requires` / `hax_lib::ensures`
attributes yet, so there is nothing here. When some are added, hax extracts
them to `<fn>.pre`, `<fn>.post` and `<fn>.spec` in
`P3BabyBear/Extraction/Specs.lean` (which `P3BabyBear/Extraction.lean` then
imports), and regenerates a `sorry` template of the obligations in
`P3BabyBear/Extraction/ProofObligations.lean` on every run. This file is the
1:1 answer to that template, one `<fn>.spec.proof` per contract, and should
contain nothing else. hax never modifies anything under `Verification/`.

Hand-written properties live in `Proofs.lean`, which this file
imports so that contract proofs can use them. -/
import P3BabyBear.Extraction
import P3BabyBear.Verification.Proofs
