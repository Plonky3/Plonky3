import p3_baby_bear
import CompPoly.Fields.BabyBear

/-!
# BabyBear constants: extraction vs. specification

Relates the constants in the hax extraction (`p3_baby_bear.baby_bear`) to the
independently-specified mathematical BabyBear field in `CompPoly.Fields.BabyBear`.

These are the first statements in this tree that are *proved* rather than
assumed. The axiomatized interface under `p3_baby_bear/` is a trust assumption;
these theorems are not. See `TCB.md`.

Scope is deliberately narrow: constants only. Nothing here says anything about
the *arithmetic* being correct.
-/

open p3_baby_bear.baby_bear
open p3_monty_31.data_traits

/-- The Monty instance's `PRIME` bitpattern is exactly `BabyBear.fieldSize`
(`2^31 - 2^27 + 1 = 2013265921`). -/
theorem monty_prime_eq_fieldSize :
    (MontyParameters.PRIME BabyBearParameters).toNat = BabyBear.fieldSize := by
  simp [BabyBear.fieldSize, MontyParameters.PRIME, Impl.PRIME_hoisted]

/-- The extracted modulus is prime. CompPoly's Pratt certificate
`BabyBear.is_prime`, transported across `monty_prime_eq_fieldSize`. -/
theorem monty_prime_is_prime :
    Nat.Prime (MontyParameters.PRIME BabyBearParameters).toNat := by
  rw [monty_prime_eq_fieldSize]
  exact BabyBear.is_prime

/-- Two-adicity declared for FFT / NTT matches `BabyBear.twoAdicity` (27). -/
theorem two_adicity_eq_spec :
    (TwoAdicData.TWO_ADICITY BabyBearParameters).toNat = BabyBear.twoAdicity := by
  simp [BabyBear.twoAdicity, TwoAdicData.TWO_ADICITY, Impl_5.TWO_ADICITY_hoisted]

/-- Extraction tripwire: `MONTY_BITS` is the literal `32` that Rust asserts in
`monty-31/src/monty_31.rs`. Both sides come from the extraction, so this is a
wiring check, not a representation theorem. The Montgomery precondition is
`monty_mu_inverse`. -/
theorem monty_bits_eq_thirtyTwo :
    (MontyParameters.MONTY_BITS BabyBearParameters).toNat = 32 := by rfl

/-- `PRIME * MONTY_MU ≡ 1 (mod 2^32)`. This is the Montgomery-reduction
precondition that Rust asserts at compile time in `monty-31/src/monty_31.rs`
and that Lean `Impl.new` does not re-check. -/
theorem monty_mu_inverse :
    ((MontyParameters.PRIME BabyBearParameters).toNat *
     (MontyParameters.MONTY_MU BabyBearParameters).toNat) % (2 ^ 32) = 1 := by
  simp [MontyParameters.PRIME, MontyParameters.MONTY_MU,
        Impl.PRIME_hoisted, Impl.MONTY_MU_hoisted]

/-- The degree-7 power map is a unit-group automorphism: `gcd(7, p - 1) = 1`.

Note this is `7`, not the `3` used for KoalaBear: BabyBear has
`p - 1 = 2^27 * 3 * 5`, so `gcd(3, p - 1) = 3` and the cube map is *not* a
bijection here. The extraction declares `RelativelyPrimePower BabyBearParameters
((7 : u64))` accordingly. -/
theorem coprime_seven_pred_prime :
    Nat.Coprime 7 ((MontyParameters.PRIME BabyBearParameters).toNat - 1) := by
  simp [MontyParameters.PRIME, Impl.PRIME_hoisted]
  decide

/-- Factorization of `p - 1` into `2^twoAdicity * 15` (two-adic FFT domain size).
For KoalaBear the odd part is `127`; for BabyBear it is `15 = 3 * 5`. -/
theorem fieldSize_sub_one_factorization' :
    (MontyParameters.PRIME BabyBearParameters).toNat - 1 =
      2 ^ (TwoAdicData.TWO_ADICITY BabyBearParameters).toNat * 15 := by
  simp [MontyParameters.PRIME, TwoAdicData.TWO_ADICITY,
        Impl.PRIME_hoisted, Impl_5.TWO_ADICITY_hoisted]
