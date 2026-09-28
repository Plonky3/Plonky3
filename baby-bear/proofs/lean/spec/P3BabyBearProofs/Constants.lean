import P3BabyBear
import CompPoly.Fields.BabyBear

/-!
# BabyBear constants: extraction vs. specification

Relates the constants in the aeneas extraction of `p3_baby_bear::baby_bear` to the
independently-specified mathematical BabyBear field in `CompPoly.Fields.BabyBear`.

Each constant is stated *through the trait instance* that the generated code
builds, e.g. `MontyParams.PRIME = ok p`. That instance is what every
`MontyField31` function reads the constant from, so the statement is about the
value the code sees rather than a free-standing literal.

Scope: constants only. `MontyField31.lean` is about code.
-/

open Aeneas Aeneas.Std CoreModels
open p3_baby_bear

namespace P3BabyBearProofs

/-- The `MontyParameters` instance aeneas generated for `BabyBearParameters`. -/
noncomputable abbrev MontyParams :=
  baby_bear.BabyBearParameters.Insts.P3_monty_31Data_traitsMontyParameters

/-- The `TwoAdicData` instance aeneas generated for `BabyBearParameters`. -/
noncomputable abbrev TwoAdicParams :=
  baby_bear.BabyBearParameters.Insts.P3_monty_31Data_traitsTwoAdicDataSharedStaticSliceMontyField31BabyBearParameters

/-- `p`, `MONTY_BITS`, `MONTY_MU` and `TWO_ADICITY` as the extraction defines
them, and as the instances above return them. -/
abbrev P : Std.U32 :=
  baby_bear.BabyBearParameters.Insts.P3_monty_31Data_traitsMontyParameters.PRIME
abbrev MU : Std.U32 :=
  baby_bear.BabyBearParameters.Insts.P3_monty_31Data_traitsMontyParameters.MONTY_MU
abbrev BITS : Std.U32 :=
  baby_bear.BabyBearParameters.Insts.P3_monty_31Data_traitsMontyParameters.MONTY_BITS
abbrev TWO_ADICITY : Std.Usize :=
  baby_bear.BabyBearParameters.Insts.P3_monty_31Data_traitsTwoAdicDataSharedStaticSliceMontyField31BabyBearParameters.TWO_ADICITY

/-! ## Wiring: the instances return the constants -/

theorem prime_ok : MontyParams.PRIME = .ok P := rfl
theorem monty_mu_ok : MontyParams.MONTY_MU = .ok MU := rfl
theorem monty_bits_ok : MontyParams.MONTY_BITS = .ok BITS := rfl
theorem two_adicity_ok : TwoAdicParams.TWO_ADICITY = .ok TWO_ADICITY := rfl

/-! ## The constants against the specification -/

/-- The Monty instance's `PRIME` is exactly `BabyBear.fieldSize`
(`2^31 - 2^27 + 1 = 2013265921`). -/
theorem monty_prime_eq_fieldSize : P.val = BabyBear.fieldSize := by
  simp [P, BabyBear.fieldSize,
    baby_bear.BabyBearParameters.Insts.P3_monty_31Data_traitsMontyParameters.PRIME]

/-- The extracted modulus is prime: CompPoly's Pratt certificate
`BabyBear.is_prime`, transported across `monty_prime_eq_fieldSize`. -/
theorem monty_prime_is_prime : Nat.Prime P.val := by
  rw [monty_prime_eq_fieldSize]
  exact BabyBear.is_prime

/-- The two-adicity declared for FFT / NTT matches `BabyBear.twoAdicity` (27). -/
theorem two_adicity_eq_spec : TWO_ADICITY.val = BabyBear.twoAdicity := by
  simp [TWO_ADICITY, BabyBear.twoAdicity,
    baby_bear.BabyBearParameters.Insts.P3_monty_31Data_traitsTwoAdicDataSharedStaticSliceMontyField31BabyBearParameters.TWO_ADICITY]

/-- Extraction tripwire: `MONTY_BITS` is the literal `32` that Rust asserts in
`monty-31/src/monty_31.rs`. Both sides come from the extraction, so this is a
wiring check, not a representation theorem. -/
theorem monty_bits_eq_thirtyTwo : BITS.val = 32 := by
  simp [BITS,
    baby_bear.BabyBearParameters.Insts.P3_monty_31Data_traitsMontyParameters.MONTY_BITS]

/-- `PRIME * MONTY_MU ≡ 1 (mod 2^32)`: the Montgomery-reduction precondition that
Rust asserts at compile time in `MontyField31::new`. aeneas keeps that assertion
as a runtime `massert`; `MontyField31.lean` shows it passes. -/
theorem monty_mu_inverse : (P.val * MU.val) % (2 ^ 32) = 1 := by
  simp [P, MU,
    baby_bear.BabyBearParameters.Insts.P3_monty_31Data_traitsMontyParameters.PRIME,
    baby_bear.BabyBearParameters.Insts.P3_monty_31Data_traitsMontyParameters.MONTY_MU]

/-- The degree-7 power map is a unit-group automorphism: `gcd(7, p - 1) = 1`.

This is `7`, not the `3` used for KoalaBear: BabyBear has `p - 1 = 2^27 * 3 * 5`,
so `gcd(3, p - 1) = 3` and the cube map is *not* a bijection here. The extraction
declares `RelativelyPrimePower BabyBearParameters 7#u64` accordingly. -/
theorem coprime_seven_pred_prime : Nat.Coprime 7 (P.val - 1) := by
  rw [monty_prime_eq_fieldSize]
  decide

/-- `p - 1 = 2^TWO_ADICITY * 15`: the size of the two-adic FFT domain. For
KoalaBear the odd part is `127`; for BabyBear it is `15 = 3 * 5`.

On its own this relates two extracted constants with an odd part chosen by
hand. Read it together with `two_adicity_eq_spec` (the constant is CompPoly's)
and `two_adicity_maximal` (it is the largest such power). -/
theorem fieldSize_sub_one_factorization :
    P.val - 1 = 2 ^ TWO_ADICITY.val * 15 := by
  rw [monty_prime_eq_fieldSize, two_adicity_eq_spec]
  exact BabyBear.fieldSize_sub_one_factorization

/-- `TWO_ADICITY` is maximal: `2^(TWO_ADICITY + 1)` does not divide `p - 1`, so
`2^TWO_ADICITY` is the largest power of two dividing it. This is what the
constant has to mean; `fieldSize_sub_one_factorization` alone would also hold
for a smaller exponent with a larger (even) cofactor. -/
theorem two_adicity_maximal : ¬ 2 ^ (TWO_ADICITY.val + 1) ∣ P.val - 1 := by
  rw [fieldSize_sub_one_factorization, pow_succ,
    Nat.mul_dvd_mul_iff_left (Nat.two_pow_pos _)]
  decide

end P3BabyBearProofs
