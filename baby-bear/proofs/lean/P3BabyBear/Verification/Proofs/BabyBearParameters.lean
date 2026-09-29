/- Hand-written properties of `BabyBearParameters`: its `MontyParameters`
and `TwoAdicData` constants, related to the independently specified BabyBear
field in `CompPoly.Fields.BabyBear`.

Each constant is read *through the trait instance* the extracted code
builds, so a statement is about the value the code sees, not a free-standing
literal. -/
import P3BabyBear.Extraction
import CompPoly.Fields.BabyBear
open Aeneas Aeneas.Std CoreModels

namespace p3_baby_bear

/-! ## Names for the trait constants -/

/-- The `MontyParameters` instance aeneas generated for `BabyBearParameters`. -/
noncomputable abbrev baby_bear.BabyBearParameters.MontyParams :=
  baby_bear.BabyBearParameters.Insts.P3_monty_31Data_traitsMontyParameters

/-- The `TwoAdicData` instance aeneas generated for `BabyBearParameters`. -/
noncomputable abbrev baby_bear.BabyBearParameters.TwoAdicParams :=
  baby_bear.BabyBearParameters.Insts.P3_monty_31Data_traitsTwoAdicDataSharedStaticSliceMontyField31BabyBearParameters

/-- `BabyBearParameters::PRIME`, `MONTY_MU`, `MONTY_BITS` and `TWO_ADICITY` as
the extraction defines them. -/
abbrev baby_bear.BabyBearParameters.PRIME : Std.U32 :=
  baby_bear.BabyBearParameters.Insts.P3_monty_31Data_traitsMontyParameters.PRIME
abbrev baby_bear.BabyBearParameters.MONTY_MU : Std.U32 :=
  baby_bear.BabyBearParameters.Insts.P3_monty_31Data_traitsMontyParameters.MONTY_MU
abbrev baby_bear.BabyBearParameters.MONTY_BITS : Std.U32 :=
  baby_bear.BabyBearParameters.Insts.P3_monty_31Data_traitsMontyParameters.MONTY_BITS
abbrev baby_bear.BabyBearParameters.TWO_ADICITY : Std.Usize :=
  baby_bear.BabyBearParameters.Insts.P3_monty_31Data_traitsTwoAdicDataSharedStaticSliceMontyField31BabyBearParameters.TWO_ADICITY

open baby_bear.BabyBearParameters (PRIME MONTY_MU MONTY_BITS TWO_ADICITY MontyParams TwoAdicParams)

/-! ## Wiring: the instances return these constants -/

theorem baby_bear.BabyBearParameters.PRIME.from_instance :
    MontyParams.PRIME = .ok PRIME := rfl
theorem baby_bear.BabyBearParameters.MONTY_MU.from_instance :
    MontyParams.MONTY_MU = .ok MONTY_MU := rfl
theorem baby_bear.BabyBearParameters.MONTY_BITS.from_instance :
    MontyParams.MONTY_BITS = .ok MONTY_BITS := rfl
theorem baby_bear.BabyBearParameters.TWO_ADICITY.from_instance :
    TwoAdicParams.TWO_ADICITY = .ok TWO_ADICITY := rfl

/-! ## The constants against the specification -/

/-- `PRIME` is exactly CompPoly's `BabyBear.fieldSize` (`2^31 - 2^27 + 1 = 2013265921`). -/
theorem baby_bear.BabyBearParameters.PRIME.eq_fieldSize :
    PRIME.val = BabyBear.fieldSize := by
  simp [PRIME, BabyBear.fieldSize,
    baby_bear.BabyBearParameters.Insts.P3_monty_31Data_traitsMontyParameters.PRIME]

/-- The modulus is prime: CompPoly's Pratt certificate `BabyBear.is_prime`,
transported across `PRIME.eq_fieldSize`. -/
theorem baby_bear.BabyBearParameters.PRIME.is_prime :
    Nat.Prime PRIME.val := by
  rw [baby_bear.BabyBearParameters.PRIME.eq_fieldSize]
  exact BabyBear.is_prime

/-- `gcd(7, p - 1) = 1`, so `x ↦ x^7` permutes the field: the fact behind
`RelativelyPrimePower<7>`. Not `3`, as for KoalaBear: BabyBear has
`p - 1 = 2^27 · 3 · 5`, so the cube map is not a bijection. -/
theorem baby_bear.BabyBearParameters.PRIME.coprime_seven_pred :
    Nat.Coprime 7 (PRIME.val - 1) := by
  rw [baby_bear.BabyBearParameters.PRIME.eq_fieldSize]
  decide

/-- Extraction tripwire: `MONTY_BITS` is the literal `32` that Rust asserts in
`monty-31/src/monty_31.rs`. Both sides come from the extraction, so this is a
wiring check, not a representation theorem. -/
theorem baby_bear.BabyBearParameters.MONTY_BITS.val_eq_32 :
    MONTY_BITS.val = 32 := by
  simp [MONTY_BITS,
    baby_bear.BabyBearParameters.Insts.P3_monty_31Data_traitsMontyParameters.MONTY_BITS]

/-- `PRIME · MONTY_MU ≡ 1 (mod 2^32)`: the Montgomery-reduction precondition that
Rust asserts at compile time in `MontyField31::new`. -/
theorem baby_bear.BabyBearParameters.MONTY_MU.inverse :
    (PRIME.val * MONTY_MU.val) % (2 ^ 32) = 1 := by
  simp [PRIME, MONTY_MU,
    baby_bear.BabyBearParameters.Insts.P3_monty_31Data_traitsMontyParameters.PRIME,
    baby_bear.BabyBearParameters.Insts.P3_monty_31Data_traitsMontyParameters.MONTY_MU]

/-- `TWO_ADICITY` is CompPoly's `BabyBear.twoAdicity` (27). -/
theorem baby_bear.BabyBearParameters.TWO_ADICITY.eq_twoAdicity :
    TWO_ADICITY.val = BabyBear.twoAdicity := by
  simp [TWO_ADICITY, BabyBear.twoAdicity,
    baby_bear.BabyBearParameters.Insts.P3_monty_31Data_traitsTwoAdicDataSharedStaticSliceMontyField31BabyBearParameters.TWO_ADICITY]

/-- `p - 1 = 2^TWO_ADICITY · 15`: the size of the two-adic FFT domain. On its
own this fixes the odd part by hand; `eq_twoAdicity` and `maximal` are what
pin `TWO_ADICITY` down. -/
theorem baby_bear.BabyBearParameters.TWO_ADICITY.factorization :
    PRIME.val - 1 = 2 ^ TWO_ADICITY.val * 15 := by
  rw [baby_bear.BabyBearParameters.PRIME.eq_fieldSize,
    baby_bear.BabyBearParameters.TWO_ADICITY.eq_twoAdicity]
  exact BabyBear.fieldSize_sub_one_factorization

/-- `TWO_ADICITY` is maximal: `2^(TWO_ADICITY + 1)` does not divide `p - 1`,
so `2^TWO_ADICITY` is the largest power of two dividing it. -/
theorem baby_bear.BabyBearParameters.TWO_ADICITY.maximal :
    ¬ 2 ^ (TWO_ADICITY.val + 1) ∣ PRIME.val - 1 := by
  rw [baby_bear.BabyBearParameters.TWO_ADICITY.factorization, pow_succ,
    Nat.mul_dvd_mul_iff_left (Nat.two_pow_pos _)]
  decide

end p3_baby_bear
