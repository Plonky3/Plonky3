/- Hand-written specifications and their proofs. These are properties stated
directly in Lean, independent of any `hax_lib::requires` / `hax_lib::ensures`
contracts in the Rust source (those are proved in `ProofObligations.lean`).

Conventions:
- Name theorems `<item>.<property>`, after the Rust item they are about
  (`BabyBearParameters::PRIME` is `baby_bear.BabyBearParameters.PRIME`).
  Avoid `<fn>.spec`, `.pre` and `.post`, which hax reserves for the
  generated contracts, and `<def>.eq_<n>`, which Lean reserves for equation
  lemmas.
- Each theorem is its own specification: the statement is the claim, and
  the proof follows it.
- Nothing here depends on the generated contracts. -/
import P3BabyBear.Extraction
import CompPoly.Fields.BabyBear
open Aeneas Aeneas.Std CoreModels

namespace p3_baby_bear

/-! ## `BabyBearParameters`

The `MontyParameters` and `TwoAdicData` constants, related to the
independently specified BabyBear field in `CompPoly.Fields.BabyBear`. Each
constant is read *through the trait instance* the extracted code builds, so a
statement is about the value the code sees, not a free-standing literal. -/

/-! ### Names for the trait constants -/

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

/-! ### Wiring: the instances return these constants -/

theorem baby_bear.BabyBearParameters.PRIME.from_instance :
    MontyParams.PRIME = .ok PRIME := rfl
theorem baby_bear.BabyBearParameters.MONTY_MU.from_instance :
    MontyParams.MONTY_MU = .ok MONTY_MU := rfl
theorem baby_bear.BabyBearParameters.MONTY_BITS.from_instance :
    MontyParams.MONTY_BITS = .ok MONTY_BITS := rfl
theorem baby_bear.BabyBearParameters.TWO_ADICITY.from_instance :
    TwoAdicParams.TWO_ADICITY = .ok TWO_ADICITY := rfl

/-! ### The constants against the specification -/

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

/-! ## The `BabyBear` constructors

`BabyBear::new` (which is `MontyField31::new` at `BabyBearParameters`) and
the table constructors `new_array` and `new_2d_array`. -/

/-- `MONTY_MU` is BabyBear's Montgomery inverse modulo `2^32`, as a `u32`
wrapping product, which is the form `MontyField31::new` asserts it in. -/
theorem mu_wrapping : U32.wrapping_mul 2013265921#u32 2281701377#u32 = 1#u32 := by
  apply UScalar.eq_of_val_eq
  simp [U32.wrapping_mul_val_eq, U32.size, U32.numBits]

/-- `BabyBear::new x` never panics and returns the Montgomery form
`x · 2^32 mod p`. Rust checks four facts about the parameters in a
`const { assert!(..) }` block; aeneas keeps them as runtime `massert`s, so this
includes that they pass. -/
theorem baby_bear.BabyBear.new.montgomery_form (x : Std.U32) :
    p3_monty_31.monty_31.MontyField31.new MontyParams x
      ⦃ r => r.value.val = x.val * 2 ^ 32 % BabyBear.fieldSize ⦄ := by
  unfold p3_monty_31.monty_31.MontyField31.new p3_monty_31.utils.to_monty
  simp only [bind_tc_ok]
  unfold baby_bear.BabyBearParameters.Insts.P3_monty_31Data_traitsMontyParameters.PRIME
    baby_bear.BabyBearParameters.Insts.P3_monty_31Data_traitsMontyParameters.MONTY_BITS
    baby_bear.BabyBearParameters.Insts.P3_monty_31Data_traitsMontyParameters.MONTY_MU
    CoreModels.core.num.U32.wrapping_mul CoreModels.rust_primitives.arithmetic.wrapping_mul_u32
  simp only [mu_wrapping]
  step as ⟨ h1, h1p ⟩
  step
  step as ⟨ i2, i2p ⟩
  step
  step
  step
  step as ⟨ y, yp ⟩
  step as ⟨ z, zp1, zp2 ⟩
  step as ⟨ q, qp ⟩
  step as ⟨ w, wp ⟩
  have hx : x.val < 2 ^ 32 := by simpa using x.hBounds
  have hy : y.val = x.val := by
    rw [yp]; exact UScalar.cast_val_mod_pow_greater_numBits_eq _ _ (by simp)
  have hq : q.val = 2013265921 := by rw [qp]; simp
  have hz : z.val = x.val * 2 ^ 32 := by
    rw [zp1, hy, Nat.shiftLeft_eq, U64.size, U64.numBits]
    apply Nat.mod_eq_of_lt
    simp only [UScalarTy.U64_numBits_eq]
    omega
  have hw : w.val = x.val * 2 ^ 32 % 2013265921 := by rw [wp, hz, hq]
  have hlt : w.val < 2 ^ UScalarTy.U32.numBits := by
    rw [hw]; have := Nat.mod_lt (x.val * 2 ^ 32) (show 2013265921 > 0 by decide)
    simp only [UScalarTy.U32_numBits_eq]; omega
  rw [UScalar.cast_val_mod_pow_of_inBounds_eq _ _ hlt, hw]
  rfl

/-- `BabyBear::new` never panics: the existential half of `new.montgomery_form`. -/
theorem baby_bear.BabyBear.new.ok (x : Std.U32) :
    ∃ v, p3_monty_31.monty_31.MontyField31.new MontyParams x = .ok v := by
  obtain ⟨v, hv, -⟩ := Aeneas.Std.WP.spec_imp_exists (baby_bear.BabyBear.new.montgomery_form x)
  exact ⟨v, hv⟩

/-- A `mapM` over a function that never fails never fails, and keeps the length. -/
theorem mapM_total {α β : Type} (f : α → RustM β) (hf : ∀ x, ∃ v, f x = .ok v)
    (l : List α) : ∃ l', l.mapM f = .ok l' ∧ l'.length = l.length := by
  induction l with
  | nil => exact ⟨[], rfl, rfl⟩
  | cons a l ih =>
    obtain ⟨v, hv⟩ := hf a
    obtain ⟨l', hl', hlen⟩ := ih
    refine ⟨v :: l', ?_, by simp [hlen]⟩
    simp [List.mapM_cons, hv, hl']
    rfl

/-- `BabyBear::new_array` never panics. It is a hand-written transcription
(`P3BabyBear/Assumptions/P3Monty31.lean`): aeneas drops the Rust function. -/
theorem baby_bear.BabyBear.new_array.never_panics {N : Std.Usize} (input : Array Std.U32 N) :
    ∃ r, p3_monty_31.monty_31.MontyField31.new_array MontyParams input = .ok r := by
  obtain ⟨l', hl', hlen⟩ := mapM_total _ baby_bear.BabyBear.new.ok input.val
  unfold p3_monty_31.monty_31.MontyField31.new_array
  rw [hl']
  have h : l'.length = N.val := by rw [hlen]; exact input.property
  exact ⟨⟨l', h⟩, by simp [h]⟩

/-- `BabyBear::new_2d_array` never panics; hand-transcribed, like `new_array`. -/
theorem baby_bear.BabyBear.new_2d_array.never_panics {N M : Std.Usize}
    (input : Array (Array Std.U32 N) M) :
    ∃ r, p3_monty_31.monty_31.MontyField31.new_2d_array MontyParams input = .ok r := by
  obtain ⟨l', hl', hlen⟩ :=
    mapM_total _ (fun a => baby_bear.BabyBear.new_array.never_panics a) input.val
  unfold p3_monty_31.monty_31.MontyField31.new_2d_array
  rw [hl']
  have h : l'.length = M.val := by rw [hlen]; exact input.property
  exact ⟨⟨l', h⟩, by simp [h]⟩

/-! ## The Poseidon round-constant length assertions

Nine of the eleven `const _: () = assert!(..)` in `baby-bear/src/poseidon{1,2}.rs`.
aeneas names the first in each module `_` and the rest `__N`; post-extraction
patch 030 renames those to `const_check_N`. `poseidon1._` (on
`BABYBEAR_POSEIDON1_RC_16`) and `poseidon2._` (on
`BABYBEAR_POSEIDON2_RC_16_EXTERNAL_INITIAL`) are not proved here. -/

/-! ### `poseidon1` -/

/-- `baby-bear/src/poseidon1.rs:70`, on `BABYBEAR_POSEIDON1_RC_24`. -/
theorem poseidon1.const_check_1.holds : poseidon1.const_check_1 ⦃ _ => True ⦄ := by
  obtain ⟨t, ht⟩ : ∃ t, poseidon1.BABYBEAR_POSEIDON1_RC_24 = .ok t := by
    unfold poseidon1.BABYBEAR_POSEIDON1_RC_24; exact baby_bear.BabyBear.new_2d_array.never_panics _
  unfold poseidon1.const_check_1
    poseidon1.BABYBEAR_POSEIDON1_HALF_FULL_ROUNDS poseidon1.BABYBEAR_POSEIDON1_PARTIAL_ROUNDS_24
  simp only [ht, bind_tc_ok, CoreModels.core.slice.Slice.len,
    CoreModels.rust_primitives.slice.slice_length]
  step*
  all_goals
    simp only [not_not]
    apply UScalar.eq_of_val_eq
    simp_all

/-! ### `poseidon2` -/

/-- `baby-bear/src/poseidon2.rs:66`, on `BABYBEAR_POSEIDON2_RC_16_EXTERNAL_FINAL`. -/
theorem poseidon2.const_check_1.holds : poseidon2.const_check_1 ⦃ _ => True ⦄ := by
  obtain ⟨t, ht⟩ : ∃ t, poseidon2.BABYBEAR_POSEIDON2_RC_16_EXTERNAL_FINAL = .ok t := by
    unfold poseidon2.BABYBEAR_POSEIDON2_RC_16_EXTERNAL_FINAL; exact baby_bear.BabyBear.new_2d_array.never_panics _
  unfold poseidon2.const_check_1 poseidon2.BABYBEAR_POSEIDON2_HALF_FULL_ROUNDS
  simp only [ht, bind_tc_ok, CoreModels.core.slice.Slice.len,
    CoreModels.rust_primitives.slice.slice_length]
  step*
  all_goals
    simp only [not_not]
    apply UScalar.eq_of_val_eq
    simp_all

/-- `baby-bear/src/poseidon2.rs:68`, on `BABYBEAR_POSEIDON2_RC_16_INTERNAL`. -/
theorem poseidon2.const_check_2.holds : poseidon2.const_check_2 ⦃ _ => True ⦄ := by
  obtain ⟨t, ht⟩ : ∃ t, poseidon2.BABYBEAR_POSEIDON2_RC_16_INTERNAL = .ok t := by
    unfold poseidon2.BABYBEAR_POSEIDON2_RC_16_INTERNAL; exact baby_bear.BabyBear.new_array.never_panics _
  unfold poseidon2.const_check_2 poseidon2.BABYBEAR_POSEIDON2_PARTIAL_ROUNDS_16
  simp only [ht, bind_tc_ok, CoreModels.core.slice.Slice.len,
    CoreModels.rust_primitives.slice.slice_length]
  step*

/-- `baby-bear/src/poseidon2.rs:70`, on `BABYBEAR_POSEIDON2_RC_24_EXTERNAL_INITIAL`. -/
theorem poseidon2.const_check_3.holds : poseidon2.const_check_3 ⦃ _ => True ⦄ := by
  obtain ⟨t, ht⟩ : ∃ t, poseidon2.BABYBEAR_POSEIDON2_RC_24_EXTERNAL_INITIAL = .ok t := by
    unfold poseidon2.BABYBEAR_POSEIDON2_RC_24_EXTERNAL_INITIAL; exact baby_bear.BabyBear.new_2d_array.never_panics _
  unfold poseidon2.const_check_3 poseidon2.BABYBEAR_POSEIDON2_HALF_FULL_ROUNDS
  simp only [ht, bind_tc_ok, CoreModels.core.slice.Slice.len,
    CoreModels.rust_primitives.slice.slice_length]
  step*
  all_goals
    simp only [not_not]
    apply UScalar.eq_of_val_eq
    simp_all

/-- `baby-bear/src/poseidon2.rs:72`, on `BABYBEAR_POSEIDON2_RC_24_EXTERNAL_FINAL`. -/
theorem poseidon2.const_check_4.holds : poseidon2.const_check_4 ⦃ _ => True ⦄ := by
  obtain ⟨t, ht⟩ : ∃ t, poseidon2.BABYBEAR_POSEIDON2_RC_24_EXTERNAL_FINAL = .ok t := by
    unfold poseidon2.BABYBEAR_POSEIDON2_RC_24_EXTERNAL_FINAL; exact baby_bear.BabyBear.new_2d_array.never_panics _
  unfold poseidon2.const_check_4 poseidon2.BABYBEAR_POSEIDON2_HALF_FULL_ROUNDS
  simp only [ht, bind_tc_ok, CoreModels.core.slice.Slice.len,
    CoreModels.rust_primitives.slice.slice_length]
  step*
  all_goals
    simp only [not_not]
    apply UScalar.eq_of_val_eq
    simp_all

/-- `baby-bear/src/poseidon2.rs:74`, on `BABYBEAR_POSEIDON2_RC_24_INTERNAL`. -/
theorem poseidon2.const_check_5.holds : poseidon2.const_check_5 ⦃ _ => True ⦄ := by
  obtain ⟨t, ht⟩ : ∃ t, poseidon2.BABYBEAR_POSEIDON2_RC_24_INTERNAL = .ok t := by
    unfold poseidon2.BABYBEAR_POSEIDON2_RC_24_INTERNAL; exact baby_bear.BabyBear.new_array.never_panics _
  unfold poseidon2.const_check_5 poseidon2.BABYBEAR_POSEIDON2_PARTIAL_ROUNDS_24
  simp only [ht, bind_tc_ok, CoreModels.core.slice.Slice.len,
    CoreModels.rust_primitives.slice.slice_length]
  step*

/-- `baby-bear/src/poseidon2.rs:76`, on `BABYBEAR_POSEIDON2_RC_32_EXTERNAL_INITIAL`. -/
theorem poseidon2.const_check_6.holds : poseidon2.const_check_6 ⦃ _ => True ⦄ := by
  obtain ⟨t, ht⟩ : ∃ t, poseidon2.BABYBEAR_POSEIDON2_RC_32_EXTERNAL_INITIAL = .ok t := by
    unfold poseidon2.BABYBEAR_POSEIDON2_RC_32_EXTERNAL_INITIAL; exact baby_bear.BabyBear.new_2d_array.never_panics _
  unfold poseidon2.const_check_6 poseidon2.BABYBEAR_POSEIDON2_HALF_FULL_ROUNDS
  simp only [ht, bind_tc_ok, CoreModels.core.slice.Slice.len,
    CoreModels.rust_primitives.slice.slice_length]
  step*
  all_goals
    simp only [not_not]
    apply UScalar.eq_of_val_eq
    simp_all

/-- `baby-bear/src/poseidon2.rs:78`, on `BABYBEAR_POSEIDON2_RC_32_EXTERNAL_FINAL`. -/
theorem poseidon2.const_check_7.holds : poseidon2.const_check_7 ⦃ _ => True ⦄ := by
  obtain ⟨t, ht⟩ : ∃ t, poseidon2.BABYBEAR_POSEIDON2_RC_32_EXTERNAL_FINAL = .ok t := by
    unfold poseidon2.BABYBEAR_POSEIDON2_RC_32_EXTERNAL_FINAL; exact baby_bear.BabyBear.new_2d_array.never_panics _
  unfold poseidon2.const_check_7 poseidon2.BABYBEAR_POSEIDON2_HALF_FULL_ROUNDS
  simp only [ht, bind_tc_ok, CoreModels.core.slice.Slice.len,
    CoreModels.rust_primitives.slice.slice_length]
  step*
  all_goals
    simp only [not_not]
    apply UScalar.eq_of_val_eq
    simp_all

/-- `baby-bear/src/poseidon2.rs:80`, on `BABYBEAR_POSEIDON2_RC_32_INTERNAL`. -/
theorem poseidon2.const_check_8.holds : poseidon2.const_check_8 ⦃ _ => True ⦄ := by
  obtain ⟨t, ht⟩ : ∃ t, poseidon2.BABYBEAR_POSEIDON2_RC_32_INTERNAL = .ok t := by
    unfold poseidon2.BABYBEAR_POSEIDON2_RC_32_INTERNAL; exact baby_bear.BabyBear.new_array.never_panics _
  unfold poseidon2.const_check_8 poseidon2.BABYBEAR_POSEIDON2_PARTIAL_ROUNDS_32
  simp only [ht, bind_tc_ok, CoreModels.core.slice.Slice.len,
    CoreModels.rust_primitives.slice.slice_length]
  step*

end p3_baby_bear
