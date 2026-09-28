import P3BabyBearProofs.Constants

/-!
# `MontyField31` construction on BabyBear: the code, not just the constants

`Constants.lean` is about literals. This file is about the generated *code*
that consumes them.

* `new_spec`: `MontyField31::new` on BabyBear never panics and returns the
  Montgomery form `x · 2^32 mod p`. Rust checks four facts about the
  parameters in a `const { assert!(..) }` block (`PRIME` odd, `PRIME < 2^31`,
  `MONTY_BITS = 32`, `PRIME · MONTY_MU ≡ 1 mod 2^32`); aeneas keeps them as
  runtime `massert`s, and this proof discharges each one.
* `new_array_ok`, `new_2d_array_ok`: the table constructors never panic. This
  is about the hand-written transcriptions in `assumptions/Interface/P3Monty31Missing.lean`,
  built on the extracted `new`.
* `poseidon{1,2}_const_check_N`: nine of the eleven Poseidon length
  assertions. aeneas names the first anonymous const in each module `_` and
  the rest `__N`; patch 030 renames the `__N` ones, and these theorems are
  those nine. `poseidon1._` (`BABYBEAR_POSEIDON1_RC_16`) and `poseidon2._`
  (`BABYBEAR_POSEIDON2_RC_16_EXTERNAL_INITIAL`) are not proved here.

Nothing here is about the Poseidon permutations or field arithmetic.
-/

open Aeneas Aeneas.Std CoreModels
open p3_baby_bear

namespace P3BabyBearProofs

theorem mu_wrapping : U32.wrapping_mul 2013265921#u32 2281701377#u32 = 1#u32 := by
  apply UScalar.eq_of_val_eq
  simp [U32.wrapping_mul_val_eq, U32.size, U32.numBits]

/-- `MontyField31::new` on BabyBear: total, and computes the Montgomery form. -/
theorem new_spec (x : Std.U32) :
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

theorem new_ok (x : Std.U32) :
    ∃ v, p3_monty_31.monty_31.MontyField31.new MontyParams x = .ok v := by
  obtain ⟨v, hv, -⟩ := Aeneas.Std.WP.spec_imp_exists (new_spec x)
  exact ⟨v, hv⟩

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

/-- `MontyField31::new_array` on BabyBear never panics. -/
theorem new_array_ok {N : Std.Usize} (input : Array Std.U32 N) :
    ∃ r, p3_monty_31.monty_31.MontyField31.new_array MontyParams input = .ok r := by
  obtain ⟨l', hl', hlen⟩ := mapM_total _ new_ok input.val
  unfold p3_monty_31.monty_31.MontyField31.new_array
  rw [hl']
  have h : l'.length = N.val := by rw [hlen]; exact input.property
  exact ⟨⟨l', h⟩, by simp [h]⟩

/-- `MontyField31::new_2d_array` on BabyBear never panics. -/
theorem new_2d_array_ok {N M : Std.Usize} (input : Array (Array Std.U32 N) M) :
    ∃ r, p3_monty_31.monty_31.MontyField31.new_2d_array MontyParams input = .ok r := by
  obtain ⟨l', hl', hlen⟩ := mapM_total _ new_array_ok input.val
  unfold p3_monty_31.monty_31.MontyField31.new_2d_array
  rw [hl']
  have h : l'.length = M.val := by rw [hlen]; exact input.property
  exact ⟨⟨l', h⟩, by simp [h]⟩

/-! ## The Poseidon table length assertions -/

/-- The length assertion `baby-bear/src/poseidon1.rs:70` on `BABYBEAR_POSEIDON1_RC_24` holds. -/
theorem poseidon1_const_check_1 : poseidon1.const_check_1 ⦃ _ => True ⦄ := by
  obtain ⟨t, ht⟩ : ∃ t, poseidon1.BABYBEAR_POSEIDON1_RC_24 = .ok t := by
    unfold poseidon1.BABYBEAR_POSEIDON1_RC_24; exact new_2d_array_ok _
  unfold poseidon1.const_check_1 poseidon1.BABYBEAR_POSEIDON1_HALF_FULL_ROUNDS poseidon1.BABYBEAR_POSEIDON1_PARTIAL_ROUNDS_24
  simp only [ht, bind_tc_ok, CoreModels.core.slice.Slice.len,
    CoreModels.rust_primitives.slice.slice_length]
  step*
  all_goals
    simp only [not_not]
    apply UScalar.eq_of_val_eq
    simp_all

/-- The length assertion `baby-bear/src/poseidon2.rs:66` on `BABYBEAR_POSEIDON2_RC_16_EXTERNAL_FINAL` holds. -/
theorem poseidon2_const_check_1 : poseidon2.const_check_1 ⦃ _ => True ⦄ := by
  obtain ⟨t, ht⟩ : ∃ t, poseidon2.BABYBEAR_POSEIDON2_RC_16_EXTERNAL_FINAL = .ok t := by
    unfold poseidon2.BABYBEAR_POSEIDON2_RC_16_EXTERNAL_FINAL; exact new_2d_array_ok _
  unfold poseidon2.const_check_1 poseidon2.BABYBEAR_POSEIDON2_HALF_FULL_ROUNDS
  simp only [ht, bind_tc_ok, CoreModels.core.slice.Slice.len,
    CoreModels.rust_primitives.slice.slice_length]
  step*
  all_goals
    simp only [not_not]
    apply UScalar.eq_of_val_eq
    simp_all

/-- The length assertion `baby-bear/src/poseidon2.rs:68` on `BABYBEAR_POSEIDON2_RC_16_INTERNAL` holds. -/
theorem poseidon2_const_check_2 : poseidon2.const_check_2 ⦃ _ => True ⦄ := by
  obtain ⟨t, ht⟩ : ∃ t, poseidon2.BABYBEAR_POSEIDON2_RC_16_INTERNAL = .ok t := by
    unfold poseidon2.BABYBEAR_POSEIDON2_RC_16_INTERNAL; exact new_array_ok _
  unfold poseidon2.const_check_2 poseidon2.BABYBEAR_POSEIDON2_PARTIAL_ROUNDS_16
  simp only [ht, bind_tc_ok, CoreModels.core.slice.Slice.len,
    CoreModels.rust_primitives.slice.slice_length]
  step*

/-- The length assertion `baby-bear/src/poseidon2.rs:70` on `BABYBEAR_POSEIDON2_RC_24_EXTERNAL_INITIAL` holds. -/
theorem poseidon2_const_check_3 : poseidon2.const_check_3 ⦃ _ => True ⦄ := by
  obtain ⟨t, ht⟩ : ∃ t, poseidon2.BABYBEAR_POSEIDON2_RC_24_EXTERNAL_INITIAL = .ok t := by
    unfold poseidon2.BABYBEAR_POSEIDON2_RC_24_EXTERNAL_INITIAL; exact new_2d_array_ok _
  unfold poseidon2.const_check_3 poseidon2.BABYBEAR_POSEIDON2_HALF_FULL_ROUNDS
  simp only [ht, bind_tc_ok, CoreModels.core.slice.Slice.len,
    CoreModels.rust_primitives.slice.slice_length]
  step*
  all_goals
    simp only [not_not]
    apply UScalar.eq_of_val_eq
    simp_all

/-- The length assertion `baby-bear/src/poseidon2.rs:72` on `BABYBEAR_POSEIDON2_RC_24_EXTERNAL_FINAL` holds. -/
theorem poseidon2_const_check_4 : poseidon2.const_check_4 ⦃ _ => True ⦄ := by
  obtain ⟨t, ht⟩ : ∃ t, poseidon2.BABYBEAR_POSEIDON2_RC_24_EXTERNAL_FINAL = .ok t := by
    unfold poseidon2.BABYBEAR_POSEIDON2_RC_24_EXTERNAL_FINAL; exact new_2d_array_ok _
  unfold poseidon2.const_check_4 poseidon2.BABYBEAR_POSEIDON2_HALF_FULL_ROUNDS
  simp only [ht, bind_tc_ok, CoreModels.core.slice.Slice.len,
    CoreModels.rust_primitives.slice.slice_length]
  step*
  all_goals
    simp only [not_not]
    apply UScalar.eq_of_val_eq
    simp_all

/-- The length assertion `baby-bear/src/poseidon2.rs:74` on `BABYBEAR_POSEIDON2_RC_24_INTERNAL` holds. -/
theorem poseidon2_const_check_5 : poseidon2.const_check_5 ⦃ _ => True ⦄ := by
  obtain ⟨t, ht⟩ : ∃ t, poseidon2.BABYBEAR_POSEIDON2_RC_24_INTERNAL = .ok t := by
    unfold poseidon2.BABYBEAR_POSEIDON2_RC_24_INTERNAL; exact new_array_ok _
  unfold poseidon2.const_check_5 poseidon2.BABYBEAR_POSEIDON2_PARTIAL_ROUNDS_24
  simp only [ht, bind_tc_ok, CoreModels.core.slice.Slice.len,
    CoreModels.rust_primitives.slice.slice_length]
  step*

/-- The length assertion `baby-bear/src/poseidon2.rs:76` on `BABYBEAR_POSEIDON2_RC_32_EXTERNAL_INITIAL` holds. -/
theorem poseidon2_const_check_6 : poseidon2.const_check_6 ⦃ _ => True ⦄ := by
  obtain ⟨t, ht⟩ : ∃ t, poseidon2.BABYBEAR_POSEIDON2_RC_32_EXTERNAL_INITIAL = .ok t := by
    unfold poseidon2.BABYBEAR_POSEIDON2_RC_32_EXTERNAL_INITIAL; exact new_2d_array_ok _
  unfold poseidon2.const_check_6 poseidon2.BABYBEAR_POSEIDON2_HALF_FULL_ROUNDS
  simp only [ht, bind_tc_ok, CoreModels.core.slice.Slice.len,
    CoreModels.rust_primitives.slice.slice_length]
  step*
  all_goals
    simp only [not_not]
    apply UScalar.eq_of_val_eq
    simp_all

/-- The length assertion `baby-bear/src/poseidon2.rs:78` on `BABYBEAR_POSEIDON2_RC_32_EXTERNAL_FINAL` holds. -/
theorem poseidon2_const_check_7 : poseidon2.const_check_7 ⦃ _ => True ⦄ := by
  obtain ⟨t, ht⟩ : ∃ t, poseidon2.BABYBEAR_POSEIDON2_RC_32_EXTERNAL_FINAL = .ok t := by
    unfold poseidon2.BABYBEAR_POSEIDON2_RC_32_EXTERNAL_FINAL; exact new_2d_array_ok _
  unfold poseidon2.const_check_7 poseidon2.BABYBEAR_POSEIDON2_HALF_FULL_ROUNDS
  simp only [ht, bind_tc_ok, CoreModels.core.slice.Slice.len,
    CoreModels.rust_primitives.slice.slice_length]
  step*
  all_goals
    simp only [not_not]
    apply UScalar.eq_of_val_eq
    simp_all

/-- The length assertion `baby-bear/src/poseidon2.rs:80` on `BABYBEAR_POSEIDON2_RC_32_INTERNAL` holds. -/
theorem poseidon2_const_check_8 : poseidon2.const_check_8 ⦃ _ => True ⦄ := by
  obtain ⟨t, ht⟩ : ∃ t, poseidon2.BABYBEAR_POSEIDON2_RC_32_INTERNAL = .ok t := by
    unfold poseidon2.BABYBEAR_POSEIDON2_RC_32_INTERNAL; exact new_array_ok _
  unfold poseidon2.const_check_8 poseidon2.BABYBEAR_POSEIDON2_PARTIAL_ROUNDS_32
  simp only [ht, bind_tc_ok, CoreModels.core.slice.Slice.len,
    CoreModels.rust_primitives.slice.slice_length]
  step*

end P3BabyBearProofs
