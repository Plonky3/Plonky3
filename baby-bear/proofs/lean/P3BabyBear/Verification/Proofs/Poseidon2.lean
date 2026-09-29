/- Hand-written properties of `baby-bear/src/poseidon2.rs`: its round-constant
length assertions, `const _: () = assert!(..)`. aeneas names the first in each
module `_` and the rest `__N`; post-extraction patch 030 renames those to
`const_check_N`. `poseidon2._` (on `BABYBEAR_POSEIDON2_RC_16_EXTERNAL_INITIAL`) is not proved here. -/
import P3BabyBear.Verification.Proofs.BabyBear
open Aeneas Aeneas.Std CoreModels

namespace p3_baby_bear

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
