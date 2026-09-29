/- Hand-written properties of `baby-bear/src/poseidon1.rs`: its round-constant
length assertions, `const _: () = assert!(..)`. aeneas names the first in each
module `_` and the rest `__N`; post-extraction patch 030 renames those to
`const_check_N`. `poseidon1._` (on `BABYBEAR_POSEIDON1_RC_16`) is not proved here. -/
import P3BabyBear.Verification.Proofs.BabyBear
open Aeneas Aeneas.Std CoreModels

namespace p3_baby_bear

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

end p3_baby_bear
