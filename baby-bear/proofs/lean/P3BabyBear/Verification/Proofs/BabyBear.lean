/- Hand-written properties of the `BabyBear` constructors: `BabyBear::new`
(which is `MontyField31::new` at `BabyBearParameters`) and the table
constructors `new_array` and `new_2d_array`. -/
import P3BabyBear.Verification.Proofs.BabyBearParameters
open Aeneas Aeneas.Std CoreModels

namespace p3_baby_bear

open baby_bear.BabyBearParameters (MontyParams)

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

end p3_baby_bear
