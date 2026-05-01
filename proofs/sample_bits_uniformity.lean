/-
# Uniformity of Plonky3 `sample_bits` (issue #613 / PR #1050)

Formalises the uniform-sampling claim behind the rejection sampler merged
in <https://github.com/Plonky3/Plonky3/pull/1050> as a resolution for
<https://github.com/Plonky3/Plonky3/issues/613>.

## Construction (from PR #1050)

To draw `k` bits uniformly from a prime field `F_P`, sample `x : F_P`, then

  * reject if `x ≥ m` where `m = ⌊P / 2^k⌋ * 2^k`,
  * otherwise return `x mod 2^k`.

The rejection step keeps the accept interval `[0, m)` divisible by `2^k`,
so modular reduction distributes accepted elements equally across all
`2^k` buckets.

## What is proved

  * `reject_count`       -- rejected count equals `P mod 2^k`.
  * `reject_lt`          -- rejection count is strictly less than `2^k`
                            (so rejection probability is `< 2^k / P`).
  * `fibre_card`         -- each bucket `j < 2^k` has exactly `P / 2^k`
                            preimages in `[0, m)`.
  * `fibre_card_eq`      -- every pair of buckets has equal preimage count
                            (the conditional-uniformity statement).

Together, `fibre_card_eq` and `reject_lt` give the two facts Daniel
Lubarov asked for in the issue: conditional uniformity on acceptance, and
a quantitative bound on the rejection probability.

## Downstream PMF corollary (not stated here)

For any `PMF ℕ` concentrated uniformly on `Finset.range P`, conditioning
on membership in `Finset.range (acceptInterval P k)` and pushing forward
by `(· % 2^k)` yields a uniform `PMF (Fin (2^k))`.  The present file
stays in `Finset` / `ℕ` so the lemma is usable from any probability
layer (Mathlib `PMF`, measure theory, or raw counting).

## Build

  $ lake new plonky3_sample_bits math
  $ cp plonky3_sample_bits_uniform.lean plonky3_sample_bits/Plonky3SampleBits.lean
  $ cd plonky3_sample_bits && lake build

Or paste into the Lean 4 web editor (https://live.lean-lang.org) with
Mathlib loaded.
-/

import Mathlib.Data.Finset.Basic
import Mathlib.Data.Finset.Card
import Mathlib.Data.Finset.Image
import Mathlib.Tactic

namespace Plonky3.SampleBits

/-- Accept interval: the largest multiple of `2 ^ k` not exceeding `P`. -/
def acceptInterval (P k : ℕ) : ℕ := (P / 2 ^ k) * 2 ^ k

/-- The number of rejected field elements equals `P mod 2 ^ k`. -/
theorem reject_count (P k : ℕ) :
    P - acceptInterval P k = P % 2 ^ k := by
  unfold acceptInterval
  have h : P / 2 ^ k * 2 ^ k + P % 2 ^ k = P := by
    rw [Nat.mul_comm]; exact Nat.div_add_mod P (2 ^ k)
  omega

/-- The rejection count is strictly less than `2 ^ k`, so the rejection
    probability is bounded by `2 ^ k / P`. -/
theorem reject_lt (P k : ℕ) (hk : 0 < 2 ^ k) :
    P - acceptInterval P k < 2 ^ k := by
  rw [reject_count]
  exact Nat.mod_lt _ hk

/-- Preimage of bucket `j` under reduction mod `2 ^ k`, restricted to the
    accept interval. -/
def fibre (P k j : ℕ) : Finset ℕ :=
  (Finset.range (acceptInterval P k)).filter (fun x => x % 2 ^ k = j)

theorem mem_fibre {P k j x : ℕ} :
    x ∈ fibre P k j ↔ x < acceptInterval P k ∧ x % 2 ^ k = j := by
  unfold fibre
  rw [Finset.mem_filter, Finset.mem_range]

/-- The fibre over bucket `j < 2 ^ k` is in bijection with
    `Finset.range (P / 2 ^ k)` via `i ↦ j + i * 2 ^ k`. -/
theorem fibre_eq_image (P k j : ℕ) (hj : j < 2 ^ k) :
    fibre P k j
      = (Finset.range (P / 2 ^ k)).image (fun i => j + i * 2 ^ k) := by
  have hk : 0 < 2 ^ k := lt_of_le_of_lt (Nat.zero_le j) hj
  ext x
  rw [mem_fibre, Finset.mem_image]
  unfold acceptInterval
  refine ⟨fun ⟨hx_lt, hx_mod⟩ => ?_, ?_⟩
  · refine ⟨x / 2 ^ k,
      Finset.mem_range.mpr ((Nat.div_lt_iff_lt_mul hk).mpr hx_lt), ?_⟩
    have hdm : 2 ^ k * (x / 2 ^ k) + x % 2 ^ k = x :=
      Nat.div_add_mod x (2 ^ k)
    linarith [hx_mod]
  · rintro ⟨i, hi, rfl⟩
    rw [Finset.mem_range] at hi
    refine ⟨?_, ?_⟩
    · have hstep : (i + 1) * 2 ^ k ≤ P / 2 ^ k * 2 ^ k :=
        Nat.mul_le_mul_right (2 ^ k) hi
      have hexp : (i + 1) * 2 ^ k = i * 2 ^ k + 2 ^ k := by ring
      omega
    · rw [Nat.add_mul_mod_self_right, Nat.mod_eq_of_lt hj]

/-- Injectivity of the fibre parameterisation `i ↦ j + i * 2 ^ k`. -/
theorem fibre_param_injective (j k : ℕ) (hk : 0 < 2 ^ k) :
    Function.Injective (fun i : ℕ => j + i * 2 ^ k) := by
  intro a b hab
  simp only at hab
  have hmul : a * 2 ^ k = b * 2 ^ k := by omega
  exact Nat.eq_of_mul_eq_mul_right hk hmul

/-- Every bucket `j < 2 ^ k` has exactly `P / 2 ^ k` preimages in the
    accept interval.  This is the core uniformity lemma. -/
theorem fibre_card (P k j : ℕ) (hj : j < 2 ^ k) :
    (fibre P k j).card = P / 2 ^ k := by
  have hk : 0 < 2 ^ k := lt_of_le_of_lt (Nat.zero_le j) hj
  rw [fibre_eq_image P k j hj,
      Finset.card_image_of_injective _ (fibre_param_injective j k hk),
      Finset.card_range]

/-- Conditional uniformity: every pair of buckets has the same cardinality
    in the accept interval.  This is the statement the `sample_bits`
    sampler needs for Fiat-Shamir soundness analysis. -/
theorem fibre_card_eq (P k j₁ j₂ : ℕ)
    (h₁ : j₁ < 2 ^ k) (h₂ : j₂ < 2 ^ k) :
    (fibre P k j₁).card = (fibre P k j₂).card := by
  rw [fibre_card P k j₁ h₁, fibre_card P k j₂ h₂]

/-- The accept interval is the disjoint union of all fibres, so its
    cardinality factors as `(P / 2 ^ k) * 2 ^ k`.  Sanity-check lemma. -/
theorem card_range_acceptInterval (P k : ℕ) :
    (Finset.range (acceptInterval P k)).card = (P / 2 ^ k) * 2 ^ k := by
  rw [Finset.card_range]
  rfl

end Plonky3.SampleBits
