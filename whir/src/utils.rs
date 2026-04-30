//! Shared helpers for HVZK-WHIR (eprint 2026/391).
//!
//! References:
//! - <https://eprint.iacr.org/2026/391> (HVZK-WHIR),
//! - <https://eprint.iacr.org/2024/1586> (base WHIR).

use p3_field::Field;

/// Action of `ze*_n(ρ) := (1, ρ, …, ρ^{n-1})` on a coefficient vector,
/// computed by Horner's method (Definition 6.1 of eprint 2026/391).
///
/// # Math
///
/// ```text
///     ze*_n(ρ) · coeffs = Σ_i c_i · ρ^i
///                       = c_0 + ρ · (c_1 + ρ · (… + ρ · c_{n-1}))
/// ```
///
/// # Zero-evader bound
///
/// For any non-zero `v ∈ F^n`, the polynomial `Σ_i v_i · X^i` has degree
/// `< n`, so at most `n - 1` roots in `F`.
///
/// ```text
///     Pr_ρ[ze*_n(ρ) · v = 0] ≤ (n - 1) / |F|.
/// ```
///
/// # Cost
///
/// `n - 1` multiplications, `n - 1` additions, no allocation.
///
/// # Edge case
///
/// `coeffs.len() == 0` returns `F::ZERO` (empty sum).
//
// TODO: remove `#[allow(dead_code)]` once the HVZK sumcheck and
// code-switching modules land and call this helper.
#[allow(dead_code)]
#[inline]
pub(crate) fn eval_ze_star_n<F: Field>(point: F, coeffs: &[F]) -> F {
    // Horner: rfold walks coefficients right-to-left:
    //
    // c_0 + ρ·(c_1 + ρ·(... + ρ·c_{n-1})) with n-1 multiplications.
    coeffs.iter().rfold(F::ZERO, |acc, &c| acc * point + c)
}

/// Evaluate the polynomial `(msg ‖ rand)` (coefficients) at point `ρ`:
///
/// ```text
///     y = eval_ze_star_n(ρ, msg) + ρ^l · eval_ze_star_n(ρ, rand)
/// ```
///
/// Used as the prover's private out-of-domain (OOD) answer in HVZK
/// code-switching at `t = 1` (Construction 9.7 of eprint 2026/391).
///
/// - `rand` is fresh prover randomness;
/// - `ρ` is the verifier challenge.
///
/// # Why `y` hides `msg`
///
/// The `rand` contribution is `ρ^l · ze*_r(ρ) · rand`.
///
/// For `ρ ≠ 0` this is a non-zero `F`-linear functional in `rand`.
///
/// So uniform `rand` makes `y` uniform in `F`, independent of `msg`.
///
/// Privacy error `ζ_ze = 0`. (`ρ = 0` has measure `1/|F|`, soundness budget.)
///
/// # Relation to Lemma 9.3
///
/// Lemma 9.3 proves abstractly that `(r, 0)`-private zero-evaders
/// (Definition 9.2) exist by padding with a uniform random mask matrix.
///
/// We instead pick the deterministic `M = ρ^l · ze*_r(ρ)`: same property, no extra sampling.
///
/// # Cost
///
/// Two Horner passes plus one `ρ^l` (`log_2 l` field multiplications).
//
// TODO: remove `#[allow(dead_code)]` once the HVZK code-switching module
// lands and calls this helper.
#[allow(dead_code)]
#[inline]
pub(crate) fn padded_ood_t1<F: Field>(point: F, msg: &[F], rand: &[F]) -> F {
    // ze*_l(ρ) · msg.
    let msg_eval = eval_ze_star_n(point, msg);

    // ze*_r(ρ) · rand: local index `j ∈ [0, r)`.
    //
    // The ρ^l shift to global index `l + j` is one scalar multiplication below.
    let rand_eval = eval_ze_star_n(point, rand);

    // ρ^l
    //
    // With msg.len() == 0 this is ρ^0 = 1, recovering the unpadded form.
    let shift = point.exp_u64(msg.len() as u64);

    msg_eval + shift * rand_eval
}

#[cfg(test)]
mod tests {
    use alloc::vec::Vec;

    use p3_baby_bear::BabyBear;
    use p3_field::PrimeCharacteristicRing;
    use p3_field::extension::BinomialExtensionField;
    use p3_koala_bear::KoalaBear;
    use proptest::prelude::*;
    use rand::rngs::SmallRng;
    use rand::{RngExt, SeedableRng};

    use super::*;

    type F = BabyBear;
    type EF = BinomialExtensionField<KoalaBear, 4>;

    #[test]
    fn ze_star_n_empty_message_is_zero() {
        // Empty sum convention; non-trivial ρ surfaces any spurious ρ factor.
        assert_eq!(eval_ze_star_n(F::from_u64(7), &[]), F::ZERO);
    }

    #[test]
    fn ze_star_n_constant_is_independent_of_point() {
        // ze*_1(ρ) · (c) = c since the only term is c · ρ^0.
        let c = F::from_u64(42);
        // Two well-separated points; same answer either way.
        assert_eq!(eval_ze_star_n(F::from_u64(7), &[c]), c);
        assert_eq!(eval_ze_star_n(F::ZERO, &[c]), c);
    }

    #[test]
    fn ze_star_n_at_zero_returns_constant_coefficient() {
        // At ρ = 0 every ρ^i with i ≥ 1 vanishes; only c_0 survives.
        let msg = [F::from_u64(7), F::from_u64(11), F::from_u64(13)];
        assert_eq!(eval_ze_star_n(F::ZERO, &msg), F::from_u64(7));
    }

    #[test]
    fn ze_star_n_at_one_returns_sum_of_coefficients() {
        // At ρ = 1 every ρ^i = 1, so the result is Σ_i coeffs[i].
        let msg = [F::from_u64(3), F::from_u64(5), F::from_u64(7)];
        // 3 + 5 + 7 = 15.
        assert_eq!(eval_ze_star_n(F::ONE, &msg), F::from_u64(15));
    }

    #[test]
    fn padded_ood_t1_empty_randomness_equals_unpadded_eval() {
        // No randomness => second Horner pass is 0; reduces to msg eval.
        let msg = [F::from_u64(3), F::from_u64(5), F::from_u64(2)];
        let sigma = F::from_u64(7);
        assert_eq!(padded_ood_t1(sigma, &msg, &[]), eval_ze_star_n(sigma, &msg));
    }

    #[test]
    fn padded_ood_t1_empty_message_equals_unpadded_eval_on_rand() {
        // No message => first pass is 0, shift ρ^0 = 1; reduces to rand eval.
        let rand = [F::from_u64(2), F::from_u64(3), F::from_u64(5)];
        let sigma = F::from_u64(11);
        assert_eq!(
            padded_ood_t1(sigma, &[], &rand),
            eval_ze_star_n(sigma, &rand)
        );
    }

    #[test]
    fn padded_ood_t1_matches_concatenated_polynomial_eval() {
        // Catches errors in the ρ^l shift exponent or the msg/rand split.
        let msg = [F::from_u64(3), F::from_u64(5), F::from_u64(2)];
        let rand = [F::from_u64(7), F::from_u64(11)];
        let sigma = F::from_u64(13);

        // msg ‖ rand as one coefficient vector.
        let concatenated: Vec<F> = msg.iter().chain(rand.iter()).copied().collect();

        // Fused form must equal the polynomial-on-concatenation form.
        assert_eq!(
            padded_ood_t1(sigma, &msg, &rand),
            eval_ze_star_n(sigma, &concatenated)
        );
    }

    proptest! {
        #[test]
        fn prop_ze_star_n_matches_power_loop_basefield(
            coeffs in prop::collection::vec(any::<u32>(), 0..32),
            sigma_raw in any::<u32>(),
        ) {
            // Sizes 0..32: cover n ≤ 1 boundary up to typical mask sizes.
            let coeffs: Vec<F> = coeffs.into_iter().map(F::from_u32).collect();
            let sigma = F::from_u32(sigma_raw);

            // Horner under test.
            let horner = eval_ze_star_n(sigma, &coeffs);

            // Reference: explicit Σ_i c_i · ρ^i with a running power.
            let mut expected = F::ZERO;
            let mut power = F::ONE;
            for &c in coeffs.iter() {
                expected += c * power;
                power *= sigma;
            }

            prop_assert_eq!(horner, expected);
        }

        #[test]
        fn prop_ze_star_n_matches_power_loop_extension(
            seed in any::<u64>(),
            n in 0usize..32,
        ) {
            // Same identity over an extension; catches bugs base-field misses.
            let mut rng = SmallRng::seed_from_u64(seed);
            let coeffs: Vec<EF> = (0..n).map(|_| rng.random()).collect();
            let sigma: EF = rng.random();

            // Horner under test.
            let horner = eval_ze_star_n(sigma, &coeffs);

            // Reference: explicit running-power loop.
            let mut expected = EF::ZERO;
            let mut power = EF::ONE;
            for &c in coeffs.iter() {
                expected += c * power;
                power *= sigma;
            }

            prop_assert_eq!(horner, expected);
        }

        #[test]
        fn prop_padded_ood_t1_matches_concatenation_basefield(
            msg in prop::collection::vec(any::<u32>(), 0..16),
            rand in prop::collection::vec(any::<u32>(), 0..16),
            sigma_raw in any::<u32>(),
        ) {
            let msg: Vec<F> = msg.into_iter().map(F::from_u32).collect();
            let rand: Vec<F> = rand.into_iter().map(F::from_u32).collect();
            let sigma = F::from_u32(sigma_raw);

            // Fused form under test.
            let fused = padded_ood_t1(sigma, &msg, &rand);

            // Reference: build msg ‖ rand, evaluate as one polynomial.
            let concatenated: Vec<F> = msg.iter().chain(rand.iter()).copied().collect();
            let unfused = eval_ze_star_n(sigma, &concatenated);

            prop_assert_eq!(fused, unfused);
        }

        #[test]
        fn prop_padded_ood_t1_lemma_9_3_programmability(
            seed in any::<u64>(),
            msg_len in 1usize..16,
        ) {
            // (r, 0)-privacy at r = 1: any y_sim is reachable from some rand[0].
            let mut rng = SmallRng::seed_from_u64(seed);

            // Random message and target output to "explain".
            let msg: Vec<EF> = (0..msg_len).map(|_| rng.random()).collect();
            let y_sim: EF = rng.random();

            // Protocol rejects ρ = 0; loop for shrinking determinism.
            let mut sigma: EF = rng.random();
            while sigma == EF::ZERO {
                sigma = rng.random();
            }

            // Closed-form solve: rand[0] = (y_sim - msg_eval) / ρ^l.
            let msg_eval = eval_ze_star_n(sigma, &msg);
            let shift = sigma.exp_u64(msg_len as u64);
            let rand_0 = (y_sim - msg_eval) / shift;

            // Honest prover with this rand[0] must hit y_sim — Lemma 9.3 bijection.
            let real = padded_ood_t1(sigma, &msg, &[rand_0]);
            prop_assert_eq!(real, y_sim);
        }
    }
}
