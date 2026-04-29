//! Shared zero-evader helpers for HVZK-WHIR.
//!
//! This module provides internal helpers used by the honest-verifier zero-knowledge
//! extensions of the WHIR protocol. None of these are part of the public crate API.
//!
//! # References
//!
//! Chiesa, Fenzi, Weissenberg. "Zero-Knowledge IOPPs for Constrained Interleaved Codes."
//! ePrint 2026/391.

use alloc::vec::Vec;

use p3_field::Field;

/// Evaluates `ze_star_n(sigma) . message` from Definition 6.1.
///
/// `ze_star_n(sigma) = (1, sigma, ..., sigma^{n-1})`, so this computes ordinary
/// univariate polynomial evaluation in monomial basis:
///
/// ```text
/// ze_star_n(sigma) . message
///     = message[0] + message[1]*sigma + ... + message[n-1]*sigma^{n-1}
/// ```
///
/// Zero-evader error is `(n-1)/|F|`: at most `n-1` roots for a nonzero degree-(n-1)
/// polynomial (Schwartz-Zippel).
///
/// Shared by HVZK sumcheck batching (#1586) and code-switching OOD masking (#1587).
pub(crate) fn eval_ze_star_n<F: Field>(sigma: F, message: &[F]) -> F {
    // Horner's method: accumulate from the highest-degree coefficient downward.
    message
        .iter()
        .rfold(F::ZERO, |acc, &coeff| acc * sigma + coeff)
}

/// Evaluates the Lemma 9.3 padded zero-evader output for general t and r.
///
/// Given:
/// - `ze_eval` = `ze(rho) . f` in `F^t` (result of a base zero-evader on the message),
/// - `mask_matrix_row_major`: a t-by-r matrix M with columns spanning `F^t` (row-major),
/// - `randomness` = `s` in `F^r` (fresh randomness),
///
/// returns `ze_eval + M * s` in `F^t`.
///
/// Since the columns of M span `F^t`, the map `s -> M * s` is surjective onto `F^t`.
/// Therefore, for uniform `s`, the output `ze_eval + M * s` is uniformly distributed
/// over `F^t` regardless of `ze_eval`, achieving perfect privacy (`zeta_ze = 0`) as in
/// Lemma 9.3. The caller is responsible for ensuring M has rank t.
///
/// For any target simulated output `y_sim` in `F^t`, there exists some `s` in `F^r` with
/// `ze_eval + M * s = y_sim`. When `r = t` and M is invertible, that s is unique;
/// when `r > t`, it is generally not unique.
///
/// Used for OOD masking in HVZK code-switching (#1587).
pub(crate) fn eval_padded_zero_evader<F: Field>(
    ze_eval: &[F],
    mask_matrix_row_major: &[F],
    randomness: &[F],
) -> Vec<F> {
    let t = ze_eval.len();
    let r = randomness.len();
    assert_eq!(
        mask_matrix_row_major.len(),
        t * r,
        "mask_matrix must have t*r = {}*{} = {} elements",
        t,
        r,
        t * r
    );

    let mut out = ze_eval.to_vec();
    for row in 0..t {
        for col in 0..r {
            out[row] += mask_matrix_row_major[row * r + col] * randomness[col];
        }
    }
    out
}

/// Scalar specialization of the Lemma 9.3 padded zero-evader for t = 1, r = 1.
///
/// Computes `ze_star_n(sigma) . message + mask * randomness`.
///
/// When `mask != 0`, the map `s -> eval_ze_star_n(sigma, message) + mask * s` is a
/// bijection on F, so the output is uniformly distributed over F for uniform `randomness`.
/// This is the t = 1 instance of the perfect-privacy guarantee from Lemma 9.3.
pub(crate) fn eval_scalar_padded_ze_star_n<F: Field>(
    sigma: F,
    message: &[F],
    mask: F,
    randomness: F,
) -> F {
    eval_ze_star_n(sigma, message) + mask * randomness
}

#[cfg(test)]
mod tests {
    use alloc::vec;
    use alloc::vec::Vec;

    use p3_baby_bear::BabyBear;
    use p3_field::PrimeCharacteristicRing;

    use super::*;

    type F = BabyBear;

    // eval_ze_star_n tests

    #[test]
    fn test_ze_star_empty_message() {
        // Empty message: dot product with empty vector is 0.
        assert_eq!(eval_ze_star_n(F::from_u64(5), &[]), F::ZERO);
    }

    #[test]
    fn test_ze_star_constant_polynomial() {
        // ze_star_1(sigma) = (1,), so ze_star_1(sigma) . [c] = c regardless of sigma.
        let c = F::from_u64(42);
        assert_eq!(eval_ze_star_n(F::from_u64(7), &[c]), c);
        assert_eq!(eval_ze_star_n(F::ZERO, &[c]), c);
    }

    #[test]
    fn test_ze_star_linear_polynomial() {
        // ze_star_2(sigma) . [a, b] = a + b*sigma
        let a = F::from_u64(3);
        let b = F::from_u64(5);
        let sigma = F::from_u64(7);
        assert_eq!(eval_ze_star_n(sigma, &[a, b]), a + b * sigma);
    }

    #[test]
    fn test_ze_star_quadratic_polynomial() {
        // ze_star_3(sigma) . [a, b, c] = a + b*sigma + c*sigma^2
        // 1 + 2*4 + 3*16 = 57
        let a = F::from_u64(1);
        let b = F::from_u64(2);
        let c = F::from_u64(3);
        let sigma = F::from_u64(4);
        assert_eq!(
            eval_ze_star_n(sigma, &[a, b, c]),
            a + b * sigma + c * sigma.square()
        );
    }

    #[test]
    fn test_ze_star_at_zero() {
        // ze_star_n(0) . f = f[0] because all higher powers of 0 vanish.
        let message = [F::from_u64(7), F::from_u64(11), F::from_u64(13)];
        assert_eq!(eval_ze_star_n(F::ZERO, &message), F::from_u64(7));
    }

    #[test]
    fn test_ze_star_at_one() {
        // ze_star_n(1) . f = sum(f) because 1^k = 1 for all k.
        let message = [F::from_u64(3), F::from_u64(5), F::from_u64(7)];
        assert_eq!(eval_ze_star_n(F::ONE, &message), F::from_u64(15));
    }

    /// Exhaustive small-value check: Horner evaluation matches explicit monomial expansion.
    #[test]
    fn test_ze_star_matches_monomial_samples() {
        for a in 0u64..8 {
            for b in 0u64..8 {
                for c in 0u64..8 {
                    for sigma in 0u64..8 {
                        let a = F::from_u64(a);
                        let b = F::from_u64(b);
                        let c = F::from_u64(c);
                        let sigma = F::from_u64(sigma);
                        let explicit = a + b * sigma + c * sigma.square();
                        assert_eq!(eval_ze_star_n(sigma, &[a, b, c]), explicit);
                    }
                }
            }
        }
    }

    // eval_padded_zero_evader tests (Lemma 9.3, general t x r)

    #[test]
    fn test_padded_ze_t1_r1_scalar_identity() {
        // t=1, r=1, M = [1]. Output = ze_eval + 1*s = ze_eval + s.
        let ze_eval = [F::from_u64(10)];
        let mask = [F::ONE];
        let s = [F::from_u64(7)];
        let out = eval_padded_zero_evader(&ze_eval, &mask, &s);
        assert_eq!(out, vec![F::from_u64(17)]);
    }

    #[test]
    fn test_padded_ze_t2_r2_concrete() {
        // t=2, r=2. M = [[1, 2], [3, 4]] (row-major: [1, 2, 3, 4]), s = [7, 11].
        // out[0] = 3 + 1*7 + 2*11 = 3 + 7 + 22 = 32
        // out[1] = 5 + 3*7 + 4*11 = 5 + 21 + 44 = 70
        let ze_eval = [F::from_u64(3), F::from_u64(5)];
        let mask = [
            F::from_u64(1),
            F::from_u64(2),
            F::from_u64(3),
            F::from_u64(4),
        ];
        let s = [F::from_u64(7), F::from_u64(11)];
        let out = eval_padded_zero_evader(&ze_eval, &mask, &s);
        assert_eq!(out[0], F::from_u64(32));
        assert_eq!(out[1], F::from_u64(70));
    }

    /// Lemma 9.3 non-square (t=2, r=3): M has rank 2 via first two identity columns.
    ///
    /// Demonstrates that the general construction works for r > t: there exists
    /// randomness s achieving any target output. Here we set the free coordinate
    /// (col 2) to zero and solve using the two identity columns.
    #[test]
    fn test_padded_ze_t2_r3_surjective_matrix() {
        // M = [[1, 0, 2],
        //      [0, 1, 3]]
        // rank = 2 because the first two columns form I_2.
        let ze_eval = [F::from_u64(3), F::from_u64(5)];
        let mask = [
            F::ONE,
            F::ZERO,
            F::from_u64(2),
            F::ZERO,
            F::ONE,
            F::from_u64(3),
        ];
        let simulated_output = [F::from_u64(10), F::from_u64(20)];

        // Set free coordinate (col 2) = 0; solve via identity columns.
        let s = [
            simulated_output[0] - ze_eval[0],
            simulated_output[1] - ze_eval[1],
            F::ZERO,
        ];

        let real_output = eval_padded_zero_evader(&ze_eval, &mask, &s);
        assert_eq!(real_output[0], simulated_output[0]);
        assert_eq!(real_output[1], simulated_output[1]);
    }

    /// Lemma 9.3 simulator-equivalence (t=2, r=2, M = I_2).
    ///
    /// For any target simulated output y_sim in F^t, when M has rank t there exists
    /// randomness s such that ze_eval + M*s = y_sim. When r = t and M is invertible,
    /// that s is unique; when r > t, it is generally not unique.
    ///
    /// Here M = I_2 (square and invertible, r = t = 2), so s = M^{-1}(y_sim - ze_eval)
    /// = y_sim - ze_eval is the unique solution.
    #[test]
    fn test_padded_ze_square_simulator_equivalence() {
        let ze_eval = [F::from_u64(3), F::from_u64(5)];
        let mask = [F::ONE, F::ZERO, F::ZERO, F::ONE]; // I_2, row-major
        let simulated_output = [F::from_u64(10), F::from_u64(20)];

        // Unique programmed randomness: s = y_sim - ze_eval.
        let s = [
            simulated_output[0] - ze_eval[0],
            simulated_output[1] - ze_eval[1],
        ];

        let real_output = eval_padded_zero_evader(&ze_eval, &mask, &s);
        assert_eq!(real_output[0], simulated_output[0]);
        assert_eq!(real_output[1], simulated_output[1]);
    }

    // eval_scalar_padded_ze_star_n tests (Lemma 9.3, t=1, r=1)

    /// Lemma 9.3 simulator-equivalence (scalar, t=1, r=1).
    ///
    /// For any target simulated output y_sim, when mask != 0 the simulator programs
    /// the real randomness as s = (y_sim - base) / mask so real and simulated outputs
    /// agree exactly. This encodes the Lemma 9.3 bijection property: the real output
    /// y = ze_star_n(sigma).message + mask*s is uniform over F for uniform s,
    /// matching the simulated uniform distribution (perfect privacy, zeta_ze = 0).
    #[test]
    fn test_scalar_padded_ze_simulator_equivalence_fixed_messages() {
        let sigma = F::from_u64(7);
        let message = [F::from_u64(3), F::from_u64(5), F::from_u64(2)];
        let mask = F::from_u64(11); // nonzero

        let base = eval_ze_star_n(sigma, &message);

        // Choose an arbitrary simulated output; the simulator programs randomness to match.
        let simulated_output = F::from_u64(123_456);
        let programmed_randomness = (simulated_output - base) / mask;

        let real_output =
            eval_scalar_padded_ze_star_n(sigma, &message, mask, programmed_randomness);
        assert_eq!(real_output, simulated_output);
    }

    /// Bijection support: for fixed message and nonzero mask, 16 distinct randomness
    /// values produce 16 distinct outputs, confirming injectivity over this sample.
    /// This is only a small deterministic sanity check for the bijection argument;
    /// the actual argument is algebraic and covered by the simulator-equivalence test above.
    #[test]
    fn test_scalar_padded_ze_injectivity_sample() {
        let sigma = F::from_u64(7);
        let message = [F::from_u64(3), F::from_u64(5), F::from_u64(2)];
        let mask = F::from_u64(11);

        let outputs: Vec<F> = (0u64..16)
            .map(|s| eval_scalar_padded_ze_star_n(sigma, &message, mask, F::from_u64(s)))
            .collect();

        for i in 0..outputs.len() {
            for j in (i + 1)..outputs.len() {
                assert_ne!(
                    outputs[i], outputs[j],
                    "outputs[{i}] == outputs[{j}]: bijection property violated"
                );
            }
        }
    }

    /// Exhaustive small-value injectivity check over sigma, mask in 1..8, and s0 != s1.
    #[test]
    fn test_scalar_padded_ze_injectivity_samples() {
        let message = [F::from_u64(3), F::from_u64(5)];
        for sigma in 0u64..8 {
            for mask in 1u64..8 {
                let sigma = F::from_u64(sigma);
                let mask = F::from_u64(mask);
                for s0 in 0u64..8 {
                    for s1 in 0u64..8 {
                        if s0 == s1 {
                            continue;
                        }
                        let y0 =
                            eval_scalar_padded_ze_star_n(sigma, &message, mask, F::from_u64(s0));
                        let y1 =
                            eval_scalar_padded_ze_star_n(sigma, &message, mask, F::from_u64(s1));
                        assert_ne!(y0, y1);
                    }
                }
            }
        }
    }
}
