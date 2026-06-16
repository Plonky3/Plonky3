//! HVZK code-switching round (Construction 9.7, eprint 2026/391 Section 9.4).
//!
//! - Reduces a proximity claim about a source oracle to one about a smaller
//!   target oracle, in zero knowledge.
//! - This module owns the deterministic batching algebra only.
//! - The round loop, proof payloads, and transcript live in the pipeline.
//!
//! # Round shape
//!
//! ```text
//! prover  : sends fresh mask oracle encoding (r || s_pad)
//!           answers OOD points  y_i = ze*(rho_i) * (f || r || s_pad)^T
//! verifier: opens f at x_1..x_t, samples batching coefficients nu
//!           batches             mu' = nu_1*mu + sum nu*y_i + sum nu*f(x_j)
//! output  : linear relation over (f, carried masks, (r || s_pad))
//! ```
//!
//! - `r`: the source encoding randomness.
//! - `s_pad`: fresh, and what hides the OOD answers.
//!
//! # Privacy preconditions (not enforced here)
//!
//! - `pad_len >= t_ood`: one fresh pad coordinate per OOD answer.
//! - OOD points pairwise distinct and nonzero.
//! - Otherwise the answers leak a linear functional of the committed data.

use alloc::vec::Vec;

use p3_field::{Field, dot_product};
use thiserror::Error;

/// Errors in the Construction 9.7 batching step.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum CodeSwitchError {
    /// The number of private OOD answers does not match the OOD batching coefficients.
    #[error("private OOD answer count mismatch: expected {expected}, got {actual}")]
    PrivateOodAnswerCountMismatch { expected: usize, actual: usize },

    /// The number of source openings does not match the in-domain batching coefficients.
    #[error("source opening count mismatch: expected {expected}, got {actual}")]
    SourceOpeningCountMismatch { expected: usize, actual: usize },
}

/// Per-round ZK mask coefficient carrier.
///
/// - Produced by the batching step (both prover and verifier).
/// - Consumed by the next ZK sumcheck relation.
///
/// The pipeline builds one per round from the combination challenge.
#[derive(Debug, Clone)]
pub struct ZkMaskClaim<EF> {
    /// Coefficient on the inherited source claim (`nu_1`).
    pub base_claim_coeff: EF,
    /// Batching coefficients for OOD answers (`nu_{1+i}` for `i in [t_ood]`).
    pub ood_coeffs: Vec<EF>,
    /// Batching coefficients for in-domain openings.
    pub in_domain_coeffs: Vec<EF>,
}

impl<EF: Field> ZkMaskClaim<EF> {
    /// Computes the verifier-side batched claim `mu'`.
    ///
    /// ```text
    /// mu' = nu_1 * mu
    ///     + sum_i nu_{1+i} * y_i
    ///     + sum_j nu_{1+t_ood+j} * f(x_j)
    /// ```
    ///
    /// - `mu`: the scalar handed off by the previous reduction.
    /// - Any sumcheck scale on its source part is already baked in.
    /// - The covector side of that scale lives in the pipeline's symbolic
    ///   claim tracking, not here.
    pub fn batched_claim(
        &self,
        inherited_claim: EF,
        private_ood_answers: &[EF],
        source_openings: &[EF],
    ) -> Result<EF, CodeSwitchError> {
        // One batching coefficient per out-of-domain answer.
        if private_ood_answers.len() != self.ood_coeffs.len() {
            return Err(CodeSwitchError::PrivateOodAnswerCountMismatch {
                expected: self.ood_coeffs.len(),
                actual: private_ood_answers.len(),
            });
        }
        // One batching coefficient per in-domain opening.
        if source_openings.len() != self.in_domain_coeffs.len() {
            return Err(CodeSwitchError::SourceOpeningCountMismatch {
                expected: self.in_domain_coeffs.len(),
                actual: source_openings.len(),
            });
        }

        // sum_i nu_{1+i} * y_i over the out-of-domain answers.
        let ood_sum = dot_product::<EF, _, _>(
            self.ood_coeffs.iter().copied(),
            private_ood_answers.iter().copied(),
        );
        // sum_j nu_{1+t_ood+j} * f(x_j) over the in-domain openings.
        let in_domain_sum = dot_product::<EF, _, _>(
            self.in_domain_coeffs.iter().copied(),
            source_openings.iter().copied(),
        );

        // nu_1 * mu plus both batched transcript contributions.
        Ok(self.base_claim_coeff * inherited_claim + ood_sum + in_domain_sum)
    }
}

/// Builds the dense covector on a fresh code-switch mask `(r || pad)`.
///
/// Construction 9.7's batched claim touches the mask through two layers:
///
/// ```text
///     OOD point rho, coeff c : slot s gains  c * rho^{l + s}     s < r_len + pad_len
///     query x,       coeff c : slot s gains  c * x^{l + s}       s < r_len
/// ```
///
/// where `l` is the new message length.
///
/// - The mask slots continue the power run of `(message || randomness || pad)`.
/// - The fresh pad never appears in openings.
/// - Query layers therefore stop at the randomness slots.
#[must_use]
pub fn switch_mask_covector<EF: Field>(
    message_len: usize,
    source_randomness_len: usize,
    pad_len: usize,
    rho_ood_points: &[EF],
    ood_coeffs: &[EF],
    query_points: &[EF],
    query_coeffs: &[EF],
) -> Vec<EF> {
    assert_eq!(rho_ood_points.len(), ood_coeffs.len());
    assert_eq!(query_points.len(), query_coeffs.len());

    let mut covector = EF::zero_vec(source_randomness_len + pad_len);

    // OOD layers reach every mask slot.
    for (&rho, &coeff) in rho_ood_points.iter().zip(ood_coeffs) {
        let mut term = coeff * rho.exp_u64(message_len as u64);
        for dst in &mut covector {
            *dst += term;
            term *= rho;
        }
    }
    // Query layers reach only the randomness slots.
    for (&x, &coeff) in query_points.iter().zip(query_coeffs) {
        let mut term = coeff * x.exp_u64(message_len as u64);
        for dst in covector.iter_mut().take(source_randomness_len) {
            *dst += term;
            term *= x;
        }
    }

    covector
}

#[cfg(test)]
mod tests {
    //! Invariant tests for the Construction 9.7 batching algebra.

    use alloc::vec;
    use alloc::vec::Vec;

    use p3_baby_bear::BabyBear;
    use p3_field::PrimeCharacteristicRing;
    use p3_field::extension::BinomialExtensionField;
    use proptest::prelude::*;
    use rand::rngs::SmallRng;
    use rand::{RngExt, SeedableRng};

    use super::{CodeSwitchError, ZkMaskClaim, switch_mask_covector};
    use crate::utils::{eval_ze_star_n, padded_ood_t1};

    type F = BabyBear;
    type EF = BinomialExtensionField<F, 4>;

    #[test]
    fn batched_claim_matches_hand_computed_value() {
        // Invariant: mu' = nu_1*mu + nu_2*y_1 + nu_3*y_2 + nu_4*f(x_1).
        //
        // Fixture state: mu = 5, y = (7, 11), one opening f(x_1) = 13,
        // coefficients nu = (2, 3, 4, 6).
        let claim = ZkMaskClaim {
            base_claim_coeff: EF::from_u64(2),
            ood_coeffs: vec![EF::from_u64(3), EF::from_u64(4)],
            in_domain_coeffs: vec![EF::from_u64(6)],
        };

        let mu_prime = claim
            .batched_claim(
                EF::from_u64(5),
                &[EF::from_u64(7), EF::from_u64(11)],
                &[EF::from_u64(13)],
            )
            .unwrap();

        // 2*5 + 3*7 + 4*11 + 6*13 = 10 + 21 + 44 + 78 = 153.
        assert_eq!(mu_prime, EF::from_u64(153));
    }

    #[test]
    fn batched_claim_rejects_ood_count_mismatch() {
        // Fixture state: 2 OOD coefficients but only 1 answer.
        //
        //     ood_coeffs: [nu_2, nu_3]
        //     answers   : [y_1]          -> 1 != 2 -> reject
        let claim = ZkMaskClaim {
            base_claim_coeff: EF::from_u64(1),
            ood_coeffs: vec![EF::from_u64(2), EF::from_u64(3)],
            in_domain_coeffs: vec![EF::from_u64(4)],
        };

        let err = claim
            .batched_claim(EF::from_u64(9), &[EF::from_u64(10)], &[EF::from_u64(11)])
            .unwrap_err();

        assert_eq!(
            err,
            CodeSwitchError::PrivateOodAnswerCountMismatch {
                expected: 2,
                actual: 1
            }
        );
    }

    #[test]
    fn batched_claim_rejects_source_opening_count_mismatch() {
        // Fixture state: 2 in-domain coefficients but only 1 opening.
        //
        //     in_domain_coeffs: [nu_3, nu_4]
        //     openings        : [f(x_1)]     -> 1 != 2 -> reject
        let claim = ZkMaskClaim {
            base_claim_coeff: EF::from_u64(1),
            ood_coeffs: vec![EF::from_u64(2)],
            in_domain_coeffs: vec![EF::from_u64(3), EF::from_u64(4)],
        };

        let err = claim
            .batched_claim(EF::from_u64(9), &[EF::from_u64(10)], &[EF::from_u64(11)])
            .unwrap_err();

        assert_eq!(
            err,
            CodeSwitchError::SourceOpeningCountMismatch {
                expected: 2,
                actual: 1
            }
        );
    }

    #[test]
    fn ood_answers_leak_committed_data_without_enough_pad() {
        // Invariant: joint privacy of t_ood answers needs pad_len >= t_ood.
        //
        // Fixture state: 2 answers share a SINGLE fresh pad coordinate s.
        //
        //     y_1 = <(f || r), ze*(rho_1)> + s * rho_1^{ell+r_len}
        //     y_2 = <(f || r), ze*(rho_2)> + s * rho_2^{ell+r_len}
        //
        //     rho_2^{ell+r_len} * y_1 - rho_1^{ell+r_len} * y_2   // s cancels
        //     -> public linear functional of the committed (f || r)
        //     -> one leaked query against the source ZK budget
        let f = vec![EF::from_u64(3), EF::from_u64(5), EF::from_u64(7)];
        let r = vec![EF::from_u64(11), EF::from_u64(13)];
        let s_pad = vec![EF::from_u64(17)];
        let mask = [r.clone(), s_pad].concat();
        let rho = [EF::from_u64(19), EF::from_u64(23)];

        // Honest prover answers for the under-padded shape.
        let y = [
            padded_ood_t1(rho[0], &f, &mask),
            padded_ood_t1(rho[1], &f, &mask),
        ];

        // The pad coefficient in answer i is rho_i^{ell + r_len}.
        let committed = [f, r].concat();
        let shift = committed.len() as u64;
        let c1 = rho[0].exp_u64(shift);
        let c2 = rho[1].exp_u64(shift);

        // The pad contributions c1 * s and c2 * s cancel in the combination,
        // leaving only committed data.
        let eliminated = c2 * y[0] - c1 * y[1];
        let predicted =
            c2 * eval_ze_star_n(rho[0], &committed) - c1 * eval_ze_star_n(rho[1], &committed);

        assert_eq!(
            eliminated, predicted,
            "under-padded OOD answers must reveal this committed functional",
        );
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(64))]

        #[test]
        fn prop_switch_mask_covector_matches_slotwise_reference(seed in any::<u64>()) {
            // Invariant: for every shape, slot s of the mask covector is
            //
            //     sum_i c_i * rho_i^{l+s}                      (all slots)
            //   + sum_q d_q * x_q^{l+s}    if s < r_len        (query layers
            //                                                   skip the pad)
            //
            // Dimensions derive from the seed; values replay deterministically.
            let mut rng = SmallRng::seed_from_u64(seed);
            let message_len = 1 + (seed % 5) as usize;
            let r_len = ((seed / 5) % 3) as usize;
            let pad_len = ((seed / 15) % 3) as usize;
            let t_ood = ((seed / 45) % 3) as usize;
            let t = ((seed / 135) % 3) as usize;

            let rho_points: Vec<EF> = (0..t_ood).map(|_| rng.random()).collect();
            let ood_coeffs: Vec<EF> = (0..t_ood).map(|_| rng.random()).collect();
            let query_points: Vec<EF> = (0..t).map(|_| rng.random()).collect();
            let query_coeffs: Vec<EF> = (0..t).map(|_| rng.random()).collect();

            // Slot-by-slot reference, written directly from the formula.
            let expected: Vec<EF> = (0..r_len + pad_len)
                .map(|slot| {
                    let exponent = (message_len + slot) as u64;
                    // OOD layers reach every slot.
                    let mut value: EF = rho_points
                        .iter()
                        .zip(&ood_coeffs)
                        .map(|(&rho, &c)| c * rho.exp_u64(exponent))
                        .sum();
                    // Query layers stop at the randomness slots.
                    if slot < r_len {
                        value += query_points
                            .iter()
                            .zip(&query_coeffs)
                            .map(|(&x, &c)| c * x.exp_u64(exponent))
                            .sum::<EF>();
                    }
                    value
                })
                .collect();

            let covector = switch_mask_covector(
                message_len,
                r_len,
                pad_len,
                &rho_points,
                &ood_coeffs,
                &query_points,
                &query_coeffs,
            );

            prop_assert_eq!(covector, expected);
        }
    }
}
