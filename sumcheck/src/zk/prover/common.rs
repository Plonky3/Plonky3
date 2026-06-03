//! Prelude helpers for the zero-knowledge sumcheck overlays.
//!
//! Covers mask sampling and auxiliary-target bookkeeping.
//! Both prefix and suffix overlays invoke them before the per-round loop starts.
//! Per-round polynomial assembly lives in a sibling module.

use alloc::vec::Vec;

use p3_challenger::{CanObserve, FieldChallenger};
use p3_commit::Mmcs;
use p3_field::{ExtensionField, Field};
use p3_matrix::Matrix;
use p3_zk_codes::ZkEncoding;
use rand::Rng;

use crate::zk::data::{MaskOracle, ZkSumcheckData};

/// Sample `k` mask polynomials, encode each, commit, and observe.
///
/// Implements step 1 of the masking layer.
///
/// # Arguments
///
/// - `k` — folding factor, one mask per sumcheck round.
/// - `encoding` — zero-knowledge encoder.
///   Defines the mask message space and draws a uniform sample on demand.
/// - `mmcs` — Merkle commitment scheme over the codeword alphabet.
/// - `challenger` — Fiat–Shamir transcript that absorbs each commitment.
/// - `rng` — driver for both the mask coefficients and the encoder's randomness budget.
///
/// # Returns
///
/// - Per-round coefficient vectors.
/// - Per-round commitment plus prover-side data.
///
/// The verifier sees only the commitments; the prover keeps the data for later openings.
#[allow(clippy::type_complexity)]
pub(super) fn sample_masks<F, Enc, M, Ch, R>(
    k: usize,
    encoding: &Enc,
    mmcs: &M,
    challenger: &mut Ch,
    rng: &mut R,
) -> (Vec<Vec<F>>, Vec<MaskOracle<F, Enc, M>>)
where
    F: Field,
    Enc: ZkEncoding<F>,
    Enc::Codeword: Matrix<F>,
    M: Mmcs<F>,
    Ch: CanObserve<M::Commitment>,
    R: Rng,
{
    // One uniform sample per round, drawn through the encoding so the message
    // space is whatever the encoding defines.
    let masks: Vec<Vec<F>> = (0..k).map(|_| encoding.sample_message(rng)).collect();

    // Encode each mask, commit, observe.
    //
    // Iterating the mask slice keeps memory layout contiguous for the wire-assembly loop that follows.
    let mask_oracles: Vec<MaskOracle<F, Enc, M>> = masks
        .iter()
        .map(|mask| {
            let codeword = encoding.encode(mask, rng);
            let (commit, prover_data) = mmcs.commit_matrix(codeword);
            challenger.observe(commit.clone());
            (commit, prover_data)
        })
        .collect();

    (masks, mask_oracles)
}

/// Compute the auxiliary target, record it on the transcript, and return the running endpoint sum.
///
/// Implements step 2 of the masking layer.
///
/// # Closed form
///
/// ```text
///     s_l(0) + s_l(1) = 2 * c_0 + sum_{i >= 1} c_i
///     aux_target      = 2^{k - 1} * sum_l ( s_l(0) + s_l(1) )
/// ```
///
/// # Returns
///
/// The running endpoint sum across all rounds.
/// The auxiliary target itself is written directly to the transcript record.
pub(super) fn observe_masks_and_mu_tilde<F, EF, Ch>(
    masks: &[Vec<EF>],
    k: usize,
    ell_zk: usize,
    challenger: &mut Ch,
    zk_data: &mut ZkSumcheckData<F, EF>,
) -> EF
where
    F: Field,
    EF: ExtensionField<F>,
    Ch: FieldChallenger<F>,
{
    // Endpoint sum used by the closed form.
    //
    //     s_l(0)        = c_0
    //     s_l(0)+s_l(1) = 2 * c_0 + sum_{i>=1} c_i
    let sum_endpoints_init: EF = masks
        .iter()
        .map(|mask| mask[0].double() + mask[1..].iter().copied().sum::<EF>())
        .sum();

    // Leading coefficient of the closed form.
    //
    // Guard the exponent: `k = 0` would underflow `k - 1` (usize) into a huge exponent.
    // The caller already enforces one round, so this is a local sanity net.
    debug_assert!(k >= 1, "auxiliary target requires at least one round");
    let two_to_k_minus_1 = EF::TWO.exp_u64((k - 1) as u64);
    let mu_tilde: EF = two_to_k_minus_1 * sum_endpoints_init;

    // Cross-check against the naive sum over the boolean cube.
    //
    // Guards against the multiplicity off-by-one that shipped in an
    // earlier external reference (eprint 2026/391 review thread).
    #[cfg(debug_assertions)]
    {
        let mut naive = EF::ZERO;
        for bits in 0..(1u64 << k) {
            for (l, mask) in masks.iter().enumerate() {
                let b_l = (bits >> l) & 1;
                // Separable mask: s_l(0) = c_0; s_l(1) = sum of all coefficients.
                let s_l_eval = if b_l == 0 {
                    mask[0]
                } else {
                    mask.iter().copied().sum::<EF>()
                };
                naive += s_l_eval;
            }
        }
        debug_assert_eq!(
            mu_tilde, naive,
            "auxiliary target closed form does not match the naive sum",
        );
    }

    // Observe the auxiliary target.
    challenger.observe_algebra_element(mu_tilde);
    // Pin the transcript record so the verifier reads the same metadata.
    zk_data.mu_tilde = mu_tilde;
    zk_data.ell_zk = ell_zk;

    sum_endpoints_init
}

#[cfg(test)]
mod tests {
    use alloc::vec;

    use p3_field::PrimeCharacteristicRing;
    use rand::SeedableRng;
    use rand::rngs::SmallRng;

    use super::*;
    use crate::zk::test_helpers::{EF, F, MyChallenger, make_setup};

    #[test]
    fn observe_masks_and_mu_tilde_matches_hand_computed_value() {
        // Fixture:
        //
        //     k       = 2
        //     ell_zk  = 3
        //     mask[0] = [1, 2, 3]
        //     mask[1] = [4, 5, 6]
        //
        // Per-mask endpoint sum (2 * c_0 + c_1 + c_2):
        //
        //     mask[0]: 2*1 + 2 + 3 = 7
        //     mask[1]: 2*4 + 5 + 6 = 19
        //
        // Aggregate:
        //
        //     sum_endpoints = 7 + 19           = 26
        //     mu_tilde      = 2^{k-1} * 26     = 52
        let masks = vec![
            vec![EF::from_u32(1), EF::from_u32(2), EF::from_u32(3)],
            vec![EF::from_u32(4), EF::from_u32(5), EF::from_u32(6)],
        ];
        let k = 2;
        let ell_zk = 3;

        let (perm, _, _) = make_setup(0, ell_zk);
        let mut challenger = MyChallenger::new(perm);
        let mut zk_data = ZkSumcheckData::<F, EF>::default();
        let endpoints = observe_masks_and_mu_tilde::<F, EF, _>(
            &masks,
            k,
            ell_zk,
            &mut challenger,
            &mut zk_data,
        );

        // Returned endpoint sum and transcript record fields.
        assert_eq!(endpoints, EF::from_u32(26));
        assert_eq!(zk_data.mu_tilde, EF::from_u32(52));
        assert_eq!(zk_data.ell_zk, ell_zk);
    }

    #[test]
    fn observe_masks_and_mu_tilde_advances_challenger() {
        // Invariant:
        //
        //     observe(mu_tilde)  ⇒  next sampled challenge differs from the
        //                           one a fresh challenger would have produced.
        //
        // Fixture: single mask of length 2; mu_tilde value is irrelevant.
        let masks = vec![vec![EF::from_u32(1), EF::from_u32(2)]];
        let k = 1;
        let ell_zk = 2;

        let (perm, _, _) = make_setup(0, ell_zk);

        // Baseline: sample without observing.
        let mut ch_baseline = MyChallenger::new(perm.clone());
        let baseline: EF = ch_baseline.sample_algebra_element();

        // Observed: sample after observing mu_tilde.
        let mut ch_observed = MyChallenger::new(perm);
        let mut zk_data = ZkSumcheckData::<F, EF>::default();
        let _ = observe_masks_and_mu_tilde::<F, EF, _>(
            &masks,
            k,
            ell_zk,
            &mut ch_observed,
            &mut zk_data,
        );
        let after_observe: EF = ch_observed.sample_algebra_element();

        assert_ne!(baseline, after_observe);
    }

    #[test]
    fn sample_masks_returns_k_masks_of_message_len() {
        // Output shape:
        //
        //     |masks|   = k
        //     |oracles| = k
        //     mask.len() = encoding.message_len()  for every mask
        let k = 3;
        let ell_zk = 4;
        let seed = 0;
        let (perm, mmcs, encoding) = make_setup(seed, ell_zk);
        let mut challenger = MyChallenger::new(perm);
        let mut rng = SmallRng::seed_from_u64(seed);

        let (masks, oracles) =
            sample_masks::<EF, _, _, _, _>(k, &encoding, &mmcs, &mut challenger, &mut rng);

        assert_eq!(masks.len(), k);
        assert_eq!(oracles.len(), k);
        for mask in &masks {
            assert_eq!(mask.len(), ell_zk);
        }
    }

    #[test]
    fn sample_masks_is_deterministic_under_matched_rng_seeds() {
        // Two parallel runs with matched RNG seeds and the same setup must
        // produce bit-identical masks and bit-identical commitments.
        //
        // Fixture: k = 2 rounds, ell_zk = 4, rng seed = 42.
        let k = 2;
        let ell_zk = 4;
        let seed = 42;
        let (perm, mmcs, encoding) = make_setup(seed, ell_zk);

        let mut ch1 = MyChallenger::new(perm.clone());
        let mut rng1 = SmallRng::seed_from_u64(seed);
        let (masks1, oracles1) =
            sample_masks::<EF, _, _, _, _>(k, &encoding, &mmcs, &mut ch1, &mut rng1);

        let mut ch2 = MyChallenger::new(perm);
        let mut rng2 = SmallRng::seed_from_u64(seed);
        let (masks2, oracles2) =
            sample_masks::<EF, _, _, _, _>(k, &encoding, &mmcs, &mut ch2, &mut rng2);

        assert_eq!(masks1, masks2);
        // Compare commitments only; prover-side data is not value-comparable.
        let commits1: Vec<_> = oracles1.iter().map(|(c, _)| c.clone()).collect();
        let commits2: Vec<_> = oracles2.iter().map(|(c, _)| c.clone()).collect();
        assert_eq!(commits1, commits2);
    }
}
