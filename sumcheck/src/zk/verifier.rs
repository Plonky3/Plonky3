//! HVZK verifier with affine-chain replay; covers both stacked binding modes.

use alloc::vec::Vec;

use p3_challenger::{CanObserve, FieldChallenger, GrindingChallenger};
use p3_commit::Mmcs;
use p3_field::{ExtensionField, Field, HornerIter};
use p3_multilinear_util::point::Point;

use super::data::{ZkSumcheckData, ZkVerifierHandoff};
use crate::error::SumcheckError;
use crate::layout::{LayoutStrategy, Verifier};
use crate::strategy::VariableOrder;
use crate::table::{OpeningEvals, OpeningRequest, TableShape};

/// HVZK verifier for the stacked sumcheck.
///
/// The wire format and the affine consistency identity match across binding modes.
/// Callers pick a constructor per binding mode so the inner layout verifier lifts opening points through the right selectors.
///
/// Per round, the verifier:
///
/// - reads wire `[c_0, c_2, c_3, ..., c_d]` (linear coefficient dropped),
/// - reconstructs `c_1` from `h_j(0) + h_j(1) = target`,
/// - checks the proof-of-work witness when enabled,
/// - samples `gamma_j` and sets the next target to `h_j(gamma_j)`.
#[derive(Debug, Clone)]
pub struct ZkVerifier<F, EF>
where
    F: Field,
    EF: ExtensionField<F>,
{
    /// Plain stacked-layout verifier holding the claims that fix `mu`.
    inner: Verifier<F, EF>,
}

impl<F, EF> ZkVerifier<F, EF>
where
    F: Field,
    EF: ExtensionField<F>,
{
    /// Build the verifier for a prover running in prefix-binding mode.
    ///
    /// The layout strategy reverses the selector bit order and folds variables low-to-high.
    /// A drift-guard test in this module pins these settings against the non-private prefix layout.
    pub fn new_prefix(table_shapes: &[TableShape]) -> Self {
        Self {
            inner: Verifier::new(
                table_shapes,
                LayoutStrategy::new(true, VariableOrder::Prefix),
            ),
        }
    }

    /// Build the verifier for a prover running in suffix-binding mode.
    ///
    /// The layout strategy leaves the selector bit order untouched and folds variables high-to-low.
    /// A drift-guard test in this module pins these settings against the non-private suffix layout.
    pub fn new_suffix(table_shapes: &[TableShape]) -> Self {
        Self {
            inner: Verifier::new(
                table_shapes,
                LayoutStrategy::new(false, VariableOrder::Suffix),
            ),
        }
    }

    /// Return the layout strategy carried by this verifier.
    ///
    /// Downstream consumers use it to dispatch on the binding direction.
    pub const fn strategy(&self) -> LayoutStrategy {
        self.inner.strategy()
    }

    fn validate_shape(
        zk_data: &ZkSumcheckData<F, EF>,
        ell_zk: usize,
        folding_factor: usize,
        pow_bits: usize,
    ) -> Result<(), SumcheckError> {
        assert!(F::TWO != F::ZERO, "Lemma 6.4 requires char(F) != 2");
        assert!(
            ell_zk >= 3,
            "mask degree ell_zk - 1 must cover the degree-2 plain piece (ell_zk >= 3)",
        );
        assert!(folding_factor >= 1, "sumcheck requires at least one round");

        if zk_data.ell_zk != ell_zk {
            return Err(SumcheckError::EllZkMismatch {
                expected: ell_zk,
                actual: zk_data.ell_zk,
            });
        }
        if zk_data.round_coefficients.len() != folding_factor {
            return Err(SumcheckError::RoundCountMismatch {
                expected: folding_factor,
                actual: zk_data.round_coefficients.len(),
            });
        }
        let expected_pow = if pow_bits > 0 { folding_factor } else { 0 };
        if zk_data.pow_witnesses.len() != expected_pow {
            return Err(SumcheckError::PowWitnessCountMismatch {
                expected: expected_pow,
                actual: zk_data.pow_witnesses.len(),
            });
        }

        let h_size = ell_zk.max(3);
        let wire_size = h_size - 1;
        for (idx, wire) in zk_data.round_coefficients.iter().enumerate() {
            if wire.len() != wire_size {
                return Err(SumcheckError::WireSizeMismatch {
                    round: idx + 1,
                    expected: wire_size,
                    actual: wire.len(),
                });
            }
        }

        Ok(())
    }

    fn replay_claim_unchecked<M, Ch>(
        zk_data: &ZkSumcheckData<F, EF>,
        mask_commitment: &M::Commitment,
        folding_factor: usize,
        pow_bits: usize,
        claimed_sum: EF,
        challenger: &mut Ch,
    ) -> Result<ZkVerifierHandoff<EF>, SumcheckError>
    where
        M: Mmcs<EF>,
        Ch: FieldChallenger<F> + GrindingChallenger<Witness = F> + CanObserve<M::Commitment>,
    {
        challenger.observe(mask_commitment.clone());
        challenger.observe_algebra_element(zk_data.mu_tilde);
        let eps: EF = challenger.sample_algebra_element();

        let mut target: EF = eps * claimed_sum + zk_data.mu_tilde;
        let mut randomness: Vec<EF> = Vec::with_capacity(folding_factor);

        for (j_idx, wire) in zk_data.round_coefficients.iter().enumerate() {
            let c0 = wire[0];
            let high_sum: EF = wire[1..].iter().copied().sum();
            let c1 = target - c0.double() - high_sum;

            challenger.observe_algebra_slice(wire);

            if pow_bits > 0 && !challenger.check_witness(pow_bits, zk_data.pow_witnesses[j_idx]) {
                return Err(SumcheckError::InvalidPowWitness);
            }

            let gamma_j: EF = challenger.sample_algebra_element();
            target = core::iter::once(c0)
                .chain(core::iter::once(c1))
                .chain(wire[1..].iter().copied())
                .horner(gamma_j);
            randomness.push(gamma_j);
        }

        Ok(ZkVerifierHandoff {
            randomness: Point::new(randomness),
            claimed_residual: target,
            eps,
        })
    }

    /// Records opening claims at the current points and at their repeat-last successor points on the inner verifier.
    ///
    /// # Errors
    ///
    /// - Propagates [`SumcheckError::OpeningShapeMismatch`] from the inner verifier.
    pub fn add_claim<Ch>(
        &mut self,
        table_idx: usize,
        batch: &OpeningRequest,
        evals: &OpeningEvals<EF>,
        challenger: &mut Ch,
    ) -> Result<(), SumcheckError>
    where
        Ch: FieldChallenger<F> + GrindingChallenger<Witness = F>,
    {
        // Delegate; the HVZK overlay carries no extra state at claim time.
        self.inner.add_claim(table_idx, batch, evals, challenger)
    }

    /// Records a virtual evaluation claim on the inner verifier.
    pub fn add_virtual_eval<Ch>(&mut self, eval: EF, challenger: &mut Ch)
    where
        Ch: FieldChallenger<F> + GrindingChallenger<Witness = F>,
    {
        // Same delegation pattern as concrete openings.
        self.inner.add_virtual_eval(eval, challenger);
    }

    /// Claim sum `mu` weighted by powers of `alpha`.
    ///
    /// Reads only the recorded claims, not any witness data.
    /// Used by the witness-free simulator to derive `mu` without re-implementing the alpha-power loop.
    pub(crate) fn sum(&self, alpha: EF) -> EF {
        self.inner.sum(alpha)
    }

    /// Replays the prover's HVZK sumcheck transcript.
    ///
    /// # Phases
    ///
    /// 1. Reject malformed shapes up front.
    /// 2. Sample alpha and derive `mu` from the recorded claims.
    /// 3. Absorb mask commits and `mu_tilde`, then sample `eps`.
    /// 4. Walk the round chain: reconstruct `c_1`, check PoW, sample `gamma_j`, advance the target by Horner evaluation.
    ///
    /// # Returns
    ///
    /// - Vector of per-round challenges `gamma_1, ..., gamma_k`.
    /// - Residual claim `target = h_k(gamma_k)`, fed to the downstream committed-sumcheck reduction.
    ///
    /// # Round-by-round soundness
    ///
    /// Each round rebuilds `c_1` from the affine identity `h(0) + h(1) = target`.
    /// A wire tampered in one coordinate still satisfies the per-round check.
    /// So this method does **not** reject it locally and may return `Ok`.
    /// The rebuilt `c_1` shifts `gamma_j`, diverging the final `target` through Fiat-Shamir.
    /// Treat the returned `target`, not the absence of an error, as the soundness-bearing output.
    ///
    /// # Errors
    ///
    /// - Mismatch between the verifier-side and proof-side mask code length.
    /// - Wrong number of rounds or PoW witnesses.
    /// - A per-round wire of the wrong shape.
    /// - A failing proof-of-work witness check.
    ///
    /// # Panics
    ///
    /// - Base field characteristic is `2`.
    /// - Mask code message length is below `3`.
    /// - Folding factor is `0`.
    #[allow(clippy::too_many_arguments)]
    pub fn into_sumcheck<M, Ch>(
        self,
        zk_data: &ZkSumcheckData<F, EF>,
        mask_commitment: &M::Commitment,
        ell_zk: usize,
        folding_factor: usize,
        pow_bits: usize,
        challenger: &mut Ch,
    ) -> Result<ZkVerifierHandoff<EF>, SumcheckError>
    where
        M: Mmcs<EF>,
        Ch: FieldChallenger<F> + GrindingChallenger<Witness = F> + CanObserve<M::Commitment>,
    {
        // Phase 1: shape checks (input validation before Construction 6.3 replay).
        Self::validate_shape(zk_data, ell_zk, folding_factor, pow_bits)?;

        // Phase 2: transcript prelude (matches the prover byte-for-byte; replays Construction 6.3 setup).

        // Sample alpha, then derive mu from the recorded claims.
        let alpha: EF = challenger.sample_algebra_element();
        let mu = self.inner.sum(alpha);

        // Phase 3: absorb the mask commitment and mu_tilde, sample eps, and walk the round chain.
        Self::replay_claim_unchecked::<M, _>(
            zk_data,
            mask_commitment,
            folding_factor,
            pow_bits,
            mu,
            challenger,
        )
    }

    /// Replays an HVZK sumcheck transcript for an already-batched scalar claim.
    ///
    /// This is the verifier-side counterpart of
    /// [`crate::strategy::SumcheckProver::into_zk_sumcheck`]. It skips the
    /// claim-batching `alpha` prelude because the caller already supplies the
    /// scalar claim that the masked sumcheck should prove. The scalar is
    /// absorbed before the masking prelude so this standalone residual-claim
    /// API is transcript-bound even without recorded layout claims.
    #[allow(clippy::too_many_arguments)]
    pub fn verify_claim<M, Ch>(
        zk_data: &ZkSumcheckData<F, EF>,
        mask_commitment: &M::Commitment,
        ell_zk: usize,
        folding_factor: usize,
        pow_bits: usize,
        claimed_sum: EF,
        challenger: &mut Ch,
    ) -> Result<ZkVerifierHandoff<EF>, SumcheckError>
    where
        M: Mmcs<EF>,
        Ch: FieldChallenger<F> + GrindingChallenger<Witness = F> + CanObserve<M::Commitment>,
    {
        Self::validate_shape(zk_data, ell_zk, folding_factor, pow_bits)?;
        challenger.observe_algebra_element(claimed_sum);
        Self::replay_claim_unchecked::<M, _>(
            zk_data,
            mask_commitment,
            folding_factor,
            pow_bits,
            claimed_sum,
            challenger,
        )
    }
}

#[cfg(test)]
mod tests {
    use p3_field::{Field, PackedValue, PrimeCharacteristicRing};
    use proptest::prelude::*;

    use super::*;
    use crate::layout::{Layout, PrefixProver, SuffixProver, TableShape};
    use crate::strategy::VariableOrder;
    use crate::zk::test_helpers::{EF, F, MyMmcs, run_prover};

    #[test]
    fn verifier_strategy_matches_non_private_layouts() {
        // Drift guard.
        //
        // Each HVZK verifier constructor must carry the same layout
        // strategy as its non-private layout counterpart.
        // A mismatch would silently lift claim points under the wrong
        // selector encoding.
        //
        // Fixture state: 1 table of arity 4 with 1 column.
        // The shape is immaterial to `strategy()` but `Verifier::new`
        // requires a non-empty slice.
        let shapes = &[TableShape::new(4, 1)];

        // Per-mode pin against the non-private strategy.
        let zk_prefix = ZkVerifier::<F, EF>::new_prefix(shapes);
        assert_eq!(zk_prefix.strategy(), PrefixProver::<F, EF>::strategy());

        let zk_suffix = ZkVerifier::<F, EF>::new_suffix(shapes);
        assert_eq!(zk_suffix.strategy(), SuffixProver::<F, EF>::strategy());

        // Cross-mode pin.
        //
        // A refactor could collapse both strategies to a single constant
        // and the per-mode checks above would still pass.
        // These three assertions catch that.
        assert_ne!(
            zk_prefix.strategy().variable_order,
            zk_suffix.strategy().variable_order,
        );
        assert_eq!(zk_prefix.strategy().variable_order, VariableOrder::Prefix);
        assert_eq!(zk_suffix.strategy().variable_order, VariableOrder::Suffix);
    }

    /// Drives the PoW-witness tampering invariant for one binding mode.
    ///
    /// - Runs an honest prover with grinding enabled.
    /// - Bumps the first round's PoW witness by one.
    /// - Asserts the verifier rejects with [`SumcheckError::InvalidPowWitness`].
    fn forged_pow_witness_rejected_case(binding: VariableOrder) {
        // Fixture state:
        //
        //     n_vars       = 6        (witness has 2^6 = 64 evaluations)
        //     folding      = 2        (two sumcheck rounds → two PoW witnesses)
        //     ell_zk       = 4        (mask polynomial degree 3)
        //     num_virtual  = 1        (one virtual claim to seed mu)
        //     pow_bits     = 16
        //     seed         = 0
        //
        // Why one seed is enough.
        //
        //     |valid preimages| / |F| = 2^{32 - 16} / 2^32 = 2^{-16}
        //
        // Random tampering passes the difficulty check with probability 2^{-16}.
        // A single concrete seed is therefore a high-confidence test.
        let n_vars = 6;
        let folding_factor = 2;
        let ell_zk = 4;
        let num_virtual = 1;
        let seed = 0u64;
        let pow_bits = 16;

        // Honest run via the binding-parameterised helper.
        let mut run = run_prover(
            binding,
            n_vars,
            folding_factor,
            ell_zk,
            0,
            num_virtual,
            pow_bits,
            seed,
        );

        // Pre-mutation sanity:
        //
        //     pow_witnesses.len() == folding_factor
        assert_eq!(run.zk_data.pow_witnesses.len(), folding_factor);

        // Mutation:
        //
        //     pow_witnesses: [ w_0,        w_1 ]
        //                    [ w_0 + 1,    w_1 ]   ← tampered
        //
        // The verifier rederives the round-1 difficulty challenge from the
        // honest transcript prefix, then checks `w_0 + 1` against it.
        run.zk_data.pow_witnesses[0] += F::ONE;

        // Verifier replay against the tampered proof.
        let result = run.verifier.clone().into_sumcheck::<MyMmcs, _>(
            &run.zk_data,
            &run.mask_commitment,
            ell_zk,
            folding_factor,
            pow_bits,
            &mut run.verifier_challenger,
        );

        assert!(
            matches!(result, Err(SumcheckError::InvalidPowWitness)),
            "verifier accepted a forged PoW witness in binding {binding:?}; got {result:?}",
        );
    }

    #[test]
    fn forged_pow_witness_rejected_prefix() {
        // Prefix path: a tampered PoW witness must be rejected with `InvalidPowWitness`.
        forged_pow_witness_rejected_case(VariableOrder::Prefix);
    }

    #[test]
    fn forged_pow_witness_rejected_suffix() {
        // Suffix path: same invariant.
        //
        // PoW handling lives in the wire schema, which both binding modes
        // share byte-for-byte; this case pins that fact against the
        // binding-mode dispatch.
        forged_pow_witness_rejected_case(VariableOrder::Suffix);
    }

    /// Drives the `ell_zk` mismatch invariant for one binding mode.
    ///
    /// - Honest prover commits with `ell_zk = 4`.
    /// - Verifier replays with `ell_zk = 5`.
    /// - Asserts the verifier rejects with [`SumcheckError::EllZkMismatch`].
    ///
    /// # Why a dedicated error is needed
    ///
    /// The wire-shape check is non-injective on `{2, 3}`:
    ///
    /// ```text
    ///     ell_zk = 2  →  wire_size = max(2, 3) - 1 = 2
    ///     ell_zk = 3  →  wire_size = max(3, 3) - 1 = 2
    /// ```
    ///
    /// A swapped mask length would slip past the shape check.
    /// The dedicated mismatch error closes that gap.
    fn ell_zk_mismatch_rejected_case(binding: VariableOrder) {
        // Fixture state:
        //
        //     n_vars       = 6
        //     folding      = 2
        //     ell_zk       = 4        (prover-side)
        //     num_virtual  = 1
        //     pow_bits     = 0        (PoW disabled to isolate the shape check)
        //     seed         = 0
        //
        // Mutation: the verifier replays with `wrong_ell_zk = 5`.
        let n_vars = 6;
        let folding_factor = 2;
        let ell_zk = 4;
        let num_virtual = 1;
        let seed = 0u64;
        let pow_bits = 0;

        let mut run = run_prover(
            binding,
            n_vars,
            folding_factor,
            ell_zk,
            0,
            num_virtual,
            pow_bits,
            seed,
        );

        // Verifier replay with the wrong ell_zk parameter.
        let wrong_ell_zk = ell_zk + 1;
        let result = run.verifier.clone().into_sumcheck::<MyMmcs, _>(
            &run.zk_data,
            &run.mask_commitment,
            wrong_ell_zk,
            folding_factor,
            pow_bits,
            &mut run.verifier_challenger,
        );

        // Expect the dedicated mismatch error with the exact (expected, actual) values.
        assert!(
            matches!(
                result,
                Err(SumcheckError::EllZkMismatch { expected, actual })
                    if expected == wrong_ell_zk && actual == ell_zk
            ),
            "verifier should have rejected ell_zk mismatch in binding {binding:?}; got {result:?}",
        );
    }

    #[test]
    fn ell_zk_mismatch_rejected_prefix() {
        // Prefix path: verifier rejects when its `ell_zk` disagrees with the proof.
        ell_zk_mismatch_rejected_case(VariableOrder::Prefix);
    }

    #[test]
    fn ell_zk_mismatch_rejected_suffix() {
        // Suffix path: same invariant, exercised through the suffix dispatch.
        ell_zk_mismatch_rejected_case(VariableOrder::Suffix);
    }

    /// Drives the wire-tampering invariant for one binding mode.
    ///
    /// - Runs an honest prover.
    /// - Picks one wire coordinate at uniformly random.
    /// - Bumps that coordinate by one on a clone of the honest transcript.
    /// - Replays the verifier from the same post-prover state on both
    ///   transcripts.
    /// - Asserts the two final targets diverge.
    ///
    /// # Why local checks are not enough
    ///
    /// - The affine reconstruction of `c_1` forces the per-round identity
    ///   to hold on a tampered wire.
    /// - The verifier therefore does not reject locally.
    /// - Round-by-round soundness still needs the cheat caught.
    /// - This driver asserts the divergence propagates through
    ///   Fiat–Shamir.
    ///
    /// # Coverage role
    ///
    /// Lemma 6.5 bounds the per-round rehabilitation probability:
    ///
    /// ```text
    ///     eps_j <= eps_mca + ell_zk * |Lambda|^2 / |F|
    /// ```
    ///
    /// That is a theorem about the abstract protocol.
    /// What an implementation can test is conformance, which this driver does.
    ///
    /// # Not tested here
    ///
    /// - The quantitative empirical-rate match.
    /// - With `gamma_j` sampled from `EF ~ 2^124`, the bound is approx `2^-110`.
    /// - CI cannot host the trial count needed to observe it.
    fn rbr_tampering_changes_verifier_output_case(
        binding: VariableOrder,
        n_vars: usize,
        ell_zk: usize,
        num_eqs: usize,
        seed: u64,
        tamper_round_seed: usize,
        tamper_pos_seed: usize,
    ) -> Result<(), TestCaseError> {
        // Per-mode folding-factor window:
        //
        //     binding | precondition         | folding range
        //     --------+----------------------+--------------------
        //     prefix  | n_vars > k_pack      | 1 ..= n_vars - k_pack
        //     suffix  | folding <= n_vars    | 1 ..= n_vars - 1
        //
        // The prefix reservation keeps at least one full SIMD lane after the
        // first packed round; suffix has no such constraint.
        let folding_factor = match binding {
            VariableOrder::Prefix => {
                let k_pack = p3_util::log2_strict_usize(<F as Field>::Packing::WIDTH);
                prop_assume!(n_vars > k_pack);
                1 + (seed as usize % (n_vars - k_pack))
            }
            VariableOrder::Suffix => 1 + (seed as usize % (n_vars - 1).max(1)),
        };

        let pow_bits = 0;

        // Honest run.
        // Both verifier replays clone its state so they observe the same
        // Fiat–Shamir history up to (but not including) the tamper.
        let run = run_prover(
            binding,
            n_vars,
            folding_factor,
            ell_zk,
            0,
            num_eqs,
            pow_bits,
            seed,
        );

        // Mutation:
        //
        //     round_coefficients[tamper_round]:
        //         [ c_0, c_2, c_3, ... ]
        //         [ c_0, c_2 + 1, c_3, ... ]   ← tampered (example: tamper_pos = 1)
        //
        // The affine reconstruction of `c_1` rewrites the local round target;
        // the verifier does not detect this locally but Fiat–Shamir downstream
        // diverges.
        let tamper_round = tamper_round_seed % run.zk_data.round_coefficients.len();
        let wire_len = run.zk_data.round_coefficients[tamper_round].len();
        let tamper_pos = tamper_pos_seed % wire_len;
        let mut tampered_zk_data = run.zk_data.clone();
        tampered_zk_data.round_coefficients[tamper_round][tamper_pos] += F::ONE;

        // Honest verifier replay against the untampered proof.
        let honest_verifier = run.verifier.clone();
        let mut honest_v_challenger = run.verifier_challenger.clone();
        let honest_result = honest_verifier.into_sumcheck::<MyMmcs, _>(
            &run.zk_data,
            &run.mask_commitment,
            ell_zk,
            folding_factor,
            pow_bits,
            &mut honest_v_challenger,
        );
        prop_assert!(honest_result.is_ok());
        let honest_target = honest_result.unwrap().claimed_residual;

        // Tampered verifier replay from the same starting state.
        let tampered_verifier = run.verifier.clone();
        let mut tampered_v_challenger = run.verifier_challenger.clone();
        let tampered_result = tampered_verifier.into_sumcheck::<MyMmcs, _>(
            &tampered_zk_data,
            &run.mask_commitment,
            ell_zk,
            folding_factor,
            pow_bits,
            &mut tampered_v_challenger,
        );
        prop_assert!(tampered_result.is_ok());
        let tampered_target = tampered_result.unwrap().claimed_residual;

        // The two targets must differ.
        // Accidental coincidence is bounded by Lemma 6.5's negligible soundness error.
        prop_assert_ne!(
            honest_target,
            tampered_target,
            "tampering with wire coordinate ({}, {}) must change target in binding {:?}",
            tamper_round,
            tamper_pos,
            binding,
        );

        Ok(())
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(8))]

        #[test]
        fn prop_rbr_tampering_changes_verifier_output_prefix(
            n_vars in 3usize..=6,
            ell_zk in 3usize..=4,
            num_eqs in 1usize..=2,
            seed in 0u64..512,
            tamper_round_seed in 0usize..16,
            tamper_pos_seed in 0usize..8,
        ) {
            // Prefix path; see the driver docstring for the soundness story.
            rbr_tampering_changes_verifier_output_case(
                VariableOrder::Prefix,
                n_vars,
                ell_zk,
                num_eqs,
                seed,
                tamper_round_seed,
                tamper_pos_seed,
            )?;
        }

        #[test]
        fn prop_rbr_tampering_changes_verifier_output_suffix(
            n_vars in 3usize..=6,
            ell_zk in 3usize..=4,
            num_eqs in 1usize..=2,
            seed in 0u64..512,
            tamper_round_seed in 0usize..16,
            tamper_pos_seed in 0usize..8,
        ) {
            // Suffix path.
            // Running on both binding modes pins that the divergence
            // invariant does not rely on prefix's packed compression.
            rbr_tampering_changes_verifier_output_case(
                VariableOrder::Suffix,
                n_vars,
                ell_zk,
                num_eqs,
                seed,
                tamper_round_seed,
                tamper_pos_seed,
            )?;
        }
    }
}
