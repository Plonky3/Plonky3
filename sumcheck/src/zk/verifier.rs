//! HVZK verifier with affine-chain replay; covers both stacked binding modes.

use alloc::vec::Vec;

use p3_challenger::fs::TranscriptField;
use p3_challenger::{CanObserve, FieldChallenger, GrindingChallenger};
use p3_commit::Mmcs;
use p3_field::{ExtensionField, Field, HornerIter};
use p3_multilinear_util::point::Point;

use super::data::{ZkSumcheckData, ZkVerifierHandoff};
use super::transcript::{ZkSumcheckShape, ZkVerifierTranscript};
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

    /// Reject a proof whose counts disagree with the described shape.
    ///
    /// # Invariant
    ///
    /// Both counts a proof carries are attacker-controlled, and the round loop reads both.
    ///
    /// ```text
    ///     round_coefficients.len()  ->  how many rounds the loop indexes
    ///     pow_witnesses.len()       ->  what the guarded rounds index into
    /// ```
    ///
    /// Comparing both against the configuration is what keeps every later index in bounds.
    /// The per-round wire width needs no check here: the described step rejects a wrong one.
    ///
    /// # Errors
    ///
    /// - The configuration cannot describe a masked batch.
    /// - The proof does not carry one wire per described round.
    /// - The witness count is not the one the difficulty implies.
    fn validate_shape(
        zk_data: &ZkSumcheckData<F, EF>,
        shape: ZkSumcheckShape,
    ) -> Result<(), SumcheckError> {
        shape.validate::<F>()?;

        if zk_data.round_coefficients.len() != shape.num_rounds {
            return Err(SumcheckError::RoundCountMismatch {
                expected: shape.num_rounds,
                actual: zk_data.round_coefficients.len(),
            });
        }
        let expected_pow = if shape.pow_bits > 0 {
            shape.num_rounds
        } else {
            0
        };
        if zk_data.pow_witnesses.len() != expected_pow {
            return Err(SumcheckError::PowWitnessCountMismatch {
                expected: expected_pow,
                actual: zk_data.pow_witnesses.len(),
            });
        }

        Ok(())
    }

    /// Replay the masking prelude and the round chain of one batch.
    ///
    /// The transcript arrives with its prelude already played.
    ///
    /// The two entry points open it differently, so only what follows is shared.
    ///
    /// # Arguments
    ///
    /// - `transcript`: driver positioned just after the prelude, and the source of the shape.
    /// - `zk_data`: the proof record, already counted against the shape.
    /// - `mask_commitment`: the batch's interleaved mask oracle.
    /// - `claimed_sum`: the scalar the batch runs against.
    ///
    /// # Errors
    ///
    /// Any rejection the round replay itself raises.
    fn replay_claim<M, Ch>(
        transcript: &mut ZkVerifierTranscript<'_, Ch, F, EF>,
        zk_data: &ZkSumcheckData<F, EF>,
        mask_commitment: &M::Commitment,
        claimed_sum: EF,
    ) -> Result<ZkVerifierHandoff<EF>, SumcheckError>
    where
        F: TranscriptField,
        M: Mmcs<EF>,
        Ch: FieldChallenger<F> + GrindingChallenger<Witness = F> + CanObserve<M::Commitment>,
    {
        // The shape the driver was seeded with, so the round count cannot drift from the description.
        let shape = transcript.shape();

        let eps = transcript.masks(mask_commitment.clone(), zk_data.mu_tilde);

        let mut target: EF = eps * claimed_sum + zk_data.mu_tilde;
        let mut randomness: Vec<EF> = Vec::with_capacity(shape.num_rounds);

        // Driven by the number the description was built from, not by a proof length.
        //
        // The two agree only because of the count check in `validate_shape`.
        // Both indices below are in bounds by that same check.
        for round in 0..shape.num_rounds {
            let wire = &zk_data.round_coefficients[round];
            let witness = (shape.pow_bits > 0).then(|| zk_data.pow_witnesses[round]);

            // One call binds the wire, re-checks the grind, and draws the challenge.
            //
            // A rejection here releases the driver's completeness check on its way out.
            let gamma_j = transcript.round(wire, witness)?;

            // Returning without error means the wire was the described width.
            //
            //     wire_len = max(ell_zk, 3) - 1 >= 2
            //
            // Both reads below are therefore in bounds.
            let c0 = wire[0];
            let high_sum: EF = wire[1..].iter().copied().sum();
            let c1 = target - c0.double() - high_sum;

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
    /// - The configuration cannot describe a masked batch.
    /// - Wrong number of rounds or PoW witnesses.
    /// - A per-round wire of the wrong width.
    /// - A failing proof-of-work witness check.
    ///
    /// # Panics
    ///
    /// Never on proof input.
    /// Only when the description is left half-played, which is a caller bug.
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
        F: TranscriptField,
        M: Mmcs<EF>,
        Ch: FieldChallenger<F> + GrindingChallenger<Witness = F> + CanObserve<M::Commitment>,
    {
        // Phase 1: shape checks (input validation before Construction 6.3 replay).
        //
        // Every number here comes from this verifier's own configuration.
        let shape = ZkSumcheckShape::new_batching(folding_factor, ell_zk, pow_bits);
        Self::validate_shape(zk_data, shape)?;

        // Phase 2: transcript prelude, seeded from the shape the prover seeded with.
        let mut transcript = ZkVerifierTranscript::<Ch, F, EF>::new(challenger, shape);

        // Draw alpha, then derive mu from the recorded claims.
        let alpha = transcript.batching_challenge();
        let mu = self.inner.sum(alpha);

        // Phase 3: bind the mask oracle and mu_tilde, draw eps, and walk the round chain.
        //
        // A rejection releases the driver here rather than relying on the step that raised it.
        let handoff = Self::replay_claim::<M, _>(&mut transcript, zk_data, mask_commitment, mu)
            .inspect_err(|_| transcript.abort())?;

        // Every described step was replayed, so the sponge goes back to the caller.
        transcript.finish();

        Ok(handoff)
    }

    /// Replays an HVZK sumcheck transcript for an already-batched scalar claim.
    ///
    /// It mirrors the prover-side run of a masked batch over an inherited claim.
    ///
    /// It plays the inherited-claim prelude rather than the batching one.
    ///
    /// The caller already supplies the scalar the masked sumcheck should prove.
    ///
    /// That scalar is bound ahead of the masking prelude.
    ///
    /// This standalone residual-claim API is therefore transcript-bound with no recorded claims.
    ///
    /// # Soundness
    ///
    /// The prover binds its own view of the same scalar.
    ///
    /// ```text
    ///     prover   ->  claimed_sum + aux_claim
    ///     verifier ->  whatever the caller hands over here
    /// ```
    ///
    /// Nothing here can compare the two, so agreeing on them stays a caller obligation.
    ///
    /// The step is described on both sides, so a disagreement is not silent.
    ///
    /// It moves `eps` and every challenge after it, and the residual no longer matches.
    ///
    /// # Errors
    ///
    /// - The configuration cannot describe a masked batch.
    /// - Wrong number of rounds or PoW witnesses.
    /// - A per-round wire of the wrong width.
    /// - A failing proof-of-work witness check.
    ///
    /// # Panics
    ///
    /// Never on proof input.
    /// Only when the description is left half-played, which is a caller bug.
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
        F: TranscriptField,
        M: Mmcs<EF>,
        Ch: FieldChallenger<F> + GrindingChallenger<Witness = F> + CanObserve<M::Commitment>,
    {
        // Every number here comes from the caller's own configuration.
        let shape = ZkSumcheckShape::new_inherited(folding_factor, ell_zk, pow_bits);
        Self::validate_shape(zk_data, shape)?;

        let mut transcript = ZkVerifierTranscript::<Ch, F, EF>::new(challenger, shape);
        transcript.bind_claim(claimed_sum);

        // A rejection releases the driver here rather than relying on the step that raised it.
        let handoff =
            Self::replay_claim::<M, _>(&mut transcript, zk_data, mask_commitment, claimed_sum)
                .inspect_err(|_| transcript.abort())?;

        transcript.finish();

        Ok(handoff)
    }
}

#[cfg(test)]
mod tests {
    use p3_field::{Field, PackedValue, PrimeCharacteristicRing};
    use proptest::prelude::*;

    use super::*;
    use crate::layout::{Layout, PrefixProver, SuffixProver, TableShape};
    use crate::strategy::VariableOrder;
    use crate::zk::test_helpers::{EF, F, MyMmcs, ProverRun, run_prover};

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
            matches!(result, Err(SumcheckError::InvalidPowWitness { .. })),
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

    /// Drives the `ell_zk` disagreement invariant for one binding mode.
    ///
    /// - Honest prover commits with `ell_zk = 4`.
    /// - Verifier replays with `ell_zk = 5`.
    /// - Asserts the verifier rejects the width disagreement.
    ///
    /// # Why the proof carries no mask length
    ///
    /// The mask length reaches the description twice.
    ///
    /// ```text
    ///     wire step   ->  Fixed(max(ell_zk, 3) - 1)
    ///     seed label  ->  ell_zk itself
    /// ```
    ///
    /// The clamp alone is non-injective on `{2, 3}`, which is why the label carries the number.
    ///
    /// A length the two sides disagree on is caught as a described-width rejection.
    ///
    /// No value the proof supplied takes part in that check.
    fn ell_zk_disagreement_rejected_case(binding: VariableOrder) {
        // Fixture state:
        //
        //     n_vars       = 6
        //     folding      = 2
        //     ell_zk       = 4        (prover-side)
        //     num_virtual  = 1
        //     pow_bits     = 0        (PoW disabled to isolate the width check)
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

        // Described width: 4 coefficients.
        // The proof carries 3.
        assert_eq!(
            result.err(),
            Some(SumcheckError::WireSizeMismatch {
                round: 0,
                expected: wrong_ell_zk - 1,
                actual: ell_zk - 1,
            }),
            "verifier should have rejected the mask-length disagreement in binding {binding:?}",
        );
    }

    /// Fixture shared by the two masking-prelude tampering tests.
    ///
    /// Returns an honest run and the residual its verifier derives from it.
    fn honest_run_and_residual() -> (ProverRun, EF) {
        // Fixture state: n_vars = 6, folding = 2, ell_zk = 4, num_virtual = 1, seed = 3.
        //
        // PoW is disabled, so only the masking prelude is under test.
        let run = run_prover(VariableOrder::Prefix, 6, 2, 4, 0, 1, 0, 3);

        // Every replay below starts from this same post-claim sponge state.
        let mut honest_challenger = run.verifier_challenger.clone();
        let residual = run
            .verifier
            .clone()
            .into_sumcheck::<MyMmcs, _>(
                &run.zk_data,
                &run.mask_commitment,
                4,
                2,
                0,
                &mut honest_challenger,
            )
            .expect("the honest run must replay")
            .claimed_residual;

        (run, residual)
    }

    #[test]
    fn a_foreign_mask_commitment_changes_the_verifier_output() {
        // Invariant: the mask oracle is bound before the challenge that combines the masks.
        //
        // Without that, a prover could pick its masks after seeing `eps`.
        //
        // Mutation: hand the verifier the mask oracle of a different run.
        //
        // The affine reconstruction keeps every round identity satisfied, so the replay
        // still returns `Ok`.
        //
        // The residual it hands back is what must move.
        let (run, honest_residual) = honest_run_and_residual();

        // A second run under another seed commits to different masks.
        let foreign = run_prover(VariableOrder::Prefix, 6, 2, 4, 0, 1, 0, 4);
        assert_ne!(run.mask_commitment, foreign.mask_commitment);

        let mut tampered_challenger = run.verifier_challenger.clone();
        let tampered_residual = run
            .verifier
            .clone()
            .into_sumcheck::<MyMmcs, _>(
                &run.zk_data,
                &foreign.mask_commitment,
                4,
                2,
                0,
                &mut tampered_challenger,
            )
            .expect("a well-shaped proof always replays")
            .claimed_residual;

        assert_ne!(honest_residual, tampered_residual);
    }

    #[test]
    fn a_perturbed_mu_tilde_changes_the_verifier_output() {
        // Invariant: the mask endpoint sum is bound before `eps` weighs it against the plain
        // piece, and it also anchors the round-1 target.
        //
        // Mutation: bump `mu_tilde` by one.
        let (run, honest_residual) = honest_run_and_residual();

        let mut tampered_zk_data = run.zk_data.clone();
        tampered_zk_data.mu_tilde += EF::ONE;

        let mut tampered_challenger = run.verifier_challenger.clone();
        let tampered_residual = run
            .verifier
            .clone()
            .into_sumcheck::<MyMmcs, _>(
                &tampered_zk_data,
                &run.mask_commitment,
                4,
                2,
                0,
                &mut tampered_challenger,
            )
            .expect("a well-shaped proof always replays")
            .claimed_residual;

        assert_ne!(honest_residual, tampered_residual);
    }

    #[test]
    fn a_configuration_no_batch_can_run_under_is_rejected() {
        // A verifier reports a configuration failure.
        // It never panics on one.
        //
        // Fixture state: an honest run at ell_zk = 4, replayed under two broken configurations.
        let run = run_prover(VariableOrder::Prefix, 6, 2, 4, 0, 1, 0, 5);

        // A mask shorter than the plain quadratic cannot hide it.
        let mut challenger = run.verifier_challenger.clone();
        assert_eq!(
            run.verifier
                .clone()
                .into_sumcheck::<MyMmcs, _>(
                    &run.zk_data,
                    &run.mask_commitment,
                    2,
                    2,
                    0,
                    &mut challenger,
                )
                .err(),
            Some(SumcheckError::MaskTooShort { ell_zk: 2 }),
        );

        // A batch of no rounds has no mask to commit and no claim to reduce.
        let mut challenger = run.verifier_challenger.clone();
        assert_eq!(
            run.verifier
                .clone()
                .into_sumcheck::<MyMmcs, _>(
                    &run.zk_data,
                    &run.mask_commitment,
                    4,
                    0,
                    0,
                    &mut challenger,
                )
                .err(),
            Some(SumcheckError::NoRounds),
        );
    }

    #[test]
    fn ell_zk_disagreement_rejected_prefix() {
        // Prefix path: verifier rejects when its `ell_zk` disagrees with the prover's.
        ell_zk_disagreement_rejected_case(VariableOrder::Prefix);
    }

    #[test]
    fn ell_zk_disagreement_rejected_suffix() {
        // Suffix path: same invariant, exercised through the suffix dispatch.
        ell_zk_disagreement_rejected_case(VariableOrder::Suffix);
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
