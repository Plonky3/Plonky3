//! HVZK-WHIR verifier.
//!
//! ```text
//!     masked sumcheck batches -> code-switching rounds -> masked base case
//! ```
//!
//! The carried claim is tracked symbolically throughout.

mod masks;

use alloc::vec;
use alloc::vec::Vec;

use masks::VerifierMasks;
use p3_challenger::{CanObserve, CanSampleUniformBits, FieldChallenger, GrindingChallenger};
use p3_commit::{ExtensionMmcs, Mmcs};
use p3_field::{ExtensionField, PrimeField64, TwoAdicField};
use p3_matrix::Dimensions;
use p3_multilinear_util::point::Point;
use p3_multilinear_util::poly::Poly;
use p3_sumcheck::SumcheckError;
use p3_sumcheck::zk::ZkVerifier;
use thiserror::Error;
use tracing::instrument;

use super::base_case::{BaseCaseZkError, BaseCaseZkVerifier};
use super::code_switch::{CodeSwitchError, ZkMaskClaim, switch_mask_covector};
use super::config::ZkWhirConfig;
use super::constraint::SourceClaim;
use super::proof::ZkWhirProof;
use crate::pcs::proof::QueryOpenings;
use crate::transcript::zk::{ZkWhirShape, ZkWhirVerifierTranscript};

/// Failure modes of the HVZK-WHIR verifier.
#[derive(Debug, PartialEq, Eq, Error)]
pub enum ZkVerifierError {
    /// The initial alpha batch cannot reach the requested security level.
    #[error(
        "initial claim combination of {num_claims} claims is below the {security_level}-bit target"
    )]
    InitialClaimsBelowTarget {
        num_claims: usize,
        security_level: usize,
    },
    /// A masked sumcheck batch failed to replay.
    #[error(transparent)]
    Sumcheck(#[from] SumcheckError),

    /// The base case rejected.
    #[error(transparent)]
    BaseCase(#[from] BaseCaseZkError),

    /// A batched-claim dimension mismatch.
    #[error(transparent)]
    CodeSwitch(#[from] CodeSwitchError),

    /// An opening point has the wrong arity for the committed polynomial.
    #[error("claim {claim}: point arity mismatch: expected {expected}, got {actual}")]
    ClaimArityMismatch {
        claim: usize,
        expected: usize,
        actual: usize,
    },

    /// The proof carries the wrong number of code-switching rounds.
    #[error("round count mismatch: expected {expected}, got {actual}")]
    RoundCountMismatch { expected: usize, actual: usize },

    /// The proof carries the wrong number of sumcheck batches.
    #[error("sumcheck batch count mismatch: expected {expected}, got {actual}")]
    SumcheckBatchCountMismatch { expected: usize, actual: usize },

    /// A round carries the wrong number of out-of-domain answers.
    #[error("round {round}: OOD answer count mismatch: expected {expected}, got {actual}")]
    OodAnswerCountMismatch {
        round: usize,
        expected: usize,
        actual: usize,
    },

    /// A round carries the wrong number of query openings.
    #[error("round {round}: query count mismatch: expected {expected}, got {actual}")]
    QueryCountMismatch {
        round: usize,
        expected: usize,
        actual: usize,
    },

    /// The proof carries the wrong number of claimed evaluations.
    #[error("claimed evaluation count mismatch: expected {expected}, got {actual}")]
    EvalCountMismatch { expected: usize, actual: usize },

    /// A Merkle multi-opening failed to verify.
    #[error("merkle verification failed in round {round}")]
    MerkleVerificationFailed { round: usize },

    /// A round failed its proof-of-work check.
    #[error("invalid proof-of-work witness in round {round}")]
    InvalidPowWitness { round: usize },

    /// A round's grinding witness is not the value its zero difficulty admits.
    ///
    /// Raised with the other structural checks, before any transcript work.
    //
    // Why: at `pow_bits = 0` neither side touches the sponge.
    //
    //     prover  : grind is skipped     -> zero on the wire
    //     verifier: check_witness(0, w)  -> returns true, absorbs nothing
    //
    // The field is then bound to nothing: any value rides along and still verifies.
    #[error("non-canonical proof-of-work witness in round {round} at zero difficulty")]
    NonCanonicalPowWitness { round: usize },
}

/// The commitment a code-switch round opens against.
enum ActiveOracle<'a, C> {
    /// Base-field initial oracle.
    Base(&'a C),
    /// Extension-field folded oracle.
    Ext(&'a C),
}

/// HVZK-WHIR verifier.
#[derive(Debug)]
pub struct HidingWhirVerifier<'a, EF, F, MT, Challenger>
where
    F: TwoAdicField,
    EF: ExtensionField<F>,
    MT: Mmcs<F>,
{
    /// Derived HVZK configuration.
    pub config: &'a ZkWhirConfig<EF, F, Challenger>,
    /// Base-field Merkle commitment scheme.
    pub mmcs: &'a MT,
    /// Extension-field commitment scheme for folded oracles and masks.
    pub extension_mmcs: ExtensionMmcs<F, EF, MT>,
}

impl<'a, EF, F, MT, Challenger> HidingWhirVerifier<'a, EF, F, MT, Challenger>
where
    F: TwoAdicField,
    EF: ExtensionField<F> + TwoAdicField,
    MT: Mmcs<F>,
    Challenger: FieldChallenger<F>
        + GrindingChallenger<Witness = F>
        + CanSampleUniformBits<F>
        + CanObserve<MT::Commitment>,
{
    /// Bundles the verifier dependencies.
    pub fn new(config: &'a ZkWhirConfig<EF, F, Challenger>, mmcs: &'a MT) -> Self {
        Self {
            config,
            mmcs,
            extension_mmcs: ExtensionMmcs::new(mmcs.clone()),
        }
    }

    /// Verifies an HVZK opening proof against the commitment and the claims
    /// `f(point_i) = eval_i`.
    ///
    /// The claims must already be bound to the transcript by the caller.
    ///
    /// # Transcript
    ///
    /// One driver spans the whole run and seeds itself from the borrowed sponge.
    ///
    /// ```text
    ///     masked sumcheck batch  ->  bracketed, then it seeds a driver of its own
    ///     masked base case       ->  bracketed, then it seeds a driver of its own
    /// ```
    ///
    /// The base-field bound is what lets every one of those seeds be encoded.
    ///
    /// A rejection releases the driver before the error travels to the caller.
    ///
    /// # Errors
    ///
    /// When any structural check, transcript step, or algebraic check rejects the proof.
    #[instrument(skip_all)]
    pub fn verify(
        &self,
        proof: &ZkWhirProof<F, EF, MT>,
        commitment: &MT::Commitment,
        claims: &[(Point<EF>, EF)],
        challenger: &mut Challenger,
    ) -> Result<(), ZkVerifierError>
    where
        F: PrimeField64,
    {
        let config = self.config;
        config.validate_initial_claims(claims.len()).map_err(|_| {
            ZkVerifierError::InitialClaimsBelowTarget {
                num_claims: claims.len(),
                security_level: config.security_level,
            }
        })?;
        let n_rounds = config.n_rounds();

        // Structural checks before any transcript work.
        if proof.rounds.len() != n_rounds {
            return Err(ZkVerifierError::RoundCountMismatch {
                expected: n_rounds,
                actual: proof.rounds.len(),
            });
        }
        // One sumcheck transcript and one interleaved mask commitment per
        // batch; check each count on its own so the error names the culprit.
        if proof.sumchecks.len() != n_rounds + 1 {
            return Err(ZkVerifierError::SumcheckBatchCountMismatch {
                expected: n_rounds + 1,
                actual: proof.sumchecks.len(),
            });
        }
        if proof.sumcheck_mask_commitments.len() != n_rounds + 1 {
            return Err(ZkVerifierError::SumcheckBatchCountMismatch {
                expected: n_rounds + 1,
                actual: proof.sumcheck_mask_commitments.len(),
            });
        }

        // A zero-difficulty site leaves its witness unread, so the value is pinned here
        // rather than by the grind.
        //
        //     pow_bits = 0 -> prover emits zero, verifier reads nothing -> pin it here
        //     pow_bits > 0 -> prover grinds,     verifier resamples     -> the grind pins it
        //
        // Each round carries its own difficulty, so each is compared against its own.
        //
        // The base case pins its own witness the same way.
        for (round, round_proof) in proof.rounds.iter().enumerate() {
            if config.round_parameters[round].pow_bits == 0 && round_proof.pow_witness != F::ZERO {
                return Err(ZkVerifierError::NonCanonicalPowWitness { round });
            }
        }

        // Reject malformed statements before any folding arithmetic runs.
        //
        //     point arity != committed arity  ->  error, never a panic
        for (claim, (point, _)) in claims.iter().enumerate() {
            if point.num_variables() != self.config.num_variables {
                return Err(ZkVerifierError::ClaimArityMismatch {
                    claim,
                    expected: self.config.num_variables,
                    actual: point.num_variables(),
                });
            }
        }

        // One driver spans the whole run.
        //
        // The description is therefore walked exactly once.
        //
        // Every check above reads the proof's own shape.
        //
        // None of them touches the sponge.
        let shape = ZkWhirShape::new(config);
        let mut transcript = ZkWhirVerifierTranscript::<Challenger, F, EF>::new(challenger, shape);

        // A rejection leaves the driver mid-description.
        //
        // Release it before the error travels out.
        match self.replay(&mut transcript, proof, commitment, claims) {
            Ok(()) => {
                transcript.finish();
                Ok(())
            }
            Err(error) => {
                transcript.abort();
                Err(error)
            }
        }
    }

    /// Walk every described step of one hiding run against the proof.
    #[allow(clippy::too_many_lines)]
    fn replay(
        &self,
        transcript: &mut ZkWhirVerifierTranscript<'_, Challenger, F, EF>,
        proof: &ZkWhirProof<F, EF, MT>,
        commitment: &MT::Commitment,
        claims: &[(Point<EF>, EF)],
    ) -> Result<(), ZkVerifierError>
    where
        F: PrimeField64,
    {
        let config = self.config;
        let n_rounds = config.n_rounds();

        // Initial relation: claims batched by powers of alpha.
        let alpha: EF = transcript.initial_batching();
        let mut source = SourceClaim::new();
        let mut target = EF::ZERO;
        for ((point, eval), coeff) in claims.iter().zip(alpha.powers()) {
            source.push_eq(point.clone(), coeff);
            target += coeff * *eval;
        }
        let mut masks = VerifierMasks::new();

        // Initial masked sumcheck batch.
        //
        // The batch is a protocol of its own.
        //
        // The run therefore records it as one bracket.
        let mut randomness = transcript.delegate_initial_fold(|challenger| {
            self.replay_sumcheck_batch(
                proof,
                0,
                config.round_folding_factor(0),
                config.starting_folding_pow_bits,
                &mut target,
                &mut source,
                &mut masks,
                challenger,
            )
        })?;

        let mut active = ActiveOracle::Base(commitment);
        let mut num_variables = config.num_variables - config.round_folding_factor(0);

        // Code-switching rounds.
        for round in 0..n_rounds {
            let round_params = &config.round_parameters[round];
            let round_proof = &proof.rounds[round];
            let folding = config.round_folding_factor(round);
            let folding_next = config.round_folding_factor(round + 1);

            // New oracle and code-switch mask commitments.
            let new_commitment = &round_proof.commitment;
            transcript.oracle_commitment(new_commitment.clone());
            let mask_commitment = &round_proof.mask_commitment;
            transcript.switch_mask_commitment(mask_commitment.clone());

            // Private out-of-domain answers.
            if round_proof.ood_answers.len() != round_params.ood_samples {
                return Err(ZkVerifierError::OodAnswerCountMismatch {
                    round,
                    expected: round_params.ood_samples,
                    actual: round_proof.ood_answers.len(),
                });
            }
            let mut rho_points = Vec::with_capacity(round_params.ood_samples);
            for &answer in &round_proof.ood_answers {
                let rho: EF = transcript.ood_point();
                transcript.ood_answer(answer);
                rho_points.push(rho);
            }

            // PoW, then STIR queries on the previous oracle.
            transcript
                .query_pow(round, round_proof.pow_witness)
                .map_err(|_| ZkVerifierError::InvalidPowWitness { round })?;
            let stir_indexes = transcript.query_indices(round);
            // Authenticate the leaves in one multiproof and fold them at the
            // batch randomness.
            let dims = vec![Dimensions {
                height: round_params.domain_size >> folding,
                width: 1 << folding,
            }];
            let folded_values = self.verify_and_fold_leaves(
                &active,
                &dims,
                &stir_indexes,
                &round_proof.openings,
                round,
                &randomness,
            )?;
            let query_points: Vec<EF> = stir_indexes
                .iter()
                .map(|&index| EF::from(round_params.folded_domain_gen.exp_u64(index as u64)))
                .collect();

            // Batch the carried claim with the fresh constraints.
            let combination: EF = transcript.round_batching();
            let coeffs: Vec<EF> = combination
                .shifted_powers(combination)
                .collect_n(rho_points.len() + query_points.len());
            let (ood_coeffs, query_coeffs) = coeffs.split_at(rho_points.len());

            let mask_claim = ZkMaskClaim {
                base_claim_coeff: EF::ONE,
                ood_coeffs: ood_coeffs.to_vec(),
                in_domain_coeffs: query_coeffs.to_vec(),
            };
            target = mask_claim.batched_claim(target, &round_proof.ood_answers, &folded_values)?;

            // Source side: fresh power constraints over the new message.
            for (&rho, &coeff) in rho_points.iter().zip(ood_coeffs) {
                source.push_pow(rho, num_variables, coeff);
            }
            for (&x, &coeff) in query_points.iter().zip(query_coeffs) {
                source.push_pow(x, num_variables, coeff);
            }

            // Mask side: the fresh code-switch mask enters the relation as
            // its own width-one group.
            masks.push_switch_mask(
                switch_mask_covector(
                    1 << num_variables,
                    config.oracle_randomness[round],
                    round_params.ood_samples,
                    &rho_points,
                    ood_coeffs,
                    &query_points,
                    query_coeffs,
                ),
                config.switch_masks[round],
                mask_commitment.clone(),
            );

            // Next masked sumcheck batch over the new oracle.
            //
            // The batch is a protocol of its own.
            //
            // The run therefore records it as one bracket.
            randomness = transcript.delegate_round_fold(|challenger| {
                self.replay_sumcheck_batch(
                    proof,
                    round + 1,
                    folding_next,
                    round_params.folding_pow_bits,
                    &mut target,
                    &mut source,
                    &mut masks,
                    challenger,
                )
            })?;

            active = ActiveOracle::Ext(new_commitment);
            num_variables -= folding_next;
        }

        // Masked base case on the virtual folded oracle.
        //
        // The closing phase reads its numbers from the configuration.
        //
        // The description read them from that same place.
        let base_config = config.base_case_config();
        // The replay rebuilt the same group list the configuration derives.
        debug_assert_eq!(base_config.mask_groups, masks.groups);
        let base_verifier = BaseCaseZkVerifier {
            config: &base_config,
            extension_mmcs: &self.extension_mmcs,
        };

        let final_config = config.final_round_config();
        let source_covector = source.materialize(final_config.num_variables);
        let dims = vec![Dimensions {
            height: final_config.domain_size >> final_config.folding_factor,
            width: 1 << final_config.folding_factor,
        }];
        // The base case is a protocol of its own.
        //
        // The run therefore records it as one bracket.
        transcript.delegate_base_case(|challenger| {
            base_verifier.verify(
                &proof.base_case,
                source_covector.as_slice(),
                &masks.claims.covectors,
                &masks.commitments,
                target,
                |positions, openings| {
                    self.verify_and_fold_leaves(
                        &active,
                        &dims,
                        positions,
                        openings,
                        n_rounds,
                        &randomness,
                    )
                    .map_err(|_| BaseCaseZkError::SourceOpeningsRejected)
                },
                challenger,
            )
        })?;

        Ok(())
    }

    /// Replays one masked sumcheck batch and updates the carried relation.
    ///
    /// Returns the batch's folding randomness.
    #[allow(clippy::too_many_arguments)]
    fn replay_sumcheck_batch(
        &self,
        proof: &ZkWhirProof<F, EF, MT>,
        batch: usize,
        folding: usize,
        pow_bits: usize,
        target: &mut EF,
        source: &mut SourceClaim<EF>,
        masks: &mut VerifierMasks<F, EF, MT>,
        challenger: &mut Challenger,
    ) -> Result<Point<EF>, ZkVerifierError>
    where
        F: PrimeField64,
    {
        let ell_zk = self.config.zk.ell_zk;
        let commitment = &proof.sumcheck_mask_commitments[batch];
        let handoff = ZkVerifier::<F, EF>::verify_claim::<ExtensionMmcs<F, EF, MT>, _>(
            &proof.sumchecks[batch],
            commitment,
            ell_zk,
            folding,
            pow_bits,
            *target,
            challenger,
        )?;

        // Source constraints fold, then absorb the combining challenge.
        source.fold(&handoff.randomness);
        for constraint in &mut source.constraints {
            constraint.coeff *= handoff.eps;
        }
        // Mask side: carried covectors absorb eps * 2^{-k}, the batch's fresh
        // sumcheck masks enter at scale one.
        masks.record_sumcheck_batch(
            handoff.eps,
            folding,
            ell_zk,
            &handoff.randomness,
            self.config.sumcheck_mask,
            commitment.clone(),
        );

        *target = handoff.claimed_residual;
        Ok(handoff.randomness)
    }

    /// Authenticates every leaf of the active oracle in one multiproof and
    /// folds each at the batch randomness.
    ///
    /// Base-field leaves fold through the mixed-field evaluator, so no lift
    /// to the extension is materialized.
    ///
    /// The variant must match the oracle: a base oracle carries base rows,
    /// an extension oracle carries extension rows. A disagreement is rejected.
    fn verify_and_fold_leaves(
        &self,
        active: &ActiveOracle<'_, MT::Commitment>,
        dims: &[Dimensions],
        indices: &[usize],
        openings: &QueryOpenings<F, EF, MT::MultiProof>,
        round: usize,
        randomness: &Point<EF>,
    ) -> Result<Vec<EF>, ZkVerifierError> {
        let width = dims.first().map_or(0, |d| d.width);
        let reject = || ZkVerifierError::MerkleVerificationFailed { round };

        // One opened row per sampled index, each of the committed leaf width.
        let check_shape = |rows: &[usize]| {
            if rows.len() != indices.len() {
                return Err(ZkVerifierError::QueryCountMismatch {
                    round,
                    expected: indices.len(),
                    actual: rows.len(),
                });
            }
            if rows.iter().any(|&len| len != width) {
                return Err(reject());
            }
            Ok(())
        };

        match (active, openings) {
            (ActiveOracle::Base(commitment), QueryOpenings::Base(opening)) => {
                check_shape(&opening.rows.iter().map(Vec::len).collect::<Vec<_>>())?;
                opening
                    .verify(self.mmcs, commitment, dims, indices)
                    .map_err(|_| reject())?;
                // Mixed-field fold: base leaves at an extension point.
                Ok(opening
                    .rows
                    .iter()
                    .map(|row| Poly::new(row.clone()).eval_base(randomness))
                    .collect())
            }
            (ActiveOracle::Ext(commitment), QueryOpenings::Extension(opening)) => {
                check_shape(&opening.rows.iter().map(Vec::len).collect::<Vec<_>>())?;
                opening
                    .verify(&self.extension_mmcs, commitment, dims, indices)
                    .map_err(|_| reject())?;
                Ok(opening
                    .rows
                    .iter()
                    .map(|row| Poly::new(row.clone()).eval_ext::<F>(randomness))
                    .collect())
            }
            _ => Err(reject()),
        }
    }
}
