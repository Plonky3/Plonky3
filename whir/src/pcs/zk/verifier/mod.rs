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
use core::slice::from_ref;

use masks::VerifierMasks;
use p3_challenger::{CanObserve, CanSampleUniformBits, FieldChallenger, GrindingChallenger};
use p3_commit::{BatchOpeningRef, ExtensionMmcs, Mmcs};
use p3_field::{ExtensionField, TwoAdicField};
use p3_matrix::Dimensions;
use p3_multilinear_util::point::Point;
use p3_multilinear_util::poly::Poly;
use p3_sumcheck::SumcheckError;
use p3_sumcheck::zk::ZkVerifier;
use thiserror::Error;
use tracing::instrument;

use super::base_case::{BaseCaseZkConfig, BaseCaseZkError, BaseCaseZkVerifier};
use super::code_switch::{CodeSwitchError, ZkMaskClaim, switch_mask_covector};
use super::config::ZkWhirConfig;
use super::constraint::SourceClaim;
use super::proof::ZkWhirProof;
use crate::pcs::proof::QueryOpening;
use crate::pcs::utils::get_challenge_stir_queries;

/// Failure modes of the HVZK-WHIR verifier.
#[derive(Debug, PartialEq, Eq, Error)]
pub enum ZkVerifierError {
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

    /// A Merkle opening failed to verify.
    #[error("merkle verification failed at position {position} in round {round}")]
    MerkleVerificationFailed { round: usize, position: usize },

    /// A round failed its proof-of-work check.
    #[error("invalid proof-of-work witness in round {round}")]
    InvalidPowWitness { round: usize },
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
    #[instrument(skip_all)]
    #[allow(clippy::too_many_lines)]
    pub fn verify(
        &self,
        proof: &ZkWhirProof<F, EF, MT>,
        commitment: &MT::Commitment,
        claims: &[(Point<EF>, EF)],
        challenger: &mut Challenger,
    ) -> Result<(), ZkVerifierError> {
        let config = self.config;
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

        // Initial relation: claims batched by powers of alpha.
        let alpha: EF = challenger.sample_algebra_element();
        let mut source = SourceClaim::new();
        let mut target = EF::ZERO;
        for ((point, eval), coeff) in claims.iter().zip(alpha.powers()) {
            source.push_eq(point.clone(), coeff);
            target += coeff * *eval;
        }
        let mut masks = VerifierMasks::new();

        // Initial masked sumcheck batch.
        let mut randomness = self.replay_sumcheck_batch(
            proof,
            0,
            config.round_folding_factor(0),
            config.starting_folding_pow_bits,
            &mut target,
            &mut source,
            &mut masks,
            challenger,
        )?;

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
            challenger.observe(new_commitment.clone());
            let mask_commitment = &round_proof.mask_commitment;
            challenger.observe(mask_commitment.clone());

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
                let rho: EF = challenger.sample_algebra_element();
                challenger.observe_algebra_element(answer);
                rho_points.push(rho);
            }

            // PoW, transcript checkpoint, STIR queries on the previous oracle.
            if round_params.pow_bits > 0
                && !challenger.check_witness(round_params.pow_bits, round_proof.pow_witness)
            {
                return Err(ZkVerifierError::InvalidPowWitness { round });
            }
            challenger.sample();
            let stir_indexes = get_challenge_stir_queries::<Challenger, F>(
                round_params.domain_size,
                folding,
                round_params.num_queries,
                challenger,
            );
            if round_proof.queries.len() != stir_indexes.len() {
                return Err(ZkVerifierError::QueryCountMismatch {
                    round,
                    expected: stir_indexes.len(),
                    actual: round_proof.queries.len(),
                });
            }

            // Authenticate the leaves and fold them at the batch randomness.
            let dims = vec![Dimensions {
                height: round_params.domain_size >> folding,
                width: 1 << folding,
            }];
            let mut folded_values = Vec::with_capacity(stir_indexes.len());
            let mut query_points = Vec::with_capacity(stir_indexes.len());
            for (&index, opening) in stir_indexes.iter().zip(&round_proof.queries) {
                folded_values.push(self.verify_and_fold_leaf(
                    &active,
                    &dims,
                    index,
                    opening,
                    round,
                    &randomness,
                )?);
                query_points.push(EF::from(
                    round_params.folded_domain_gen.exp_u64(index as u64),
                ));
            }

            // Batch the carried claim with the fresh constraints.
            let combination: EF = challenger.sample_algebra_element();
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
            randomness = self.replay_sumcheck_batch(
                proof,
                round + 1,
                folding_next,
                round_params.folding_pow_bits,
                &mut target,
                &mut source,
                &mut masks,
                challenger,
            )?;

            active = ActiveOracle::Ext(new_commitment);
            num_variables -= folding_next;
        }

        // Masked base case on the virtual folded oracle.
        let final_config = config.final_round_config();
        let source_code = super::committer::FoldedRsCode::<F>::new(
            1 << final_config.num_variables,
            config.oracle_randomness[n_rounds],
            final_config.domain_size >> final_config.folding_factor,
        );
        let base_config = BaseCaseZkConfig {
            code: source_code,
            mask_groups: masks.groups,
            num_queries: config.final_queries,
            mask_queries: config.mask_queries,
            pow_bits: config.final_pow_bits,
        };
        let base_verifier = BaseCaseZkVerifier {
            config: &base_config,
            extension_mmcs: &self.extension_mmcs,
        };

        let source_covector = source.materialize(final_config.num_variables);
        let dims = vec![Dimensions {
            height: final_config.domain_size >> final_config.folding_factor,
            width: 1 << final_config.folding_factor,
        }];
        base_verifier.verify(
            &proof.base_case,
            source_covector.as_slice(),
            &masks.claims.covectors,
            &masks.commitments,
            target,
            |position, opening| {
                self.verify_and_fold_leaf(&active, &dims, position, opening, n_rounds, &randomness)
                    .map_err(|_| BaseCaseZkError::SourceOpeningRejected { position })
            },
            challenger,
        )?;

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
    ) -> Result<Point<EF>, ZkVerifierError> {
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

    /// Authenticates one leaf of the active oracle and folds it at the
    /// batch randomness.
    ///
    /// Base-field leaves fold through the mixed-field evaluator, so no lift
    /// to the extension is materialized.
    #[allow(clippy::too_many_arguments)]
    fn verify_and_fold_leaf(
        &self,
        active: &ActiveOracle<'_, MT::Commitment>,
        dims: &[Dimensions],
        index: usize,
        opening: &QueryOpening<F, EF, MT::Proof>,
        round: usize,
        randomness: &Point<EF>,
    ) -> Result<EF, ZkVerifierError> {
        let width = dims.first().map_or(0, |d| d.width);
        let opened_len = match opening {
            QueryOpening::Base { values, .. } => values.len(),
            QueryOpening::Extension { values, .. } => values.len(),
        };
        if opened_len != width {
            return Err(ZkVerifierError::MerkleVerificationFailed {
                round,
                position: index,
            });
        }
        match (active, opening) {
            (ActiveOracle::Base(commitment), QueryOpening::Base { values, proof }) => {
                self.mmcs
                    .verify_batch(
                        commitment,
                        dims,
                        index,
                        BatchOpeningRef {
                            opened_values: from_ref(values),
                            opening_proof: proof,
                        },
                    )
                    .map_err(|_| ZkVerifierError::MerkleVerificationFailed {
                        round,
                        position: index,
                    })?;
                // Mixed-field fold: base leaf at an extension point.
                Ok(Poly::new(values.clone()).eval_base(randomness))
            }
            (ActiveOracle::Ext(commitment), QueryOpening::Extension { values, proof }) => {
                self.extension_mmcs
                    .verify_batch(
                        commitment,
                        dims,
                        index,
                        BatchOpeningRef {
                            opened_values: from_ref(values),
                            opening_proof: proof,
                        },
                    )
                    .map_err(|_| ZkVerifierError::MerkleVerificationFailed {
                        round,
                        position: index,
                    })?;
                Ok(Poly::new(values.clone()).eval_ext::<F>(randomness))
            }
            _ => Err(ZkVerifierError::MerkleVerificationFailed {
                round,
                position: index,
            }),
        }
    }
}
