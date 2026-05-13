use alloc::vec::Vec;
use alloc::{format, vec};
use core::fmt::Debug;
use core::ops::Deref;
use core::slice::from_ref;

use errors::VerifierError;
use p3_challenger::{CanObserve, FieldChallenger, GrindingChallenger};
use p3_commit::{BatchOpeningRef, ExtensionMmcs, Mmcs};
use p3_field::{ExtensionField, Field, TwoAdicField};
use p3_matrix::Dimensions;
use p3_multilinear_util::point::Point;
use p3_multilinear_util::poly::Poly;
use tracing::instrument;

use super::committer::reader::ParsedCommitment;
use super::utils::get_challenge_stir_queries;
use crate::alloc::string::ToString;
use crate::constraints::Constraint;
use crate::constraints::statement::{EqStatement, SelectStatement};
use crate::parameters::{RoundConfig, WhirConfig};
use crate::pcs::proof::{QueryOpening, WhirProof};
use crate::sumcheck::strategy::VariableOrder;
use crate::sumcheck::verify_final_sumcheck_rounds;

pub mod errors;

fn accumulate_scaled<EF: Field>(acc: &mut Option<Vec<EF>>, coeff: EF, values: &[EF]) {
    match acc {
        Some(acc) => {
            for (slot, &value) in acc.iter_mut().zip(values) {
                *slot += coeff * value;
            }
        }
        None => {
            *acc = Some(values.iter().map(|&value| coeff * value).collect());
        }
    }
}

fn accumulate_scaled_base<F, EF>(acc: &mut Option<Vec<EF>>, coeff: EF, values: &[F])
where
    F: Field,
    EF: ExtensionField<F>,
{
    match acc {
        Some(acc) => {
            for (slot, &value) in acc.iter_mut().zip(values) {
                *slot += coeff * EF::from(value);
            }
        }
        None => {
            *acc = Some(
                values
                    .iter()
                    .map(|&value| coeff * EF::from(value))
                    .collect(),
            );
        }
    }
}

/// One independently committed initial oracle participating in a batched WHIR
/// proof for the virtual polynomial `sum_i coeff_i * f_i`.
#[derive(Clone, Debug)]
pub enum WhirBatchedInitialVerifierOracle<EF, D> {
    Base { coeff: EF, root: D },
    Extension { coeff: EF, root: D },
    SharedBase { coeffs: Vec<EF>, root: D },
    SharedExtension { coeffs: Vec<EF>, root: D },
}

/// WHIR protocol verifier.
#[derive(Debug)]
pub struct WhirVerifier<'a, EF, F, MT, Challenger>(
    pub(crate) &'a WhirConfig<EF, F, MT, Challenger>,
)
where
    F: Field,
    EF: ExtensionField<F>;

impl<'a, EF, F, MT, Challenger> WhirVerifier<'a, EF, F, MT, Challenger>
where
    F: TwoAdicField,
    EF: ExtensionField<F> + TwoAdicField,
    Challenger: FieldChallenger<F> + GrindingChallenger<Witness = F>,
    MT: Mmcs<F>,
{
    pub const fn new(params: &'a WhirConfig<EF, F, MT, Challenger>) -> Self {
        Self(params)
    }

    /// Verify a WHIR proof against a commitment and statement.
    ///
    /// Returns the folding randomness point on success.
    #[instrument(skip_all)]
    #[allow(clippy::too_many_lines)]
    pub fn verify(
        &self,
        proof: &WhirProof<F, EF, MT>,
        challenger: &mut Challenger,
        parsed_commitment: &ParsedCommitment<EF, MT::Commitment>,
        mut statement: EqStatement<EF>,
    ) -> Result<Point<EF>, VerifierError>
    where
        Challenger: CanObserve<MT::Commitment>,
    {
        let mut constraints = Vec::new();
        let mut round_folding_randomness = Vec::new();
        let mut claimed_eval = EF::ZERO;
        let mut prev_commitment = parsed_commitment.clone();

        // Combine initial OOD constraints with the public statement.
        statement.concatenate(&prev_commitment.ood_statement);

        let constraint = Constraint::new(
            challenger.sample_algebra_element(),
            statement,
            SelectStatement::initialize(self.num_variables),
        );
        constraint.combine_evals(&mut claimed_eval);
        constraints.push(constraint);

        // Verify the initial sumcheck.
        let folding_randomness = proof.initial_sumcheck.verify_rounds(
            challenger,
            &mut claimed_eval,
            self.starting_folding_pow_bits,
        )?;
        round_folding_randomness.push(folding_randomness);

        // Verify each intermediate round.
        for round_index in 0..self.n_rounds() {
            let round_params = &self.round_parameters[round_index];

            // Parse the round commitment from the proof.
            let new_commitment = ParsedCommitment::<_, MT::Commitment>::parse_with_round(
                proof,
                challenger,
                round_params.num_variables,
                round_params.ood_samples,
                Some(round_index),
            );

            // Verify STIR in-domain challenges against the previous commitment.
            let stir_statement = self.verify_stir_challenges(
                proof,
                challenger,
                round_params,
                &prev_commitment,
                round_folding_randomness.last().unwrap(),
                round_index,
            )?;

            let constraint = Constraint::new(
                challenger.sample_algebra_element(),
                new_commitment.ood_statement.clone(),
                stir_statement,
            );
            constraint.combine_evals(&mut claimed_eval);
            constraints.push(constraint);

            let folding_randomness = proof.rounds[round_index].sumcheck.verify_rounds(
                challenger,
                &mut claimed_eval,
                round_params.folding_pow_bits,
            )?;
            round_folding_randomness.push(folding_randomness);

            prev_commitment = new_commitment;
        }

        // Final round: receive the polynomial in the clear.
        let Some(final_evaluations) = proof.final_poly.clone() else {
            panic!("Expected final polynomial");
        };
        challenger.observe_algebra_slice(final_evaluations.as_slice());

        // Verify final STIR challenges.
        let stir_statement = self.verify_stir_challenges(
            proof,
            challenger,
            &self.final_round_config(),
            &prev_commitment,
            round_folding_randomness.last().unwrap(),
            self.n_rounds(),
        )?;

        stir_statement
            .verify(&final_evaluations)
            .then_some(())
            .ok_or_else(|| VerifierError::StirChallengeFailed {
                challenge_id: 0,
                details: "STIR constraint verification failed on final polynomial".to_string(),
            })?;

        let final_sumcheck_randomness = verify_final_sumcheck_rounds(
            proof.final_sumcheck.as_ref(),
            challenger,
            &mut claimed_eval,
            self.final_sumcheck_rounds,
            self.final_folding_pow_bits,
        )?;
        round_folding_randomness.push(final_sumcheck_randomness.clone());

        // Compute the full folding randomness across all rounds.
        let folding_randomness = Point::new(
            round_folding_randomness
                .into_iter()
                .flat_map(IntoIterator::into_iter)
                .collect(),
        );

        // Evaluate the constraint polynomial at the folding point.
        let evaluation_of_weights =
            VariableOrder::Prefix.eval_constraints_poly(&constraints, &folding_randomness);

        // Final consistency check: claimed_eval == weight * f(r).
        let final_value = final_evaluations.eval_ext::<F>(&final_sumcheck_randomness);
        if claimed_eval != evaluation_of_weights * final_value {
            return Err(VerifierError::SumcheckFailed {
                round: self.final_sumcheck_rounds,
                expected: (evaluation_of_weights * final_value).to_string(),
                actual: claimed_eval.to_string(),
            });
        }

        Ok(folding_randomness)
    }

    /// Verify a WHIR proof whose initial oracle is the virtual linear
    /// combination of several independently committed initial oracles.
    #[instrument(skip_all)]
    #[allow(clippy::too_many_lines)]
    pub fn verify_batched_initial(
        &self,
        proof: &WhirProof<F, EF, MT>,
        challenger: &mut Challenger,
        initial_oracles: &[WhirBatchedInitialVerifierOracle<EF, MT::Commitment>],
        statement: EqStatement<EF>,
    ) -> Result<Point<EF>, VerifierError>
    where
        Challenger: CanObserve<MT::Commitment>,
    {
        let initial_select = SelectStatement::initialize(statement.num_variables());
        self.verify_batched_initial_with_select(
            proof,
            challenger,
            initial_oracles,
            statement,
            initial_select,
        )
    }

    /// Verify a WHIR proof whose virtual initial oracle carries both
    /// multilinear-message equality constraints and RS-codeword select
    /// constraints.
    ///
    /// This is the verifier-side form of WHIR Construction 7.4 for relations
    /// such as WARP: ordinary message claims use `EqStatement`, while
    /// codeword-domain claims stay in `SelectStatement` and are checked by the
    /// constrained-RS proximity protocol instead of a verifier-side RS adjoint.
    #[instrument(skip_all)]
    #[allow(clippy::too_many_lines)]
    pub fn verify_batched_initial_with_select(
        &self,
        proof: &WhirProof<F, EF, MT>,
        challenger: &mut Challenger,
        initial_oracles: &[WhirBatchedInitialVerifierOracle<EF, MT::Commitment>],
        mut statement: EqStatement<EF>,
        initial_select: SelectStatement<F, EF>,
    ) -> Result<Point<EF>, VerifierError>
    where
        Challenger: CanObserve<MT::Commitment>,
    {
        assert_eq!(
            statement.num_variables(),
            initial_select.num_variables(),
            "initial WHIR eq/select arity mismatch"
        );

        let mut constraints = Vec::new();
        let mut round_folding_randomness = Vec::new();
        let mut claimed_eval = EF::ZERO;

        let mut initial_ood_statement = EqStatement::initialize(self.num_variables);
        for oracle in initial_oracles {
            match oracle {
                WhirBatchedInitialVerifierOracle::Base { root, .. }
                | WhirBatchedInitialVerifierOracle::Extension { root, .. }
                | WhirBatchedInitialVerifierOracle::SharedBase { root, .. }
                | WhirBatchedInitialVerifierOracle::SharedExtension { root, .. } => {
                    challenger.observe(root.clone());
                }
            }
        }
        for i in 0..self.commitment_ood_samples {
            let point = Point::expand_from_univariate(
                challenger.sample_algebra_element(),
                self.num_variables,
            );
            let eval = proof.initial_ood_answers[i];
            challenger.observe_algebra_element(eval);
            initial_ood_statement.add_evaluated_constraint(point, eval);
        }

        statement.concatenate(&initial_ood_statement);
        let constraint = Constraint::new(
            challenger.sample_algebra_element(),
            statement,
            initial_select,
        );
        constraint.combine_evals(&mut claimed_eval);
        constraints.push(constraint);

        let folding_randomness = proof.initial_sumcheck.verify_rounds(
            challenger,
            &mut claimed_eval,
            self.starting_folding_pow_bits,
        )?;
        round_folding_randomness.push(folding_randomness);

        let mut prev_commitment: Option<ParsedCommitment<EF, MT::Commitment>> = None;
        for round_index in 0..self.n_rounds() {
            let round_params = &self.round_parameters[round_index];
            let new_commitment = ParsedCommitment::<_, MT::Commitment>::parse_with_round(
                proof,
                challenger,
                round_params.num_variables,
                round_params.ood_samples,
                Some(round_index),
            );

            let stir_statement = if round_index == 0 {
                self.verify_batched_stir_challenges(
                    proof,
                    challenger,
                    round_params,
                    initial_oracles,
                    round_folding_randomness.last().unwrap(),
                    round_index,
                )?
            } else {
                self.verify_stir_challenges(
                    proof,
                    challenger,
                    round_params,
                    prev_commitment.as_ref().unwrap(),
                    round_folding_randomness.last().unwrap(),
                    round_index,
                )?
            };

            let constraint = Constraint::new(
                challenger.sample_algebra_element(),
                new_commitment.ood_statement.clone(),
                stir_statement,
            );
            constraint.combine_evals(&mut claimed_eval);
            constraints.push(constraint);

            let folding_randomness = proof.rounds[round_index].sumcheck.verify_rounds(
                challenger,
                &mut claimed_eval,
                round_params.folding_pow_bits,
            )?;
            round_folding_randomness.push(folding_randomness);
            prev_commitment = Some(new_commitment);
        }

        let Some(final_evaluations) = proof.final_poly.clone() else {
            panic!("Expected final polynomial");
        };
        challenger.observe_algebra_slice(final_evaluations.as_slice());

        let stir_statement = if self.n_rounds() == 0 {
            self.verify_batched_stir_challenges(
                proof,
                challenger,
                &self.final_round_config(),
                initial_oracles,
                round_folding_randomness.last().unwrap(),
                self.n_rounds(),
            )?
        } else {
            self.verify_stir_challenges(
                proof,
                challenger,
                &self.final_round_config(),
                prev_commitment.as_ref().unwrap(),
                round_folding_randomness.last().unwrap(),
                self.n_rounds(),
            )?
        };

        stir_statement
            .verify(&final_evaluations)
            .then_some(())
            .ok_or_else(|| VerifierError::StirChallengeFailed {
                challenge_id: 0,
                details: "STIR constraint verification failed on final polynomial".to_string(),
            })?;

        let final_sumcheck_randomness = verify_final_sumcheck_rounds(
            proof.final_sumcheck.as_ref(),
            challenger,
            &mut claimed_eval,
            self.final_sumcheck_rounds,
            self.final_folding_pow_bits,
        )?;
        round_folding_randomness.push(final_sumcheck_randomness.clone());

        let folding_randomness = Point::new(
            round_folding_randomness
                .into_iter()
                .flat_map(IntoIterator::into_iter)
                .collect(),
        );
        let evaluation_of_weights =
            VariableOrder::Prefix.eval_constraints_poly(&constraints, &folding_randomness);
        let final_value = final_evaluations.eval_ext::<F>(&final_sumcheck_randomness);
        if claimed_eval != evaluation_of_weights * final_value {
            return Err(VerifierError::SumcheckFailed {
                round: self.final_sumcheck_rounds,
                expected: (evaluation_of_weights * final_value).to_string(),
                actual: claimed_eval.to_string(),
            });
        }

        Ok(folding_randomness)
    }

    /// Verify STIR in-domain queries and produce associated constraints.
    ///
    /// Checks PoW witness, generates query indices, verifies Merkle proofs,
    /// and evaluates folded polynomials at the queried positions.
    pub fn verify_stir_challenges(
        &self,
        proof: &WhirProof<F, EF, MT>,
        challenger: &mut Challenger,
        params: &RoundConfig<F>,
        commitment: &ParsedCommitment<EF, MT::Commitment>,
        folding_randomness: &Point<EF>,
        round_index: usize,
    ) -> Result<SelectStatement<F, EF>, VerifierError> {
        // Verify PoW witness before generating challenges.
        let pow_witness = if round_index < self.n_rounds() {
            proof
                .get_pow_after_commitment(round_index)
                .ok_or(VerifierError::InvalidRoundIndex { index: round_index })?
        } else {
            proof.final_pow_witness
        };
        if params.pow_bits > 0 && !challenger.check_witness(params.pow_bits, pow_witness) {
            return Err(VerifierError::InvalidPowWitness);
        }

        // Transcript checkpoint after PoW.
        if round_index < self.n_rounds() {
            challenger.sample();
        }

        // Sample STIR query positions.
        let stir_challenges_indexes = get_challenge_stir_queries::<Challenger, F, EF>(
            params.domain_size,
            params.folding_factor,
            params.num_queries,
            challenger,
        )?;

        let dimensions = vec![Dimensions {
            height: params.domain_size >> params.folding_factor,
            width: 1 << params.folding_factor,
        }];
        let answers = self.verify_merkle_proof(
            proof,
            &commitment.root,
            &stir_challenges_indexes,
            &dimensions,
            round_index,
        )?;

        // Evaluate folded polynomial at each queried position.
        let folds: Vec<_> = answers
            .into_iter()
            .map(|answer| Poly::new(answer).eval_ext::<F>(folding_randomness))
            .collect();

        let stir_constraints = stir_challenges_indexes
            .iter()
            .map(|&index| params.folded_domain_gen.exp_u64(index as u64))
            .collect();

        Ok(SelectStatement::new(
            params.num_variables,
            stir_constraints,
            folds,
        ))
    }

    /// Batched-initial variant of [`Self::verify_stir_challenges`].
    pub fn verify_batched_stir_challenges(
        &self,
        proof: &WhirProof<F, EF, MT>,
        challenger: &mut Challenger,
        params: &RoundConfig<F>,
        initial_oracles: &[WhirBatchedInitialVerifierOracle<EF, MT::Commitment>],
        folding_randomness: &Point<EF>,
        round_index: usize,
    ) -> Result<SelectStatement<F, EF>, VerifierError>
    where
        Challenger: CanObserve<MT::Commitment>,
    {
        let pow_witness = if round_index < self.n_rounds() {
            proof
                .get_pow_after_commitment(round_index)
                .ok_or(VerifierError::InvalidRoundIndex { index: round_index })?
        } else {
            proof.final_pow_witness
        };
        if params.pow_bits > 0 && !challenger.check_witness(params.pow_bits, pow_witness) {
            return Err(VerifierError::InvalidPowWitness);
        }

        if round_index < self.n_rounds() {
            challenger.sample();
        }

        let stir_challenges_indexes = get_challenge_stir_queries::<Challenger, F, EF>(
            params.domain_size,
            params.folding_factor,
            params.num_queries,
            challenger,
        )?;

        let dimensions = vec![Dimensions {
            height: params.domain_size >> params.folding_factor,
            width: 1 << params.folding_factor,
        }];
        let answers = self.verify_batched_merkle_proof(
            proof,
            initial_oracles,
            &stir_challenges_indexes,
            &dimensions,
            round_index,
        )?;

        let folds: Vec<_> = answers
            .into_iter()
            .map(|answer| Poly::new(answer).eval_ext::<F>(folding_randomness))
            .collect();

        let stir_constraints = stir_challenges_indexes
            .iter()
            .map(|&index| params.folded_domain_gen.exp_u64(index as u64))
            .collect();

        Ok(SelectStatement::new(
            params.num_variables,
            stir_constraints,
            folds,
        ))
    }

    fn verify_batched_merkle_proof(
        &self,
        proof: &WhirProof<F, EF, MT>,
        initial_oracles: &[WhirBatchedInitialVerifierOracle<EF, MT::Commitment>],
        indices: &[usize],
        dimensions: &[Dimensions],
        round_index: usize,
    ) -> Result<Vec<Vec<EF>>, VerifierError> {
        let extension_mmcs = ExtensionMmcs::new(self.mmcs.clone());
        let queries = if round_index == self.n_rounds() {
            &proof.final_queries
        } else {
            &proof
                .rounds
                .get(round_index)
                .ok_or_else(|| VerifierError::MerkleProofInvalid {
                    position: 0,
                    reason: format!("Round {round_index} not found in proof"),
                })?
                .queries
        };

        let mut results = Vec::with_capacity(indices.len());
        for (&index, query) in indices.iter().zip(queries.iter()) {
            let QueryOpening::Batched { openings } = query else {
                return Err(VerifierError::MerkleProofInvalid {
                    position: index,
                    reason: "expected batched initial Merkle opening".to_string(),
                });
            };
            if openings.len() != initial_oracles.len() {
                return Err(VerifierError::MerkleProofInvalid {
                    position: index,
                    reason: "batched opening arity mismatch".to_string(),
                });
            }

            let mut combined: Option<Vec<EF>> = None;
            for (oracle, opening) in initial_oracles.iter().zip(openings.iter()) {
                match (oracle, opening) {
                    (
                        WhirBatchedInitialVerifierOracle::Base { coeff, root },
                        QueryOpening::Base { values, proof },
                    ) => {
                        self.mmcs
                            .verify_batch(
                                root,
                                dimensions,
                                index,
                                BatchOpeningRef {
                                    opened_values: from_ref(values),
                                    opening_proof: proof,
                                },
                            )
                            .map_err(|_| VerifierError::MerkleProofInvalid {
                                position: index,
                                reason: "base batched Merkle proof failed".to_string(),
                            })?;
                        let values = values
                            .iter()
                            .map(|&value| EF::from(value))
                            .collect::<Vec<_>>();
                        accumulate_scaled(&mut combined, *coeff, &values);
                    }
                    (
                        WhirBatchedInitialVerifierOracle::Extension { coeff, root },
                        QueryOpening::Extension { values, proof },
                    ) => {
                        extension_mmcs
                            .verify_batch(
                                root,
                                dimensions,
                                index,
                                BatchOpeningRef {
                                    opened_values: from_ref(values),
                                    opening_proof: proof,
                                },
                            )
                            .map_err(|_| VerifierError::MerkleProofInvalid {
                                position: index,
                                reason: "extension batched Merkle proof failed".to_string(),
                            })?;
                        accumulate_scaled(&mut combined, *coeff, values);
                    }
                    (
                        WhirBatchedInitialVerifierOracle::SharedBase { coeffs, root },
                        QueryOpening::SharedBase { values, proof },
                    ) => {
                        if values.len() != coeffs.len() {
                            return Err(VerifierError::MerkleProofInvalid {
                                position: index,
                                reason: "shared base opening arity mismatch".to_string(),
                            });
                        }
                        let shared_dimensions = vec![dimensions[0]; values.len()];
                        self.mmcs
                            .verify_batch(
                                root,
                                &shared_dimensions,
                                index,
                                BatchOpeningRef {
                                    opened_values: values,
                                    opening_proof: proof,
                                },
                            )
                            .map_err(|_| VerifierError::MerkleProofInvalid {
                                position: index,
                                reason: "shared base Merkle proof failed".to_string(),
                            })?;
                        for (coeff, values) in coeffs.iter().zip(values.iter()) {
                            accumulate_scaled_base::<F, EF>(&mut combined, *coeff, values);
                        }
                    }
                    (
                        WhirBatchedInitialVerifierOracle::SharedExtension { coeffs, root },
                        QueryOpening::SharedExtension { values, proof },
                    ) => {
                        if values.len() != coeffs.len() {
                            return Err(VerifierError::MerkleProofInvalid {
                                position: index,
                                reason: "shared extension opening arity mismatch".to_string(),
                            });
                        }
                        let shared_dimensions = vec![dimensions[0]; values.len()];
                        extension_mmcs
                            .verify_batch(
                                root,
                                &shared_dimensions,
                                index,
                                BatchOpeningRef {
                                    opened_values: values,
                                    opening_proof: proof,
                                },
                            )
                            .map_err(|_| VerifierError::MerkleProofInvalid {
                                position: index,
                                reason: "shared extension Merkle proof failed".to_string(),
                            })?;
                        for (coeff, values) in coeffs.iter().zip(values.iter()) {
                            accumulate_scaled(&mut combined, *coeff, values);
                        }
                    }
                    _ => {
                        return Err(VerifierError::MerkleProofInvalid {
                            position: index,
                            reason: "batched opening field mismatch".to_string(),
                        });
                    }
                }
            }
            results.push(combined.unwrap_or_default());
        }

        Ok(results)
    }

    /// Verify Merkle multi-opening proofs at the given indices.
    pub fn verify_merkle_proof(
        &self,
        proof: &WhirProof<F, EF, MT>,
        root: &MT::Commitment,
        indices: &[usize],
        dimensions: &[Dimensions],
        round_index: usize,
    ) -> Result<Vec<Vec<EF>>, VerifierError> {
        let extension_mmcs = ExtensionMmcs::new(self.mmcs.clone());

        let queries = if round_index == self.n_rounds() {
            &proof.final_queries
        } else {
            &proof
                .rounds
                .get(round_index)
                .ok_or_else(|| VerifierError::MerkleProofInvalid {
                    position: 0,
                    reason: format!("Round {round_index} not found in proof"),
                })?
                .queries
        };

        let mut results = Vec::with_capacity(indices.len());

        for (&index, query) in indices.iter().zip(queries.iter()) {
            let values_ef = match query {
                QueryOpening::Base { values, proof } => {
                    self.mmcs
                        .verify_batch(
                            root,
                            dimensions,
                            index,
                            BatchOpeningRef {
                                opened_values: from_ref(values),
                                opening_proof: proof,
                            },
                        )
                        .map_err(|_| VerifierError::MerkleProofInvalid {
                            position: index,
                            reason: "Base field Merkle proof verification failed".to_string(),
                        })?;

                    values.iter().map(|&f| f.into()).collect()
                }
                QueryOpening::Extension { values, proof } => {
                    extension_mmcs
                        .verify_batch(
                            root,
                            dimensions,
                            index,
                            BatchOpeningRef {
                                opened_values: from_ref(values),
                                opening_proof: proof,
                            },
                        )
                        .map_err(|_| VerifierError::MerkleProofInvalid {
                            position: index,
                            reason: "Extension field Merkle proof verification failed".to_string(),
                        })?;

                    values.clone()
                }
                QueryOpening::Batched { .. } => {
                    return Err(VerifierError::MerkleProofInvalid {
                        position: index,
                        reason: "unexpected batched Merkle opening for single-root verifier"
                            .to_string(),
                    });
                }
                QueryOpening::SharedBase { .. } => {
                    return Err(VerifierError::MerkleProofInvalid {
                        position: index,
                        reason: "unexpected shared-base Merkle opening for single-root verifier"
                            .to_string(),
                    });
                }
                QueryOpening::SharedExtension { .. } => {
                    return Err(VerifierError::MerkleProofInvalid {
                        position: index,
                        reason:
                            "unexpected shared-extension Merkle opening for single-root verifier"
                                .to_string(),
                    });
                }
            };

            results.push(values_ef);
        }

        Ok(results)
    }
}

impl<EF, F, MT, Challenger> Deref for WhirVerifier<'_, EF, F, MT, Challenger>
where
    F: Field,
    EF: ExtensionField<F>,
{
    type Target = WhirConfig<EF, F, MT, Challenger>;

    fn deref(&self) -> &Self::Target {
        self.0
    }
}
