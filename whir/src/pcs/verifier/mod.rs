use alloc::vec::Vec;
use alloc::{format, vec};
use core::fmt::Debug;
use core::ops::Deref;
use core::slice::from_ref;

use errors::VerifierError;
use p3_challenger::{CanObserve, CanSampleUniformBits, FieldChallenger, GrindingChallenger};
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
use crate::constraints::statement::SelectStatement;
use crate::parameters::{RoundConfig, WhirConfig};
use crate::pcs::proof::{QueryOpening, WhirProof};
use crate::sumcheck::strategy::VariableOrder;
use crate::sumcheck::{SumcheckError, verify_final_sumcheck_rounds};

pub mod errors;

/// Replays a WHIR opening proof against a public commitment and the
/// constraint built by the layout adapter.
///
/// # Borrowing
///
/// - Config and Merkle scheme are borrowed for the lifetime of the check.
/// - Nothing is cloned across `verify`.
/// - Construction is `const`; spinning up a fresh verifier per proof is free.
///
/// # Variable order
///
/// Tag declared by the prover at commit time. Selects which way folding
/// randomness is consumed in the final identity and STIR unfold:
///
/// ```text
///     Prefix:  fold(rs)         -> final eval, query unfold
///     Suffix:  fold(rs.rev())   -> same checks, reversed binding
/// ```
#[derive(Debug)]
pub struct WhirVerifier<'a, EF, F, MT, Challenger>
where
    F: Field,
    EF: ExtensionField<F>,
    MT: Mmcs<F>,
{
    /// Derived per-protocol parameters and per-round configuration.
    pub(crate) config: &'a WhirConfig<EF, F, Challenger>,
    /// Base-field Merkle commitment scheme used to authenticate STIR queries.
    pub(crate) mmcs: &'a MT,
    /// Binding direction used to interpret folding randomness.
    pub(crate) variable_order: VariableOrder,
}

impl<'a, EF, F, MT, Challenger> WhirVerifier<'a, EF, F, MT, Challenger>
where
    F: TwoAdicField,
    EF: ExtensionField<F> + TwoAdicField,
    MT: Mmcs<F>,
    Challenger: FieldChallenger<F> + GrindingChallenger<Witness = F> + CanSampleUniformBits<F>,
{
    /// Wraps the verifier-side dependencies into a single replay context.
    ///
    /// # Arguments
    ///
    /// - `config`         — derived per-protocol parameters and per-round configuration.
    /// - `mmcs`           — base-field Merkle commitment scheme used to authenticate STIR queries.
    /// - `variable_order` — binding direction the prover declared at commit time.
    pub const fn new(
        config: &'a WhirConfig<EF, F, Challenger>,
        mmcs: &'a MT,
        variable_order: VariableOrder,
    ) -> Self {
        Self {
            config,
            mmcs,
            variable_order,
        }
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
        parsed_commitment: &MT::Commitment,
        initial_constraint: Constraint<F, EF>,
        mut claimed_eval: EF,
    ) -> Result<Point<EF>, VerifierError>
    where
        Challenger: CanObserve<MT::Commitment>,
    {
        // Reject a proof that carries the wrong number of rounds before any
        // transcript work. The per-round commitment slot is checked further
        // down, where each round is parsed.
        let expected_rounds = self.n_rounds();
        if proof.rounds.len() != expected_rounds {
            return Err(VerifierError::RoundCountMismatch {
                expected: expected_rounds,
                actual: proof.rounds.len(),
            });
        }

        let mut constraints = Vec::new();
        let mut round_folding_randomness = Vec::new();
        let mut prev_commitment = parsed_commitment.clone();

        constraints.push(initial_constraint);

        // Initial sumcheck rounds == first-round folding factor.
        let expected_initial_rounds = self.folding_factor(0);
        let actual_initial_rounds = proof.initial_sumcheck.polynomial_evaluations().len();
        if actual_initial_rounds != expected_initial_rounds {
            return Err(VerifierError::Sumcheck(SumcheckError::RoundCountMismatch {
                expected: expected_initial_rounds,
                actual: actual_initial_rounds,
            }));
        }
        let folding_randomness = proof.initial_sumcheck.verify_rounds(
            challenger,
            &mut claimed_eval,
            self.starting_folding_pow_bits,
        )?;
        round_folding_randomness.push(folding_randomness);

        // Verify each intermediate round.
        for round_index in 0..self.n_rounds() {
            let round_params = &self.round_parameters[round_index];

            // Index is in bounds thanks to the length check at function entry,
            // so only a missing commitment slot can fail here.
            let new_commitment = ParsedCommitment::<_, MT::Commitment>::parse_with_round(
                proof,
                challenger,
                round_params.num_variables,
                round_params.ood_samples,
                round_index,
            )?;

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

            prev_commitment = new_commitment.root;
        }

        // Final round: receive the polynomial in the clear.
        let final_evaluations = proof
            .final_poly
            .clone()
            .ok_or(VerifierError::MissingFinalPoly)?;
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
        let evaluation_of_weights = self
            .variable_order
            .eval_constraints_poly(&constraints, &folding_randomness);

        // Final consistency check: claimed_eval == weight * f(r).
        let final_value = match self.variable_order {
            VariableOrder::Prefix => final_evaluations.eval_ext::<F>(&final_sumcheck_randomness),
            VariableOrder::Suffix => {
                final_evaluations.eval_ext::<F>(&final_sumcheck_randomness.reversed())
            }
        };
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
        commitment: &MT::Commitment,
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
        );

        let dimensions = vec![Dimensions {
            height: params.domain_size >> params.folding_factor,
            width: 1 << params.folding_factor,
        }];
        let answers = self.verify_merkle_proof(
            proof,
            commitment,
            &stir_challenges_indexes,
            &dimensions,
            round_index,
        )?;
        let query_randomness = match self.variable_order {
            VariableOrder::Prefix => folding_randomness.clone(),
            VariableOrder::Suffix => folding_randomness.reversed(),
        };

        // Evaluate folded polynomial at each queried position.
        let folds: Vec<_> = answers
            .into_iter()
            .map(|answer| Poly::new(answer).eval_ext::<F>(&query_randomness))
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

    /// Verify Merkle multi-opening proofs at the given indices.
    pub fn verify_merkle_proof(
        &self,
        proof: &WhirProof<F, EF, MT>,
        root: &MT::Commitment,
        indices: &[usize],
        dimensions: &[Dimensions],
        round_index: usize,
    ) -> Result<Vec<Vec<EF>>, VerifierError> {
        let extension_mmcs = ExtensionMmcs::new((*self.mmcs).clone());

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
    MT: Mmcs<F>,
{
    type Target = WhirConfig<EF, F, Challenger>;

    fn deref(&self) -> &Self::Target {
        self.config
    }
}
