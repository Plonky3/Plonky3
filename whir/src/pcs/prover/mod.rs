use alloc::vec::Vec;
use core::ops::Deref;

use p3_challenger::{CanObserve, CanSampleUniformBits, FieldChallenger, GrindingChallenger};
use p3_commit::{ExtensionMmcs, Mmcs};
use p3_dft::TwoAdicSubgroupDft;
use p3_field::{ExtensionField, Field, TwoAdicField};
use p3_matrix::Matrix;
use p3_matrix::dense::{DenseMatrix, RowMajorMatrixView};
use p3_matrix::extension::FlatMatrixView;
use p3_multilinear_util::point::Point;
use p3_multilinear_util::poly::Poly;
use round_state::RoundState;
use tracing::{info_span, instrument};

use crate::constraints::Constraint;
use crate::constraints::statement::initial::InitialStatement;
use crate::constraints::statement::{EqStatement, SelectStatement};
use crate::fiat_shamir::errors::FiatShamirError;
use crate::parameters::WhirConfig;
use crate::pcs::proof::{QueryOpening, SumcheckData, WhirProof};
use crate::pcs::utils::get_challenge_stir_queries;

pub mod round_state;

pub type Proof<W, const DIGEST_ELEMS: usize> = Vec<Vec<[W; DIGEST_ELEMS]>>;
pub type Leafs<F> = Vec<Vec<F>>;

/// Per-round prover state with the Merkle authentication shapes
/// baked in for the WHIR commitment scheme.
///
/// - The first round commits to evaluations laid out as a base-field
///   row-major matrix.
/// - Subsequent rounds commit to folded extension-field evaluations,
///   reinterpreted as wider base-field rows so a single Merkle backend
///   handles both shapes.
type WhirRoundState<EF, F, MT> = RoundState<
    EF,
    F,
    <MT as Mmcs<F>>::ProverData<DenseMatrix<F>>,
    <MT as Mmcs<F>>::ProverData<FlatMatrixView<F, EF, DenseMatrix<EF>>>,
>;

/// Orchestrates the full WHIR proving protocol.
#[derive(Debug)]
pub struct WhirProver<'a, EF, F, MT, Challenger>(pub &'a WhirConfig<EF, F, MT, Challenger>)
where
    F: Field,
    EF: ExtensionField<F>;

impl<EF, F, MT, Challenger> Deref for WhirProver<'_, EF, F, MT, Challenger>
where
    F: Field,
    EF: ExtensionField<F>,
{
    type Target = WhirConfig<EF, F, MT, Challenger>;

    fn deref(&self) -> &Self::Target {
        self.0
    }
}

impl<EF, F, MT, Challenger> WhirProver<'_, EF, F, MT, Challenger>
where
    F: TwoAdicField + Ord,
    EF: ExtensionField<F> + TwoAdicField,
    Challenger: FieldChallenger<F> + GrindingChallenger<Witness = F> + CanSampleUniformBits<F>,
    MT: Mmcs<F>,
{
    const fn validate_parameters(&self) -> bool {
        self.0.num_variables
            == self.0.folding_factor.total_number(self.0.n_rounds()) + self.0.final_sumcheck_rounds
    }

    /// Execute the full WHIR proving protocol.
    ///
    /// Performs multi-round sumcheck-based polynomial folding,
    /// producing Merkle authentication paths and constraint evaluations.
    #[instrument(skip_all)]
    pub fn prove<Dft>(
        &self,
        dft: &Dft,
        proof: &mut WhirProof<F, EF, MT>,
        challenger: &mut Challenger,
        statement: &InitialStatement<F, EF>,
        prover_data: MT::ProverData<DenseMatrix<F>>,
    ) -> Result<(), FiatShamirError>
    where
        Dft: TwoAdicSubgroupDft<F>,
        Challenger: CanObserve<MT::Commitment>,
    {
        assert!(self.validate_parameters(), "Invalid prover parameters");

        // Pre-allocate the extension MMCS wrapper once for all rounds.
        let extension_mmcs = ExtensionMmcs::new(self.mmcs.clone());

        // Initialize the round state with the committed polynomial.
        let mut round_state = RoundState::initialize_first_round_state(
            &mut proof.initial_sumcheck,
            challenger,
            statement,
            prover_data,
            self.folding_factor.at_round(0),
            self.starting_folding_pow_bits,
        )?;

        // Run each WHIR folding round.
        for round in 0..=self.n_rounds() {
            self.round(
                dft,
                round,
                proof,
                challenger,
                &mut round_state,
                &extension_mmcs,
            )?;
        }

        Ok(())
    }

    #[instrument(skip_all, fields(round_number = round_index, log_size = self.num_variables - self.folding_factor.total_number(round_index)))]
    #[allow(clippy::too_many_lines)]
    fn round<Dft: TwoAdicSubgroupDft<F>>(
        &self,
        dft: &Dft,
        round_index: usize,
        proof: &mut WhirProof<F, EF, MT>,
        challenger: &mut Challenger,
        round_state: &mut WhirRoundState<EF, F, MT>,
        extension_mmcs: &ExtensionMmcs<F, EF, MT>,
    ) -> Result<(), FiatShamirError>
    where
        Challenger: CanObserve<MT::Commitment>,
    {
        let folded_evaluations = &round_state.sumcheck_prover.evals();
        let num_variables = self.num_variables - self.folding_factor.total_number(round_index);
        assert_eq!(num_variables, folded_evaluations.num_variables());

        // Final round: send polynomial in the clear.
        if round_index == self.n_rounds() {
            return self.final_round(round_index, proof, challenger, round_state, extension_mmcs);
        }

        let round_params = &self.round_parameters[round_index];
        let folding_factor_next = self.folding_factor.at_round(round_index + 1);
        let inv_rate = self.inv_rate(round_index);

        // Transpose and zero-pad for DFT.
        let padded = info_span!("transpose & pad").in_scope(|| {
            let num_vars = folded_evaluations.num_variables();
            let mut mat = RowMajorMatrixView::new(
                folded_evaluations.as_slice(),
                1 << (num_vars - folding_factor_next),
            )
            .transpose();

            mat.pad_to_height(inv_rate * (1 << (num_vars - folding_factor_next)), EF::ZERO);
            mat
        });

        // DFT to produce the codeword.
        let folded_matrix = info_span!("dft", height = padded.height(), width = padded.width())
            .in_scope(|| dft.dft_algebra_batch(padded).to_row_major_matrix());

        let (root, prover_data) =
            info_span!("commit matrix").in_scope(|| extension_mmcs.commit_matrix(folded_matrix));

        // Observe the round commitment.
        challenger.observe(root.clone());
        proof.rounds[round_index].commitment = Some(root);

        // OOD sampling.
        let mut ood_statement = EqStatement::initialize(num_variables);
        let mut ood_answers = Vec::with_capacity(round_params.ood_samples);
        (0..round_params.ood_samples).for_each(|_| {
            let point =
                Point::expand_from_univariate(challenger.sample_algebra_element(), num_variables);
            let eval = round_state.sumcheck_prover.eval(&point);
            challenger.observe_algebra_element(eval);

            ood_answers.push(eval);
            ood_statement.add_evaluated_constraint(point, eval);
        });
        proof.rounds[round_index].ood_answers = ood_answers;

        // PoW grinding: prevents query manipulation by forcing work after committing.
        if round_params.pow_bits > 0 {
            proof.rounds[round_index].pow_witness = challenger.grind(round_params.pow_bits);
        }

        challenger.sample();

        // STIR query sampling.
        let stir_challenges_indexes = get_challenge_stir_queries::<Challenger, F, EF>(
            round_params.domain_size,
            self.folding_factor.at_round(round_index),
            round_params.num_queries,
            challenger,
        )?;

        let mut stir_statement = SelectStatement::initialize(num_variables);
        let mut queries = Vec::with_capacity(stir_challenges_indexes.len());

        // Open Merkle proofs and evaluate folded polynomials at each queried position.
        match &round_state.merkle_prover_data {
            None => {
                for &challenge in &stir_challenges_indexes {
                    let commitment = self
                        .mmcs
                        .open_batch(challenge, &round_state.commitment_merkle_prover_data);
                    let answer = commitment.opened_values[0].clone();

                    let eval = Poly::new(answer.clone()).eval_base(&round_state.folding_randomness);
                    let var = round_params.folded_domain_gen.exp_u64(challenge as u64);
                    stir_statement.add_constraint(var, eval);

                    queries.push(QueryOpening::Base {
                        values: answer,
                        proof: commitment.opening_proof,
                    });
                }
            }
            Some(data) => {
                for &challenge in &stir_challenges_indexes {
                    let commitment = extension_mmcs.open_batch(challenge, data);
                    let answer = commitment.opened_values[0].clone();

                    let eval =
                        Poly::new(answer.clone()).eval_ext::<F>(&round_state.folding_randomness);
                    let var = round_params.folded_domain_gen.exp_u64(challenge as u64);
                    stir_statement.add_constraint(var, eval);

                    queries.push(QueryOpening::Extension {
                        values: answer,
                        proof: commitment.opening_proof,
                    });
                }
            }
        }

        proof.rounds[round_index].queries = queries;

        let constraint = Constraint::new(
            challenger.sample_algebra_element(),
            ood_statement,
            stir_statement,
        );

        // Run sumcheck and fold the polynomial.
        let mut sumcheck_data: SumcheckData<F, EF> = SumcheckData::default();
        let folding_randomness = round_state.sumcheck_prover.compute_sumcheck_polynomials(
            &mut sumcheck_data,
            challenger,
            folding_factor_next,
            round_params.folding_pow_bits,
            Some(constraint),
        );
        proof.set_sumcheck_data_at(sumcheck_data, round_index);

        // Update round state for next iteration.
        round_state.folding_randomness = folding_randomness;
        round_state.merkle_prover_data = Some(prover_data);

        Ok(())
    }

    #[instrument(skip_all)]
    fn final_round(
        &self,
        round_index: usize,
        proof: &mut WhirProof<F, EF, MT>,
        challenger: &mut Challenger,
        round_state: &mut WhirRoundState<EF, F, MT>,
        extension_mmcs: &ExtensionMmcs<F, EF, MT>,
    ) -> Result<(), FiatShamirError> {
        // Send final polynomial coefficients in the clear.
        challenger.observe_algebra_slice(round_state.sumcheck_prover.evals().as_slice());
        proof.final_poly = Some(round_state.sumcheck_prover.evals());

        // PoW grinding for the final round.
        if self.final_pow_bits > 0 {
            proof.final_pow_witness = challenger.grind(self.final_pow_bits);
        }

        // Final STIR queries.
        let final_challenge_indexes = get_challenge_stir_queries::<Challenger, F, EF>(
            self.final_round_config().domain_size,
            self.folding_factor.at_round(round_index),
            self.final_queries,
            challenger,
        )?;

        // Open Merkle proofs at the queried positions.
        match &round_state.merkle_prover_data {
            None => {
                for challenge in final_challenge_indexes {
                    let commitment = self
                        .mmcs
                        .open_batch(challenge, &round_state.commitment_merkle_prover_data);

                    proof.final_queries.push(QueryOpening::Base {
                        values: commitment.opened_values[0].clone(),
                        proof: commitment.opening_proof,
                    });
                }
            }

            Some(data) => {
                for challenge in final_challenge_indexes {
                    let commitment = extension_mmcs.open_batch(challenge, data);
                    proof.final_queries.push(QueryOpening::Extension {
                        values: commitment.opened_values[0].clone(),
                        proof: commitment.opening_proof,
                    });
                }
            }
        }

        // Optional final sumcheck.
        if self.final_sumcheck_rounds > 0 {
            let mut sumcheck_data: SumcheckData<F, EF> = SumcheckData::default();
            round_state.sumcheck_prover.compute_sumcheck_polynomials(
                &mut sumcheck_data,
                challenger,
                self.final_sumcheck_rounds,
                self.final_folding_pow_bits,
                None,
            );
            proof.set_final_sumcheck_data(sumcheck_data);
        }

        Ok(())
    }
}
