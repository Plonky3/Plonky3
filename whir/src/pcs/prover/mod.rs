use alloc::vec;
use alloc::vec::Vec;
use core::marker::PhantomData;
use core::mem;
use core::ops::Deref;

use p3_challenger::{CanObserve, CanSampleUniformBits, FieldChallenger, GrindingChallenger};
use p3_commit::{ExtensionMmcs, Mmcs};
use p3_dft::TwoAdicSubgroupDft;
use p3_field::{ExtensionField, Field, TwoAdicField};
use p3_matrix::dense::DenseMatrix;
use p3_matrix::extension::FlatMatrixView;
use p3_multilinear_util::point::Point;
use p3_multilinear_util::poly::Poly;
use p3_sumcheck::constraints::statement::{EqStatement, SelectStatement};
use p3_sumcheck::constraints::{Constraint, Statements};
use p3_sumcheck::layout::Layout;
use p3_sumcheck::strategy::{SumcheckProver, VariableOrder};
use tracing::instrument;

use crate::fiat_shamir::domain_separator::DomainSeparator;
use crate::parameters::WhirConfig;
use crate::pcs::committer::writer::commit_extension;
use crate::pcs::proof::{
    QueryOpenings, SharedProofOpening, SumcheckData, WhirProof, WhirRoundProof,
};
use crate::pcs::utils::get_challenge_stir_queries;

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

/// Active Merkle prover data for the polynomial currently being queried.
#[derive(Debug)]
enum RoundData<BaseData, ExtData> {
    /// Base-field commitment produced by the initial round.
    Base(BaseData),
    /// Extension-field commitment produced by every subsequent folded round.
    Ext(ExtData),
}

/// Per-round state during WHIR proof generation.
///
/// Tracks the sumcheck prover, folding randomness, and Merkle
/// commitments across base and extension field rounds.
#[derive(Debug)]
struct RoundState<EF, F, BaseData, ExtData>
where
    F: TwoAdicField,
    EF: ExtensionField<F> + TwoAdicField,
{
    /// Sumcheck prover managing constraint batching and polynomial folding.
    sumcheck_prover: SumcheckProver<F, EF>,
    /// Folding challenges (alpha_1, ..., alpha_k) for the current round.
    folding_randomness: Point<EF>,
    /// Active Merkle prover data for the polynomial currently being queried.
    round_data: RoundData<BaseData, ExtData>,
}

/// WHIR prover bundling the protocol config with its FFT and commitment backends.
#[derive(Debug)]
pub struct WhirProver<EF, F, Dft, MT, Challenger, Layout>
where
    F: Field,
    EF: ExtensionField<F>,
{
    /// Derived per-protocol parameters and per-round configuration.
    pub config: WhirConfig<EF, F, Challenger>,
    /// FFT engine used to encode polynomials before each commitment.
    pub dft: Dft,
    /// Base-field Merkle commitment scheme used in the initial round.
    pub mmcs: MT,
    /// Extension-field commitment scheme used in every folded round.
    pub extension_mmcs: ExtensionMmcs<F, EF, MT>,
    /// Marker tying the prover to a specific stacked-layout binding mode.
    _marker: PhantomData<Layout>,
}

impl<EF, F, Dft, MT, Challenger, Layout> Deref for WhirProver<EF, F, Dft, MT, Challenger, Layout>
where
    F: Field,
    EF: ExtensionField<F>,
{
    type Target = WhirConfig<EF, F, Challenger>;

    fn deref(&self) -> &Self::Target {
        &self.config
    }
}

impl<EF, F, Dft, MT, Challenger, L> WhirProver<EF, F, Dft, MT, Challenger, L>
where
    F: TwoAdicField + Ord,
    EF: ExtensionField<F> + TwoAdicField,
    Dft: TwoAdicSubgroupDft<F>,
    Challenger: FieldChallenger<F> + GrindingChallenger<Witness = F> + CanSampleUniformBits<F>,
    MT: Mmcs<F>,
    L: Layout<F, EF>,
{
    /// Builds a prover from a derived config, an FFT engine, and a base-field MMCS.
    ///
    /// The extension-field MMCS is constructed by wrapping the base-field one,
    /// so callers never have to thread it through manually.
    pub fn new(config: WhirConfig<EF, F, Challenger>, dft: Dft, mmcs: MT) -> Self {
        let extension_mmcs = ExtensionMmcs::new(mmcs.clone());
        Self {
            config,
            dft,
            mmcs,
            extension_mmcs,
            _marker: PhantomData,
        }
    }

    /// Build the Fiat-Shamir domain separator for this protocol instance.
    ///
    /// The domain separator encodes all public protocol parameters into
    /// the transcript so the verifier's challenges are bound to this
    /// specific configuration (see Construction 5.1, step 1).
    pub fn add_domain_separator<const DIGEST_ELEMS: usize>(&self, ds: &mut DomainSeparator<EF, F>)
    where
        EF: TwoAdicField,
    {
        // Encode the public parameters (num_variables, security, rate, etc.).
        ds.commit_statement::<Challenger, DIGEST_ELEMS>(&self.config);
        // Encode the full proof structure (round counts, query counts, etc.).
        ds.add_whir_proof::<Challenger, DIGEST_ELEMS>(&self.config);
    }

    /// Execute the full WHIR proving protocol.
    ///
    /// Performs multi-round sumcheck-based polynomial folding,
    /// producing Merkle authentication paths and constraint evaluations.
    #[instrument(skip_all)]
    pub fn prove(
        &self,
        initial_ood_answers: Vec<EF>,
        challenger: &mut Challenger,
        layout: L,
        prover_data: MT::ProverData<DenseMatrix<F>>,
    ) -> WhirProof<F, EF, MT>
    where
        Dft: TwoAdicSubgroupDft<F>,
        Challenger: CanObserve<MT::Commitment>,
    {
        assert_eq!(self.round_folding_factor(0), layout.folding());
        let variable_order = L::variable_order();

        let mut initial_sumcheck = SumcheckData::default();
        let (sumcheck_prover, folding_randomness) = layout.into_sumcheck(
            &mut initial_sumcheck,
            self.starting_folding_pow_bits,
            challenger,
        );

        let mut round_state = RoundState {
            sumcheck_prover,
            folding_randomness,
            round_data: RoundData::Base(prover_data),
        };

        // Build one round proof per intermediate folding round.
        let mut rounds = Vec::with_capacity(self.n_rounds());
        for round_index in 0..self.n_rounds() {
            rounds.push(self.round(round_index, challenger, &mut round_state, variable_order));
        }

        // Final round: send the polynomial in the clear, open the last queries.
        let (final_poly, final_pow_witness, final_openings, final_sumcheck) =
            self.final_round(self.n_rounds(), challenger, &mut round_state);

        WhirProof {
            initial_ood_answers,
            initial_sumcheck,
            rounds,
            final_poly,
            final_pow_witness,
            final_openings,
            final_sumcheck,
        }
    }

    #[instrument(skip_all, fields(round_number = round_index, log_size = self.num_variables - self.total_folded_through(round_index)))]
    #[allow(clippy::too_many_lines)]
    fn round(
        &self,
        round_index: usize,
        challenger: &mut Challenger,
        round_state: &mut WhirRoundState<EF, F, MT>,
        variable_order: VariableOrder,
    ) -> WhirRoundProof<F, EF, MT>
    where
        Challenger: CanObserve<MT::Commitment>,
    {
        let num_variables = self.num_variables - self.total_folded_through(round_index);
        assert_eq!(num_variables, round_state.sumcheck_prover.num_variables());

        let round_params = &self.round_parameters[round_index];
        let folding_factor_next = self.round_folding_factor(round_index + 1);
        let inv_rate = self.inv_rate(round_index);

        // Commit straight from the live sumcheck buffer; no scalar copy is materialized.
        let (root, prover_data) = commit_extension(
            variable_order,
            &self.dft,
            &self.extension_mmcs,
            round_state.sumcheck_prover.evals_view(),
            folding_factor_next,
            inv_rate,
        );

        // Observe the round commitment.
        challenger.observe(root.clone());
        let commitment = Some(root);

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

        // PoW grinding: prevents query manipulation by forcing work after committing.
        let pow_witness = if round_params.pow_bits > 0 {
            challenger.grind(round_params.pow_bits)
        } else {
            F::ZERO
        };

        challenger.sample();

        // STIR query sampling.
        let stir_challenges_indexes = get_challenge_stir_queries::<Challenger, F>(
            round_params.domain_size,
            self.round_folding_factor(round_index),
            round_params.num_queries,
            challenger,
        );

        let mut stir_statement = SelectStatement::initialize(num_variables);
        let query_randomness = match variable_order {
            VariableOrder::Prefix => round_state.folding_randomness.clone(),
            VariableOrder::Suffix => round_state.folding_randomness.reversed(),
        };

        // Open all queried positions in one multiproof.
        // Each row folds in place; the same allocation then travels into the proof.
        let openings = match &round_state.round_data {
            RoundData::Base(data) => {
                let mut opening =
                    SharedProofOpening::open(&self.mmcs, &stir_challenges_indexes, data);
                for (row, &challenge) in opening.rows.iter_mut().zip(&stir_challenges_indexes) {
                    let poly = Poly::new(mem::take(row));
                    let eval = poly.eval_base(&query_randomness);
                    let var = round_params.folded_domain_gen.exp_u64(challenge as u64);
                    stir_statement.add_constraint(var, eval);
                    *row = poly.into_evals();
                }
                QueryOpenings::Base(opening)
            }
            RoundData::Ext(data) => {
                let mut opening =
                    SharedProofOpening::open(&self.extension_mmcs, &stir_challenges_indexes, data);
                for (row, &challenge) in opening.rows.iter_mut().zip(&stir_challenges_indexes) {
                    let poly = Poly::new(mem::take(row));
                    let eval = poly.eval_ext::<F>(&query_randomness);
                    let var = round_params.folded_domain_gen.exp_u64(challenge as u64);
                    stir_statement.add_constraint(var, eval);
                    *row = poly.into_evals();
                }
                QueryOpenings::Extension(opening)
            }
        };

        // Batch the two statement groups into one constraint over the same cube.
        // The out-of-domain claims form the equality group.
        // The query openings form the selection group.
        // A freshly sampled challenge weights the groups by its successive powers,
        // and the verifier samples the same challenge to rebuild the identical batch.
        let num_variables = ood_statement.num_variables();
        let constraint = Constraint::new(
            challenger.sample_algebra_element(),
            num_variables,
            vec![
                Statements::Eq(ood_statement),
                Statements::Select(stir_statement),
            ],
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

        // Update round state for next iteration.
        round_state.folding_randomness = folding_randomness;
        round_state.round_data = RoundData::Ext(prover_data);

        WhirRoundProof {
            commitment,
            ood_answers,
            pow_witness,
            openings,
            sumcheck: sumcheck_data,
        }
    }

    #[instrument(skip_all)]
    #[allow(clippy::type_complexity)]
    fn final_round(
        &self,
        round_index: usize,
        challenger: &mut Challenger,
        round_state: &mut WhirRoundState<EF, F, MT>,
    ) -> (
        Option<Poly<EF>>,
        F,
        QueryOpenings<F, EF, MT::MultiProof>,
        Option<SumcheckData<F, EF>>,
    ) {
        let num_variables = self.num_variables - self.total_folded_through(round_index);
        assert_eq!(
            num_variables,
            round_state.sumcheck_prover.evals().num_variables()
        );

        // Send final polynomial coefficients in the clear.
        // Unpack once; the transcript and the returned proof share the same copy.
        let final_poly = round_state.sumcheck_prover.evals();
        challenger.observe_algebra_slice(final_poly.as_slice());
        let final_poly = Some(final_poly);

        // PoW grinding for the final round.
        let final_pow_witness = if self.final_pow_bits > 0 {
            challenger.grind(self.final_pow_bits)
        } else {
            F::ZERO
        };

        // Final STIR queries.
        let final_challenge_indexes = get_challenge_stir_queries::<Challenger, F>(
            self.final_round_config().domain_size,
            self.round_folding_factor(round_index),
            self.final_queries,
            challenger,
        );

        // Open all queried positions in one multiproof.
        let final_openings = match &round_state.round_data {
            RoundData::Base(data) => QueryOpenings::Base(SharedProofOpening::open(
                &self.mmcs,
                &final_challenge_indexes,
                data,
            )),
            RoundData::Ext(data) => QueryOpenings::Extension(SharedProofOpening::open(
                &self.extension_mmcs,
                &final_challenge_indexes,
                data,
            )),
        };

        // Optional final sumcheck.
        let final_sumcheck = (self.final_sumcheck_rounds > 0).then(|| {
            let mut sumcheck_data: SumcheckData<F, EF> = SumcheckData::default();
            round_state.sumcheck_prover.compute_sumcheck_polynomials(
                &mut sumcheck_data,
                challenger,
                self.final_sumcheck_rounds,
                self.final_folding_pow_bits,
                None,
            );
            sumcheck_data
        });

        (
            final_poly,
            final_pow_witness,
            final_openings,
            final_sumcheck,
        )
    }
}
