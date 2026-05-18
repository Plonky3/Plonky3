use alloc::vec::Vec;
use core::marker::PhantomData;
use core::ops::Deref;

use p3_challenger::{CanObserve, CanSampleUniformBits, FieldChallenger, GrindingChallenger};
use p3_commit::{ExtensionMmcs, Mmcs};
use p3_dft::{Radix2Dit, TwoAdicSubgroupDft};
use p3_field::{ExtensionField, Field, TwoAdicField};
use p3_matrix::dense::DenseMatrix;
use p3_matrix::extension::FlatMatrixView;
use p3_multilinear_util::point::Point;
use p3_multilinear_util::poly::Poly;
use p3_zk_codes::ReedSolomonZkEncoding;
use rand::distr::{Distribution, StandardUniform};
use rand::{Rng, RngExt};
use tracing::instrument;

use crate::constraints::Constraint;
use crate::constraints::statement::{EqStatement, SelectStatement};
use crate::fiat_shamir::domain_separator::DomainSeparator;
use crate::parameters::WhirConfig;
use crate::pcs::code_switch_zk;
use crate::pcs::committer::writer::{commit_extension, commit_extension_zk};
use crate::pcs::proof::{QueryOpening, SumcheckData, WhirProof};
use crate::pcs::utils::get_challenge_stir_queries;
use crate::sumcheck::layout::Layout;
use crate::sumcheck::strategy::{SumcheckProver, VariableOrder};
use crate::utils::padded_ood_t1;

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
pub struct RoundState<EF, F, BaseData, ExtData>
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
    /// Encoding randomness `r'` from the current round's ZK commitment
    /// (Construction 9.7, step 1). Carried forward so the next round's
    /// mask oracle can hide it. `None` when `zk == false`.
    encoding_randomness: Option<Vec<EF>>,
    /// Mask oracle Merkle prover data for opening at STIR positions (W4).
    /// Commits `DenseMatrix<EF>` via `ExtensionMmcs`, producing the same
    /// `ExtData` type as the target oracle. `None` when `zk == false`.
    mask_prover_data: Option<ExtData>,
    /// Mask message `(r ∥ s̃)` for the padded OOD answer (W3).
    mask_msg: Option<Vec<EF>>,
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
    ///
    /// When `self.params.zk` is `true`, each round additionally commits a
    /// mask oracle and uses private OOD answers (Construction 9.7).
    /// The `rng` is used for ZK randomness generation; it is never sampled
    /// when `zk == false`.
    #[instrument(skip_all)]
    pub fn prove<R: Rng>(
        &self,
        proof: &mut WhirProof<F, EF, MT>,
        challenger: &mut Challenger,
        layout: L,
        prover_data: MT::ProverData<DenseMatrix<F>>,
        rng: &mut R,
    ) where
        Dft: TwoAdicSubgroupDft<F>,
        Challenger: CanObserve<MT::Commitment>,
        StandardUniform: Distribution<EF>,
    {
        assert_eq!(self.folding_factor.at_round(0), layout.folding());
        let variable_order = L::variable_order();

        let (sumcheck_prover, folding_randomness) = layout.into_sumcheck(
            &mut proof.initial_sumcheck,
            self.starting_folding_pow_bits,
            challenger,
        );

        let mut round_state = RoundState {
            sumcheck_prover,
            folding_randomness,
            round_data: RoundData::Base(prover_data),
            encoding_randomness: None,
            mask_prover_data: None,
            mask_msg: None,
        };

        // Run each WHIR folding round.
        for round in 0..=self.n_rounds() {
            self.round(
                round,
                proof,
                challenger,
                &mut round_state,
                variable_order,
                rng,
            );
        }
    }

    #[instrument(skip_all, fields(round_number = round_index, log_size = self.num_variables - self.params.folding_factor.total_number(round_index)))]
    #[allow(clippy::too_many_lines)]
    fn round<R: Rng>(
        &self,
        round_index: usize,
        proof: &mut WhirProof<F, EF, MT>,
        challenger: &mut Challenger,
        round_state: &mut WhirRoundState<EF, F, MT>,
        variable_order: VariableOrder,
        rng: &mut R,
    ) where
        Challenger: CanObserve<MT::Commitment>,
        StandardUniform: Distribution<EF>,
    {
        let folded_evaluations = round_state.sumcheck_prover.evals();
        let num_variables =
            self.num_variables - self.params.folding_factor.total_number(round_index);
        assert_eq!(num_variables, folded_evaluations.num_variables());

        // Final round: send polynomial in the clear.
        if round_index == self.n_rounds() {
            return self.final_round(round_index, proof, challenger, round_state);
        }

        let round_params = &self.round_parameters[round_index];
        let folding_factor_next = self.params.folding_factor.at_round(round_index + 1);
        let inv_rate = self.inv_rate(round_index);
        let zk = self.params.zk;

        // --- Step 1: Commit g (target oracle) ---
        //
        // Construction 9.7: g = Enc_{C'}(f, r') with ZK randomness appended.
        // The last intermediate round uses plain encoding (base-case IOPP
        // handles terminal ZK via #1635).
        let is_last_intermediate = round_index + 1 == self.n_rounds();
        let zk_this_round = zk && !is_last_intermediate;

        // Save the PREVIOUS round's encoding randomness before this
        // round's commit overwrites it. This r' was used to pad the
        // codeword that this round's STIR queries will open.
        let prev_r_prime = round_state.encoding_randomness.clone();

        let (root, prover_data) = if zk_this_round {
            let r_prime: Vec<EF> = (0..round_params.num_queries)
                .map(|_| rng.random())
                .collect();
            let result = commit_extension_zk(
                variable_order,
                &self.dft,
                &self.extension_mmcs,
                &folded_evaluations,
                &r_prime,
                folding_factor_next,
                inv_rate,
            );
            // W5: store r' for next round's mask oracle.
            round_state.encoding_randomness = Some(r_prime);
            result
        } else {
            commit_extension(
                variable_order,
                &self.dft,
                &self.extension_mmcs,
                &folded_evaluations,
                folding_factor_next,
                inv_rate,
            )
        };

        // Observe the round commitment.
        challenger.observe(root.clone());
        proof.rounds[round_index].commitment = Some(root);

        // --- Step 1b: Commit mask oracle s (ZK only) ---
        //
        // The mask hides the previous round's encoding randomness r.
        // Must be observed before OOD sampling so the verifier's challenges
        // depend on the mask commitment (Fiat-Shamir binding).
        if zk_this_round {
            let prev_rand = round_state.encoding_randomness.as_deref().unwrap_or(&[]);

            // Mask encoding parameters (Construction 9.7, Lemma 9.9):
            // - mask_msg_len = ℓ_zk: message length for C_zk. Must hold
            //   (r ∥ s̃) where |r| = prev_rand.len() and |s̃| = ℓ_zk - r.
            // - mask_t = t_zk: ZK query bound for the mask oracle.
            //   Lemma 9.9 requires t_zk ≥ num_queries for RBR soundness.
            // - mask_m = m_zk: codeword length (power of 2, ≥ ℓ_zk + t_zk).
            let mask_msg_len = prev_rand.len().max(1).next_power_of_two();
            let mask_t = round_params.num_queries.max(1);
            let mask_m = (mask_msg_len + mask_t).next_power_of_two();
            let mask_dft = Radix2Dit::default();
            let enc_zk =
                ReedSolomonZkEncoding::<EF, _>::new(mask_t, mask_msg_len, mask_m, mask_dft);

            let (mask_root, mask_msg, mask_prover_data) = code_switch_zk::commit_mask(
                prev_rand,
                &enc_zk,
                &self.extension_mmcs,
                challenger,
                rng,
            );
            proof.rounds[round_index].mask_commitment = Some(mask_root);
            round_state.mask_prover_data = Some(mask_prover_data);
            round_state.mask_msg = Some(mask_msg);
        }

        // --- Steps 2-3: OOD sampling ---
        //
        // Construction 9.7 step 3: y = ze_ood(ρ) · [f; r; s̃].
        // When ZK, the OOD answer is padded_ood_t1(ρ, f_evals, mask_msg).
        // When non-ZK, it's the plain multilinear evaluation f(point).
        let mut ood_statement = EqStatement::initialize(num_variables);
        let mut ood_answers = Vec::with_capacity(round_params.ood_samples);
        let mut ood_corrections = Vec::new();
        (0..round_params.ood_samples).for_each(|_| {
            let ood_univariate: EF = challenger.sample_algebra_element();
            let point = Point::expand_from_univariate(ood_univariate, num_variables);

            let plain_eval = round_state.sumcheck_prover.eval(&point);

            let transcript_eval = if zk_this_round {
                let mask_msg = round_state.mask_msg.as_deref().unwrap();
                let padded = padded_ood_t1(ood_univariate, folded_evaluations.as_slice(), mask_msg);
                ood_corrections.push(padded - plain_eval);
                padded
            } else {
                plain_eval
            };

            challenger.observe_algebra_element(transcript_eval);
            ood_answers.push(transcript_eval);

            ood_statement.add_evaluated_constraint(point, plain_eval);
        });
        proof.rounds[round_index].ood_answers = ood_answers;
        proof.rounds[round_index].zk_ood_corrections = ood_corrections;

        // PoW grinding: prevents query manipulation by forcing work after committing.
        if round_params.pow_bits > 0 {
            proof.rounds[round_index].pow_witness = challenger.grind(round_params.pow_bits);
        }

        challenger.sample();

        // STIR query sampling.
        let stir_challenges_indexes = get_challenge_stir_queries::<Challenger, F, EF>(
            round_params.domain_size,
            self.params.folding_factor.at_round(round_index),
            round_params.num_queries,
            challenger,
        );

        let mut stir_statement = SelectStatement::initialize(num_variables);
        let mut queries = Vec::with_capacity(stir_challenges_indexes.len());
        let query_randomness = match variable_order {
            VariableOrder::Prefix => round_state.folding_randomness.clone(),
            VariableOrder::Suffix => round_state.folding_randomness.reversed(),
        };

        // Open Merkle proofs and evaluate folded polynomials at each queried position.
        match &round_state.round_data {
            RoundData::Base(data) => {
                for &challenge in &stir_challenges_indexes {
                    let commitment = self.mmcs.open_batch(challenge, data);
                    let answer = commitment.opened_values[0].clone();

                    let eval = Poly::new(answer.clone()).eval_base(&query_randomness);
                    let var = round_params.folded_domain_gen.exp_u64(challenge as u64);
                    stir_statement.add_constraint(var, eval);

                    queries.push(QueryOpening::Base {
                        values: answer,
                        proof: commitment.opening_proof,
                    });
                }
            }
            RoundData::Ext(data) => {
                let mut zk_corrections = Vec::new();

                let correction_params = prev_r_prime.as_ref().map(|r_prime| {
                    let prev_folding = self.params.folding_factor.at_round(round_index);
                    let prev_num_vars = num_variables + prev_folding;
                    let prev_msg_len = 1usize << prev_num_vars;
                    let prev_width = 1usize << prev_folding;
                    let prev_inv_rate = self.inv_rate(round_index - 1);
                    let prev_height = prev_inv_rate * (prev_msg_len / prev_width);
                    let dft_root = F::two_adic_generator(p3_util::log2_strict_usize(prev_height));
                    (
                        r_prime.as_slice(),
                        prev_msg_len,
                        prev_width,
                        prev_height,
                        dft_root,
                    )
                });

                for &challenge in &stir_challenges_indexes {
                    let commitment = self.extension_mmcs.open_batch(challenge, data);
                    let answer = commitment.opened_values[0].clone();

                    let mut eval = Poly::new(answer.clone()).eval_ext::<F>(&query_randomness);

                    if let Some((r_prime, msg_len, width, height, dft_root)) = &correction_params {
                        let correction = crate::pcs::utils::zk_stir_correction(
                            r_prime,
                            *msg_len,
                            *width,
                            *height,
                            challenge,
                            *dft_root,
                            query_randomness.as_slice(),
                        );
                        eval -= correction;
                        zk_corrections.push(correction);
                    }

                    let var = round_params.folded_domain_gen.exp_u64(challenge as u64);
                    stir_statement.add_constraint(var, eval);

                    queries.push(QueryOpening::Extension {
                        values: answer,
                        proof: commitment.opening_proof,
                    });
                }

                proof.rounds[round_index].zk_stir_corrections = zk_corrections;
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
        round_state.round_data = RoundData::Ext(prover_data);
    }

    #[instrument(skip_all)]
    fn final_round(
        &self,
        round_index: usize,
        proof: &mut WhirProof<F, EF, MT>,
        challenger: &mut Challenger,
        round_state: &mut WhirRoundState<EF, F, MT>,
    ) {
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
            self.params.folding_factor.at_round(round_index),
            self.final_queries,
            challenger,
        );

        // Open Merkle proofs at the queried positions.
        match &round_state.round_data {
            RoundData::Base(data) => {
                for challenge in final_challenge_indexes {
                    let commitment = self.mmcs.open_batch(challenge, data);

                    proof.final_queries.push(QueryOpening::Base {
                        values: commitment.opened_values[0].clone(),
                        proof: commitment.opening_proof,
                    });
                }
            }

            RoundData::Ext(data) => {
                for challenge in final_challenge_indexes {
                    let commitment = self.extension_mmcs.open_batch(challenge, data);
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
    }
}
