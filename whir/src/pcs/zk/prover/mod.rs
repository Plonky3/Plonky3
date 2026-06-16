//! HVZK-WHIR prover (Construction 8.2 / Theorem 10.2 of eprint 2026/391).
//!
//! ```text
//!     masked sumcheck batches -> code-switching rounds -> masked base case
//! ```

mod data;
mod masks;

use alloc::vec::Vec;
use core::marker::PhantomData;

pub use data::HidingWhirProverData;
use data::ZkRoundData;
use masks::{ProverMasks, fold_limb_chunks};
use p3_challenger::{CanObserve, CanSampleUniformBits, FieldChallenger, GrindingChallenger};
use p3_commit::{ExtensionMmcs, Mmcs};
use p3_dft::TwoAdicSubgroupDft;
use p3_field::{ExtensionField, PackedValue, PrimeCharacteristicRing, TwoAdicField, dot_product};
use p3_matrix::Matrix;
use p3_matrix::dense::RowMajorMatrixView;
use p3_maybe_rayon::prelude::*;
use p3_multilinear_util::eq_batch::eval_eq_batch;
use p3_multilinear_util::point::Point;
use p3_multilinear_util::poly::Poly;
use p3_sumcheck::constraints::statement::SelectStatement;
use p3_sumcheck::product_polynomial::ProductPolynomial;
use p3_sumcheck::strategy::{SumcheckProver, VariableOrder};
use p3_sumcheck::zk::ZkSumcheckData;
use p3_util::log2_strict_usize;
use p3_zk_codes::ZkEncodingWithRandomness;
use rand::distr::{Distribution, StandardUniform};
use rand::{Rng, RngExt};
use tracing::instrument;

use crate::pcs::proof::QueryOpening;
use crate::pcs::utils::get_challenge_stir_queries;
use crate::pcs::zk::base_case::{BaseCaseZkConfig, BaseCaseZkProver, MaskGroupWitness};
use crate::pcs::zk::code_switch::{ZkMaskClaim, switch_mask_covector};
use crate::pcs::zk::committer::{FoldedRsCode, zk_padded_matrix};
use crate::pcs::zk::config::ZkWhirConfig;
use crate::pcs::zk::proof::{ZkRoundProof, ZkWhirProof};
use crate::utils::padded_ood_t1;

/// HVZK-WHIR prover.
#[derive(Debug)]
pub struct HidingWhirProver<'a, EF, F, Dft, MT, Challenger>
where
    F: TwoAdicField,
    EF: ExtensionField<F>,
    MT: Mmcs<F>,
{
    /// Derived HVZK configuration.
    pub config: &'a ZkWhirConfig<EF, F, Challenger>,
    /// FFT engine for every codeword encoding.
    pub dft: &'a Dft,
    /// Base-field Merkle commitment scheme.
    pub mmcs: &'a MT,
    /// Extension-field commitment scheme for folded oracles and masks.
    pub extension_mmcs: ExtensionMmcs<F, EF, MT>,
}

impl<'a, EF, F, Dft, MT, Challenger> HidingWhirProver<'a, EF, F, Dft, MT, Challenger>
where
    F: TwoAdicField,
    EF: ExtensionField<F> + TwoAdicField,
    Dft: TwoAdicSubgroupDft<F>,
    MT: Mmcs<F>,
    Challenger: FieldChallenger<F>
        + GrindingChallenger<Witness = F>
        + CanSampleUniformBits<F>
        + CanObserve<MT::Commitment>,
    StandardUniform: Distribution<EF> + Distribution<F>,
{
    /// Bundles the prover dependencies.
    pub fn new(config: &'a ZkWhirConfig<EF, F, Challenger>, dft: &'a Dft, mmcs: &'a MT) -> Self {
        Self {
            config,
            dft,
            mmcs,
            extension_mmcs: ExtensionMmcs::new(mmcs.clone()),
        }
    }

    /// Commits the witness as an interleaved ZK Reed-Solomon codeword.
    pub fn commit<R: Rng>(
        &self,
        message: Poly<F>,
        challenger: &mut Challenger,
        rng: &mut R,
    ) -> (MT::Commitment, HidingWhirProverData<F, EF, MT>) {
        assert_eq!(message.num_variables(), self.config.num_variables);
        let folding = self.config.round_folding_factor(0);
        let randomness: Vec<F> = (0..(self.config.oracle_randomness[0] << folding))
            .map(|_| rng.random())
            .collect();
        // Interleaved ZK encoding: pad with limb-major randomness, then DFT.
        let height =
            (1 << (message.num_variables() - folding)) << self.config.starting_log_inv_rate;
        let padded = zk_padded_matrix(message.as_slice(), &randomness, folding, height);
        let encoded = self.dft.dft_batch(padded).to_row_major_matrix();
        let (commitment, merkle) = self.mmcs.commit_matrix(encoded);
        challenger.observe(commitment.clone());
        (
            commitment,
            HidingWhirProverData {
                message,
                randomness,
                merkle,
                _marker: PhantomData,
            },
        )
    }

    /// Runs the full HVZK opening protocol for evaluation claims
    /// `f(point_i) = eval_i`.
    ///
    /// The claims must already be bound to the transcript by the caller.
    #[instrument(skip_all)]
    #[allow(clippy::too_many_lines)]
    pub fn prove<R: Rng>(
        &self,
        prover_data: HidingWhirProverData<F, EF, MT>,
        claims: &[(Point<EF>, EF)],
        challenger: &mut Challenger,
        rng: &mut R,
    ) -> ZkWhirProof<F, EF, MT> {
        let config = self.config;
        let num_variables = config.num_variables;
        let sumcheck_mask_encoding = config.sumcheck_mask.encoding::<EF>();

        // Claimed evaluations
        let claimed_evals: Vec<EF> = claims.iter().map(|(_, eval)| *eval).collect();

        // Initial relation: claims batched by powers of alpha.
        //
        //     W = sum_i alpha^i eq(z_i, .)        claim = sum_i alpha^i v_i
        let alpha: EF = challenger.sample_algebra_element();
        let mut weights = EF::zero_vec(1 << num_variables);
        let mut claim = EF::ZERO;
        if !claims.is_empty() {
            let coeffs: Vec<EF> = alpha.powers().collect_n(claims.len());
            // All claim tables land in one batched parallel sweep.
            // Row `var` of the point matrix holds variable `var` across claims.
            let mut points_flat = Vec::with_capacity(num_variables * claims.len());
            for var in 0..num_variables {
                for (point, _) in claims {
                    assert_eq!(point.num_variables(), num_variables);
                    points_flat.push(point.as_slice()[var]);
                }
            }
            eval_eq_batch::<F, EF, false>(
                RowMajorMatrixView::new(&points_flat, claims.len()),
                &mut weights,
                &coeffs,
            );
            for ((_, eval), coeff) in claims.iter().zip(&coeffs) {
                claim += *coeff * *eval;
            }
        }
        // Lift the base-field message into the extension once, in parallel.
        let evals: Vec<EF> = prover_data
            .message
            .as_slice()
            .par_iter()
            .map(|&v| v.into())
            .collect();
        let product = ProductPolynomial::new_unpacked(
            VariableOrder::Prefix,
            Poly::new(evals),
            Poly::new(weights),
        );
        let sumcheck_prover = SumcheckProver::new(product, claim);

        // Initial masked sumcheck batch.
        let mut masks = ProverMasks::<F, EF, MT>::new();
        let mut zk_data = ZkSumcheckData::default();
        let handoff = sumcheck_prover.into_zk_sumcheck(
            &mut zk_data,
            &sumcheck_mask_encoding,
            &self.extension_mmcs,
            config.round_folding_factor(0),
            config.starting_folding_pow_bits,
            EF::ZERO,
            challenger,
            rng,
        );
        // Proof accumulators
        let mut sumchecks = Vec::new();
        let mut sumcheck_mask_commitments = Vec::new();
        let mut rounds = Vec::with_capacity(config.n_rounds());

        let mut batch = masks.record_batch(
            &mut sumchecks,
            &mut sumcheck_mask_commitments,
            handoff,
            &mut zk_data,
            config.zk.ell_zk,
        );

        // Current oracle state: message randomness folds along with the
        // message (Lemma 3.26), the Merkle data answers the spot checks.
        // Mixed-field fold: base-field chunks against the extension eq table.
        let mut oracle_randomness: Vec<EF> = fold_limb_chunks(
            &prover_data.randomness,
            config.oracle_randomness[0],
            &batch.randomness,
        );
        let mut round_data = ZkRoundData::<F, EF, MT>::Base(prover_data.merkle);

        // Code-switching rounds.
        for round in 0..config.n_rounds() {
            let round_params = &config.round_parameters[round];
            let folding = config.round_folding_factor(round);
            let folding_next = config.round_folding_factor(round + 1);
            let next_randomness_len = config.oracle_randomness[round + 1];

            let message = batch.residual_prover.evals();
            let message_len = message.num_evals();

            // Commit the folded message into the next interleaved ZK oracle.
            let fresh_randomness: Vec<EF> = (0..next_randomness_len << folding_next)
                .map(|_| rng.random())
                .collect();
            // Interleaved ZK encoding over the extension, base-field DFT.
            let height = config.inv_rate(round) * (1 << (message.num_variables() - folding_next));
            let padded =
                zk_padded_matrix(message.as_slice(), &fresh_randomness, folding_next, height);
            let encoded = self.dft.dft_algebra_batch(padded);
            let (commitment, merkle) = self.extension_mmcs.commit_matrix(encoded);
            challenger.observe(commitment.clone());

            // Commit the code-switch mask (folded randomness || pad).
            let mask_shape = &config.switch_masks[round];
            let mask_encoding = mask_shape.encoding::<EF>();
            let pad: Vec<EF> = (0..round_params.ood_samples)
                .map(|_| rng.random())
                .collect();
            let mut mask_message = oracle_randomness.clone();
            mask_message.extend_from_slice(&pad);
            let mask_encoding_randomness: Vec<EF> = (0..mask_shape.randomness_len)
                .map(|_| rng.random())
                .collect();
            let mask_codeword =
                mask_encoding.encode_with_randomness(&mask_message, &mask_encoding_randomness);
            let (mask_commitment, mask_data) = self.extension_mmcs.commit_matrix(mask_codeword);
            challenger.observe(mask_commitment.clone());

            // Private out-of-domain answers over (message || randomness || pad).
            //
            // OOD privacy needs the pad-coefficient matrix {rho_i^{l+r+s}}
            // invertible, i.e. the rho_i pairwise distinct and nonzero.
            // Over the quartic extension both hold but for a 1/|F| event,
            // folded into the HVZK error; the debug_assert flags a future
            // small-field instantiation loudly rather than leaking silently.
            let mut rho_points = Vec::with_capacity(round_params.ood_samples);
            let mut ood_answers = Vec::with_capacity(round_params.ood_samples);
            for _ in 0..round_params.ood_samples {
                let rho: EF = challenger.sample_algebra_element();
                debug_assert!(!rho.is_zero(), "OOD point must be nonzero");
                debug_assert!(
                    !rho_points.contains(&rho),
                    "OOD points must be pairwise distinct",
                );
                let answer = padded_ood_t1(rho, message.as_slice(), &mask_message);
                challenger.observe_algebra_element(answer);
                rho_points.push(rho);
                ood_answers.push(answer);
            }

            // PoW, transcript checkpoint, STIR queries on the previous oracle.
            //
            //     pow_bits = 0  ->  no grind, zero witness on the wire
            let pow_witness = if round_params.pow_bits > 0 {
                challenger.grind(round_params.pow_bits)
            } else {
                F::ZERO
            };
            challenger.sample();
            let stir_indexes = get_challenge_stir_queries::<Challenger, F>(
                round_params.domain_size,
                folding,
                round_params.num_queries,
                challenger,
            );

            // Open the previous oracle and fold each leaf at the batch
            // randomness; the verifier recomputes the same folds.
            let mut queries = Vec::with_capacity(stir_indexes.len());
            let mut folded_values = Vec::with_capacity(stir_indexes.len());
            let mut query_vars = Vec::with_capacity(stir_indexes.len());
            for &index in &stir_indexes {
                let (opening, folded) = self.open_and_fold(&round_data, index, &batch.randomness);
                queries.push(opening);
                folded_values.push(folded);
                query_vars.push(round_params.folded_domain_gen.exp_u64(index as u64));
            }
            let query_points: Vec<EF> = query_vars.iter().map(|&x| EF::from(x)).collect();

            // Batch the carried claim with the fresh constraints.
            //
            //     carried claim     ->  coefficient 1
            //     OOD answer i      ->  coefficient gamma^{1+i}
            //     query opening q   ->  coefficient gamma^{1+t_ood+q}
            //
            // Starting at the first power keeps every fresh constraint
            // independent of the carried claim.
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
            // Carried scalar entering this round's batching:
            //
            //     carried = source residual + sum_i <xi_i, u_i>
            //
            // The mask total is read from the running value.
            let carried = batch.residual_prover.claimed_sum() + masks.aux;
            let joint = mask_claim
                .batched_claim(carried, &ood_answers, &folded_values)
                .expect("prover-built dimensions always match");

            // Source side: fold the fresh power constraints into the
            // running sumcheck prover.
            let mut sumcheck_prover = batch.residual_prover;
            // Source side: the fresh constraints land as power covectors.
            //
            //     delta[b]    = sum_j c_j rho_j^b  +  sum_q c'_q x_q^b
            //     claim_delta = <message, delta>
            // The base-field query covectors fill through the packed
            // SelectStatement kernel; the few extension-field OOD points
            // follow with chunked parallel power runs.
            let k = log2_strict_usize(message_len);
            let k_pack = log2_strict_usize(F::Packing::WIDTH);
            let mut weight_delta = if k >= k_pack {
                let mut packed_delta =
                    Poly::new(EF::ExtensionPacking::zero_vec(message_len >> k_pack));
                let mut select = SelectStatement::<F, EF>::initialize(k);
                for &var in &query_vars {
                    // Evaluations are unused: only the covector side is read.
                    select.add_constraint(var, EF::ZERO);
                }
                // Query coefficients are gamma^{1 + t_ood + q}.
                let mut unused_sum = EF::ZERO;
                select.combine_packed(
                    &mut packed_delta,
                    &mut unused_sum,
                    combination,
                    1 + rho_points.len(),
                );
                packed_delta.unpack::<F, EF>().into_evals()
            } else {
                // Too few variables for the packed kernel: power runs only.
                let mut weight_delta = EF::zero_vec(message_len);
                for (&var, &coeff) in query_points.iter().zip(query_coeffs) {
                    let mut term = coeff;
                    for dst in &mut weight_delta {
                        *dst += term;
                        term *= var;
                    }
                }
                weight_delta
            };
            // Chunked parallel power run: chunk `c` starts at coeff * rho^(c * CHUNK).
            const POW_CHUNK: usize = 1 << 12;
            for (&rho, &coeff) in rho_points.iter().zip(ood_coeffs) {
                weight_delta.par_chunks_mut(POW_CHUNK).enumerate().for_each(
                    |(chunk_idx, chunk)| {
                        let mut term = coeff * rho.exp_u64((chunk_idx * POW_CHUNK) as u64);
                        for dst in chunk {
                            *dst += term;
                            term *= rho;
                        }
                    },
                );
            }
            let claim_delta = message
                .as_slice()
                .par_chunks(POW_CHUNK)
                .zip(weight_delta.par_chunks(POW_CHUNK))
                .map(|(m, w)| dot_product::<EF, _, _>(m.iter().copied(), w.iter().copied()))
                .sum::<EF>();
            sumcheck_prover.accumulate_claim(&weight_delta, claim_delta);

            // Mask side: the fresh mask enters the relation.
            let mask_covector = switch_mask_covector(
                message_len,
                oracle_randomness.len(),
                pad.len(),
                &rho_points,
                ood_coeffs,
                &query_points,
                query_coeffs,
            );
            // The running total must match a full re-evaluation.
            debug_assert_eq!(masks.aux, masks.claims.evaluate(&masks.messages));
            // Cross-check the batched-claim identity:
            //
            //     residual + aux + <mask covector, mask message> = mu'
            debug_assert_eq!(
                sumcheck_prover.claimed_sum()
                    + masks.aux
                    + dot_product::<EF, _, _>(
                        mask_covector.iter().copied(),
                        mask_message.iter().copied(),
                    ),
                joint,
            );
            masks.push_switch_mask(
                mask_covector,
                mask_message,
                mask_encoding_randomness,
                mask_data,
            );

            rounds.push(ZkRoundProof {
                commitment,
                mask_commitment,
                ood_answers,
                pow_witness,
                queries,
            });

            // Next masked sumcheck batch over the new oracle.
            //
            // The mask-claim total rides the batch as its auxiliary constant.
            let aux = masks.aux;
            let mut zk_data = ZkSumcheckData::default();
            let handoff = sumcheck_prover.into_zk_sumcheck(
                &mut zk_data,
                &sumcheck_mask_encoding,
                &self.extension_mmcs,
                folding_next,
                round_params.folding_pow_bits,
                aux,
                challenger,
                rng,
            );
            batch = masks.record_batch(
                &mut sumchecks,
                &mut sumcheck_mask_commitments,
                handoff,
                &mut zk_data,
                config.zk.ell_zk,
            );

            oracle_randomness =
                fold_limb_chunks(&fresh_randomness, next_randomness_len, &batch.randomness);
            round_data = ZkRoundData::Ext(merkle);
        }

        // Masked base case on the virtual folded oracle.
        let final_config = config.final_round_config();
        let source_code = FoldedRsCode::<F>::new(
            1 << final_config.num_variables,
            config.oracle_randomness[config.n_rounds()],
            final_config.domain_size >> final_config.folding_factor,
        );
        let base_config = BaseCaseZkConfig {
            code: source_code,
            mask_groups: config.mask_groups(),
            num_queries: config.final_queries,
            mask_queries: config.mask_queries,
            pow_bits: config.final_pow_bits,
        };
        let base_prover = BaseCaseZkProver {
            config: &base_config,
            extension_mmcs: &self.extension_mmcs,
        };

        let source_message = batch.residual_prover.evals();
        let source_covector = batch.residual_prover.weights();
        // Slice the flat mask state into the committed groups.
        let mut group_offset = 0;
        let mask_witnesses: Vec<MaskGroupWitness<'_, F, EF, MT>> = masks
            .groups
            .iter()
            .map(|(width, data)| {
                let range = group_offset..group_offset + width;
                group_offset += width;
                MaskGroupWitness {
                    messages: &masks.messages[range.clone()],
                    randomness: &masks.randomness[range.clone()],
                    covectors: &masks.claims.covectors[range],
                    data,
                }
            })
            .collect();

        let base_case = base_prover.prove(
            self.dft,
            source_message.as_slice(),
            &oracle_randomness,
            source_covector.as_slice(),
            &mask_witnesses,
            |position| {
                self.open_and_fold(&round_data, position, &batch.randomness)
                    .0
            },
            challenger,
            rng,
        );

        ZkWhirProof {
            evals: claimed_evals,
            sumchecks,
            sumcheck_mask_commitments,
            rounds,
            base_case,
        }
    }

    /// Opens one leaf of the active oracle and folds it at the randomness.
    fn open_and_fold(
        &self,
        round_data: &ZkRoundData<F, EF, MT>,
        index: usize,
        randomness: &Point<EF>,
    ) -> (QueryOpening<F, EF, MT::Proof>, EF) {
        match round_data {
            ZkRoundData::Base(data) => {
                let opening = self.mmcs.open_batch(index, data);
                let values = opening.opened_values.into_iter().next().unwrap();
                let folded = Poly::new(values.clone()).eval_base(randomness);
                (
                    QueryOpening::Base {
                        values,
                        proof: opening.opening_proof,
                    },
                    folded,
                )
            }
            ZkRoundData::Ext(data) => {
                let opening = self.extension_mmcs.open_batch(index, data);
                let values = opening.opened_values.into_iter().next().unwrap();
                let folded = Poly::new(values.clone()).eval_ext::<F>(randomness);
                (
                    QueryOpening::Extension {
                        values,
                        proof: opening.opening_proof,
                    },
                    folded,
                )
            }
        }
    }
}
