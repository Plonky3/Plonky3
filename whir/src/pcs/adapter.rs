//! Adapter implementing the multilinear PCS trait for the WHIR protocol.

use alloc::string::ToString;
use alloc::vec::Vec;
use core::marker::PhantomData;
use core::slice::from_ref;

use p3_challenger::{CanObserve, CanSampleUniformBits, FieldChallenger, GrindingChallenger};
use p3_commit::{BatchOpeningRef, Mmcs, MultilinearPcs};
use p3_dft::TwoAdicSubgroupDft;
use p3_field::{ExtensionField, TwoAdicField};
use p3_matrix::dense::DenseMatrix;
use p3_matrix::extension::FlatMatrixView;
use p3_matrix::{Dimensions, Matrix};
use p3_zk_codes::{LinearZkEncoding, ZkEncoding};
use rand::Rng;
use rand::distr::{Distribution, StandardUniform};

use super::prover::WhirProver;
use super::verifier::WhirVerifier;
use super::verifier::errors::VerifierError;
use crate::pcs::code_switch::{
    ZkMaskClaim, batched_claim, batching_coefficients, output_relation, private_ood_answers,
};
use crate::pcs::committer::writer::commit_extension;
use crate::pcs::proof::{PcsProof, QueryOpening, WhirInitialZkProof, WhirRoundZkProof};
use crate::pcs::utils::get_challenge_stir_queries;
use crate::sumcheck::OpeningProtocol;
use crate::sumcheck::layout::{Layout, PrefixProver, Verifier, Witness};
use crate::sumcheck::zk::{
    ZkPrefixProver, ZkSumcheckData, ZkSumcheckHandoff, ZkVerifier, ZkVerifierHandoff,
};

/// Prover-side handoff between the commit and open phases of the PCS.
///
/// # Lifecycle
///
/// - Built by the commit phase alongside the public commitment.
/// - Stored by the caller while the public transcript advances.
/// - Consumed by the opening phase; never reused afterwards.
pub struct WhirProverData<F, EF, MT, L>
where
    F: TwoAdicField,
    EF: ExtensionField<F>,
    MT: Mmcs<F>,
    L: Layout<F, EF>,
{
    /// Layout-mode prover holding the per-table opening claims accumulator.
    pub layout: L,
    /// Merkle prover data behind the initial commitment; reused to open STIR queries.
    pub merkle_data: MT::ProverData<DenseMatrix<F>>,
    /// Marker tying the data to its extension field; carries no runtime state.
    _marker: PhantomData<EF>,
}

/// Prefix-only ZK opening state after the initial HVZK sumcheck.
///
/// This is the dedicated API boundary for the ZK path. It deliberately does
/// not implement `MultilinearPcs::open`, because the ZK protocol needs an RNG
/// and an explicit mask encoding. The future `round_zk_prefix` flow consumes
/// `initial_handoff` together with `source_merkle_data`.
pub struct WhirZkPrefixOpenState<F, EF, Enc, MT>
where
    F: TwoAdicField,
    EF: ExtensionField<F>,
    Enc: ZkEncoding<F>,
    MT: Mmcs<F>,
{
    /// Partial proof containing public opening evaluations and the initial ZK sumcheck transcript.
    pub proof: PcsProof<F, EF, MT>,
    /// Typed Construction 6.3 handoff consumed by the first code-switch round.
    pub initial_handoff: ZkSumcheckHandoff<F, EF, Enc, MT>,
    /// Merkle prover data for the inherited source oracle queried by the first code-switch round.
    pub source_merkle_data: MT::ProverData<DenseMatrix<F>>,
}

/// Prefix-ZK state after one code-switch round.
pub struct WhirZkPrefixRoundState<F, EF, Enc, MT>
where
    F: TwoAdicField,
    EF: ExtensionField<F> + TwoAdicField,
    Enc: ZkEncoding<F>,
    MT: Mmcs<F>,
{
    /// Partial proof with `rounds[round_index].zk` populated.
    pub proof: PcsProof<F, EF, MT>,
    /// Typed nested ZK sumcheck handoff for the next round.
    pub handoff: ZkSumcheckHandoff<F, EF, Enc, MT>,
    /// Merkle prover data for the newly committed folded oracle.
    pub target_merkle_data: <MT as Mmcs<F>>::ProverData<FlatMatrixView<F, EF, DenseMatrix<EF>>>,
}

impl<EF, F, Dft, MT, Challenger> WhirProver<EF, F, Dft, MT, Challenger, PrefixProver<F, EF>>
where
    F: TwoAdicField + Ord,
    EF: ExtensionField<F> + TwoAdicField,
    Dft: TwoAdicSubgroupDft<F>,
    MT: Mmcs<F>,
    Challenger: FieldChallenger<F>
        + GrindingChallenger<Witness = F>
        + CanSampleUniformBits<F>
        + CanObserve<MT::Commitment>,
{
    /// Start the prefix-only ZK WHIR opening flow.
    ///
    /// This records the same initial public opening claims as the plain
    /// `MultilinearPcs::open` path, then runs the #1605 HVZK sumcheck overlay
    /// and returns the typed handoff needed by Construction 9.7.
    ///
    /// The method intentionally stops before the WHIR round loop. The next
    /// implementation step is `round_zk_prefix`, which consumes the returned
    /// handoff and fills each `WhirRoundProof::zk` payload.
    pub fn begin_zk_prefix_open<Enc, R>(
        &self,
        mut prover_data: WhirProverData<F, EF, MT, PrefixProver<F, EF>>,
        protocol: &OpeningProtocol,
        challenger: &mut Challenger,
        encoding: Enc,
        rng: &mut R,
    ) -> WhirZkPrefixOpenState<F, EF, Enc, MT>
    where
        Enc: ZkEncoding<F>,
        Enc::Codeword: Matrix<F>,
        R: Rng,
        StandardUniform: Distribution<F>,
    {
        let zk_config = self
            .config
            .zk
            .as_ref()
            .expect("ZK prefix opening requires WhirConfig::with_zk_config");
        assert!(
            zk_config.only_prefix,
            "ZK WHIR currently supports only prefix layout"
        );
        assert_eq!(
            encoding.message_len(),
            zk_config.mask_message_len,
            "ZK encoding message length must match WhirZkConfig",
        );
        let first_round_zk = self
            .round_parameters
            .first()
            .and_then(|round| round.zk.as_ref())
            .expect("ZK prefix opening requires at least one ZK code-switch round");
        let required_query_bound = first_round_zk.mask_query_budget;
        assert!(
            encoding.query_bound() >= required_query_bound,
            "ZK encoding query bound must cover the derived mask query budget",
        );
        let expected_domain_size = first_round_zk.mask_domain_size;
        assert_eq!(
            encoding.codeword_len(),
            expected_domain_size,
            "ZK encoding codeword length must match the derived mask domain",
        );

        let mut whir_proof = self.config.empty_proof();
        tracing::info_span!("zk prefix ood claims").in_scope(|| {
            whir_proof.initial_ood_answers = (0..self.commitment_ood_samples)
                .map(|_| prover_data.layout.add_virtual_eval(challenger))
                .collect::<Vec<_>>();
        });

        let evals = protocol
            .iter_openings()
            .map(|(table_idx, polys)| prover_data.layout.eval(table_idx, polys, challenger))
            .collect::<Vec<_>>();

        let mut zk_sumcheck = ZkSumcheckData::<F, EF>::default();
        let zk_prover = ZkPrefixProver::new(prover_data.layout, encoding, self.mmcs.clone());
        let initial_handoff = zk_prover.into_sumcheck(
            &mut zk_sumcheck,
            self.starting_folding_pow_bits,
            challenger,
            rng,
        );
        whir_proof.initial_zk = Some(WhirInitialZkProof {
            zk_sumcheck,
            zk_sumcheck_mask_commitments: initial_handoff
                .mask_oracles
                .iter()
                .map(|(commitment, _)| commitment.clone())
                .collect(),
        });

        WhirZkPrefixOpenState {
            proof: PcsProof {
                whir: whir_proof,
                evals,
            },
            initial_handoff,
            source_merkle_data: prover_data.merkle_data,
        }
    }

    /// Prove one prefix-only ZK code-switching round.
    ///
    /// This is the first real Construction 9.7 wiring point: it consumes the
    /// typed Construction 6.3 handoff, commits the folded target oracle, commits
    /// a fresh mask oracle, records private OOD answers and source/mask openings,
    /// derives the `mu'` relation, and runs the next HVZK residual sumcheck.
    #[allow(clippy::too_many_lines)]
    pub fn round_zk_prefix<SourceEnc, Enc, R>(
        &self,
        state: WhirZkPrefixOpenState<F, EF, Enc, MT>,
        round_index: usize,
        source_encoding: &SourceEnc,
        mask_encoding: &Enc,
        challenger: &mut Challenger,
        rng: &mut R,
    ) -> WhirZkPrefixRoundState<F, EF, Enc, MT>
    where
        SourceEnc: LinearZkEncoding<F>,
        Enc: ZkEncoding<F>,
        Enc::Codeword: Matrix<F>,
        R: Rng,
        StandardUniform: Distribution<F>,
    {
        assert_eq!(
            round_index, 0,
            "first ZK round helper currently handles round 0"
        );
        assert!(
            self.config.zk.as_ref().is_some_and(|zk| zk.only_prefix),
            "ZK WHIR currently supports only prefix layout",
        );

        let mut proof = state.proof;
        let round_params = &self.round_parameters[round_index];
        let round_zk = round_params
            .zk
            .as_ref()
            .expect("round_zk_prefix requires RoundConfig::zk");
        assert_eq!(
            mask_encoding.message_len(),
            round_zk.mask_message_len,
            "mask encoding message length must match the round ZK config",
        );
        assert!(
            mask_encoding.query_bound() >= round_zk.mask_query_budget,
            "mask encoding query bound must cover the round mask query budget",
        );
        assert_eq!(
            mask_encoding.codeword_len(),
            round_zk.mask_domain_size,
            "mask encoding domain must match the round ZK config",
        );

        let handoff = state.initial_handoff;
        let folded_evaluations = handoff.residual_prover.evals();
        let num_variables =
            self.num_variables - self.params.folding_factor.total_number(round_index);
        assert_eq!(num_variables, folded_evaluations.num_variables());
        assert_eq!(
            source_encoding.message_len(),
            folded_evaluations.as_slice().len(),
            "source encoding message length must match the residual polynomial",
        );

        let folding_factor_next = self.params.folding_factor.at_round(round_index + 1);
        let inv_rate = self.inv_rate(round_index);
        let (target_root, target_merkle_data) = commit_extension(
            crate::sumcheck::strategy::VariableOrder::Prefix,
            &self.dft,
            &self.extension_mmcs,
            &folded_evaluations,
            folding_factor_next,
            inv_rate,
        );
        challenger.observe(target_root.clone());
        proof.whir.rounds[round_index].commitment = Some(target_root);

        let mask_message = F::zero_vec(round_zk.mask_message_len);
        let mask_codeword = mask_encoding.encode(&mask_message, rng);
        let (mask_commitment, mask_prover_data) = self.mmcs.commit_matrix(mask_codeword);
        challenger.observe(mask_commitment.clone());

        let rho_ood_points = (0..round_zk.ood_samples)
            .map(|_| challenger.sample_algebra_element())
            .collect::<Vec<EF>>();
        let source_message = folded_evaluations.as_slice();
        let mask_message_ext = mask_message
            .iter()
            .copied()
            .map(EF::from)
            .collect::<Vec<_>>();
        let private_ood_answers =
            private_ood_answers(&rho_ood_points, source_message, &mask_message_ext);
        challenger.observe_algebra_slice(&private_ood_answers);

        if round_params.pow_bits > 0 {
            proof.whir.rounds[round_index].pow_witness = challenger.grind(round_params.pow_bits);
        }
        challenger.sample();

        let row_indices = get_challenge_stir_queries::<Challenger, F, EF>(
            round_params.domain_size,
            self.params.folding_factor.at_round(round_index),
            round_zk.mask_query_budget,
            challenger,
        );
        let row_width = 1usize << self.params.folding_factor.at_round(round_index);
        let mut source_queries = Vec::with_capacity(row_indices.len());
        let mut source_openings = Vec::with_capacity(row_indices.len() * row_width);
        let mut query_positions = Vec::with_capacity(row_indices.len() * row_width);
        for &row in &row_indices {
            let opening = self.mmcs.open_batch(row, &state.source_merkle_data);
            let values = opening.opened_values[0].clone();
            for (limb, &value) in values.iter().enumerate() {
                source_openings.push(EF::from(value));
                query_positions.push(row * row_width + limb);
            }
            source_queries.push(QueryOpening::Base {
                values,
                proof: opening.opening_proof,
            });
        }

        let mask_indices = get_challenge_stir_queries::<Challenger, F, EF>(
            round_zk.mask_domain_size,
            0,
            round_zk.mask_query_budget,
            challenger,
        );
        let mut mask_queries = Vec::with_capacity(mask_indices.len());
        for &position in &mask_indices {
            let opening = self.mmcs.open_batch(position, &mask_prover_data);
            mask_queries.push(QueryOpening::Base {
                values: opening.opened_values[0].clone(),
                proof: opening.opening_proof,
            });
        }

        let batching_dim = 1 + private_ood_answers.len() + source_openings.len();
        let batching_challenge = challenger.sample_algebra_element();
        let coeffs = batching_coefficients(batching_challenge, batching_dim);
        let claim = ZkMaskClaim {
            base_claim_coeff: coeffs[0],
            residual_sumcheck_scale: handoff.eps,
            ood_coeffs: coeffs[1..1 + private_ood_answers.len()].to_vec(),
            in_domain_coeffs: coeffs[1 + private_ood_answers.len()..].to_vec(),
            source_randomness_weights: Vec::new(),
            pad_weights: Vec::new(),
        };
        let _mu_prime = batched_claim(
            handoff
                .residual_prover
                .evals()
                .as_slice()
                .iter()
                .copied()
                .sum(),
            &private_ood_answers,
            &source_openings,
            &claim,
        )
        .expect("honest code-switch batching dimensions should match");
        let _output_relation = output_relation(
            source_encoding,
            &EF::zero_vec(source_encoding.message_len()),
            &[],
            0,
            round_zk.mask_message_len,
            &rho_ood_points,
            &query_positions,
            &claim,
        )
        .expect("honest code-switch relation dimensions should match");

        let mut zk_sumcheck = ZkSumcheckData::default();
        let next_handoff = handoff.residual_prover.into_zk_sumcheck(
            &mut zk_sumcheck,
            mask_encoding,
            &self.mmcs,
            folding_factor_next,
            round_params.folding_pow_bits,
            challenger,
            rng,
        );
        proof.whir.rounds[round_index].zk = Some(WhirRoundZkProof {
            mask_commitment,
            private_ood_answers,
            source_queries,
            mask_queries,
            zk_sumcheck,
            zk_sumcheck_mask_commitments: next_handoff
                .mask_oracles
                .iter()
                .map(|(commitment, _)| commitment.clone())
                .collect(),
        });

        WhirZkPrefixRoundState {
            proof,
            handoff: next_handoff,
            target_merkle_data,
        }
    }

    /// Replay the initial ZK handoff and the first prefix-only ZK code-switch round.
    ///
    /// This verifier helper is intentionally scoped to the dedicated ZK API. It
    /// does not route through the plain PCS verifier, and it checks the source
    /// and mask openings carried by `WhirRoundZkProof`.
    #[allow(clippy::too_many_lines)]
    pub fn verify_round_zk_prefix<SourceEnc>(
        &self,
        commitment: &MT::Commitment,
        proof: &PcsProof<F, EF, MT>,
        protocol: &OpeningProtocol,
        source_encoding: &SourceEnc,
        challenger: &mut Challenger,
    ) -> Result<ZkVerifierHandoff<EF>, VerifierError>
    where
        SourceEnc: LinearZkEncoding<F>,
    {
        let zk_config = self
            .config
            .zk
            .as_ref()
            .ok_or(VerifierError::ZkVerifierRequiresPrefixPath)?;
        assert!(
            zk_config.only_prefix,
            "ZK WHIR currently supports only prefix layout"
        );

        challenger.observe(commitment.clone());

        let mut initial_verifier = ZkVerifier::<F, EF>::new(&protocol.table_shapes());
        for &eval in &proof.whir.initial_ood_answers {
            initial_verifier.add_virtual_eval(eval, challenger);
        }
        if protocol.num_openings() != proof.evals.len() {
            return Err(VerifierError::OpeningBatchCountMismatch {
                expected: protocol.num_openings(),
                actual: proof.evals.len(),
            });
        }
        for ((table_idx, polys), evals) in protocol.iter_openings().zip(&proof.evals) {
            if polys.len() != evals.len() {
                return Err(VerifierError::OpeningBatchSizeMismatch {
                    table_idx,
                    expected: polys.len(),
                    actual: evals.len(),
                });
            }
            initial_verifier.add_claim(table_idx, polys, evals, challenger);
        }

        let initial_zk = proof
            .whir
            .initial_zk
            .as_ref()
            .ok_or(VerifierError::UnexpectedInitialZkPayloadInPlainProof)?;
        let initial_handoff = initial_verifier.into_sumcheck::<MT, _>(
            &initial_zk.zk_sumcheck,
            &initial_zk.zk_sumcheck_mask_commitments,
            zk_config.mask_message_len,
            self.params.folding_factor.at_round(0),
            self.starting_folding_pow_bits,
            challenger,
        )?;

        let round_index = 0;
        let round_params = &self.round_parameters[round_index];
        let round_zk = round_params
            .zk
            .as_ref()
            .expect("verify_round_zk_prefix requires RoundConfig::zk");
        let round = proof
            .whir
            .rounds
            .get(round_index)
            .ok_or(VerifierError::InvalidRoundIndex { index: round_index })?;
        let target_commitment = round
            .commitment
            .clone()
            .ok_or(VerifierError::MissingRoundCommitment { round: round_index })?;
        challenger.observe(target_commitment);

        let round_zk_proof = round
            .zk
            .as_ref()
            .ok_or(VerifierError::UnexpectedZkPayloadInPlainProof { round: 0 })?;
        challenger.observe(round_zk_proof.mask_commitment.clone());

        let rho_ood_points = (0..round_zk.ood_samples)
            .map(|_| challenger.sample_algebra_element())
            .collect::<Vec<EF>>();
        if round_zk_proof.private_ood_answers.len() != rho_ood_points.len() {
            return Err(VerifierError::StirQueryCountMismatch {
                round_index,
                expected: rho_ood_points.len(),
                actual: round_zk_proof.private_ood_answers.len(),
            });
        }
        challenger.observe_algebra_slice(&round_zk_proof.private_ood_answers);

        if round_params.pow_bits > 0
            && !challenger.check_witness(round_params.pow_bits, round.pow_witness)
        {
            return Err(VerifierError::InvalidPowWitness);
        }
        challenger.sample();

        let row_indices = get_challenge_stir_queries::<Challenger, F, EF>(
            round_params.domain_size,
            self.params.folding_factor.at_round(round_index),
            round_zk.mask_query_budget,
            challenger,
        );
        if round_zk_proof.source_queries.len() != row_indices.len() {
            return Err(VerifierError::StirQueryCountMismatch {
                round_index,
                expected: row_indices.len(),
                actual: round_zk_proof.source_queries.len(),
            });
        }
        let row_width = 1usize << self.params.folding_factor.at_round(round_index);
        let source_dimensions = [Dimensions {
            height: round_params.domain_size >> self.params.folding_factor.at_round(round_index),
            width: row_width,
        }];
        let mut source_openings = Vec::with_capacity(row_indices.len() * row_width);
        let mut query_positions = Vec::with_capacity(row_indices.len() * row_width);
        for (&row, query) in row_indices.iter().zip(&round_zk_proof.source_queries) {
            let QueryOpening::Base { values, proof } = query else {
                return Err(VerifierError::MerkleProofInvalid {
                    position: row,
                    reason: "Expected base-field source opening in first ZK round".into(),
                });
            };
            self.mmcs
                .verify_batch(
                    commitment,
                    &source_dimensions,
                    row,
                    BatchOpeningRef {
                        opened_values: from_ref(values),
                        opening_proof: proof,
                    },
                )
                .map_err(|_| VerifierError::MerkleProofInvalid {
                    position: row,
                    reason: "ZK source opening verification failed".into(),
                })?;
            for (limb, &value) in values.iter().enumerate() {
                source_openings.push(EF::from(value));
                query_positions.push(row * row_width + limb);
            }
        }

        let mask_indices = get_challenge_stir_queries::<Challenger, F, EF>(
            round_zk.mask_domain_size,
            0,
            round_zk.mask_query_budget,
            challenger,
        );
        if round_zk_proof.mask_queries.len() != mask_indices.len() {
            return Err(VerifierError::StirQueryCountMismatch {
                round_index,
                expected: mask_indices.len(),
                actual: round_zk_proof.mask_queries.len(),
            });
        }
        let mask_dimensions = [Dimensions {
            height: round_zk.mask_domain_size,
            width: 1,
        }];
        for (&position, query) in mask_indices.iter().zip(&round_zk_proof.mask_queries) {
            let QueryOpening::Base { values, proof } = query else {
                return Err(VerifierError::MerkleProofInvalid {
                    position,
                    reason: "Expected base-field mask opening".into(),
                });
            };
            self.mmcs
                .verify_batch(
                    &round_zk_proof.mask_commitment,
                    &mask_dimensions,
                    position,
                    BatchOpeningRef {
                        opened_values: from_ref(values),
                        opening_proof: proof,
                    },
                )
                .map_err(|_| VerifierError::MerkleProofInvalid {
                    position,
                    reason: "ZK mask opening verification failed".into(),
                })?;
        }

        let batching_dim = 1 + round_zk_proof.private_ood_answers.len() + source_openings.len();
        let batching_challenge = challenger.sample_algebra_element();
        let coeffs = batching_coefficients(batching_challenge, batching_dim);
        let claim = ZkMaskClaim {
            base_claim_coeff: coeffs[0],
            residual_sumcheck_scale: initial_handoff.eps,
            ood_coeffs: coeffs[1..1 + round_zk_proof.private_ood_answers.len()].to_vec(),
            in_domain_coeffs: coeffs[1 + round_zk_proof.private_ood_answers.len()..].to_vec(),
            source_randomness_weights: Vec::new(),
            pad_weights: Vec::new(),
        };
        let _ = batched_claim(
            initial_handoff.claimed_residual,
            &round_zk_proof.private_ood_answers,
            &source_openings,
            &claim,
        )
        .map_err(|err| VerifierError::MerkleProofInvalid {
            position: 0,
            reason: err.to_string(),
        })?;
        let _ = output_relation(
            source_encoding,
            &EF::zero_vec(source_encoding.message_len()),
            &[],
            0,
            round_zk.mask_message_len,
            &rho_ood_points,
            &query_positions,
            &claim,
        )
        .map_err(|err| VerifierError::MerkleProofInvalid {
            position: 0,
            reason: err.to_string(),
        })?;

        let nested_verifier = ZkVerifier::<F, EF>::new(&[]);
        nested_verifier
            .into_sumcheck::<MT, _>(
                &round_zk_proof.zk_sumcheck,
                &round_zk_proof.zk_sumcheck_mask_commitments,
                round_zk.mask_message_len,
                self.params.folding_factor.at_round(round_index + 1),
                round_params.folding_pow_bits,
                challenger,
            )
            .map_err(VerifierError::from)
    }
}

impl<EF, F, Dft, MT, Challenger, L> MultilinearPcs<EF, Challenger>
    for WhirProver<EF, F, Dft, MT, Challenger, L>
where
    F: TwoAdicField + Ord,
    EF: ExtensionField<F> + TwoAdicField,
    Dft: TwoAdicSubgroupDft<F>,
    MT: Mmcs<F>,
    Challenger: FieldChallenger<F>
        + GrindingChallenger<Witness = F>
        + CanSampleUniformBits<F>
        + CanObserve<MT::Commitment>,
    L: Layout<F, EF>,
{
    type Commitment = MT::Commitment;
    type Val = F;
    type ProverData = WhirProverData<F, EF, MT, L>;
    type Proof = PcsProof<F, EF, MT>;
    type Error = VerifierError;
    type Witness = Witness<F>;
    type OpeningProtocol = OpeningProtocol;

    fn num_vars(&self) -> usize {
        self.config.num_variables
    }

    fn commit(
        &self,
        witness: Self::Witness,
        challenger: &mut Challenger,
    ) -> (Self::Commitment, Self::ProverData) {
        assert_eq!(witness.num_variables(), self.config.num_variables);
        let (layout, commitment, merkle_data) = L::commit(
            &self.dft,
            &self.mmcs,
            challenger,
            witness,
            self.config.folding_factor.at_round(0),
            self.config.starting_log_inv_rate,
        );
        (
            commitment,
            WhirProverData {
                layout,
                merkle_data,
                _marker: PhantomData,
            },
        )
    }

    fn open(
        &self,
        mut prover_data: Self::ProverData,
        protocol: Self::OpeningProtocol,
        challenger: &mut Challenger,
    ) -> Self::Proof {
        let mut whir_proof = self.config.empty_proof();
        tracing::info_span!("ood claims").in_scope(|| {
            whir_proof.initial_ood_answers = (0..self.commitment_ood_samples)
                .map(|_| prover_data.layout.add_virtual_eval(challenger))
                .collect::<Vec<_>>();
        });

        let evals = protocol
            .iter_openings()
            .map(|(table_idx, polys)| prover_data.layout.eval(table_idx, polys, challenger))
            .collect::<Vec<_>>();

        self.prove(
            &mut whir_proof,
            challenger,
            prover_data.layout,
            prover_data.merkle_data,
        );

        PcsProof {
            whir: whir_proof,
            evals,
        }
    }

    fn verify(
        &self,
        commitment: &Self::Commitment,
        proof: &Self::Proof,
        challenger: &mut Challenger,
        protocol: Self::OpeningProtocol,
    ) -> Result<(), Self::Error> {
        challenger.observe(commitment.clone());

        let mut layout_verifier = Verifier::<F, EF>::new(&protocol.table_shapes(), L::strategy());
        for &eval in &proof.whir.initial_ood_answers {
            layout_verifier.add_virtual_eval(eval, challenger);
        }
        if protocol.num_openings() != proof.evals.len() {
            return Err(VerifierError::OpeningBatchCountMismatch {
                expected: protocol.num_openings(),
                actual: proof.evals.len(),
            });
        }
        for ((table_idx, polys), evals) in protocol.iter_openings().zip(&proof.evals) {
            if polys.len() != evals.len() {
                return Err(VerifierError::OpeningBatchSizeMismatch {
                    table_idx,
                    expected: polys.len(),
                    actual: evals.len(),
                });
            }
            layout_verifier.add_claim(table_idx, polys, evals, challenger);
        }

        let alpha = challenger.sample_algebra_element();
        let constraint = layout_verifier.constraint(alpha);
        let mut claimed_eval = EF::ZERO;
        constraint.combine_evals(&mut claimed_eval);

        let verifier = WhirVerifier::new(&self.config, &self.mmcs, L::variable_order());
        verifier.verify(
            &proof.whir,
            challenger,
            commitment,
            constraint,
            claimed_eval,
        )?;

        Ok(())
    }
}
