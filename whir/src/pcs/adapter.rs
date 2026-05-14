//! Adapter implementing the multilinear PCS trait for the WHIR protocol.

use alloc::vec::Vec;
use core::marker::PhantomData;

use p3_challenger::{CanObserve, CanSampleUniformBits, FieldChallenger, GrindingChallenger};
use p3_commit::{Mmcs, MultilinearPcs};
use p3_dft::TwoAdicSubgroupDft;
use p3_field::{ExtensionField, TwoAdicField};
use p3_matrix::Matrix;
use p3_matrix::dense::DenseMatrix;
use p3_zk_codes::ZkEncoding;
use rand::Rng;
use rand::distr::{Distribution, StandardUniform};

use super::prover::WhirProver;
use super::verifier::WhirVerifier;
use super::verifier::errors::VerifierError;
use crate::pcs::proof::{PcsProof, WhirInitialZkProof};
use crate::sumcheck::OpeningProtocol;
use crate::sumcheck::layout::{Layout, PrefixProver, Verifier, Witness};
use crate::sumcheck::zk::{ZkPrefixProver, ZkSumcheckData, ZkSumcheckHandoff};

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
