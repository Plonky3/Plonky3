//! Adapter implementing the multilinear PCS trait for the WHIR protocol.

use alloc::vec::Vec;
use core::marker::PhantomData;

use p3_challenger::{CanObserve, CanSampleUniformBits, FieldChallenger, GrindingChallenger};
use p3_commit::{Mmcs, MultilinearPcs};
use p3_dft::TwoAdicSubgroupDft;
use p3_field::{ExtensionField, TwoAdicField};
use p3_matrix::dense::DenseMatrix;

use super::prover::WhirProver;
use super::verifier::WhirVerifier;
use super::verifier::errors::VerifierError;
use crate::pcs::proof::PcsProof;
use crate::sumcheck::OpeningProtocol;
use crate::sumcheck::layout::{Layout, Verifier, Witness};

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
