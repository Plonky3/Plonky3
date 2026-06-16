//! Adapter implementing the multilinear PCS trait for the WHIR protocol.

use alloc::vec::Vec;
use core::marker::PhantomData;

use p3_challenger::{CanObserve, CanSampleUniformBits, FieldChallenger, GrindingChallenger};
use p3_commit::{Mmcs, MultilinearPcs};
use p3_dft::TwoAdicSubgroupDft;
use p3_field::{ExtensionField, TwoAdicField};
use p3_matrix::dense::DenseMatrix;
use p3_sumcheck::OpeningProtocol;
use p3_sumcheck::layout::{Layout, Verifier, Witness};

use super::prover::WhirProver;
use super::verifier::WhirVerifier;
use super::verifier::errors::VerifierError;
use crate::pcs::proof::PcsProof;

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
            self.config.round_folding_factor(0),
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
            .map(|(table_idx, batch)| prover_data.layout.eval(table_idx, batch, challenger))
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

        // The commitment fixes how many initial OOD answers exist.
        // Each one drives a transcript draw, so a wrong count desyncs Fiat-Shamir.
        // Reject it up front, matching the per-round OOD guard.
        if proof.whir.initial_ood_answers.len() != self.commitment_ood_samples {
            return Err(VerifierError::InitialOodAnswerCountMismatch {
                expected: self.commitment_ood_samples,
                actual: proof.whir.initial_ood_answers.len(),
            });
        }
        for &eval in &proof.whir.initial_ood_answers {
            layout_verifier.add_virtual_eval(eval, challenger);
        }
        if protocol.num_openings() != proof.evals.len() {
            return Err(VerifierError::OpeningBatchCountMismatch {
                expected: protocol.num_openings(),
                actual: proof.evals.len(),
            });
        }
        for ((table_idx, batch), evals) in protocol.iter_openings().zip(&proof.evals) {
            // The supplied evaluations must match the schedule entry on both counts:
            // the current-point list and the repeat-last successor list.
            if !batch.has_same_shape(evals) {
                return Err(VerifierError::OpeningBatchSizeMismatch {
                    table_idx,
                    expected: batch.len(),
                    actual: evals.len(),
                });
            }
            // The shape was pre-checked above, so this never errors in practice.
            // The inner check is defense in depth.
            // Its error maps into a verifier error if ever reached.
            layout_verifier.add_claim(table_idx, batch, evals, challenger)?;
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
