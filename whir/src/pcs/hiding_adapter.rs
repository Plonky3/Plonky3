//! Hiding (HVZK) adapter for the WHIR polynomial commitment scheme.
//!
//! Wraps the standard [`WhirProver`] with `zk: true` enforced and a
//! caller-provided RNG for ZK randomness generation.
//!
//! Mirrors the design of `HidingFriPcs` in the `p3-fri` crate.

use alloc::vec::Vec;

use p3_challenger::{CanObserve, CanSampleUniformBits, FieldChallenger, GrindingChallenger};
use p3_commit::{Mmcs, MultilinearPcs};
use p3_dft::TwoAdicSubgroupDft;
use p3_field::{ExtensionField, TwoAdicField};
use rand::Rng;
use rand::distr::{Distribution, StandardUniform};
use spin::Mutex;

use super::adapter::WhirProverData;
use super::prover::WhirProver;
use super::verifier::errors::VerifierError;
use crate::pcs::proof::PcsProof;
use crate::sumcheck::OpeningProtocol;
use crate::sumcheck::layout::{Layout, Witness};

/// A hiding (HVZK) WHIR PCS adapter.
///
/// Enforces `zk: true` in the protocol parameters and provides
/// private randomness via the stored RNG. The non-ZK [`WhirProver`]
/// adapter uses a dummy RNG that is never sampled.
#[derive(Debug)]
pub struct HidingWhirPcs<EF, F, Dft, MT, Challenger, Layout, R>
where
    F: TwoAdicField,
    EF: ExtensionField<F>,
{
    inner: WhirProver<EF, F, Dft, MT, Challenger, Layout>,
    rng: Mutex<R>,
}

impl<EF, F, Dft, MT, Challenger, Layout, R: Clone> Clone
    for HidingWhirPcs<EF, F, Dft, MT, Challenger, Layout, R>
where
    F: TwoAdicField,
    EF: ExtensionField<F>,
    WhirProver<EF, F, Dft, MT, Challenger, Layout>: Clone,
{
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
            rng: Mutex::new(self.rng.lock().clone()),
        }
    }
}

impl<EF, F, Dft, MT, Challenger, L, R> HidingWhirPcs<EF, F, Dft, MT, Challenger, L, R>
where
    F: TwoAdicField + Ord,
    EF: ExtensionField<F> + TwoAdicField,
    Dft: TwoAdicSubgroupDft<F>,
    Challenger: FieldChallenger<F> + GrindingChallenger<Witness = F> + CanSampleUniformBits<F>,
    MT: Mmcs<F>,
    L: Layout<F, EF>,
{
    /// Build a hiding WHIR PCS from a prover and an RNG.
    ///
    /// Delegate to the inner prover's domain separator builder.
    pub fn add_domain_separator<const DIGEST_ELEMS: usize>(
        &self,
        ds: &mut crate::fiat_shamir::domain_separator::DomainSeparator<EF, F>,
    ) where
        EF: TwoAdicField,
    {
        self.inner.add_domain_separator::<DIGEST_ELEMS>(ds);
    }

    /// # Panics
    ///
    /// Panics if `inner.config.params.zk` is `false`.
    pub fn new(inner: WhirProver<EF, F, Dft, MT, Challenger, L>, rng: R) -> Self {
        assert!(
            inner.config.params.zk,
            "HidingWhirPcs requires zk: true in ProtocolParameters"
        );
        Self {
            inner,
            rng: Mutex::new(rng),
        }
    }
}

impl<EF, F, Dft, MT, Challenger, L, R> MultilinearPcs<EF, Challenger>
    for HidingWhirPcs<EF, F, Dft, MT, Challenger, L, R>
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
    R: Rng + Send + Sync,
    StandardUniform: Distribution<EF>,
{
    type Commitment = MT::Commitment;
    type Val = F;
    type ProverData = WhirProverData<F, EF, MT, L>;
    type Proof = PcsProof<F, EF, MT>;
    type Error = VerifierError;
    type Witness = Witness<F>;
    type OpeningProtocol = OpeningProtocol;

    fn num_vars(&self) -> usize {
        self.inner.config.num_variables
    }

    fn commit(
        &self,
        witness: Self::Witness,
        challenger: &mut Challenger,
    ) -> (Self::Commitment, Self::ProverData) {
        <WhirProver<EF, F, Dft, MT, Challenger, L> as MultilinearPcs<EF, Challenger>>::commit(
            &self.inner,
            witness,
            challenger,
        )
    }

    fn open(
        &self,
        mut prover_data: Self::ProverData,
        protocol: Self::OpeningProtocol,
        challenger: &mut Challenger,
    ) -> Self::Proof {
        let mut whir_proof = self.inner.config.empty_proof();
        tracing::info_span!("ood claims").in_scope(|| {
            whir_proof.initial_ood_answers = (0..self.inner.commitment_ood_samples)
                .map(|_| prover_data.layout.add_virtual_eval(challenger))
                .collect::<Vec<_>>();
        });

        let evals = protocol
            .iter_openings()
            .map(|(table_idx, polys)| prover_data.layout.eval(table_idx, polys, challenger))
            .collect::<Vec<_>>();

        self.inner.prove(
            &mut whir_proof,
            challenger,
            prover_data.layout,
            prover_data.merkle_data,
            &mut *self.rng.lock(),
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
        <WhirProver<EF, F, Dft, MT, Challenger, L> as MultilinearPcs<EF, Challenger>>::verify(
            &self.inner,
            commitment,
            proof,
            challenger,
            protocol,
        )
    }
}
