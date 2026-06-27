//! Hiding multilinear PCS adapter over the HVZK-WHIR pipeline.

use alloc::vec::Vec;

use p3_challenger::{CanObserve, CanSampleUniformBits, FieldChallenger, GrindingChallenger};
use p3_commit::{Mmcs, MultilinearPcs};
use p3_dft::TwoAdicSubgroupDft;
use p3_field::{ExtensionField, TwoAdicField};
use p3_multilinear_util::point::Point;
use p3_multilinear_util::poly::Poly;
use rand::Rng;
use rand::distr::{Distribution, StandardUniform};
use spin::Mutex;

use super::config::ZkWhirConfig;
use super::proof::ZkWhirProof;
use super::prover::{HidingWhirProver, HidingWhirProverData};
use super::verifier::{HidingWhirVerifier, ZkVerifierError};
use crate::fiat_shamir::domain_separator::DomainSeparator;

/// A hiding WHIR PCS, mirroring the hiding FRI adapter.
///
/// - Opening a set of evaluation claims reveals exactly those evaluations.
/// - Nothing else about the committed polynomial leaks.
/// - The guarantee is honest-verifier zero knowledge (eprint 2026/391).
#[derive(Debug)]
pub struct HidingWhirPcs<EF, F, Dft, MT, Challenger, R>
where
    F: TwoAdicField,
    EF: ExtensionField<F>,
{
    /// Derived HVZK configuration.
    pub config: ZkWhirConfig<EF, F, Challenger>,
    /// FFT engine for every codeword encoding.
    pub dft: Dft,
    /// Base-field Merkle commitment scheme.
    pub mmcs: MT,
    /// Source of the prover-side hiding randomness.
    rng: Mutex<R>,
}

impl<EF, F, Dft, MT, Challenger, R> HidingWhirPcs<EF, F, Dft, MT, Challenger, R>
where
    F: TwoAdicField,
    EF: ExtensionField<F>,
{
    /// Bundles the PCS dependencies.
    ///
    /// # Security
    ///
    /// All hiding randomness — encoding masks, sumcheck masks, out-of-domain
    /// pads, and base-case one-time pads — is drawn from `rng`.
    /// Zero knowledge holds only if `rng` is a cryptographically secure
    /// generator; a predictable stream lets an observer strip every mask and
    /// recover the witness from the reveals.
    /// Tests seed a deterministic generator on purpose.
    pub const fn new(config: ZkWhirConfig<EF, F, Challenger>, dft: Dft, mmcs: MT, rng: R) -> Self {
        Self {
            config,
            dft,
            mmcs,
            rng: Mutex::new(rng),
        }
    }
}

impl<EF, F, Dft, MT, Challenger, R> HidingWhirPcs<EF, F, Dft, MT, Challenger, R>
where
    F: TwoAdicField,
    EF: ExtensionField<F> + TwoAdicField,
    Challenger: FieldChallenger<F> + GrindingChallenger<Witness = F>,
{
    /// Builds the Fiat-Shamir domain separator for this protocol instance.
    ///
    /// Encodes the protocol parameters and the full HVZK transcript shape.
    /// Every challenge is thereby bound to this configuration.
    pub fn add_domain_separator<const DIGEST_ELEMS: usize>(&self, ds: &mut DomainSeparator<EF, F>) {
        ds.commit_statement_hvzk::<Challenger, DIGEST_ELEMS>(&self.config.inner);
        ds.add_zk_whir_proof::<Challenger, DIGEST_ELEMS>(&self.config);
    }
}

impl<EF, F, Dft, MT, Challenger, R> MultilinearPcs<EF, Challenger>
    for HidingWhirPcs<EF, F, Dft, MT, Challenger, R>
where
    F: TwoAdicField,
    EF: ExtensionField<F> + TwoAdicField,
    Dft: TwoAdicSubgroupDft<F>,
    MT: Mmcs<F>,
    Challenger: FieldChallenger<F>
        + GrindingChallenger<Witness = F>
        + CanSampleUniformBits<F>
        + CanObserve<MT::Commitment>,
    R: Rng + Send + Sync,
    StandardUniform: Distribution<EF> + Distribution<F>,
{
    type Commitment = MT::Commitment;
    type Val = F;
    type ProverData = HidingWhirProverData<F, EF, MT>;
    type Proof = ZkWhirProof<F, EF, MT>;
    type Error = ZkVerifierError;
    type Witness = Poly<F>;
    type OpeningProtocol = Vec<Point<EF>>;

    fn num_vars(&self) -> usize {
        self.config.num_variables
    }

    fn commit(
        &self,
        witness: Self::Witness,
        challenger: &mut Challenger,
    ) -> (Self::Commitment, Self::ProverData) {
        let prover = HidingWhirProver::new(&self.config, &self.dft, &self.mmcs);
        let mut rng = self.rng.lock();
        prover.commit(witness, challenger, &mut *rng)
    }

    fn open(
        &self,
        prover_data: Self::ProverData,
        protocol: Self::OpeningProtocol,
        challenger: &mut Challenger,
    ) -> Self::Proof {
        // Evaluate and bind the public claims: points and values.
        let claims: Vec<(Point<EF>, EF)> = protocol
            .into_iter()
            .map(|point| {
                let eval = prover_data.message.eval_base(&point);
                challenger.observe_algebra_slice(point.as_slice());
                challenger.observe_algebra_element(eval);
                (point, eval)
            })
            .collect();

        let prover = HidingWhirProver::new(&self.config, &self.dft, &self.mmcs);
        let mut rng = self.rng.lock();
        prover.prove(prover_data, &claims, challenger, &mut *rng)
    }

    fn verify(
        &self,
        commitment: &Self::Commitment,
        proof: &Self::Proof,
        challenger: &mut Challenger,
        protocol: Self::OpeningProtocol,
    ) -> Result<(), Self::Error> {
        challenger.observe(commitment.clone());

        if proof.evals.len() != protocol.len() {
            return Err(ZkVerifierError::EvalCountMismatch {
                expected: protocol.len(),
                actual: proof.evals.len(),
            });
        }
        // Bind the public claims exactly as the prover did.
        let claims: Vec<(Point<EF>, EF)> = protocol
            .into_iter()
            .zip(proof.evals.iter().copied())
            .map(|(point, eval)| {
                challenger.observe_algebra_slice(point.as_slice());
                challenger.observe_algebra_element(eval);
                (point, eval)
            })
            .collect();

        let verifier = HidingWhirVerifier::new(&self.config, &self.mmcs);
        verifier.verify(proof, commitment, &claims, challenger)
    }
}
