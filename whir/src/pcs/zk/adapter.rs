//! Hiding multilinear PCS adapter over the HVZK-WHIR pipeline.

use alloc::vec::Vec;

use p3_challenger::{CanObserve, CanSampleUniformBits, FieldChallenger, GrindingChallenger};
use p3_commit::{Mmcs, MultilinearPcs};
use p3_dft::TwoAdicSubgroupDft;
use p3_field::{ExtensionField, PrimeField64, TwoAdicField};
use p3_multilinear_util::point::Point;
use p3_multilinear_util::poly::Poly;
use rand::distr::{Distribution, StandardUniform};
use rand::rngs::StdRng;
use rand::{CryptoRng, SeedableRng};
use spin::Mutex;

use super::config::ZkWhirConfig;
use super::proof::ZkWhirProof;
use super::prover::{HidingWhirProver, HidingWhirProverData};
use super::verifier::{HidingWhirVerifier, ZkVerifierError};
use crate::WhirConfigError;
use crate::pcs::zk::verifier::check_claim_arity;
use crate::transcript::zk::{observe_claims, observe_commitment};

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
    R: CryptoRng,
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
    /// The [`CryptoRng`] bound rules out known non-cryptographic generators,
    /// but production callers must still seed their generator from
    /// unpredictable entropy. Tests use a deterministic seed only for
    /// reproducibility.
    ///
    /// `rng` itself is only ever used to seed a fresh `StdRng` (a CSPRNG) for
    /// each `commit`/`open` call, under a briefly-held lock; the forked generator
    /// then drives the (possibly long) proving work without holding the lock.
    pub const fn new(config: ZkWhirConfig<EF, F, Challenger>, dft: Dft, mmcs: MT, rng: R) -> Self {
        Self {
            config,
            dft,
            mmcs,
            rng: Mutex::new(rng),
        }
    }
}

impl<EF, F, Dft, MT, Challenger, R> MultilinearPcs<EF, Challenger>
    for HidingWhirPcs<EF, F, Dft, MT, Challenger, R>
where
    F: TwoAdicField + PrimeField64,
    EF: ExtensionField<F> + TwoAdicField,
    Dft: TwoAdicSubgroupDft<F>,
    MT: Mmcs<F>,
    Challenger: FieldChallenger<F>
        + GrindingChallenger<Witness = F>
        + CanSampleUniformBits<F>
        + CanObserve<MT::Commitment>,
    R: CryptoRng + Send + Sync,
    StandardUniform: Distribution<EF> + Distribution<F>,
{
    type Commitment = MT::Commitment;
    type Val = F;
    type ProverData = HidingWhirProverData<F, EF, MT>;
    type Proof = ZkWhirProof<F, EF, MT>;
    type Error = ZkVerifierError;
    type ProverError = WhirConfigError;
    type Witness = Poly<F>;
    type OpeningProtocol = Vec<Point<EF>>;

    fn num_vars(&self) -> usize {
        self.config.num_variables
    }

    fn commit(
        &self,
        witness: Self::Witness,
        challenger: &mut Challenger,
    ) -> Result<(Self::Commitment, Self::ProverData), Self::ProverError> {
        let prover = HidingWhirProver::new(&self.config, &self.dft, &self.mmcs);
        let mut rng = StdRng::from_rng(&mut *self.rng.lock());
        let (commitment, prover_data) = prover.commit(witness, &mut rng);

        // The verifier reaches the same call, so neither side can bind differently.
        self.observe_commitment(&commitment, challenger);

        Ok((commitment, prover_data))
    }

    fn observe_commitment(&self, commitment: &Self::Commitment, challenger: &mut Challenger) {
        observe_commitment::<F, _, _>(challenger, commitment.clone());
    }

    fn open(
        &self,
        prover_data: Self::ProverData,
        protocol: Self::OpeningProtocol,
        challenger: &mut Challenger,
    ) -> Result<Self::Proof, Self::ProverError> {
        self.config.validate_initial_claims(protocol.len())?;

        // Evaluate the public claims, then bind the whole statement in one phase.
        let claims: Vec<(Point<EF>, EF)> = protocol
            .into_iter()
            .map(|point| {
                let eval = prover_data.message.eval_base(&point);
                (point, eval)
            })
            .collect();
        observe_claims::<F, EF, _>(challenger, &claims, self.config.num_variables);

        // The claims are bound and the hiding run starts here.
        // Its driver therefore seeds here, ahead of the run's first challenge.
        let prover = HidingWhirProver::new(&self.config, &self.dft, &self.mmcs);
        let mut rng = StdRng::from_rng(&mut *self.rng.lock());
        prover.prove(prover_data, &claims, challenger, &mut rng)
    }

    fn verify(
        &self,
        commitment: &Self::Commitment,
        proof: &Self::Proof,
        challenger: &mut Challenger,
        protocol: Self::OpeningProtocol,
    ) -> Result<(), Self::Error> {
        self.observe_commitment(commitment, challenger);

        if proof.evals.len() != protocol.len() {
            return Err(ZkVerifierError::EvalCountMismatch {
                expected: protocol.len(),
                actual: proof.evals.len(),
            });
        }
        // Pair each requested point with the value the proof claims at it.
        let claims: Vec<(Point<EF>, EF)> = protocol
            .into_iter()
            .zip(proof.evals.iter().copied())
            .collect();

        // The statement is the caller's, so its arity is checked before it is bound.
        //
        //     point arity != committed arity  ->  error, never a panic
        //
        // Binding first would describe a step width the point cannot fill.
        check_claim_arity(&claims, self.config.num_variables)?;

        // Bind the public claims exactly as the prover did.
        observe_claims::<F, EF, _>(challenger, &claims, self.config.num_variables);

        // The claims are bound and the hiding run starts here.
        // Its driver therefore seeds here, ahead of the run's first challenge.
        let verifier = HidingWhirVerifier::new(&self.config, &self.mmcs);
        verifier.verify(proof, commitment, &claims, challenger)
    }
}
