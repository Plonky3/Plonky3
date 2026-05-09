//! Root-level WARP proof composition.
//!
//! A WARP accumulation step proves `VACC` for one transition. A complete
//! application-level proof also needs to show that every transition in the
//! chain verifies and that the final accumulator satisfies the decider
//! relation `DACC`. This module packages that composition as an executable
//! root proof system:
//!
//! ```text
//!     VACC(step_0) ∧ VACC(step_1) ∧ ... ∧ DACC(final_acc)
//! ```
//!
//! This is the sound boundary described by WARP Construction 10.4:
//! `PACC/VACC` maintain the accumulator, and `DACC` checks the final
//! `(rt, alpha, mu, beta, eta)` against a witness `(f, w)`.
//!
//! This module intentionally keeps only the native root composition and the
//! reusable external-commitment hooks used by the WHIR compiler.

use alloc::vec::Vec;
use core::marker::PhantomData;

use p3_challenger::{CanFinalizeDigest, CanObserve, FieldChallenger, GrindingChallenger};
use p3_commit::Mmcs;
use p3_dft::TwoAdicSubgroupDft;
use p3_field::{ExtensionField, TwoAdicField};
use serde::{Deserialize, Serialize};

use crate::accumulator::{
    Accumulator, AccumulatorInstance, WarpProof, WarpProofExternal, WarpProofExternalBatched,
};
use crate::code::ReedSolomonCode;
use crate::error::WarpError;
use crate::finalize::{AccumulatorFinalizer, Finalizer};
use crate::protocol::prover::ExtProverData;
use crate::protocol::{
    AccumulatorBatchOpeningBackend, AccumulatorCommitmentBackend,
    ExternalCodewordBatchOpeningProver, ExternalCodewordBatchOpeningVerifier,
    ExternalCodewordOpeningProver, ExternalCodewordOpeningVerifier, ExternalCommitmentObserver,
    ExternalCommittedCodeword, WarpParams, WarpProver, WarpVerifier,
};
use crate::relation::BundledPesat;

/// One verified edge in a linear WARP accumulation chain.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound = "F: Serialize + serde::de::DeserializeOwned,
                  EF: Serialize + serde::de::DeserializeOwned,
                  Comm: Serialize + serde::de::DeserializeOwned,
                  Proof: Serialize + serde::de::DeserializeOwned")]
pub struct WarpRootStep<F, EF, Comm, Proof> {
    /// Number of fresh witnesses accumulated by this step (`ell_1`).
    pub num_fresh: usize,
    /// New accumulator instance output by this step.
    pub instance: AccumulatorInstance<EF, Comm>,
    /// Ordinary WARP accumulation proof for this step.
    pub proof: WarpProof<F, EF, Comm, Proof>,
}

/// Root proof for a linear WARP chain plus a finalizer proof.
///
/// For step `0`, the verifier checks `VACC` with no prior accumulator. For
/// every later step, the verifier uses the previous step's accumulator
/// instance as the single prior accumulator. The finalizer then verifies
/// `DACC` for the last accumulator instance.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound = "F: Serialize + serde::de::DeserializeOwned,
                  EF: Serialize + serde::de::DeserializeOwned,
                  Comm: Serialize + serde::de::DeserializeOwned,
                  Proof: Serialize + serde::de::DeserializeOwned,
                  FinalProof: Serialize + serde::de::DeserializeOwned")]
pub struct WarpRootProof<F, EF, Comm, Proof, FinalProof> {
    /// Linear sequence of WARP accumulation steps.
    pub steps: Vec<WarpRootStep<F, EF, Comm, Proof>>,
    /// Proof produced by the configured finalizer for the last accumulator.
    pub final_proof: FinalProof,
}

/// One verified edge in a linear WARP chain whose fresh inputs were committed
/// by an external PCS.
///
/// The fresh commitments are verifier-visible. The WARP accumulator itself
/// still uses the local Plonky3 `Mmcs`, so prior-accumulator openings keep the
/// local proof type while fresh openings use the external PCS proof type.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound = "F: Serialize + serde::de::DeserializeOwned,
                  EF: Serialize + serde::de::DeserializeOwned,
                  AccComm: Serialize + serde::de::DeserializeOwned,
                  FreshComm: Serialize + serde::de::DeserializeOwned,
                  FreshProof: Serialize + serde::de::DeserializeOwned,
                  AccProof: Serialize + serde::de::DeserializeOwned")]
pub struct WarpExternalRootStep<F, EF, AccComm, FreshComm, FreshProof, AccProof> {
    /// Verifier-visible commitments for this step's fresh inputs.
    pub fresh_commitments: Vec<FreshComm>,
    /// New accumulator instance output by this step.
    pub instance: AccumulatorInstance<EF, AccComm>,
    /// WARP accumulation proof for externally committed fresh inputs.
    pub proof: WarpProofExternal<F, EF, AccComm, FreshProof, AccProof>,
}

/// Root proof for a linear WARP chain whose fresh inputs come from an
/// external PCS, plus a finalizer proof for the last accumulator.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound = "F: Serialize + serde::de::DeserializeOwned,
                  EF: Serialize + serde::de::DeserializeOwned,
                  AccComm: Serialize + serde::de::DeserializeOwned,
                  FreshComm: Serialize + serde::de::DeserializeOwned,
                  FreshProof: Serialize + serde::de::DeserializeOwned,
                  AccProof: Serialize + serde::de::DeserializeOwned,
                  FinalProof: Serialize + serde::de::DeserializeOwned")]
pub struct WarpExternalRootProof<F, EF, AccComm, FreshComm, FreshProof, AccProof, FinalProof> {
    /// Linear sequence of externally committed WARP accumulation steps.
    pub steps: Vec<WarpExternalRootStep<F, EF, AccComm, FreshComm, FreshProof, AccProof>>,
    /// Proof produced by the configured finalizer for the last accumulator.
    pub final_proof: FinalProof,
}

/// One verified edge in a linear WARP chain whose shift openings are batched
/// per external fresh commitment and per prior accumulator.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound = "F: Serialize + serde::de::DeserializeOwned,
                  EF: Serialize + serde::de::DeserializeOwned,
                  AccComm: Serialize + serde::de::DeserializeOwned,
                  FreshComm: Serialize + serde::de::DeserializeOwned,
                  FreshProof: Serialize + serde::de::DeserializeOwned,
                  AccProof: Serialize + serde::de::DeserializeOwned")]
pub struct WarpExternalRootStepBatched<F, EF, AccComm, FreshComm, FreshProof, AccProof> {
    /// Verifier-visible commitments for this step's fresh inputs.
    pub fresh_commitments: Vec<FreshComm>,
    /// New accumulator instance output by this step.
    pub instance: AccumulatorInstance<EF, AccComm>,
    /// Batched-opening WARP accumulation proof for externally committed fresh inputs.
    pub proof: WarpProofExternalBatched<F, EF, AccComm, FreshProof, AccProof>,
}

/// Root proof for a linear external WARP chain using batched PCS openings.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound = "F: Serialize + serde::de::DeserializeOwned,
                  EF: Serialize + serde::de::DeserializeOwned,
                  AccComm: Serialize + serde::de::DeserializeOwned,
                  FreshComm: Serialize + serde::de::DeserializeOwned,
                  FreshProof: Serialize + serde::de::DeserializeOwned,
                  AccProof: Serialize + serde::de::DeserializeOwned,
                  FinalProof: Serialize + serde::de::DeserializeOwned")]
pub struct WarpExternalRootProofBatched<F, EF, AccComm, FreshComm, FreshProof, AccProof, FinalProof>
{
    /// Linear sequence of externally committed WARP accumulation steps.
    pub steps: Vec<WarpExternalRootStepBatched<F, EF, AccComm, FreshComm, FreshProof, AccProof>>,
    /// Proof produced by the configured finalizer for the last accumulator.
    pub final_proof: FinalProof,
}

/// Public shape of a linear WARP root proof.
///
/// This is the statement boundary a future succinct root proof should bind:
/// it records the arity of each `VACC` edge and the public accumulator
/// instance output by each edge. The step proofs and final decider witness
/// are private to the root proof backend.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(bound = "EF: Serialize + serde::de::DeserializeOwned,
                  Comm: Serialize + serde::de::DeserializeOwned")]
pub struct WarpRootClaim<EF, Comm> {
    /// Number of fresh witnesses accumulated at each step.
    pub step_num_fresh: Vec<usize>,
    /// Public accumulator instance after each step.
    pub step_instances: Vec<AccumulatorInstance<EF, Comm>>,
}

/// Public shape of a linear WARP root proof with external fresh commitments.
///
/// This is the statement boundary for an outer succinct proof: a verifier must
/// bind every external fresh commitment, every intermediate accumulator
/// instance, and the final accumulator instance before accepting a proof that
/// the WARP chain and finalizer verified.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(bound = "EF: Serialize + serde::de::DeserializeOwned,
                  AccComm: Serialize + serde::de::DeserializeOwned,
                  FreshComm: Serialize + serde::de::DeserializeOwned")]
pub struct WarpExternalRootClaim<EF, AccComm, FreshComm> {
    /// Verifier-visible fresh commitments used by each step.
    pub step_fresh_commitments: Vec<Vec<FreshComm>>,
    /// Public accumulator instance after each step.
    pub step_instances: Vec<AccumulatorInstance<EF, AccComm>>,
}

/// Native receipt for a successfully verified WARP root proof.
///
/// This is not yet a succinct proof. It is the public statement contract for
/// the outer WHIR proof: the outer proof should expose `claim_digest`
/// and prove that native verification of `claim` accepted.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(bound = "EF: Serialize + serde::de::DeserializeOwned,
                  Comm: Serialize + serde::de::DeserializeOwned,
                  Digest: Serialize + serde::de::DeserializeOwned")]
pub struct WarpRootReceipt<EF, Comm, Digest> {
    /// Verifier-visible WARP root claim.
    pub claim: WarpRootClaim<EF, Comm>,
    /// Transcript digest binding the root claim as public input.
    pub claim_digest: Digest,
    /// Final accumulator instance returned by the accepted root verifier.
    pub final_instance: AccumulatorInstance<EF, Comm>,
}

/// Native receipt for a successfully verified externally committed WARP root
/// proof.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(bound = "EF: Serialize + serde::de::DeserializeOwned,
                  AccComm: Serialize + serde::de::DeserializeOwned,
                  FreshComm: Serialize + serde::de::DeserializeOwned,
                  Digest: Serialize + serde::de::DeserializeOwned")]
pub struct WarpExternalRootReceipt<EF, AccComm, FreshComm, Digest> {
    /// Verifier-visible WARP root claim, including every external fresh
    /// commitment.
    pub claim: WarpExternalRootClaim<EF, AccComm, FreshComm>,
    /// Transcript digest binding the external root claim as public input.
    pub claim_digest: Digest,
    /// Final accumulator instance returned by the accepted root verifier.
    pub final_instance: AccumulatorInstance<EF, AccComm>,
}

impl<EF, AccComm, FreshComm> WarpExternalRootClaim<EF, AccComm, FreshComm> {
    /// Final accumulator instance claimed by the root proof.
    pub fn final_instance(&self) -> Option<&AccumulatorInstance<EF, AccComm>> {
        self.step_instances.last()
    }

    /// Observe this public claim into a challenger in canonical order.
    ///
    /// External fresh commitments are observed through the same backend
    /// verifier hook used by `VACC`, so external claims bind all metadata
    /// required by their opening verifier.
    pub fn observe_into<F, Challenger, FreshVerifier>(
        &self,
        challenger: &mut Challenger,
        fresh_verifier: &FreshVerifier,
    ) where
        F: TwoAdicField,
        EF: ExtensionField<F>,
        AccComm: Clone,
        FreshVerifier: ExternalCodewordOpeningVerifier<F, Challenger, Commitment = FreshComm>,
        Challenger: FieldChallenger<F> + CanObserve<AccComm>,
    {
        observe_root_domain::<F, _>(challenger, b"external");
        observe_usize::<F, _>(challenger, self.step_fresh_commitments.len());
        observe_usize::<F, _>(challenger, self.step_instances.len());
        for (fresh_commitments, instance) in self
            .step_fresh_commitments
            .iter()
            .zip(self.step_instances.iter())
        {
            observe_usize::<F, _>(challenger, fresh_commitments.len());
            for commitment in fresh_commitments {
                fresh_verifier.observe_commitment(challenger, commitment);
            }
            observe_accumulator_instance::<F, EF, AccComm, _>(challenger, instance);
        }
    }

    /// Observe this public claim when accumulator commitments are bound by a
    /// caller-provided backend rather than by `CanObserve<AccComm>` directly.
    pub fn observe_into_accumulator<F, Challenger, FreshVerifier, AccBackend>(
        &self,
        challenger: &mut Challenger,
        fresh_verifier: &FreshVerifier,
        acc_backend: &AccBackend,
    ) where
        F: TwoAdicField,
        EF: ExtensionField<F>,
        FreshVerifier: ExternalCodewordOpeningVerifier<F, Challenger, Commitment = FreshComm>,
        AccBackend: AccumulatorCommitmentBackend<F, EF, Challenger, Commitment = AccComm>,
        Challenger: FieldChallenger<F>,
    {
        observe_root_domain::<F, _>(challenger, b"external");
        observe_usize::<F, _>(challenger, self.step_fresh_commitments.len());
        observe_usize::<F, _>(challenger, self.step_instances.len());
        for (fresh_commitments, instance) in self
            .step_fresh_commitments
            .iter()
            .zip(self.step_instances.iter())
        {
            observe_usize::<F, _>(challenger, fresh_commitments.len());
            for commitment in fresh_commitments {
                fresh_verifier.observe_commitment(challenger, commitment);
            }
            observe_accumulator_instance_with_backend::<F, EF, AccBackend, _>(
                challenger,
                acc_backend,
                instance,
            );
        }
    }

    /// Return a digest binding this claim.
    pub fn digest<F, Challenger, FreshVerifier>(
        &self,
        mut challenger: Challenger,
        fresh_verifier: &FreshVerifier,
    ) -> Challenger::Digest
    where
        F: TwoAdicField,
        EF: ExtensionField<F>,
        AccComm: Clone,
        FreshVerifier: ExternalCodewordOpeningVerifier<F, Challenger, Commitment = FreshComm>,
        Challenger: FieldChallenger<F> + CanObserve<AccComm> + CanFinalizeDigest,
    {
        self.observe_into::<F, Challenger, FreshVerifier>(&mut challenger, fresh_verifier);
        challenger.finalize()
    }

    /// Return a digest binding this claim through an external accumulator
    /// backend.
    pub fn digest_accumulator<F, Challenger, FreshVerifier, AccBackend>(
        &self,
        mut challenger: Challenger,
        fresh_verifier: &FreshVerifier,
        acc_backend: &AccBackend,
    ) -> Challenger::Digest
    where
        F: TwoAdicField,
        EF: ExtensionField<F>,
        FreshVerifier: ExternalCodewordOpeningVerifier<F, Challenger, Commitment = FreshComm>,
        AccBackend: AccumulatorCommitmentBackend<F, EF, Challenger, Commitment = AccComm>,
        Challenger: FieldChallenger<F> + CanFinalizeDigest,
    {
        self.observe_into_accumulator::<F, Challenger, FreshVerifier, AccBackend>(
            &mut challenger,
            fresh_verifier,
            acc_backend,
        );
        challenger.finalize()
    }
}

impl<EF, Comm> WarpRootClaim<EF, Comm> {
    /// Final accumulator instance claimed by the root proof.
    pub fn final_instance(&self) -> Option<&AccumulatorInstance<EF, Comm>> {
        self.step_instances.last()
    }

    /// Observe this public claim into a challenger in canonical order.
    pub fn observe_into<F, Challenger>(&self, challenger: &mut Challenger)
    where
        F: TwoAdicField,
        EF: ExtensionField<F>,
        Comm: Clone,
        Challenger: FieldChallenger<F> + CanObserve<Comm>,
    {
        observe_root_domain::<F, _>(challenger, b"local");
        observe_usize::<F, _>(challenger, self.step_num_fresh.len());
        observe_usize::<F, _>(challenger, self.step_instances.len());
        for (&num_fresh, instance) in self.step_num_fresh.iter().zip(self.step_instances.iter()) {
            observe_usize::<F, _>(challenger, num_fresh);
            observe_accumulator_instance::<F, EF, Comm, _>(challenger, instance);
        }
    }

    /// Return a digest binding this claim.
    pub fn digest<F, Challenger>(&self, mut challenger: Challenger) -> Challenger::Digest
    where
        F: TwoAdicField,
        EF: ExtensionField<F>,
        Comm: Clone,
        Challenger: FieldChallenger<F> + CanObserve<Comm> + CanFinalizeDigest,
    {
        self.observe_into::<F, Challenger>(&mut challenger);
        challenger.finalize()
    }
}

fn observe_root_domain<F, Challenger>(challenger: &mut Challenger, variant: &[u8])
where
    F: TwoAdicField,
    Challenger: CanObserve<F>,
{
    observe_bytes::<F, _>(challenger, b"p3-warp-root-v1");
    observe_bytes::<F, _>(challenger, variant);
}

fn observe_bytes<F, Challenger>(challenger: &mut Challenger, bytes: &[u8])
where
    F: TwoAdicField,
    Challenger: CanObserve<F>,
{
    observe_usize(challenger, bytes.len());
    for &byte in bytes {
        challenger.observe(F::from_u8(byte));
    }
}

fn observe_usize<F, Challenger>(challenger: &mut Challenger, value: usize)
where
    F: TwoAdicField,
    Challenger: CanObserve<F>,
{
    challenger.observe(F::from_usize(value));
}

fn observe_accumulator_instance<F, EF, Comm, Challenger>(
    challenger: &mut Challenger,
    instance: &AccumulatorInstance<EF, Comm>,
) where
    F: TwoAdicField,
    EF: ExtensionField<F>,
    Comm: Clone,
    Challenger: FieldChallenger<F> + CanObserve<Comm>,
{
    challenger.observe(instance.rt.clone());
    observe_usize::<F, _>(challenger, instance.alpha.len());
    for &alpha in &instance.alpha {
        challenger.observe_algebra_element(alpha);
    }
    challenger.observe_algebra_element(instance.mu);
    observe_usize::<F, _>(challenger, instance.beta.len());
    for &beta in &instance.beta {
        challenger.observe_algebra_element(beta);
    }
    challenger.observe_algebra_element(instance.eta);
}

fn observe_accumulator_instance_with_backend<F, EF, AccBackend, Challenger>(
    challenger: &mut Challenger,
    acc_backend: &AccBackend,
    instance: &AccumulatorInstance<EF, AccBackend::Commitment>,
) where
    F: TwoAdicField,
    EF: ExtensionField<F>,
    AccBackend: AccumulatorCommitmentBackend<F, EF, Challenger>,
    Challenger: FieldChallenger<F>,
{
    acc_backend.observe_commitment(challenger, &instance.rt);
    observe_usize::<F, _>(challenger, instance.alpha.len());
    for &alpha in &instance.alpha {
        challenger.observe_algebra_element(alpha);
    }
    challenger.observe_algebra_element(instance.mu);
    observe_usize::<F, _>(challenger, instance.beta.len());
    for &beta in &instance.beta {
        challenger.observe_algebra_element(beta);
    }
    challenger.observe_algebra_element(instance.eta);
}

impl<F, EF, Comm, Proof, FinalProof> WarpRootProof<F, EF, Comm, Proof, FinalProof>
where
    AccumulatorInstance<EF, Comm>: Clone,
{
    /// Extract the verifier-visible root claim from this proof.
    pub fn claim(&self) -> WarpRootClaim<EF, Comm> {
        WarpRootClaim {
            step_num_fresh: self.steps.iter().map(|step| step.num_fresh).collect(),
            step_instances: self
                .steps
                .iter()
                .map(|step| step.instance.clone())
                .collect(),
        }
    }
}

impl<F, EF, AccComm, FreshComm, FreshProof, AccProof, FinalProof>
    WarpExternalRootProof<F, EF, AccComm, FreshComm, FreshProof, AccProof, FinalProof>
where
    AccumulatorInstance<EF, AccComm>: Clone,
    FreshComm: Clone,
{
    /// Extract the verifier-visible external root claim from this proof.
    pub fn claim(&self) -> WarpExternalRootClaim<EF, AccComm, FreshComm> {
        WarpExternalRootClaim {
            step_fresh_commitments: self
                .steps
                .iter()
                .map(|step| step.fresh_commitments.clone())
                .collect(),
            step_instances: self
                .steps
                .iter()
                .map(|step| step.instance.clone())
                .collect(),
        }
    }
}

/// Convenience alias for a root proof whose finalizer is witness-carrying.
pub type WitnessRootProof<F, EF, Comm, Proof> =
    WarpRootProof<F, EF, Comm, Proof, crate::WitnessProof<EF>>;

/// Prover for a linear WARP root proof.
pub struct WarpRootProver<'a, F, EF, MT, Dft, Pesat>
where
    F: TwoAdicField,
    EF: ExtensionField<F> + TwoAdicField,
    MT: Mmcs<F>,
    Dft: TwoAdicSubgroupDft<F>,
    Pesat: BundledPesat<F, EF>,
{
    step_prover: WarpProver<'a, F, EF, MT, Dft, Pesat>,
    _ph: PhantomData<(F, EF)>,
}

impl<'a, F, EF, MT, Dft, Pesat> WarpRootProver<'a, F, EF, MT, Dft, Pesat>
where
    F: TwoAdicField,
    EF: ExtensionField<F> + TwoAdicField,
    MT: Mmcs<F> + Sync,
    Dft: TwoAdicSubgroupDft<F>,
    Pesat: BundledPesat<F, EF>,
{
    /// Create a root prover over the same components as ordinary WARP.
    pub fn new(
        mmcs: &'a MT,
        code: &'a ReedSolomonCode<F, Dft>,
        pesat: &'a Pesat,
        params: WarpParams,
    ) -> Self {
        Self {
            step_prover: WarpProver::new(mmcs, code, pesat, params),
            _ph: PhantomData,
        }
    }

    /// Prove a linear chain and finalise its last accumulator.
    ///
    /// `step_witnesses[k]` contains the fresh witnesses for step `k`.
    /// Step `0` has no prior accumulator. Every later step folds exactly one
    /// prior: the accumulator produced by the preceding step.
    pub fn prove_linear_chain<Challenger, Fin>(
        &self,
        base_challenger: &Challenger,
        step_witnesses: &[Vec<Vec<F>>],
        finalizer: &Fin,
    ) -> Result<
        (
            AccumulatorInstance<EF, MT::Commitment>,
            WarpRootProof<F, EF, MT::Commitment, MT::Proof, Fin::Proof>,
        ),
        WarpError,
    >
    where
        Challenger: FieldChallenger<F>
            + GrindingChallenger<Witness = F>
            + CanObserve<MT::Commitment>
            + Clone,
        Fin: Finalizer<F, EF, MT, ExtProverData<F, EF, MT>>,
    {
        if step_witnesses.is_empty() {
            return Err(WarpError::Config("root proof requires at least one step"));
        }

        let mut current: Option<Accumulator<EF, MT::Commitment, ExtProverData<F, EF, MT>>> = None;
        let mut steps = Vec::with_capacity(step_witnesses.len());

        for fresh in step_witnesses {
            let prior_count = usize::from(current.is_some());
            let total = fresh.len() + prior_count;
            if !(total >= 2 && total.is_power_of_two()) {
                return Err(WarpError::Config(
                    "each root step must have ell = ell_1 + ell_2 as a power of two",
                ));
            }

            let priors = match current.take() {
                Some(acc) => alloc::vec![acc],
                None => Vec::new(),
            };
            let mut challenger = base_challenger.clone();
            let (next, proof) = self.step_prover.prove(&mut challenger, fresh, &priors);

            steps.push(WarpRootStep {
                num_fresh: fresh.len(),
                instance: next.instance.clone(),
                proof,
            });
            current = Some(next);
        }

        let final_acc = current.expect("non-empty step_witnesses checked above");
        let final_proof = finalizer.finalize(&final_acc.instance, &final_acc.witness)?;
        Ok((final_acc.instance, WarpRootProof { steps, final_proof }))
    }

    /// Prove a linear chain whose fresh inputs were already committed by an
    /// external PCS.
    ///
    /// This is the reusable external-handoff path used in the benchmarks. The
    /// returned proof contains all verifier-visible external
    /// fresh commitments, all intermediate accumulator instances, every WARP
    /// step proof, and the finalizer proof. A future succinct outer proof
    /// should prove verification of exactly this object.
    pub fn prove_external_linear_chain<Challenger, Fresh, FreshOpenings, Fin>(
        &self,
        base_challenger: &Challenger,
        fresh_openings: &FreshOpenings,
        step_fresh_committed: Vec<Vec<Fresh>>,
        finalizer: &Fin,
    ) -> Result<
        (
            AccumulatorInstance<EF, MT::Commitment>,
            WarpExternalRootProof<
                F,
                EF,
                MT::Commitment,
                Fresh::Commitment,
                FreshOpenings::Proof,
                MT::Proof,
                Fin::Proof,
            >,
        ),
        WarpError,
    >
    where
        Challenger: FieldChallenger<F>
            + GrindingChallenger<Witness = F>
            + CanObserve<MT::Commitment>
            + Clone,
        Fresh: ExternalCommittedCodeword<F> + ExternalCommitmentObserver<F, Challenger>,
        FreshOpenings: ExternalCodewordOpeningProver<F, Fresh>,
        Fin: Finalizer<F, EF, MT, ExtProverData<F, EF, MT>>,
    {
        if step_fresh_committed.is_empty() {
            return Err(WarpError::Config("root proof requires at least one step"));
        }

        let mut current: Option<Accumulator<EF, MT::Commitment, ExtProverData<F, EF, MT>>> = None;
        let mut steps = Vec::with_capacity(step_fresh_committed.len());

        for fresh in step_fresh_committed {
            let prior_count = usize::from(current.is_some());
            let total = fresh.len() + prior_count;
            if !(total >= 2 && total.is_power_of_two()) {
                return Err(WarpError::Config(
                    "each root step must have ell = ell_1 + ell_2 as a power of two",
                ));
            }

            let fresh_commitments = fresh.iter().map(Fresh::commitment).collect();
            let priors = match current.take() {
                Some(acc) => alloc::vec![acc],
                None => Vec::new(),
            };
            let mut challenger = base_challenger.clone();
            let (next, proof) = self.step_prover.prove_with_external_committed(
                &mut challenger,
                fresh_openings,
                fresh,
                &priors,
            );

            steps.push(WarpExternalRootStep {
                fresh_commitments,
                instance: next.instance.clone(),
                proof,
            });
            current = Some(next);
        }

        let final_acc = current.expect("non-empty step_fresh_committed checked above");
        let final_proof = finalizer.finalize(&final_acc.instance, &final_acc.witness)?;
        Ok((
            final_acc.instance,
            WarpExternalRootProof { steps, final_proof },
        ))
    }

    /// Prove a linear externally committed chain using an external
    /// accumulator commitment backend.
    ///
    /// Fresh inputs and every intermediate WARP accumulator are bound with
    /// their caller-provided backend layouts, so the root proof does not fall
    /// back to Plonky3 `ExtensionMmcs` for accumulated codewords.
    pub fn prove_external_linear_chain_accumulator<
        Challenger,
        Fresh,
        FreshOpenings,
        AccBackend,
        Fin,
    >(
        &self,
        base_challenger: &Challenger,
        fresh_openings: &FreshOpenings,
        acc_backend: &AccBackend,
        step_fresh_committed: Vec<Vec<Fresh>>,
        finalizer: &Fin,
    ) -> Result<
        (
            AccumulatorInstance<EF, AccBackend::Commitment>,
            WarpExternalRootProof<
                F,
                EF,
                AccBackend::Commitment,
                Fresh::Commitment,
                FreshOpenings::Proof,
                AccBackend::Proof,
                Fin::Proof,
            >,
        ),
        WarpError,
    >
    where
        Challenger: FieldChallenger<F> + GrindingChallenger<Witness = F> + Clone,
        Fresh: ExternalCommittedCodeword<F> + ExternalCommitmentObserver<F, Challenger>,
        FreshOpenings: ExternalCodewordOpeningProver<F, Fresh>,
        AccBackend: AccumulatorCommitmentBackend<F, EF, Challenger>,
        Fin: AccumulatorFinalizer<F, EF, AccBackend::Commitment, AccBackend::ProverData>,
    {
        if step_fresh_committed.is_empty() {
            return Err(WarpError::Config("root proof requires at least one step"));
        }

        let mut current: Option<Accumulator<EF, AccBackend::Commitment, AccBackend::ProverData>> =
            None;
        let mut steps = Vec::with_capacity(step_fresh_committed.len());

        for fresh in step_fresh_committed {
            let prior_count = usize::from(current.is_some());
            let total = fresh.len() + prior_count;
            if !(total >= 2 && total.is_power_of_two()) {
                return Err(WarpError::Config(
                    "each root step must have ell = ell_1 + ell_2 as a power of two",
                ));
            }

            let fresh_commitments = fresh.iter().map(Fresh::commitment).collect();
            let priors = match current.take() {
                Some(acc) => alloc::vec![acc],
                None => Vec::new(),
            };
            let mut challenger = base_challenger.clone();
            let (next, proof) = self.step_prover.prove_with_external_committed_accumulator(
                &mut challenger,
                fresh_openings,
                acc_backend,
                fresh,
                &priors,
            );

            steps.push(WarpExternalRootStep {
                fresh_commitments,
                instance: next.instance.clone(),
                proof,
            });
            current = Some(next);
        }

        let final_acc = current.expect("non-empty step_fresh_committed checked above");
        let final_proof = finalizer.finalize(&final_acc.instance, &final_acc.witness)?;
        Ok((
            final_acc.instance,
            WarpExternalRootProof { steps, final_proof },
        ))
    }

    /// Prove a linear externally committed chain using batched PCS openings
    /// for every WARP shift-query set.
    pub fn prove_external_linear_chain_accumulator_batched<
        Challenger,
        Fresh,
        FreshOpenings,
        AccBackend,
        Fin,
    >(
        &self,
        base_challenger: &Challenger,
        fresh_openings: &FreshOpenings,
        acc_backend: &AccBackend,
        step_fresh_committed: Vec<Vec<Fresh>>,
        finalizer: &Fin,
    ) -> Result<
        (
            AccumulatorInstance<EF, AccBackend::Commitment>,
            WarpExternalRootProofBatched<
                F,
                EF,
                AccBackend::Commitment,
                Fresh::Commitment,
                FreshOpenings::BatchProof,
                AccBackend::BatchProof,
                Fin::Proof,
            >,
        ),
        WarpError,
    >
    where
        Challenger: FieldChallenger<F> + GrindingChallenger<Witness = F> + Clone,
        Fresh: ExternalCommittedCodeword<F> + ExternalCommitmentObserver<F, Challenger>,
        FreshOpenings: ExternalCodewordBatchOpeningProver<F, Fresh>,
        AccBackend: AccumulatorBatchOpeningBackend<F, EF, Challenger>,
        Fin: AccumulatorFinalizer<F, EF, AccBackend::Commitment, AccBackend::ProverData>,
    {
        if step_fresh_committed.is_empty() {
            return Err(WarpError::Config("root proof requires at least one step"));
        }

        let mut current: Option<Accumulator<EF, AccBackend::Commitment, AccBackend::ProverData>> =
            None;
        let mut steps = Vec::with_capacity(step_fresh_committed.len());

        for fresh in step_fresh_committed {
            let prior_count = usize::from(current.is_some());
            let total = fresh.len() + prior_count;
            if !(total >= 2 && total.is_power_of_two()) {
                return Err(WarpError::Config(
                    "each root step must have ell = ell_1 + ell_2 as a power of two",
                ));
            }

            let fresh_commitments = fresh.iter().map(Fresh::commitment).collect();
            let priors = match current.take() {
                Some(acc) => alloc::vec![acc],
                None => Vec::new(),
            };
            let mut challenger = base_challenger.clone();
            let (next, proof) = self
                .step_prover
                .prove_with_external_committed_accumulator_batched(
                    &mut challenger,
                    fresh_openings,
                    acc_backend,
                    fresh,
                    priors,
                );

            steps.push(WarpExternalRootStepBatched {
                fresh_commitments,
                instance: next.instance.clone(),
                proof,
            });
            current = Some(next);
        }

        let final_acc = current.expect("non-empty step_fresh_committed checked above");
        let final_proof = finalizer.finalize(&final_acc.instance, &final_acc.witness)?;
        Ok((
            final_acc.instance,
            WarpExternalRootProofBatched { steps, final_proof },
        ))
    }
}

/// Verifier for a linear WARP root proof.
pub struct WarpRootVerifier<'a, F, EF, MT, Dft, Pesat>
where
    F: TwoAdicField,
    EF: ExtensionField<F> + TwoAdicField,
    MT: Mmcs<F>,
    Dft: TwoAdicSubgroupDft<F>,
    Pesat: BundledPesat<F, EF>,
{
    step_verifier: WarpVerifier<'a, F, EF, MT, Dft, Pesat>,
    _ph: PhantomData<(F, EF)>,
}

impl<'a, F, EF, MT, Dft, Pesat> WarpRootVerifier<'a, F, EF, MT, Dft, Pesat>
where
    F: TwoAdicField,
    EF: ExtensionField<F> + TwoAdicField,
    MT: Mmcs<F> + Sync,
    Dft: TwoAdicSubgroupDft<F>,
    Pesat: BundledPesat<F, EF>,
{
    /// Create a root verifier over the same components as ordinary WARP.
    pub fn new(
        mmcs: &'a MT,
        code: &'a ReedSolomonCode<F, Dft>,
        pesat: &'a Pesat,
        params: WarpParams,
    ) -> Self {
        Self {
            step_verifier: WarpVerifier::new(mmcs, code, pesat, params),
            _ph: PhantomData,
        }
    }

    /// Verify every WARP step in the chain and then verify the finalizer.
    ///
    /// Returns the final accumulator instance on success.
    pub fn verify_linear_chain<Challenger, Fin>(
        &self,
        base_challenger: &Challenger,
        proof: &WarpRootProof<F, EF, MT::Commitment, MT::Proof, Fin::Proof>,
        finalizer: &Fin,
    ) -> Result<AccumulatorInstance<EF, MT::Commitment>, WarpError>
    where
        Challenger: FieldChallenger<F>
            + GrindingChallenger<Witness = F>
            + CanObserve<MT::Commitment>
            + Clone,
        Fin: Finalizer<F, EF, MT, ExtProverData<F, EF, MT>>,
    {
        if proof.steps.is_empty() {
            return Err(WarpError::Config("root proof requires at least one step"));
        }

        let mut previous: Option<AccumulatorInstance<EF, MT::Commitment>> = None;
        for step in &proof.steps {
            let priors: &[AccumulatorInstance<EF, MT::Commitment>] = match previous.as_ref() {
                Some(instance) => core::slice::from_ref(instance),
                None => &[],
            };
            let mut challenger = base_challenger.clone();
            self.step_verifier.verify(
                &mut challenger,
                step.num_fresh,
                priors,
                &step.instance,
                &step.proof,
            )?;
            previous = Some(step.instance.clone());
        }

        let final_instance = previous.expect("non-empty proof.steps checked above");
        finalizer.verify(&final_instance, &proof.final_proof)?;
        Ok(final_instance)
    }

    /// Verify a local root proof and return the public claim receipt that an
    /// outer proof should expose.
    pub fn verify_linear_chain_with_receipt<Challenger, Fin>(
        &self,
        base_challenger: &Challenger,
        proof: &WarpRootProof<F, EF, MT::Commitment, MT::Proof, Fin::Proof>,
        finalizer: &Fin,
    ) -> Result<
        WarpRootReceipt<EF, MT::Commitment, <Challenger as CanFinalizeDigest>::Digest>,
        WarpError,
    >
    where
        Challenger: FieldChallenger<F>
            + GrindingChallenger<Witness = F>
            + CanObserve<MT::Commitment>
            + CanFinalizeDigest
            + Clone,
        MT::Commitment: Clone,
        Fin: Finalizer<F, EF, MT, ExtProverData<F, EF, MT>>,
    {
        let final_instance = self.verify_linear_chain(base_challenger, proof, finalizer)?;
        let claim = proof.claim();
        let claim_digest = claim.digest::<F, Challenger>(base_challenger.clone());
        Ok(WarpRootReceipt {
            claim,
            claim_digest,
            final_instance,
        })
    }

    /// Verify every externally committed WARP step in the chain and then
    /// verify the finalizer.
    ///
    /// This is the native verifier for the statement a future outer
    /// WHIR proof should arithmetize.
    pub fn verify_external_linear_chain<Challenger, FreshVerifier, Fin>(
        &self,
        base_challenger: &Challenger,
        fresh_verifier: &FreshVerifier,
        proof: &WarpExternalRootProof<
            F,
            EF,
            MT::Commitment,
            FreshVerifier::Commitment,
            FreshVerifier::Proof,
            MT::Proof,
            Fin::Proof,
        >,
        finalizer: &Fin,
    ) -> Result<AccumulatorInstance<EF, MT::Commitment>, WarpError>
    where
        Challenger: FieldChallenger<F>
            + GrindingChallenger<Witness = F>
            + CanObserve<MT::Commitment>
            + Clone,
        FreshVerifier: ExternalCodewordOpeningVerifier<F, Challenger>,
        Fin: Finalizer<F, EF, MT, ExtProverData<F, EF, MT>>,
    {
        if proof.steps.is_empty() {
            return Err(WarpError::Config("root proof requires at least one step"));
        }

        let mut previous: Option<AccumulatorInstance<EF, MT::Commitment>> = None;
        for step in &proof.steps {
            let priors: &[AccumulatorInstance<EF, MT::Commitment>] = match previous.as_ref() {
                Some(instance) => core::slice::from_ref(instance),
                None => &[],
            };
            let mut challenger = base_challenger.clone();
            self.step_verifier.verify_with_external_committed(
                &mut challenger,
                fresh_verifier,
                &step.fresh_commitments,
                priors,
                &step.instance,
                &step.proof,
            )?;
            previous = Some(step.instance.clone());
        }

        let final_instance = previous.expect("non-empty proof.steps checked above");
        finalizer.verify(&final_instance, &proof.final_proof)?;
        Ok(final_instance)
    }

    /// Verify an externally committed WARP chain whose accumulator
    /// commitments use a caller-provided external backend.
    pub fn verify_external_linear_chain_accumulator<Challenger, FreshVerifier, AccBackend, Fin>(
        &self,
        base_challenger: &Challenger,
        fresh_verifier: &FreshVerifier,
        acc_backend: &AccBackend,
        proof: &WarpExternalRootProof<
            F,
            EF,
            AccBackend::Commitment,
            FreshVerifier::Commitment,
            FreshVerifier::Proof,
            AccBackend::Proof,
            Fin::Proof,
        >,
        finalizer: &Fin,
    ) -> Result<AccumulatorInstance<EF, AccBackend::Commitment>, WarpError>
    where
        Challenger: FieldChallenger<F> + GrindingChallenger<Witness = F> + Clone,
        FreshVerifier: ExternalCodewordOpeningVerifier<F, Challenger>,
        AccBackend: AccumulatorCommitmentBackend<F, EF, Challenger>,
        Fin: AccumulatorFinalizer<F, EF, AccBackend::Commitment, AccBackend::ProverData>,
    {
        if proof.steps.is_empty() {
            return Err(WarpError::Config("root proof requires at least one step"));
        }

        let mut previous: Option<AccumulatorInstance<EF, AccBackend::Commitment>> = None;
        for step in &proof.steps {
            let priors: &[AccumulatorInstance<EF, AccBackend::Commitment>] = match previous.as_ref()
            {
                Some(instance) => core::slice::from_ref(instance),
                None => &[],
            };
            let mut challenger = base_challenger.clone();
            self.step_verifier
                .verify_with_external_committed_accumulator(
                    &mut challenger,
                    fresh_verifier,
                    acc_backend,
                    &step.fresh_commitments,
                    priors,
                    &step.instance,
                    &step.proof,
                )?;
            previous = Some(step.instance.clone());
        }

        let final_instance = previous.expect("non-empty proof.steps checked above");
        finalizer.verify(&final_instance, &proof.final_proof)?;
        Ok(final_instance)
    }

    /// Verify a batched-opening externally committed WARP chain.
    pub fn verify_external_linear_chain_accumulator_batched<
        Challenger,
        FreshVerifier,
        AccBackend,
        Fin,
    >(
        &self,
        base_challenger: &Challenger,
        fresh_verifier: &FreshVerifier,
        acc_backend: &AccBackend,
        proof: &WarpExternalRootProofBatched<
            F,
            EF,
            AccBackend::Commitment,
            FreshVerifier::Commitment,
            FreshVerifier::BatchProof,
            AccBackend::BatchProof,
            Fin::Proof,
        >,
        finalizer: &Fin,
    ) -> Result<AccumulatorInstance<EF, AccBackend::Commitment>, WarpError>
    where
        Challenger: FieldChallenger<F> + GrindingChallenger<Witness = F> + Clone,
        FreshVerifier: ExternalCodewordBatchOpeningVerifier<F, Challenger>,
        AccBackend: AccumulatorBatchOpeningBackend<F, EF, Challenger>,
        Fin: AccumulatorFinalizer<F, EF, AccBackend::Commitment, AccBackend::ProverData>,
    {
        if proof.steps.is_empty() {
            return Err(WarpError::Config("root proof requires at least one step"));
        }

        let mut previous: Option<AccumulatorInstance<EF, AccBackend::Commitment>> = None;
        for step in &proof.steps {
            let priors: &[AccumulatorInstance<EF, AccBackend::Commitment>] = match previous.as_ref()
            {
                Some(instance) => core::slice::from_ref(instance),
                None => &[],
            };
            let mut challenger = base_challenger.clone();
            self.step_verifier
                .verify_with_external_committed_accumulator_batched(
                    &mut challenger,
                    fresh_verifier,
                    acc_backend,
                    &step.fresh_commitments,
                    priors,
                    &step.instance,
                    &step.proof,
                )?;
            previous = Some(step.instance.clone());
        }

        let final_instance = previous.expect("non-empty proof.steps checked above");
        finalizer.verify(&final_instance, &proof.final_proof)?;
        Ok(final_instance)
    }

    /// Verify an externally committed root proof with an external
    /// accumulator backend and return the public claim receipt.
    pub fn verify_external_linear_chain_accumulator_with_receipt<
        Challenger,
        FreshVerifier,
        AccBackend,
        Fin,
    >(
        &self,
        base_challenger: &Challenger,
        fresh_verifier: &FreshVerifier,
        acc_backend: &AccBackend,
        proof: &WarpExternalRootProof<
            F,
            EF,
            AccBackend::Commitment,
            FreshVerifier::Commitment,
            FreshVerifier::Proof,
            AccBackend::Proof,
            Fin::Proof,
        >,
        finalizer: &Fin,
    ) -> Result<
        WarpExternalRootReceipt<
            EF,
            AccBackend::Commitment,
            FreshVerifier::Commitment,
            <Challenger as CanFinalizeDigest>::Digest,
        >,
        WarpError,
    >
    where
        Challenger:
            FieldChallenger<F> + GrindingChallenger<Witness = F> + CanFinalizeDigest + Clone,
        FreshVerifier: ExternalCodewordOpeningVerifier<F, Challenger>,
        FreshVerifier::Commitment: Clone,
        AccBackend: AccumulatorCommitmentBackend<F, EF, Challenger>,
        Fin: AccumulatorFinalizer<F, EF, AccBackend::Commitment, AccBackend::ProverData>,
    {
        let final_instance = self.verify_external_linear_chain_accumulator(
            base_challenger,
            fresh_verifier,
            acc_backend,
            proof,
            finalizer,
        )?;
        let claim = proof.claim();
        let claim_digest = claim.digest_accumulator::<F, Challenger, FreshVerifier, AccBackend>(
            base_challenger.clone(),
            fresh_verifier,
            acc_backend,
        );
        Ok(WarpExternalRootReceipt {
            claim,
            claim_digest,
            final_instance,
        })
    }

    /// Verify an externally committed root proof and return the public claim
    /// receipt that an outer WHIR proof should expose.
    pub fn verify_external_linear_chain_with_receipt<Challenger, FreshVerifier, Fin>(
        &self,
        base_challenger: &Challenger,
        fresh_verifier: &FreshVerifier,
        proof: &WarpExternalRootProof<
            F,
            EF,
            MT::Commitment,
            FreshVerifier::Commitment,
            FreshVerifier::Proof,
            MT::Proof,
            Fin::Proof,
        >,
        finalizer: &Fin,
    ) -> Result<
        WarpExternalRootReceipt<
            EF,
            MT::Commitment,
            FreshVerifier::Commitment,
            <Challenger as CanFinalizeDigest>::Digest,
        >,
        WarpError,
    >
    where
        Challenger: FieldChallenger<F>
            + GrindingChallenger<Witness = F>
            + CanObserve<MT::Commitment>
            + CanFinalizeDigest
            + Clone,
        FreshVerifier: ExternalCodewordOpeningVerifier<F, Challenger>,
        FreshVerifier::Commitment: Clone,
        MT::Commitment: Clone,
        Fin: Finalizer<F, EF, MT, ExtProverData<F, EF, MT>>,
    {
        let final_instance =
            self.verify_external_linear_chain(base_challenger, fresh_verifier, proof, finalizer)?;
        let claim = proof.claim();
        let claim_digest =
            claim.digest::<F, Challenger, FreshVerifier>(base_challenger.clone(), fresh_verifier);
        Ok(WarpExternalRootReceipt {
            claim,
            claim_digest,
            final_instance,
        })
    }
}
