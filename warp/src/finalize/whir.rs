//! WHIR-facing opening layer for a WARP accumulator.
//!
//! This module is the WHIR-facing boundary for replacing the current
//! FRI-backed root proof with a WHIR-native final proof. It provides a
//! precommitted opening proof for the RS/MLE accumulator opening
//!
//! ```text
//!     f_hat(alpha) = mu
//! ```
//!
//! and a sumcheck proof for the Boolean PESAT decider claim
//! `Pb(beta, C^{-1}(f)) = eta`, using Plonky3's
//! [`MultilinearPcs`](p3_commit::MultilinearPcs) abstraction. The important
//! soundness condition is enforced explicitly throughout: PCS openings are
//! checked against the accumulator's existing commitment `rt`. A WHIR wrapper
//! that opens a fresh unrelated commitment would be unsound as a WARP
//! finalizer.

use alloc::format;
use alloc::vec;
use alloc::vec::Vec;
use core::marker::PhantomData;

use p3_challenger::{CanObserve, FieldChallenger, GrindingChallenger};
use p3_commit::{MultilinearOpenedValues, MultilinearPcs};
use p3_dft::TwoAdicSubgroupDft;
use p3_field::{BasedVectorSpace, ExtensionField, Field, PrimeCharacteristicRing, TwoAdicField};
use p3_matrix::Matrix;
use p3_matrix::dense::RowMajorMatrix;
use p3_multilinear_util::point::Point;
use p3_multilinear_util::poly::Poly;
use p3_whir::pcs::proof::WhirProof;
use p3_whir::pcs::{WhirDeferredProverData, WhirPcs};
use serde::{Deserialize, Serialize};

use crate::accumulator::{AccumulatorInstance, AccumulatorWitness};
use crate::code::ReedSolomonCode;
use crate::error::{DeciderError, FinalizerError};
use crate::protocol::{
    AccumulatorBatchOpeningBackend, AccumulatorCommitmentBackend,
    ExternalCodewordBatchOpeningProver, ExternalCodewordBatchOpeningVerifier,
    ExternalCodewordOpeningProver, ExternalCodewordOpeningVerifier, ExternalCommitmentObserver,
    ExternalCommittedCodeword,
};
use crate::relation::{BooleanPesat, BundledPesat};
use crate::sumcheck::{SumcheckProof, observe_and_sample, verify_sumcheck};

/// PCS proof for the accumulator codeword opening `f_hat(alpha) = mu`.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound = "PcsProof: Serialize + serde::de::DeserializeOwned")]
pub struct WhirAccumulatorOpeningProof<PcsProof> {
    /// Opening proof produced by the underlying multilinear PCS.
    pub pcs_proof: PcsProof,
}

/// WHIR-facing proof of the final PESAT decider claim.
///
/// The final decider equation `Pb(beta, C^{-1}(f)) = eta` is reduced by a
/// sumcheck to one terminal witness claim. In systematic RS mode, `C^{-1}(f)`
/// is the message subspace of the committed codeword.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(
    bound = "EF: Serialize + serde::de::DeserializeOwned, PcsProof: Serialize + serde::de::DeserializeOwned"
)]
pub struct WhirPesatProof<EF, PcsProof> {
    /// Sumcheck over the PESAT witness hypercube.
    pub decider_sumcheck: SumcheckProof<EF>,
    /// Claimed terminal witness value at the sampled point.
    pub terminal_values: Vec<EF>,
    /// Reserved for legacy serialization compatibility; always empty on the
    /// direct Boolean path.
    pub next_terminal_values: Vec<EF>,
    /// Reserved for legacy serialization compatibility; always empty on the
    /// direct Boolean path.
    pub next_opened_row_values: Vec<EF>,
    /// PCS opening proof for terminal values on the systematic RS oracle.
    pub pcs_proof: PcsProof,
}

/// WHIR-facing final WARP proof.
///
/// This is the reusable assembly point for a WHIR-native WARP finalizer. The
/// two subproofs certify the two non-local decider equations against the same
/// accumulator commitment:
///
/// ```text
///     f_hat(alpha) = mu
///     Pb(beta, C^{-1}(f)) = eta
/// ```
///
/// Soundness depends on the `Pcs` commitment being the same public commitment
/// layout as the accumulator's `rt`. A PCS that commits to a fresh unrelated
/// oracle must not be used here.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(
    bound = "EF: Serialize + serde::de::DeserializeOwned, PcsProof: Serialize + serde::de::DeserializeOwned"
)]
pub struct WhirWarpFinalizerProof<EF, PcsProof> {
    /// Opening proof for `f_hat(alpha) = mu`.
    pub accumulator_opening: WhirAccumulatorOpeningProof<PcsProof>,
    /// PESAT decider proof for `Pb(beta, C^{-1}(f)) = eta`.
    pub pesat: WhirPesatProof<EF, PcsProof>,
}

/// Opening backend for arbitrary MLE points on an already committed accumulator.
///
/// This is the extra capability needed by the WHIR-backed finalizer to avoid
/// recommitting the final accumulator. The ordinary
/// [`AccumulatorCommitmentBackend`] trait only opens Boolean codeword indices for
/// WARP shift queries; the final decider also needs extension-point openings
/// such as `f_hat(alpha)` and terminal PESAT claims.
pub trait AccumulatorPointOpeningBackend<F, EF, Challenger>:
    AccumulatorCommitmentBackend<F, EF, Challenger>
where
    F: Field,
    EF: ExtensionField<F>,
    Challenger: FieldChallenger<F>,
{
    /// Proof for arbitrary MLE point openings.
    type PointProof: Clone + Serialize + serde::de::DeserializeOwned;
    /// Verification/proving error from the opening backend.
    type PointError: core::fmt::Debug;

    /// Number of variables in the committed accumulator codeword MLE.
    fn num_vars(&self) -> usize;

    /// Prove openings against an existing accumulator commitment/prover data.
    fn prove_points(
        &self,
        prover_data: &Self::ProverData,
        opening_points: &[Vec<Point<EF>>],
    ) -> Result<(MultilinearOpenedValues<EF>, Self::PointProof), Self::PointError>;

    /// Verify openings against an existing accumulator commitment.
    fn verify_points(
        &self,
        commitment: &Self::Commitment,
        opening_claims: &[Vec<(Point<EF>, EF)>],
        proof: &Self::PointProof,
    ) -> Result<(), Self::PointError>;
}

#[derive(Clone, Debug)]
pub struct PrecommittedAccumulatorPcs<'a, F, EF, Backend, Challenger>
where
    F: Field,
    EF: ExtensionField<F>,
    Challenger: FieldChallenger<F>,
    Backend: AccumulatorPointOpeningBackend<F, EF, Challenger>,
{
    backend: &'a Backend,
    commitment: Option<&'a Backend::Commitment>,
    prover_data: Option<&'a Backend::ProverData>,
    _ph: PhantomData<(F, EF, Challenger)>,
}

impl<'a, F, EF, Backend, Challenger> PrecommittedAccumulatorPcs<'a, F, EF, Backend, Challenger>
where
    F: Field,
    EF: ExtensionField<F>,
    Challenger: FieldChallenger<F>,
    Backend: AccumulatorPointOpeningBackend<F, EF, Challenger>,
{
    pub const fn prover(
        backend: &'a Backend,
        commitment: &'a Backend::Commitment,
        prover_data: &'a Backend::ProverData,
    ) -> Self {
        Self {
            backend,
            commitment: Some(commitment),
            prover_data: Some(prover_data),
            _ph: PhantomData,
        }
    }

    pub const fn verifier(backend: &'a Backend) -> Self {
        Self {
            backend,
            commitment: None,
            prover_data: None,
            _ph: PhantomData,
        }
    }
}

#[derive(Clone, Debug)]
pub struct PrecommittedAccumulatorPcsProverData<EF, Proof> {
    opened_values: MultilinearOpenedValues<EF>,
    proof: Proof,
}

impl<'a, F, EF, Backend, Challenger> MultilinearPcs<EF, Challenger>
    for PrecommittedAccumulatorPcs<'a, F, EF, Backend, Challenger>
where
    F: Field,
    EF: ExtensionField<F>,
    Challenger: FieldChallenger<F>,
    Backend: AccumulatorPointOpeningBackend<F, EF, Challenger>,
{
    type Val = EF;
    type Commitment = Backend::Commitment;
    type ProverData = PrecommittedAccumulatorPcsProverData<EF, Backend::PointProof>;
    type Proof = Backend::PointProof;
    type Error = Backend::PointError;

    fn num_vars(&self) -> usize {
        self.backend.num_vars()
    }

    fn commit(
        &self,
        evaluations: RowMajorMatrix<Self::Val>,
        opening_points: &[Vec<Point<EF>>],
        _challenger: &mut Challenger,
    ) -> (Self::Commitment, Self::ProverData) {
        assert_eq!(
            evaluations.width(),
            1,
            "precommitted accumulator finalizer opens one accumulator polynomial"
        );
        assert_eq!(
            evaluations.height(),
            1 << self.backend.num_vars(),
            "precommitted accumulator length must match backend variable count"
        );
        let commitment = self
            .commitment
            .expect("precommitted accumulator PCS prover missing commitment")
            .clone();
        let prover_data = self
            .prover_data
            .expect("precommitted accumulator PCS prover missing prover data");
        let (opened_values, proof) = self
            .backend
            .prove_points(prover_data, opening_points)
            .unwrap_or_else(|err| panic!("precommitted accumulator opening failed: {err:?}"));
        (
            commitment,
            PrecommittedAccumulatorPcsProverData {
                opened_values,
                proof,
            },
        )
    }

    fn open(
        &self,
        prover_data: Self::ProverData,
        _challenger: &mut Challenger,
    ) -> (MultilinearOpenedValues<EF>, Self::Proof) {
        (prover_data.opened_values, prover_data.proof)
    }

    fn verify(
        &self,
        commitment: &Self::Commitment,
        opening_claims: &[Vec<(Point<EF>, EF)>],
        proof: &Self::Proof,
        _challenger: &mut Challenger,
    ) -> Result<(), Self::Error> {
        self.backend
            .verify_points(commitment, opening_claims, proof)
    }
}

/// Adapter exposing a base-field multilinear PCS as an extension-field PCS.
///
/// WARP accumulator codewords live over `EF`, while WHIR's native PCS commits
/// base-field polynomials and evaluates them at extension-field points. This
/// adapter bridges that mismatch by decomposing every `EF` coefficient into
/// its fixed `F`-basis limbs, committing one limb polynomial per basis element,
/// and recomposing the opened limb evaluations in `EF`.
///
/// The proof carries the opened limb evaluations explicitly. That is necessary:
/// an evaluation of a base-field limb polynomial at an extension point is an
/// `EF` value, so the verifier cannot recover all limb evaluations from the
/// final recomposed `EF` claim alone.
#[derive(Clone, Debug)]
pub struct ExtensionLimbPcs<'a, F, EF, Inner>
where
    F: Field,
    EF: ExtensionField<F>,
{
    inner: &'a Inner,
    _ph: PhantomData<(F, EF)>,
}

impl<'a, F, EF, Inner> ExtensionLimbPcs<'a, F, EF, Inner>
where
    F: Field,
    EF: ExtensionField<F>,
{
    /// Wrap a base-field PCS whose challenge field is `EF`.
    pub const fn new(inner: &'a Inner) -> Self {
        Self {
            inner,
            _ph: PhantomData,
        }
    }
}

/// Prover data for [`ExtensionLimbPcs`].
pub struct ExtensionLimbPcsProverData<InnerData, Challenger> {
    limb_prover_data: Vec<InnerData>,
    limb_challengers: Vec<Challenger>,
}

/// Opening proof for [`ExtensionLimbPcs`].
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(
    bound = "EF: Serialize + serde::de::DeserializeOwned, InnerProof: Serialize + serde::de::DeserializeOwned"
)]
pub struct ExtensionLimbPcsProof<EF, InnerProof> {
    /// Opened values for each committed base-field limb polynomial.
    ///
    /// `limb_opened_values[i][p][j]` is the `EF` evaluation of limb `i`,
    /// polynomial `p`, at opening point `j`.
    pub limb_opened_values: Vec<MultilinearOpenedValues<EF>>,
    /// Inner PCS proofs, one per extension-field basis limb.
    pub limb_proofs: Vec<InnerProof>,
}

/// Verification failures for [`ExtensionLimbPcs`].
#[derive(Debug)]
pub enum ExtensionLimbPcsError<InnerError> {
    /// The commitment/proof did not contain one entry per extension basis limb.
    WrongLimbCount { expected: usize, actual: usize },
    /// A limb opening did not match the opening-claim shape.
    ShapeMismatch(&'static str),
    /// Limb openings did not recombine to the claimed extension-field values.
    RecompositionMismatch,
    /// The wrapped PCS rejected one limb opening.
    Inner { limb: usize, error: InnerError },
}

/// WHIR-backed fresh codeword committed outside a WARP step.
///
/// This is the Plonky3-native analogue of an upstream segment commitment:
/// the codeword is committed once with WHIR, then WARP's VACC step opens it at
/// sampled shift indices via WHIR proofs.
pub struct WhirCommittedCodeword<F, EF, MT, Challenger, const DIGEST_ELEMS: usize>
where
    F: Field,
    EF: ExtensionField<F>,
    MT: p3_commit::Mmcs<F>,
{
    commitment: MT::Commitment,
    prover_data: WhirDeferredProverData<F, EF, MT, DIGEST_ELEMS>,
    challenger: Challenger,
    codeword: Vec<F>,
    witness: Vec<F>,
}

/// WHIR-backed opening backend for fresh base-field WARP codewords.
#[derive(Debug)]
pub struct WhirCodewordBackend<'a, F, EF, MT, Challenger, Dft, const DIGEST_ELEMS: usize>
where
    F: Field,
    EF: ExtensionField<F>,
    MT: p3_commit::Mmcs<F>,
{
    pcs: &'a WhirPcs<EF, F, MT, Challenger, Dft, DIGEST_ELEMS>,
    challenger_seed: Challenger,
    _ph: PhantomData<(F, EF)>,
}

impl<'a, F, EF, MT, Challenger, Dft, const DIGEST_ELEMS: usize>
    WhirCodewordBackend<'a, F, EF, MT, Challenger, Dft, DIGEST_ELEMS>
where
    F: Field,
    EF: ExtensionField<F>,
    MT: p3_commit::Mmcs<F>,
{
    /// Create a WHIR backend for fresh base-field codewords.
    pub const fn new(
        pcs: &'a WhirPcs<EF, F, MT, Challenger, Dft, DIGEST_ELEMS>,
        challenger_seed: Challenger,
    ) -> Self {
        Self {
            pcs,
            challenger_seed,
            _ph: PhantomData,
        }
    }
}

impl<'a, F, EF, MT, Challenger, Dft, const DIGEST_ELEMS: usize>
    WhirCodewordBackend<'a, F, EF, MT, Challenger, Dft, DIGEST_ELEMS>
where
    F: TwoAdicField + Ord,
    EF: ExtensionField<F> + TwoAdicField,
    MT: p3_commit::Mmcs<F>,
    MT::Commitment: Clone + PartialEq,
    Challenger:
        FieldChallenger<F> + GrindingChallenger<Witness = F> + CanObserve<MT::Commitment> + Clone,
    Dft: TwoAdicSubgroupDft<F>,
    WhirDeferredProverData<F, EF, MT, DIGEST_ELEMS>: Clone,
{
    fn codeword_challenger(&self) -> Challenger {
        let mut challenger = self.challenger_seed.clone();
        challenger.observe(F::from_u64(WHIR_CODEWORD_BACKEND_TAG));
        challenger
    }

    /// Commit a fresh WARP input codeword with WHIR.
    pub fn commit_codeword(
        &self,
        codeword: Vec<F>,
        witness: Vec<F>,
    ) -> Result<WhirCommittedCodeword<F, EF, MT, Challenger, DIGEST_ELEMS>, &'static str> {
        if codeword.len() != 1 << self.pcs.num_vars() {
            return Err("fresh codeword length does not match WHIR variable count");
        }

        let mut challenger = self.codeword_challenger();
        let (commitment, prover_data) = self
            .pcs
            .commit_deferred(RowMajorMatrix::new(codeword.clone(), 1), &mut challenger);

        Ok(WhirCommittedCodeword {
            commitment,
            prover_data,
            challenger,
            codeword,
            witness,
        })
    }
}

impl<F, EF, MT, Challenger, const DIGEST_ELEMS: usize> ExternalCommittedCodeword<F>
    for WhirCommittedCodeword<F, EF, MT, Challenger, DIGEST_ELEMS>
where
    F: Field,
    EF: ExtensionField<F>,
    MT: p3_commit::Mmcs<F>,
    MT::Commitment: Clone + Serialize + serde::de::DeserializeOwned,
{
    type Commitment = MT::Commitment;

    fn commitment(&self) -> Self::Commitment {
        self.commitment.clone()
    }

    fn codeword(&self) -> &[F] {
        &self.codeword
    }

    fn witness(&self) -> &[F] {
        &self.witness
    }
}

impl<F, EF, MT, Challenger, const DIGEST_ELEMS: usize> ExternalCommitmentObserver<F, Challenger>
    for WhirCommittedCodeword<F, EF, MT, Challenger, DIGEST_ELEMS>
where
    F: Field,
    EF: ExtensionField<F>,
    MT: p3_commit::Mmcs<F>,
    MT::Commitment: Clone + Serialize + serde::de::DeserializeOwned,
    Challenger: FieldChallenger<F> + CanObserve<MT::Commitment>,
{
    fn observe_commitment(&self, challenger: &mut Challenger) {
        challenger.observe(self.commitment.clone());
    }
}

impl<'a, F, EF, MT, Challenger, Dft, const DIGEST_ELEMS: usize>
    ExternalCodewordOpeningProver<F, WhirCommittedCodeword<F, EF, MT, Challenger, DIGEST_ELEMS>>
    for WhirCodewordBackend<'a, F, EF, MT, Challenger, Dft, DIGEST_ELEMS>
where
    F: TwoAdicField + Ord,
    EF: ExtensionField<F> + TwoAdicField + Serialize + serde::de::DeserializeOwned,
    MT: p3_commit::Mmcs<F>,
    MT::Commitment: Clone + PartialEq + Serialize + serde::de::DeserializeOwned,
    WhirProof<F, EF, MT>: Clone + Serialize + serde::de::DeserializeOwned,
    WhirDeferredProverData<F, EF, MT, DIGEST_ELEMS>: Clone,
    Challenger:
        FieldChallenger<F> + GrindingChallenger<Witness = F> + CanObserve<MT::Commitment> + Clone,
    Dft: TwoAdicSubgroupDft<F>,
{
    type Proof = WhirProof<F, EF, MT>;
    type Error = p3_whir::pcs::verifier::errors::VerifierError;

    fn open(
        &self,
        committed: &WhirCommittedCodeword<F, EF, MT, Challenger, DIGEST_ELEMS>,
        index: usize,
    ) -> Result<(F, Self::Proof), Self::Error> {
        let num_vars = self.pcs.num_vars();
        let opening_points = [vec![Point::new(boolean_index_point(index, num_vars))]];
        let mut challenger = committed.challenger.clone();
        let (opened_values, proof) = self.pcs.open_deferred(
            committed.prover_data.clone(),
            &opening_points,
            &mut challenger,
        );
        let opened = opened_values
            .first()
            .and_then(|values| values.first())
            .copied()
            .expect("WHIR fresh opening returned no value");
        debug_assert_eq!(opened, EF::from(committed.codeword[index]));
        Ok((committed.codeword[index], proof))
    }
}

impl<'a, F, EF, MT, Challenger, Dft, const DIGEST_ELEMS: usize>
    ExternalCodewordBatchOpeningProver<
        F,
        WhirCommittedCodeword<F, EF, MT, Challenger, DIGEST_ELEMS>,
    > for WhirCodewordBackend<'a, F, EF, MT, Challenger, Dft, DIGEST_ELEMS>
where
    F: TwoAdicField + Ord,
    EF: ExtensionField<F> + TwoAdicField + Serialize + serde::de::DeserializeOwned,
    MT: p3_commit::Mmcs<F>,
    MT::Commitment: Clone + PartialEq + Serialize + serde::de::DeserializeOwned,
    WhirProof<F, EF, MT>: Clone + Serialize + serde::de::DeserializeOwned,
    WhirDeferredProverData<F, EF, MT, DIGEST_ELEMS>: Clone,
    Challenger:
        FieldChallenger<F> + GrindingChallenger<Witness = F> + CanObserve<MT::Commitment> + Clone,
    Dft: TwoAdicSubgroupDft<F>,
{
    type BatchProof = WhirProof<F, EF, MT>;

    fn open_batch(
        &self,
        committed: &WhirCommittedCodeword<F, EF, MT, Challenger, DIGEST_ELEMS>,
        indices: &[usize],
    ) -> Result<(Vec<F>, Self::BatchProof), Self::Error> {
        let num_vars = self.pcs.num_vars();
        let len = 1usize << num_vars;
        let points = indices
            .iter()
            .map(|&index| {
                assert!(index < len, "fresh WHIR opening index out of range");
                Point::new(boolean_index_point(index, num_vars))
            })
            .collect::<Vec<_>>();
        let opening_points = [points];
        let mut challenger = committed.challenger.clone();
        let (opened_values, proof) = self.pcs.open_deferred(
            committed.prover_data.clone(),
            &opening_points,
            &mut challenger,
        );
        let opened = opened_values
            .first()
            .expect("WHIR fresh opening returned no polynomial values");
        debug_assert_eq!(opened.len(), indices.len());
        let values = indices
            .iter()
            .zip(opened.iter())
            .map(|(&index, &opened)| {
                debug_assert_eq!(opened, EF::from(committed.codeword[index]));
                committed.codeword[index]
            })
            .collect();
        Ok((values, proof))
    }

    fn open_batch_owned(
        &self,
        committed: WhirCommittedCodeword<F, EF, MT, Challenger, DIGEST_ELEMS>,
        indices: &[usize],
    ) -> Result<(Vec<F>, Self::BatchProof), Self::Error> {
        let num_vars = self.pcs.num_vars();
        let len = 1usize << num_vars;
        let points = indices
            .iter()
            .map(|&index| {
                assert!(index < len, "fresh WHIR opening index out of range");
                Point::new(boolean_index_point(index, num_vars))
            })
            .collect::<Vec<_>>();
        let opening_points = [points];
        let mut challenger = committed.challenger;
        let (opened_values, proof) =
            self.pcs
                .open_deferred(committed.prover_data, &opening_points, &mut challenger);
        let opened = opened_values
            .first()
            .expect("WHIR fresh opening returned no polynomial values");
        debug_assert_eq!(opened.len(), indices.len());
        let values = indices
            .iter()
            .zip(opened.iter())
            .map(|(&index, &opened)| {
                debug_assert_eq!(opened, EF::from(committed.codeword[index]));
                committed.codeword[index]
            })
            .collect();
        Ok((values, proof))
    }
}

impl<'a, F, EF, MT, Challenger, Dft, const DIGEST_ELEMS: usize>
    ExternalCodewordOpeningVerifier<F, Challenger>
    for WhirCodewordBackend<'a, F, EF, MT, Challenger, Dft, DIGEST_ELEMS>
where
    F: TwoAdicField + Ord,
    EF: ExtensionField<F> + TwoAdicField,
    MT: p3_commit::Mmcs<F>,
    MT::Commitment: Clone + PartialEq + Serialize + serde::de::DeserializeOwned,
    WhirProof<F, EF, MT>: Clone + Serialize + serde::de::DeserializeOwned,
    WhirDeferredProverData<F, EF, MT, DIGEST_ELEMS>: Clone,
    Challenger:
        FieldChallenger<F> + GrindingChallenger<Witness = F> + CanObserve<MT::Commitment> + Clone,
    Dft: TwoAdicSubgroupDft<F>,
{
    type Commitment = MT::Commitment;
    type Proof = WhirProof<F, EF, MT>;
    type Error = p3_whir::pcs::verifier::errors::VerifierError;

    fn observe_commitment(&self, challenger: &mut Challenger, commitment: &Self::Commitment) {
        challenger.observe(commitment.clone());
    }

    fn verify_opening(
        &self,
        commitment: &Self::Commitment,
        log_codeword_len: usize,
        index: usize,
        value: F,
        proof: &Self::Proof,
    ) -> Result<(), Self::Error> {
        assert_eq!(
            log_codeword_len,
            self.pcs.num_vars(),
            "fresh codeword variable count mismatch"
        );
        let opening_claims = [vec![(
            Point::new(boolean_index_point(index, log_codeword_len)),
            EF::from(value),
        )]];
        let mut challenger = self.codeword_challenger();
        self.pcs
            .verify_deferred(commitment, &opening_claims, proof, &mut challenger)
    }
}

impl<'a, F, EF, MT, Challenger, Dft, const DIGEST_ELEMS: usize>
    ExternalCodewordBatchOpeningVerifier<F, Challenger>
    for WhirCodewordBackend<'a, F, EF, MT, Challenger, Dft, DIGEST_ELEMS>
where
    F: TwoAdicField + Ord,
    EF: ExtensionField<F> + TwoAdicField,
    MT: p3_commit::Mmcs<F>,
    MT::Commitment: Clone + PartialEq + Serialize + serde::de::DeserializeOwned,
    WhirProof<F, EF, MT>: Clone + Serialize + serde::de::DeserializeOwned,
    WhirDeferredProverData<F, EF, MT, DIGEST_ELEMS>: Clone,
    Challenger:
        FieldChallenger<F> + GrindingChallenger<Witness = F> + CanObserve<MT::Commitment> + Clone,
    Dft: TwoAdicSubgroupDft<F>,
{
    type BatchProof = WhirProof<F, EF, MT>;

    fn verify_batch_opening(
        &self,
        commitment: &Self::Commitment,
        log_codeword_len: usize,
        indices: &[usize],
        values: &[F],
        proof: &Self::BatchProof,
    ) -> Result<(), Self::Error> {
        assert_eq!(
            log_codeword_len,
            self.pcs.num_vars(),
            "fresh codeword variable count mismatch"
        );
        assert_eq!(indices.len(), values.len());
        let opening_claims = [indices
            .iter()
            .zip(values.iter())
            .map(|(&index, &value)| {
                (
                    Point::new(boolean_index_point(index, log_codeword_len)),
                    EF::from(value),
                )
            })
            .collect::<Vec<_>>()];
        let mut challenger = self.codeword_challenger();
        self.pcs
            .verify_deferred(commitment, &opening_claims, proof, &mut challenger)
    }
}

/// WHIR-backed commitment backend for WARP accumulator codewords.
///
/// The accumulator codeword lives over `EF`, while `p3-whir` commits one
/// base-field multilinear polynomial at a time. This backend decomposes the
/// accumulator into `EF` basis limbs, commits each limb once with
/// [`WhirPcs::commit_deferred`], and later proves VACC shift openings against
/// the saved WHIR prover data. The initial WHIR Merkle tree for a limb is not
/// rebuilt when a later WARP step asks for an opening.
#[derive(Debug)]
pub struct WhirLimbAccumulatorBackend<'a, F, EF, MT, Challenger, Dft, const DIGEST_ELEMS: usize>
where
    F: Field,
    EF: ExtensionField<F>,
    MT: p3_commit::Mmcs<F>,
{
    pcs: &'a WhirPcs<EF, F, MT, Challenger, Dft, DIGEST_ELEMS>,
    challenger_seed: Challenger,
    _ph: PhantomData<(F, EF)>,
}

impl<'a, F, EF, MT, Challenger, Dft, const DIGEST_ELEMS: usize>
    WhirLimbAccumulatorBackend<'a, F, EF, MT, Challenger, Dft, DIGEST_ELEMS>
where
    F: Field,
    EF: ExtensionField<F>,
    MT: p3_commit::Mmcs<F>,
{
    /// Create a WHIR-limb accumulator backend from a base-field WHIR PCS.
    ///
    /// `challenger_seed` is cloned and domain-separated per limb, so commit,
    /// open, and verify can reconstruct the same WHIR Fiat-Shamir transcript
    /// without borrowing WARP's step challenger after the step challenges have
    /// already been sampled.
    pub const fn new(
        pcs: &'a WhirPcs<EF, F, MT, Challenger, Dft, DIGEST_ELEMS>,
        challenger_seed: Challenger,
    ) -> Self {
        Self {
            pcs,
            challenger_seed,
            _ph: PhantomData,
        }
    }
}

/// Prover data for [`WhirLimbAccumulatorBackend`].
pub struct WhirLimbAccumulatorProverData<F, EF, MT, Challenger, const DIGEST_ELEMS: usize>
where
    F: Field,
    EF: ExtensionField<F>,
    MT: p3_commit::Mmcs<F>,
{
    limb_prover_data: Vec<WhirDeferredProverData<F, EF, MT, DIGEST_ELEMS>>,
    limb_challengers: Vec<Challenger>,
}

/// Opening proof for [`WhirLimbAccumulatorBackend`].
pub type WhirLimbAccumulatorOpeningProof<F, EF, MT> =
    ExtensionLimbPcsProof<EF, WhirProof<F, EF, MT>>;

impl<'a, F, EF, MT, Challenger, Dft, const DIGEST_ELEMS: usize>
    WhirLimbAccumulatorBackend<'a, F, EF, MT, Challenger, Dft, DIGEST_ELEMS>
where
    F: TwoAdicField + Ord,
    EF: ExtensionField<F> + TwoAdicField,
    MT: p3_commit::Mmcs<F>,
    MT::Commitment: Clone + PartialEq,
    Challenger: FieldChallenger<F>
        + GrindingChallenger<Witness = F>
        + CanObserve<F>
        + CanObserve<MT::Commitment>
        + Clone,
    Dft: TwoAdicSubgroupDft<F>,
{
    fn limb_challenger(&self, limb: usize) -> Challenger {
        extension_limb_challenger::<F, _>(&self.challenger_seed, limb)
    }

    /// Prove arbitrary multilinear openings against a previously committed
    /// accumulator codeword.
    pub fn prove_points(
        &self,
        prover_data: &WhirLimbAccumulatorProverData<F, EF, MT, Challenger, DIGEST_ELEMS>,
        opening_points: &[Vec<Point<EF>>],
    ) -> Result<
        (
            MultilinearOpenedValues<EF>,
            WhirLimbAccumulatorOpeningProof<F, EF, MT>,
        ),
        ExtensionLimbPcsError<p3_whir::pcs::verifier::errors::VerifierError>,
    >
    where
        WhirDeferredProverData<F, EF, MT, DIGEST_ELEMS>: Clone,
    {
        check_limb_count::<F, EF, p3_whir::pcs::verifier::errors::VerifierError>(
            prover_data.limb_prover_data.len(),
        )?;
        check_limb_count::<F, EF, p3_whir::pcs::verifier::errors::VerifierError>(
            prover_data.limb_challengers.len(),
        )?;

        let mut limb_opened_values = Vec::with_capacity(EF::DIMENSION);
        let mut limb_proofs = Vec::with_capacity(EF::DIMENSION);
        for limb in 0..EF::DIMENSION {
            let mut challenger = prover_data.limb_challengers[limb].clone();
            let (opened_values, proof) = self.pcs.open_deferred(
                prover_data.limb_prover_data[limb].clone(),
                opening_points,
                &mut challenger,
            );
            limb_opened_values.push(opened_values);
            limb_proofs.push(proof);
        }

        let opened_values = recompose_limb_opened_values::<F, EF>(&limb_opened_values).ok_or(
            ExtensionLimbPcsError::ShapeMismatch("limb opened values have inconsistent shape"),
        )?;

        Ok((
            opened_values,
            ExtensionLimbPcsProof {
                limb_opened_values,
                limb_proofs,
            },
        ))
    }

    /// Verify arbitrary multilinear openings against a WHIR-limb accumulator
    /// commitment.
    pub fn verify_points(
        &self,
        commitment: &[MT::Commitment],
        opening_claims: &[Vec<(Point<EF>, EF)>],
        proof: &WhirLimbAccumulatorOpeningProof<F, EF, MT>,
    ) -> Result<(), ExtensionLimbPcsError<p3_whir::pcs::verifier::errors::VerifierError>> {
        check_limb_count::<F, EF, p3_whir::pcs::verifier::errors::VerifierError>(commitment.len())?;
        check_limb_count::<F, EF, p3_whir::pcs::verifier::errors::VerifierError>(
            proof.limb_opened_values.len(),
        )?;
        check_limb_count::<F, EF, p3_whir::pcs::verifier::errors::VerifierError>(
            proof.limb_proofs.len(),
        )?;

        let recomposed = recompose_limb_opened_values::<F, EF>(&proof.limb_opened_values).ok_or(
            ExtensionLimbPcsError::ShapeMismatch("limb opened values have inconsistent shape"),
        )?;
        if !opening_claims_match_values(opening_claims, &recomposed) {
            return Err(ExtensionLimbPcsError::RecompositionMismatch);
        }

        for limb in 0..EF::DIMENSION {
            let limb_claims = limb_opening_claims(opening_claims, &proof.limb_opened_values[limb])?;
            let mut challenger = self.limb_challenger(limb);
            self.pcs
                .verify_deferred(
                    &commitment[limb],
                    &limb_claims,
                    &proof.limb_proofs[limb],
                    &mut challenger,
                )
                .map_err(|error| ExtensionLimbPcsError::Inner { limb, error })?;
        }

        Ok(())
    }
}

impl<'a, F, EF, MT, Challenger, Dft, const DIGEST_ELEMS: usize>
    AccumulatorCommitmentBackend<F, EF, Challenger>
    for WhirLimbAccumulatorBackend<'a, F, EF, MT, Challenger, Dft, DIGEST_ELEMS>
where
    F: TwoAdicField + Ord + Serialize + serde::de::DeserializeOwned,
    EF: ExtensionField<F> + TwoAdicField + Serialize + serde::de::DeserializeOwned,
    MT: p3_commit::Mmcs<F>,
    MT::Commitment: Clone + PartialEq + Serialize + serde::de::DeserializeOwned,
    WhirProof<F, EF, MT>: Clone + Serialize + serde::de::DeserializeOwned,
    WhirDeferredProverData<F, EF, MT, DIGEST_ELEMS>: Clone,
    Challenger: FieldChallenger<F>
        + GrindingChallenger<Witness = F>
        + CanObserve<F>
        + CanObserve<MT::Commitment>
        + Clone,
    Dft: TwoAdicSubgroupDft<F>,
{
    type Commitment = Vec<MT::Commitment>;
    type ProverData = WhirLimbAccumulatorProverData<F, EF, MT, Challenger, DIGEST_ELEMS>;
    type Proof = WhirLimbAccumulatorOpeningProof<F, EF, MT>;
    type Error = ExtensionLimbPcsError<p3_whir::pcs::verifier::errors::VerifierError>;

    fn commit(
        &self,
        codeword: Vec<EF>,
    ) -> Result<(Self::Commitment, Self::ProverData), Self::Error> {
        if codeword.len() != 1 << self.pcs.num_vars() {
            return Err(ExtensionLimbPcsError::ShapeMismatch(
                "accumulator codeword length does not match WHIR variable count",
            ));
        }

        let dimension = <EF as BasedVectorSpace<F>>::DIMENSION;
        let mut commitments = Vec::with_capacity(dimension);
        let mut limb_prover_data = Vec::with_capacity(dimension);
        let mut limb_challengers = Vec::with_capacity(dimension);

        for limb in 0..dimension {
            let limb_values = codeword
                .iter()
                .map(|value| <EF as BasedVectorSpace<F>>::as_basis_coefficients_slice(value)[limb])
                .collect::<Vec<_>>();
            let mut challenger = self.limb_challenger(limb);
            let (commitment, prover_data) = self
                .pcs
                .commit_deferred(RowMajorMatrix::new(limb_values, 1), &mut challenger);
            commitments.push(commitment);
            limb_prover_data.push(prover_data);
            limb_challengers.push(challenger);
        }

        Ok((
            commitments,
            WhirLimbAccumulatorProverData {
                limb_prover_data,
                limb_challengers,
            },
        ))
    }

    fn open(
        &self,
        prover_data: &Self::ProverData,
        index: usize,
    ) -> Result<(EF, Self::Proof), Self::Error> {
        let num_vars = self.pcs.num_vars();
        if index >= (1usize << num_vars) {
            return Err(ExtensionLimbPcsError::ShapeMismatch(
                "accumulator opening index out of range",
            ));
        }

        let opening_points = [vec![Point::new(boolean_index_point(index, num_vars))]];
        let (opened_values, proof) = self.prove_points(prover_data, &opening_points)?;
        let value = opened_values
            .first()
            .and_then(|values| values.first())
            .copied()
            .ok_or(ExtensionLimbPcsError::ShapeMismatch(
                "WHIR accumulator opening returned no value",
            ))?;
        Ok((value, proof))
    }

    fn observe_commitment(&self, challenger: &mut Challenger, commitment: &Self::Commitment) {
        for limb_commitment in commitment {
            challenger.observe(limb_commitment.clone());
        }
    }

    fn verify_opening(
        &self,
        commitment: &Self::Commitment,
        log_codeword_len: usize,
        index: usize,
        value: EF,
        proof: &Self::Proof,
    ) -> Result<(), Self::Error> {
        if log_codeword_len != self.pcs.num_vars() || index >= (1usize << log_codeword_len) {
            return Err(ExtensionLimbPcsError::ShapeMismatch(
                "accumulator opening shape does not match WHIR variable count",
            ));
        }

        let opening_claims = [vec![(
            Point::new(boolean_index_point(index, log_codeword_len)),
            value,
        )]];
        self.verify_points(commitment, &opening_claims, proof)
    }
}

impl<'a, F, EF, MT, Challenger, Dft, const DIGEST_ELEMS: usize>
    AccumulatorBatchOpeningBackend<F, EF, Challenger>
    for WhirLimbAccumulatorBackend<'a, F, EF, MT, Challenger, Dft, DIGEST_ELEMS>
where
    F: TwoAdicField + Ord + Serialize + serde::de::DeserializeOwned,
    EF: ExtensionField<F> + TwoAdicField + Serialize + serde::de::DeserializeOwned,
    MT: p3_commit::Mmcs<F>,
    MT::Commitment: Clone + PartialEq + Serialize + serde::de::DeserializeOwned,
    WhirProof<F, EF, MT>: Clone + Serialize + serde::de::DeserializeOwned,
    WhirDeferredProverData<F, EF, MT, DIGEST_ELEMS>: Clone,
    Challenger: FieldChallenger<F>
        + GrindingChallenger<Witness = F>
        + CanObserve<F>
        + CanObserve<MT::Commitment>
        + Clone,
    Dft: TwoAdicSubgroupDft<F>,
{
    type BatchProof = WhirLimbAccumulatorOpeningProof<F, EF, MT>;

    fn open_batch(
        &self,
        prover_data: &Self::ProverData,
        indices: &[usize],
    ) -> Result<(Vec<EF>, Self::BatchProof), Self::Error> {
        let num_vars = self.pcs.num_vars();
        let len = 1usize << num_vars;
        let points = indices
            .iter()
            .map(|&index| {
                if index >= len {
                    return Err(ExtensionLimbPcsError::ShapeMismatch(
                        "accumulator opening index out of range",
                    ));
                }
                Ok(Point::new(boolean_index_point(index, num_vars)))
            })
            .collect::<Result<Vec<_>, Self::Error>>()?;
        let opening_points = [points];
        let (opened_values, proof) = self.prove_points(prover_data, &opening_points)?;
        let values = opened_values
            .first()
            .ok_or(ExtensionLimbPcsError::ShapeMismatch(
                "WHIR accumulator opening returned no polynomial values",
            ))?
            .clone();
        if values.len() != indices.len() {
            return Err(ExtensionLimbPcsError::ShapeMismatch(
                "WHIR accumulator opening returned wrong number of values",
            ));
        }
        Ok((values, proof))
    }

    fn verify_batch_opening(
        &self,
        commitment: &Self::Commitment,
        log_codeword_len: usize,
        indices: &[usize],
        values: &[EF],
        proof: &Self::BatchProof,
    ) -> Result<(), Self::Error> {
        if log_codeword_len != self.pcs.num_vars() || indices.len() != values.len() {
            return Err(ExtensionLimbPcsError::ShapeMismatch(
                "accumulator opening shape does not match WHIR variable count",
            ));
        }
        let opening_claims = [indices
            .iter()
            .zip(values.iter())
            .map(|(&index, &value)| {
                if index >= (1usize << log_codeword_len) {
                    return Err(ExtensionLimbPcsError::ShapeMismatch(
                        "accumulator opening index out of range",
                    ));
                }
                Ok((
                    Point::new(boolean_index_point(index, log_codeword_len)),
                    value,
                ))
            })
            .collect::<Result<Vec<_>, Self::Error>>()?];
        self.verify_points(commitment, &opening_claims, proof)
    }
}

impl<'a, F, EF, MT, Challenger, Dft, const DIGEST_ELEMS: usize>
    AccumulatorPointOpeningBackend<F, EF, Challenger>
    for WhirLimbAccumulatorBackend<'a, F, EF, MT, Challenger, Dft, DIGEST_ELEMS>
where
    F: TwoAdicField + Ord,
    EF: ExtensionField<F> + TwoAdicField,
    MT: p3_commit::Mmcs<F>,
    MT::Commitment: PartialEq,
    Challenger:
        FieldChallenger<F> + GrindingChallenger<Witness = F> + CanObserve<MT::Commitment> + Clone,
    Dft: TwoAdicSubgroupDft<F>,
    WhirDeferredProverData<F, EF, MT, DIGEST_ELEMS>: Clone,
{
    type PointProof = WhirLimbAccumulatorOpeningProof<F, EF, MT>;
    type PointError = ExtensionLimbPcsError<p3_whir::pcs::verifier::errors::VerifierError>;

    fn num_vars(&self) -> usize {
        self.pcs.num_vars()
    }

    fn prove_points(
        &self,
        prover_data: &Self::ProverData,
        opening_points: &[Vec<Point<EF>>],
    ) -> Result<(MultilinearOpenedValues<EF>, Self::PointProof), Self::PointError> {
        WhirLimbAccumulatorBackend::prove_points(self, prover_data, opening_points)
    }

    fn verify_points(
        &self,
        commitment: &Self::Commitment,
        opening_claims: &[Vec<(Point<EF>, EF)>],
        proof: &Self::PointProof,
    ) -> Result<(), Self::PointError> {
        WhirLimbAccumulatorBackend::verify_points(self, commitment, opening_claims, proof)
    }
}

impl<'a, F, EF, Inner, Challenger> MultilinearPcs<EF, Challenger>
    for ExtensionLimbPcs<'a, F, EF, Inner>
where
    F: Field,
    EF: ExtensionField<F>,
    Inner: MultilinearPcs<EF, Challenger, Val = F>,
    Challenger: Clone + CanObserve<F>,
{
    type Val = EF;
    type Commitment = Vec<Inner::Commitment>;
    type ProverData = ExtensionLimbPcsProverData<Inner::ProverData, Challenger>;
    type Proof = ExtensionLimbPcsProof<EF, Inner::Proof>;
    type Error = ExtensionLimbPcsError<Inner::Error>;

    fn num_vars(&self) -> usize {
        self.inner.num_vars()
    }

    fn commit(
        &self,
        evaluations: RowMajorMatrix<Self::Val>,
        opening_points: &[Vec<Point<EF>>],
        challenger: &mut Challenger,
    ) -> (Self::Commitment, Self::ProverData) {
        let width = evaluations.width();
        let height = evaluations.height();
        let dimension = <EF as BasedVectorSpace<F>>::DIMENSION;
        let mut commitments = Vec::with_capacity(dimension);
        let mut limb_prover_data = Vec::with_capacity(dimension);
        let mut limb_challengers = Vec::with_capacity(dimension);

        for limb in 0..dimension {
            let mut limb_challenger = extension_limb_challenger::<F, _>(challenger, limb);
            let limb_values = evaluations
                .values
                .iter()
                .map(|value| <EF as BasedVectorSpace<F>>::as_basis_coefficients_slice(value)[limb])
                .collect::<Vec<_>>();
            let limb_matrix = RowMajorMatrix::new(limb_values, width);
            debug_assert_eq!(limb_matrix.height(), height);
            let (commitment, prover_data) =
                self.inner
                    .commit(limb_matrix, opening_points, &mut limb_challenger);
            commitments.push(commitment);
            limb_prover_data.push(prover_data);
            limb_challengers.push(limb_challenger);
        }

        (
            commitments,
            ExtensionLimbPcsProverData {
                limb_prover_data,
                limb_challengers,
            },
        )
    }

    fn open(
        &self,
        prover_data: Self::ProverData,
        _challenger: &mut Challenger,
    ) -> (MultilinearOpenedValues<EF>, Self::Proof) {
        let mut limb_opened_values = Vec::with_capacity(EF::DIMENSION);
        let mut limb_proofs = Vec::with_capacity(EF::DIMENSION);

        for (limb_data, mut limb_challenger) in prover_data
            .limb_prover_data
            .into_iter()
            .zip(prover_data.limb_challengers)
        {
            let (opened_values, proof) = self.inner.open(limb_data, &mut limb_challenger);
            limb_opened_values.push(opened_values);
            limb_proofs.push(proof);
        }

        let opened_values = recompose_limb_opened_values::<F, EF>(&limb_opened_values)
            .expect("inner PCS returned inconsistent limb opening shapes");
        (
            opened_values,
            ExtensionLimbPcsProof {
                limb_opened_values,
                limb_proofs,
            },
        )
    }

    fn verify(
        &self,
        commitment: &Self::Commitment,
        opening_claims: &[Vec<(Point<EF>, EF)>],
        proof: &Self::Proof,
        challenger: &mut Challenger,
    ) -> Result<(), Self::Error> {
        check_limb_count::<F, EF, Inner::Error>(commitment.len())?;
        check_limb_count::<F, EF, Inner::Error>(proof.limb_opened_values.len())?;
        check_limb_count::<F, EF, Inner::Error>(proof.limb_proofs.len())?;

        let recomposed = recompose_limb_opened_values::<F, EF>(&proof.limb_opened_values).ok_or(
            ExtensionLimbPcsError::ShapeMismatch("limb opened values have inconsistent shape"),
        )?;
        if !opening_claims_match_values(opening_claims, &recomposed) {
            return Err(ExtensionLimbPcsError::RecompositionMismatch);
        }

        for limb in 0..EF::DIMENSION {
            let limb_claims = limb_opening_claims(opening_claims, &proof.limb_opened_values[limb])?;
            let mut limb_challenger = extension_limb_challenger::<F, _>(challenger, limb);
            self.inner
                .verify(
                    &commitment[limb],
                    &limb_claims,
                    &proof.limb_proofs[limb],
                    &mut limb_challenger,
                )
                .map_err(|error| ExtensionLimbPcsError::Inner { limb, error })?;
        }

        Ok(())
    }
}

/// WHIR-native decider proof for the direct Boolean PESAT relation.
pub struct WhirBooleanPesatProtocol<'a, F, EF, Pcs, Challenger, Dft>
where
    F: TwoAdicField,
    EF: ExtensionField<F>,
    Dft: TwoAdicSubgroupDft<F>,
{
    pcs: &'a Pcs,
    code: &'a ReedSolomonCode<F, Dft>,
    pesat: &'a BooleanPesat<F, EF>,
    challenger_seed: Challenger,
}

impl<'a, F, EF, Pcs, Challenger, Dft> WhirBooleanPesatProtocol<'a, F, EF, Pcs, Challenger, Dft>
where
    F: TwoAdicField,
    EF: ExtensionField<F>,
    Dft: TwoAdicSubgroupDft<F>,
{
    /// Create a Boolean PESAT finalizer from a compatible PCS.
    pub const fn new(
        pcs: &'a Pcs,
        code: &'a ReedSolomonCode<F, Dft>,
        pesat: &'a BooleanPesat<F, EF>,
        challenger_seed: Challenger,
    ) -> Self {
        Self {
            pcs,
            code,
            pesat,
            challenger_seed,
        }
    }
}

impl<'a, F, EF, Pcs, Challenger, Dft> WhirBooleanPesatProtocol<'a, F, EF, Pcs, Challenger, Dft>
where
    F: TwoAdicField,
    EF: ExtensionField<F> + TwoAdicField,
    Pcs: MultilinearPcs<EF, Challenger, Val = EF>,
    Pcs::Commitment: Clone + PartialEq,
    Challenger:
        FieldChallenger<F> + GrindingChallenger<Witness = F> + CanObserve<Pcs::Commitment> + Clone,
    Dft: TwoAdicSubgroupDft<F>,
{
    /// Prove `sum_i eq(beta, i) * w_i * (w_i - 1) = eta`.
    pub fn prove<ProverData>(
        &self,
        instance: &AccumulatorInstance<EF, Pcs::Commitment>,
        witness: &AccumulatorWitness<EF, ProverData>,
    ) -> Result<WhirPesatProof<EF, Pcs::Proof>, FinalizerError> {
        self.validate_pesat_shape(instance)?;
        if witness.f.len() != self.code.codeword_len() {
            return Err(FinalizerError::Decider(DeciderError::EncodingMismatch));
        }
        let committed_witness = self.systematic_witness_from_codeword(&witness.f)?;

        let mut challenger = self.challenger_seed.clone();
        self.observe_statement(instance, &mut challenger);
        let (sumcheck, point, final_claim, terminal_value) =
            self.prove_boolean_sumcheck(instance, &committed_witness, &mut challenger);
        self.check_terminal_claim(instance, point.as_slice(), terminal_value, final_claim)?;

        challenger.observe_algebra_element(terminal_value);
        let opening_points = [vec![self.opening_point(point.as_slice())]];
        let evaluations = RowMajorMatrix::new(witness.f.clone(), 1);
        let (commitment, prover_data) =
            self.pcs
                .commit(evaluations, &opening_points, &mut challenger);
        if commitment != instance.rt {
            return Err(FinalizerError::Decider(DeciderError::MerkleRoot));
        }
        let (opened_values, pcs_proof) = self.pcs.open(prover_data, &mut challenger);
        let opened_values = opened_values.first().ok_or(FinalizerError::Unsupported(
            "PCS did not return terminal opening",
        ))?;
        if opened_values != &vec![terminal_value] {
            return Err(FinalizerError::Decider(DeciderError::EncodingMismatch));
        }

        Ok(WhirPesatProof {
            decider_sumcheck: sumcheck,
            terminal_values: vec![terminal_value],
            next_terminal_values: Vec::new(),
            next_opened_row_values: Vec::new(),
            pcs_proof,
        })
    }

    /// Verify the direct Boolean PESAT decider proof.
    pub fn verify(
        &self,
        instance: &AccumulatorInstance<EF, Pcs::Commitment>,
        proof: &WhirPesatProof<EF, Pcs::Proof>,
    ) -> Result<(), FinalizerError> {
        self.validate_pesat_shape(instance)?;
        if proof.terminal_values.len() != 1
            || !proof.next_terminal_values.is_empty()
            || !proof.next_opened_row_values.is_empty()
        {
            return Err(FinalizerError::Decider(DeciderError::EncodingMismatch));
        }

        let mut challenger = self.challenger_seed.clone();
        self.observe_statement(instance, &mut challenger);
        let (point, final_claim) = verify_sumcheck::<F, EF, _>(
            &proof.decider_sumcheck,
            &mut challenger,
            instance.eta,
            instance.beta.len(),
            3,
            "whir-boolean-pesat",
        )?;
        let terminal_value = proof.terminal_values[0];
        self.check_terminal_claim(instance, point.as_slice(), terminal_value, final_claim)?;

        challenger.observe_algebra_element(terminal_value);
        let opening_claims = [vec![(self.opening_point(point.as_slice()), terminal_value)]];
        self.pcs
            .verify(
                &instance.rt,
                &opening_claims,
                &proof.pcs_proof,
                &mut challenger,
            )
            .map_err(|err| FinalizerError::OpeningProof(format!("{err:?}")))
    }

    fn validate_pesat_shape(
        &self,
        instance: &AccumulatorInstance<EF, Pcs::Commitment>,
    ) -> Result<(), FinalizerError> {
        if !self.code.is_systematic() {
            return Err(FinalizerError::Unsupported(
                "WHIR-native Boolean PESAT proof requires systematic RS encoding",
            ));
        }
        if self.pcs.num_vars() != self.code.log_codeword_len() {
            return Err(FinalizerError::Unsupported(
                "PCS variable count must match the WARP codeword MLE dimension",
            ));
        }
        if self.pesat.shape().witness_len() != self.code.msg_len() {
            return Err(FinalizerError::Decider(DeciderError::EncodingMismatch));
        }
        if instance.beta.len() != self.pesat.shape().log_constraints {
            return Err(FinalizerError::Decider(DeciderError::BundledPesat));
        }
        Ok(())
    }

    fn observe_statement(
        &self,
        instance: &AccumulatorInstance<EF, Pcs::Commitment>,
        challenger: &mut Challenger,
    ) {
        challenger.observe(instance.rt.clone());
        challenger.observe_algebra_slice(&instance.beta);
        challenger.observe_algebra_element(instance.eta);
    }

    fn prove_boolean_sumcheck(
        &self,
        instance: &AccumulatorInstance<EF, Pcs::Commitment>,
        witness: &[EF],
        challenger: &mut Challenger,
    ) -> (SumcheckProof<EF>, Point<EF>, EF, EF) {
        let rounds = instance.beta.len();
        let mut proof = SumcheckProof::<EF>::new();
        let mut prefix = Vec::with_capacity(rounds);
        let mut prefix_eq = EF::ONE;
        let mut folded = witness.to_vec();
        let mut claim = instance.eta;

        for round in 0..rounds {
            let suffix_bits = rounds - round - 1;
            let half_len = 1usize << suffix_bits;
            let suffix_eq = Poly::<EF>::new_from_point(&instance.beta[round + 1..], prefix_eq);
            let beta = instance.beta[round];
            let eq_const = EF::ONE - beta;
            let eq_linear = beta + beta - EF::ONE;
            let mut coeffs = vec![EF::ZERO; 4];

            for suffix in 0..half_len {
                let lo = folded[suffix];
                let hi = folded[suffix + half_len];
                let diff = hi - lo;
                let q0 = lo * (lo - EF::ONE);
                let q1 = diff * (lo + lo - EF::ONE);
                let q2 = diff * diff;
                let weight = suffix_eq.as_slice()[suffix];
                coeffs[0] += weight * eq_const * q0;
                coeffs[1] += weight * (eq_const * q1 + eq_linear * q0);
                coeffs[2] += weight * (eq_const * q2 + eq_linear * q1);
                coeffs[3] += weight * eq_linear * q2;
            }

            debug_assert_eq!(coeffs[0] + coeffs.iter().copied().sum::<EF>(), claim);
            let challenge = observe_and_sample::<F, EF, _>(&mut proof, challenger, coeffs.clone());
            claim = coeffs
                .iter()
                .rev()
                .fold(EF::ZERO, |acc, &c| acc * challenge + c);
            prefix.push(challenge);
            prefix_eq *= beta * challenge + (EF::ONE - beta) * (EF::ONE - challenge);

            for suffix in 0..half_len {
                let lo = folded[suffix];
                let hi = folded[suffix + half_len];
                folded[suffix] = lo + challenge * (hi - lo);
            }
            folded.truncate(half_len);
        }

        debug_assert_eq!(folded.len(), 1);
        (proof, Point::new(prefix), claim, folded[0])
    }

    fn check_terminal_claim(
        &self,
        instance: &AccumulatorInstance<EF, Pcs::Commitment>,
        point: &[EF],
        terminal_value: EF,
        final_claim: EF,
    ) -> Result<(), FinalizerError> {
        let expected =
            eval_eq_ext(&instance.beta, point) * terminal_value * (terminal_value - EF::ONE);
        if expected != final_claim {
            return Err(FinalizerError::Decider(DeciderError::BundledPesat));
        }
        Ok(())
    }

    fn systematic_witness_from_codeword(&self, codeword: &[EF]) -> Result<Vec<EF>, FinalizerError> {
        if codeword.len() != self.code.codeword_len() {
            return Err(FinalizerError::Decider(DeciderError::EncodingMismatch));
        }
        Ok((0..self.code.msg_len())
            .map(|index| codeword[self.code.systematic_codeword_index(index)])
            .collect())
    }

    fn opening_point(&self, point: &[EF]) -> Point<EF> {
        self.code.systematic_message_point(point)
    }
}

/// Composed WHIR-native finalizer for direct Boolean PESAT.
pub struct WhirBooleanWarpFinalizerProtocol<'a, F, EF, Pcs, Challenger, Dft>
where
    F: TwoAdicField + PrimeCharacteristicRing,
    EF: ExtensionField<F>,
    Dft: TwoAdicSubgroupDft<F>,
{
    pcs: &'a Pcs,
    code: &'a ReedSolomonCode<F, Dft>,
    pesat: &'a BooleanPesat<F, EF>,
    challenger_seed: Challenger,
}

impl<'a, F, EF, Pcs, Challenger, Dft>
    WhirBooleanWarpFinalizerProtocol<'a, F, EF, Pcs, Challenger, Dft>
where
    F: TwoAdicField + PrimeCharacteristicRing,
    EF: ExtensionField<F>,
    Dft: TwoAdicSubgroupDft<F>,
{
    /// Create a composed Boolean WHIR finalizer from a compatible PCS.
    pub const fn new(
        pcs: &'a Pcs,
        code: &'a ReedSolomonCode<F, Dft>,
        pesat: &'a BooleanPesat<F, EF>,
        challenger_seed: Challenger,
    ) -> Self {
        Self {
            pcs,
            code,
            pesat,
            challenger_seed,
        }
    }
}

impl<'a, F, EF, Pcs, Challenger, Dft>
    WhirBooleanWarpFinalizerProtocol<'a, F, EF, Pcs, Challenger, Dft>
where
    F: TwoAdicField + PrimeCharacteristicRing,
    EF: ExtensionField<F> + TwoAdicField,
    Pcs: MultilinearPcs<EF, Challenger, Val = EF>,
    Pcs::Commitment: Clone + PartialEq,
    Challenger:
        FieldChallenger<F> + GrindingChallenger<Witness = F> + CanObserve<Pcs::Commitment> + Clone,
    Dft: TwoAdicSubgroupDft<F>,
{
    /// Prove the final accumulator opening and direct Boolean PESAT claim.
    pub fn prove<ProverData>(
        &self,
        instance: &AccumulatorInstance<EF, Pcs::Commitment>,
        witness: &AccumulatorWitness<EF, ProverData>,
    ) -> Result<WhirWarpFinalizerProof<EF, Pcs::Proof>, FinalizerError> {
        let opening_protocol = WhirAccumulatorOpeningProtocol::<F, EF, Pcs, Challenger, Dft>::new(
            self.pcs,
            self.code,
            self.tagged_challenger(WHIR_WARP_OPENING_TAG),
        );
        let boolean_protocol = WhirBooleanPesatProtocol::<F, EF, Pcs, Challenger, Dft>::new(
            self.pcs,
            self.code,
            self.pesat,
            self.tagged_challenger(WHIR_WARP_PESAT_TAG),
        );

        let accumulator_opening = opening_protocol.prove(instance, witness)?;
        let pesat = boolean_protocol.prove(instance, witness)?;

        Ok(WhirWarpFinalizerProof {
            accumulator_opening,
            pesat,
        })
    }

    /// Verify both final WARP decider subclaims against the same commitment.
    pub fn verify(
        &self,
        instance: &AccumulatorInstance<EF, Pcs::Commitment>,
        proof: &WhirWarpFinalizerProof<EF, Pcs::Proof>,
    ) -> Result<(), FinalizerError> {
        let opening_protocol = WhirAccumulatorOpeningProtocol::<F, EF, Pcs, Challenger, Dft>::new(
            self.pcs,
            self.code,
            self.tagged_challenger(WHIR_WARP_OPENING_TAG),
        );
        let boolean_protocol = WhirBooleanPesatProtocol::<F, EF, Pcs, Challenger, Dft>::new(
            self.pcs,
            self.code,
            self.pesat,
            self.tagged_challenger(WHIR_WARP_PESAT_TAG),
        );

        opening_protocol.verify(instance, &proof.accumulator_opening)?;
        boolean_protocol.verify(instance, &proof.pesat)
    }

    fn tagged_challenger(&self, tag: u64) -> Challenger {
        let mut challenger = self.challenger_seed.clone();
        challenger.observe(F::from_u64(tag));
        challenger
    }
}

impl<'a, F, EF, Pcs, Challenger, Dft, ProverData>
    crate::finalize::AccumulatorFinalizer<F, EF, Pcs::Commitment, ProverData>
    for WhirBooleanWarpFinalizerProtocol<'a, F, EF, Pcs, Challenger, Dft>
where
    F: TwoAdicField + PrimeCharacteristicRing,
    EF: ExtensionField<F> + TwoAdicField,
    Pcs: MultilinearPcs<EF, Challenger, Val = EF>,
    Pcs::Commitment: Clone + PartialEq,
    Challenger:
        FieldChallenger<F> + GrindingChallenger<Witness = F> + CanObserve<Pcs::Commitment> + Clone,
    Dft: TwoAdicSubgroupDft<F>,
{
    type Proof = WhirWarpFinalizerProof<EF, Pcs::Proof>;

    fn finalize(
        &self,
        instance: &AccumulatorInstance<EF, Pcs::Commitment>,
        witness: &AccumulatorWitness<EF, ProverData>,
    ) -> Result<Self::Proof, FinalizerError> {
        self.prove(instance, witness)
    }

    fn verify(
        &self,
        instance: &AccumulatorInstance<EF, Pcs::Commitment>,
        proof: &Self::Proof,
    ) -> Result<(), FinalizerError> {
        WhirBooleanWarpFinalizerProtocol::verify(self, instance, proof)
    }
}

/// Precommitted Boolean finalizer over an accumulator opening backend.
pub struct WhirPrecommittedBooleanWarpFinalizerProtocol<'a, F, EF, Backend, Challenger, Dft>
where
    F: TwoAdicField + PrimeCharacteristicRing,
    EF: ExtensionField<F>,
    Challenger: FieldChallenger<F>,
    Backend: AccumulatorPointOpeningBackend<F, EF, Challenger>,
    Dft: TwoAdicSubgroupDft<F>,
{
    backend: &'a Backend,
    code: &'a ReedSolomonCode<F, Dft>,
    pesat: &'a BooleanPesat<F, EF>,
    challenger_seed: Challenger,
}

impl<'a, F, EF, Backend, Challenger, Dft>
    WhirPrecommittedBooleanWarpFinalizerProtocol<'a, F, EF, Backend, Challenger, Dft>
where
    F: TwoAdicField + PrimeCharacteristicRing,
    EF: ExtensionField<F>,
    Challenger: FieldChallenger<F>,
    Backend: AccumulatorPointOpeningBackend<F, EF, Challenger>,
    Dft: TwoAdicSubgroupDft<F>,
{
    /// Create a finalizer over an already committed accumulator.
    pub const fn new(
        backend: &'a Backend,
        code: &'a ReedSolomonCode<F, Dft>,
        pesat: &'a BooleanPesat<F, EF>,
        challenger_seed: Challenger,
    ) -> Self {
        Self {
            backend,
            code,
            pesat,
            challenger_seed,
        }
    }
}

impl<'a, F, EF, Backend, Challenger, Dft>
    WhirPrecommittedBooleanWarpFinalizerProtocol<'a, F, EF, Backend, Challenger, Dft>
where
    F: TwoAdicField + PrimeCharacteristicRing,
    EF: ExtensionField<F> + TwoAdicField,
    Backend: AccumulatorPointOpeningBackend<F, EF, Challenger>,
    Backend::Commitment: Clone + PartialEq,
    Challenger: FieldChallenger<F>
        + GrindingChallenger<Witness = F>
        + CanObserve<Backend::Commitment>
        + Clone,
    Dft: TwoAdicSubgroupDft<F>,
{
    /// Prove both final decider claims against `instance.rt`.
    pub fn prove(
        &self,
        instance: &AccumulatorInstance<EF, Backend::Commitment>,
        witness: &AccumulatorWitness<EF, Backend::ProverData>,
    ) -> Result<WhirWarpFinalizerProof<EF, Backend::PointProof>, FinalizerError> {
        let pcs = PrecommittedAccumulatorPcs::<F, EF, Backend, Challenger>::prover(
            self.backend,
            &instance.rt,
            &witness.td,
        );
        let finalizer = WhirBooleanWarpFinalizerProtocol::<F, EF, _, Challenger, Dft>::new(
            &pcs,
            self.code,
            self.pesat,
            self.challenger_seed.clone(),
        );
        finalizer.prove(instance, witness)
    }

    /// Verify both final decider claims against `instance.rt`.
    pub fn verify(
        &self,
        instance: &AccumulatorInstance<EF, Backend::Commitment>,
        proof: &WhirWarpFinalizerProof<EF, Backend::PointProof>,
    ) -> Result<(), FinalizerError> {
        let pcs = PrecommittedAccumulatorPcs::<F, EF, Backend, Challenger>::verifier(self.backend);
        let finalizer = WhirBooleanWarpFinalizerProtocol::<F, EF, _, Challenger, Dft>::new(
            &pcs,
            self.code,
            self.pesat,
            self.challenger_seed.clone(),
        );
        finalizer.verify(instance, proof)
    }
}

impl<'a, F, EF, Backend, Challenger, Dft>
    crate::finalize::AccumulatorFinalizer<F, EF, Backend::Commitment, Backend::ProverData>
    for WhirPrecommittedBooleanWarpFinalizerProtocol<'a, F, EF, Backend, Challenger, Dft>
where
    F: TwoAdicField + PrimeCharacteristicRing,
    EF: ExtensionField<F> + TwoAdicField,
    Backend: AccumulatorPointOpeningBackend<F, EF, Challenger>,
    Backend::Commitment: Clone + PartialEq,
    Challenger: FieldChallenger<F>
        + GrindingChallenger<Witness = F>
        + CanObserve<Backend::Commitment>
        + Clone,
    Dft: TwoAdicSubgroupDft<F>,
{
    type Proof = WhirWarpFinalizerProof<EF, Backend::PointProof>;

    fn finalize(
        &self,
        instance: &AccumulatorInstance<EF, Backend::Commitment>,
        witness: &AccumulatorWitness<EF, Backend::ProverData>,
    ) -> Result<Self::Proof, FinalizerError> {
        self.prove(instance, witness)
    }

    fn verify(
        &self,
        instance: &AccumulatorInstance<EF, Backend::Commitment>,
        proof: &Self::Proof,
    ) -> Result<(), FinalizerError> {
        WhirPrecommittedBooleanWarpFinalizerProtocol::verify(self, instance, proof)
    }
}

const WHIR_WARP_OPENING_TAG: u64 = 0x5741_5250_4f50_454e;
const WHIR_WARP_PESAT_TAG: u64 = 0x5741_5250_5045_5341;
const EXTENSION_LIMB_PCS_TAG: u64 = 0x5741_5250_4c49_4d42;
const WHIR_CODEWORD_BACKEND_TAG: u64 = 0x5741_5250_434f_4445;

fn eval_eq_ext<EF: Field>(lhs: &[EF], rhs: &[EF]) -> EF {
    debug_assert_eq!(lhs.len(), rhs.len());
    lhs.iter()
        .zip(rhs.iter())
        .map(|(&l, &r)| l * r + (EF::ONE - l) * (EF::ONE - r))
        .product()
}

fn boolean_index_point<EF: Field>(index: usize, log_len: usize) -> Vec<EF> {
    (0..log_len)
        .map(|i| {
            if (index >> (log_len - 1 - i)) & 1 == 1 {
                EF::ONE
            } else {
                EF::ZERO
            }
        })
        .collect()
}

fn extension_limb_challenger<F, Challenger>(challenger: &Challenger, limb: usize) -> Challenger
where
    F: Field,
    Challenger: Clone + CanObserve<F>,
{
    let mut limb_challenger = challenger.clone();
    limb_challenger.observe(F::from_u64(EXTENSION_LIMB_PCS_TAG));
    limb_challenger.observe(F::from_u64(limb as u64));
    limb_challenger
}

fn check_limb_count<F, EF, InnerError>(
    actual: usize,
) -> Result<(), ExtensionLimbPcsError<InnerError>>
where
    F: Field,
    EF: ExtensionField<F>,
{
    let dimension = <EF as BasedVectorSpace<F>>::DIMENSION;
    if actual != dimension {
        return Err(ExtensionLimbPcsError::WrongLimbCount {
            expected: dimension,
            actual,
        });
    }
    Ok(())
}

fn recompose_limb_opened_values<F, EF>(
    limb_opened_values: &[MultilinearOpenedValues<EF>],
) -> Option<MultilinearOpenedValues<EF>>
where
    F: Field,
    EF: ExtensionField<F>,
{
    let dimension = <EF as BasedVectorSpace<F>>::DIMENSION;
    if limb_opened_values.len() != dimension {
        return None;
    }
    if limb_opened_values.is_empty() {
        return Some(Vec::new());
    }

    let num_polys = limb_opened_values[0].len();
    let basis = (0..dimension)
        .map(|limb| {
            <EF as BasedVectorSpace<F>>::ith_basis_element(limb).expect("basis limb is in range")
        })
        .collect::<Vec<_>>();

    let mut recomposed = Vec::with_capacity(num_polys);
    for poly_idx in 0..num_polys {
        let num_points = limb_opened_values[0][poly_idx].len();
        let mut values = Vec::with_capacity(num_points);
        for point_idx in 0..num_points {
            let mut value = EF::ZERO;
            for limb in 0..dimension {
                if limb_opened_values[limb].len() != num_polys
                    || limb_opened_values[limb][poly_idx].len() != num_points
                {
                    return None;
                }
                value += limb_opened_values[limb][poly_idx][point_idx] * basis[limb];
            }
            values.push(value);
        }
        recomposed.push(values);
    }

    Some(recomposed)
}

fn opening_claims_match_values<EF: Field>(
    opening_claims: &[Vec<(Point<EF>, EF)>],
    opened_values: &MultilinearOpenedValues<EF>,
) -> bool {
    opening_claims.len() == opened_values.len()
        && opening_claims
            .iter()
            .zip(opened_values)
            .all(|(claims_for_poly, values_for_poly)| {
                claims_for_poly.len() == values_for_poly.len()
                    && claims_for_poly
                        .iter()
                        .zip(values_for_poly)
                        .all(|((_, claim), value)| claim == value)
            })
}

fn limb_opening_claims<EF: Field, InnerError>(
    opening_claims: &[Vec<(Point<EF>, EF)>],
    limb_values: &MultilinearOpenedValues<EF>,
) -> Result<Vec<Vec<(Point<EF>, EF)>>, ExtensionLimbPcsError<InnerError>> {
    if opening_claims.len() != limb_values.len() {
        return Err(ExtensionLimbPcsError::ShapeMismatch(
            "limb opened values do not match polynomial count",
        ));
    }

    opening_claims
        .iter()
        .zip(limb_values)
        .map(|(claims_for_poly, values_for_poly)| {
            if claims_for_poly.len() != values_for_poly.len() {
                return Err(ExtensionLimbPcsError::ShapeMismatch(
                    "limb opened values do not match opening point count",
                ));
            }
            Ok(claims_for_poly
                .iter()
                .zip(values_for_poly)
                .map(|((point, _), value)| (point.clone(), *value))
                .collect())
        })
        .collect()
}

/// Precommitted opening prover/verifier for the final accumulator codeword.
///
/// The type is generic over the PCS because the soundness requirement is about
/// the trait contract, not a particular implementation detail: `commit(f)`
/// must produce the same public commitment type and layout as `acc.x.rt`.
/// Once `p3-whir` exposes that precommitted layout for WARP's RS oracle, it can
/// instantiate this layer directly.
pub struct WhirAccumulatorOpeningProtocol<'a, F, EF, Pcs, Challenger, Dft>
where
    F: TwoAdicField,
    EF: ExtensionField<F>,
    Dft: TwoAdicSubgroupDft<F>,
{
    pcs: &'a Pcs,
    code: &'a ReedSolomonCode<F, Dft>,
    challenger_seed: Challenger,
    _ph: PhantomData<EF>,
}

impl<'a, F, EF, Pcs, Challenger, Dft>
    WhirAccumulatorOpeningProtocol<'a, F, EF, Pcs, Challenger, Dft>
where
    F: TwoAdicField,
    EF: ExtensionField<F>,
    Dft: TwoAdicSubgroupDft<F>,
{
    /// Create a new opening protocol from a compatible PCS and transcript seed.
    ///
    /// `challenger_seed` is cloned for proving and verification, so callers
    /// should pass the same transcript state that should prefix this final
    /// opening protocol.
    pub const fn new(
        pcs: &'a Pcs,
        code: &'a ReedSolomonCode<F, Dft>,
        challenger_seed: Challenger,
    ) -> Self {
        Self {
            pcs,
            code,
            challenger_seed,
            _ph: PhantomData,
        }
    }
}

impl<'a, F, EF, Pcs, Challenger, Dft>
    WhirAccumulatorOpeningProtocol<'a, F, EF, Pcs, Challenger, Dft>
where
    F: TwoAdicField,
    EF: ExtensionField<F> + TwoAdicField,
    Pcs: MultilinearPcs<EF, Challenger, Val = EF>,
    Pcs::Commitment: PartialEq,
    Challenger: Clone,
    Dft: TwoAdicSubgroupDft<F>,
{
    /// Prove `f_hat(alpha) = mu` for the accumulator's committed final
    /// codeword.
    ///
    /// This is fail-closed:
    /// - `code` must be systematic, since the later PESAT terminal openings
    ///   depend on the same message-subspace layout.
    /// - `pcs.num_vars()` must match the codeword MLE dimension.
    /// - `pcs.commit(witness.f)` must equal `instance.rt`.
    /// - the opened PCS value must equal `instance.mu`.
    pub fn prove<ProverData>(
        &self,
        instance: &AccumulatorInstance<EF, Pcs::Commitment>,
        witness: &AccumulatorWitness<EF, ProverData>,
    ) -> Result<WhirAccumulatorOpeningProof<Pcs::Proof>, FinalizerError> {
        self.validate_instance_shape(instance)?;
        if witness.f.len() != self.code.codeword_len() {
            return Err(FinalizerError::Decider(DeciderError::EncodingMismatch));
        }

        let opening_points = self.opening_points(instance);
        let mut challenger = self.challenger_seed.clone();
        let evaluations = RowMajorMatrix::new(witness.f.clone(), 1);
        let (commitment, prover_data) =
            self.pcs
                .commit(evaluations, &opening_points, &mut challenger);
        if commitment != instance.rt {
            return Err(FinalizerError::Decider(DeciderError::MerkleRoot));
        }

        let (opened_values, pcs_proof) = self.pcs.open(prover_data, &mut challenger);
        let opened_mu = opened_values
            .first()
            .and_then(|poly_values| poly_values.first())
            .copied()
            .ok_or(FinalizerError::Unsupported(
                "PCS did not return the accumulator opening value",
            ))?;
        if opened_mu != instance.mu {
            return Err(FinalizerError::Decider(DeciderError::MlEval));
        }

        Ok(WhirAccumulatorOpeningProof { pcs_proof })
    }

    /// Verify the accumulator codeword opening proof against `acc.x.rt`.
    pub fn verify(
        &self,
        instance: &AccumulatorInstance<EF, Pcs::Commitment>,
        proof: &WhirAccumulatorOpeningProof<Pcs::Proof>,
    ) -> Result<(), FinalizerError> {
        self.validate_instance_shape(instance)?;
        let opening_claims = [vec![(Point::new(instance.alpha.clone()), instance.mu)]];
        let mut challenger = self.challenger_seed.clone();
        self.pcs
            .verify(
                &instance.rt,
                &opening_claims,
                &proof.pcs_proof,
                &mut challenger,
            )
            .map_err(|err| FinalizerError::OpeningProof(format!("{err:?}")))
    }

    fn validate_instance_shape(
        &self,
        instance: &AccumulatorInstance<EF, Pcs::Commitment>,
    ) -> Result<(), FinalizerError> {
        if !self.code.is_systematic() {
            return Err(FinalizerError::Unsupported(
                "WHIR-native WARP finalization requires systematic RS encoding",
            ));
        }
        if self.pcs.num_vars() != self.code.log_codeword_len() {
            return Err(FinalizerError::Unsupported(
                "PCS variable count must match the WARP codeword MLE dimension",
            ));
        }
        if instance.alpha.len() != self.code.log_codeword_len() {
            return Err(FinalizerError::Decider(DeciderError::MlEval));
        }
        Ok(())
    }

    fn opening_points(
        &self,
        instance: &AccumulatorInstance<EF, Pcs::Commitment>,
    ) -> [Vec<Point<EF>>; 1] {
        [vec![Point::new(instance.alpha.clone())]]
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec::Vec;
    use core::marker::PhantomData;

    use p3_baby_bear::{BabyBear, Poseidon2BabyBear};
    use p3_challenger::{
        CanObserve, CanSample, CanSampleBits, DuplexChallenger, FieldChallenger, GrindingChallenger,
    };
    use p3_commit::{MultilinearOpenedValues, MultilinearPcs};
    use p3_dft::Radix2DFTSmallBatch;
    use p3_field::BasedVectorSpace;
    use p3_field::extension::BinomialExtensionField;
    use p3_field::{ExtensionField, Field, PrimeCharacteristicRing};
    use p3_matrix::Matrix;
    use p3_matrix::dense::RowMajorMatrix;
    use p3_merkle_tree::MerkleTreeMmcs;
    use p3_multilinear_util::point::Point;
    use p3_multilinear_util::poly::Poly;
    use p3_symmetric::{MerkleCap, PaddingFreeSponge, TruncatedPermutation};
    use p3_whir::parameters::{
        FoldingFactor, ProtocolParameters, SecurityAssumption, SumcheckStrategy,
    };
    use p3_whir::pcs::WhirPcs;
    use rand::SeedableRng;
    use rand::rngs::SmallRng;

    use super::*;

    type F = BabyBear;
    type EF = BinomialExtensionField<F, 4>;
    type Dft = Radix2DFTSmallBatch<F>;
    type Perm = Poseidon2BabyBear<16>;
    type MyHash = PaddingFreeSponge<Perm, 16, 8, 8>;
    type MyCompress = TruncatedPermutation<Perm, 2, 8, 16>;
    type MyChallenger = DuplexChallenger<F, Perm, 16, 8>;
    type PackedF = <F as Field>::Packing;
    type MyMmcs = MerkleTreeMmcs<PackedF, PackedF, MyHash, MyCompress, 2, 8>;
    type MyCommitment = MerkleCap<F, [F; 8]>;
    type TestWhirPcs = WhirPcs<EF, F, MyMmcs, MyChallenger, Dft, 8>;
    type TestChallengerWhirPcs = WhirPcs<EF, F, MyMmcs, TestChallenger<F>, Dft, 8>;
    type TestWhirProof = WhirProof<F, EF, MyMmcs>;
    type TestWhirLimbProof = ExtensionLimbPcsProof<EF, TestWhirProof>;
    type TestWhirFinalizerProof = WhirWarpFinalizerProof<EF, TestWhirLimbProof>;
    type TestWhirRootProof = crate::WarpExternalRootProofBatched<
        F,
        EF,
        Vec<MyCommitment>,
        MyCommitment,
        TestWhirProof,
        TestWhirLimbProof,
        TestWhirFinalizerProof,
    >;

    #[derive(Clone, Debug)]
    struct TestChallenger<F> {
        state: F,
    }

    impl<F: Field> TestChallenger<F> {
        const fn new(state: F) -> Self {
            Self { state }
        }
    }

    impl<F: Field> CanObserve<F> for TestChallenger<F> {
        fn observe(&mut self, value: F) {
            self.state += value;
        }
    }

    impl CanObserve<Vec<EF>> for TestChallenger<F> {
        fn observe(&mut self, values: Vec<EF>) {
            for value in values {
                for &coeff in <EF as BasedVectorSpace<F>>::as_basis_coefficients_slice(&value) {
                    <Self as CanObserve<F>>::observe(self, coeff);
                }
            }
        }
    }

    impl<F: Field> CanObserve<Vec<Vec<F>>> for TestChallenger<F> {
        fn observe(&mut self, valuess: Vec<Vec<F>>) {
            for values in valuess {
                for value in values {
                    <Self as CanObserve<F>>::observe(self, value);
                }
            }
        }
    }

    impl CanObserve<MyCommitment> for TestChallenger<F> {
        fn observe(&mut self, commitment: MyCommitment) {
            for root in commitment.roots() {
                for &value in root {
                    <Self as CanObserve<F>>::observe(self, value);
                }
            }
        }
    }

    impl CanObserve<Vec<MyCommitment>> for TestChallenger<F> {
        fn observe(&mut self, commitments: Vec<MyCommitment>) {
            for commitment in commitments {
                <Self as CanObserve<MyCommitment>>::observe(self, commitment);
            }
        }
    }

    impl<F: Field> CanSample<F> for TestChallenger<F> {
        fn sample(&mut self) -> F {
            let out = self.state;
            self.state += F::ONE;
            out
        }
    }

    impl<F: Field> CanSampleBits<usize> for TestChallenger<F> {
        fn sample_bits(&mut self, _bits: usize) -> usize {
            0
        }
    }

    impl<F> FieldChallenger<F> for TestChallenger<F> where F: Field + Sync {}

    impl<F> GrindingChallenger for TestChallenger<F>
    where
        F: Field + Sync,
    {
        type Witness = F;

        fn grind(&mut self, _bits: usize) -> Self::Witness {
            F::ZERO
        }
    }

    #[derive(Clone, Debug)]
    struct RawEvalPcs<Val> {
        num_vars: usize,
        _ph: PhantomData<Val>,
    }

    impl<Val> RawEvalPcs<Val> {
        const fn new(num_vars: usize) -> Self {
            Self {
                num_vars,
                _ph: PhantomData,
            }
        }
    }

    struct RawEvalProverData<Val> {
        evaluations: Vec<Val>,
        opening_points: Vec<Vec<Point<Val>>>,
    }

    impl<Val, Challenger> MultilinearPcs<Val, Challenger> for RawEvalPcs<Val>
    where
        Val: ExtensionField<Val> + Serialize + serde::de::DeserializeOwned,
    {
        type Val = Val;
        type Commitment = Vec<Val>;
        type ProverData = RawEvalProverData<Val>;
        type Proof = ();
        type Error = ();

        fn num_vars(&self) -> usize {
            self.num_vars
        }

        fn commit(
            &self,
            evaluations: RowMajorMatrix<Self::Val>,
            opening_points: &[Vec<Point<Val>>],
            _challenger: &mut Challenger,
        ) -> (Self::Commitment, Self::ProverData) {
            assert_eq!(evaluations.width(), 1);
            assert_eq!(evaluations.height(), 1 << self.num_vars);
            (
                evaluations.values.clone(),
                RawEvalProverData {
                    evaluations: evaluations.values,
                    opening_points: opening_points.to_vec(),
                },
            )
        }

        fn open(
            &self,
            prover_data: Self::ProverData,
            _challenger: &mut Challenger,
        ) -> (MultilinearOpenedValues<Val>, Self::Proof) {
            let poly = Poly::<Val>::new(prover_data.evaluations);
            let values = prover_data
                .opening_points
                .iter()
                .map(|points| {
                    points
                        .iter()
                        .map(|point| poly.eval_ext::<Val>(point))
                        .collect()
                })
                .collect();
            (values, ())
        }

        fn verify(
            &self,
            commitment: &Self::Commitment,
            opening_claims: &[Vec<(Point<Val>, Val)>],
            _proof: &Self::Proof,
            _challenger: &mut Challenger,
        ) -> Result<(), Self::Error> {
            let poly = Poly::<Val>::new(commitment.clone());
            for claims_for_poly in opening_claims {
                for (point, value) in claims_for_poly {
                    if poly.eval_ext::<Val>(point) != *value {
                        return Err(());
                    }
                }
            }
            Ok(())
        }
    }

    #[derive(Clone, Debug)]
    struct BaseEvalPcs<Base, Challenge> {
        num_vars: usize,
        _ph: PhantomData<(Base, Challenge)>,
    }

    impl<Base, Challenge> BaseEvalPcs<Base, Challenge> {
        const fn new(num_vars: usize) -> Self {
            Self {
                num_vars,
                _ph: PhantomData,
            }
        }
    }

    struct BaseEvalProverData<Base, Challenge> {
        evaluations: Vec<Base>,
        opening_points: Vec<Vec<Point<Challenge>>>,
    }

    impl<Base, Challenge, Challenger> MultilinearPcs<Challenge, Challenger>
        for BaseEvalPcs<Base, Challenge>
    where
        Base: Field + Serialize + serde::de::DeserializeOwned,
        Challenge: ExtensionField<Base> + Serialize + serde::de::DeserializeOwned,
    {
        type Val = Base;
        type Commitment = Vec<Base>;
        type ProverData = BaseEvalProverData<Base, Challenge>;
        type Proof = ();
        type Error = ();

        fn num_vars(&self) -> usize {
            self.num_vars
        }

        fn commit(
            &self,
            evaluations: RowMajorMatrix<Self::Val>,
            opening_points: &[Vec<Point<Challenge>>],
            _challenger: &mut Challenger,
        ) -> (Self::Commitment, Self::ProverData) {
            assert_eq!(evaluations.width(), 1);
            assert_eq!(evaluations.height(), 1 << self.num_vars);
            (
                evaluations.values.clone(),
                BaseEvalProverData {
                    evaluations: evaluations.values,
                    opening_points: opening_points.to_vec(),
                },
            )
        }

        fn open(
            &self,
            prover_data: Self::ProverData,
            _challenger: &mut Challenger,
        ) -> (MultilinearOpenedValues<Challenge>, Self::Proof) {
            let poly = Poly::<Base>::new(prover_data.evaluations);
            let values = prover_data
                .opening_points
                .iter()
                .map(|points| {
                    points
                        .iter()
                        .map(|point| poly.eval_base::<Challenge>(point))
                        .collect()
                })
                .collect();
            (values, ())
        }

        fn verify(
            &self,
            commitment: &Self::Commitment,
            opening_claims: &[Vec<(Point<Challenge>, Challenge)>],
            _proof: &Self::Proof,
            _challenger: &mut Challenger,
        ) -> Result<(), Self::Error> {
            let poly = Poly::<Base>::new(commitment.clone());
            for claims_for_poly in opening_claims {
                for (point, value) in claims_for_poly {
                    if poly.eval_base::<Challenge>(point) != *value {
                        return Err(());
                    }
                }
            }
            Ok(())
        }
    }

    fn systematic_code() -> ReedSolomonCode<F, Dft> {
        ReedSolomonCode::new_systematic(4, 1, Dft::default())
    }

    fn whir_pcs(num_vars: usize) -> TestWhirPcs {
        let mut rng = SmallRng::seed_from_u64(0x5748_4952);
        let perm = Perm::new_from_rng_128(&mut rng);
        let merkle_hash = MyHash::new(perm.clone());
        let merkle_compress = MyCompress::new(perm);
        let mmcs = MyMmcs::new(merkle_hash, merkle_compress, 0);
        let whir_params = ProtocolParameters {
            security_level: 32,
            pow_bits: 0,
            rs_domain_initial_reduction_factor: 1,
            folding_factor: FoldingFactor::Constant(2),
            mmcs,
            soundness_type: SecurityAssumption::CapacityBound,
            starting_log_inv_rate: 1,
        };
        TestWhirPcs::new(
            num_vars,
            whir_params,
            Dft::default(),
            SumcheckStrategy::default(),
        )
    }

    fn whir_pcs_test_challenger(num_vars: usize) -> TestChallengerWhirPcs {
        let mut rng = SmallRng::seed_from_u64(0x5748_4952);
        let perm = Perm::new_from_rng_128(&mut rng);
        let merkle_hash = MyHash::new(perm.clone());
        let merkle_compress = MyCompress::new(perm);
        let mmcs = MyMmcs::new(merkle_hash, merkle_compress, 0);
        let whir_params = ProtocolParameters {
            security_level: 32,
            pow_bits: 0,
            rs_domain_initial_reduction_factor: 1,
            folding_factor: FoldingFactor::Constant(2),
            mmcs,
            soundness_type: SecurityAssumption::CapacityBound,
            starting_log_inv_rate: 1,
        };
        TestChallengerWhirPcs::new(
            num_vars,
            whir_params,
            Dft::default(),
            SumcheckStrategy::default(),
        )
    }

    fn whir_challenger(seed: u64) -> MyChallenger {
        let mut rng = SmallRng::seed_from_u64(seed);
        MyChallenger::new(Perm::new_from_rng_128(&mut rng))
    }

    fn accumulator_fixture() -> (
        ReedSolomonCode<F, Dft>,
        AccumulatorInstance<EF, Vec<EF>>,
        AccumulatorWitness<EF, ()>,
    ) {
        use rand::RngExt;

        let code = systematic_code();
        let mut rng = SmallRng::seed_from_u64(0x57484952);
        let w: Vec<EF> = (0..code.msg_len()).map(|_| rng.random()).collect();
        let f = code.encode_algebra::<EF>(&w);
        let alpha = Point::<EF>::rand(&mut rng, code.log_codeword_len());
        let mu = Poly::<EF>::new(f.clone()).eval_ext::<F>(&alpha);
        let instance = AccumulatorInstance {
            rt: f.clone(),
            alpha: alpha.as_slice().to_vec(),
            mu,
            beta: Vec::new(),
            eta: EF::ZERO,
        };
        let witness = AccumulatorWitness { td: (), f, w };
        (code, instance, witness)
    }

    fn boolean_pesat_fixture() -> (
        ReedSolomonCode<F, Dft>,
        BooleanPesat<F, EF>,
        AccumulatorInstance<EF, Vec<EF>>,
        AccumulatorWitness<EF, ()>,
    ) {
        let log_witness = 4;
        let pesat = BooleanPesat::<F, EF>::new(log_witness, b"BooleanPesat/whir".to_vec());
        let code = ReedSolomonCode::new_systematic(log_witness, 1, Dft::default());
        let w = (0..pesat.shape().witness_len())
            .map(|i| EF::from_bool(i % 3 == 0))
            .collect::<Vec<_>>();
        let f = code.encode_algebra::<EF>(&w);
        let beta = (0..pesat.shape().log_constraints)
            .map(|i| EF::from_u64((i as u64) + 2))
            .collect::<Vec<_>>();
        let beta_eq = Poly::<EF>::new_from_point(&beta, EF::ONE);
        let eta = pesat.evaluate_bundled(beta_eq.as_slice(), &w);
        let instance = AccumulatorInstance {
            rt: f.clone(),
            alpha: vec![EF::ZERO; code.log_codeword_len()],
            mu: Poly::<EF>::new(f.clone())
                .eval_ext::<F>(&Point::new(vec![EF::ZERO; code.log_codeword_len()])),
            beta,
            eta,
        };
        let witness = AccumulatorWitness { td: (), f, w };
        (code, pesat, instance, witness)
    }

    fn whir_native_root_fixture() -> (
        ReedSolomonCode<F, Dft>,
        BooleanPesat<F, EF>,
        MyMmcs,
        crate::WarpParams,
        TestChallenger<F>,
        TestChallengerWhirPcs,
        AccumulatorInstance<EF, Vec<MyCommitment>>,
        TestWhirRootProof,
    ) {
        use crate::{WarpParams, WarpRootProver};

        let code = systematic_code();
        let pesat =
            BooleanPesat::<F, EF>::new(code.log_msg_len(), b"BooleanPesat/whir-root".to_vec());
        let mut rng = SmallRng::seed_from_u64(0x5254);
        let perm = Perm::new_from_rng_128(&mut rng);
        let mmcs = MyMmcs::new(MyHash::new(perm.clone()), MyCompress::new(perm), 0);
        let params = WarpParams::new(1, 2);
        let base_challenger = TestChallenger::new(F::from_u64(17));
        let whir_pcs = whir_pcs_test_challenger(code.log_codeword_len());
        let fresh_backend = WhirCodewordBackend::<F, EF, _, TestChallenger<F>, Dft, 8>::new(
            &whir_pcs,
            TestChallenger::new(F::from_u64(23)),
        );
        let acc_backend = WhirLimbAccumulatorBackend::<F, EF, _, TestChallenger<F>, Dft, 8>::new(
            &whir_pcs,
            TestChallenger::new(F::from_u64(29)),
        );
        let finalizer =
            WhirPrecommittedBooleanWarpFinalizerProtocol::<F, EF, _, TestChallenger<F>, Dft>::new(
                &acc_backend,
                &code,
                &pesat,
                TestChallenger::new(F::from_u64(31)),
            );

        let make_witness = |seed: u64| -> Vec<F> {
            (0..code.msg_len())
                .map(|i| F::from_bool(((seed + i as u64) & 1) == 1))
                .collect()
        };
        let step_fresh_committed = vec![
            (0..4)
                .map(|i| {
                    let witness = make_witness(100 + i as u64);
                    fresh_backend
                        .commit_codeword(code.encode(&witness), witness)
                        .expect("WHIR fresh commit")
                })
                .collect::<Vec<_>>(),
            (0..3)
                .map(|i| {
                    let witness = make_witness(200 + i as u64);
                    fresh_backend
                        .commit_codeword(code.encode(&witness), witness)
                        .expect("WHIR fresh commit")
                })
                .collect::<Vec<_>>(),
        ];

        let root_prover = WarpRootProver::new(&mmcs, &code, &pesat, params);
        let (claimed_final, root_proof) = root_prover
            .prove_external_linear_chain_accumulator_batched(
                &base_challenger,
                &fresh_backend,
                &acc_backend,
                step_fresh_committed,
                &finalizer,
            )
            .expect("WHIR-native root prove");

        (
            code,
            pesat,
            mmcs,
            params,
            base_challenger,
            whir_pcs,
            claimed_final,
            root_proof,
        )
    }

    fn verify_whir_native_root(
        code: &ReedSolomonCode<F, Dft>,
        pesat: &BooleanPesat<F, EF>,
        mmcs: &MyMmcs,
        params: crate::WarpParams,
        base_challenger: &TestChallenger<F>,
        whir_pcs: &TestChallengerWhirPcs,
        root_proof: &TestWhirRootProof,
    ) -> Result<AccumulatorInstance<EF, Vec<MyCommitment>>, crate::WarpError> {
        use crate::WarpRootVerifier;

        let fresh_backend = WhirCodewordBackend::<F, EF, _, TestChallenger<F>, Dft, 8>::new(
            whir_pcs,
            TestChallenger::new(F::from_u64(23)),
        );
        let acc_backend = WhirLimbAccumulatorBackend::<F, EF, _, TestChallenger<F>, Dft, 8>::new(
            whir_pcs,
            TestChallenger::new(F::from_u64(29)),
        );
        let finalizer =
            WhirPrecommittedBooleanWarpFinalizerProtocol::<F, EF, _, TestChallenger<F>, Dft>::new(
                &acc_backend,
                code,
                pesat,
                TestChallenger::new(F::from_u64(31)),
            );
        let root_verifier = WarpRootVerifier::new(mmcs, code, pesat, params);
        root_verifier.verify_external_linear_chain_accumulator_batched(
            base_challenger,
            &fresh_backend,
            &acc_backend,
            root_proof,
            &finalizer,
        )
    }

    #[test]
    fn opening_protocol_proves_and_verifies_accumulator_mle_claim() {
        let (code, instance, witness) = accumulator_fixture();
        let pcs = RawEvalPcs::<EF>::new(code.log_codeword_len());
        let protocol = WhirAccumulatorOpeningProtocol::<F, EF, _, (), _>::new(&pcs, &code, ());

        let proof = protocol.prove(&instance, &witness).unwrap();
        protocol.verify(&instance, &proof).unwrap();
    }

    #[test]
    fn opening_protocol_rejects_unrelated_commitment() {
        let (code, mut instance, witness) = accumulator_fixture();
        let pcs = RawEvalPcs::<EF>::new(code.log_codeword_len());
        let protocol = WhirAccumulatorOpeningProtocol::<F, EF, _, (), _>::new(&pcs, &code, ());
        instance.rt[0] += EF::ONE;

        let err = protocol.prove(&instance, &witness).unwrap_err();
        assert!(matches!(
            err,
            FinalizerError::Decider(DeciderError::MerkleRoot)
        ));
    }

    #[test]
    fn opening_protocol_rejects_bad_claimed_mu() {
        let (code, mut instance, witness) = accumulator_fixture();
        let pcs = RawEvalPcs::<EF>::new(code.log_codeword_len());
        let protocol = WhirAccumulatorOpeningProtocol::<F, EF, _, (), _>::new(&pcs, &code, ());
        instance.mu += EF::ONE;

        let err = protocol.prove(&instance, &witness).unwrap_err();
        assert!(matches!(err, FinalizerError::Decider(DeciderError::MlEval)));
    }

    #[test]
    fn extension_limb_pcs_recomposes_base_field_openings() {
        let (code, instance, witness) = accumulator_fixture();
        let inner = BaseEvalPcs::<F, EF>::new(code.log_codeword_len());
        let pcs = ExtensionLimbPcs::<F, EF, _>::new(&inner);
        let opening_points = [vec![Point::new(instance.alpha.clone())]];

        let (commitment, prover_data) = pcs.commit(
            RowMajorMatrix::new(witness.f.clone(), 1),
            &opening_points,
            &mut TestChallenger::new(F::ZERO),
        );
        let (opened_values, proof) = pcs.open(prover_data, &mut TestChallenger::new(F::ZERO));

        assert_eq!(commitment.len(), <EF as BasedVectorSpace<F>>::DIMENSION);
        assert_eq!(opened_values[0][0], instance.mu);
        pcs.verify(
            &commitment,
            &[vec![(Point::new(instance.alpha), instance.mu)]],
            &proof,
            &mut TestChallenger::new(F::ZERO),
        )
        .unwrap();
    }

    #[test]
    fn extension_limb_pcs_rejects_tampered_limb_opening() {
        let (code, instance, witness) = accumulator_fixture();
        let inner = BaseEvalPcs::<F, EF>::new(code.log_codeword_len());
        let pcs = ExtensionLimbPcs::<F, EF, _>::new(&inner);
        let opening_points = [vec![Point::new(instance.alpha.clone())]];

        let (commitment, prover_data) = pcs.commit(
            RowMajorMatrix::new(witness.f.clone(), 1),
            &opening_points,
            &mut TestChallenger::new(F::ZERO),
        );
        let (_, mut proof) = pcs.open(prover_data, &mut TestChallenger::new(F::ZERO));
        proof.limb_opened_values[0][0][0] += EF::ONE;

        let err = pcs
            .verify(
                &commitment,
                &[vec![(Point::new(instance.alpha), instance.mu)]],
                &proof,
                &mut TestChallenger::new(F::ZERO),
            )
            .unwrap_err();
        assert!(matches!(err, ExtensionLimbPcsError::RecompositionMismatch));
    }

    #[test]
    fn opening_protocol_accepts_extension_limb_pcs_commitment() {
        let (code, raw_instance, witness) = accumulator_fixture();
        let inner = BaseEvalPcs::<F, EF>::new(code.log_codeword_len());
        let pcs = ExtensionLimbPcs::<F, EF, _>::new(&inner);
        let opening_points = [vec![Point::new(raw_instance.alpha.clone())]];
        let (rt, _) = pcs.commit(
            RowMajorMatrix::new(witness.f.clone(), 1),
            &opening_points,
            &mut TestChallenger::new(F::ZERO),
        );
        let instance = AccumulatorInstance {
            rt,
            alpha: raw_instance.alpha,
            mu: raw_instance.mu,
            beta: raw_instance.beta,
            eta: raw_instance.eta,
        };
        let protocol = WhirAccumulatorOpeningProtocol::<F, EF, _, TestChallenger<F>, _>::new(
            &pcs,
            &code,
            TestChallenger::new(F::ONE),
        );

        let proof = protocol.prove(&instance, &witness).unwrap();
        protocol.verify(&instance, &proof).unwrap();
    }

    #[test]
    fn whir_limb_accumulator_backend_opens_committed_accumulator_index() {
        let (code, _raw_instance, witness) = accumulator_fixture();
        let pcs = whir_pcs(code.log_codeword_len());
        let backend = WhirLimbAccumulatorBackend::<F, EF, _, MyChallenger, Dft, 8>::new(
            &pcs,
            whir_challenger(0xacc),
        );

        let (rt, td) = backend.commit(witness.f.clone()).unwrap();
        let index = 7;
        let (value, proof) = backend.open(&td, index).unwrap();
        assert_eq!(value, witness.f[index]);
        backend
            .verify_opening(&rt, code.log_codeword_len(), index, value, &proof)
            .unwrap();

        let err = backend
            .verify_opening(&rt, code.log_codeword_len(), index, value + EF::ONE, &proof)
            .unwrap_err();
        assert!(matches!(err, ExtensionLimbPcsError::RecompositionMismatch));
    }

    #[test]
    fn whir_codeword_backend_opens_fresh_codeword_index() {
        let code = systematic_code();
        let witness = (0..code.msg_len())
            .map(|i| F::from_u64(i as u64 + 3))
            .collect::<Vec<_>>();
        let codeword = code.encode(&witness);
        let pcs = whir_pcs(code.log_codeword_len());
        let backend =
            WhirCodewordBackend::<F, EF, _, MyChallenger, Dft, 8>::new(&pcs, whir_challenger(0x51));
        let committed = backend
            .commit_codeword(codeword.clone(), witness)
            .expect("WHIR fresh commit");

        let index = 9;
        let (value, proof) = backend.open(&committed, index).unwrap();
        assert_eq!(value, codeword[index]);
        backend
            .verify_opening(
                &committed.commitment(),
                code.log_codeword_len(),
                index,
                value,
                &proof,
            )
            .unwrap();

        let err = backend
            .verify_opening(
                &committed.commitment(),
                code.log_codeword_len(),
                index,
                value + F::ONE,
                &proof,
            )
            .unwrap_err();
        assert!(matches!(
            err,
            p3_whir::pcs::verifier::errors::VerifierError::SumcheckFailed { .. }
                | p3_whir::pcs::verifier::errors::VerifierError::StirChallengeFailed { .. }
                | p3_whir::pcs::verifier::errors::VerifierError::MerkleProofInvalid { .. }
        ));
    }

    #[test]
    fn whir_native_root_proof_verifies_steps_and_whir_finalizer() {
        let (code, pesat, mmcs, params, base_challenger, whir_pcs, claimed_final, root_proof) =
            whir_native_root_fixture();
        let verified_final = verify_whir_native_root(
            &code,
            &pesat,
            &mmcs,
            params,
            &base_challenger,
            &whir_pcs,
            &root_proof,
        )
        .expect("WHIR-native root verify");

        assert_eq!(verified_final.mu, claimed_final.mu);
        assert_eq!(verified_final.eta, claimed_final.eta);
        assert_eq!(root_proof.steps.len(), 2);
    }

    #[test]
    fn whir_native_root_rejects_tampered_fresh_commitment() {
        let (code, pesat, mmcs, params, base_challenger, whir_pcs, _, mut root_proof) =
            whir_native_root_fixture();
        let mut roots = root_proof.steps[0].fresh_commitments[0].roots().to_vec();
        roots[0][0] += F::ONE;
        root_proof.steps[0].fresh_commitments[0] = MerkleCap::new(roots);

        verify_whir_native_root(
            &code,
            &pesat,
            &mmcs,
            params,
            &base_challenger,
            &whir_pcs,
            &root_proof,
        )
        .expect_err("tampered fresh WHIR commitment must be rejected");
    }

    #[test]
    fn whir_native_root_rejects_valid_fresh_commitment_substitution() {
        let (code, pesat, mmcs, params, base_challenger, whir_pcs, _, mut root_proof) =
            whir_native_root_fixture();
        let fresh_backend = WhirCodewordBackend::<F, EF, _, TestChallenger<F>, Dft, 8>::new(
            &whir_pcs,
            TestChallenger::new(F::from_u64(23)),
        );
        let witness = (0..code.msg_len())
            .map(|i| F::from_bool(i % 2 == 0))
            .collect::<Vec<_>>();
        let alternate = fresh_backend
            .commit_codeword(code.encode(&witness), witness)
            .expect("alternate WHIR fresh commit");

        root_proof.steps[0].fresh_commitments[0] = alternate.commitment();

        verify_whir_native_root(
            &code,
            &pesat,
            &mmcs,
            params,
            &base_challenger,
            &whir_pcs,
            &root_proof,
        )
        .expect_err("substituting a different valid fresh WHIR commitment must be rejected");
    }

    #[test]
    fn whir_native_root_rejects_dropped_or_reordered_steps() {
        let (code, pesat, mmcs, params, base_challenger, whir_pcs, _, root_proof) =
            whir_native_root_fixture();

        let mut dropped = root_proof.clone();
        dropped.steps.remove(0);
        verify_whir_native_root(
            &code,
            &pesat,
            &mmcs,
            params,
            &base_challenger,
            &whir_pcs,
            &dropped,
        )
        .expect_err("dropping a WHIR-native WARP root step must be rejected");

        let mut reordered = root_proof;
        reordered.steps.swap(0, 1);
        verify_whir_native_root(
            &code,
            &pesat,
            &mmcs,
            params,
            &base_challenger,
            &whir_pcs,
            &reordered,
        )
        .expect_err("reordering WHIR-native WARP root steps must be rejected");
    }

    #[test]
    fn whir_native_root_rejects_tampered_step_instance() {
        let (code, pesat, mmcs, params, base_challenger, whir_pcs, _, mut root_proof) =
            whir_native_root_fixture();
        root_proof.steps[0].instance.mu += EF::ONE;

        verify_whir_native_root(
            &code,
            &pesat,
            &mmcs,
            params,
            &base_challenger,
            &whir_pcs,
            &root_proof,
        )
        .expect_err("tampered WARP step accumulator instance must be rejected");
    }

    #[test]
    fn whir_native_root_rejects_tampered_finalizer_terminal() {
        let (code, pesat, mmcs, params, base_challenger, whir_pcs, _, mut root_proof) =
            whir_native_root_fixture();
        root_proof.final_proof.pesat.terminal_values[0] += EF::ONE;

        verify_whir_native_root(
            &code,
            &pesat,
            &mmcs,
            params,
            &base_challenger,
            &whir_pcs,
            &root_proof,
        )
        .expect_err("tampered WHIR finalizer terminal value must be rejected");
    }

    #[test]
    fn whir_boolean_pesat_protocol_proves_and_verifies_decider_claim() {
        let (code, pesat, instance, witness) = boolean_pesat_fixture();
        let pcs = RawEvalPcs::<EF>::new(code.log_codeword_len());
        let protocol = WhirBooleanPesatProtocol::<F, EF, _, TestChallenger<F>, Dft>::new(
            &pcs,
            &code,
            &pesat,
            TestChallenger::new(F::ONE),
        );

        let proof = protocol.prove(&instance, &witness).unwrap();
        protocol.verify(&instance, &proof).unwrap();
    }

    #[test]
    fn whir_boolean_pesat_protocol_rejects_bad_eta() {
        let (code, pesat, mut instance, witness) = boolean_pesat_fixture();
        let pcs = RawEvalPcs::<EF>::new(code.log_codeword_len());
        let protocol = WhirBooleanPesatProtocol::<F, EF, _, TestChallenger<F>, Dft>::new(
            &pcs,
            &code,
            &pesat,
            TestChallenger::new(F::ONE),
        );

        let proof = protocol.prove(&instance, &witness).unwrap();
        instance.eta += EF::ONE;
        assert!(protocol.verify(&instance, &proof).is_err());
    }

    #[test]
    fn whir_boolean_warp_finalizer_proves_and_verifies_both_decider_claims() {
        let (code, pesat, instance, witness) = boolean_pesat_fixture();
        let pcs = RawEvalPcs::<EF>::new(code.log_codeword_len());
        let protocol = WhirBooleanWarpFinalizerProtocol::<F, EF, _, TestChallenger<F>, Dft>::new(
            &pcs,
            &code,
            &pesat,
            TestChallenger::new(F::ONE),
        );

        let proof = protocol.prove(&instance, &witness).unwrap();
        protocol.verify(&instance, &proof).unwrap();
    }

    #[test]
    fn whir_boolean_warp_finalizer_accepts_extension_limb_pcs_commitment() {
        let (code, pesat, raw_instance, witness) = boolean_pesat_fixture();
        let inner = BaseEvalPcs::<F, EF>::new(code.log_codeword_len());
        let pcs = ExtensionLimbPcs::<F, EF, _>::new(&inner);
        let opening_points = [vec![Point::new(raw_instance.alpha.clone())]];
        let (rt, _) = pcs.commit(
            RowMajorMatrix::new(witness.f.clone(), 1),
            &opening_points,
            &mut TestChallenger::new(F::ZERO),
        );
        let instance = AccumulatorInstance {
            rt,
            alpha: raw_instance.alpha,
            mu: raw_instance.mu,
            beta: raw_instance.beta,
            eta: raw_instance.eta,
        };
        let protocol = WhirBooleanWarpFinalizerProtocol::<F, EF, _, TestChallenger<F>, Dft>::new(
            &pcs,
            &code,
            &pesat,
            TestChallenger::new(F::ONE),
        );

        let proof = protocol.prove(&instance, &witness).unwrap();
        protocol.verify(&instance, &proof).unwrap();
    }

    #[test]
    fn whir_boolean_warp_finalizer_rejects_bad_accumulator_opening_claim() {
        let (code, pesat, mut instance, witness) = boolean_pesat_fixture();
        let pcs = RawEvalPcs::<EF>::new(code.log_codeword_len());
        let protocol = WhirBooleanWarpFinalizerProtocol::<F, EF, _, TestChallenger<F>, Dft>::new(
            &pcs,
            &code,
            &pesat,
            TestChallenger::new(F::ONE),
        );
        let proof = protocol.prove(&instance, &witness).unwrap();
        instance.mu += EF::ONE;

        let err = protocol.verify(&instance, &proof).unwrap_err();
        assert!(matches!(err, FinalizerError::OpeningProof(_)));
    }
}
