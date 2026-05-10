//! Finalisation strategies for a WARP accumulator.
//!
//! After a chain of accumulation steps the prover is left with an
//! accumulator `(acc.x, acc.w)`. The chain of `verify` calls only attests
//! that `acc.x` is well-formed *relative to a witness side that the prover
//! claims to hold*. To convert the accumulator into a verdict on the
//! underlying PESAT instances, someone must run the
//! [`WarpDecider`](crate::WarpDecider). The natural question is: by whom,
//! and is the verdict transmissible?
//!
//! WARP's paper ([eprint 2025/753, Appendix A]) frames this as a
//! direct-vs-indirect trade-off:
//!
//! - **Direct accumulation (default)**: the prover runs the decider locally.
//!   Cheapest, but the verdict isn't transmissible.
//! - **SNARG-wrapped accumulation**: the prover produces a succinct proof
//!   that the decider would accept. Transmissible and succinct, at the cost
//!   of the SNARG's prover work.
//!
//! This module abstracts both modes behind a single [`Finalizer`] trait so
//! downstream users can compose either against the same `Accumulator`
//! types. v1 ships:
//!
//! - [`local::LocalDeciderFinalizer`]: thin wrapper over the existing
//!   [`WarpDecider`](crate::WarpDecider). The "proof" is `()`. `verify`
//!   re-runs the decider — only useful if the verifier holds the witness
//!   side already.
//! - [`witness::WitnessFinalizer`]: bundles `acc.w` into the proof so any
//!   third party can re-run the four decider checks. Linear-size in `n`,
//!   not succinct, but transmissible.
//! - [`whir::WhirAccumulatorOpeningProtocol`]: fail-closed precommitted
//!   multilinear opening layer for the accumulator claim `f_hat(α) = µ`.
//! - [`whir::WhirBooleanPesatProtocol`]: WHIR-native sumcheck plus
//!   systematic-RS openings for the Boolean PESAT decider claim
//!   `Pb(β, w) = η`.
//! - [`whir::WhirPrecommittedBooleanWarpFinalizerProtocol`]: composed finalizer
//!   proof for both decider claims, using a caller-provided backend that opens
//!   the accumulator's existing commitment `rt`. Creating a fresh unrelated
//!   WHIR commitment is not a sound WARP finalizer.

pub mod local;
pub mod whir;
pub mod witness;

pub use local::LocalDeciderFinalizer;
pub use whir::{
    AccumulatorPointOpeningBackend, PrecommittedAccumulatorPcs, WhirAccumulatorOpeningProof,
    WhirAccumulatorOpeningProtocol, WhirBooleanPesatProtocol, WhirBooleanWarpFinalizerProtocol,
    WhirPesatProof, WhirPrecommittedBooleanWarpFinalizerProtocol, WhirWarpFinalizerProof,
};
pub use witness::{WitnessFinalizer, WitnessProof};

use p3_commit::Mmcs;
use p3_field::{ExtensionField, Field};

use crate::accumulator::{AccumulatorInstance, AccumulatorWitness};
use crate::error::FinalizerError;

/// A [`Finalizer`] turns a WARP `Accumulator` into a verdict — either
/// locally (no transmissible proof) or by producing a `Self::Proof` that
/// any third party with `acc.x` can verify.
///
/// # Type parameters
///
/// - `F`, `EF`: base / extension fields.
/// - `MT`: the Merkle commitment scheme used by the WARP accumulator.
/// - `ProverData`: the Mmcs prover-data type for the (extension-field)
///   merged-codeword commitment held in `acc.w.td`.
pub trait Finalizer<F, EF, MT, ProverData>
where
    F: Field,
    EF: ExtensionField<F>,
    MT: Mmcs<F>,
{
    /// Transmissible proof produced by [`finalize`](Self::finalize).
    /// Use `()` for finalizers that don't produce a transmissible proof
    /// (e.g., [`LocalDeciderFinalizer`]).
    type Proof;

    /// Run the prover-side finalisation. Consumes `(acc.x, acc.w)` and
    /// produces a `Self::Proof`. The prover's `acc.w` typically isn't
    /// needed after this call.
    fn finalize(
        &self,
        instance: &AccumulatorInstance<EF, MT::Commitment>,
        witness: &AccumulatorWitness<EF, ProverData>,
    ) -> Result<Self::Proof, FinalizerError>;

    /// Verify a finalisation proof against the accumulator's instance
    /// half. Used by parties who do **not** hold the witness side.
    ///
    /// Some finalizers (notably [`LocalDeciderFinalizer`]) cannot support
    /// remote verification and return `FinalizerError::NoTransmissibleProof`.
    fn verify(
        &self,
        instance: &AccumulatorInstance<EF, MT::Commitment>,
        proof: &Self::Proof,
    ) -> Result<(), FinalizerError>;
}

/// Finalizer abstraction for accumulator commitments that are not represented
/// as a Plonky3 [`Mmcs`] commitment.
///
/// This is used by root proof paths where the WARP accumulator is committed
/// with an external layout rather than `ExtensionMmcs`.
pub trait AccumulatorFinalizer<F, EF, Comm, ProverData>
where
    F: Field,
    EF: ExtensionField<F>,
{
    type Proof;

    fn finalize(
        &self,
        instance: &AccumulatorInstance<EF, Comm>,
        witness: &AccumulatorWitness<EF, ProverData>,
    ) -> Result<Self::Proof, FinalizerError>;

    fn verify(
        &self,
        instance: &AccumulatorInstance<EF, Comm>,
        proof: &Self::Proof,
    ) -> Result<(), FinalizerError>;
}
