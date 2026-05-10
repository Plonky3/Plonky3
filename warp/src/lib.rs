//! WARP: linear-time accumulation scheme over Reed-Solomon codes.
//!
//! Reference: <https://eprint.iacr.org/2025/753>
//!
//! This crate implements the WARP accumulation scheme (Construction 10.4)
//! specialised to DFT Reed-Solomon codes and direct Boolean PESAT relations.
//! The protocol bundles `ℓ` PESAT instances into a
//! single accumulator carrying:
//!
//! ```text
//!     acc.x = (rt, α, µ, β, η)        — verifier-visible part
//!     acc.w = (td, f, w)              — prover-side witness
//! ```
//!
//! Each accumulation step runs the combined IOR of Construction 9.4
//! (twin-constraint pseudo-batching § 6.3 + codeword batching § 7.2 +
//! multilinear-constraint batching § 8.2). The
//! [`decider`](protocol::WarpDecider) re-checks `f̂(α) == µ`,
//! `Pb(β, w) == η`, and `f == C(w)`.

#![no_std]

extern crate alloc;

pub mod accumulator;
pub mod code;
pub mod error;
pub mod finalize;
pub mod protocol;
pub mod relation;
pub mod root;
pub mod root_iop;
pub mod sumcheck;
pub mod transcript;
pub mod whir_compiler;

pub use accumulator::{
    Accumulator, AccumulatorInstance, AccumulatorWitness, WarpProof, WarpProofCommitted,
    WarpProofExternal, WarpProofExternalBatched,
};
pub use code::{ReedSolomonCode, ReedSolomonEncoding};
pub use error::{DeciderError, FinalizerError, VerifierError, WarpError};
pub use finalize::{
    AccumulatorFinalizer, AccumulatorPointOpeningBackend, Finalizer, LocalDeciderFinalizer,
    PrecommittedAccumulatorPcs, WhirAccumulatorOpeningProof, WhirAccumulatorOpeningProtocol,
    WhirBooleanPesatProtocol, WhirBooleanWarpFinalizerProtocol, WhirPesatProof,
    WhirPrecommittedBooleanWarpFinalizerProtocol, WhirWarpFinalizerProof, WitnessFinalizer,
    WitnessProof,
};
pub use protocol::{
    AccumulatorBatchOpeningBackend, AccumulatorCommitmentBackend, CommittedCodeword,
    ExternalCodewordBatchOpeningProver, ExternalCodewordBatchOpeningVerifier,
    ExternalCodewordOpeningProver, ExternalCodewordOpeningVerifier, ExternalCommitmentObserver,
    ExternalCommittedCodeword, MmcsExternalOpeningVerifier, WarpDecider, WarpParams, WarpProver,
    WarpVerifier,
};
pub use relation::{BooleanPesat, BundledPesat, PesatShape};
pub use root::{
    WarpExternalRootClaim, WarpExternalRootProof, WarpExternalRootProofBatched,
    WarpExternalRootReceipt, WarpExternalRootStep, WarpExternalRootStepBatched, WarpRootClaim,
    WarpRootProof, WarpRootProver, WarpRootReceipt, WarpRootStep, WarpRootVerifier,
    WitnessRootProof,
};
pub use root_iop::{
    RootIopBoundAccumulatorProverData, RootIopBoundCommitment, RootIopBoundCommittedCodeword,
    RootIopBoundProofSystem, RootIopBoundProver, RootIopBoundTranscript, RootIopBoundVerifier,
    RootIopCommitment, RootIopCommittedCodeword, RootIopError, RootIopOpeningClaim,
    RootIopOpeningPoint, RootIopOpeningProof, RootIopOpeningValue, RootIopOracleField,
    RootIopOracleValues, RootIopProofSystem, RootIopProver, RootIopTranscript, RootIopVerifier,
    RootIopWarpFinalizerProof, RootIopWarpRootProof, WitnessRootIopBoundProof, WitnessRootIopProof,
};
pub use whir_compiler::{
    NativeWarpWhirClaimCompileError, NativeWarpWhirCompiler, NativeWarpWhirEvalClaim,
    NativeWarpWhirOracleStatement, NativeWarpWhirRootBaseProverData, NativeWarpWhirRootCommitment,
    NativeWarpWhirRootOracleProverData, NativeWarpWhirRootProof, NativeWarpWhirRootProofError,
    NativeWarpWhirRootProofSystem, NativeWarpWhirRootProverData, NativeWarpWhirRootReductionError,
    eval_claims_from_parts,
};
