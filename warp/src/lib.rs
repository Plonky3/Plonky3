//! WARP: linear-time accumulation scheme over Reed-Solomon codes.
//!
//! Reference: <https://eprint.iacr.org/2025/753>
//!
//! This crate implements the WARP accumulation scheme (Construction 10.4)
//! specialised to DFT Reed-Solomon codes and instance-free Plonky3
//! [`Air`](p3_air::Air) constraint systems (`κ = 0` in the paper's PESAT
//! notation). The protocol bundles `ℓ` PESAT instances into a
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
#[cfg(feature = "stark-backend")]
extern crate std;

pub mod accumulator;
pub mod bool_pesat_air;
pub mod code;
pub mod error;
pub mod finalize;
pub mod mle_eval_air;
pub mod protocol;
pub mod relation;
pub mod root;
pub mod root_iop;
#[cfg(feature = "stark-backend")]
pub mod root_receipt_air;
pub mod rs_encoding_air;
#[cfg(feature = "stark-backend")]
pub mod stark_backend;
pub mod sumcheck;
pub mod sumcheck_air;
pub mod transcript;
pub mod whir_compiler;

pub use accumulator::{
    Accumulator, AccumulatorInstance, AccumulatorWitness, WarpProof, WarpProofCommitted,
    WarpProofExternal, WarpProofExternalBatched,
};
pub use bool_pesat_air::{BinomialBoolPesatAir, binomial_bool_pesat_air_trace};
#[cfg(feature = "stark-backend")]
pub use bool_pesat_air::{
    binomial_bool_pesat_air, binomial_bool_pesat_air_context, binomial_bool_pesat_proving_context,
};
pub use code::{ReedSolomonCode, ReedSolomonEncoding};
pub use error::{DeciderError, FinalizerError, VerifierError, WarpError};
pub use finalize::{
    AccumulatorFinalizer, AccumulatorPointOpeningBackend, BackendWitnessFinalizer,
    ExtensionLimbPcs, ExtensionLimbPcsError, ExtensionLimbPcsProof, ExtensionLimbPcsProverData,
    Finalizer, LocalDeciderFinalizer, PrecommittedAccumulatorPcs, WhirAccumulatorOpeningProof,
    WhirAccumulatorOpeningProtocol, WhirAirPesatProof, WhirAirPesatProtocol,
    WhirBooleanPesatProtocol, WhirBooleanWarpFinalizerProtocol, WhirCodewordBackend,
    WhirCommittedCodeword, WhirCurrentRowPesatProof, WhirCurrentRowPesatProtocol,
    WhirLimbAccumulatorBackend, WhirLimbAccumulatorOpeningProof, WhirLimbAccumulatorProverData,
    WhirPrecommittedBooleanWarpFinalizerProtocol, WhirPrecommittedWarpFinalizerProtocol,
    WhirWarpFinalizerProof, WhirWarpFinalizerProtocol, WitnessFinalizer, WitnessProof,
};
pub use mle_eval_air::{BinomialMleEvalAir, binomial_mle_eval_air_trace};
#[cfg(feature = "stark-backend")]
pub use mle_eval_air::{
    binomial_mle_eval_air, binomial_mle_eval_air_context, binomial_mle_eval_proving_context,
};
pub use protocol::{
    AccumulatorBatchOpeningBackend, AccumulatorCommitmentBackend, CommittedCodeword,
    ExternalCodewordBatchOpeningProver, ExternalCodewordBatchOpeningVerifier,
    ExternalCodewordOpeningProver, ExternalCodewordOpeningVerifier, ExternalCommitmentObserver,
    ExternalCommittedCodeword, MmcsExternalOpeningVerifier, WarpDecider, WarpParams, WarpProver,
    WarpStepSumcheckAirWitness, WarpVerifier,
};
pub use relation::{AirAsPesat, BooleanPesat, BundledPesat, PesatShape};
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
#[cfg(feature = "stark-backend")]
pub use root_receipt_air::{
    WarpRootReceiptAir, root_receipt_air, root_receipt_air_context, root_receipt_proving_context,
};
pub use rs_encoding_air::{BinomialRsEncodingAir, binomial_rs_encoding_air_trace};
#[cfg(feature = "stark-backend")]
pub use rs_encoding_air::{
    binomial_rs_encoding_air, binomial_rs_encoding_air_context,
    binomial_rs_encoding_proving_context,
};
#[cfg(feature = "stark-backend")]
pub use stark_backend::{
    CpuCommittedProvingContext, StarkBackendAccumulatorBackend, StarkBackendAccumulatorClaim,
    StarkBackendAccumulatorOpeningProof, StarkBackendAccumulatorProverData,
    StarkBackendExternalOpeningError, StarkBackendOpeningBackend, StarkBackendOpeningProof,
    StarkBackendOpeningVerifier, StarkBackendSegment, StarkBackendSegmentClaim,
    StarkBackendStackedColumn, StarkBackendStackedLayout, StarkBackendTraceVData,
};
pub use sumcheck_air::{BinomialSumcheckAir, binomial_sumcheck_air_trace};
#[cfg(feature = "stark-backend")]
pub use sumcheck_air::{
    binomial_sumcheck_air, binomial_sumcheck_air_context, binomial_sumcheck_proving_context,
};
pub use whir_compiler::{
    NativeWarpWhirClaimCompileError, NativeWarpWhirCompiler, NativeWarpWhirCompilerError,
    NativeWarpWhirEvalClaim, NativeWarpWhirOracleStatement, NativeWarpWhirPointProof,
    NativeWarpWhirRootBaseProverData, NativeWarpWhirRootCommitment,
    NativeWarpWhirRootOracleOpeningProof, NativeWarpWhirRootOracleProverData,
    NativeWarpWhirRootOracleReductionProof, NativeWarpWhirRootProof, NativeWarpWhirRootProofError,
    NativeWarpWhirRootProofSystem, NativeWarpWhirRootProverData, NativeWarpWhirRootReductionError,
    NativeWarpWhirRootReductionProof, NativeWarpWhirRootResidualClaim, eval_claims_from_parts,
};
