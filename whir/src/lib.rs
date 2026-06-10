//! WHIR: Reed-Solomon proximity testing with super-fast verification.
//!
//! An IOP of proximity for constrained Reed-Solomon codes that serves as
//! a multilinear polynomial commitment scheme.
//!
//! A hiding variant lives in the zero-knowledge PCS module.
//! Masked sumcheck batches, HVZK code-switching rounds, and a masked base
//! case compose into a commitment that reveals only the requested
//! evaluations.
//!
//! References:
//! - <https://eprint.iacr.org/2024/1586> (WHIR),
//! - <https://eprint.iacr.org/2026/391> (HVZK-WHIR).

#![no_std]

extern crate alloc;

pub mod fiat_shamir;
pub mod parameters;
pub mod pcs;
pub(crate) mod utils;

pub use fiat_shamir::domain_separator::DomainSeparator;
pub use p3_sumcheck::{self as sumcheck, constraints};
pub use parameters::{
    DEFAULT_MAX_POW, FoldingFactor, FoldingFactorError, ProtocolParameters, RoundConfig,
    SecurityAssumption, WhirConfig, WhirConfigError,
};
pub use pcs::WhirProverData;
pub use pcs::proof::{PcsProof, QueryOpening, WhirProof, WhirRoundProof};
pub use pcs::prover::WhirProver;
pub use pcs::verifier::WhirVerifier;
pub use pcs::verifier::errors::VerifierError;
pub use pcs::zk::{
    BaseCaseZkError, BaseCaseZkProof, BlindedMask, CodeSwitchError, HidingWhirPcs,
    HidingWhirProverData, MaskCodeShape, MaskGroupShape, MaskOpeningPair, ZkConfigError,
    ZkParameters, ZkRoundProof, ZkVerifierError, ZkWhirConfig, ZkWhirProof,
};
