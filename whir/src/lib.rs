#![doc = include_str!("../README.md")]
#![no_std]

extern crate alloc;

// Only the transcript's unwind tests reach for `std`, and only where unwinding exists.
#[cfg(all(test, panic = "unwind"))]
extern crate std;

pub mod parameters;
pub mod pcs;
pub mod transcript;
pub(crate) mod utils;

pub use parameters::{
    DEFAULT_MAX_POW, FoldingFactor, FoldingFactorError, ProtocolParameters, RoundConfig,
    SecurityAssumption, WhirConfig, WhirConfigError,
};
pub use pcs::WhirProverData;
pub use pcs::proof::{PcsProof, QueryOpenings, SharedProofOpening, WhirProof, WhirRoundProof};
pub use pcs::prover::WhirProver;
pub use pcs::verifier::WhirVerifier;
pub use pcs::verifier::errors::VerifierError;
pub use pcs::zk::{
    BaseCaseZkError, BaseCaseZkProof, BlindedMask, CodeSwitchError, HidingBaseCaseSecurityReport,
    HidingBoundClassification, HidingCodeRole, HidingCodeSecurityReport, HidingErrorBound,
    HidingGammaRoundReport, HidingListBound, HidingQueryRoundReport, HidingWhirPcs,
    HidingWhirProverData, MaskCodeShape, MaskGroupShape, MaskOpeningPair, ZkConfigError,
    ZkParameters, ZkRoundProof, ZkVerifierError, ZkWhirConfig, ZkWhirProof,
};
pub use transcript::WhirShape;
pub use transcript::zk::{ZkBaseCaseShape, ZkWhirShape};
