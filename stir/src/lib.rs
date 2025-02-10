//! An implementation of the STIR low-degree test (LDT).
//! https://eprint.iacr.org/2024/390.

// NP TODO re-introduce no_std
// #![no_std]

// NP TODOs
// - Credit Giacomo and link to his code
// - Think about MMCS
// - Batching (fold multiple words)
// - Protocol builder

// NP TODO profusely add documentation

// NP TODO remove
// - optimisations for fold_evaluations
// - optimisations for oracle computation
// - More tests

extern crate alloc;

mod config;
mod proof;
pub mod prover;
mod proximity_gaps;
mod utils;
pub mod verifier;

#[cfg(any(test, feature = "test-utils"))]
pub mod test_utils;

pub use config::{StirConfig, StirParameters};
pub use proof::StirProof;
pub use proximity_gaps::*;

// Used for domain separation in the Fiat-Shamir transcript
pub(crate) enum Messages {
    Commitment,
    RoundCommitment,
    FoldingRandomness,
    OodSamples,
    Betas,
    CombRandomness,
    QueryIndices,
    AnsPolynomial,
    ShakePolynomial,
    ShakeRandomness,
    FinalPolynomial,
    FinalQueryIndices,
}
