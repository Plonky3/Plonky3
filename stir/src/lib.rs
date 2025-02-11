//! An implementation of the STIR low-degree test. Based on the article
//! https://eprint.iacr.org/2024/390 and the
//! [implementation](https://github.com/WizardOfMenlo/stir) by the co-author
//! Giacomo Fenzi.

// NP TODO re-introduce no_std
// #![no_std]

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
