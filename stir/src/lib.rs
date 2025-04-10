//! An implementation of the STIR low-degree test. Based on the article
//! <https://eprint.iacr.org/2024/390> and the
//! [implementation](https://github.com/WizardOfMenlo/stir) by the co-author
//! Giacomo Fenzi.

// NP TODO re-introduce
// #![no_std]

extern crate alloc;

mod config;
mod proof;
// NP TODO refactor once one prover is left
pub mod prover;
pub mod prover_eval;
mod proximity_gaps;
mod utils;
mod verifier;

#[cfg(any(test, feature = "test-utils"))]
pub mod test_utils;

pub use config::{StirConfig, StirParameters};
pub use proof::StirProof;
pub use proximity_gaps::SecurityAssumption;
pub use verifier::verify;

// If the configuration requires the prover to compute a proof of work of more
// bits than this limit, calling prove() or verify() will log a WARN message (as
// long as the tracing logger has been init()iated)
const POW_BITS_WARNING: usize = 25;

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
