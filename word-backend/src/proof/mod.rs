//! The complete word-level proof, ending at one authenticated Boolean-trace opening.
//!
//! Its committed words are the bits of one Boolean commitment, word-major and bit-minor.

mod error;
mod key;
mod prover;
mod record;
mod relation;
mod transcript;
mod verifier;

pub use error::WordProofError;
pub use key::WordProofKey;
pub use record::{ProvedStatement, WordProof};
