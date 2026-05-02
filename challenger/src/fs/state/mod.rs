//! Prover-side and verifier-side drivers that consume a recorded pattern in lockstep with a sponge.

pub mod prover;
pub mod verifier;

pub use prover::ProverState;
pub use verifier::VerifierState;
