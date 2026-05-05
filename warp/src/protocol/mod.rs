//! WARP protocol orchestrator: prover, verifier, decider.
//!
//! This module ties together the four constructions of the WARP paper
//! (5.10, 6.3, 7.2, 8.2) into the single accumulation-step protocol of
//! Construction 10.4, specialised to Reed-Solomon codes (`code::ReedSolomonCode`)
//! and PESAT instances (`relation::BundledPesat`).

mod decider;
mod external;
pub mod prover;
mod verifier;

pub use decider::WarpDecider;
pub use external::{
    AccumulatorBatchOpeningBackend, AccumulatorCommitmentBackend,
    ExternalCodewordBatchOpeningProver, ExternalCodewordBatchOpeningVerifier,
    ExternalCodewordOpeningProver, ExternalCodewordOpeningVerifier, ExternalCommitmentObserver,
    ExternalCommittedCodeword, MmcsExternalOpeningVerifier,
};
pub use prover::{CommittedCodeword, ExtProverData, WarpProver};
pub use verifier::WarpVerifier;

/// Soundness parameters for one WARP step.
#[derive(Clone, Copy, Debug)]
pub struct WarpParams {
    /// Number of out-of-domain samples in §7.2 (`s` in the paper).
    pub num_ood: usize,
    /// Number of shift queries in §7.2 (`t` in the paper). Must satisfy
    /// `t ≥ λ / -log(1 − δ)` for soundness; the caller picks based on the
    /// security target and chosen proximity bound.
    pub num_shift_queries: usize,
}

impl WarpParams {
    /// Build WarpParams. Theorem 9.1 of the paper requires `1 + s + t` to
    /// be a power of two; the constructor enforces this.
    pub fn new(num_ood: usize, num_shift_queries: usize) -> Self {
        let r = 1 + num_ood + num_shift_queries;
        assert!(
            r.is_power_of_two(),
            "WarpParams: 1 + s + t = {r} must be a power of two (paper Theorem 9.1)"
        );
        Self {
            num_ood,
            num_shift_queries,
        }
    }

    /// Total number of evaluation claims `r = 1 + s + t` carried into §8.2.
    #[inline]
    pub const fn r(&self) -> usize {
        1 + self.num_ood + self.num_shift_queries
    }

    /// `log_2 r`. By construction `r` is a power of two.
    pub fn log_r(&self) -> usize {
        self.r().trailing_zeros() as usize
    }
}
