#![doc = include_str!("../README.md")]
#![no_std]

extern crate alloc;

use p3_sumcheck::layout::SuffixProver;

mod boolean;
mod boolean_trace;
mod boolean_trace_transcript;
mod error;
mod fold;
mod grouped_mmcs;
mod mixed_trace;
mod packing;
mod params;
mod pcs;
mod proof;
mod prover;
#[cfg(test)]
mod security_tests;
#[cfg(test)]
pub(crate) mod test_util;
pub mod transcript;
mod verifier;
pub mod whir;

pub use boolean::{
    BitOpening, BitReadings, BooleanBackend, BooleanMultilinearPcs, BooleanPcs, BooleanPcsError,
    BooleanProof,
};
pub use boolean_trace::{
    BooleanTraceCommitment, BooleanTraceCommitmentData, BooleanTraceCommitmentError,
    BooleanTraceCommitmentProof, BooleanTraceData, BooleanTraceError, BooleanTracePcs,
    BooleanTraceProof,
};
pub use error::BinaryPcsError;
pub use fold::{ChallengeField, FoldAlphabet, fold_codeword, fold_pair};
pub use grouped_mmcs::GroupedCodewordMmcs;
pub use mixed_trace::{
    MixedTraceCommitment, MixedTraceData, MixedTracePcs, committed_shapes, coordinate_basis,
};
pub use packing::{Coordinates, PackError, PackedStack, coordinate_bytes, pack, unpack};
pub use params::{BinaryPcsConfig, BinaryPcsConfigError, BinaryPcsParams};
pub use pcs::BinaryPcs;
pub use proof::{BinaryPcsProof, RoundProof};
pub use prover::BinaryPcsProverData;

/// The stacked-layout binding mode this scheme commits in.
///
/// Fixed rather than chosen by the caller.
/// The codeword fold merges adjacent pairs, which only suffix-order binding matches.
///
/// Prefix binding merges halves, and driving it through the same fold falls out of lockstep.
/// The prover's own tests pin that as a property of the two orders, not a tuning detail.
pub(crate) type PcsLayout<F, EF> = SuffixProver<F, EF>;
