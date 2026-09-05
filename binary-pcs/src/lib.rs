#![doc = include_str!("../README.md")]
#![no_std]

extern crate alloc;

use p3_binary_field::BinaryField128;
use p3_sumcheck::layout::SuffixProver;

mod error;
mod fold;
mod params;
mod pcs;
mod proof;
mod prover;
#[cfg(test)]
pub(crate) mod test_util;
mod verifier;

pub use error::BinaryPcsError;
pub use fold::{fold_codeword, fold_pair};
pub use params::{BinaryPcsConfig, BinaryPcsConfigError, BinaryPcsParams};
pub use pcs::BinaryPcs;
pub use proof::{BinaryPcsProof, RoundProof};
pub use prover::BinaryPcsProverData;

/// The stacked-layout binding mode this scheme commits in.
///
/// Fixed rather than chosen by the caller: the codeword fold merges *adjacent pairs*, which
/// only suffix-order binding matches. Prefix binding merges halves, and driving it through the
/// same fold does not stay in lockstep — see `prover.rs`'s
/// `prefix_layout_does_not_stay_in_lockstep`, which pins that as a property of the two binding
/// orders rather than a tuning detail.
pub(crate) type PcsLayout = SuffixProver<BinaryField128, BinaryField128>;
