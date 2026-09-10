#![doc = include_str!("../README.md")]
#![no_std]

extern crate alloc;

// Only the transcript's unwind test reaches for `std`, and only where unwinding exists.
#[cfg(all(test, panic = "unwind"))]
extern crate std;

mod config;
mod hiding_pcs;
mod pcs_transcript;
mod periodic;
mod proof;
pub mod prover;
mod transcript;
mod two_adic_pcs;
pub mod verifier;

pub use config::*;
pub use hiding_pcs::*;
pub use pcs_transcript::{
    PcsProverTranscript, PcsShape, PcsTranscriptFailure, PcsVerifierTranscript,
};
pub use periodic::*;
pub use proof::*;
pub use transcript::{FriShape, ProverTranscript, TranscriptFailure, VerifierTranscript};
pub use two_adic_pcs::*;
