//! Flock: batched Boolean R1CS over binary fields.
//!
//! A circuit is a gate list; a batch repeats it on independent instances.
//!
//! Reference: Bünz, Rothblum and Wang, "Flock: Fast Proving for Batch Boolean Computations", ePrint 2026/1329.

#![no_std]

extern crate alloc;
#[cfg(test)]
extern crate std;

mod batch;
mod byteslice;
mod circuit;
mod config;
mod error;
pub mod lincheck;
pub mod opening;
mod point;
mod proof;
mod prover;
mod skip;
mod stream;
mod verifier;
mod witness;
pub mod zerocheck;

#[cfg(test)]
mod tests;

pub use batch::{Batch, BatchError};
pub use circuit::{Builder, Circuit, Wire, Word};
pub use config::{FlockChallenger, FlockConfig, PcsParams};
pub use error::{ProvingError, VerificationError};
pub use point::{QuirkyClaim, QuirkyPoint};
pub use proof::FlockProof;
pub use prover::prove;
pub use verifier::verify;
pub use witness::Witness;

/// The committed field: a witness packs 64 bits per element.
pub type Val = p3_binary_field::Poly64;

/// The challenge field, a cubic extension of the committed field.
pub type Challenge = p3_binary_field::Poly192;
