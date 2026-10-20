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
mod point;
mod skip;
mod stream;
mod witness;
pub mod zerocheck;

#[cfg(test)]
mod tests;

pub use batch::{Batch, BatchError};
pub use circuit::{Builder, Circuit, Wire, Word};
pub use point::QuirkyPoint;
pub use witness::Witness;

/// The committed field: a witness packs 64 bits per element.
pub type Val = p3_binary_field::Poly64;

/// The challenge field, a cubic extension of the committed field.
pub type Challenge = p3_binary_field::Poly192;
