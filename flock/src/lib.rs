//! Flock: batched Boolean R1CS over binary fields.
//!
//! A circuit is a gate list; a batch repeats it on independent instances.
//!
//! Reference: Bünz, Rothblum and Wang, "Flock: Fast Proving for Batch Boolean Computations", ePrint 2026/1329.

#![no_std]

extern crate alloc;
#[cfg(test)]
extern crate std;

mod circuit;
mod witness;

#[cfg(test)]
mod tests;

pub use circuit::{Builder, Circuit, Wire, Word};
pub use witness::Witness;
