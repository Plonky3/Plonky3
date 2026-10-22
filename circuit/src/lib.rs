//! A fixed table machine that verifies Plonky3 proofs inside a proof.
//!
//! A circuit is a list of operations on wires, each wire written once.
//!
//! Tables prove the operations and share every wire over one bus.
//!
//! Verifiers written over [`p3_verifier_arith::Arith`] build circuits through [`CircuitArith`].

#![no_std]

extern crate alloc;

mod builder;
mod chip;
mod circuit;
mod expr;
pub mod recursion;
pub mod tables;
mod verifier;

pub use builder::{CircuitBuilder, Unsatisfied};
pub use chip::{PermutationChip, Poseidon2Chip};
pub use circuit::Circuit;
pub use expr::{Base, Ext};
pub use verifier::*;

/// A wire of a circuit, written by exactly one operation.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Wire(pub(crate) u32);

impl Wire {
    /// Position of the wire in the circuit's value list.
    pub(crate) const fn index(self) -> usize {
        self.0 as usize
    }
}
