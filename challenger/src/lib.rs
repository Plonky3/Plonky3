//! Utilities for generating Fiat-Shamir challenges based on an IOP's transcript.

#![no_std]
#![allow(incomplete_features)]

extern crate alloc;

pub mod duplex_challenger;
pub mod hash_challenger;

use p3_field::field::Field;

/// Observes prover messages during an IOP, and generates Fiat-Shamir challenges in response.
pub trait Challenger<F: Field> {
    // Could use Field::map_components() to get functionality similar
    // to observe_extension_element(), though as mentioned there, it's
    // better to avoid depending on the individual components of the
    // representation of a field element.
    fn observe_element(&mut self, element: F);

    fn random_element(&mut self) -> F;

    fn random_array<const N: usize>(&mut self) -> [F; N] {
        core::array::from_fn(|_| self.random_element())
    }
}
