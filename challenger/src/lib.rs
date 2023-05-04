//! Utilities for generating Fiat-Shamir challenges based on an IOP's transcript.

#![no_std]

extern crate alloc;

pub mod duplex_challenger;
pub mod hash_challenger;

use p3_field::field::{Field, FieldExtension, Field32};

/// Observes prover messages during an IOP, and generates Fiat-Shamir challenges in response.
pub trait Challenger<F: Field> {
    fn observe_element(&mut self, element: F);

    fn observe_ext_element<FE: FieldExtension<F>>(&mut self, ext: FE) {
        for coeff in ext.as_base_slice() {
            self.observe_element(*coeff);
        }
    }

    fn random_element(&mut self) -> F;

    fn random_array<const N: usize>(&mut self) -> [F; N] {
        core::array::from_fn(|_| self.random_element())
    }

    fn random_u32(&mut self, bound: u32) -> u32 where F: Field32 {
        self.random_element().as_canonical_u32() % bound
    }
}
