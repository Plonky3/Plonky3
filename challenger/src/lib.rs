//! Utilities for generating Fiat-Shamir challenges based on an IOP's transcript.

#![no_std]

extern crate alloc;

pub mod duplex_challenger;
pub mod hash_challenger;

use p3_field::field::{Field, FieldExtension};

/// Observes prover messages during an IOP, and generates Fiat-Shamir challenges in response.
pub trait Challenger<F: Field> {
    fn observe_element(&mut self, element: F);

    fn observe_elements(&mut self, elements: &[F]) {
        for &elt in elements {
            self.observe_element(elt);
        }
    }

    fn observe_ext_element<FE: FieldExtension<F>>(&mut self, ext: FE) {
        self.observe_elements(ext.as_base_slice());
    }

    fn random_element(&mut self) -> F;

    fn random_array<const N: usize>(&mut self) -> [F; N] {
        core::array::from_fn(|_| self.random_element())
    }
}
