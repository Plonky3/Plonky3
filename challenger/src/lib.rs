//! Utilities for generating Fiat-Shamir challenges based on an IOP's transcript.

#![no_std]
#![allow(incomplete_features)]
#![feature(generic_const_exprs)]

extern crate alloc;

pub mod duplex_challenger;
pub mod hash_challenger;

use p3_field::field::{Field, FieldExtension};

/// Observes prover messages during an IOP, and generates Fiat-Shamir challenges in response.
pub trait Challenger<F: Field> {
    fn observe_element(&mut self, element: F);

    fn observe_extension_element<FE: FieldExtension<F>>(&mut self, element: FE)
    where
        [(); FE::D]:,
    {
        for base in element.to_base_array() {
            self.observe_element(base);
        }
    }

    fn random_element(&mut self) -> F;

    fn random_array<const N: usize>(&mut self) -> [F; N] {
        core::array::from_fn(|_| self.random_element())
    }
}
