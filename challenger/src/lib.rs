//! Utilities for generating Fiat-Shamir challenges based on an IOP's transcript.

#![no_std]

extern crate alloc;

mod duplex_challenger;
mod hash_challenger;

use alloc::vec::Vec;
use core::mem::size_of;

pub use duplex_challenger::*;
pub use hash_challenger::*;
use p3_field::{AbstractExtensionField, Field, PrimeField64};

/// Observes prover messages during an IOP, and generates Fiat-Shamir challenges in response.
pub trait Challenger<F: Field> {
    fn observe_element(&mut self, element: F);

    fn observe_elements(&mut self, elements: &[F]) {
        for &elt in elements {
            self.observe_element(elt);
        }
    }

    fn observe_ext_element<EF: AbstractExtensionField<F>>(&mut self, ext: EF) {
        self.observe_elements(ext.as_base_slice());
    }

    fn random_element(&mut self) -> F;

    fn random_usize(&mut self, bits: usize) -> usize
    where
        F: PrimeField64,
    {
        debug_assert!(bits < size_of::<usize>());
        let rand_f = self.random_element();
        let rand_usize = rand_f.as_canonical_u64() as usize;
        rand_usize & ((1 << bits) - 1)
    }

    fn random_ext_element<EF: AbstractExtensionField<F>>(&mut self) -> EF {
        let vec = self.random_vec(EF::D);
        EF::from_base_slice(&vec)
    }

    fn random_array<const N: usize>(&mut self) -> [F; N] {
        core::array::from_fn(|_| self.random_element())
    }

    fn random_vec(&mut self, n: usize) -> Vec<F> {
        (0..n).map(|_| self.random_element()).collect()
    }
}
