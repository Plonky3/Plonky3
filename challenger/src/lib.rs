//! Utilities for generating Fiat-Shamir challenges based on an IOP's transcript.

#![no_std]

extern crate alloc;

mod duplex_challenger;
mod hash_challenger;

use alloc::vec::Vec;
use core::array;

pub use duplex_challenger::*;
pub use hash_challenger::*;
use p3_field::{AbstractExtensionField, Field};

pub trait CanObserve<T> {
    fn observe(&mut self, value: T);

    fn observe_slice(&mut self, values: &[T])
    where
        T: Clone,
    {
        for value in values {
            self.observe(value.clone());
        }
    }
}

pub trait CanSample<T> {
    fn sample(&mut self) -> T;

    fn sample_array<const N: usize>(&mut self) -> [T; N] {
        array::from_fn(|_| self.sample())
    }

    fn sample_vec(&mut self, n: usize) -> Vec<T> {
        (0..n).map(|_| self.sample()).collect()
    }
}

pub trait CanSampleBits<T> {
    fn sample_bits(&mut self, bits: usize) -> T;
}

pub trait FieldChallenger<F: Field>: CanObserve<F> + CanSample<F> + CanSampleBits<usize> {
    fn observe_ext_element<EF: AbstractExtensionField<F>>(&mut self, ext: EF) {
        self.observe_slice(ext.as_base_slice());
    }

    fn sample_ext_element<EF: AbstractExtensionField<F>>(&mut self) -> EF {
        let vec = self.sample_vec(EF::D);
        EF::from_base_slice(&vec)
    }
}
