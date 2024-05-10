//! Utilities for generating Fiat-Shamir challenges based on an IOP's transcript.

#![no_std]

extern crate alloc;

mod duplex_challenger;
mod grinding_challenger;
mod hash_challenger;
mod multi_field_challenger;
mod serializing_challenger;

use alloc::vec::Vec;
use core::array;

pub use duplex_challenger::*;
pub use grinding_challenger::*;
pub use hash_challenger::*;
pub use multi_field_challenger::*;
use p3_field::{AbstractExtensionField, Field};
pub use serializing_challenger::*;

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

pub trait FieldChallenger<F: Field>:
    CanObserve<F> + CanSample<F> + CanSampleBits<usize> + Sync
{
    fn observe_ext_element<EF: AbstractExtensionField<F>>(&mut self, ext: EF) {
        self.observe_slice(ext.as_base_slice());
    }

    fn sample_ext_element<EF: AbstractExtensionField<F>>(&mut self) -> EF {
        let vec = self.sample_vec(EF::D);
        EF::from_base_slice(&vec)
    }
}

impl<'a, C, T> CanObserve<T> for &'a mut C
where
    C: CanObserve<T>,
{
    #[inline(always)]
    fn observe(&mut self, value: T) {
        (**self).observe(value)
    }

    #[inline(always)]
    fn observe_slice(&mut self, values: &[T])
    where
        T: Clone,
    {
        (**self).observe_slice(values)
    }
}

impl<'a, C, T> CanSample<T> for &'a mut C
where
    C: CanSample<T>,
{
    #[inline(always)]
    fn sample(&mut self) -> T {
        (**self).sample()
    }

    #[inline(always)]
    fn sample_array<const N: usize>(&mut self) -> [T; N] {
        (**self).sample_array()
    }

    #[inline(always)]
    fn sample_vec(&mut self, n: usize) -> Vec<T> {
        (**self).sample_vec(n)
    }
}

impl<'a, C, T> CanSampleBits<T> for &'a mut C
where
    C: CanSampleBits<T>,
{
    #[inline(always)]
    fn sample_bits(&mut self, bits: usize) -> T {
        (**self).sample_bits(bits)
    }
}

impl<'a, C, F: Field> FieldChallenger<F> for &'a mut C
where
    C: FieldChallenger<F>,
{
    #[inline(always)]
    fn observe_ext_element<EF: AbstractExtensionField<F>>(&mut self, ext: EF) {
        (**self).observe_ext_element(ext)
    }

    #[inline(always)]
    fn sample_ext_element<EF: AbstractExtensionField<F>>(&mut self) -> EF {
        (**self).sample_ext_element()
    }
}
