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
use p3_field::{BasedVectorSpace, Field};
pub use serializing_challenger::*;

/// A generic trait for absorbing elements into the transcript.
///
/// Absorbed elements update the internal sponge state,
/// preparing it to deterministically produce future challenges.
pub trait CanObserve<T> {
    /// Absorb a single value into the transcript.
    fn observe(&mut self, value: T);

    /// Absorb a slice of values into the transcript.
    fn observe_slice(&mut self, values: &[T])
    where
        T: Clone,
    {
        for value in values {
            self.observe(value.clone());
        }
    }
}

/// A trait for sampling challenge elements from the Fiat-Shamir transcript.
///
/// Sampling produces pseudo-random elements deterministically derived
/// from the absorbed inputs and the sponge state.
pub trait CanSample<T> {
    /// Sample a single challenge value from the transcript.
    fn sample(&mut self) -> T;

    /// Sample an array of `N` challenge values from the transcript.
    fn sample_array<const N: usize>(&mut self) -> [T; N] {
        array::from_fn(|_| self.sample())
    }

    /// Sample a `Vec` of `n` challenge values from the transcript.
    fn sample_vec(&mut self, n: usize) -> Vec<T> {
        (0..n).map(|_| self.sample()).collect()
    }
}

/// A trait for sampling random bitstrings from the Fiat-Shamir transcript.
pub trait CanSampleBits<T> {
    /// Sample a uniformly random `bits`-bit integer from the transcript.
    ///
    /// Guarantees that the returned value fits within the requested bit width.
    fn sample_bits(&mut self, bits: usize) -> T;
}

/// A high-level trait combining observation and sampling over a finite field.
pub trait FieldChallenger<F: Field>:
    CanObserve<F> + CanSample<F> + CanSampleBits<usize> + Sync
{
    /// Absorb an element from a vector space over the base field.
    ///
    /// Decomposes the element into its basis coefficients and absorbs each.
    fn observe_algebra_element<A: BasedVectorSpace<F>>(&mut self, alg_elem: A) {
        self.observe_slice(alg_elem.as_basis_coefficients_slice());
    }

    /// Sample an element of a vector space over the base field.
    ///
    /// Constructs the element by sampling basis coefficients.
    fn sample_algebra_element<A: BasedVectorSpace<F>>(&mut self) -> A {
        A::from_basis_coefficients_fn(|_| self.sample())
    }
}

impl<C, T> CanObserve<T> for &mut C
where
    C: CanObserve<T>,
{
    #[inline(always)]
    fn observe(&mut self, value: T) {
        (*self).observe(value)
    }

    #[inline(always)]
    fn observe_slice(&mut self, values: &[T])
    where
        T: Clone,
    {
        (*self).observe_slice(values)
    }
}

impl<C, T> CanSample<T> for &mut C
where
    C: CanSample<T>,
{
    #[inline(always)]
    fn sample(&mut self) -> T {
        (*self).sample()
    }

    #[inline(always)]
    fn sample_array<const N: usize>(&mut self) -> [T; N] {
        (*self).sample_array()
    }

    #[inline(always)]
    fn sample_vec(&mut self, n: usize) -> Vec<T> {
        (*self).sample_vec(n)
    }
}

impl<C, T> CanSampleBits<T> for &mut C
where
    C: CanSampleBits<T>,
{
    #[inline(always)]
    fn sample_bits(&mut self, bits: usize) -> T {
        (*self).sample_bits(bits)
    }
}

impl<C, F: Field> FieldChallenger<F> for &mut C
where
    C: FieldChallenger<F>,
{
    #[inline(always)]
    fn observe_algebra_element<EF: BasedVectorSpace<F>>(&mut self, ext: EF) {
        (*self).observe_algebra_element(ext)
    }

    #[inline(always)]
    fn sample_algebra_element<EF: BasedVectorSpace<F>>(&mut self) -> EF {
        (*self).sample_algebra_element()
    }
}
