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
use p3_field::{Algebra, BasedVectorSpace, Field, PrimeField32};
use p3_maybe_rayon::prelude::ParIterExt;
pub use serializing_challenger::*;
use tracing::info;

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
    /// Sample a random `bits`-bit integer from the transcript.
    ///
    /// The distribution should be reasonably close to uniform.
    /// (In practice, a small bias may arise when bit-decomposing a uniformly
    /// sampled field element)
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
        info!("Observing extension field element:",);
        self.observe_slice(alg_elem.as_basis_coefficients_slice());
    }

    /// Sample an element of a vector space over the base field.
    ///
    /// Constructs the element by sampling basis coefficients.
    fn sample_algebra_element<A: BasedVectorSpace<F>>(&mut self) -> A {
        info!("Sampling extension field element:");
        A::from_basis_coefficients_fn(|_| self.sample())
    }

    /// Observe base field elements as extension field elements for recursion-friendly transcripts.
    ///
    /// This simplifies recursive verifier circuits by using a uniform extension field challenger.
    /// Instead of observing a mix of base and extension field elements, we convert all base field
    /// observations (metadata, public values) to extension field elements before passing to the challenger.
    ///
    /// # Recursion Benefits
    ///
    /// In recursive proof systems, the verifier circuit needs to verify the inner proof. Since STARK
    /// verification operates entirely in the extension field (challenges, opened values, constraint
    /// evaluation), having a challenger that only observes extension field elements significantly
    /// simplifies the recursive circuit implementation.
    #[inline]
    fn observe_base_as_algebra_element<EF>(&mut self, val: F)
    where
        EF: Algebra<F> + BasedVectorSpace<F>,
    {
        info!("Observing base value (as extension element): {:?}", val);
        self.observe_algebra_element(EF::from(val));
    }

    fn check_witness_algebra<A: BasedVectorSpace<F>>(&mut self, bits: usize, witness: A) -> bool {
        self.observe_algebra_element(witness);
        let res = self.sample_bits(bits) == 0;
        // Sample more to correctly update the challenger.
        for _ in 0..A::DIMENSION - 1 {
            self.sample_bits(0);
        }
        res
    }

    fn grind_algebra<A: BasedVectorSpace<F>>(mut self, bits: usize) -> A
    where
        Self: Clone,
    {
        assert!(bits < (usize::BITS as usize));
        // assert!((1 << bits) < F::ORDER_U64);
        info!("Grinding, ignore these...");
        let good_index = (0..1 << 32)
            .find_any(|&i| {
                let base_field_element = F::from_usize(i);
                let witness = A::from_basis_coefficients_fn(|j| {
                    if j == 0 { base_field_element } else { F::ZERO }
                });
                // Clone the challenger itself, not the mutable reference
                let mut cloned_challenger = self.clone();
                cloned_challenger.check_witness_algebra(bits, witness)
            })
            .expect("failed to find witness");

        let base_field_element = F::from_usize(good_index);
        let good_witness_assert =
            A::from_basis_coefficients_fn(|j| if j == 0 { base_field_element } else { F::ZERO });
        assert!(self.check_witness_algebra(bits, good_witness_assert));
        info!("Grinding done, witness found.");
        A::from_basis_coefficients_fn(|j| if j == 0 { base_field_element } else { F::ZERO })
    }
}

impl<C, T> CanObserve<T> for &mut C
where
    C: CanObserve<T>,
{
    #[inline(always)]
    fn observe(&mut self, value: T) {
        (*self).observe(value);
    }

    #[inline(always)]
    fn observe_slice(&mut self, values: &[T])
    where
        T: Clone,
    {
        (*self).observe_slice(values);
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
        (*self).observe_algebra_element(ext);
    }

    #[inline(always)]
    fn sample_algebra_element<EF: BasedVectorSpace<F>>(&mut self) -> EF {
        (*self).sample_algebra_element()
    }

    #[inline(always)]
    fn observe_base_as_algebra_element<EF>(&mut self, val: F)
    where
        EF: Algebra<F> + BasedVectorSpace<F>,
    {
        (*self).observe_base_as_algebra_element::<EF>(val);
    }
}
