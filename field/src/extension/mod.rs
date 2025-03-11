use core::{debug_assert, debug_assert_eq, iter};

use crate::field::Field;
use crate::{ExtensionField, naive_poly_mul, prime_factorization};

mod binomial_extension;
mod complex;
mod packed_binomial_extension;

use alloc::vec;
use alloc::vec::Vec;

pub use binomial_extension::*;
pub use complex::*;
use num_bigint::BigUint;
use num_traits::One;
pub use packed_binomial_extension::*;

/// Binomial extension field trait.
///
/// This exists if the polynomial ring `F[X]` has an irreducible polynomial `X^d-W`
/// allowing us to define the binomial extension field `F[X]/(X^d-W)`.
pub trait BinomiallyExtendable<const D: usize>: Field {
    const W: Self;

    // DTH_ROOT = W^((n - 1)/D).
    // n is the order of base field.
    // Only works when exists k such that n = kD + 1.
    const DTH_ROOT: Self;

    const EXT_GENERATOR: [Self; D];

    /// A list of (factor, exponent) pairs.
    ///
    /// If a list is not provided, we attempt to compute it but this may
    /// be slow for large extensions.
    fn extension_multiplicative_group_factors() -> Vec<(BigUint, usize)> {
        let multiplicative_order = Self::order().pow(D as u32) - BigUint::one();
        prime_factorization(&multiplicative_order)
    }
}

pub trait HasFrobenius<F: Field>: ExtensionField<F> {
    fn frobenius(&self) -> Self;
    fn repeated_frobenius(&self, count: usize) -> Self;
    fn frobenius_inv(&self) -> Self;

    fn minimal_poly(mut self) -> Vec<F> {
        let mut m = vec![Self::ONE];
        for _ in 0..Self::DIMENSION {
            m = naive_poly_mul(&m, &[-self, Self::ONE]);
            self = self.frobenius();
        }
        let mut m_iter = m
            .into_iter()
            .map(|c| c.as_base().expect("Extension is not algebraic?"));
        let m: Vec<F> = m_iter.by_ref().take(Self::DIMENSION + 1).collect();
        debug_assert_eq!(m.len(), Self::DIMENSION + 1);
        debug_assert_eq!(m.last(), Some(&F::ONE));
        debug_assert!(m_iter.all(|c| c.is_zero()));
        m
    }

    fn galois_group(self) -> Vec<Self> {
        iter::successors(Some(self), |x| Some(x.frobenius()))
            .take(Self::DIMENSION)
            .collect()
    }
}

/// Optional trait for implementing Two Adic Binomial Extension Field.
pub trait HasTwoAdicBinomialExtension<const D: usize>: BinomiallyExtendable<D> {
    const EXT_TWO_ADICITY: usize;

    /// Assumes the multiplicative group size has at least `bits` powers of two, otherwise the
    /// behavior is undefined.
    fn ext_two_adic_generator(bits: usize) -> [Self; D];
}
