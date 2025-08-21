use core::iter;

use crate::field::Field;
use crate::{Algebra, ExtensionField};

mod binomial_extension;
mod complex;
mod packed_binomial_extension;

use alloc::vec::Vec;

pub use binomial_extension::*;
pub use complex::*;
pub use packed_binomial_extension::*;

/// Trait for fields that support binomial extension of the form `F[X]/(X^D - W)`.
///
/// A type implementing this trait can define a degree-`D` extension field using an
/// irreducible binomial polynomial `X^D - W`, where `W` is a nonzero constant in the base field.
///
/// This is used to construct extension fields with efficient arithmetic.
pub trait BinomiallyExtendable<const D: usize>:
    Field + BinomiallyExtendableAlgebra<Self, D>
{
    /// The constant coefficient `W` in the binomial `X^D - W`.
    const W: Self;

    /// A `D`-th root of unity derived from `W`.
    ///
    /// This is `W^((n - 1)/D)`, where `n` is the order of the field.
    /// Valid only when `n = kD + 1` for some `k`.
    const DTH_ROOT: Self;

    /// A generator for the extension field, expressed as a degree-`D` polynomial.
    ///
    /// This is an array of size `D`, where each entry is a base field element.
    const EXT_GENERATOR: [Self; D];
}

/// Trait for algebras which support binomial extensions of the form `A[X]/(X^D - W)`
/// with `W` in the base field `F`.
pub trait BinomiallyExtendableAlgebra<F: Field, const D: usize>: Algebra<F> {
    /// Multiplication in the algebra extension ring `A<X> / (X^D - W)`.
    ///
    /// Some algebras may want to reimplement this with faster methods.
    #[inline]
    fn binomial_mul(a: &[Self; D], b: &[Self; D], res: &mut [Self; D], w: F) {
        binomial_mul::<F, Self, Self, D>(a, b, res, w);
    }

    /// Addition of elements in the algebra extension ring `A<X> / (X^D - W)`.
    ///
    /// As addition has no dependence on `W` so this is equivalent
    /// to an algorithm for adding arrays of elements of `A`.
    ///
    /// Some algebras may want to reimplement this with faster methods.
    #[inline]
    #[must_use]
    fn binomial_add(a: &[Self; D], b: &[Self; D]) -> [Self; D] {
        vector_add(a, b)
    }

    /// Subtraction of elements in the algebra extension ring `A<X> / (X^D - W)`.
    ///
    /// As subtraction has no dependence on `W` so this is equivalent
    /// to an algorithm for subtracting arrays of elements of `A`.
    ///
    /// Some algebras may want to reimplement this with faster methods.
    #[inline]
    #[must_use]
    fn binomial_sub(a: &[Self; D], b: &[Self; D]) -> [Self; D] {
        vector_sub(a, b)
    }

    #[inline]
    fn binomial_base_mul(lhs: [Self; D], rhs: Self) -> [Self; D] {
        lhs.map(|x| x * rhs.clone())
    }
}

/// Trait for extension fields that support Frobenius automorphisms.
///
/// The Frobenius automorphism is a field map `x ↦ x^n`,
/// where `n` is the order of the base field.
///
/// This map is an automorphism of the field that fixes the base field.
pub trait HasFrobenius<F: Field>: ExtensionField<F> {
    /// Apply the Frobenius automorphism once.
    ///
    /// Equivalent to raising the element to the `n`th power.
    #[must_use]
    fn frobenius(&self) -> Self;

    /// Apply the Frobenius automorphism `count` times.
    ///
    /// Equivalent to raising to the `n^count` power.
    #[must_use]
    fn repeated_frobenius(&self, count: usize) -> Self;

    /// Compute the inverse Frobenius map.
    ///
    /// Returns the unique element `y` such that `self = y^n`.
    #[must_use]
    fn frobenius_inv(&self) -> Self;

    /// Returns the full Galois orbit of the element under Frobenius.
    ///
    /// This is the sequence `[x, x^n, x^{n^2}, ..., x^{n^{D-1}}]`,
    /// where `D` is the extension degree.
    #[must_use]
    fn galois_orbit(self) -> Vec<Self> {
        iter::successors(Some(self), |x| Some(x.frobenius()))
            .take(Self::DIMENSION)
            .collect()
    }
}

/// Trait for binomial extensions that support a two-adic subgroup generator.
pub trait HasTwoAdicBinomialExtension<const D: usize>: BinomiallyExtendable<D> {
    /// Two-adicity of the multiplicative group of the extension field.
    ///
    /// This is the number of times 2 divides the order of the field minus 1.
    const EXT_TWO_ADICITY: usize;

    /// Returns a two-adic generator for the extension field.
    ///
    /// This is used to generate the 2^bits-th roots of unity in the extension field.
    /// Behavior is undefined if `bits > EXT_TWO_ADICITY`.
    #[must_use]
    fn ext_two_adic_generator(bits: usize) -> [Self; D];
}
