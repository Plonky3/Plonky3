use core::iter;

use crate::ExtensionField;
use crate::field::Field;

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
pub trait BinomiallyExtendable<const D: usize>: Field {
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
    fn frobenius(&self) -> Self;

    /// Apply the Frobenius automorphism `count` times.
    ///
    /// Equivalent to raising to the `n^count` power.
    fn repeated_frobenius(&self, count: usize) -> Self;

    /// Compute the inverse Frobenius map.
    ///
    /// Returns the unique element `y` such that `self = y^n`.
    fn frobenius_inv(&self) -> Self;

    /// Returns the full Galois orbit of the element under Frobenius.
    ///
    /// This is the sequence `[x, x^n, x^{n^2}, ..., x^{n^{D-1}}]`,
    /// where `D` is the extension degree.
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
    fn ext_two_adic_generator(bits: usize) -> [Self; D];
}
