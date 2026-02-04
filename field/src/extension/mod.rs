use core::iter;

use crate::field::Field;
use crate::{Algebra, ExtensionField};

mod binomial_extension;
mod complex;
mod packed_binomial_extension;
mod packed_quintic_extension;
mod quintic_extension;

use alloc::vec::Vec;

pub use binomial_extension::*;
pub use complex::*;
pub use packed_binomial_extension::*;
pub use packed_quintic_extension::PackedQuinticExtensionField;
pub use quintic_extension::QuinticExtensionField;

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

    /// Computes the pseudo inverse of the given field element.
    ///
    /// Returns `0` if `self == 0`, and `1/self` otherwise.
    /// In other words, returns `self^(n^D - 2)` where `D` is the extension degree.
    #[must_use]
    fn pseudo_inv(&self) -> Self;

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

/// Trait for fields that support a degree-5 extension using the trinomial `X^5 + X^2 - 1`.
///
/// # Mathematical Background
///
/// For a prime field `F_p`:
/// - If `5 | (p - 1)`: Use [`BinomiallyExtendable<5>`] with polynomial `X^5 - W`
/// - If `5 ∤ (p - 1)`: Use this trait with polynomial `X^5 + X^2 - 1`
///
/// The key reduction identity is: `X^5 ≡ 1 - X^2 (mod X^5 + X^2 - 1)`
///
/// **Important**: The irreducibility of `X^5 + X^2 - 1` must be verified for each specific
/// field. This trinomial is NOT irreducible over all fields where `5 ∤ (p - 1)`.
/// For example, `X = 2` is a root in `F_7`.
///
/// # Example Fields
///
/// - **BabyBear** (`P = 2^31 - 2^27 + 1`): `(P-1) mod 5 = 0` → uses `BinomiallyExtendable<5>`
/// - **KoalaBear** (`P = 2^31 - 2^24 + 1`): `(P-1) mod 5 = 2` → uses `QuinticExtendable`
///   (irreducibility verified for this field)
pub trait QuinticExtendable: Field + QuinticExtendableAlgebra<Self> {
    /// Frobenius coefficients for the quintic extension.
    ///
    /// `FROBENIUS_COEFFS[k]` represents `X^{(k+1)*p} mod (X^5 + X^2 - 1)` as a polynomial
    /// with coefficients `[c_0, c_1, c_2, c_3, c_4]` where `X^{(k+1)*p} = Σ c_i * X^i`.
    ///
    /// These precomputed values enable efficient Frobenius automorphism computation.
    const FROBENIUS_COEFFS: [[Self; 5]; 4];

    /// A generator for the multiplicative group of the extension field `F_{p^5}*`.
    ///
    /// Represented as polynomial coefficients `[g_0, g_1, g_2, g_3, g_4]`.
    const EXT_GENERATOR: [Self; 5];
}

/// Trait for algebras supporting quintic extension arithmetic over `A[X]/(X^5 + X^2 - 1)`.
///
/// The reduction identity `X^5 = 1 - X^2` yields higher power reductions:
/// - `X^6 = X - X^3`
/// - `X^7 = X^2 - X^4`
/// - `X^8 = X^3 + X^2 - 1`
///
/// Implementors may override the default methods with optimized versions
/// (e.g., SIMD implementations for packed fields).
pub trait QuinticExtendableAlgebra<F: Field>: Algebra<F> {
    /// Multiply two elements in the quintic extension ring.
    ///
    /// Computes `a * b mod (X^5 + X^2 - 1)` and stores the result in `res`.
    #[inline]
    fn quintic_mul(a: &[Self; 5], b: &[Self; 5], res: &mut [Self; 5]) {
        quintic_extension::quintic_mul(a, b, res);
    }

    /// Square an element in the quintic extension ring.
    ///
    /// Computes `a^2 mod (X^5 + X^2 - 1)` and stores the result in `res`.
    /// Uses optimized formulas exploiting the symmetry `a_i * a_j = a_j * a_i`.
    #[inline]
    fn quintic_square(a: &[Self; 5], res: &mut [Self; 5]) {
        quintic_extension::quintic_square(a, res);
    }

    /// Add two elements in the quintic extension ring.
    ///
    /// Addition is coefficient-wise and independent of the modulus polynomial.
    #[inline]
    #[must_use]
    fn quintic_add(a: &[Self; 5], b: &[Self; 5]) -> [Self; 5] {
        vector_add(a, b)
    }

    /// Subtract two elements in the quintic extension ring.
    ///
    /// Subtraction is coefficient-wise and independent of the modulus polynomial.
    #[inline]
    #[must_use]
    fn quintic_sub(a: &[Self; 5], b: &[Self; 5]) -> [Self; 5] {
        vector_sub(a, b)
    }

    /// Multiply a quintic extension element by a base field scalar.
    #[inline]
    fn quintic_base_mul(lhs: [Self; 5], rhs: Self) -> [Self; 5] {
        lhs.map(|x| x * rhs.clone())
    }
}

/// Trait for quintic extensions that support two-adic subgroup generators.
pub trait HasTwoAdicQuinticExtension: QuinticExtendable {
    /// Two-adicity of the multiplicative group order `p^5 - 1`.
    const EXT_TWO_ADICITY: usize;

    /// Type of array-like objects that can be used to store the two-adic generators.
    type ArrayLike: AsRef<[[Self; 5]]> + Sized;

    /// Additional two-adic generators for the extension field.
    ///
    /// These generators, combined with base field generators, provide roots of unity
    /// for all powers of 2 up to `EXT_TWO_ADICITY`.
    const TWO_ADIC_EXTENSION_GENERATORS: Self::ArrayLike;

    /// Returns a two-adic generator for the specified bit count.
    ///
    /// # Panics
    /// Panics if `bits > EXT_TWO_ADICITY`.
    #[must_use]
    fn ext_two_adic_generator(bits: usize) -> [Self; 5];
}
