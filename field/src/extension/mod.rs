use core::iter;
use core::marker::PhantomData;

use crate::field::Field;
use crate::{Algebra, ExtensionField, PackedField};

mod binomial_extension;
mod complex;
mod cubic_extension;
mod packed_binomial_extension;
mod packed_cubic_extension;
mod packed_quintic_extension;
mod quintic_extension;

use alloc::vec::Vec;

pub use binomial_extension::*;
pub use complex::*;
pub use cubic_extension::{CubicTrinomialExtensionField, cubic_square, trinomial_cubic_mul};
pub use packed_binomial_extension::*;
pub use packed_cubic_extension::PackedCubicTrinomialExtensionField;
pub use packed_quintic_extension::PackedQuinticTrinomialExtensionField;
pub use quintic_extension::{
    QuinticTrinomialExtensionField, quintic_square, trinomial_quintic_mul,
};

// ---------------------------------------------------------------------------
// Shape-parameterized algebra for extension fields
// ---------------------------------------------------------------------------
//
// Extension-ring arithmetic (mul / square / add / sub / base_mul) is exposed
// via a single trait `ExtensionAlgebra<F, D, Shape>`, parameterized by:
//   - the base field `F`,
//   - the extension degree `D`,
//   - a marker `Shape: ExtensionShape` selecting the reducing polynomial.
//
// The three supported shapes are:
//   - `Binomial<F>`     — `F[X] / (X^D - W)`, degree-generic, `W = F::W`.
//   - `CubicTrinomial`  — `F[X] / (X^3 - X - 1)`.
//   - `QuinticTrinomial`— `F[X] / (X^5 + X^2 - 1)`.
//
// SIMD-packed types override `ext_mul` / `ext_square` with optimized kernels;
// the other three methods have sensible coefficient-wise defaults.

/// Sealed marker for the reducing polynomial shape of an extension field.
///
/// The three concrete shapes — [`Binomial`], [`CubicTrinomial`], and
/// [`QuinticTrinomial`] — correspond to the reducers
/// `X^D - W`, `X^3 - X - 1`, and `X^5 + X^2 - 1` respectively.
pub trait ExtensionShape: 'static + sealed::Sealed {}

mod sealed {
    pub trait Sealed {}
}

/// Marker for the binomial reducer `X^D - W` (degree-generic).
#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug, PartialOrd, Ord)]
pub struct Binomial<F>(PhantomData<F>);

/// Marker for the trinomial reducer `X^3 - X - 1`.
#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug, PartialOrd, Ord)]
pub struct CubicTrinomial;

/// Marker for the trinomial reducer `X^5 + X^2 - 1`.
#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug, PartialOrd, Ord)]
pub struct QuinticTrinomial;

impl<F: Field> sealed::Sealed for Binomial<F> {}
impl sealed::Sealed for CubicTrinomial {}
impl sealed::Sealed for QuinticTrinomial {}

impl<F: Field> ExtensionShape for Binomial<F> {}
impl ExtensionShape for CubicTrinomial {}
impl ExtensionShape for QuinticTrinomial {}

/// Unified extension-field representation.
///
/// An `ExtField<F, D, Shape, A>` represents an element of a degree-`D` extension
/// of the base field `F`, with the reducing polynomial determined by `Shape`.
/// The `A` parameter is the algebra over `F` storing each coefficient — usually
/// `F` itself for scalar elements, or `F::Packing` for SIMD-packed elements.
///
/// The three public-facing extension types — [`BinomialExtensionField`],
/// [`CubicTrinomialExtensionField`], [`QuinticTrinomialExtensionField`] — are
/// type aliases over this struct with their respective `Shape`s.
#[derive(
    Copy, Clone, Eq, PartialEq, Hash, Debug, serde::Serialize, serde::Deserialize, PartialOrd, Ord,
)]
#[repr(transparent)] // Needed to make various casts safe.
#[must_use]
pub struct ExtField<F, const D: usize, Shape, A = F> {
    #[serde(
        with = "p3_util::array_serialization",
        bound(
            serialize = "A: serde::Serialize",
            deserialize = "A: serde::Deserialize<'de>"
        )
    )]
    pub(crate) value: [A; D],

    _phantom: PhantomData<(F, Shape)>,
}

impl<F, const D: usize, Shape, A> ExtField<F, D, Shape, A> {
    /// Create an extension field element from an array of base/algebra elements.
    ///
    /// Any array is accepted. No reduction is required since each entry is
    /// already a valid element of the algebra over `F`.
    ///
    /// # Panics
    /// Panics (at compile time) if `D <= 1`. A degree-0 or degree-1 "extension"
    /// is degenerate — use `F` directly instead.
    #[inline]
    pub const fn new(value: [A; D]) -> Self {
        const { assert!(D > 1) }
        Self {
            value,
            _phantom: PhantomData,
        }
    }
}

/// Unified packed extension-field representation (SIMD-lane parallel).
///
/// `PackedExtField<F, PF, D, Shape>` stores `D` packed coefficients (one per
/// SIMD lane), representing the `WIDTH` extension-field elements that fit in
/// one SIMD register simultaneously.
///
/// The three public-facing packed types — [`PackedBinomialExtensionField`],
/// [`PackedCubicTrinomialExtensionField`], [`PackedQuinticTrinomialExtensionField`]
/// — are type aliases over this struct with their respective `Shape`s.
#[derive(
    Copy, Clone, Eq, PartialEq, Hash, Debug, serde::Serialize, serde::Deserialize, PartialOrd, Ord,
)]
#[repr(transparent)]
#[must_use]
pub struct PackedExtField<F: Field, PF: PackedField<Scalar = F>, const D: usize, Shape> {
    #[serde(
        with = "p3_util::array_serialization",
        bound(
            serialize = "PF: serde::Serialize",
            deserialize = "PF: serde::Deserialize<'de>"
        )
    )]
    pub(crate) value: [PF; D],
    _phantom: PhantomData<Shape>,
}

impl<F: Field, PF: PackedField<Scalar = F>, const D: usize, Shape> PackedExtField<F, PF, D, Shape> {
    /// Create a packed extension-field element from an array of packed coefficients.
    #[inline]
    pub const fn new(value: [PF; D]) -> Self {
        Self {
            value,
            _phantom: PhantomData,
        }
    }
}

// ---------------------------------------------------------------------------
// Shape-agnostic impls on the unified struct
// ---------------------------------------------------------------------------
//
// These traits — `Default`, `From<A>`, `From<[A; D]>`, `BasedVectorSpace`,
// `Packable`, `Distribution` — have the same implementation across every
// shape. They live here on the unified struct rather than being triplicated
// across the per-shape files.

impl<F: Field, A: Algebra<F>, const D: usize, Shape: ExtensionShape> Default
    for ExtField<F, D, Shape, A>
{
    #[inline]
    fn default() -> Self {
        Self::new(core::array::from_fn(|_| A::ZERO))
    }
}

impl<F: Field, A: Algebra<F>, const D: usize, Shape: ExtensionShape> From<A>
    for ExtField<F, D, Shape, A>
{
    #[inline]
    fn from(x: A) -> Self {
        Self::new(crate::field_to_array(x))
    }
}

impl<F, A, const D: usize, Shape: ExtensionShape> From<[A; D]> for ExtField<F, D, Shape, A> {
    #[inline]
    fn from(x: [A; D]) -> Self {
        Self::new(x)
    }
}

impl<F, const D: usize, Shape> crate::Packable for ExtField<F, D, Shape>
where
    F: Field + ExtensionAlgebra<F, D, Shape>,
    Shape: ExtensionShape + Copy + Eq + core::hash::Hash + Send + Sync,
{
}

impl<F, A: Algebra<F>, const D: usize, Shape: ExtensionShape + Clone> crate::BasedVectorSpace<A>
    for ExtField<F, D, Shape, A>
where
    F: Field + ExtensionAlgebra<F, D, Shape>,
{
    const DIMENSION: usize = D;

    #[inline]
    fn as_basis_coefficients_slice(&self) -> &[A] {
        &self.value
    }

    #[inline]
    fn from_basis_coefficients_fn<Fn: FnMut(usize) -> A>(f: Fn) -> Self {
        Self::new(core::array::from_fn(f))
    }

    #[inline]
    fn from_basis_coefficients_iter<I: ExactSizeIterator<Item = A>>(mut iter: I) -> Option<Self> {
        // The unwrap is safe as we just checked the length of iter.
        (iter.len() == D).then(|| Self::new(core::array::from_fn(|_| iter.next().unwrap())))
    }

    #[inline]
    fn flatten_to_base(vec: Vec<Self>) -> Vec<A> {
        // Safety: `Self` is `#[repr(transparent)]` over `[A; D]`.
        unsafe { p3_util::flatten_to_base::<A, Self>(vec) }
    }

    #[inline]
    fn reconstitute_from_base(vec: Vec<A>) -> Vec<Self> {
        // Safety: `Self` is `#[repr(transparent)]` over `[A; D]`.
        unsafe { p3_util::reconstitute_from_base::<A, Self>(vec) }
    }
}

impl<F, const D: usize, Shape: ExtensionShape> rand::distr::Distribution<ExtField<F, D, Shape>>
    for rand::distr::StandardUniform
where
    F: Field + ExtensionAlgebra<F, D, Shape>,
    Self: rand::distr::Distribution<F>,
{
    #[inline]
    fn sample<R: rand::Rng + ?Sized>(&self, rng: &mut R) -> ExtField<F, D, Shape> {
        ExtField::new(core::array::from_fn(|_| self.sample(rng)))
    }
}

/// Algebra over `F` that supports degree-`D` extension arithmetic with a given reducer `Shape`.
pub trait ExtensionAlgebra<F: Field, const D: usize, Shape: ExtensionShape>: Algebra<F> {
    /// Multiplication in the algebra extension ring.
    fn ext_mul(a: &[Self; D], b: &[Self; D], res: &mut [Self; D]);

    /// Squaring in the algebra extension ring.
    ///
    /// Override when a dedicated symmetry-exploiting kernel beats a general multiply.
    #[inline]
    fn ext_square(a: &[Self; D], res: &mut [Self; D]) {
        Self::ext_mul(a, a, res);
    }

    /// Coefficient-wise addition.
    #[inline]
    #[must_use]
    fn ext_add(a: &[Self; D], b: &[Self; D]) -> [Self; D] {
        vector_add(a, b)
    }

    /// Coefficient-wise subtraction.
    #[inline]
    #[must_use]
    fn ext_sub(a: &[Self; D], b: &[Self; D]) -> [Self; D] {
        vector_sub(a, b)
    }

    /// Multiply an extension element by a base-field scalar.
    #[inline]
    #[must_use]
    fn ext_base_mul(lhs: [Self; D], rhs: Self) -> [Self; D] {
        lhs.map(|x| x * rhs.dup())
    }
}

/// Trait for fields that support binomial extension of the form `F[X]/(X^D - W)`.
///
/// A type implementing this trait can define a degree-`D` extension field using an
/// irreducible binomial polynomial `X^D - W`, where `W` is a nonzero constant in the base field.
///
/// This is used to construct extension fields with efficient arithmetic.
pub trait BinomiallyExtendable<const D: usize>:
    Field + ExtensionAlgebra<Self, D, Binomial<Self>>
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
    ///
    /// # Panics
    /// Panics if `bits > EXT_TWO_ADICITY`.
    #[must_use]
    fn ext_two_adic_generator(bits: usize) -> [Self; D];
}

/// Trait for fields that support a degree-3 extension using the trinomial `X^3 - X - 1`.
///
/// Implement only when `X^3 - X - 1` is irreducible over the base field.
pub trait CubicTrinomialExtendable: Field + ExtensionAlgebra<Self, 3, CubicTrinomial> {
    /// Linear map for the Frobenius automorphism on `Σ a_i X^i` in the power basis `(1, X, X^2)`.
    ///
    /// Row `i` contains the coefficients of the image of `X^i` under Frobenius (the first row
    /// includes the fixed contribution from `a_0`).
    const FROBENIUS_MATRIX: [[Self; 3]; 3];

    /// A generator of the multiplicative group of `F_{p^3}^*`, as polynomial coefficients.
    const EXT_GENERATOR: [Self; 3];
}

/// Trait for cubic trinomial extensions that expose two-adic subgroup generators.
pub trait HasTwoAdicCubicExtension: CubicTrinomialExtendable {
    /// Two-adicity of `p^3 - 1`.
    const EXT_TWO_ADICITY: usize;

    /// Two-adic generator for the extension field; `bits` must be at most `EXT_TWO_ADICITY`.
    #[must_use]
    fn ext_two_adic_generator(bits: usize) -> [Self; 3];
}

/// Trait for fields that support a degree-5 extension using the trinomial `X^5 + X^2 - 1`.
///
/// This trait should only be implemented for fields where `X^5 + X^2 - 1` is irreducible.
/// The implementor must verify irreducibility for their specific field.
pub trait QuinticTrinomialExtendable: Field + ExtensionAlgebra<Self, 5, QuinticTrinomial> {
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

/// Trait for quintic extensions that support two-adic subgroup generators.
pub trait HasTwoAdicQuinticExtension: QuinticTrinomialExtendable {
    /// Two-adicity of the multiplicative group order `p^5 - 1`.
    const EXT_TWO_ADICITY: usize;

    /// Returns a two-adic generator for the specified bit count.
    ///
    /// # Panics
    /// Panics if `bits > EXT_TWO_ADICITY`.
    #[must_use]
    fn ext_two_adic_generator(bits: usize) -> [Self; 5];
}
