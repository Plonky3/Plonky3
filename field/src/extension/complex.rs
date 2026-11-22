use super::{
    Binomial, BinomialExtensionField, BinomiallyExtendable, ExtensionAlgebra,
    HasTwoAdicBinomialExtension, binomial_mul, binomial_square,
};
use crate::{Algebra, Field, PrimeCharacteristicRing};

pub type Complex<F> = BinomialExtensionField<F, 2>;

/// A field for which `p = 3 (mod 4)`. Equivalently, `-1` is not a square,
/// so the complex extension can be defined `F[i] = F[X]/(X^2+1)`.
pub trait ComplexExtendable: Field {
    /// The two-adicity of `p+1`, the order of the circle group.
    const CIRCLE_TWO_ADICITY: usize;

    const COMPLEX_GENERATOR: Complex<Self>;

    fn circle_two_adic_generator(bits: usize) -> Complex<Self>;

    /// Multiply two elements in `Self[i] = Self[X] / (X^2 + 1)`.
    ///
    /// Uses `-b[1]` directly rather than multiplying by `W = -1`.
    #[inline]
    fn complex_mul(a: &[Self; 2], b: &[Self; 2], res: &mut [Self; 2]) {
        res[0] = Self::dot_product(a, &[b[0], -b[1]]);
        res[1] = Self::dot_product(a, &[b[1], b[0]]);
    }

    /// Square an element in `Self[i] = Self[X] / (X^2 + 1)` in 2 base multiplications:
    /// `(a_0 + a_1 i)^2 = (a_0 + a_1)(a_0 - a_1) + 2 a_0 a_1 i`.
    #[inline]
    fn complex_square(a: &[Self; 2], res: &mut [Self; 2]) {
        res[0] = (a[0] + a[1]) * (a[0] - a[1]);
        res[1] = (a[0] * a[1]).double();
    }
}

impl<F: ComplexExtendable> ExtensionAlgebra<F, 2, Binomial<F>> for F {
    #[inline]
    fn ext_mul(a: &[Self; 2], b: &[Self; 2], res: &mut [Self; 2]) {
        F::complex_mul(a, b, res);
    }

    #[inline]
    fn ext_square(a: &[Self; 2], res: &mut [Self; 2]) {
        F::complex_square(a, res);
    }
}

impl<F: ComplexExtendable> BinomiallyExtendable<2> for F {
    fn binomial_algebra_id() -> alloc::vec::Vec<u8> {
        b"p3-power-basis-v1:X^2+1".to_vec()
    }

    const W: Self = F::NEG_ONE;

    // since `p = 3 (mod 4)`, `(p-1)/2` is always odd,
    // so `(-1)^((p-1)/2) = -1`
    const DTH_ROOT: Self = F::NEG_ONE;

    const EXT_GENERATOR: [Self; 2] = F::COMPLEX_GENERATOR.value;
}

/// Convenience methods for complex extensions
impl<R: PrimeCharacteristicRing> Complex<R> {
    #[inline(always)]
    pub const fn new_complex(real: R, imag: R) -> Self {
        Self::new([real, imag])
    }

    #[inline(always)]
    pub const fn new_real(real: R) -> Self {
        Self::new_complex(real, R::ZERO)
    }

    #[inline(always)]
    pub const fn new_imag(imag: R) -> Self {
        Self::new_complex(R::ZERO, imag)
    }

    #[inline(always)]
    #[must_use]
    pub fn real(&self) -> R {
        self.value[0].dup()
    }

    #[inline(always)]
    #[must_use]
    pub fn imag(&self) -> R {
        self.value[1].dup()
    }

    #[inline(always)]
    pub fn conjugate(&self) -> Self {
        Self::new_complex(self.real(), self.imag().neg())
    }

    #[inline]
    #[must_use]
    pub fn norm(&self) -> R {
        self.real().square() + self.imag().square()
    }

    #[inline(always)]
    #[must_use]
    pub fn to_array(&self) -> [R; 2] {
        core::array::from_fn(|i| self.value[i].dup())
    }

    // Sometimes we want to rotate over an extension that's not necessarily ComplexExtendable,
    // but still on the circle.
    #[inline]
    pub fn rotate<Ext: Algebra<R>>(&self, rhs: &Complex<Ext>) -> Complex<Ext> {
        Complex::<Ext>::new_complex(
            rhs.real() * self.real() - rhs.imag() * self.imag(),
            rhs.imag() * self.real() + rhs.real() * self.imag(),
        )
    }
}

/// The complex extension of this field has a binomial extension.
///
/// This exists if the polynomial ring `F[i][X]` has an irreducible polynomial `X^d-W`
/// allowing us to define the binomial extension field `F[i][X]/(X^d-W)`.
pub trait HasComplexBinomialExtension<const D: usize>: ComplexExtendable {
    /// Stable identity of the power basis modulo `X^D - W` over `Self[i]`.
    fn complex_binomial_algebra_id() -> alloc::vec::Vec<u8>;

    const W: Complex<Self>;

    // DTH_ROOT = W^((n - 1)/D).
    // n is the order of base field.
    // Only works when exists k such that n = kD + 1.
    const DTH_ROOT: Complex<Self>;

    const EXT_GENERATOR: [Complex<Self>; D];

    /// Multiply a `Complex<Self>` element by `W = Self::W`.
    ///
    /// The default is a general complex multiplication. Override when `W` has
    /// structure that makes multiplication cheaper (e.g. add-only when all
    /// coefficients of `W` are small).
    #[inline]
    fn mul_by_w(z: Complex<Self>) -> Complex<Self> {
        <Complex<Self> as BinomiallyExtendable<D>>::W * z
    }

    /// Multiply two elements in `Complex<Self>[X] / (X^D - W)`.
    ///
    /// Specializes `D = 2` (3-mul Karatsuba + [`Self::mul_by_w`]) and `D = 3`
    /// (6-mul Karatsuba + [`Self::mul_by_w`]) to avoid full complex multiplications
    /// by `W`. Base fields may override this with delayed-reduction kernels.
    #[inline]
    fn complex_ext_mul(
        a: &[Complex<Self>; D],
        b: &[Complex<Self>; D],
        res: &mut [Complex<Self>; D],
    ) {
        match D {
            2 => {
                let m0 = a[0] * b[0];
                let m1 = a[1] * b[1];
                let m2 = (a[0] + a[1]) * (b[0] + b[1]);
                res[0] = m0 + Self::mul_by_w(m1);
                res[1] = m2 - m0 - m1;
            }
            3 => {
                let m0 = a[0] * b[0];
                let m1 = a[1] * b[1];
                let m2 = a[2] * b[2];
                let t0 = (a[1] + a[2]) * (b[1] + b[2]) - m1 - m2;
                let t1 = (a[0] + a[1]) * (b[0] + b[1]) - m0 - m1;
                let t2 = (a[0] + a[2]) * (b[0] + b[2]) - m0 - m2;
                res[0] = m0 + Self::mul_by_w(t0);
                res[1] = t1 + Self::mul_by_w(m2);
                res[2] = t2 + m1;
            }
            _ => binomial_mul::<Complex<Self>, Complex<Self>, Complex<Self>, D>(
                a,
                b,
                res,
                <Complex<Self> as BinomiallyExtendable<D>>::W,
            ),
        }
    }

    /// Square an element in `Complex<Self>[X] / (X^D - W)`.
    ///
    /// Specializes `D = 2` and `D = 3` using [`Self::mul_by_w`] so base fields with
    /// cheap `W` scaling avoid full complex multiplications by `W`.
    #[inline]
    fn complex_ext_square(a: &[Complex<Self>; D], res: &mut [Complex<Self>; D]) {
        match D {
            2 => {
                res[0] = a[0].square() + Self::mul_by_w(a[1].square());
                res[1] = (a[0] * a[1]).double();
            }
            3 => {
                res[0] = a[0].square() + Self::mul_by_w(a[1] * a[2]).double();
                res[1] = Self::mul_by_w(a[2].square()) + (a[0] * a[1]).double();
                res[2] = a[1].square() + (a[0] * a[2]).double();
            }
            _ => binomial_square::<Complex<Self>, Complex<Self>, D>(
                a,
                res,
                <Complex<Self> as BinomiallyExtendable<D>>::W,
            ),
        }
    }
}

impl<F, const D: usize> ExtensionAlgebra<Self, D, Binomial<Self>> for Complex<F>
where
    F: HasComplexBinomialExtension<D>,
{
    #[inline]
    fn ext_mul(a: &[Self; D], b: &[Self; D], res: &mut [Self; D]) {
        F::complex_ext_mul(a, b, res);
    }

    #[inline]
    fn ext_square(a: &[Self; D], res: &mut [Self; D]) {
        F::complex_ext_square(a, res);
    }
}

impl<F, const D: usize> BinomiallyExtendable<D> for Complex<F>
where
    F: HasComplexBinomialExtension<D>,
{
    fn binomial_algebra_id() -> alloc::vec::Vec<u8> {
        F::complex_binomial_algebra_id()
    }

    const W: Self = <F as HasComplexBinomialExtension<D>>::W;

    const DTH_ROOT: Self = <F as HasComplexBinomialExtension<D>>::DTH_ROOT;

    const EXT_GENERATOR: [Self; D] = F::EXT_GENERATOR;
}

/// The complex extension of this field has a two-adic binomial extension.
pub trait HasTwoAdicComplexBinomialExtension<const D: usize>:
    HasComplexBinomialExtension<D>
{
    const COMPLEX_EXT_TWO_ADICITY: usize;

    fn complex_ext_two_adic_generator(bits: usize) -> [Complex<Self>; D];
}

impl<F, const D: usize> HasTwoAdicBinomialExtension<D> for Complex<F>
where
    F: HasTwoAdicComplexBinomialExtension<D>,
{
    const EXT_TWO_ADICITY: usize = F::COMPLEX_EXT_TWO_ADICITY;

    #[inline(always)]
    fn ext_two_adic_generator(bits: usize) -> [Self; D] {
        F::complex_ext_two_adic_generator(bits)
    }
}
