//! Implementation of the quadratic extension of the Mersenne31 field
//! by X^2 + 1.
//!
//! Note that X^2 + 1 is irreducible over p = Mersenne31 field because
//! kronecker(-1, p) = -1, that is, -1 is not square in F_p.

use core::fmt::{Debug, Display, Formatter};
use core::iter::{Product, Sum};
use core::ops::{Add, AddAssign, Div, Mul, MulAssign, Neg, Sub, SubAssign};

use p3_field::{AbstractExtensionField, AbstractField, Field, TwoAdicField};
use rand::distributions::{Distribution, Standard};
use rand::Rng;

use crate::Mersenne31;

#[derive(PartialEq, Eq, Hash, Copy, Clone, Debug, Default)]
pub struct Mersenne31Complex<AF: AbstractField<F = Mersenne31>> {
    pub(crate) parts: [AF; 2],
}

impl<AF: AbstractField<F = Mersenne31>> Mersenne31Complex<AF> {
    pub const fn new(real: AF, imag: AF) -> Self {
        Self {
            parts: [real, imag],
        }
    }

    pub fn new_real(real: AF) -> Self {
        Self::new(real, AF::zero())
    }

    pub fn new_imag(imag: AF) -> Self {
        Self::new(AF::zero(), imag)
    }

    pub fn real(&self) -> AF {
        self.parts[0].clone()
    }

    pub fn imag(&self) -> AF {
        self.parts[1].clone()
    }

    pub fn conjugate(&self) -> Self {
        Self::new(self.real(), self.imag().neg())
    }

    pub fn magnitude_squared(&self) -> AF {
        self.real() * self.real() + self.imag() * self.imag()
    }
}

impl<AF: AbstractField<F = Mersenne31>> Add for Mersenne31Complex<AF> {
    type Output = Self;

    fn add(self, rhs: Self) -> Self {
        Self::new(self.real() + rhs.real(), self.imag() + rhs.imag())
    }
}

impl<AF: AbstractField<F = Mersenne31>> Add<AF> for Mersenne31Complex<AF> {
    type Output = Self;

    fn add(self, rhs: AF) -> Self {
        Self::new(self.real() + rhs, self.imag())
    }
}

impl<AF: AbstractField<F = Mersenne31>> AddAssign for Mersenne31Complex<AF> {
    fn add_assign(&mut self, rhs: Self) {
        self.parts[0] += rhs.real();
        self.parts[1] += rhs.imag();
    }
}

impl<AF: AbstractField<F = Mersenne31>> AddAssign<AF> for Mersenne31Complex<AF> {
    fn add_assign(&mut self, rhs: AF) {
        self.parts[0] += rhs;
    }
}

impl<AF: AbstractField<F = Mersenne31>> Sum for Mersenne31Complex<AF> {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.reduce(|x, y| x + y)
            .unwrap_or(Self::new_real(AF::zero()))
    }
}

impl<AF: AbstractField<F = Mersenne31>> Sub for Mersenne31Complex<AF> {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self {
        Self::new(self.real() - rhs.real(), self.imag() - rhs.imag())
    }
}

impl<AF: AbstractField<F = Mersenne31>> Sub<AF> for Mersenne31Complex<AF> {
    type Output = Self;

    fn sub(self, rhs: AF) -> Self {
        Self::new(self.real() - rhs, self.imag())
    }
}

impl<AF: AbstractField<F = Mersenne31>> SubAssign for Mersenne31Complex<AF> {
    fn sub_assign(&mut self, rhs: Self) {
        self.parts[0] -= rhs.real();
        self.parts[1] -= rhs.imag();
    }
}

impl<AF: AbstractField<F = Mersenne31>> SubAssign<AF> for Mersenne31Complex<AF> {
    fn sub_assign(&mut self, rhs: AF) {
        self.parts[0] -= rhs;
    }
}

impl<AF: AbstractField<F = Mersenne31>> Neg for Mersenne31Complex<AF> {
    type Output = Self;

    fn neg(self) -> Self {
        Self::new(-self.real(), -self.imag())
    }
}

impl<AF: AbstractField<F = Mersenne31>> Mul for Mersenne31Complex<AF> {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self {
        // TODO: Try the Gauss algorithm for complex multiplication, see if it's faster for us.
        // Method for (a + b i) * (c + d i):
        //     t1 = a * c
        //     t2 = b * d
        //     real = t1 - t2
        //     imag = (a + b) * (c + d) - t1 - t2

        let real = self.real() * rhs.real() - self.imag() * rhs.imag();
        let imag = self.real() * rhs.imag() + self.imag() * rhs.real();
        Self::new(real, imag)
    }
}

impl<AF: AbstractField<F = Mersenne31>> Mul<AF> for Mersenne31Complex<AF> {
    type Output = Self;

    fn mul(self, rhs: AF) -> Self {
        Self::new(self.real() * rhs.clone(), self.imag() * rhs)
    }
}

impl<AF: AbstractField<F = Mersenne31>> MulAssign for Mersenne31Complex<AF> {
    fn mul_assign(&mut self, rhs: Self) {
        *self = self.clone() * rhs;
    }
}

impl<AF: AbstractField<F = Mersenne31>> MulAssign<AF> for Mersenne31Complex<AF> {
    fn mul_assign(&mut self, rhs: AF) {
        self.parts[0] *= rhs.clone();
        self.parts[1] *= rhs;
    }
}

impl<AF: AbstractField<F = Mersenne31>> Product for Mersenne31Complex<AF> {
    fn product<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.reduce(|x, y| x * y)
            .unwrap_or(Self::new_real(AF::one()))
    }
}

impl<AF: AbstractField<F = Mersenne31>> AbstractField for Mersenne31Complex<AF> {
    type F = Mersenne31Complex<Mersenne31>;

    fn zero() -> Self {
        Self::new_real(AF::zero())
    }
    fn one() -> Self {
        Self::new_real(AF::one())
    }
    fn two() -> Self {
        Self::new_real(AF::two())
    }
    fn neg_one() -> Self {
        Self::new_real(AF::neg_one())
    }

    #[inline]
    fn from_f(f: Self::F) -> Self {
        Self::new(AF::from_f(f.real()), AF::from_f(f.imag()))
    }

    fn from_bool(b: bool) -> Self {
        Self::new_real(AF::from_bool(b))
    }

    fn from_canonical_u8(n: u8) -> Self {
        Self::new_real(AF::from_canonical_u8(n))
    }

    fn from_canonical_u16(n: u16) -> Self {
        Self::new_real(AF::from_canonical_u16(n))
    }

    fn from_canonical_u32(n: u32) -> Self {
        Self::new_real(AF::from_canonical_u32(n))
    }

    fn from_canonical_u64(n: u64) -> Self {
        Self::new_real(AF::from_canonical_u64(n))
    }

    fn from_canonical_usize(n: usize) -> Self {
        Self::new_real(AF::from_canonical_usize(n))
    }

    fn from_wrapped_u32(n: u32) -> Self {
        Self::new_real(AF::from_wrapped_u32(n))
    }

    fn from_wrapped_u64(n: u64) -> Self {
        Self::new_real(AF::from_wrapped_u64(n))
    }

    // sage: p = 2^31 - 1
    // sage: F = GF(p)
    // sage: R.<x> = F[]
    // sage: F2.<u> = F.extension(x^2 + 1)
    // sage: F2.multiplicative_generator()
    // u + 12
    fn generator() -> Self {
        Self::new(AF::from_canonical_u8(12), AF::one())
    }
}

impl Div<Self> for Mersenne31Complex<Mersenne31> {
    type Output = Self;

    #[allow(clippy::suspicious_arithmetic_impl)]
    fn div(self, rhs: Self) -> Self::Output {
        self * rhs.inverse()
    }
}

impl Display for Mersenne31Complex<Mersenne31> {
    fn fmt(&self, f: &mut Formatter<'_>) -> core::fmt::Result {
        write!(f, "{} + {}i", self.real(), self.imag())
    }
}

impl Field for Mersenne31Complex<Mersenne31> {
    type Packing = Self;

    fn try_inverse(&self) -> Option<Self> {
        self.magnitude_squared()
            .try_inverse()
            .map(|x| self.conjugate() * x)
    }

    fn mul_2exp_u64(&self, exp: u64) -> Self {
        Self::new(
            self.parts[0].mul_2exp_u64(exp),
            self.parts[1].mul_2exp_u64(exp),
        )
    }

    fn div_2exp_u64(&self, exp: u64) -> Self {
        Self::new(
            self.parts[0].div_2exp_u64(exp),
            self.parts[1].div_2exp_u64(exp),
        )
    }
}

impl TwoAdicField for Mersenne31Complex<Mersenne31> {
    const TWO_ADICITY: usize = 32;

    fn two_adic_generator(bits: usize) -> Self {
        // TODO: Consider a `match` which may speed this up.
        assert!(bits <= Self::TWO_ADICITY);
        // Generator of the whole 2^TWO_ADICITY group
        // sage: p = 2^31 - 1
        // sage: F = GF(p)
        // sage: R.<x> = F[]
        // sage: F2.<u> = F.extension(x^2 + 1)
        // sage: g = F2.multiplicative_generator()^((p^2 - 1) / 2^32); g
        // 1117296306*u + 1166849849
        // sage: assert(g.multiplicative_order() == 2^32)
        let base = Self::new(
            Mersenne31::new(1_166_849_849),
            Mersenne31::new(1_117_296_306),
        );
        base.exp_power_of_2(Self::TWO_ADICITY - bits)
    }
}

impl<AF: AbstractField<F = Mersenne31>> AbstractExtensionField<AF> for Mersenne31Complex<AF> {
    const D: usize = 2;

    fn from_base(b: AF) -> Self {
        Self::new_real(b)
    }

    fn from_base_slice(bs: &[AF]) -> Self {
        assert_eq!(bs.len(), 2);
        Self::new(bs[0].clone(), bs[1].clone())
    }

    fn as_base_slice(&self) -> &[AF] {
        &self.parts
    }
}

impl Distribution<Mersenne31Complex<Mersenne31>> for Standard {
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> Mersenne31Complex<Mersenne31> {
        Mersenne31Complex::<Mersenne31>::new(rng.gen(), rng.gen())
    }
}

#[cfg(test)]
mod tests {
    use p3_field::PrimeField32;
    use p3_field_testing::{test_field, test_two_adic_field};

    use super::*;

    type Fi = Mersenne31Complex<Mersenne31>;
    type F = Mersenne31;

    #[test]
    fn add() {
        // real part
        assert_eq!(Fi::one() + Fi::one(), Fi::two());
        assert_eq!(Fi::neg_one() + Fi::one(), Fi::zero());
        assert_eq!(Fi::neg_one() + Fi::two(), Fi::one());
        assert_eq!(
            (Fi::neg_one() + Fi::neg_one()).real(),
            F::new(F::ORDER_U32 - 2)
        );

        // complex part
        assert_eq!(
            Fi::new_imag(F::one()) + Fi::new_imag(F::one()),
            Fi::new_imag(F::two())
        );
        assert_eq!(
            Fi::new_imag(F::neg_one()) + Fi::new_imag(F::one()),
            Fi::new_imag(F::zero())
        );
        assert_eq!(
            Fi::new_imag(F::neg_one()) + Fi::new_imag(F::two()),
            Fi::new_imag(F::one())
        );
        assert_eq!(
            (Fi::new_imag(F::neg_one()) + Fi::new_imag(F::neg_one())).imag(),
            F::new(F::ORDER_U32 - 2)
        );

        // further tests
        assert_eq!(
            Fi::new(F::one(), F::two()) + Fi::new(F::one(), F::one()),
            Fi::new(F::two(), F::new(3))
        );
        assert_eq!(
            Fi::new(F::neg_one(), F::neg_one()) + Fi::new(F::one(), F::one()),
            Fi::zero()
        );
        assert_eq!(
            Fi::new(F::neg_one(), F::one()) + Fi::new(F::two(), F::new(F::ORDER_U32 - 2)),
            Fi::new(F::one(), F::neg_one())
        );
    }

    #[test]
    fn sub() {
        // real part
        assert_eq!(Fi::one() - Fi::one(), Fi::zero());
        assert_eq!(Fi::two() - Fi::two(), Fi::zero());
        assert_eq!(Fi::neg_one() - Fi::neg_one(), Fi::zero());
        assert_eq!(Fi::two() - Fi::one(), Fi::one());
        assert_eq!(Fi::neg_one() - Fi::zero(), Fi::neg_one());

        // complex part
        assert_eq!(Fi::new_imag(F::one()) - Fi::new_imag(F::one()), Fi::zero());
        assert_eq!(Fi::new_imag(F::two()) - Fi::new_imag(F::two()), Fi::zero());
        assert_eq!(
            Fi::new_imag(F::neg_one()) - Fi::new_imag(F::neg_one()),
            Fi::zero()
        );
        assert_eq!(
            Fi::new_imag(F::two()) - Fi::new_imag(F::one()),
            Fi::new_imag(F::one())
        );
        assert_eq!(
            Fi::new_imag(F::neg_one()) - Fi::zero(),
            Fi::new_imag(F::neg_one())
        );
    }

    #[test]
    fn mul() {
        assert_eq!(
            Fi::new(F::two(), F::two()) * Fi::new(F::new(4), F::new(5)),
            Fi::new(-F::two(), F::new(18))
        );
    }

    #[test]
    fn mul_2exp_u64() {
        // real part
        // 1 * 2^0 = 1.
        assert_eq!(Fi::one().mul_2exp_u64(0), Fi::one());
        // 2 * 2^30 = 2^31 = 1.
        assert_eq!(Fi::two().mul_2exp_u64(30), Fi::one());
        // 5 * 2^2 = 20.
        assert_eq!(
            Fi::new_real(F::new(5)).mul_2exp_u64(2),
            Fi::new_real(F::new(20))
        );

        // complex part
        // i * 2^0 = i.
        assert_eq!(
            Fi::new_imag(F::one()).mul_2exp_u64(0),
            Fi::new_imag(F::one())
        );
        // (2i) * 2^30 = (2^31) * i = i.
        assert_eq!(
            Fi::new_imag(F::two()).mul_2exp_u64(30),
            Fi::new_imag(F::one())
        );
        // 5i * 2^2 = 20i.
        assert_eq!(
            Fi::new_imag(F::new(5)).mul_2exp_u64(2),
            Fi::new_imag(F::new(20))
        );
    }

    test_field!(crate::Mersenne31Complex<crate::Mersenne31>);
    test_two_adic_field!(crate::Mersenne31Complex<crate::Mersenne31>);
}
