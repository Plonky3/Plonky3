use core::fmt::{Display, Formatter};
use core::iter::{Product, Sum};
use core::ops::{Add, AddAssign, Div, Mul, MulAssign, Neg, Sub, SubAssign};

use p3_field::{AbstractExtensionField, AbstractField, AbstractionOf, Field, TwoAdicField};
use rand::distributions::{Distribution, Standard};
use rand::Rng;

use crate::Mersenne31;

#[derive(PartialEq, Eq, Hash, Copy, Clone, Debug, Default)]
pub struct Mersenne31Complex<AF: AbstractionOf<Mersenne31>> {
    parts: [AF; 2],
}

impl<AF: AbstractionOf<Mersenne31>> Mersenne31Complex<AF> {
    pub const fn new(real: AF, imag: AF) -> Self {
        Self {
            parts: [real, imag],
        }
    }

    pub const fn new_real(real: AF) -> Self {
        Self::new(real, AF::ZERO)
    }

    pub const fn new_imag(imag: AF) -> Self {
        Self::new(AF::ZERO, imag)
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

impl<AF: AbstractionOf<Mersenne31>> Add for Mersenne31Complex<AF> {
    type Output = Self;

    fn add(self, rhs: Self) -> Self {
        Self::new(self.real() + rhs.real(), self.imag() + rhs.imag())
    }
}

impl<AF: AbstractionOf<Mersenne31>> Add<AF> for Mersenne31Complex<AF> {
    type Output = Self;

    fn add(self, rhs: AF) -> Self {
        Self::new(self.real() + rhs, self.imag())
    }
}

impl<AF: AbstractionOf<Mersenne31>> AddAssign for Mersenne31Complex<AF> {
    fn add_assign(&mut self, rhs: Self) {
        self.parts[0] += rhs.real();
        self.parts[1] += rhs.imag();
    }
}

impl<AF: AbstractionOf<Mersenne31>> AddAssign<AF> for Mersenne31Complex<AF> {
    fn add_assign(&mut self, rhs: AF) {
        self.parts[0] += rhs;
    }
}

impl<AF: AbstractionOf<Mersenne31>> Sum for Mersenne31Complex<AF> {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.reduce(|x, y| x + y).unwrap_or(Self::ZERO)
    }
}

impl<AF: AbstractionOf<Mersenne31>> Sub for Mersenne31Complex<AF> {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self {
        Self::new(self.real() - rhs.real(), self.imag() - rhs.imag())
    }
}

impl<AF: AbstractionOf<Mersenne31>> Sub<AF> for Mersenne31Complex<AF> {
    type Output = Self;

    fn sub(self, rhs: AF) -> Self {
        Self::new(self.real() - rhs, self.imag())
    }
}

impl<AF: AbstractionOf<Mersenne31>> SubAssign for Mersenne31Complex<AF> {
    fn sub_assign(&mut self, rhs: Self) {
        self.parts[0] -= rhs.real();
        self.parts[1] -= rhs.imag();
    }
}

impl<AF: AbstractionOf<Mersenne31>> SubAssign<AF> for Mersenne31Complex<AF> {
    fn sub_assign(&mut self, rhs: AF) {
        self.parts[0] -= rhs;
    }
}

impl<AF: AbstractionOf<Mersenne31>> Neg for Mersenne31Complex<AF> {
    type Output = Self;

    fn neg(self) -> Self {
        Self::new(-self.real(), -self.imag())
    }
}

impl<AF: AbstractionOf<Mersenne31>> Mul for Mersenne31Complex<AF> {
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

impl<AF: AbstractionOf<Mersenne31>> Mul<AF> for Mersenne31Complex<AF> {
    type Output = Self;

    fn mul(self, rhs: AF) -> Self {
        Self::new(self.real() * rhs.clone(), self.imag() * rhs)
    }
}

impl<AF: AbstractionOf<Mersenne31>> MulAssign for Mersenne31Complex<AF> {
    fn mul_assign(&mut self, rhs: Self) {
        *self = self.clone() * rhs;
    }
}

impl<AF: AbstractionOf<Mersenne31>> MulAssign<AF> for Mersenne31Complex<AF> {
    fn mul_assign(&mut self, rhs: AF) {
        self.parts[0] *= rhs.clone();
        self.parts[1] *= rhs;
    }
}

impl<AF: AbstractionOf<Mersenne31>> Product for Mersenne31Complex<AF> {
    fn product<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.reduce(|x, y| x * y).unwrap_or(Self::ONE)
    }
}

impl<AF: AbstractionOf<Mersenne31>> AbstractField for Mersenne31Complex<AF> {
    const ZERO: Self = Self::new_real(AF::ZERO);
    const ONE: Self = Self::new_real(AF::ONE);
    const TWO: Self = Self::new_real(AF::TWO);
    const NEG_ONE: Self = Self::new_real(AF::NEG_ONE);

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

    fn from_wrapped_u128(n: u128) -> Self {
        Self::new_real(AF::from_wrapped_u128(n))
    }

    // sage: p = 2^31 - 1
    // sage: F = GF(p)
    // sage: R.<x> = F[]
    // sage: F2.<u> = F.extension(x^2 + 1)
    // sage: F2.multiplicative_generator()
    // u + 12
    fn multiplicative_group_generator() -> Self {
        Self::new(AF::ONE, AF::from(Mersenne31::new(12)))
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
}

impl TwoAdicField for Mersenne31Complex<Mersenne31> {
    const TWO_ADICITY: usize = 32;

    // sage: p = 2^31 - 1
    // sage: F = GF(p)
    // sage: R.<x> = F[]
    // sage: F2.<u> = F.extension(x^2 + 1)
    // sage: g = F2.multiplicative_generator()^((p^2 - 1) / 2^32); g
    // 1117296306*u + 1166849849
    // sage: assert(g.multiplicative_order() == 2^32)
    fn power_of_two_generator() -> Self {
        Self::new(
            Mersenne31::new(1_117_296_306),
            Mersenne31::new(1_166_849_849),
        )
    }
}

impl<AF: AbstractField + AbstractionOf<Mersenne31>> AbstractExtensionField<AF>
    for Mersenne31Complex<AF>
{
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

    use super::*;

    // The Euler criteria:
    // https://en.wikipedia.org/wiki/Euler%27s_criterion,
    // implies that every prime field whose order is not divisible by 4,
    // does not admit a square root of -1.
    // Over the Mersenne31 prime field (of order 2^31 - 1), we know
    // that p = (2^31 - 1) - 1 = 2^31 - 2 = 2 * (2^30 - 1), is not divisible
    // by 4. Therefore, it makes sense to consider a complex field extension,
    // F_p[X] / (X^2 + 1) = F_[i] / F_p, of the Mersennes31 field.
    type Fi = Mersenne31Complex<Mersenne31>;
    type F = Mersenne31;

    #[test]
    fn add() {
        // real part
        assert_eq!(Fi::ONE + Fi::ONE, Fi::TWO);
        assert_eq!(Fi::NEG_ONE + Fi::ONE, Fi::ZERO);
        assert_eq!(Fi::NEG_ONE + Fi::TWO, Fi::ONE);
        assert_eq!((Fi::NEG_ONE + Fi::NEG_ONE).real(), F::new(F::ORDER_U32 - 2));

        // complex part
        assert_eq!(
            Fi::new_imag(F::ONE) + Fi::new_imag(F::ONE),
            Fi::new_imag(F::TWO)
        );
        assert_eq!(
            Fi::new_imag(F::NEG_ONE) + Fi::new_imag(F::ONE),
            Fi::new_imag(F::ZERO)
        );
        assert_eq!(
            Fi::new_imag(F::NEG_ONE) + Fi::new_imag(F::TWO),
            Fi::new_imag(F::ONE)
        );
        assert_eq!(
            (Fi::new_imag(F::NEG_ONE) + Fi::new_imag(F::NEG_ONE)).imag(),
            F::new(F::ORDER_U32 - 2)
        );

        // further tests
        assert_eq!(
            Fi::new(F::ONE, F::TWO) + Fi::new(F::ONE, F::ONE),
            Fi::new(F::TWO, F::new(3))
        );
        assert_eq!(
            Fi::new(F::NEG_ONE, F::NEG_ONE) + Fi::new(F::ONE, F::ONE),
            Fi::ZERO
        );
        assert_eq!(
            Fi::new(F::NEG_ONE, F::ONE) + Fi::new(F::TWO, F::new(F::ORDER_U32 - 2)),
            Fi::new(F::ONE, F::NEG_ONE)
        );
    }

    #[test]
    fn sub() {
        // real part
        assert_eq!(Fi::ONE - Fi::ONE, Fi::ZERO);
        assert_eq!(Fi::TWO - Fi::TWO, Fi::ZERO);
        assert_eq!(Fi::NEG_ONE - Fi::NEG_ONE, Fi::ZERO);
        assert_eq!(Fi::TWO - Fi::ONE, Fi::ONE);
        assert_eq!(Fi::NEG_ONE - Fi::ZERO, Fi::NEG_ONE);

        // complex part
        assert_eq!(Fi::new_imag(F::ONE) - Fi::new_imag(F::ONE), Fi::ZERO);
        assert_eq!(Fi::new_imag(F::TWO) - Fi::new_imag(F::TWO), Fi::ZERO);
        assert_eq!(
            Fi::new_imag(F::NEG_ONE) - Fi::new_imag(F::NEG_ONE),
            Fi::ZERO
        );
        assert_eq!(
            Fi::new_imag(F::TWO) - Fi::new_imag(F::ONE),
            Fi::new_imag(F::ONE)
        );
        assert_eq!(
            Fi::new_imag(F::NEG_ONE) - Fi::ZERO,
            Fi::new_imag(F::NEG_ONE)
        );
    }

    #[test]
    fn mul() {
        assert_eq!(
            Fi::new(F::TWO, F::TWO) * Fi::new(F::new(4), F::new(5)),
            Fi::new(-F::TWO, F::new(18))
        );
    }

    #[test]
    fn mul_2exp_u64() {
        // real part
        // 1 * 2^0 = 1.
        assert_eq!(Fi::ONE.mul_2exp_u64(0), Fi::ONE);
        // 2 * 2^30 = 2^31 = 1.
        assert_eq!(Fi::TWO.mul_2exp_u64(30), Fi::ONE);
        // 5 * 2^2 = 20.
        assert_eq!(
            Fi::new_real(F::new(5)).mul_2exp_u64(2),
            Fi::new_real(F::new(20))
        );

        // complex part
        // i * 2^0 = i.
        assert_eq!(Fi::new_imag(F::ONE).mul_2exp_u64(0), Fi::new_imag(F::ONE));
        // (2i) * 2^30 = (2^31) * i = i.
        assert_eq!(Fi::new_imag(F::TWO).mul_2exp_u64(30), Fi::new_imag(F::ONE));
        // 5i * 2^2 = 20i.
        assert_eq!(
            Fi::new_imag(F::new(5)).mul_2exp_u64(2),
            Fi::new_imag(F::new(20))
        );
    }
}
