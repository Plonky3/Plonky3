use crate::Mersenne31;
use core::fmt::{Display, Formatter};
use core::iter::{Product, Sum};
use core::ops::{Add, AddAssign, Div, Mul, MulAssign, Neg, Sub, SubAssign};
use p3_field::{AbstractExtensionField, AbstractField, AbstractionOf, Field, TwoAdicField};

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
        todo!()
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
        Self::new(Mersenne31::new(1117296306), Mersenne31::new(1166849849))
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
