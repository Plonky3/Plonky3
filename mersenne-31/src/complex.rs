use crate::Mersenne31;
use core::fmt::{Display, Formatter};
use core::iter::{Product, Sum};
use core::ops::{Add, AddAssign, Div, Mul, MulAssign, Neg, Sub, SubAssign};
use p3_field::field::{AbstractField, AbstractionOf, Field};

#[derive(PartialEq, Eq, Copy, Clone, Debug, Default)]
pub struct ComplexExtension<AF: AbstractionOf<Mersenne31>> {
    real: AF,
    imag: AF,
}

impl<AF: AbstractionOf<Mersenne31>> ComplexExtension<AF> {
    pub const fn new(real: AF, imag: AF) -> Self {
        Self { real, imag }
    }

    pub const fn real(real: AF) -> Self {
        Self::new(real, AF::ZERO)
    }

    pub const fn imag(imag: AF) -> Self {
        Self::new(AF::ZERO, imag)
    }
}

impl<AF: AbstractionOf<Mersenne31>> Add<Self> for ComplexExtension<AF> {
    type Output = Self;

    fn add(self, rhs: Self) -> Self {
        Self::new(self.real + rhs.real, self.imag + rhs.imag)
    }
}

impl<AF: AbstractionOf<Mersenne31>> Add<Mersenne31> for ComplexExtension<AF> {
    type Output = Self;

    fn add(self, rhs: Mersenne31) -> Self {
        Self::new(self.real + AF::from(rhs), self.imag)
    }
}

impl<AF: AbstractionOf<Mersenne31>> AddAssign<Self> for ComplexExtension<AF> {
    fn add_assign(&mut self, rhs: Self) {
        self.real += rhs.real;
        self.imag += rhs.imag;
    }
}

impl<AF: AbstractionOf<Mersenne31>> AddAssign<Mersenne31> for ComplexExtension<AF> {
    fn add_assign(&mut self, rhs: Mersenne31) {
        self.real += AF::from(rhs);
    }
}

impl<AF: AbstractionOf<Mersenne31>> Sum for ComplexExtension<AF> {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.reduce(|x, y| x + y).unwrap_or(Self::ZERO)
    }
}

impl<AF: AbstractionOf<Mersenne31>> Sub<Self> for ComplexExtension<AF> {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self {
        Self::new(self.real - rhs.real, self.imag - rhs.imag)
    }
}

impl<AF: AbstractionOf<Mersenne31>> Sub<Mersenne31> for ComplexExtension<AF> {
    type Output = Self;

    fn sub(self, rhs: Mersenne31) -> Self {
        Self::new(self.real - rhs, self.imag)
    }
}

impl<AF: AbstractionOf<Mersenne31>> SubAssign<Self> for ComplexExtension<AF> {
    fn sub_assign(&mut self, rhs: Self) {
        self.real -= rhs.real;
        self.imag -= rhs.imag;
    }
}

impl<AF: AbstractionOf<Mersenne31>> SubAssign<Mersenne31> for ComplexExtension<AF> {
    fn sub_assign(&mut self, rhs: Mersenne31) {
        self.real -= rhs;
    }
}

impl<AF: AbstractionOf<Mersenne31>> Neg for ComplexExtension<AF> {
    type Output = Self;

    fn neg(self) -> Self {
        Self::new(-self.real, -self.imag)
    }
}

impl<AF: AbstractionOf<Mersenne31>> Mul<Self> for ComplexExtension<AF> {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self {
        // TODO: Try the Gauss algorithm for complex multiplication, see if it's faster for us.
        // Method for (a + b i) * (c + d i):
        //     t1 = a * c
        //     t2 = b * d
        //     real = t1 - t2
        //     imag = (a + b) * (c + d) - t1 - t2

        let real = self.real.clone() * rhs.real.clone() - self.imag.clone() * rhs.imag.clone();
        let imag = self.real * rhs.imag + self.imag * rhs.real;
        Self::new(real, imag)
    }
}

impl<AF: AbstractionOf<Mersenne31>> Mul<Mersenne31> for ComplexExtension<AF> {
    type Output = Self;

    fn mul(self, rhs: Mersenne31) -> Self {
        Self::new(self.real * rhs, self.imag * rhs)
    }
}

impl<AF: AbstractionOf<Mersenne31>> MulAssign<Self> for ComplexExtension<AF> {
    fn mul_assign(&mut self, rhs: Self) {
        *self = self.clone() * rhs;
    }
}

impl<AF: AbstractionOf<Mersenne31>> MulAssign<Mersenne31> for ComplexExtension<AF> {
    fn mul_assign(&mut self, rhs: Mersenne31) {
        self.real *= rhs;
        self.imag *= rhs;
    }
}

impl<AF: AbstractionOf<Mersenne31>> Product for ComplexExtension<AF> {
    fn product<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.reduce(|x, y| x * y).unwrap_or(Self::ONE)
    }
}

impl<AF: AbstractionOf<Mersenne31>> AbstractField for ComplexExtension<AF> {
    const ZERO: Self = Self::real(AF::ZERO);
    const ONE: Self = Self::real(AF::ONE);
    const TWO: Self = Self::real(AF::TWO);
    const NEG_ONE: Self = Self::real(AF::NEG_ONE);

    // sage: p = 2^31 - 1
    // sage: F = GF(p)
    // sage: F2.<u> = F.extension(x^2 + 1)
    // sage: F2.multiplicative_generator()
    // u + 12
    const MULTIPLICATIVE_GROUP_GENERATOR: Self = Self::ZERO; // TODO Self::new(AF::ONE, AF::from(Mersenne31::new(12)));
}

impl Div<Self> for ComplexExtension<Mersenne31> {
    type Output = Self;

    fn div(self, rhs: Self) -> Self::Output {
        self * rhs.inverse()
    }
}

impl Display for ComplexExtension<Mersenne31> {
    fn fmt(&self, f: &mut Formatter<'_>) -> core::fmt::Result {
        write!(f, "{} + {}i", self.real, self.imag)
    }
}

impl Field for ComplexExtension<Mersenne31> {
    type Packing = Self;

    fn try_inverse(&self) -> Option<Self> {
        todo!()
    }
}

// impl<AF: AbstractField<Mersenne31>> FieldExtension<AF> for ComplexExt<AF> {}
