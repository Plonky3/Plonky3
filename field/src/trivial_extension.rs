use crate::field::{Field, FieldExtension};
use core::fmt::{Debug, Display, Formatter};
use core::iter::{Product, Sum};
use core::ops::{Add, AddAssign, Div, Mul, MulAssign, Neg, Sub, SubAssign};

#[derive(Copy, Clone, Eq, PartialEq, Default, Debug)]
pub struct TrivialExtension<F: Field> {
    value: F,
}

impl<F: Field> Add<Self> for TrivialExtension<F> {
    type Output = Self;

    fn add(self, rhs: Self) -> Self {
        Self {
            value: self.value + rhs.value,
        }
    }
}

impl<F: Field> AddAssign<Self> for TrivialExtension<F> {
    fn add_assign(&mut self, rhs: Self) {
        self.value += rhs.value;
    }
}

impl<F: Field> Sum for TrivialExtension<F> {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::ZERO, |acc, x| acc + x)
    }
}

impl<F: Field> Sub<Self> for TrivialExtension<F> {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self {
        Self {
            value: self.value - rhs.value,
        }
    }
}

impl<F: Field> SubAssign<Self> for TrivialExtension<F> {
    fn sub_assign(&mut self, rhs: Self) {
        self.value -= rhs.value;
    }
}

impl<F: Field> Neg for TrivialExtension<F> {
    type Output = Self;

    fn neg(self) -> Self {
        Self { value: -self.value }
    }
}

impl<F: Field> Mul<Self> for TrivialExtension<F> {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self {
        Self {
            value: self.value * rhs.value,
        }
    }
}

impl<F: Field> MulAssign<Self> for TrivialExtension<F> {
    fn mul_assign(&mut self, rhs: Self) {
        self.value *= rhs.value;
    }
}

impl<F: Field> Product for TrivialExtension<F> {
    fn product<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::ONE, |acc, x| acc * x)
    }
}

impl<F: Field> Div<Self> for TrivialExtension<F> {
    type Output = Self;

    #[allow(clippy::suspicious_arithmetic_impl)]
    fn div(self, rhs: Self) -> Self {
        self * rhs.inverse()
    }
}

impl<F: Field> Display for TrivialExtension<F> {
    fn fmt(&self, f: &mut Formatter<'_>) -> core::fmt::Result {
        Display::fmt(&self.value, f)
    }
}

impl<F: Field> Field for TrivialExtension<F> {
    type Packing = Self; // TODO: Should there be a wrapper packing for trivial extensions?
    const ZERO: Self = Self { value: F::ZERO };
    const ONE: Self = Self { value: F::ONE };
    const TWO: Self = Self { value: F::TWO };

    fn try_inverse(&self) -> Option<Self> {
        self.value.try_inverse().map(|inv| Self { value: inv })
    }
}

impl<F: Field> FieldExtension for TrivialExtension<F> {
    type Base = F;
    const D: usize = 1;

    fn to_base_array(&self) -> [Self::Base; Self::D] {
        [self.value]
    }

    fn from_base_array(b: [Self::Base; Self::D]) -> Self {
        Self { value: b[0] }
    }

    fn from_base(b: F) -> Self {
        Self { value: b }
    }
}
