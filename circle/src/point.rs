use core::ops::{Add, AddAssign, Mul, Neg, Sub};

use p3_field::{extension::ComplexExtendable, ExtensionField, Field};

#[derive(Copy, Clone, PartialEq, Eq, Debug, Hash)]
pub struct Point<F> {
    pub x: F,
    pub y: F,
}

impl<F: Field> Point<F> {
    pub fn zero() -> Self {
        Self {
            x: F::one(),
            y: F::zero(),
        }
    }
    pub fn from_projective_line(t: F) -> Self {
        let t2 = t.square();
        let inv_denom = (F::one() + t2).try_inverse().expect("t^2 = -1");
        Self {
            x: (F::one() - t2) * inv_denom,
            y: t.double() * inv_denom,
        }
    }
    // Returns None if self.x = -1, corresponding to Inf on the projective line
    pub fn to_projective_line(&self) -> Option<F> {
        self.y.try_div(self.x + F::one())
    }

    pub fn double(self) -> Self {
        Self {
            x: self.x.square().double() - F::one(),
            y: self.x.double() * self.y,
        }
    }

    pub fn v_n(mut self, log_n: usize) -> F {
        for _ in 0..(log_n - 1) {
            self.x = self.x.square().double() - F::one();
        }
        self.x
    }
}

impl<F: ComplexExtendable> Point<F> {
    pub fn generator(log_n: usize) -> Self {
        let g = F::circle_two_adic_generator(log_n);
        Self {
            x: g.real(),
            y: g.imag(),
        }
    }
}

impl<F: Field> Neg for Point<F> {
    type Output = Self;
    fn neg(mut self) -> Self::Output {
        self.y = -self.y;
        self
    }
}

impl<F: Field, EF: ExtensionField<F>> Add<Point<F>> for Point<EF> {
    type Output = Self;
    fn add(self, rhs: Point<F>) -> Self::Output {
        Self {
            x: self.x * rhs.x - self.y * rhs.y,
            y: self.x * rhs.y + self.y * rhs.x,
        }
    }
}

impl<F: Field> AddAssign for Point<F> {
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl<F: Field, EF: ExtensionField<F>> Sub<Point<F>> for Point<EF> {
    type Output = Self;
    fn sub(self, rhs: Point<F>) -> Self::Output {
        self + (-rhs)
    }
}

impl<F: Field> Mul<usize> for Point<F> {
    type Output = Self;
    fn mul(mut self, mut rhs: usize) -> Self::Output {
        let mut res = Self::zero();
        while rhs != 0 {
            if rhs & 1 == 1 {
                res += self;
            }
            rhs >>= 1;
            self = self.double();
        }
        res
    }
}

#[cfg(test)]
mod tests {
    use p3_mersenne_31::Mersenne31;

    use super::*;

    type F = Mersenne31;
    type Pt = Point<F>;

    #[test]
    fn test_arithmetic() {
        let one = Pt::generator(3);
        assert_eq!(one - one, Pt::zero());
        assert_eq!(one + one, one * 2);
        assert_eq!(one + one + one, one * 3);
        assert_eq!(one * 7, -one);
        assert_eq!(one * 8, Pt::zero());
    }
}
