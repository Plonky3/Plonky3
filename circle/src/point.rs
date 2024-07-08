use core::ops::{Add, AddAssign, Mul, Neg, Sub};

use p3_field::extension::ComplexExtendable;
use p3_field::{ExtensionField, Field};

/// Affine representation of a point on the circle.
/// x^2 + y^2 == 1
// _private is to prevent construction so we can debug assert the invariant
#[allow(clippy::manual_non_exhaustive)]
#[derive(Copy, Clone, PartialEq, Eq, Debug, Hash)]
pub struct Point<F> {
    pub x: F,
    pub y: F,
    _private: (),
}

impl<F: Field> Point<F> {
    #[inline]
    pub fn new(x: F, y: F) -> Point<F> {
        debug_assert_eq!(x.square() + y.square(), F::one());
        Point { x, y, _private: () }
    }
    pub fn zero() -> Self {
        Self::new(F::one(), F::zero())
    }

    /// Circle STARKs, Section 3, Lemma 1: (page 4 of the first revision PDF)
    /// ```ignore
    /// (x, y) = ((1-t^2)/(1+t^2), 2t/(1+t^2))
    /// ```
    /// Panics if t^2 = -1, corresponding to either of the points at infinity
    /// (on the projective *circle*) (1 : ±i : 0)
    pub fn from_projective_line(t: F) -> Self {
        let t2 = t.square();
        let inv_denom = (F::one() + t2).try_inverse().expect("t^2 = -1");
        Self::new((F::one() - t2) * inv_denom, t.double() * inv_denom)
    }

    /// Circle STARKs, Section 3, Lemma 1: (page 4 of the first revision PDF)
    /// ```ignore
    /// t = y / (x + 1)
    /// ```
    /// Returns None if self.x = -1, corresponding to Inf on the projective line
    ///
    /// This is also used as a selector polynomial, with a simple zero at (1,0)
    /// and a simple pole at (-1,0), which in the paper is called v_0
    /// Circle STARKs, Section 5.1, Lemma 11 (page 21 of the first revision PDF)
    pub fn to_projective_line(self) -> Option<F> {
        self.y.try_div(self.x + F::one())
    }

    /// The "squaring map", or doubling in additive notation, denoted π(x,y)
    /// Circle STARKs, Section 3.1, Equation 1: (page 5 of the first revision PDF)
    pub fn double(self) -> Self {
        Self::new(
            self.x.square().double() - F::one(),
            self.x.double() * self.y,
        )
    }

    /// Evaluate the vanishing polynomial for the standard position coset of size 2^log_n
    /// at this point
    /// Circle STARKs, Section 3.3, Equation 8 (page 10 of the first revision PDF)
    pub fn v_n(mut self, log_n: usize) -> F {
        for _ in 0..(log_n - 1) {
            self.x = self.x.square().double() - F::one();
        }
        self.x
    }

    /// Evaluate the formal derivative of `v_n` at `self`
    /// Circle STARKs, Section 5.1, Remark 15 (page 21 of the first revision PDF)
    pub fn v_n_prime(self, log_n: usize) -> F {
        F::two().exp_u64((2 * (log_n - 1)) as u64) * (1..log_n).map(|i| self.v_n(i)).product()
    }

    /// Evaluate the selector function which is zero at `self` and nonzero elsewhere, at `at`.
    /// Called v_0 . T_p⁻¹ or ṽ_p(x,y) in the paper, used for constraint selectors.
    /// Panics if p = -self, the pole.
    /// Section 5.1, Lemma 11 of Circle Starks (page 21 of first edition PDF)
    pub fn v_tilde_p<EF: ExtensionField<F>>(self, at: Point<EF>) -> EF {
        (at - self).to_projective_line().unwrap()
    }

    /// The concrete value of the selector s_P = v_n / (v_0 . T_p⁻¹) at P=self, used for normalization.
    /// Circle STARKs, Section 5.1, Remark 16 (page 22 of the first revision PDF)
    pub fn s_p_at_p(self, log_n: usize) -> F {
        -F::two() * self.v_n_prime(log_n) * self.y
    }

    /// Evaluate the alternate single-point vanishing function v_p(x), used for DEEP quotient.
    /// Returns (a, b), representing the complex number a + bi.
    /// Simple zero at p, simple pole at +-infinity.
    /// Circle STARKs, Section 3.3, Equation 11 (page 11 of the first edition PDF).
    pub fn v_p<EF: ExtensionField<F>>(self, at: Point<EF>) -> (EF, EF) {
        let diff = -at + self;
        (EF::one() - diff.x, -diff.y)
    }
}

impl<F: ComplexExtendable> Point<F> {
    pub fn generator(log_n: usize) -> Self {
        let g = F::circle_two_adic_generator(log_n);
        Self::new(g.real(), g.imag())
    }
}

/// Circle STARKs, Section 3.1, Equation 2: (page 5 of the first revision PDF)
/// The inverse map J(x,y) = (x,-y)
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
        Self::new(
            self.x * rhs.x - self.y * rhs.y,
            self.x * rhs.y + self.y * rhs.x,
        )
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
