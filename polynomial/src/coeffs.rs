use core::fmt::Debug;
use core::ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign};
use std::usize;

use p3_field::TwoAdicField;
use rand::distributions::{Distribution, Standard};

use crate::evals::CyclicPolynomialEvaluations;
use crate::{
    dft, fft, AbstractCyclicPolynomial, AbstractPolynomial, AbstractPolynomialCoefficients,
};

#[derive(PartialEq, Eq, Hash, Copy, Clone, Debug)]
pub struct CyclicPolynomialCoefficients<F: TwoAdicField, const N: usize> {
    pub vals: [F; N],
}

impl<F: TwoAdicField, const N: usize> CyclicPolynomialCoefficients<F, N> {
    pub fn new(vals: [F; N]) -> Self {
        Self { vals }
    }
    pub fn from_vec(vals: Vec<F>) -> Self {
        let mut result = Self::default();
        for (i, v) in vals.iter().enumerate() {
            result.vals[i % N] += *v;
        }
        result
    }
}
impl<F: TwoAdicField, const N: usize> Default for CyclicPolynomialCoefficients<F, N> {
    fn default() -> Self {
        Self::new([F::ZERO; N])
    }
}
impl<F: TwoAdicField, const N: usize> AddAssign for CyclicPolynomialCoefficients<F, N> {
    fn add_assign(&mut self, rhs: Self) {
        for i in 0..N {
            self.vals[i] += rhs.vals[i];
        }
    }
}
impl<F: TwoAdicField, const N: usize> Add for CyclicPolynomialCoefficients<F, N> {
    type Output = Self;
    fn add(self, rhs: Self) -> Self::Output {
        let mut result = self;
        result += rhs;
        result
    }
}
impl<F: TwoAdicField, const N: usize> SubAssign for CyclicPolynomialCoefficients<F, N> {
    fn sub_assign(&mut self, rhs: Self) {
        for i in 0..N {
            self.vals[i] -= rhs.vals[i];
        }
    }
}
impl<F: TwoAdicField, const N: usize> Sub for CyclicPolynomialCoefficients<F, N> {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self::Output {
        let mut result = self;
        result -= rhs;
        result
    }
}
impl<F: TwoAdicField, const N: usize> Neg for CyclicPolynomialCoefficients<F, N> {
    type Output = Self;
    fn neg(self) -> Self {
        let mut result = self;
        for i in 0..N {
            result.vals[i] = -result.vals[i];
        }
        result
    }
}
impl<F: TwoAdicField, const N: usize> Mul for CyclicPolynomialCoefficients<F, N> {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self {
        let mut result = Self::default();
        for i in 0..N {
            for j in 0..N {
                result.vals[(i + j) % N] += self.vals[i] * rhs.vals[j];
            }
        }
        result
    }
}
impl<F: TwoAdicField, const N: usize> MulAssign for CyclicPolynomialCoefficients<F, N> {
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}
impl<F: TwoAdicField, const N: usize> AbstractPolynomial<F> for CyclicPolynomialCoefficients<F, N> {}

impl<F: TwoAdicField, const N: usize> Distribution<CyclicPolynomialCoefficients<F, N>> for Standard
where
    Standard: Distribution<F>,
{
    fn sample<R: rand::Rng + ?Sized>(&self, rng: &mut R) -> CyclicPolynomialCoefficients<F, N> {
        let mut result = CyclicPolynomialCoefficients::<F, N>::default();
        for i in 0..N {
            result.vals[i] = rng.gen();
        }
        result
    }
}

impl<F: TwoAdicField, const N: usize> AbstractCyclicPolynomial<F, N>
    for CyclicPolynomialCoefficients<F, N>
{
}

impl<F: TwoAdicField, const N: usize>
    AbstractPolynomialCoefficients<F, CyclicPolynomialEvaluations<F, N>>
    for CyclicPolynomialCoefficients<F, N>
{
    fn eval(&self, x: F) -> F {
        let mut result = F::ZERO;
        let mut x_pow = F::ONE;
        for i in 0..N {
            result += self.vals[i] * x_pow;
            x_pow *= x;
        }
        result
    }
    fn fft(&self) -> CyclicPolynomialEvaluations<F, N> {
        let mut vals = self.vals;
        fft::fft(&mut vals);

        CyclicPolynomialEvaluations::new(vals)
    }
    fn dft(&self) -> CyclicPolynomialEvaluations<F, N> {
        let vals = dft::dft(self.vals);
        CyclicPolynomialEvaluations::new(vals)
    }
}
