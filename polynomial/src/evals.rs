use core::fmt::Debug;
use core::ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign};
use std::usize;

use p3_field::TwoAdicField;
use rand::distributions::{Distribution, Standard};

use crate::coeffs::CyclicPolynomialCoefficients;
use crate::{
    dft, fft, AbstractCyclicPolynomial, AbstractPolynomial, AbstractPolynomialEvaluations,
};

#[derive(PartialEq, Eq, Hash, Copy, Clone, Debug)]
pub struct CyclicPolynomialEvaluations<F: TwoAdicField, const N: usize> {
    pub vals: [F; N],
}
impl<F: TwoAdicField, const N: usize> CyclicPolynomialEvaluations<F, N> {
    pub fn new(vals: [F; N]) -> Self {
        Self { vals }
    }
}
impl<F: TwoAdicField, const N: usize> Default for CyclicPolynomialEvaluations<F, N> {
    fn default() -> Self {
        Self::new([F::ZERO; N])
    }
}

impl<F: TwoAdicField, const N: usize> AddAssign for CyclicPolynomialEvaluations<F, N> {
    fn add_assign(&mut self, rhs: Self) {
        for i in 0..N {
            self.vals[i] += rhs.vals[i];
        }
    }
}
impl<F: TwoAdicField, const N: usize> Add for CyclicPolynomialEvaluations<F, N> {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        let mut result = self;
        result += rhs;
        result
    }
}
impl<F: TwoAdicField, const N: usize> SubAssign for CyclicPolynomialEvaluations<F, N> {
    fn sub_assign(&mut self, rhs: Self) {
        for i in 0..N {
            self.vals[i] -= rhs.vals[i];
        }
    }
}
impl<F: TwoAdicField, const N: usize> Sub for CyclicPolynomialEvaluations<F, N> {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        let mut result = self;
        result -= rhs;
        result
    }
}
impl<F: TwoAdicField, const N: usize> Neg for CyclicPolynomialEvaluations<F, N> {
    type Output = Self;
    fn neg(self) -> Self {
        let mut result = self;
        for i in 0..N {
            result.vals[i] = -result.vals[i];
        }
        result
    }
}

impl<F: TwoAdicField, const N: usize> Mul for CyclicPolynomialEvaluations<F, N> {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        let mut result = self;
        result *= rhs;
        result
    }
}

impl<F: TwoAdicField, const N: usize> MulAssign for CyclicPolynomialEvaluations<F, N> {
    fn mul_assign(&mut self, rhs: Self) {
        for i in 0..N {
            self.vals[i] *= rhs.vals[i];
        }
    }
}

impl<F: TwoAdicField, const N: usize> AbstractPolynomial<F> for CyclicPolynomialEvaluations<F, N> {}

impl<F: TwoAdicField, const N: usize> Distribution<CyclicPolynomialEvaluations<F, N>> for Standard
where
    Standard: Distribution<F>,
{
    fn sample<R: rand::Rng + ?Sized>(&self, rng: &mut R) -> CyclicPolynomialEvaluations<F, N> {
        let mut result = CyclicPolynomialEvaluations::<F, N>::default();
        for i in 0..N {
            result.vals[i] = rng.gen();
        }
        result
    }
}

impl<F: TwoAdicField, const N: usize> AbstractCyclicPolynomial<F, N>
    for CyclicPolynomialEvaluations<F, N>
{
}

impl<F: TwoAdicField, const N: usize>
    AbstractPolynomialEvaluations<F, CyclicPolynomialCoefficients<F, N>>
    for CyclicPolynomialEvaluations<F, N>
{
    fn ifft(&self) -> CyclicPolynomialCoefficients<F, N> {
        let mut vals = self.vals;
        fft::ifft(&mut vals);

        CyclicPolynomialCoefficients::new(vals)
    }
    fn idft(&self) -> CyclicPolynomialCoefficients<F, N> {
        let vals = dft::idft(self.vals);
        CyclicPolynomialCoefficients::new(vals)
    }
}
