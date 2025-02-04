#![no_std]

extern crate alloc;

use core::clone::Clone;
use core::iter;
use core::iter::Product;
use core::ops::{Add, AddAssign, Div, Mul, Neg, Sub};

use alloc::{vec, vec::Vec};
use itertools::Itertools;
use p3_dft::{NaiveDft, TwoAdicSubgroupDft};
use p3_field::{Field, TwoAdicField};
use p3_matrix::dense::RowMajorMatrix;
use p3_matrix::Matrix;

#[cfg(test)]
mod tests;

#[cfg(any(test, feature = "test-utils"))]
pub mod test_utils;

/// Polynomial stored as a list of coefficients
#[derive(Clone, PartialEq, Eq, Hash, Default, Debug)]
pub struct Polynomial<F: Field> {
    // The coefficient of `x^i` is stored at location `i` in `self.coeffs`.
    coeffs: Vec<F>,
}

impl<F: Field> Polynomial<F> {
    pub fn coeffs(&self) -> &[F] {
        &self.coeffs
    }

    /// Returns the zero polynomial
    pub fn zero() -> Self {
        Self { coeffs: vec![] }
    }

    /// Returns the constant polynomial 1
    pub fn one() -> Self {
        Self::constant(F::ONE)
    }

    // Returns the constant polynomial with the given constant term
    pub fn constant(constant: F) -> Self {
        Self {
            coeffs: vec![constant],
        }
    }

    /// Returns the monic polynomial of degree 1 with no constant term
    pub fn x() -> Self {
        Self {
            coeffs: vec![F::ZERO, F::ONE],
        }
    }

    /// Returns the linear polynomial x - point
    pub fn vanishing_linear_polynomial(point: F) -> Self {
        Self {
            coeffs: vec![-point, F::ONE],
        }
    }

    /// Returns the quotient and remainder of the division of `polynomial` by `x
    /// - point`. The remainder is the constant polynomial `polynomial(point)`,
    /// and it is returned as an element of the field for convenience.
    pub fn divide_by_vanishing_linear_polynomial(polynomial: &Self, point: F) -> (Self, F) {
        let mut quotient_coeffs = polynomial.coeffs().to_vec();

        let mut last = *quotient_coeffs.iter().last().unwrap();

        for new_c in quotient_coeffs.iter_mut().rev().skip(1) {
            *new_c += point * last;
            last = *new_c;
        }

        let remainder = quotient_coeffs.remove(0);

        (Polynomial::from_coeffs(quotient_coeffs), remainder)
    }

    pub fn from_coeffs(coeffs: Vec<F>) -> Self {
        Self { coeffs }.truncate_leading_zeros()
    }

    pub fn constant_term(&self) -> F {
        *self.coeffs.first().unwrap_or(&F::ZERO)
    }

    fn truncate_leading_zeros(mut self) -> Self {
        if self.is_zero() || !self.coeffs.last().unwrap().is_zero() {
            return self;
        }

        let mut leading_index = self.coeffs.len() - 1;

        while self.coeffs[leading_index].is_zero() {
            if leading_index == 0 {
                return Self::zero();
            }

            leading_index -= 1;
        }

        self.coeffs.truncate(leading_index + 1);

        self
    }

    // Horner's method for polynomial evaluation
    fn horner_evaluate(poly_coeffs: &[F], point: &F) -> F {
        poly_coeffs
            .iter()
            .rfold(F::ZERO, move |result, coeff| result * *point + *coeff)
    }

    pub fn evaluate(&self, point: &F) -> F {
        if self.is_zero() {
            return F::ZERO;
        }
        Self::horner_evaluate(&self.coeffs, point)
    }

    pub fn degree(&self) -> Option<usize> {
        if self.is_zero() {
            None
        } else {
            Some(self.coeffs.len() - 1)
        }
    }

    pub fn is_zero(&self) -> bool {
        self.coeffs.is_empty()
    }

    pub fn is_constant(&self) -> bool {
        self.coeffs.len() <= 1
    }

    pub fn divide_with_q_and_r(&self, divisor: &Self) -> (Self, Self) {
        assert!(!divisor.is_zero(), "Divisor is zero");

        let d_deg = divisor.degree().unwrap();

        if self.is_zero() {
            return (Self::zero(), Self::zero());
        } else if self.degree() < divisor.degree() {
            return (Self::zero(), self.clone());
        }

        let mut quotient_coeffs =
            vec![F::ZERO; self.degree().unwrap() - divisor.degree().unwrap() + 1];
        let mut remainder = self.clone();

        let divisor_leading_coeff_inv = divisor.coeffs.last().unwrap().inverse();

        while !remainder.is_zero() && remainder.degree().unwrap() >= d_deg {
            let cur_q_coeff = *remainder.coeffs.last().unwrap() * divisor_leading_coeff_inv;
            let cur_q_degree = remainder.degree().unwrap() - d_deg;
            quotient_coeffs[cur_q_degree] = cur_q_coeff;

            for (i, div_coeff) in divisor.coeffs.iter().cloned().enumerate() {
                remainder.coeffs[cur_q_degree + i] -= cur_q_coeff * div_coeff;
            }
            while let Some(true) = remainder.coeffs.last().map(|c| c.is_zero()) {
                remainder.coeffs.pop();
            }
        }

        (Polynomial::from_coeffs(quotient_coeffs), remainder)
    }
}

impl<F: TwoAdicField> Polynomial<F> {
    // NP TODO: Confirm the naive algorithm is best
    // 1 (naive)
    // (x - x_0)(x - x_1): four products, results in 3 coefficients
    // [(x - x_0)(x - x_1)] * (x - x_1): six products, results in 4 coefficients
    // [[(x - x_0)(x - x_1)] * (x - x_1)] * (x - x_2): eight products, results in 5 coefficients

    // 2 (as it works now)
    // (x - x_0)(x - x_1): 2 FFT of size 4, 4 products, 1 IDFT of size 4                                (4
    // [(x - x_0)(x - x_1)] * (x - x_1): 2 FFT of size 4, 4 products, 1 IDFT of size 4                  (4
    // [[(x - x_0)(x - x_1)] * (x - x_1)] * (x - x_2): 2 FFT of size 8, 8 products, 1 IDFT of size 8    (8
    // another 3 times: 2 FFT of size 8, 8 products, 1 IDFT of size 8

    // 2.5:
    // (worse)

    // 2.75: Tree version of 2.5

    // 3 (bad)
    // FFT:
    // n times FFT of size n = n * n * log(n)
    // n products each of size n = n^2
    // one time FFT of size n = n * log(n)

    // NP TODO doc
    // mention empty lists are mapped to zero
    // mention dedup, careful with the expected degree!
    pub fn vanishing_polynomial(points: impl IntoIterator<Item = F>) -> Polynomial<F> {
        let mut points = points.into_iter().unique().collect_vec();

        if points.is_empty() {
            panic!("The vanishing polynomial of an empty set is undefined");
        }

        let mut coeffs = vec![-points.pop().unwrap(), F::ONE];

        while let Some(point) = points.pop() {
            // Basic idea: add shifted and scaled versions of the current polynomial
            // For instance, if f has coefficients
            //   [2, -3, 4, 1],
            // then (x - 5) * f has coefficients
            //   [0, 2, -3, 4, 1] + (-5) * [2, -3, 4, 1, 0]

            let mut prev_coeff = F::ZERO;

            for coeff in coeffs.iter_mut() {
                let current_coeff = *coeff;
                *coeff = prev_coeff - *coeff * point;
                prev_coeff = current_coeff;
            }

            coeffs.push(F::ONE);
        }

        Polynomial::from_coeffs(coeffs)
    }

    // NP TODO lagrange_interpolate_and_eval(

    /// Returns the unique polynomial of degree less than `point_to_evals.len()`
    /// that evaluates to `y_i` at `x_i`, where `(x_i, y_i)` refers to
    /// `point_to_evals[i]`. If two points in `point_to_evals` have the same x
    /// coordinate, the following happens:
    /// - If their evaluations do not match, the function `panic`s.
    /// - If their evaluations match, the function proceeds normally (note that
    ///   this reduces expected degree by one).
    ///
    /// This allows the function to be called in situations where, for instance,
    /// the x coordinates are generated randomly and the evaluations come from
    /// evaluating a (larger-degree) polynomial.
    ///
    /// This function uses naive Lagrange interpolation and is not optimal (cost
    /// `O(n^2)`)
    pub fn lagrange_interpolation(point_to_evals: Vec<(F, F)>) -> Polynomial<F> {
        // Testing for consistency and removing duplicate points
        let point_to_evals = point_to_evals.into_iter().unique().collect_vec();

        let points = point_to_evals
            .iter()
            .map(|(x, _)| *x)
            .unique()
            .collect_vec();

        if points.len() != point_to_evals.len() {
            panic!("Two points with the same x coordinate have different evaluations");
        }

        // Computing interpolator
        let vanishing_poly = Self::vanishing_polynomial(points);

        let mut result = Polynomial::zero();

        for (point, eval) in point_to_evals.into_iter() {
            let polynomial = &vanishing_poly / &Polynomial::vanishing_linear_polynomial(point);
            let denominator = polynomial.evaluate(&point);
            result += &(&polynomial * &(eval / denominator));
        }

        result
    }

    /// Given f(x) and e, returns f(x^e)
    pub fn compose_with_exponent(&self, exponent: usize) -> Polynomial<F> {
        let d = if let Some(d) = self.degree() {
            d
        } else {
            return Polynomial::zero();
        };

        let mut coeffs = vec![F::ZERO; d * exponent + 1];
        for (i, coeff) in self.coeffs.iter().enumerate() {
            coeffs[i * exponent] = *coeff;
        }
        Polynomial::from_coeffs(coeffs)
    }

    // Compute the scaling polynomial, 1 + rx + r^2 x^2 + ... + r^n x^n with n = |quotient_set|
    pub fn power_polynomial(r: F, degree: usize) -> Polynomial<F> {
        Polynomial::from_coeffs(
            iter::successors(Some(F::ONE), |&prev| Some(prev * r))
                .take(degree + 1)
                .collect_vec(),
        )
    }
}

impl<'a, 'b, F: Field> Add<&'a Polynomial<F>> for &'b Polynomial<F> {
    type Output = Polynomial<F>;

    fn add(self, other: &'a Polynomial<F>) -> Polynomial<F> {
        if self.is_zero() {
            return other.clone();
        } else if other.is_zero() {
            return self.clone();
        };

        let (mut high, low) = if self.degree() >= other.degree() {
            (self.clone(), other.clone())
        } else {
            (other.clone(), self.clone())
        };

        high.coeffs.iter_mut().zip(&low.coeffs).for_each(|(a, b)| {
            *a += *b;
        });

        high.truncate_leading_zeros()
    }
}

impl<F: Field> AddAssign<&Polynomial<F>> for Polynomial<F> {
    fn add_assign(&mut self, other: &Polynomial<F>) {
        *self = &*self + other;
    }
}

impl<F: Field> Neg for &Polynomial<F> {
    type Output = Polynomial<F>;

    #[inline]
    fn neg(self) -> Polynomial<F> {
        Polynomial {
            coeffs: self.coeffs.iter().map(|c| -*c).collect(),
        }
    }
}

impl<F: Field> Sub<&Polynomial<F>> for &Polynomial<F> {
    type Output = Polynomial<F>;

    fn sub(self, other: &Polynomial<F>) -> Polynomial<F> {
        self + &(-other)
    }
}

impl<F: TwoAdicField> Mul<&Polynomial<F>> for &Polynomial<F> {
    type Output = Polynomial<F>;

    // NP TODO: Definitely a better way to do this
    fn mul(self, other: &Polynomial<F>) -> Polynomial<F> {
        if self.is_zero() || other.is_zero() {
            return Polynomial::zero();
        }

        if self.is_constant() {
            return Polynomial::from_coeffs(
                other
                    .coeffs
                    .iter()
                    .map(|c| *c * self.coeffs[0])
                    .collect_vec(),
            );
        }

        if other.is_constant() {
            return Polynomial::from_coeffs(
                self.coeffs
                    .iter()
                    .map(|c| *c * other.coeffs[0])
                    .collect_vec(),
            );
        }

        // NP TODO add check that FFT fits into field; ow use traditional algorithm
        let mut extended_self = self.coeffs.clone();
        let mut extended_other = other.coeffs.clone();

        let domain_size = (self.coeffs.len() + other.coeffs.len() - 1).next_power_of_two();
        extended_self.resize(domain_size, F::ZERO);
        extended_other.resize(domain_size, F::ZERO);

        // NP TODO transposing?
        let coeffs = RowMajorMatrix::new(
            extended_self.into_iter().chain(extended_other).collect(),
            domain_size,
        )
        .transpose();

        let dft: RowMajorMatrix<F> = NaiveDft.dft_batch(coeffs).transpose();

        let (first_row, second_row) = (dft.first_row(), dft.last_row());
        let pointwise_multiplication = first_row
            .zip(second_row)
            .map(|(a, b): (F, F)| a * b)
            .collect_vec();

        let pointwise_multiplication =
            RowMajorMatrix::new(pointwise_multiplication, domain_size).transpose();

        let inverse_dft = NaiveDft.idft_batch(pointwise_multiplication);

        Polynomial::from_coeffs(inverse_dft.values.clone())
    }
}

impl<F: TwoAdicField> Div<&Polynomial<F>> for &Polynomial<F> {
    type Output = Polynomial<F>;

    // NP TODO think about FFT
    fn div(self, other: &Polynomial<F>) -> Polynomial<F> {
        let (q, r) = self.divide_with_q_and_r(other);
        assert!(
            r.is_zero(),
            "Polynomial division failed, remainder is not zero"
        );
        q
    }
}

impl<F: TwoAdicField> Product<Polynomial<F>> for Polynomial<F> {
    fn product<I: Iterator<Item = Polynomial<F>>>(iter: I) -> Self {
        iter.fold(Polynomial::one(), |acc, p| &acc * &p)
    }
}

impl<F: Field> Add<&F> for &Polynomial<F> {
    type Output = Polynomial<F>;

    fn add(self, other: &F) -> Polynomial<F> {
        self + &Polynomial::from_coeffs(vec![*other])
    }
}

impl<F: Field> Sub<&F> for &Polynomial<F> {
    type Output = Polynomial<F>;

    fn sub(self, other: &F) -> Polynomial<F> {
        self - &Polynomial::from_coeffs(vec![*other])
    }
}

impl<F: TwoAdicField> Mul<&F> for &Polynomial<F> {
    type Output = Polynomial<F>;

    fn mul(self, other: &F) -> Polynomial<F> {
        self * &Polynomial::from_coeffs(vec![*other])
    }
}
