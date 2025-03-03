//! Interface for working with dense univariate polynomials. It offers
//! polynomial arithmetic, Lagrange interpolation, vanishing polynomials and
//! many other convenience methods.
//!

// N. B.: The standard operators are implemented on references to polynomials
// only.

#![no_std]

extern crate alloc;

use alloc::vec;
use alloc::vec::Vec;
use core::clone::Clone;
use core::iter::Product;
use core::ops::{Add, AddAssign, Div, Mul, Neg, Sub};

use itertools::{iterate, Itertools};
use p3_dft::{Radix2Dit, TwoAdicSubgroupDft};
use p3_field::{Field, TwoAdicField};
use p3_matrix::dense::RowMajorMatrix;
use p3_matrix::Matrix;
use serde::{Deserialize, Serialize};

#[cfg(test)]
mod tests;

#[cfg(any(test, feature = "test-utils"))]
pub mod test_utils;

/// Polynomial stored as a dense list of coefficients
///
/// # Examples
///
/// ```
/// # use p3_poly::Polynomial;
/// # use p3_field::{Field, TwoAdicField, PrimeCharacteristicRing};
/// # use p3_baby_bear::BabyBear;
/// # use rand::Rng;
/// #
/// type F = BabyBear;
/// let mut rng = rand::rng();
/// let poly = Polynomial::from_coeffs(
///     (0..10).map(|_| rng.random()).collect()
/// );
///
/// // Polynomial evaluation
/// let point = rng.random();
/// assert_eq!(
///     poly.evaluate(&point),
///     poly.coeffs().iter().rfold(F::ZERO, |result, coeff| result * point + *coeff)
/// );
///
/// // Polynomials support arithmetic operations with polynomials and field elements
/// // Note that the operations are implemented on references to polynomials and field elements
/// let other_poly = Polynomial::from_coeffs((0..10).map(|_| rng.random()).collect());
/// let constant: F = rng.random();
///
/// let _ = &poly + &other_poly;
/// let _ = &poly - &other_poly;
/// let _ = &poly + &constant;
/// let _ = &poly - &constant;
/// let _ = &poly * &constant;
/// let _ = &poly / &constant;
///
/// // Multiplication can be done using `mul_naive` which uses the
/// // naive multiplication algorithm, and `mul` which can be used if the
/// // base field is two-adic. The latter internally chooses whether to use
/// // FFTs or the naive algorithm depending on the degrees of the polynomials.
/// assert_eq!(&poly * &other_poly, poly.mul_naive(&other_poly));
///
/// // Division can be done using `divide_with_remainder`
/// // or `Div::div` (i. e. the operator `/`). Note that `Div::div` will panic
/// // if the remainder is not zero.
/// let (quotient, remainder) = poly.divide_with_remainder(&other_poly);
/// assert_eq!(quotient, &(&poly - &remainder) / &other_poly);
///
/// // Polynomials can be interpolated at an arbitrary set of points using
/// // `lagrange_interpolation`. For efficient interpolation over two-adic cosets,
/// // cf. the `p3-coset` crate.
/// assert_eq!(
///     Polynomial::lagrange_interpolation(
///         vec![(F::ONE, F::ONE), (F::TWO, F::ZERO), (F::ZERO, F::ONE)]
///     ),
///     Polynomial::from_coeffs(
///         vec![F::ONE, F::TWO.inverse(), F::TWO.inverse() - F::ONE]
///     )
/// );
///
/// // Other utility methods
/// assert_eq!(
///     Polynomial::vanishing_linear_polynomial(constant),
///     Polynomial::from_coeffs(vec![-constant, F::ONE])
/// );
/// assert_eq!(poly.compose_with_exponent(2).degree(), Some(18));
///
/// let (quotient, remainder) = poly.divide_with_remainder(
///     &Polynomial::from_coeffs(vec![-constant, F::ONE])
/// );
/// assert_eq!(
///     poly.divide_by_vanishing_linear_polynomial(constant),
///     (quotient, remainder.coeffs()[0])
/// );
/// assert_eq!(
///     Polynomial::power_polynomial(constant, 3).degree(),
///     Polynomial::from_coeffs(
///         vec![F::ONE, constant, constant.exp_const_u64::<2>(), constant.exp_const_u64::<3>()]
///     )
///     .degree()
/// );
/// ```
#[derive(Clone, PartialEq, Eq, Hash, Default, Debug, Serialize, Deserialize)]
#[serde(bound(deserialize = "Vec<F>: Deserialize<'de>",))]
pub struct Polynomial<F: Field> {
    // The coefficient of `x^i` is stored at location `i` in `self.coeffs`. It
    // is important for this field to remain private, as leading-zeroes trimming
    // is handled internally and is abstracted from the user.
    coeffs: Vec<F>,
}

impl<F: Field> Polynomial<F> {
    /// Returns the coefficients of the polynomial in increasing-degree order
    /// with no leading zeros
    pub fn coeffs(&self) -> &[F] {
        // This never has leading zeros for users of the public interface
        &self.coeffs
    }

    /// Returns the leading coefficient of the polynomial
    pub fn leading_coeff(&self) -> F {
        *self.coeffs.last().unwrap_or(&F::ZERO)
    }

    /// Returns the zero polynomial
    pub fn zero() -> Self {
        Self { coeffs: vec![] }
    }

    /// Returns the constant polynomial 1
    pub fn one() -> Self {
        Self::constant(F::ONE)
    }

    /// Returns the polynomial with the given coefficients. Leading zeros are automatically trimmed.
    pub fn from_coeffs(coeffs: Vec<F>) -> Self {
        Self { coeffs }.truncate_leading_zeros()
    }

    /// Returns the constant term of the polynomial
    pub fn constant_term(&self) -> F {
        *self.coeffs.first().unwrap_or(&F::ZERO)
    }

    /// Returns the constant polynomial with the given constant term
    pub fn constant(constant: F) -> Self {
        Self {
            coeffs: vec![constant],
        }
    }

    /// Returns the unique monic polynomial of degree 1 with no constant term
    pub fn x() -> Self {
        Self {
            coeffs: vec![F::ZERO, F::ONE],
        }
    }

    /// Returns the linear polynomial `x - point`
    pub fn vanishing_linear_polynomial(point: F) -> Self {
        Self {
            coeffs: vec![-point, F::ONE],
        }
    }

    // Internal method which eliminates leading zeros from the polynomial,
    // mutating the polynomial and returning it.
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

    /// Evaluates `self` at the given `point` using Horner's method
    pub fn evaluate(&self, point: &F) -> F {
        self.coeffs
            .iter()
            .rfold(F::ZERO, move |result, coeff| result * *point + *coeff)
    }

    /// Returns `None` if self is the zero polynomial and `Some(d)` if `self` is
    /// a (non-zero) polynomial of degree `d`
    pub fn degree(&self) -> Option<usize> {
        if self.is_zero() {
            None
        } else {
            Some(self.coeffs.len() - 1)
        }
    }

    /// Returns `true` if and only if `self` is the zero polynomial
    pub fn is_zero(&self) -> bool {
        self.coeffs.is_empty()
    }

    /// Returns `true` if and only if `self` is a constant polynomial
    pub fn is_constant(&self) -> bool {
        self.coeffs.len() <= 1
    }

    /// Returns the unique polynomials `q` and `r` such that
    /// `self = q * divisor + r` and `r` is zero or has degree less than
    /// `divisor`
    ///
    /// # Panics
    ///
    /// Panics if `divisor` is the zero polynomial
    // ** Credit to Arkworks/algebra for the core of the algorithm **
    pub fn divide_with_remainder(&self, divisor: &Self) -> (Self, Self) {
        let d_deg = divisor
            .degree()
            .expect("Cannot divide by the zero polynomial");

        // Trivial division cases
        if self.is_zero() {
            return (Self::zero(), Self::zero());
        }

        let d_self = self.degree().unwrap();

        if d_self < d_deg {
            return (Self::zero(), self.clone());
        }

        let mut quotient_coeffs = vec![F::ZERO; d_self - d_deg + 1];
        let mut remainder = self.clone();

        let divisor_leading_coeff_inv = divisor.coeffs.last().unwrap().inverse();

        // Ieratively compute the coefficients of the quotient
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

        (
            Polynomial::from_coeffs(quotient_coeffs),
            remainder.truncate_leading_zeros(),
        )
    }

    /// Returns the quotient and remainder of the division of `self` by `x -
    /// point`. Since the remainder is the constant polynomial with value
    /// `self(point)`, it is directly returned as field element for convenience.
    pub fn divide_by_vanishing_linear_polynomial(&self, point: F) -> (Self, F) {
        if self.is_zero() {
            return (Self::zero(), F::ZERO);
        }

        // Special case: division by x - 0 = 0
        if point == F::ZERO {
            let mut quotient_coeffs = self.coeffs.clone();
            let remainder = quotient_coeffs.remove(0);
            return (Polynomial::from_coeffs(quotient_coeffs), remainder);
        }

        // General case: use Ruffini's rule
        let mut quotient_coeffs = self.coeffs.clone();

        let mut quotient_coeffs_iter = quotient_coeffs.iter_mut().rev();

        let mut last = *quotient_coeffs_iter.next().unwrap();

        for new_c in quotient_coeffs_iter {
            *new_c += point * last;
            last = *new_c;
        }

        let remainder = quotient_coeffs.remove(0);

        (Polynomial::from_coeffs(quotient_coeffs), remainder)
    }

    /// Returns the unique monic polynomial of degree equal to the number
    /// of _distinct_ elements in `points` that vanishes at each
    /// of those elements, that is, `(x - distinct_points[0]) * (x -
    /// distinct_points[1]) * ... * (x - distinct_points[n - 1])`, where
    /// `distinct_points` contains the distinct elements of `points` and `n` is
    /// its length.
    ///     
    /// # Panics
    ///
    /// Panics if `points` is empty
    pub fn vanishing_polynomial(points: impl IntoIterator<Item = F>) -> Polynomial<F> {
        // Deduplicating the points
        let mut points = points.into_iter().unique().collect_vec();

        assert!(
            !points.is_empty(),
            "The vanishing polynomial of an empty set is undefined"
        );

        // We iteratively multiply the polynomial (x - points[0]) by each of the
        // vanishing polynomials (x - points[i]) for i > 0
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

    /// Returns the unique polynomial of degree less than the number of
    /// (distinct) pairs in `point_to_evals` that evaluates to `y_i` at `x_i`
    /// for all `i`, where `(x_i, y_i)` denotes `point_to_evals[i]`. If two
    /// points in `point_to_evals` have the same x coordinate, the following
    /// happens:
    /// - If their evaluations do not match, the function `panic`s.
    /// - If their evaluations match, the function proceeds normally (note that
    ///   this reduces expected degree by one).
    ///
    /// This distinction allows the function to be called transparently in
    /// situations where, for instance, the x coordinates are generated randomly
    /// and the evaluations come from evaluating a (larger-degree) polynomial.
    ///
    /// This method uses Lagrange interpolation, which has quadratic runtime in
    /// the number of (distinct) pairs in `point_to_evals`.
    ///
    /// # Panics
    ///
    /// Panics if
    ///
    /// - `point_to_evals` is empty
    /// - `point_to_evals` has two points (i. e. x coordinates) with different
    ///    requested evaluation (i. e. y coordinates)
    pub fn lagrange_interpolation(point_to_evals: Vec<(F, F)>) -> Polynomial<F> {
        if point_to_evals.is_empty() {
            panic!("The Lagrange interpolation of an empty set is undefined");
        }

        // Testing for consistency and removing duplicate points
        let point_to_evals = point_to_evals.into_iter().unique().collect_vec();

        let points = point_to_evals
            .iter()
            .map(|(x, _)| *x)
            .unique()
            .collect_vec();

        assert_eq!(
            points.len(),
            point_to_evals.len(),
            "One point has two different requested evaluations"
        );

        let vanishing_poly = Self::vanishing_polynomial(points);

        let mut result = Polynomial::zero();

        for (point, eval) in point_to_evals.into_iter() {
            // We obtain the (non-normalised) vanishing polynomial at all points
            // other than point by removing the (x - point) factor from the full
            // vanishing polynomial
            let (polynomial, _) = vanishing_poly.divide_by_vanishing_linear_polynomial(point);

            // We normalise it so that it takes the value `eval` at `point`
            let denominator = polynomial.evaluate(&point);
            result += &(&polynomial * &(eval / denominator));
        }

        result
    }

    /// Returns the composition of `self` with the polynomial `x^exponent`. In
    /// other words, if `self` is given by `f(x)`, the result is `f(x^exponent)`.
    pub fn compose_with_exponent(&self, exponent: usize) -> Polynomial<F> {
        let d = if let Some(d) = self.degree() {
            d
        } else {
            return Polynomial::zero();
        };

        // We "stretch out" the vector of coefficients by a factor of exponent
        // filling the gaps with zeros
        let mut coeffs = vec![F::ZERO; d * exponent + 1];

        for (i, coeff) in self.coeffs.iter().enumerate() {
            coeffs[i * exponent] = *coeff;
        }

        Polynomial::from_coeffs(coeffs)
    }

    /// Returns the polynomial `1 + r * x + r^2 * x^2 + ... + r^degree * x^degree`
    pub fn power_polynomial(r: F, degree: usize) -> Polynomial<F> {
        if r == F::ZERO {
            Polynomial::one()
        } else {
            Polynomial::from_coeffs(iterate(F::ONE, |&prev| prev * r).take(degree + 1).collect())
        }
    }

    /// Multiplies `self` and `other` using the standard naive algorithm (i. e.
    /// the Cauchy product of the coefficients). If `F: TwoAdicField`, instead
    /// consider using [`mul`](Mul::mul) or, equivalently, the operator `*`, which selects
    /// the naive algorithm or the FFT depending on the degrees of the two
    /// factors.
    pub fn mul_naive(&self, other: &Self) -> Self {
        if self.is_zero() || other.is_zero() {
            return Self::zero();
        }

        let mut coeffs = vec![F::ZERO; self.coeffs.len() + other.coeffs.len() - 1];

        for (i, &c1) in self.coeffs.iter().enumerate() {
            for (j, &c2) in other.coeffs.iter().enumerate() {
                coeffs[i + j] += c1 * c2;
            }
        }

        Polynomial::from_coeffs(coeffs)
    }
}

impl<'a, F: Field> Add<&'a Polynomial<F>> for &Polynomial<F> {
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

        high.coeffs.iter_mut().zip(&low.coeffs).for_each(|(a, &b)| {
            *a += b;
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
            coeffs: self.coeffs.iter().map(|&c| -c).collect(),
        }
    }
}

impl<F: Field> Sub<&Polynomial<F>> for &Polynomial<F> {
    type Output = Polynomial<F>;

    fn sub(self, other: &Polynomial<F>) -> Polynomial<F> {
        self + &(-other)
    }
}

/// Multiply the two polynomials using FFTs or the naive multiplication
/// algorithm depending on what is expected to be faster based on their
/// degrees.
impl<F: TwoAdicField> Mul<&Polynomial<F>> for &Polynomial<F> {
    type Output = Polynomial<F>;

    fn mul(self, other: &Polynomial<F>) -> Polynomial<F> {
        if self.is_zero() || other.is_zero() {
            return Polynomial::zero();
        }

        let d_self = self.degree().unwrap();
        let d_other = other.degree().unwrap();

        let fft_domain_size = (d_self + d_other + 1).next_power_of_two();
        let fft_domain_size_log = fft_domain_size.ilog2() as usize;

        // This is only a rough estimate to avoid doing three [i]FFTs in very
        // imbalanced cases (such as large poly times constant or deg-two poly).
        // Only multiplications are taken into account.
        let fft_cost = 3 * fft_domain_size * fft_domain_size_log + fft_domain_size;
        let naive_cost = (d_self + 1) * (d_other + 1);

        // We also use the naive algorithm in the unlikely case the poylnomials
        // are so large that the two-adicity of F* does not support an FFT
        // therein
        if fft_cost > naive_cost || fft_domain_size_log > F::TWO_ADICITY {
            return self.mul_naive(other);
        }

        let mut extended_self = self.coeffs.clone();
        let mut extended_other = other.coeffs.clone();

        extended_self.resize(fft_domain_size, F::ZERO);
        extended_other.resize(fft_domain_size, F::ZERO);

        let coeffs = RowMajorMatrix::new(
            extended_self.into_iter().chain(extended_other).collect(),
            fft_domain_size,
        )
        .transpose();

        let fft = Radix2Dit::default();

        // Evaluate the polynomials over the domain
        let dft: RowMajorMatrix<F> = fft.dft_batch(coeffs).transpose();

        // Multiply the polynomial evaluations pointwise
        let eval_products = dft
            .first_row()
            .zip(dft.last_row())
            .map(|(a, b): (F, F)| a * b)
            .collect_vec();

        // Interpolating the evaluations with an inverse FFT
        Polynomial::from_coeffs(fft.idft(eval_products))
    }
}

/// Exact polynomial division (using the classical algorithm).
///
/// # Panics
///
/// Panics if the remainder is not zero. If this is not guaranteed, use
/// [`divide_with_remainder`](Polynomial::divide_with_remainder) instead.
impl<F: TwoAdicField> Div<&Polynomial<F>> for &Polynomial<F> {
    type Output = Polynomial<F>;

    fn div(self, other: &Polynomial<F>) -> Polynomial<F> {
        let (q, r) = self.divide_with_remainder(other);
        assert!(
            r.is_zero(),
            "Non-zero remainder is not zero. Consider using `divide_with_remainder` instead."
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

impl<F: Field> Mul<&F> for &Polynomial<F> {
    type Output = Polynomial<F>;

    fn mul(self, other: &F) -> Polynomial<F> {
        Polynomial::from_coeffs(self.coeffs.iter().map(|&c| c * *other).collect())
    }
}

impl<F: Field> Div<&F> for &Polynomial<F> {
    type Output = Polynomial<F>;

    fn div(self, other: &F) -> Polynomial<F> {
        Polynomial::from_coeffs(self.coeffs.iter().map(|&c| c * other.inverse()).collect())
    }
}
