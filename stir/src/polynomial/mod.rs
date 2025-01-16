use core::clone::Clone;
use core::iter::Product;
use core::ops::{Add, AddAssign, Div, Mul, Neg, Sub};

use itertools::Itertools;
use p3_dft::{NaiveDft, TwoAdicSubgroupDft};
use p3_field::{Field, TwoAdicField};
use p3_matrix::dense::RowMajorMatrix;
use p3_matrix::Matrix;

#[cfg(test)]
mod tests;

#[cfg(test)]
pub use tests::rand_poly;

/// Stores a polynomial in coefficient form.
#[derive(Clone, PartialEq, Eq, Hash, Default, Debug)]
pub struct Polynomial<F: Field> {
    // The coefficient of `x^i` is stored at location `i` in `self.coeffs`.
    coeffs: Vec<F>,
}

impl<F: Field> Polynomial<F> {
    pub fn coeffs(&self) -> &[F] {
        &self.coeffs
    }

    pub fn zero() -> Self {
        Self { coeffs: vec![] }
    }

    pub fn one() -> Self {
        Self {
            coeffs: vec![F::one()],
        }
    }

    pub fn monomial(coeff: F) -> Self {
        Self {
            coeffs: vec![coeff, F::one()],
        }
    }

    pub fn from_coeffs(coeffs: Vec<F>) -> Self {
        Self { coeffs }.truncate_leading_zeros()
    }

    fn truncate_leading_zeros(mut self) -> Self {
        if self.is_zero() || !self.coeffs.last().unwrap().is_zero() {
            return self;
        }

        let mut leading_index = self.coeffs.len() - 1;

        while self.coeffs[leading_index].is_zero() {
            leading_index -= 1;
        }

        self.coeffs.truncate(leading_index + 1);

        self
    }

    // Horner's method for polynomial evaluation
    fn horner_evaluate(poly_coeffs: &[F], point: &F) -> F {
        poly_coeffs
            .iter()
            .rfold(F::zero(), move |result, coeff| result * *point + *coeff)
    }

    pub fn evaluate(&self, point: &F) -> F {
        if self.is_zero() {
            return F::zero();
        }
        Self::horner_evaluate(&self.coeffs, point)
    }

    pub fn degree(&self) -> usize {
        // NP TODO Option
        if self.is_zero() {
            panic!("The degree of the zero polynomial is undefined");
        }
        // All operations internally ensure that the result has no leading zeros
        self.coeffs.len() - 1
    }

    pub fn is_zero(&self) -> bool {
        self.coeffs.is_empty()
    }

    pub fn is_constant(&self) -> bool {
        self.coeffs.len() <= 1
    }

    pub fn divide_with_q_and_r(&self, divisor: &Self) -> (Self, Self) {
        assert!(!divisor.is_zero());

        if self.is_zero() {
            return (Self::zero(), Self::zero());
        } else if self.degree() < divisor.degree() {
            return (Self::zero(), self.clone());
        }

        let mut quotient_coeffs = vec![F::zero(); self.degree() - divisor.degree() + 1];
        let mut remainder = self.clone();

        let divisor_leading_coeff_inv = divisor.coeffs.last().unwrap().inverse();

        while !remainder.is_zero() && remainder.degree() >= divisor.degree() {
            let cur_q_coeff = *remainder.coeffs.last().unwrap() * divisor_leading_coeff_inv;
            let cur_q_degree = remainder.degree() - divisor.degree();
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
    // NP TODO: This is far from optimal
    pub fn vanishing_polynomial(points: impl IntoIterator<Item = F>) -> Polynomial<F> {
        points
            .into_iter()
            .map(|point| Polynomial::monomial(-point))
            .product()
    }

    // NP TODO lagrange_interpolate_and_eval(
    // NP TODO lagrange_interpolate(
    pub fn naive_interpolate(point_to_evals: Vec<(F, F)>) -> Polynomial<F> {
        let points = point_to_evals.iter().map(|(p, _)| *p).collect_vec();
        let vanishing_poly = Self::vanishing_polynomial(points);
        let mut result = Polynomial::zero();
        for (point, eval) in point_to_evals.into_iter() {
            let term = &vanishing_poly / &Polynomial::monomial(-point);
            let scale = eval / term.evaluate(&point);
            let coeffs = term.coeffs().iter().map(|c| *c * scale).collect_vec();
            result += &Polynomial::from_coeffs(coeffs);
        }
        result
    }

    /// Given f(x) and e, returns f(x^e)
    pub fn compose_with_exponent(&self, exponent: usize) -> Polynomial<F> {
        let mut coeffs = vec![F::zero(); self.degree() * exponent + 1];
        for (i, coeff) in self.coeffs.iter().enumerate() {
            coeffs[i * exponent] = *coeff;
        }
        Polynomial::from_coeffs(coeffs)
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
        extended_self.resize(domain_size, F::zero());
        extended_other.resize(domain_size, F::zero());

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
