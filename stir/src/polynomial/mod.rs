use core::ops::{Add, Mul, Neg, Sub};

use itertools::Itertools;
use p3_dft::{NaiveDft, TwoAdicSubgroupDft};
use p3_field::{Field, TwoAdicField};
use p3_matrix::dense::RowMajorMatrix;
use p3_matrix::Matrix;

#[cfg(test)]
mod tests;

/// Stores a polynomial in coefficient form.
#[derive(Clone, PartialEq, Eq, Hash, Default)]
pub struct Polynomial<F: Field> {
    /// The coefficient of `x^i` is stored at location `i` in `self.coeffs`.
    pub coeffs: Vec<F>,
}

impl<F: Field> Polynomial<F> {
    fn from_coeffs(coeffs: Vec<F>) -> Self {
        Self { coeffs }
    }

    fn truncate_leading_zeros(&mut self) {
        while self.coeffs.last().map_or(false, |c| c.is_zero()) {
            self.coeffs.pop();
        }
    }

    // Horner's method for polynomial evaluation
    fn horner_evaluate(poly_coeffs: &[F], point: &F) -> F {
        poly_coeffs
            .iter()
            .rfold(F::zero(), move |result, coeff| result * *point + *coeff)
    }

    fn evaluate(&self, point: &F) -> F {
        if self.is_zero() {
            return F::zero();
        }
        Self::horner_evaluate(&self.coeffs, point)
    }

    fn degree(&self) -> usize {
        if self.is_zero() {
            return 0;
        }
        self.coeffs.len() - 1
    }

    fn is_zero(&self) -> bool {
        self.coeffs.is_empty()
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

        high.truncate_leading_zeros();
        high
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
        let mut extended_self = self.coeffs.clone();
        let mut extended_other = other.coeffs.clone();

        let domain_size = (self.coeffs.len() + other.coeffs.len() - 1).next_power_of_two();
        extended_self.resize(domain_size, F::zero());
        extended_other.resize(domain_size, F::zero());

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

        let mut result = Polynomial {
            coeffs: inverse_dft.values.clone(),
        };

        result.truncate_leading_zeros();
        result
    }
}
