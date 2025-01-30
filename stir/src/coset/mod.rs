use alloc::vec::Vec;

use p3_dft::{NaiveDft, TwoAdicSubgroupDft};
use p3_field::TwoAdicField;
use p3_interpolation::interpolate_coset;
use p3_matrix::Matrix;

use crate::polynomial::Polynomial;

#[cfg(test)]
mod tests;

/// Coset of a smooth subgroup of the group of units of a finite field (smooth
/// meaning: having power-of-2 order).
#[derive(Clone, Debug)]
pub struct Radix2Coset<F: TwoAdicField> {
    generator: F,
    generator_inv: F,
    shift: F,
    log_size: usize,
}

pub struct Radix2Iterator<F: TwoAdicField> {
    current: F,
    generator: F,
    shift: F,
    consumed: bool,
}

impl<F: TwoAdicField> Radix2Coset<F> {
    pub fn new(shift: F, log_size: usize) -> Self {
        let generator = F::two_adic_generator(log_size);
        Self {
            generator,
            generator_inv: generator.inverse(),
            shift,
            log_size,
        }
    }

    pub fn generator(&self) -> F {
        self.generator
    }

    pub fn shift(&self) -> F {
        self.shift
    }

    pub fn generator_inv(&self) -> F {
        self.generator_inv
    }

    pub fn log_size(&self) -> usize {
        self.log_size
    }

    pub fn size(&self) -> usize {
        1 << self.log_size
    }

    /// Reduce the size of the subgroup by a factor of 2^log_scale_factor
    /// this leaves the shift untouched
    pub fn shrink_subgroup(&self, log_scale_factor: usize) -> Radix2Coset<F> {
        assert!(
            log_scale_factor <= self.log_size,
            "The domain size (2 ^ {}) is not large enough to be shrunk by a factor of 2^{}",
            self.log_size,
            log_scale_factor
        );

        let generator = self.generator.exp_power_of_2(log_scale_factor);
        Radix2Coset {
            generator,
            generator_inv: generator.inverse(),
            shift: self.shift,
            log_size: self.log_size - log_scale_factor,
        }
    }

    /// Reduce the size of the coset by a factor of 2^log_scale_factor.
    /// The shift is also raised to the power of 2^log_scale_factor.
    pub fn shrink_coset(&self, log_scale_factor: usize) -> Radix2Coset<F> {
        assert!(
            log_scale_factor <= self.log_size,
            "The domain size (2 ^ {}) is not large enough to be shrunk by a factor of 2^{}",
            self.log_size,
            log_scale_factor
        );

        let generator = self.generator.exp_power_of_2(log_scale_factor);
        let shift = self.shift.exp_power_of_2(log_scale_factor);
        Radix2Coset {
            generator,
            generator_inv: generator.inverse(),
            shift,
            log_size: self.log_size - log_scale_factor,
        }
    }

    /// Shift the coset by an element of the field
    pub fn shift_by(&self, shift: F) -> Radix2Coset<F> {
        let mut shifted = self.clone();
        shifted.shift = self.shift * shift;
        shifted
    }

    /// Set the shift of the coset to a given element
    pub fn set_shift(&self, shift: F) -> Radix2Coset<F> {
        let mut shifted = self.clone();
        shifted.shift = shift;
        shifted
    }

    /// Checks if a given element is in the coset
    pub fn contains(&self, element: F) -> bool {
        // Note that, in a field (this is not true of a general commutative
        // ring), there is exactly one subgroup of |F^*| of order n for each
        // divisor n of |F| - 1, and its elements e are uniquely caracterised by
        // the condition e^n = 1.

        // NP TODO think about early termination either here or in field: exp_power_of_2
        (self.shift.inverse() * element).exp_power_of_2(self.log_size) == F::ONE
    }

    pub fn evaluate_interpolation<Mat>(&self, coset_evals: &Mat, point: F) -> Vec<F>
    where
        Mat: Matrix<F>,
    {
        // NP TODO: Make use of the denominator diffs for efficiency
        interpolate_coset(coset_evals, self.shift, point, None)
    }

    pub fn interpolate(&self, evals: Vec<F>) -> Polynomial<F> {
        let mut evals = evals;
        evals.resize(1 << self.log_size, F::ZERO);
        // NP TODO is there a better impl?
        let dft = NaiveDft.coset_idft(evals, self.shift);
        Polynomial::from_coeffs(dft)
    }

    /// Give the `i`-th element of the coset: `shift * generator^i` (this wraps
    /// around `2^log_size`)
    // We are limited to u64 because of the `TwoAdicField` interface. If the
    // latter is expanded, this function can be so as well.
    pub fn element(&self, index: u64) -> F {
        self.shift * self.generator.exp_u64(index)
    }

    pub fn evaluate_polynomial(&self, polynomial: &Polynomial<F>) -> Vec<F> {
        // NP TODO
        assert!(polynomial.degree() < (1 << self.log_size), "TODO");

        let mut coeffs = polynomial.coeffs().to_vec();
        coeffs.resize(1 << self.log_size, F::ZERO);
        let dft = NaiveDft.coset_dft(coeffs, self.shift);
        dft
    }

    pub fn iter(&self) -> Radix2Iterator<F> {
        Radix2Iterator {
            current: self.shift,
            generator: self.generator,
            shift: self.shift,
            consumed: false,
        }
    }
}

impl<F: TwoAdicField> PartialEq for Radix2Coset<F> {
    fn eq(&self, other: &Self) -> bool {
        // The first equality assumes generators are chosen canonically. If not,
        // simply assert self.generator has the same order as other.generator
        // (this is enough by the cyclicity of the group of units)
        self.generator == other.generator && other.contains(self.shift)
    }
}

impl<F: TwoAdicField> Eq for Radix2Coset<F> {}

impl<F: TwoAdicField> Iterator for Radix2Iterator<F> {
    type Item = F;

    fn next(&mut self) -> Option<Self::Item> {
        if self.consumed {
            return None;
        }

        let current = self.current;
        self.current = self.current * self.generator;

        if self.current == self.shift {
            self.consumed = true;
        }

        Some(current)
    }
}

impl<F: TwoAdicField> IntoIterator for Radix2Coset<F> {
    type Item = F;
    type IntoIter = Radix2Iterator<F>;

    fn into_iter(self) -> Self::IntoIter {
        Radix2Iterator {
            current: self.shift,
            generator: self.generator,
            shift: self.shift,
            consumed: false,
        }
    }
}
