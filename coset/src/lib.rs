//! Interface for handling smooth cosets in the group of units of finite fields.

// NP TODO re-introduce
// #![no_std]

extern crate alloc;

use alloc::{vec, vec::Vec};

use p3_dft::{NaiveDft, TwoAdicSubgroupDft};
use p3_field::TwoAdicField;
use p3_interpolation::interpolate_coset;
use p3_matrix::Matrix;
use p3_poly::Polynomial;

#[cfg(test)]
mod tests;

/// Coset of a subgroup of the group of units of a finite field of order equal to a power of two.
#[derive(Clone, Debug)]
pub struct TwoAdicCoset<F: TwoAdicField> {
    generator: F,
    generator_inv: F,
    shift: F,
    log_size: usize,
    // The i-th element, if present, is generator^2^i. The vector starts off as
    // vec![generator] and is expanded every time a higher iterated square is
    // computed.
    generator_iter_squares: Vec<F>,
}

pub struct TwoAdicCosetIterator<F: TwoAdicField> {
    current: F,
    generator: F,
    shift: F,
    consumed: bool,
}

impl<F: TwoAdicField> TwoAdicCoset<F> {
    pub fn new(shift: F, log_size: usize) -> Self {
        let generator = F::two_adic_generator(log_size);
        Self {
            generator,
            generator_inv: generator.inverse(),
            shift,
            log_size,
            generator_iter_squares: vec![generator],
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
    pub fn shrink_subgroup(&self, log_scale_factor: usize) -> TwoAdicCoset<F> {
        assert!(
            log_scale_factor <= self.log_size,
            "The domain size (2^{}) is not large enough to be shrunk by a factor of 2^{}",
            self.log_size,
            log_scale_factor
        );

        // If we had already computed some iterated squares of the generator, we
        // can keep the relevant ones (and spare ourselves the computation of
        // the new generator)
        let (generator, generator_iter_squares) = {
            if self.generator_iter_squares.len() > log_scale_factor {
                let iter_squares = self.generator_iter_squares[log_scale_factor..].to_vec();
                let generator = iter_squares[0];
                (generator, iter_squares)
            } else {
                let generator = self.generator.exp_power_of_2(log_scale_factor);
                let iter_squares = vec![generator];
                (generator, iter_squares)
            }
        };

        TwoAdicCoset {
            generator,
            generator_inv: generator.inverse(),
            shift: self.shift,
            log_size: self.log_size - log_scale_factor,
            generator_iter_squares,
        }
    }

    /// Reduce the size of the coset by a factor of 2^log_scale_factor.
    /// The shift is also raised to the power of 2^log_scale_factor.
    pub fn shrink_coset(&self, log_scale_factor: usize) -> TwoAdicCoset<F> {
        let new_coset = self.shrink_subgroup(log_scale_factor);
        new_coset.set_shift(self.shift.exp_power_of_2(log_scale_factor))
    }

    /// Shift the coset by an element of the field
    pub fn shift_by(&self, shift: F) -> TwoAdicCoset<F> {
        let mut shifted = self.clone();
        shifted.shift = self.shift * shift;
        shifted
    }

    /// Set the shift of the coset to a given element
    pub fn set_shift(&self, shift: F) -> TwoAdicCoset<F> {
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
    // NP TODO change this comment, now we are handling it
    // We are limited to u64 because of the `TwoAdicField` interface. If the
    // latter is expanded, this function can be so as well.
    pub fn element(&mut self, exp: usize) -> F {
        self.shift * self.generator_exp_usize(exp)
    }

    // NP TODO explain this
    pub fn element_immutable(&self, exp: u64) -> F {
        self.shift * self.generator.exp_u64(exp)
    }

    pub fn evaluate_polynomial(&self, polynomial: &Polynomial<F>) -> Vec<F> {
        let coeffs = polynomial.coeffs();

        if coeffs.len() == 0 {
            return vec![F::ZERO; 1 << self.log_size];
        } else if coeffs.len() == 1 {
            return vec![coeffs[0]; 1 << self.log_size];
        }

        // NP TODO
        assert!(polynomial.degree().unwrap() < (1 << self.log_size), "TODO");

        let mut coeffs = coeffs.to_vec();
        coeffs.resize(1 << self.log_size, F::ZERO);
        let dft = NaiveDft.coset_dft(coeffs, self.shift);
        dft
    }

    pub fn generator_exp_usize(&mut self, exp: usize) -> F {
        // This case needs to be handled separately: otherwise msb_index would
        // be ill defined and could trigger a panic
        if exp == 0 {
            return self.shift;
        }

        let mut exp = exp;
        let msb_index = (usize::BITS - exp.leading_zeros() - 1) as usize;

        // We make use of the available iterated squares of the generator and
        // compute the rest
        for i in self.generator_iter_squares.len()..=msb_index {
            self.generator_iter_squares
                .push(self.generator_iter_squares[i - 1].square());
        }

        let mut gen_power = F::ONE;
        let mut gen_squares = self.generator_iter_squares.iter();

        while exp > 0 {
            let gen_square = gen_squares.next().unwrap();
            if exp & 1 != 0 {
                gen_power *= *gen_square;
            }
            exp >>= 1;
        }

        gen_power
    }

    pub fn iter(&self) -> TwoAdicCosetIterator<F> {
        TwoAdicCosetIterator {
            current: self.shift,
            generator: self.generator,
            shift: self.shift,
            consumed: false,
        }
    }
}

impl<F: TwoAdicField> PartialEq for TwoAdicCoset<F> {
    fn eq(&self, other: &Self) -> bool {
        // The first equality assumes generators are chosen canonically (as
        // ensured by the TwoAdicField interface). If not, one could simply
        // assert self.generator has the same order as other.generator (this is
        // enough by the cyclicity of the group of units)
        self.generator == other.generator && other.contains(self.shift)
    }
}

impl<F: TwoAdicField> Eq for TwoAdicCoset<F> {}

impl<F: TwoAdicField> Iterator for TwoAdicCosetIterator<F> {
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

impl<F: TwoAdicField> IntoIterator for TwoAdicCoset<F> {
    type Item = F;
    type IntoIter = TwoAdicCosetIterator<F>;

    fn into_iter(self) -> Self::IntoIter {
        TwoAdicCosetIterator {
            current: self.shift,
            generator: self.generator,
            shift: self.shift,
            consumed: false,
        }
    }
}
