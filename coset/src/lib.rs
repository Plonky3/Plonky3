//! Interface for handling smooth cosets in the group of units of finite fields.

#![no_std]

extern crate alloc;

use alloc::{vec, vec::Vec};

use p3_dft::{Radix2Dit, TwoAdicSubgroupDft};
use p3_field::TwoAdicField;
use p3_poly::Polynomial;

#[cfg(test)]
mod tests;

/// Coset of a subgroup of the group of units of a finite field of order equal
/// to a power of two.
#[derive(Clone, Debug)]
pub struct TwoAdicCoset<F: TwoAdicField> {
    // Letting s = shift, and g = generator (of order 2^log_size), the coset in
    // question is
    //     s * <g> = {s, s * g, shift * g^2, ..., s * g^(2^log_size - 1)]
    generator: F,
    shift: F,
    log_size: usize,
    // The i-th element, if present, is generator^(2^i). The vector starts off as
    // vec![generator] and is expanded every time a higher iterated square is
    // computed.
    generator_iter_squares: Vec<F>,
}

/// Iterator over the elements of a two-adic coset.
pub struct TwoAdicCosetIterator<F: TwoAdicField> {
    current: F,
    generator: F,
    shift: F,
    consumed: bool,
}

impl<F: TwoAdicField> TwoAdicCoset<F> {
    /// Returns the coset `shift * <g>`, where g is a canonical (i. e. fixed in
    /// the implementation of `F: TwoAdicField`) generator of the unique
    /// subgroup of the units of `F` of order `2 ^ log_size`.
    ///
    /// # Arguments
    ///
    ///  - `shift`: the value by which the subgroup is (multiplicatively)
    ///    shifted
    ///  - `log_size`: the size of the subgroup (and hence of the coset) is
    ///    `2 ^ log_size`. This determines the subgroup uniquely.
    ///
    /// # Panics
    ///
    ///  - If `F: TwoAdicField` does not provide an element of order `2 ^
    /// log_size` (e. g. if `2 ^ log_size` does not divide `|F| - 1`)
    ///  - If `log_size` is greater than `usize::BITS`
    pub fn new(shift: F, log_size: usize) -> Self {
        assert!(
            log_size <= usize::BITS as usize,
            "log_size must be <= the number of bits in usize ({})",
            usize::BITS
        );

        let generator = F::two_adic_generator(log_size);
        Self {
            generator,
            shift,
            log_size,
            generator_iter_squares: vec![generator],
        }
    }

    /// Returns the generator of the coset.
    pub fn generator(&self) -> F {
        self.generator
    }

    /// Returns the shift of the coset.
    pub fn shift(&self) -> F {
        self.shift
    }

    /// Returns the log2 of the size of the coset.
    pub fn log_size(&self) -> usize {
        self.log_size
    }

    /// Returns the size of the coset.
    pub fn size(&self) -> usize {
        1 << self.log_size
    }

    /// Returns a new coset with its subgroup reduced by a factor of
    /// `2^log_scale_factor` in size (i. e. with generator equal to the
    /// `2^log_scale_factor`-th power of that of the original coset), leaving
    /// the shift untouched
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
            shift: self.shift,
            log_size: self.log_size - log_scale_factor,
            generator_iter_squares,
        }
    }

    /// Returns the coset `self^(2^log_scale_factor)` (i. e. with shift and
    /// generator equal to the `2^log_scale_factor`-th power of the original
    /// ones).
    pub fn shrink_coset(&self, log_scale_factor: usize) -> TwoAdicCoset<F> {
        let new_coset = self.shrink_subgroup(log_scale_factor);
        new_coset.set_shift(self.shift.exp_power_of_2(log_scale_factor))
    }

    /// Returns a new coset where the shift has been set to `shift` times the
    /// original shift.
    pub fn shift_by(&self, shift: F) -> TwoAdicCoset<F> {
        TwoAdicCoset {
            shift: self.shift * shift,
            ..self.clone()
        }
    }

    /// Returns a new coset where the shift has been set to `shift`
    pub fn set_shift(&self, shift: F) -> TwoAdicCoset<F> {
        TwoAdicCoset {
            shift,
            ..self.clone()
        }
    }

    /// Checks if the given field element is in the coset
    pub fn contains(&self, element: F) -> bool {
        // Note that, in a finite field F (this is not true of a general finite
        // commutative ring), there is exactly one subgroup of |F^*| of order n
        // for each divisor n of |F| - 1, and its elements e are uniquely
        // caracterised by the condition e^n = 1.

        // We check (shift * element)^(2^log_size) = 1, terminating early if
        // possible.
        let mut e = self.shift.inverse() * element;

        for _ in 0..self.log_size {
            if e == F::ONE {
                return true;
            }
            e = e.square();
        }

        e == F::ONE
    }

    /// Returns the unique polynomial of degree less than `2^log_size` that
    /// interpolates the given evaluations on the coset (with the canonical order
    /// "shift * g^0, shift * g^1, ...`).
    ///
    /// # Panics
    ///
    /// Panics if the number of evaluations is greater than the size of the
    /// coset.
    pub fn interpolate(&self, evals: Vec<F>) -> Polynomial<F> {
        assert!(
            evals.len() == 1 << self.log_size,
            "The number of evaluations must be equal to the size of the coset."
        );
        let dft = Radix2Dit::default().coset_idft(evals, self.shift);
        Polynomial::from_coeffs(dft)
    }

    /// Returns the `index`-th element of the coset `shift * g^index`. To prevent
    /// unnecessary computation, `index` is asserted to be < `2^log_size`.
    ///
    /// *Note*: Because `TwoAdicCoset` memoizes the iterated squares `g^(2^0)`,
    /// `g^(2^1)`, ... of `g` (so that subsequent element queries do not require
    /// recomputing those squares), this function might modify the internal
    /// vector of memoised values - hence the `mut` requirement. In situations
    /// where `mut` is not available, consider [`element_immutable`]
    ///
    /// # Panics
    ///
    /// Panics if `index >= 2^log_size`.
    pub fn element(&mut self, index: usize) -> F {
        // Transferring the responsibility of the modular reduction to the
        // caller spares handling cases where self.log_size >= usize::BITS here,
        // which would come at an unnecessary performance cost in the vast
        // majority of use cases.
        assert!(
            index < 1 << self.log_size,
            "index must be less than the size of the coset.\
            Consider passing the equivalent index % (1 << log_size) instead"
        );
        self.shift * self.generator_exp_usize(index)
    }

    /// Returns the `index`-th element of the coset `shift * g^index`. To
    /// prevent unnecessary computation, `index` is asserted to be <
    /// `2^log_size`.
    ///
    /// *Note*: If `self` is `mut` and several elements will be queried,
    /// consider the more optimal [`element`] which memoizes intermediate
    /// computations.
    ///
    /// # Panics
    ///
    /// Panics if `index >= 2^log_size`.
    pub fn element_immutable(&self, index: u64) -> F {
        assert!(
            index < 1 << self.log_size,
            "index must be less than the size of the coset.\
            Consider passing the equivalent index % (1 << log_size) instead"
        );

        self.shift * self.generator.exp_u64(index)
    }

    /// Returns the list of evaluations over the coset, that is,
    /// `polynomial(shift * g^0), polynomial(shift * g^1), ..., polynomial(shift
    /// * g^(2^log_size - 1))`.
    ///
    /// # Panics
    ///
    /// Panics if the degree of the polynomial is greater than or equal to the
    /// size of the coset. In this case, a larger domain should be used instead.
    pub fn evaluate_polynomial(&self, polynomial: &Polynomial<F>) -> Vec<F> {
        assert!(
            polynomial.degree().is_none_or(|d| d < (1 << self.log_size)),
            "The degree of the polynomial must be less than the size of the\
            coset. Consider constructing a larger coset, evaluating therein\
            and retaining the appropriate evaluations (which will be\
            interleaved with those in the rest of the large domain)"
        );

        let coeffs = polynomial.coeffs();

        if coeffs.len() == 0 {
            return vec![F::ZERO; 1 << self.log_size];
        } else if coeffs.len() == 1 {
            return vec![coeffs[0]; 1 << self.log_size];
        }

        let mut coeffs = coeffs.to_vec();
        coeffs.resize(1 << self.log_size, F::ZERO);
        Radix2Dit::default().coset_dft(coeffs, self.shift)
    }

    // Internal function which computes `generator^exp`. It uses the previously
    // stored iterated squares of the generator and stores any new ones arising
    // during the computation.
    fn generator_exp_usize(&mut self, exp: usize) -> F {
        // This case needs to be handled separately: otherwise msb_index would
        // be ill defined and could trigger a panic
        if exp == 0 {
            return F::ONE;
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

    /// Returns an iterator over the elements of the coset in the canonical order
    /// "shift * generator^0, shift * generator^1, ...,
    /// shift * generator^(2^log_size - 1)`.
    pub fn iter(&self) -> TwoAdicCosetIterator<F> {
        TwoAdicCosetIterator {
            current: self.shift,
            generator: self.generator,
            shift: self.shift,
            consumed: false,
        }
    }
}

/// This tests the equality of the two cosets as mathematical objects, that is:
/// whether they contain the exact same elements (not necessarily in the same
/// order) or not.
///
/// *Note*: The iterated squares of the generator memoised in each coset are
/// not taken into consideration.
impl<F: TwoAdicField> PartialEq for TwoAdicCoset<F> {
    fn eq(&self, other: &Self) -> bool {
        if self.generator == other.generator && self.shift == other.shift {
            return true;
        }

        // The first equality assumes generators are chosen canonically (as
        // ensured by the TwoAdicField interface). If not, one could simply
        // assert self.generator has the same order as other.generator (this is
        // enough by the cyclicity of the group of units)
        self.generator == other.generator && other.contains(self.shift)
    }
}

/// This tests the equality of the two cosets as mathematical objects, that is:
/// whether they contain the exact same elements (not necessarily in the same
/// order) or not.
///
/// *Note*: The iterated squares of the generator memoised in each coset are
/// not taken into consideration.
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
