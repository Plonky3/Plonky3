//! Convenience methods to work with smooth cosets in the group of units of
//! finite fields.

#![no_std]

extern crate alloc;

use core::iter::Take;

use p3_field::{Powers, TwoAdicField};
#[cfg(test)]
mod tests;

/// Coset of a subgroup of the group of units of a finite field of order equal
/// to a power of two.
///
/// # Examples
///
/// ```
/// # use p3_coset::TwoAdicCoset;
/// # use p3_field::{TwoAdicField, PrimeCharacteristicRing};
/// # use itertools::Itertools;
/// # use p3_baby_bear::BabyBear;
/// #
/// type F = BabyBear;
/// let log_size = 3;
/// let shift = F::from_u64(7);
/// let mut coset = TwoAdicCoset::new(shift, log_size);
/// let generator = coset.generator();
///
///
/// // Coset elements can be queried by index
/// assert_eq!(coset.element(4), shift * generator.exp_u64(4));
///
/// // Coset elements can be iterated over in the canonical order
/// assert_eq!(
///     coset.iter().collect_vec(),
///     (0..1 << log_size).map(|i| shift * generator.exp_u64(i)).collect_vec()
/// );
///
/// // Cosets can be (element-wise) raised to a power of 2, either maintaining
/// // the shift and raising only the subgroup, or raising both.
/// assert_eq!(coset.shrink_coset(2), TwoAdicCoset::new(shift.exp_power_of_2(2), log_size - 2));
/// assert_eq!(coset.shrink_subgroup(2), TwoAdicCoset::new(shift, log_size - 2));
/// ```
#[derive(Clone, Debug)]
pub struct TwoAdicCoset<F: TwoAdicField> {
    // Letting s = shift, and g = generator (of order 2^log_size), the coset in
    // question is
    //     s * <g> = {s, s * g, shift * g^2, ..., s * g^(2^log_size - 1)]
    shift: F,
    log_size: usize,
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
    ///  - `log_size`: the size of the subgroup (and hence of the coset) is `2 ^
    ///    log_size`. This determines the subgroup uniquely.
    ///
    /// # Panics
    ///
    ///  If `F: TwoAdicField` does not provide an element of order `2 ^
    /// log_size` (e. g. if `2 ^ log_size` does not divide `|F| - 1`)
    pub fn new(shift: F, log_size: usize) -> Self {
        assert!(
            log_size <= F::TWO_ADICITY,
            "log_size must be <= the two_adicity of the field ({})",
            F::TWO_ADICITY
        );

        Self { shift, log_size }
    }

    /// Returns the generator of the coset.
    #[inline]
    pub fn generator(&self) -> F {
        F::two_adic_generator(self.log_size)
    }

    /// Returns the shift of the coset.
    #[inline]
    pub fn shift(&self) -> F {
        self.shift
    }

    /// Returns the log2 of the size of the coset.
    #[inline]
    pub fn log_size(&self) -> usize {
        self.log_size
    }

    /// Returns the size of the coset.
    #[inline]
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

        TwoAdicCoset {
            shift: self.shift,
            log_size: self.log_size - log_scale_factor,
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
            log_size: self.log_size,
        }
    }

    /// Returns a new coset where the shift has been set to `shift`
    pub fn set_shift(&self, shift: F) -> TwoAdicCoset<F> {
        TwoAdicCoset {
            shift,
            log_size: self.log_size,
        }
    }

    /// Checks if the given field element is in the coset
    pub fn contains(&self, element: F) -> bool {
        // Note that, in a finite field F (this is not true of a general finite
        // commutative ring), there is exactly one subgroup of |F^*| of order n
        // for each divisor n of |F| - 1, and its elements e are uniquely
        // caracterised by the condition e^n = 1.

        // We check (shift^{-1} * element)^(2^log_size) = 1, terminating early if
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

    /// Returns the `index`-th element of the coset `shift * g^index`. To prevent
    /// unnecessary computation, `index` is asserted to be < `2^log_size`.
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
            "index must be less than the size of the coset. \
            Consider passing the equivalent index % (1 << log_size) instead"
        );
        self.shift * self.generator_exp(index)
    }

    // Internal function which computes `generator^exp`. It uses the
    // square-and-multiply algorithm with the caveat that squares of the
    // generator are queried from the field (which typically should have them
    // stored), i. e. rather "fetch-and-multiply"
    fn generator_exp(&mut self, exp: usize) -> F {
        let mut gen_power = F::ONE;
        let mut exp = exp;
        let mut i = self.log_size();

        while exp > 0 {
            if exp & 1 != 0 {
                gen_power *= F::two_adic_generator(i);
            }
            exp >>= 1;

            // This cannot be added to the loop condition cleanly because
            // casting it to an isize would give a different range than usize
            if i == 0 {
                break;
            }

            i -= 1;
        }

        gen_power
    }

    /// Returns an iterator over the elements of the coset in the canonical order
    /// "shift * generator^0, shift * generator^1, ...,
    /// shift * generator^(2^log_size - 1)`.
    pub fn iter(&self) -> Take<Powers<F>> {
        self.generator()
            .shifted_powers(self.shift)
            .take(1 << self.log_size)
    }
}

/// This tests the equality of the two cosets as mathematical objects, that is:
/// whether they contain the exact same elements (not necessarily in the same
/// order) or not.
impl<F: TwoAdicField> PartialEq for TwoAdicCoset<F> {
    fn eq(&self, other: &Self) -> bool {
        if self.generator() == other.generator() && self.shift == other.shift {
            return true;
        }

        // The first equality assumes generators are chosen canonically (as
        // ensured by the TwoAdicField interface). If not, one could simply
        // assert self.generator has the same order as other.generator (this is
        // enough by the cyclicity of the group of units)
        self.generator() == other.generator() && other.contains(self.shift)
    }
}

/// This tests the equality of the two cosets as mathematical objects, that is:
/// whether they contain the exact same elements (not necessarily in the same
/// order) or not.
///
// This falls back to PartialEq
impl<F: TwoAdicField> Eq for TwoAdicCoset<F> {}

impl<F: TwoAdicField> IntoIterator for TwoAdicCoset<F> {
    type Item = F;
    type IntoIter = Take<Powers<F>>;

    fn into_iter(self) -> Self::IntoIter {
        self.generator()
            .shifted_powers(self.shift)
            .take(1 << self.log_size)
    }
}
