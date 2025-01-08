use p3_field::TwoAdicField;
use p3_interpolation::interpolate_coset;
use p3_matrix::Matrix;

/// Coset of a smooth subgroup of the group of units of a finite field (smooth
/// meaning: having power-of-2 order).
pub struct Radix2Coset<F: TwoAdicField> {
    generator: F,
    generator_inv: F,
    shift: F,
    log_size: usize,
}

impl<F: TwoAdicField> Radix2Coset<F> {
    pub fn new(shift: F, log_size: usize) -> Self {
        let generator = F::two_adic_generator(log_size);
        Self {
            generator,
            generator_inv: generator.inv(),
            shift,
            log_size,
        }
    }

    pub fn size(&self) -> usize {
        1 << self.log_size
    }

    /// Reduce the size of the coset by a factor of 2^log_scale_factor (leaving the shift untouched)
    pub fn shrink(&self, log_scale_factor: usize) -> Radix2Coset<F> {
        let generator = self.generator.exp_power_of_2(log_scale_factor);
        Radix2Coset {
            generator,
            generator_inv: generator.inv(),
            shift: self.shift,
            log_size: self.log_size - log_scale_factor,
        }
    }

    /// Shift the coset by an element of the field
    pub fn shift_by(&self, shift: F) -> Radix2Coset<F> {
        Radix2Coset {
            shift: self.shift * shift,
            ..self
        }
    }

    /// Set the shift of the coset to a given element
    pub fn set_shift(&self, shift: F) -> Radix2Coset<F> {
        Radix2Coset { shift, ..self }
    }

    /// Checks if a given element is in the coset
    pub fn contains(&self, element: F) -> bool {
        // Note that a subgroup of order n of the group of units of a field is
        // necessarily the group of n-th roots of unity. Therefore, testing for
        // belonging to that group can be done by raising to its order.
        element.exp_power_of_2(self.log_size) == F::one()
    }

    pub fn interpolate<Mat>(coset_evals: &Mat, shift: F, point: F) -> Vec<F>
    where
        Mat: Matrix<F>,
    {
        interpolate_coset(coset_evals, shift, point)
    }
}

impl<F: TwoAdicField> PartialEq for Radix2Coset<F> {
    fn eq(&self, other: &Self) -> bool {
        self.generator == other.generator && other.contains(self.shift)
    }
}

impl<F: TwoAdicField> Eq for Radix2Coset<F> {}
