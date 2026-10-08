//! Additive transforms on an explicitly supplied ordered binary basis.

use alloc::vec::Vec;

use p3_binary_field::TowerLevel;
use p3_field::Algebra;

use crate::ButterflyField;

/// An additive transform whose domain is a coset of an ordered binary basis.
///
/// Coefficients use the normalized Lin-Chung-Han novel basis of that subspace.
/// Output index bits select basis vectors in the supplied order.
#[derive(Clone, Debug)]
pub struct BasisNtt<F> {
    /// Normalized subspace evaluations, including the coset offset as the last entry.
    rows: Vec<Vec<F>>,
    /// The ordered, linearly independent domain vectors.
    basis: Vec<F>,
    /// The affine offset of the domain.
    shift: F,
}

impl<F: TowerLevel> BasisNtt<F> {
    /// Construct a transform over a coset of an explicitly ordered binary basis.
    ///
    /// # Panics
    /// Panics if the basis is dependent or exceeds the field dimension.
    pub fn new(basis: &[F], shift: F) -> Self {
        // A binary field has one independent vector per coordinate bit.
        assert!(
            basis.len() <= 1 << F::LOG_BITS,
            "domain exceeds field dimension"
        );
        let mut row = basis.to_vec();
        row.push(shift);
        let mut rows = Vec::with_capacity(basis.len());
        for _ in 0..basis.len() {
            // Why: the value on the next basis vector normalizes the novel polynomial.
            let root = row[0];
            assert!(!root.is_zero(), "domain basis is linearly dependent");
            let inverse = root.inverse();
            rows.push(row.iter().map(|&v| v * inverse).collect());
            // The next subspace polynomial vanishes on this vector and every earlier one.
            row = row[1..].iter().map(|&v| v * (v + root)).collect();
        }
        Self {
            rows,
            basis: basis.to_vec(),
            shift,
        }
    }

    /// Construct a domain from the low coordinate bits in this field representation.
    ///
    /// # Panics
    /// Panics if the dimension exceeds the field's coordinate width.
    pub fn polynomial(log_n: usize, shift: F) -> Self {
        // Select each coordinate separately without integer-to-prime-field conversion.
        assert!(log_n <= 1 << F::LOG_BITS, "domain exceeds field dimension");
        let basis: Vec<_> = (0..log_n)
            .map(|bit| {
                F::from_le_byte_iter(
                    (0..).map(move |byte| if byte == bit / 8 { 1 << (bit % 8) } else { 0 }),
                )
            })
            .collect();
        Self::new(&basis, shift)
    }

    /// The base-two logarithm of the evaluation count.
    pub const fn log_domain_size(&self) -> usize {
        self.basis.len()
    }

    /// The affine offset of the evaluation domain.
    pub const fn shift(&self) -> F {
        self.shift
    }

    /// Carry this plan through a field embedding that preserves zero, one, addition and multiplication.
    pub fn map<G: TowerLevel>(&self, embed: impl Fn(F) -> G) -> BasisNtt<G> {
        // Embedding the domain preserves its ordered basis and all normalized subspace polynomials.
        BasisNtt {
            rows: self
                .rows
                .iter()
                .map(|row| row.iter().copied().map(&embed).collect())
                .collect(),
            basis: self.basis.iter().copied().map(&embed).collect(),
            shift: embed(self.shift),
        }
    }

    /// The normalized twiddle for a stage numbered from the root of the butterfly tree.
    ///
    /// # Panics
    /// Panics if the stage or block lies outside the domain.
    pub fn twiddle(&self, layer: usize, block: usize) -> F {
        // Each block starts at a subset sum of the basis vectors above this stage.
        assert!(layer < self.log_domain_size(), "stage outside domain");
        assert!(block < (1usize << layer), "block outside domain");
        let row = &self.rows[self.log_domain_size() - layer - 1];
        let mut t = *row.last().expect("a subspace row includes its shift");
        for (bit, &v) in row[1..row.len() - 1].iter().enumerate() {
            if block >> bit & 1 != 0 {
                t += v;
            }
        }
        t
    }

    /// The seven twiddles of three adjacent stages, in breadth-first order.
    ///
    /// # Panics
    /// Panics unless three stages remain and the block lies in the first stage.
    pub fn twiddles_radix8(&self, layer: usize, block: usize) -> [F; 7] {
        assert!(
            layer + 2 < self.log_domain_size(),
            "three stages must remain"
        );
        assert!(block < 1usize << layer, "block outside domain");
        let dim = self.log_domain_size();
        let (v0, v1, v2) = (
            &self.rows[dim - layer - 1],
            &self.rows[dim - layer - 2],
            &self.rows[dim - layer - 3],
        );
        let (mut t, mut a, mut c) = (
            *v0.last().expect("shift present"),
            *v1.last().expect("shift present"),
            *v2.last().expect("shift present"),
        );
        // A single walk shares the block's subset-sum decisions across all three rows.
        for bit in 0..layer {
            if block >> bit & 1 != 0 {
                t += v0[1 + bit];
                a += v1[2 + bit];
                c += v2[3 + bit];
            }
        }
        let (d, e0, e1) = (v1[1], v2[1], v2[2]);
        [t, a, a + d, c, c + e0, c + e1, c + e0 + e1]
    }

    /// Transform extension-valued rows with this base-field plan.
    ///
    /// The inverse flag selects interpolation instead of evaluation.
    ///
    /// # Panics
    /// Panics unless the width is positive and the buffer has exactly one row per domain point.
    pub fn transform_algebra<A: Algebra<F> + Copy, const INVERSE: bool>(
        &self,
        values: &mut [A],
        width: usize,
    ) {
        // Extension multiplication by a base twiddle keeps the domain unchanged.
        self.transform::<A, INVERSE>(values, width, |lo, hi, t| {
            for (a, b) in lo.iter_mut().zip(hi) {
                if INVERSE {
                    *b += *a;
                    *a += *b * t;
                } else {
                    *a += *b * t;
                    *b += *a;
                }
            }
        });
    }

    fn transform<A, const INVERSE: bool>(
        &self,
        values: &mut [A],
        width: usize,
        butterfly: impl Fn(&mut [A], &mut [A], F),
    ) {
        // These equalities make each split cover complete rows and the entire domain.
        assert!(width > 0, "matrix width must be positive");
        assert_eq!(
            values.len() / width,
            1usize << self.log_domain_size(),
            "matrix height differs from domain"
        );
        assert_eq!(values.len() % width, 0, "incomplete matrix row");
        for step in 0..self.log_domain_size() {
            let layer = if INVERSE {
                self.log_domain_size() - 1 - step
            } else {
                step
            };
            let size = values.len() >> layer;
            for (block, values) in values.chunks_exact_mut(size).enumerate() {
                let (lo, hi) = values.split_at_mut(size / 2);
                butterfly(lo, hi, self.twiddle(layer, block));
            }
        }
    }
}

impl<F: ButterflyField> BasisNtt<F> {
    /// Evaluate one coefficient column on the ordered domain.
    pub fn forward(&self, values: &mut [F]) {
        self.forward_batch(values, 1);
    }

    /// Interpolate one evaluation column on the ordered domain.
    pub fn inverse(&self, values: &mut [F]) {
        self.inverse_batch(values, 1);
    }

    /// Evaluate a row-major coefficient matrix on the ordered domain.
    pub fn forward_batch(&self, values: &mut [F], width: usize) {
        // Dispatch each contiguous row group to the field's optimized butterfly.
        self.transform_field::<false>(values, width);
    }

    /// Interpolate a row-major evaluation matrix on the ordered domain.
    pub fn inverse_batch(&self, values: &mut [F], width: usize) {
        // Reverse the stage order and each local butterfly.
        self.transform_field::<true>(values, width);
    }

    /// Group three adjacent stages so the field can keep their rows in registers.
    fn transform_field<const INVERSE: bool>(&self, values: &mut [F], width: usize) {
        assert!(width > 0, "matrix width must be positive");
        let dim = self.log_domain_size();
        assert_eq!(
            values.len() / width,
            1usize << dim,
            "matrix height differs from domain"
        );
        assert_eq!(values.len() % width, 0, "incomplete matrix row");
        let mut done = 0;
        while done < dim {
            let stages = (dim - done).min(3);
            if stages == 3 {
                let layer = if INVERSE { dim - done - 3 } else { done };
                let size = values.len() >> layer;
                for (block, values) in values.chunks_exact_mut(size).enumerate() {
                    let mut chunks = values.chunks_exact_mut(size / 8);
                    let mut rows =
                        core::array::from_fn(|_| chunks.next().expect("eight equal slabs"));
                    F::butterfly_radix8::<INVERSE>(&mut rows, &self.twiddles_radix8(layer, block));
                }
            } else {
                for step in 0..stages {
                    let layer = if INVERSE {
                        dim - done - step - 1
                    } else {
                        done + step
                    };
                    let size = values.len() >> layer;
                    for (block, values) in values.chunks_exact_mut(size).enumerate() {
                        let (lo, hi) = values.split_at_mut(size / 2);
                        F::butterfly::<INVERSE>(lo, hi, self.twiddle(layer, block));
                    }
                }
            }
            done += stages;
        }
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec;

    use p3_binary_field::{Poly64, Rijndael8b};
    use p3_field::PrimeCharacteristicRing;
    use proptest::prelude::*;

    use super::*;

    fn check<F: ButterflyField>(basis: &[F], shift: F, coefficients: &[F], width: usize) {
        let plan = BasisNtt::new(basis, shift);
        // The fused three-stage twiddles retain the same breadth-first block order.
        for layer in 0..basis.len().saturating_sub(2) {
            for block in 0..1usize << layer {
                assert_eq!(
                    plan.twiddles_radix8(layer, block),
                    [
                        plan.twiddle(layer, block),
                        plan.twiddle(layer + 1, block * 2),
                        plan.twiddle(layer + 1, block * 2 + 1),
                        plan.twiddle(layer + 2, block * 4),
                        plan.twiddle(layer + 2, block * 4 + 1),
                        plan.twiddle(layer + 2, block * 4 + 2),
                        plan.twiddle(layer + 2, block * 4 + 3)
                    ]
                );
            }
        }
        let mut values = coefficients.to_vec();
        plan.forward_batch(&mut values, width);
        // Invariant: each novel-basis polynomial evaluates directly on the supplied domain.
        for (index, row) in values.chunks_exact(width).enumerate() {
            let x =
                basis.iter().enumerate().fold(
                    shift,
                    |x, (bit, &b)| if index >> bit & 1 != 0 { x + b } else { x },
                );
            let mut numerator = x;
            let mut images = basis.to_vec();
            let mut normalized = Vec::new();
            for i in 0..basis.len() {
                let root = images[i];
                normalized.push(numerator * root.inverse());
                numerator *= numerator + root;
                for value in &mut images[i + 1..] {
                    *value *= *value + root;
                }
            }
            for column in 0..width {
                let expected =
                    coefficients
                        .chunks_exact(width)
                        .enumerate()
                        .fold(F::ZERO, |sum, (term, c)| {
                            let weight =
                                normalized
                                    .iter()
                                    .enumerate()
                                    .fold(F::ONE, |weight, (bit, &v)| {
                                        if term >> bit & 1 != 0 {
                                            weight * v
                                        } else {
                                            weight
                                        }
                                    });
                            sum + c[column] * weight
                        });
                assert_eq!(row[column], expected);
            }
        }
        // The algebra path and the optimized field butterflies must agree in both directions.
        let mut algebra = coefficients.to_vec();
        plan.transform_algebra::<F, false>(&mut algebra, width);
        assert_eq!(values, algebra);
        plan.inverse_batch(&mut values, width);
        plan.transform_algebra::<F, true>(&mut algebra, width);
        assert_eq!(values, coefficients);
        assert_eq!(algebra, coefficients);
    }

    proptest! {
        #[test]
        fn explicit_domains_match_direct_evaluation(log_n in 0usize..=6, width in 1usize..=5, shift in any::<u8>(), seed in any::<u64>()) {
            // Reversing coordinate vectors exercises a basis other than the default Cantor order.
            let basis: Vec<_> = (0..log_n).rev().map(|bit| Rijndael8b::from_byte(1 << bit)).collect();
            let values: Vec<_> = (0..width << log_n).map(|i| Rijndael8b::from_byte(seed.wrapping_add(i as u64).wrapping_mul(0x9e3779b9) as u8)).collect();
            check(&basis, Rijndael8b::from_byte(shift), &values, width);
        }
    }

    #[test]
    fn wide_rows_and_polynomial_domains_match_the_algebra_path() {
        // Several register widths and an odd tail reach every optimized butterfly boundary.
        for log_n in 0..=5 {
            for width in [1, 3, 8, 17, 65] {
                let basis: Vec<_> = (0..log_n).map(|bit| Poly64::new(1 << bit)).collect();
                let values: Vec<_> = (0..width << log_n)
                    .map(|i| Poly64::new((i as u64).wrapping_mul(0x9e3779b97f4a7c15)))
                    .collect();
                check(&basis, Poly64::new(0xfedcba9876543210), &values, width);
            }
        }
    }

    #[test]
    #[should_panic(expected = "linearly dependent")]
    fn dependent_basis_is_refused() {
        let _ = BasisNtt::new(&[Rijndael8b::ONE; 2], Rijndael8b::ZERO);
    }

    #[test]
    #[should_panic(expected = "matrix height")]
    fn wrong_height_is_refused() {
        BasisNtt::polynomial(2, Rijndael8b::ZERO).forward(&mut [Rijndael8b::ZERO; 2]);
    }

    #[test]
    fn polynomial_constructor_preserves_coordinate_order() {
        // A unit novel coefficient on the bottom stage evaluates as the domain coordinate itself.
        let ntt = BasisNtt::polynomial(3, Rijndael8b::from_byte(0x80));
        let mut values = vec![Rijndael8b::ZERO; 8];
        values[1] = Rijndael8b::ONE;
        ntt.forward(&mut values);
        assert_eq!(
            values,
            (0..8)
                .map(|i| Rijndael8b::from_byte(0x80 ^ i))
                .collect::<Vec<_>>()
        );
    }
}
