//! The Lin–Chung–Han transform carried out in the polynomial basis of `GF(2^128)`.

use alloc::vec::Vec;

use p3_binary_field::{BinaryField128, poly_basis};
use p3_matrix::Matrix;
use p3_matrix::dense::RowMajorMatrix;
use p3_maybe_rayon::prelude::*;
use p3_util::log2_strict_usize;

use crate::domain::{domain_point, subspace_polynomial};
use crate::lch::{BUTTERFLY_GRAIN, LchNtt};
use crate::traits::AdditiveNtt;

/// [`LchNtt`] over `BinaryField128`, with the data held in the polynomial basis throughout.
///
/// A tower-basis product converts both operands into the polynomial basis and the result back,
/// sixteen dependent table lookups apiece, which is most of what a butterfly costs. Converting
/// the whole matrix once on the way in and once on the way out pays `2` conversions per element
/// instead of `3ℓ/2`, and every twiddle multiply in between is a bare carryless multiply and a
/// reduction. Additions are `XOR` in both bases, so they are unaffected.
///
/// Without a carryless-multiply instruction that product is a bit-serial loop and slower than
/// the tower arithmetic it replaces, so on such a target the transform runs in the tower basis
/// instead. The choice is a constant and only one arm survives compilation.
#[derive(Clone, Debug, Default)]
pub struct PolyBasisNtt {
    tower: LchNtt<BinaryField128>,
}

/// The twiddle of block `blk` of stage `j`, in the polynomial basis.
#[inline]
fn twiddle(base: BinaryField128, blk: usize) -> u128 {
    poly_basis::from_tower(base + domain_point::<BinaryField128>(blk << 1))
}

impl AdditiveNtt<BinaryField128> for PolyBasisNtt {
    fn shifted_ntt_batch(
        &self,
        mut mat: RowMajorMatrix<BinaryField128>,
        shift: BinaryField128,
    ) -> RowMajorMatrix<BinaryField128> {
        if !poly_basis::HAS_HARDWARE_CLMUL {
            return self.tower.shifted_ntt_batch(mat, shift);
        }

        let width = mat.width();
        let log_n = log2_strict_usize(mat.height());
        let mut values: Vec<u128> = mat
            .values
            .par_iter()
            .map(|&x| poly_basis::from_tower(x))
            .collect();

        for j in (0..log_n).rev() {
            let half = (1 << j) * width;
            let base = subspace_polynomial::<BinaryField128>(j, shift);
            values
                .par_chunks_mut(half << 1)
                .enumerate()
                .for_each(|(blk, block)| {
                    let t = twiddle(base, blk);
                    let (lo, hi) = block.split_at_mut(half);
                    lo.par_chunks_mut(BUTTERFLY_GRAIN)
                        .zip(hi.par_chunks_mut(BUTTERFLY_GRAIN))
                        .for_each(|(lo, hi)| {
                            if t == 0 {
                                for (u, v) in lo.iter_mut().zip(hi) {
                                    *v ^= *u;
                                }
                            } else {
                                for (u, v) in lo.iter_mut().zip(hi) {
                                    *u ^= poly_basis::mul(t, *v);
                                    *v ^= *u;
                                }
                            }
                        });
                });
        }

        mat.values
            .par_iter_mut()
            .zip(values.par_iter())
            .for_each(|(x, &v)| *x = poly_basis::to_tower(v));
        mat
    }

    fn shifted_intt_batch(
        &self,
        mut mat: RowMajorMatrix<BinaryField128>,
        shift: BinaryField128,
    ) -> RowMajorMatrix<BinaryField128> {
        if !poly_basis::HAS_HARDWARE_CLMUL {
            return self.tower.shifted_intt_batch(mat, shift);
        }

        let width = mat.width();
        let log_n = log2_strict_usize(mat.height());
        let mut values: Vec<u128> = mat
            .values
            .par_iter()
            .map(|&x| poly_basis::from_tower(x))
            .collect();

        for j in 0..log_n {
            let half = (1 << j) * width;
            let base = subspace_polynomial::<BinaryField128>(j, shift);
            values
                .par_chunks_mut(half << 1)
                .enumerate()
                .for_each(|(blk, block)| {
                    let t = twiddle(base, blk);
                    let (lo, hi) = block.split_at_mut(half);
                    lo.par_chunks_mut(BUTTERFLY_GRAIN)
                        .zip(hi.par_chunks_mut(BUTTERFLY_GRAIN))
                        .for_each(|(lo, hi)| {
                            if t == 0 {
                                for (u, v) in lo.iter_mut().zip(hi) {
                                    *v ^= *u;
                                }
                            } else {
                                for (u, v) in lo.iter_mut().zip(hi) {
                                    *v ^= *u;
                                    *u ^= poly_basis::mul(t, *v);
                                }
                            }
                        });
                });
        }

        mat.values
            .par_iter_mut()
            .zip(values.par_iter())
            .for_each(|(x, &v)| *x = poly_basis::to_tower(v));
        mat
    }
}

#[cfg(test)]
mod tests {
    use p3_binary_field::{BinaryField128, TowerLevel};
    use p3_matrix::dense::RowMajorMatrix;
    use proptest::prelude::*;

    use super::PolyBasisNtt;
    use crate::naive::NaiveAdditiveNtt;
    use crate::traits::AdditiveNtt;

    /// Builds a matrix whose entries are distinct functions of the seed and the position.
    fn matrix(log_n: usize, width: usize, seed: u64) -> RowMajorMatrix<BinaryField128> {
        RowMajorMatrix::new(
            (0..(width << log_n))
                .map(|i| {
                    let bits = seed
                        .wrapping_mul(0x9e37_79b9_7f4a_7c15)
                        .wrapping_add(i as u64);
                    BinaryField128::from_le_byte_iter(bits.to_le_bytes().into_iter().cycle())
                })
                .collect(),
            width,
        )
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(32))]

        /// The polynomial-basis transform is the same map as the reference oracle.
        #[test]
        fn poly_basis_matches_naive(
            log_n in 0usize..=8,
            width in 1usize..=5,
            seed in any::<u64>(),
            shift in any::<u64>(),
        ) {
            let coeffs = matrix(log_n, width, seed);
            let shift = BinaryField128::from_le_byte_iter(
                shift.to_le_bytes().into_iter().cycle(),
            );

            let fast = PolyBasisNtt::default().shifted_ntt_batch(coeffs.clone(), shift);
            let slow = NaiveAdditiveNtt::<BinaryField128>::default()
                .shifted_ntt_batch(coeffs, shift);
            prop_assert_eq!(fast, slow);
        }

        #[test]
        fn poly_basis_round_trips(
            log_n in 0usize..=8,
            width in 1usize..=3,
            seed in any::<u64>(),
            shift in any::<u64>(),
        ) {
            let coeffs = matrix(log_n, width, seed);
            let shift = BinaryField128::from_le_byte_iter(
                shift.to_le_bytes().into_iter().cycle(),
            );
            let ntt = PolyBasisNtt::default();
            let evals = ntt.shifted_ntt_batch(coeffs.clone(), shift);
            prop_assert_eq!(ntt.shifted_intt_batch(evals, shift), coeffs);
        }
    }
}
