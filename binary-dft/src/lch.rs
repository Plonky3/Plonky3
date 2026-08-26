//! The Lin–Chung–Han additive NTT.

use core::marker::PhantomData;

use p3_binary_field::TowerLevel;
use p3_matrix::Matrix;
use p3_matrix::dense::RowMajorMatrix;
use p3_maybe_rayon::prelude::*;
use p3_util::log2_strict_usize;

use crate::domain::{domain_point, subspace_polynomial};
use crate::traits::AdditiveNtt;

/// The Lin–Chung–Han additive NTT over the Cantor-basis domain.
///
/// Twiddles are index shifts (D8): at stage `j` and butterfly block `blk` the twiddle is
/// `W_j(shift) + domain_point(blk << 1)`, so there is no twiddle table and no per-size
/// precomputation.
///
/// `W_j` is `F_2`-linear and `domain_point(0)` is zero, so over the subspace itself — the
/// coset with `shift = 0` — the first block of every stage has a zero twiddle and its
/// butterfly collapses to a single addition. The inner loop takes that as a separate case,
/// which removes one multiply in `2/ℓ` of them.
#[derive(Clone, Debug, Default)]
pub struct LchNtt<F> {
    _marker: PhantomData<F>,
}

/// The number of field elements one butterfly task covers on each side of a block.
///
/// Stage `j` has `2^(ℓ − 1 − j)` blocks of `half = 2^j · width` elements per side, so cutting
/// each side into pieces of this size leaves `n · width / (2 · BUTTERFLY_GRAIN)` pieces at
/// every stage, independent of `j`: the wide stages, which have too few blocks to fill a
/// machine, are split from within instead. Stages with `half ≤ BUTTERFLY_GRAIN` keep a single
/// piece per side and pay nothing for the extra level.
///
/// At roughly twenty nanoseconds per `GF(2^128)` butterfly a piece of this size is tens of
/// microseconds of work, orders of magnitude above the cost of handing a task to another
/// thread, while still leaving hundreds of pieces per stage at the smallest useful heights.
pub(crate) const BUTTERFLY_GRAIN: usize = 1 << 10;

impl<F: TowerLevel> AdditiveNtt<F> for LchNtt<F> {
    fn shifted_ntt_batch(&self, mut mat: RowMajorMatrix<F>, shift: F) -> RowMajorMatrix<F> {
        let width = mat.width();
        let log_n = log2_strict_usize(mat.height());

        for j in (0..log_n).rev() {
            let half = (1 << j) * width;
            // D8: the block starting at row `b` of the coset `shift + S_ℓ` has twiddle
            // `W_j(shift) + point(b >> j)`, and block `blk` starts at row `blk << (j + 1)`.
            let base = subspace_polynomial::<F>(j, shift);
            mat.values
                .par_chunks_mut(half << 1)
                .enumerate()
                .for_each(|(blk, block)| {
                    let t = base + domain_point::<F>(blk << 1);
                    let (lo, hi) = block.split_at_mut(half);
                    // Pairs are independent across the block, so a block wider than the grain
                    // is split further rather than run on a single thread.
                    lo.par_chunks_mut(BUTTERFLY_GRAIN)
                        .zip(hi.par_chunks_mut(BUTTERFLY_GRAIN))
                        .for_each(|(lo, hi)| {
                            if t.is_zero() {
                                // (u, v) ↦ (u, u + v)
                                for (u, v) in lo.iter_mut().zip(hi) {
                                    *v += *u;
                                }
                            } else {
                                for (u, v) in lo.iter_mut().zip(hi) {
                                    // (u, v) ↦ (u + t·v, u + t·v + v)
                                    *u += t * *v;
                                    *v += *u;
                                }
                            }
                        });
                });
        }
        mat
    }

    fn shifted_intt_batch(&self, mut mat: RowMajorMatrix<F>, shift: F) -> RowMajorMatrix<F> {
        let width = mat.width();
        let log_n = log2_strict_usize(mat.height());

        for j in 0..log_n {
            let half = (1 << j) * width;
            // Twiddles as derived in `shifted_ntt_batch`, with the stages run in reverse.
            let base = subspace_polynomial::<F>(j, shift);
            mat.values
                .par_chunks_mut(half << 1)
                .enumerate()
                .for_each(|(blk, block)| {
                    let t = base + domain_point::<F>(blk << 1);
                    let (lo, hi) = block.split_at_mut(half);
                    // Pairs are independent across the block, so a block wider than the grain
                    // is split further rather than run on a single thread.
                    lo.par_chunks_mut(BUTTERFLY_GRAIN)
                        .zip(hi.par_chunks_mut(BUTTERFLY_GRAIN))
                        .for_each(|(lo, hi)| {
                            if t.is_zero() {
                                // (u', v') ↦ (u = u', v = u' + v')
                                for (u, v) in lo.iter_mut().zip(hi) {
                                    *v += *u;
                                }
                            } else {
                                for (u, v) in lo.iter_mut().zip(hi) {
                                    // (u', v') ↦ (u = u' + t·v, v = u' + v')
                                    *v += *u;
                                    *u += t * *v;
                                }
                            }
                        });
                });
        }
        mat
    }
}

#[cfg(test)]
mod tests {
    use p3_binary_field::{
        BinaryField8, BinaryField16, BinaryField32, BinaryField64, BinaryField128, TowerLevel,
    };
    use p3_matrix::dense::RowMajorMatrix;
    use proptest::prelude::*;

    use super::LchNtt;
    use crate::naive::NaiveAdditiveNtt;
    use crate::traits::AdditiveNtt;

    /// Builds an element of any level from a 64-bit pattern, repeating it for wider levels.
    fn sample<F: TowerLevel>(bits: u64) -> F {
        F::from_le_byte_iter(bits.to_le_bytes().into_iter().cycle())
    }

    /// Builds a matrix whose entries are distinct functions of the seed and the position.
    fn matrix<F: TowerLevel>(log_n: usize, width: usize, seed: u64) -> RowMajorMatrix<F> {
        RowMajorMatrix::new(
            (0..(width << log_n))
                .map(|i| {
                    sample::<F>(
                        seed.wrapping_mul(0x9e37_79b9_7f4a_7c15)
                            .wrapping_add(i as u64),
                    )
                })
                .collect(),
            width,
        )
    }

    /// `LchNtt` agrees with the oracle on a random matrix and a random coset.
    fn check_matches_naive<F: TowerLevel>(log_n: usize, width: usize, seed: u64, shift: u64) {
        let coeffs = matrix::<F>(log_n, width, seed);
        let shift = sample::<F>(shift);

        let fast = LchNtt::<F>::default().shifted_ntt_batch(coeffs.clone(), shift);
        let slow = NaiveAdditiveNtt::<F>::default().shifted_ntt_batch(coeffs, shift);
        assert_eq!(fast, slow);
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(32))]

        #[test]
        fn lch_matches_naive_at_8_bits(
            log_n in 0usize..=8,
            width in 1usize..=5,
            seed in any::<u64>(),
            shift in any::<u64>(),
        ) {
            check_matches_naive::<BinaryField8>(log_n, width, seed, shift);
        }

        #[test]
        fn lch_matches_naive_at_16_bits(
            log_n in 0usize..=8,
            width in 1usize..=5,
            seed in any::<u64>(),
            shift in any::<u64>(),
        ) {
            check_matches_naive::<BinaryField16>(log_n, width, seed, shift);
        }

        #[test]
        fn lch_matches_naive_at_32_bits(
            log_n in 0usize..=8,
            width in 1usize..=5,
            seed in any::<u64>(),
            shift in any::<u64>(),
        ) {
            check_matches_naive::<BinaryField32>(log_n, width, seed, shift);
        }

        #[test]
        fn lch_matches_naive_at_64_bits(
            log_n in 0usize..=8,
            width in 1usize..=5,
            seed in any::<u64>(),
            shift in any::<u64>(),
        ) {
            check_matches_naive::<BinaryField64>(log_n, width, seed, shift);
        }

        #[test]
        fn lch_matches_naive_at_128_bits(
            log_n in 0usize..=8,
            width in 1usize..=5,
            seed in any::<u64>(),
            shift in any::<u64>(),
        ) {
            check_matches_naive::<BinaryField128>(log_n, width, seed, shift);
        }

        #[test]
        fn lch_round_trips(
            log_n in 0usize..=10,
            width in 1usize..=3,
            seed in any::<u64>(),
            shift in any::<u64>(),
        ) {
            let coeffs = matrix::<BinaryField16>(log_n, width, seed);
            let shift = sample::<BinaryField16>(shift);
            let ntt = LchNtt::<BinaryField16>::default();
            let evals = ntt.shifted_ntt_batch(coeffs.clone(), shift);
            prop_assert_eq!(ntt.shifted_intt_batch(evals, shift), coeffs);
        }
    }

    /// The transform is `F_2`-linear in the coefficient vector.
    #[test]
    fn lch_is_linear() {
        let ntt = LchNtt::<BinaryField16>::default();
        let a = matrix::<BinaryField16>(5, 2, 1);
        let b = matrix::<BinaryField16>(5, 2, 2);
        let sum = RowMajorMatrix::new(
            a.values
                .iter()
                .zip(&b.values)
                .map(|(x, y)| *x + *y)
                .collect(),
            2,
        );

        let lhs = ntt.ntt_batch(sum);
        let rhs_a = ntt.ntt_batch(a);
        let rhs_b = ntt.ntt_batch(b);
        for (i, v) in lhs.values.iter().enumerate() {
            assert_eq!(*v, rhs_a.values[i] + rhs_b.values[i]);
        }
    }

    /// The same data transformed at two levels agrees after embedding: the domain does not
    /// depend on the level, so neither does the transform.
    #[test]
    fn levels_agree_after_embedding() {
        const LOG_N: usize = 6;
        let coeffs32 = matrix::<BinaryField32>(LOG_N, 1, 7);
        let coeffs128 = RowMajorMatrix::new(
            coeffs32
                .values
                .iter()
                .map(|v| BinaryField128::from_repr(u128::from(v.to_repr())))
                .collect(),
            1,
        );

        let small = LchNtt::<BinaryField32>::default().ntt_batch(coeffs32);
        let large = LchNtt::<BinaryField128>::default().ntt_batch(coeffs128);
        for (s, l) in small.values.iter().zip(&large.values) {
            assert_eq!(u128::from(s.to_repr()), l.to_repr());
        }
    }
}
