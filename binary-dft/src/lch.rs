//! The Lin–Chung–Han additive NTT.

use core::marker::PhantomData;

use p3_binary_field::TowerLevel;
use p3_field::{PackedValue, PrimeCharacteristicRing};
use p3_matrix::Matrix;
use p3_matrix::dense::RowMajorMatrix;
use p3_maybe_rayon::prelude::*;
use p3_util::log2_strict_usize;

use crate::domain::{domain_point, domain_point_steps, subspace_polynomial};
use crate::traits::AdditiveNtt;

/// The Lin–Chung–Han additive NTT over a Cantor-basis domain.
///
/// At stage `j`, block `blk` has twiddle `W_j(shift) + domain_point(blk << 1)`.
/// There is no twiddle table.
///
/// Consecutive blocks differ by a fixed increment, shared by every stage.
///
/// Butterflies run on SIMD packings, and drop the multiply where the twiddle is zero.
#[derive(Clone, Debug, Default)]
pub struct LchNtt<F> {
    _marker: PhantomData<F>,
}

/// Elements per side of a butterfly task.
///
/// Narrow blocks share a task.
/// Wide blocks split into pieces of this size, which have too few blocks to fill a machine.
/// Either way a stage leaves enough independent tasks to be worth handing to other threads.
pub(crate) const BUTTERFLY_GRAIN: usize = 1 << 10;

impl<F: TowerLevel> AdditiveNtt<F> for LchNtt<F> {
    fn shifted_ntt_batch(&self, mut mat: RowMajorMatrix<F>, shift: F) -> RowMajorMatrix<F> {
        let width = mat.width();
        let log_n = log2_strict_usize(mat.height());

        // An increment depends only on a block index's trailing-zero count, never on the stage,
        // so one table serves every stage and each stage uses the prefix it reaches.
        // A height of one runs no stage and needs no table.
        let steps = domain_point_steps::<F>(log_n.saturating_sub(1));
        for j in (0..log_n).rev() {
            let half = (1 << j) * width;
            // D8: the block starting at row `b` of the coset `shift + S_ℓ` has twiddle
            // `W_j(shift) + point(b >> j)`, and block `blk` starts at row `blk << (j + 1)`.
            let base = subspace_polynomial::<F>(j, shift);
            let per_task = (BUTTERFLY_GRAIN / half).max(1);
            mat.values
                .par_chunks_mut(per_task * (half << 1))
                .enumerate()
                .for_each(|(task, group)| {
                    let first = task * per_task;
                    let mut t = base + domain_point::<F>(first << 1);
                    // Invariant: blocks are visited in ascending index order.
                    // Carrying the twiddle from one block to the next relies on it.
                    for (i, block) in group.chunks_mut(half << 1).enumerate() {
                        if i != 0 {
                            t += steps[(first + i).trailing_zeros() as usize];
                        }
                        let (lo, hi) = block.split_at_mut(half);
                        let butterfly = |lo: &mut [F], hi: &mut [F]| {
                            packed_butterfly::<F, false>(lo, hi, t);
                        };
                        // Pairs are independent across the block, so a block wider than the
                        // grain is split further rather than run on a single thread.
                        if half <= BUTTERFLY_GRAIN {
                            butterfly(lo, hi);
                        } else {
                            lo.par_chunks_mut(BUTTERFLY_GRAIN)
                                .zip(hi.par_chunks_mut(BUTTERFLY_GRAIN))
                                .for_each(|(lo, hi)| butterfly(lo, hi));
                        }
                    }
                });
        }
        mat
    }

    fn shifted_intt_batch(&self, mut mat: RowMajorMatrix<F>, shift: F) -> RowMajorMatrix<F> {
        let width = mat.width();
        let log_n = log2_strict_usize(mat.height());

        // An increment depends only on a block index's trailing-zero count, never on the stage,
        // so one table serves every stage and each stage uses the prefix it reaches.
        // A height of one runs no stage and needs no table.
        let steps = domain_point_steps::<F>(log_n.saturating_sub(1));
        for j in 0..log_n {
            let half = (1 << j) * width;
            // Twiddles as derived in `shifted_ntt_batch`, with the stages run in reverse.
            let base = subspace_polynomial::<F>(j, shift);
            let per_task = (BUTTERFLY_GRAIN / half).max(1);
            mat.values
                .par_chunks_mut(per_task * (half << 1))
                .enumerate()
                .for_each(|(task, group)| {
                    let first = task * per_task;
                    let mut t = base + domain_point::<F>(first << 1);
                    // Invariant: blocks are visited in ascending index order.
                    // Carrying the twiddle from one block to the next relies on it.
                    for (i, block) in group.chunks_mut(half << 1).enumerate() {
                        if i != 0 {
                            t += steps[(first + i).trailing_zeros() as usize];
                        }
                        let (lo, hi) = block.split_at_mut(half);
                        let butterfly = |lo: &mut [F], hi: &mut [F]| {
                            packed_butterfly::<F, true>(lo, hi, t);
                        };
                        // Pairs are independent across the block, so a block wider than the
                        // grain is split further rather than run on a single thread.
                        if half <= BUTTERFLY_GRAIN {
                            butterfly(lo, hi);
                        } else {
                            lo.par_chunks_mut(BUTTERFLY_GRAIN)
                                .zip(hi.par_chunks_mut(BUTTERFLY_GRAIN))
                                .for_each(|(lo, hi)| butterfly(lo, hi));
                        }
                    }
                });
        }
        mat
    }
}

/// Apply a butterfly to full SIMD vectors and any remaining scalar elements.
#[inline]
fn packed_butterfly<F: TowerLevel, const INVERSE: bool>(lo: &mut [F], hi: &mut [F], t: F) {
    // Both sides have equal length, so their packed prefixes and tails pair exactly.
    let (lo, lo_tail) = F::Packing::pack_slice_with_suffix_mut(lo);
    let (hi, hi_tail) = F::Packing::pack_slice_with_suffix_mut(hi);
    let zero = t.is_zero();
    butterfly_values::<_, INVERSE>(lo, hi, t.into(), zero);
    butterfly_values::<_, INVERSE>(lo_tail, hi_tail, t, zero);
}

/// Apply the same field identities to scalar or packed values.
#[inline]
fn butterfly_values<R: PrimeCharacteristicRing + Copy, const INVERSE: bool>(
    lo: &mut [R],
    hi: &mut [R],
    t: R,
    zero: bool,
) {
    if zero {
        // A zero twiddle reduces both transform directions to (u, u + v).
        for (u, v) in lo.iter_mut().zip(hi) {
            *v += *u;
        }
    } else if INVERSE {
        // Recover v first, then remove its twiddle contribution from u.
        for (u, v) in lo.iter_mut().zip(hi) {
            *v += *u;
            *u += t * *v;
        }
    } else {
        // Evaluate the pair as (u + t*v, u + t*v + v).
        for (u, v) in lo.iter_mut().zip(hi) {
            *u += t * *v;
            *v += *u;
        }
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec;

    use p3_binary_field::{
        BinaryField8, BinaryField16, BinaryField32, BinaryField64, BinaryField128, Ghash128,
        TowerLevel,
    };
    use p3_field::PrimeCharacteristicRing;
    use p3_matrix::Matrix;
    use p3_matrix::dense::RowMajorMatrix;
    use p3_util::log2_strict_usize;
    use proptest::prelude::*;

    use super::LchNtt;
    use crate::domain::{domain_point, subspace_polynomial};
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

    /// The forward transform with every twiddle walked out from its own block index, in one
    /// serial pass and with no zero shortcut.
    fn twiddle_walk_ntt<F: TowerLevel>(mut mat: RowMajorMatrix<F>, shift: F) -> RowMajorMatrix<F> {
        let width = mat.width();
        let log_n = log2_strict_usize(mat.height());
        for j in (0..log_n).rev() {
            let half = (1 << j) * width;
            let base = subspace_polynomial::<F>(j, shift);
            for (blk, block) in mat.values.chunks_mut(half << 1).enumerate() {
                let t = base + domain_point::<F>(blk << 1);
                let (lo, hi) = block.split_at_mut(half);
                for (u, v) in lo.iter_mut().zip(hi) {
                    *u += t * *v;
                    *v += *u;
                }
            }
        }
        mat
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
        fn ghash_packing_matches_naive_and_round_trips(
            log_n in 0usize..=7,
            width in 1usize..=9,
            seed: u64,
            shift: u64,
        ) {
            // Odd widths force scalar tails alongside SIMD prefixes.
            check_matches_naive::<Ghash128>(log_n, width, seed, shift);
            let coeffs = matrix::<Ghash128>(log_n, width, seed);
            let shift = sample::<Ghash128>(shift);
            let ntt = LchNtt::<Ghash128>::default();
            let transformed = ntt.shifted_ntt_batch(coeffs.clone(), shift);
            prop_assert_eq!(ntt.shifted_intt_batch(transformed, shift), coeffs);
        }

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

        /// The low-degree extension agrees with the oracle, on a coset too.
        /// The input rows reappear as the prefix of the output.
        #[test]
        fn lch_lde_matches_naive(
            log_n in 0usize..=6,
            added in 0usize..=3,
            width in 1usize..=3,
            seed in any::<u64>(),
            shift in any::<u64>(),
        ) {
            let coeffs = matrix::<BinaryField16>(log_n, width, seed);
            let shift = sample::<BinaryField16>(shift);

            let lde = LchNtt::<BinaryField16>::default()
                .shifted_lde_batch(coeffs.clone(), added, shift);
            let naive = NaiveAdditiveNtt::<BinaryField16>::default()
                .shifted_lde_batch(coeffs.clone(), added, shift);
            prop_assert_eq!(&lde, &naive);
            prop_assert_eq!(&lde.values[..coeffs.values.len()], &coeffs.values[..]);
        }
    }

    /// A height whose stages take more than one butterfly task, so a task seeds its twiddle at
    /// a block index of its own rather than at zero.
    ///
    /// The round-trip test is blind to the schedule.
    /// Both directions read the same twiddles, so they invert each other regardless.
    #[test]
    fn lch_matches_a_twiddle_walk_across_several_tasks() {
        const LOG_N: usize = 12;
        let ntt = LchNtt::<BinaryField128>::default();
        for width in [1usize, 3] {
            for shift_bits in [0u64, 0x1234_5678_9abc_def0] {
                let coeffs = matrix::<BinaryField128>(LOG_N, width, 5);
                let shift = sample::<BinaryField128>(shift_bits);

                let walked = twiddle_walk_ntt::<BinaryField128>(coeffs.clone(), shift);
                assert_eq!(
                    ntt.shifted_ntt_batch(coeffs.clone(), shift),
                    walked,
                    "ntt width={width} shift={shift_bits:#x}"
                );
                // The inverse has its own copy of the schedule, and undoing a codeword the walk
                // produced is what holds that copy to the same twiddles.
                assert_eq!(
                    ntt.shifted_intt_batch(walked, shift),
                    coeffs,
                    "intt width={width} shift={shift_bits:#x}"
                );
            }
        }
    }

    /// An index past the field's bit width needs a Cantor vector it does not have.
    #[test]
    #[should_panic]
    fn shifted_ntt_batch_rejects_l_past_the_bit_width() {
        // `BinaryField8` has `2^LOG_BITS = 8` Cantor basis vectors, indices `0..8`.
        let coeffs = matrix::<BinaryField8>((1 << BinaryField8::LOG_BITS) + 1, 1, 0);
        let _ = LchNtt::<BinaryField8>::default().ntt_batch(coeffs);
    }

    /// A height that is not a power of two has no well-defined `l`.
    #[test]
    #[should_panic]
    fn shifted_ntt_batch_rejects_a_non_power_of_two_height() {
        let coeffs = RowMajorMatrix::new(vec![sample::<BinaryField8>(0); 3], 1);
        let _ = LchNtt::<BinaryField8>::default().ntt_batch(coeffs);
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

    /// The domain does not depend on the level, so neither does the transform.
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

    #[test]
    fn ghash_packing_matches_scalar_across_tasks() {
        // These blocks cross the task-size boundary and leave scalar tails at narrow stages.
        for width in [3, 5] {
            for shift in [Ghash128::ZERO, sample::<Ghash128>(17)] {
                let coeffs = matrix::<Ghash128>(11, width, 29);
                let ntt = LchNtt::<Ghash128>::default();
                let actual = ntt.shifted_ntt_batch(coeffs.clone(), shift);
                // The serial oracle uses scalar products and recomputes every twiddle.
                assert_eq!(actual, twiddle_walk_ntt(coeffs.clone(), shift));
                assert_eq!(ntt.shifted_intt_batch(actual, shift), coeffs);
            }
        }
    }

    /// The transform must commute with the change of basis between the two representations.
    ///
    /// It is built from twiddle multiplies.
    /// Only a field isomorphism preserves multiplication, so this pins that too.
    #[test]
    fn the_two_representations_of_the_widest_level_transform_alike() {
        const LOG_N: usize = 7;
        const WIDTH: usize = 3;

        // Express one matrix in both field bases.
        let tower_coeffs = matrix::<BinaryField128>(LOG_N, WIDTH, 11);
        let ghash_coeffs = RowMajorMatrix::new(
            tower_coeffs
                .values
                .iter()
                .copied()
                .map(Ghash128::from)
                .collect(),
            WIDTH,
        );

        let shift = sample::<BinaryField128>(0x0123_4567_89ab_cdef);

        let tower = LchNtt::<BinaryField128>::default().shifted_ntt_batch(tower_coeffs, shift);
        let ghash =
            LchNtt::<Ghash128>::default().shifted_ntt_batch(ghash_coeffs, Ghash128::from(shift));

        // Converting before or after the transform must give the same values.
        for (t, g) in tower.values.iter().zip(&ghash.values) {
            assert_eq!(Ghash128::from(*t), *g);
        }
    }
}
