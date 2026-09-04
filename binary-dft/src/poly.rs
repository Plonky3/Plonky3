//! The Lin–Chung–Han transform carried out in the polynomial basis of `GF(2^128)`.

use alloc::vec::Vec;

use p3_binary_field::{BinaryField128, TowerLevel, poly_basis};
use p3_matrix::Matrix;
use p3_matrix::dense::RowMajorMatrix;
use p3_maybe_rayon::prelude::*;
use p3_util::log2_strict_usize;

use crate::domain::{domain_point, domain_point_steps, subspace_polynomial};
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

/// [`domain_point_steps`] carried into the polynomial basis.
///
/// The change of basis is additive, so an increment converted here applies to a twiddle
/// already in this basis by `XOR`.
fn twiddle_steps(count: usize) -> Vec<u128> {
    domain_point_steps::<BinaryField128>(count)
        .into_iter()
        .map(poly_basis::from_tower)
        .collect()
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
        // `BinaryField128` is `#[repr(transparent)]` over `u128`, so this reuses `mat.values`'s
        // allocation instead of allocating a second codeword buffer: `to_repr` is the identity
        // bit pattern, and the specialised `Vec` collect below reinterprets the buffer in place.
        let mut values: Vec<u128> = core::mem::take(&mut mat.values)
            .into_iter()
            .map(BinaryField128::to_repr)
            .collect();
        values
            .par_iter_mut()
            .for_each(|v| *v = poly_basis::from_tower(BinaryField128::from_repr(*v)));

        // An increment depends only on a block index's trailing-zero count, never on the stage,
        // so one table serves every stage and each stage uses the prefix it reaches.
        // A height of one runs no stage and needs no table.
        let steps = twiddle_steps(log_n.saturating_sub(1));
        for j in (0..log_n).rev() {
            let half = (1 << j) * width;
            let base = subspace_polynomial::<BinaryField128>(j, shift);
            let per_task = (BUTTERFLY_GRAIN / half).max(1);
            values
                .par_chunks_mut(per_task * (half << 1))
                .enumerate()
                .for_each(|(task, group)| {
                    let first = task * per_task;
                    let mut t = twiddle(base, first);
                    // Invariant: blocks are visited in ascending index order.
                    // Carrying the twiddle from one block to the next relies on it.
                    for (i, block) in group.chunks_mut(half << 1).enumerate() {
                        if i != 0 {
                            t ^= steps[(first + i).trailing_zeros() as usize];
                        }
                        let zero = t == 0;
                        let (lo, hi) = block.split_at_mut(half);
                        let butterfly = |lo: &mut [u128], hi: &mut [u128]| {
                            if zero {
                                for (u, v) in lo.iter_mut().zip(hi) {
                                    *v ^= *u;
                                }
                            } else {
                                for (u, v) in lo.iter_mut().zip(hi) {
                                    *u ^= poly_basis::mul(t, *v);
                                    *v ^= *u;
                                }
                            }
                        };
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

        values
            .par_iter_mut()
            .for_each(|v| *v = poly_basis::to_tower(*v).to_repr());
        mat.values = values.into_iter().map(BinaryField128::from_repr).collect();
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
        // See `shifted_ntt_batch`: reuses `mat.values`'s allocation instead of a second buffer.
        let mut values: Vec<u128> = core::mem::take(&mut mat.values)
            .into_iter()
            .map(BinaryField128::to_repr)
            .collect();
        values
            .par_iter_mut()
            .for_each(|v| *v = poly_basis::from_tower(BinaryField128::from_repr(*v)));

        // An increment depends only on a block index's trailing-zero count, never on the stage,
        // so one table serves every stage and each stage uses the prefix it reaches.
        // A height of one runs no stage and needs no table.
        let steps = twiddle_steps(log_n.saturating_sub(1));
        for j in 0..log_n {
            let half = (1 << j) * width;
            let base = subspace_polynomial::<BinaryField128>(j, shift);
            let per_task = (BUTTERFLY_GRAIN / half).max(1);
            values
                .par_chunks_mut(per_task * (half << 1))
                .enumerate()
                .for_each(|(task, group)| {
                    let first = task * per_task;
                    let mut t = twiddle(base, first);
                    // Invariant: blocks are visited in ascending index order.
                    // Carrying the twiddle from one block to the next relies on it.
                    for (i, block) in group.chunks_mut(half << 1).enumerate() {
                        if i != 0 {
                            t ^= steps[(first + i).trailing_zeros() as usize];
                        }
                        let zero = t == 0;
                        let (lo, hi) = block.split_at_mut(half);
                        let butterfly = |lo: &mut [u128], hi: &mut [u128]| {
                            if zero {
                                for (u, v) in lo.iter_mut().zip(hi) {
                                    *v ^= *u;
                                }
                            } else {
                                for (u, v) in lo.iter_mut().zip(hi) {
                                    *v ^= *u;
                                    *u ^= poly_basis::mul(t, *v);
                                }
                            }
                        };
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

        values
            .par_iter_mut()
            .for_each(|v| *v = poly_basis::to_tower(*v).to_repr());
        mat.values = values.into_iter().map(BinaryField128::from_repr).collect();
        mat
    }
}

#[cfg(test)]
mod tests {
    use p3_binary_field::{BinaryField128, TowerLevel};
    use p3_matrix::dense::RowMajorMatrix;
    use proptest::prelude::*;

    use super::PolyBasisNtt;
    use crate::lch::LchNtt;
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

        /// `PolyBasisNtt`'s low-degree extension agrees with the oracle's, on a coset too, and
        /// the input rows reappear as the prefix: the correspondence Phase 3 folds along.
        #[test]
        fn poly_basis_lde_matches_naive(
            log_n in 0usize..=6,
            added in 0usize..=3,
            width in 1usize..=3,
            seed in any::<u64>(),
            shift in any::<u64>(),
        ) {
            let coeffs = matrix(log_n, width, seed);
            let shift = BinaryField128::from_le_byte_iter(
                shift.to_le_bytes().into_iter().cycle(),
            );

            let lde = PolyBasisNtt::default().shifted_lde_batch(coeffs.clone(), added, shift);
            let naive = NaiveAdditiveNtt::<BinaryField128>::default()
                .shifted_lde_batch(coeffs.clone(), added, shift);
            prop_assert_eq!(&lde, &naive);
            prop_assert_eq!(&lde.values[..coeffs.values.len()], &coeffs.values[..]);
        }
    }

    /// A height whose stages take more than one butterfly task, so a task seeds its twiddle at
    /// a block index of its own rather than at zero. The oracle tests all sit below that
    /// height, so `LchNtt` stands in for the oracle here, itself held to an independent
    /// twiddle walk at this same height by `lch_matches_a_twiddle_walk_across_several_tasks`.
    #[test]
    fn poly_basis_matches_the_tower_across_several_tasks() {
        const LOG_N: usize = 12;
        let poly = PolyBasisNtt::default();
        let tower = LchNtt::<BinaryField128>::default();
        for width in [1usize, 3] {
            for shift_bits in [0u64, 0x1234_5678_9abc_def0] {
                let coeffs = matrix(LOG_N, width, 5);
                let shift =
                    BinaryField128::from_le_byte_iter(shift_bits.to_le_bytes().into_iter().cycle());

                let evals = tower.shifted_ntt_batch(coeffs.clone(), shift);
                assert_eq!(
                    poly.shifted_ntt_batch(coeffs.clone(), shift),
                    evals,
                    "ntt width={width} shift={shift_bits:#x}"
                );
                assert_eq!(
                    poly.shifted_intt_batch(evals, shift),
                    coeffs,
                    "intt width={width} shift={shift_bits:#x}"
                );
            }
        }
    }

    /// A height that is not a power of two has no well-defined `ℓ`. `ℓ` exceeding the bit width
    /// of `BinaryField128` is covered generically at [`LchNtt`](crate::LchNtt), where the level
    /// is a type parameter and the panic is cheap to reach; at a fixed `BinaryField128` it is
    /// only reachable through a matrix of `2^129` rows, which is not a test worth writing.
    #[test]
    #[should_panic]
    fn shifted_ntt_batch_rejects_a_non_power_of_two_height() {
        let coeffs = RowMajorMatrix::new(matrix(0, 1, 0).values.repeat(3), 1);
        let _ = PolyBasisNtt::default().ntt_batch(coeffs);
    }
}
