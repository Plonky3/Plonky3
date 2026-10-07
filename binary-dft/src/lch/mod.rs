//! The Lin-Chung-Han additive NTT.

mod schedule;
mod twiddles;

use alloc::vec::Vec;
use core::marker::PhantomData;

use p3_matrix::Matrix;
use p3_matrix::dense::RowMajorMatrix;
use p3_maybe_rayon::prelude::*;
use p3_util::log2_strict_usize;
pub(crate) use schedule::BUTTERFLY_GRAIN;
use schedule::{DEEP_TILE_BYTES, Schedule, run, run_cosets, stage_pass};
use twiddles::Twiddles;

use crate::butterfly::ButterflyField;
use crate::domain::domain_point;
use crate::traits::AdditiveNtt;

/// The Lin-Chung-Han additive NTT over the Cantor-basis domain.
///
/// - Twiddles are index shifts, so there is no twiddle table.
/// - Each block steps its twiddle from the block before it.
/// - Over the subspace itself, most twiddles lie in a small subfield, which the butterfly exploits.
/// - Adjacent stages are grouped into cache-sized row sets, so the matrix is swept a few times, not once per stage.
#[derive(Clone, Debug, Default)]
pub struct LchNtt<F> {
    _marker: PhantomData<F>,
}

impl<F: ButterflyField> AdditiveNtt<F> for LchNtt<F> {
    fn shifted_ntt_batch(&self, mut mat: RowMajorMatrix<F>, shift: F) -> RowMajorMatrix<F> {
        let width = mat.width();
        F::lch_transform::<false>(&mut mat.values, width, shift);
        mat
    }

    fn shifted_intt_batch(&self, mut mat: RowMajorMatrix<F>, shift: F) -> RowMajorMatrix<F> {
        let width = mat.width();
        F::lch_transform::<true>(&mut mat.values, width, shift);
        mat
    }

    fn ntt_batch_padded(
        &self,
        mut mat: RowMajorMatrix<F>,
        log_inv_rate: usize,
    ) -> RowMajorMatrix<F> {
        let width = mat.width();
        let log_n = log2_strict_usize(mat.height());
        assert!(log_inv_rate <= log_n, "padding exceeds matrix height");
        assert!(log_n <= 1 << F::LOG_BITS, "domain exceeds field dimension");

        // The zero tail starts where the message ends, which fixes the coset size.
        F::lch_transform_cosets(&mut mat.values, width, log_n - log_inv_rate);
        mat
    }
}

/// Transform a row-major buffer of `width` columns in place, over the coset `shift + S_l`.
///
/// The network runs in the level's own representation.
///
/// # Panics
///
/// - Panics if the row count is not a power of two.
/// - Panics if the domain dimension exceeds the bit width of the level.
pub(crate) fn transform<F: ButterflyField, const INVERSE: bool>(
    values: &mut [F],
    width: usize,
    shift: F,
) {
    let log_n = log2_strict_usize(values.len() / width);
    assert!(log_n <= 1 << F::LOG_BITS, "domain exceeds field dimension");
    let twiddles = Twiddles::new(log_n, shift);

    // A matrix within one tile stays in cache throughout, so blocking would only add bookkeeping.
    if size_of_val(values) <= DEEP_TILE_BYTES {
        for step in 0..log_n {
            let j = if INVERSE { step } else { log_n - 1 - step };
            stage_pass::<F, INVERSE>(values, width, j, &twiddles);
        }
        return;
    }

    run::<F, INVERSE>(
        values,
        width,
        log_n,
        &twiddles,
        &Schedule::new::<F>(width, log_n),
    );
}

/// Transform a buffer whose leading `2^log_message` rows hold the message and the rest zeros.
///
/// # Algorithm
///
/// The stages that cross the padding read a zero on the high side of every butterfly:
///
/// ```text
///     (u, 0)  ->  (u + t*0, u + t*0 + 0)  =  (u, u)
/// ```
///
/// So they copy the message into each coset of its own subspace and compute nothing.
///
/// What is left is one shifted transform per coset, over that coset's own rows.
///
/// Coset `c` is the message transformed over `domain_point(c * 2^log_message) + S_log_message`.
///
/// The subspace polynomials are linear, so every coset's blocks read the full network's twiddles.
///
/// The network runs in the level's own representation.
pub(crate) fn transform_cosets<F: ButterflyField>(
    values: &mut [F],
    width: usize,
    log_message: usize,
) {
    let len = width << log_message;

    // A coset within one tile is cheap to copy and transform on its own.
    if size_of::<F>() * len <= DEEP_TILE_BYTES {
        let (message, rest) = values.split_at_mut(len);
        rest.par_chunks_mut(len).enumerate().for_each(|(c, coset)| {
            coset.copy_from_slice(message);
            transform::<F, false>(coset, width, domain_point::<F>((c + 1) << log_message));
        });
        transform::<F, false>(message, width, F::ZERO);
        return;
    }

    // Coset c starts at domain point c * 2^log_message, which is its shift.
    let twiddles: Vec<_> = (0..values.len() / len)
        .map(|c| Twiddles::new(log_message, domain_point::<F>(c << log_message)))
        .collect();
    run_cosets(
        values,
        width,
        log_message,
        &twiddles,
        &Schedule::new::<F>(width, log_message),
    );
}

#[cfg(test)]
mod tests {
    use alloc::vec::Vec;
    use alloc::{format, vec};

    use p3_binary_field::{
        BinaryField8, BinaryField16, BinaryField32, BinaryField64, BinaryField128, Ghash128,
        Poly64, TowerLevel,
    };
    use p3_field::PrimeCharacteristicRing;
    use p3_matrix::Matrix;
    use p3_matrix::dense::RowMajorMatrix;
    use p3_util::log2_strict_usize;
    use proptest::prelude::*;

    use super::schedule::{
        DEEP_TILE_BYTES, MIN_FUSED_STAGES, STAGED_WORKERS, Schedule, run, run_cosets, stage_pass,
        staged_runs,
    };
    use super::{ButterflyField, LchNtt, Twiddles};
    use crate::butterfly::{ghash_transform, ghash_transform_cosets};
    use crate::domain::{domain_point, subspace_polynomial};
    use crate::naive::NaiveAdditiveNtt;
    use crate::traits::AdditiveNtt;

    /// The widths a sweep covers.
    ///
    /// `1 ..= 8` walks a row across the SIMD lane count of every level.
    /// So a stage meets a packed prefix with a scalar tail, a packed prefix, and a tail alone.
    ///
    /// 16 is the folding-block width the encoder commits at, and 64 is the width whose row
    /// passes a cache line at every element size, so a staged row of it is a single row.
    const WIDTHS: [usize; 10] = [1, 2, 3, 4, 5, 6, 7, 8, 16, 64];

    /// The widths a tile-boundary sweep covers.
    ///
    /// A shape's boundary height is `2^t`, where `2^t * row_bytes` is one tile budget.
    /// So the matrix a boundary sweep builds is four budgets whatever the width, and the cost
    /// is one budget per width rather than one per element.
    ///
    /// This subset keeps the two rows the heavy sweep leaves below its boundary - one
    /// element and two of them - plus an odd row, a SIMD-sized row and a wide one.
    const BOUNDARY_WIDTHS: [usize; 5] = [1, 2, 3, 16, 64];

    /// The heights every sweep covers.
    ///
    /// At production budgets only the wide end reaches the blocked schedule, and no staged run
    /// executes. The staged path is pinned by the boundary and production sweeps instead, at
    /// named worker counts.
    const LOG_HEIGHTS: core::ops::RangeInclusive<usize> = 0..=10;

    /// The heights the heavy sweep covers on top of the ordinary sweep.
    ///
    /// By `2^14` every row wider than 8 bytes has passed one tile budget; the three narrower
    /// shapes block only in the boundary sweep. At width 64 this is minutes of oracle in a
    /// debug build, so the sweep is ignored.
    const HEAVY_LOG_HEIGHTS: core::ops::RangeInclusive<usize> = 0..=14;

    /// The shifts a sweep covers.
    ///
    /// Zero is the subspace itself, where the first block of every stage has a zero twiddle and
    /// takes the butterfly's separate case. The other shift has bits throughout, so no stage's
    /// twiddle is accidentally zero.
    const SHIFTS: [u64; 2] = [0, 0x5555_1234_9abc_def0];

    /// Builds an element of any level from a 64-bit pattern, repeating it for wider levels.
    fn sample<F: ButterflyField>(bits: u64) -> F {
        F::from_le_byte_iter(bits.to_le_bytes().into_iter().cycle())
    }

    /// Builds a matrix whose entries are distinct functions of the seed and the position.
    fn matrix<F: ButterflyField>(log_n: usize, width: usize, seed: u64) -> RowMajorMatrix<F> {
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
    fn twiddle_walk_ntt<F: ButterflyField>(
        mut mat: RowMajorMatrix<F>,
        shift: F,
    ) -> RowMajorMatrix<F> {
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

    /// The inverse transform walked the same way, every twiddle from its own block index.
    /// One serial pass, with the stage order reversed and the butterfly undone.
    fn twiddle_walk_intt<F: ButterflyField>(
        mut mat: RowMajorMatrix<F>,
        shift: F,
    ) -> RowMajorMatrix<F> {
        let width = mat.width();
        let log_n = log2_strict_usize(mat.height());
        for j in 0..log_n {
            let half = (1 << j) * width;
            let base = subspace_polynomial::<F>(j, shift);
            for (blk, block) in mat.values.chunks_mut(half << 1).enumerate() {
                let t = base + domain_point::<F>(blk << 1);
                let (lo, hi) = block.split_at_mut(half);
                for (u, v) in lo.iter_mut().zip(hi) {
                    *v += *u;
                    *u += t * *v;
                }
            }
        }
        mat
    }

    /// One stage at a time over the whole matrix, which is the schedule blocking has to match.
    fn stage_by_stage<F: ButterflyField, const INVERSE: bool>(
        mat: &mut RowMajorMatrix<F>,
        twiddles: &Twiddles<F>,
    ) {
        let width = mat.width();
        let log_n = log2_strict_usize(mat.height());
        for step in 0..log_n {
            let j = if INVERSE { step } else { log_n - 1 - step };
            stage_pass::<F, INVERSE>(&mut mat.values, width, j, twiddles);
        }
    }

    /// The blocked schedule and the plain one agree element for element, in both directions.
    fn check_schedules_agree<F: ButterflyField>(log_n: usize, width: usize, schedule: &Schedule) {
        let coeffs = matrix::<F>(log_n, width, 3);
        for shift_bits in SHIFTS {
            let shift = sample::<F>(shift_bits);
            let twiddles = Twiddles::new(log_n, shift);
            let label = format!("log_n={log_n} width={width} shift={shift_bits:#x}");

            // Forward: the reference runs stage by stage over the whole matrix.
            let mut expected = coeffs.clone();
            stage_by_stage::<F, false>(&mut expected, &twiddles);

            let mut blocked = coeffs.clone();
            run::<F, false>(&mut blocked.values, width, log_n, &twiddles, schedule);
            assert_eq!(blocked, expected, "forward {label}");

            // Inverse: same comparison with the stage order reversed.
            let mut expected = coeffs.clone();
            stage_by_stage::<F, true>(&mut expected, &twiddles);

            let mut blocked = coeffs.clone();
            run::<F, true>(&mut blocked.values, width, log_n, &twiddles, schedule);
            assert_eq!(blocked, expected, "inverse {label}");
        }
    }

    /// `LchNtt` agrees with the oracle on a random matrix and a random coset.
    fn check_matches_naive<F: ButterflyField>(log_n: usize, width: usize, seed: u64, shift: u64) {
        let coeffs = matrix::<F>(log_n, width, seed);
        let shift = sample::<F>(shift);

        let fast = LchNtt::<F>::default().shifted_ntt_batch(coeffs.clone(), shift);
        let slow = NaiveAdditiveNtt::<F>::default().shifted_ntt_batch(coeffs, shift);
        assert_eq!(fast, slow);
    }

    /// The twiddle a block is handed, walked from the block before it and checked against the
    /// twiddle that block's own index defines.
    ///
    /// Every schedule below carries a twiddle forward across blocks instead of computing it.
    /// An error in the increment table is therefore a drift that starts at one block index.
    /// So the starts below seed the walk at several block indices per stage, not at zero alone.
    fn check_the_twiddle_walk<F: ButterflyField>(log_n: usize, shift_bits: u64) {
        let shift = sample::<F>(shift_bits);
        let twiddles = Twiddles::new(log_n, shift);

        for stage in 0..log_n {
            // Stage `stage` pairs rows `2^stage` apart, so a block spans `2^(stage + 1)` rows.
            let blocks = 1usize << (log_n - 1 - stage);
            for start in [0, 1, 3, 63, blocks.saturating_sub(5)] {
                if start >= blocks {
                    continue;
                }
                let mut t = twiddles.at(stage, start);
                for block in start..blocks.min(start + 5) {
                    // The independent formula: `W_stage(shift)` plus the block's domain point.
                    let expected =
                        subspace_polynomial::<F>(stage, shift) + domain_point::<F>(block << 1);
                    assert_eq!(t, expected, "stage={stage} block={block}");
                    if block + 1 < blocks {
                        t += twiddles.step(block + 1);
                    }
                }
            }
        }
    }

    #[test]
    fn incremental_twiddles_match_independent_domain_points() {
        for shift_bits in SHIFTS {
            check_the_twiddle_walk::<BinaryField32>(16, shift_bits);
            check_the_twiddle_walk::<BinaryField64>(16, shift_bits);
            check_the_twiddle_walk::<BinaryField128>(16, shift_bits);
            check_the_twiddle_walk::<Ghash128>(16, shift_bits);
        }
    }

    /// The transform agrees with the serial walk, element for element, in both directions.
    fn check_walk_agrees<F: ButterflyField>(log_n: usize, width: usize, shift_bits: u64) {
        let coeffs = matrix::<F>(log_n, width, 41);
        let shift = sample::<F>(shift_bits);
        let ntt = LchNtt::<F>::default();
        let label = format!("log_n={log_n} width={width} shift={shift_bits:#x}");

        assert_eq!(
            ntt.shifted_ntt_batch(coeffs.clone(), shift),
            twiddle_walk_ntt::<F>(coeffs.clone(), shift),
            "ntt {label}"
        );
        assert_eq!(
            ntt.shifted_intt_batch(coeffs.clone(), shift),
            twiddle_walk_intt::<F>(coeffs, shift),
            "intt {label}"
        );
    }

    /// The blocked schedule of a named worker count, against the same independent walk.
    ///
    /// The transform picks its own schedule from the shape and the thread count, so what it
    /// covers moves with the machine. This drives `run` directly instead: the walk then pins the
    /// blocked path whatever the ambient thread count is.
    fn check_the_blocked_walk_agrees<F: ButterflyField>(
        log_n: usize,
        width: usize,
        shift_bits: u64,
        workers: &[usize],
    ) {
        let coeffs = matrix::<F>(log_n, width, 41);
        let shift = sample::<F>(shift_bits);
        let twiddles = Twiddles::new(log_n, shift);
        let walked = twiddle_walk_ntt::<F>(coeffs.clone(), shift);

        for &workers in workers {
            let schedule = Schedule::for_workers::<F>(width, log_n, workers);
            let label =
                format!("log_n={log_n} width={width} shift={shift_bits:#x} workers={workers}");

            let mut blocked = coeffs.clone();
            run::<F, false>(&mut blocked.values, width, log_n, &twiddles, &schedule);
            assert_eq!(blocked, walked, "ntt {label}");

            // Undoing the walk's own codeword holds the inverse schedule to the same twiddles.
            let mut blocked = walked.clone();
            run::<F, true>(&mut blocked.values, width, log_n, &twiddles, &schedule);
            assert_eq!(blocked, coeffs, "intt {label}");
        }
    }

    /// `LchNtt` agrees with the oracle in both directions, on the subspace and on a coset.
    fn check_oracle_agrees<F: ButterflyField>(log_n: usize, width: usize, shift_bits: u64) {
        let coeffs = matrix::<F>(log_n, width, 53);
        let shift = sample::<F>(shift_bits);
        let fast = LchNtt::<F>::default();
        let slow = NaiveAdditiveNtt::<F>::default();
        let label = format!("log_n={log_n} width={width} shift={shift_bits:#x}");

        assert_eq!(
            fast.shifted_ntt_batch(coeffs.clone(), shift),
            slow.shifted_ntt_batch(coeffs.clone(), shift),
            "ntt {label}"
        );
        assert_eq!(
            fast.shifted_intt_batch(coeffs.clone(), shift),
            slow.shifted_intt_batch(coeffs, shift),
            "intt {label}"
        );
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

        /// `LchNtt`'s low-degree extension agrees with the oracle's, on a coset too, and the
        /// input rows reappear as the prefix: the correspondence Phase 3 folds along.
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

    proptest! {
        // A blocked height is a whole matrix per case, so this block draws fewer of them.
        #![proptest_config(ProptestConfig::with_cases(8))]

        /// The two directions invert each other at heights the schedule blocks.
        #[test]
        fn lch_round_trips_past_the_blocking_threshold(
            log_n in 11usize..=14,
            width in 1usize..=4,
            seed in any::<u64>(),
            shift in any::<u64>(),
        ) {
            // Fixture state: the two extreme element sizes, hence the two extreme tile shapes.
            // A 4-byte element gets the deepest contiguous tile of the four levels and a
            // 16-byte one the shallowest, so between them the drawn heights land on either side
            // of a boundary.
            let coeffs = matrix::<BinaryField32>(log_n, width, seed);
            let ntt = LchNtt::<BinaryField32>::default();
            let shift32 = sample::<BinaryField32>(shift);
            let evals = ntt.shifted_ntt_batch(coeffs.clone(), shift32);
            prop_assert_eq!(ntt.shifted_intt_batch(evals, shift32), coeffs);

            let coeffs = matrix::<Ghash128>(log_n, width, seed);
            let ntt = LchNtt::<Ghash128>::default();
            let shift128 = sample::<Ghash128>(shift);
            let evals = ntt.shifted_ntt_batch(coeffs.clone(), shift128);
            prop_assert_eq!(ntt.shifted_intt_batch(evals, shift128), coeffs);
        }
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(32))]

        /// Random padded shapes, against running the skipped layers as ordinary butterflies.
        #[test]
        fn random_padded_shapes_match_the_full_transform(
            log_message in 0usize..=7,
            width in 1usize..=5,
            log_inv_rate in 0usize..=3,
            seed in any::<u64>(),
        ) {
            let message = matrix::<BinaryField128>(log_message, width, seed);
            let mut padded = message.values;
            padded.resize(padded.len() << log_inv_rate, BinaryField128::ZERO);
            let padded = RowMajorMatrix::new(padded, width);

            let ntt = LchNtt::<BinaryField128>::default();
            let expected = ntt.ntt_batch(padded.clone());
            prop_assert_eq!(ntt.ntt_batch_padded(padded, log_inv_rate), expected);
        }
    }

    #[test]
    fn lch_matches_a_twiddle_walk_across_several_tasks() {
        // Fixture state: a height whose stages take more than one butterfly task, so a task
        // seeds its twiddle at a block index of its own rather than at zero.
        //
        // The oracle tests all sit below this height, and `lch_round_trips` is blind to the
        // schedule: both directions read the same twiddles, so they invert each other whatever
        // those twiddles are. Only a comparison against an independent walk pins them.
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
                // The inverse has its own copy of the schedule.
                //
                // Undoing a codeword the walk produced holds that copy to the same twiddles.
                assert_eq!(
                    ntt.shifted_intt_batch(walked, shift),
                    coeffs,
                    "intt width={width} shift={shift_bits:#x}"
                );
            }
        }
    }

    #[test]
    fn blocked_schedule_matches_the_plain_one() {
        // Invariant: the blocked schedule is a pure reordering of the plain one, so it
        // reproduces it bit for bit wherever the tile boundaries fall differently. A round trip
        // cannot see this: both directions read the same twiddle, so they invert each other
        // whatever index that twiddle came from.
        //
        // The tile shapes here are far smaller than a cache, which is what lets a test-sized
        // matrix reach the branches a production-sized one does. They also reach `run` whole:
        // the worker count reshapes a schedule inside `Schedule::for_workers`, so these
        // fixtures run the same stages at any thread count.

        // Fixture state: rows per contiguous tile, staged rows per group, rows per staged row.
        let shape = |tile: usize, staged: usize, slab: usize| Schedule {
            log_tile_rows: tile,
            log_staged_rows: staged,
            log_slab_rows: slab,
        };

        for width in WIDTHS {
            // A three-stage tile with two-stage groups above it.
            //
            // Heights 2 and 3 stay inside a single tile; 4 leaves one stage over it, which runs
            // as a plain pass; 5 is exactly one fused group.
            //
            // 6 is a fused group and a plain pass, 7 two fused groups, and 8 two of them with a
            // plain pass on top.
            for log_n in [2usize, 3, 4, 5, 6, 7, 8] {
                let schedule = shape(3, 2, 0);
                check_schedules_agree::<BinaryField32>(log_n, width, &schedule);
                check_schedules_agree::<BinaryField64>(log_n, width, &schedule);
                check_schedules_agree::<BinaryField128>(log_n, width, &schedule);
                check_schedules_agree::<Ghash128>(log_n, width, &schedule);
            }

            // Groups deeper than the stages left over them, so the top group is clipped: at 7 to
            // a single stage, which is a plain pass, and at 8 to a fused group of two.
            check_schedules_agree::<BinaryField128>(7, width, &shape(2, 4, 0));
            check_schedules_agree::<BinaryField128>(8, width, &shape(2, 4, 0));
            // A single-stage group everywhere, which is the plain pass under another name.
            check_schedules_agree::<BinaryField128>(7, width, &shape(2, 1, 0));
            // No tile at all, so every stage belongs to a group.
            check_schedules_agree::<BinaryField128>(6, width, &shape(0, 3, 0));
            // Staged rows holding a run of matrix rows each, as narrow rows call for.
            check_schedules_agree::<BinaryField32>(8, width, &shape(4, 3, 2));
            check_schedules_agree::<BinaryField32>(9, width, &shape(3, 2, 1));
        }
    }

    #[test]
    fn production_schedule_matches_the_plain_one() {
        // Fixture state: the production tile shapes, past the height where blocking switches
        // on. The shapes come from cache budgets, so the heights that clear them differ by
        // shape, and each pairing below leaves at least one group above the contiguous tile.
        //
        // The worker count is named rather than read: a lone worker takes the deepest tile the
        // budget allows, and 32 workers the shallowest the clamp leaves.
        for workers in [1usize, 32] {
            for (log_n, width) in [(14usize, 3usize), (15, 3), (13, 16)] {
                let schedule = Schedule::for_workers::<BinaryField32>(width, log_n, workers);
                assert!(schedule.log_tile_rows < log_n, "no group above the tile");
                check_schedules_agree::<BinaryField32>(log_n, width, &schedule);
            }
            for (log_n, width) in [(11usize, 16usize), (12, 16), (13, 5)] {
                let schedule = Schedule::for_workers::<BinaryField128>(width, log_n, workers);
                assert!(schedule.log_tile_rows < log_n, "no group above the tile");
                check_schedules_agree::<BinaryField128>(log_n, width, &schedule);
                check_schedules_agree::<Ghash128>(log_n, width, &schedule);
            }
        }
    }

    /// One padded shape: skipping the layers that cross the padding against running them.
    fn check_the_padded_transform_agrees<F: ButterflyField>(
        log_message: usize,
        width: usize,
        log_inv_rate: usize,
    ) {
        let message = matrix::<F>(log_message, width, 59);

        // The reference pads the coefficients and transforms every layer of the result.
        let mut padded = message.values;
        padded.resize(padded.len() << log_inv_rate, F::ZERO);
        let padded = RowMajorMatrix::new(padded, width);

        let ntt = LchNtt::<F>::default();
        let expected = ntt.ntt_batch(padded.clone());
        assert_eq!(
            ntt.ntt_batch_padded(padded, log_inv_rate),
            expected,
            "log_message={log_message} width={width} rate={log_inv_rate}"
        );
    }

    #[test]
    fn the_padded_transform_matches_the_full_one() {
        // Invariant: a layer whose high side is all zero copies, so skipping it changes nothing.
        //
        //     rate 0   one coset, and the skip has no layer to skip
        //     rate 3   eight cosets, seven of them a copy of the first
        for width in WIDTHS {
            for log_message in 0..=5 {
                for log_inv_rate in 0..=3 {
                    check_the_padded_transform_agrees::<BinaryField32>(
                        log_message,
                        width,
                        log_inv_rate,
                    );
                    check_the_padded_transform_agrees::<BinaryField128>(
                        log_message,
                        width,
                        log_inv_rate,
                    );
                    check_the_padded_transform_agrees::<Ghash128>(log_message, width, log_inv_rate);
                }
            }
        }
    }

    /// Every coset of one padded shape through a named schedule, against the serial walk of the whole codeword.
    fn check_the_shared_cosets_agree<F: ButterflyField>(
        log_message: usize,
        width: usize,
        log_inv_rate: usize,
        schedule: &Schedule,
    ) {
        // The reference pads the message and walks every stage of the whole codeword.
        let message = matrix::<F>(log_message, width, 67);
        let mut padded = message.values;
        padded.resize(padded.len() << log_inv_rate, F::ZERO);
        let expected = twiddle_walk_ntt::<F>(RowMajorMatrix::new(padded.clone(), width), F::ZERO);

        // Coset c is the message transformed over the domain point it starts at.
        let twiddles: Vec<_> = (0..1usize << log_inv_rate)
            .map(|c| Twiddles::new(log_message, domain_point::<F>(c << log_message)))
            .collect();
        run_cosets(&mut padded, width, log_message, &twiddles, schedule);
        assert_eq!(
            padded, expected.values,
            "log_message={log_message} width={width} rate={log_inv_rate} {schedule:?}"
        );
    }

    #[test]
    fn cosets_sharing_their_first_pass_match_the_full_walk() {
        // Invariant: one gather of the message feeds every coset's first pass.
        // Each coset then finishes alone, and the result is the codeword a full transform gives.
        //
        // Fixture state: rows per contiguous tile, staged rows per group, rows per staged row.
        //
        //     tile 3, staged 2    groups above the tile, so the top group is the shared pass
        //     tile 8, staged 2    the tile covers every stage, so the tile pass is the shared one
        //     tile 0, staged 3    no tile at all, every stage in a group
        //     tile 2, staged 0    groups of one stage, a plain pass shared through a staging tile
        //     tile 4, staged 3, slab 2   staged rows holding runs of matrix rows
        let shape = |tile: usize, staged: usize, slab: usize| Schedule {
            log_tile_rows: tile,
            log_staged_rows: staged,
            log_slab_rows: slab,
        };
        let shapes = [
            shape(3, 2, 0),
            shape(8, 2, 0),
            shape(0, 3, 0),
            shape(2, 0, 0),
            shape(4, 3, 2),
        ];
        for schedule in &shapes {
            for width in [1usize, 3, 16] {
                for log_message in 0..=7 {
                    for log_inv_rate in 0..=3 {
                        check_the_shared_cosets_agree::<BinaryField32>(
                            log_message,
                            width,
                            log_inv_rate,
                            schedule,
                        );
                        check_the_shared_cosets_agree::<Poly64>(
                            log_message,
                            width,
                            log_inv_rate,
                            schedule,
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn a_padded_transform_past_one_tile_matches_the_full_one() {
        // Invariant: a coset larger than one tile takes the shared-pass route, at production budgets.
        //
        // Fixture state: each message below is 256 KiB, twice the tile budget.
        //
        //     BinaryField32   2^16 rows of 1 column
        //     Poly64          2^12 rows of 8 columns
        //     BinaryField128  2^10 rows of 16 columns
        for log_inv_rate in 0..=2 {
            check_the_padded_transform_agrees::<BinaryField32>(16, 1, log_inv_rate);
            check_the_padded_transform_agrees::<Poly64>(12, 8, log_inv_rate);
            check_the_padded_transform_agrees::<BinaryField128>(10, 16, log_inv_rate);
        }

        // The same shapes at named worker counts, since the pool size moves the tile and the groups.
        for workers in [1usize, 32] {
            for log_inv_rate in 0..=2 {
                check_the_shared_cosets_agree::<Poly64>(
                    12,
                    8,
                    log_inv_rate,
                    &Schedule::for_workers::<Poly64>(8, 12, workers),
                );
            }
        }
    }

    #[test]
    fn the_padded_transform_matches_the_reference_oracle() {
        // The oracle evaluates the novel basis straight from its product definition.
        // So it pins the coset split to that basis, not to another split of the same network.
        for width in [1usize, 3] {
            for log_message in 0..=4 {
                for log_inv_rate in 0..=2 {
                    let message = matrix::<BinaryField64>(log_message, width, 61);
                    let mut padded = message.values;
                    padded.resize(padded.len() << log_inv_rate, BinaryField64::ZERO);
                    let padded = RowMajorMatrix::new(padded, width);

                    let expected =
                        NaiveAdditiveNtt::<BinaryField64>::default().ntt_batch(padded.clone());
                    let actual =
                        LchNtt::<BinaryField64>::default().ntt_batch_padded(padded, log_inv_rate);
                    let label = format!("log_message={log_message} width={width}");
                    assert_eq!(actual, expected, "{label} rate={log_inv_rate}");
                }
            }
        }
    }

    #[test]
    #[should_panic = "padding exceeds matrix height"]
    fn the_padded_transform_rejects_padding_past_the_height() {
        // A 2^2-row matrix cannot have been padded from a fraction of a row.
        let mat = matrix::<BinaryField32>(2, 4, 0);
        let _ = LchNtt::<BinaryField32>::default().ntt_batch_padded(mat, 3);
    }

    #[test]
    #[should_panic = "domain exceeds field dimension"]
    fn the_padded_transform_rejects_a_domain_past_the_field_dimension() {
        // A 2^9-row domain asks for nine Cantor basis vectors, and a byte level has eight.
        let mat = matrix::<BinaryField8>(9, 1, 0);
        let _ = LchNtt::<BinaryField8>::default().ntt_batch_padded(mat, 1);
    }

    #[test]
    fn the_contiguous_tile_leaves_one_for_every_worker() {
        // Invariant: a contiguous tile is a task, so the schedule keeps one per worker wherever
        // the height allows, and is otherwise as deep as the byte budget permits.
        //
        // No comparison against the plain schedule can see this: the clamp only moves where a
        // stage runs, never what it computes. Replacing it with `.min(log_n)` leaves the rest of
        // this module passing at any thread count.
        for workers in [1usize, 2, 3, 4, 14, 16, 32, 4096] {
            for log_n in 0..=24usize {
                for width in [1usize, 3, 16, 64, 1 << 14] {
                    let tile =
                        Schedule::for_workers::<BinaryField32>(width, log_n, workers).log_tile_rows;
                    assert!(tile <= log_n, "a tile taller than the matrix");
                    let tiles = 1usize << (log_n - tile);
                    assert!(
                        tiles >= workers.min(1 << log_n),
                        "workers={workers} log_n={log_n} width={width}: {tiles} tiles"
                    );

                    // And no shallower than it has to be: one more row per tile would either
                    // halve the tile count below the worker count or overrun the budget.
                    let row_bytes = core::mem::size_of::<BinaryField32>() * width;
                    let deeper_fits = (row_bytes << (tile + 1)) <= DEEP_TILE_BYTES;
                    assert!(
                        tile == log_n || !deeper_fits || tiles / 2 < workers,
                        "workers={workers} log_n={log_n} width={width}: tile {tile} is short"
                    );
                }
            }
        }

        // One shape where the clamp bites: a 4-byte row of one element fills a tile at 2^15
        // rows, so a lone worker takes all of a 2^16-row matrix in two tiles, and 32 workers
        // give up four stages of tile depth to have one each.
        assert_eq!(
            Schedule::for_workers::<BinaryField32>(1, 16, 1).log_tile_rows,
            15
        );
        assert_eq!(
            Schedule::for_workers::<BinaryField32>(1, 16, 32).log_tile_rows,
            11
        );
    }

    #[test]
    fn staging_waits_for_enough_workers() {
        // Invariant: below `STAGED_WORKERS` the schedule leaves every stage above the contiguous
        // tile a plain pass, however much staging the byte budget would have paid for.
        //
        // The tile itself is unaffected at these counts, so they differ in the groups alone.
        for workers in [1usize, STAGED_WORKERS - 1, STAGED_WORKERS] {
            let schedule = Schedule::for_workers::<Ghash128>(16, 20, workers);
            assert_eq!(
                schedule.log_tile_rows, 9,
                "a 256-byte row fills the tile budget at 2^9 rows"
            );
            assert_eq!(
                schedule.log_staged_rows >= MIN_FUSED_STAGES,
                workers >= STAGED_WORKERS,
                "workers={workers}"
            );
        }
    }

    #[test]
    fn a_staging_group_stages_every_row_exactly_once() {
        // Invariant: the map a staging task walks is a bijection onto the matrix rows, and the
        // butterfly block it hands the tile pass is the block those rows really sit in.
        //
        // Both hold at runtime under `debug_assert` alone, and the second only indirectly. So
        // the model below walks every shape the schedule can produce, against the definitions
        // rather than against another copy of the map.
        let mut shapes = 0;
        for log_n in 1..=10usize {
            for top in 0..log_n {
                // A group is at least one stage, and a staged run may not straddle the butterfly
                // of its narrowest stage: `log_slab + depth <= top + 1`.
                for depth in 1..=top + 1 {
                    for log_slab in 0..=top + 1 - depth {
                        shapes += 1;
                        let map = staged_runs(1, top, depth, log_slab);
                        let mut seen = vec![false; 1 << log_n];
                        for task in 0..1usize << (log_n - depth - log_slab) {
                            for k in 0..1usize << depth {
                                let first = map.run_index(task, k) << log_slab;

                                // A staged row is the run of `2^log_slab` rows from there.
                                for row in first..first + (1 << log_slab) {
                                    let hit =
                                        seen.get_mut(row).expect("the walk leaves the matrix");
                                    assert!(!*hit, "row {row} staged twice");
                                    *hit = true;
                                }

                                // Stage `top - s` puts a row in block `row >> (top - s + 1)`.
                                //
                                // `tile_stages` is told `map.block(task)` and adds the block
                                // index a tile of `2^depth` rows walks on its own.
                                for s in 0..depth {
                                    assert_eq!(
                                        first >> (top - s + 1),
                                        (map.block(task) << s) + (k >> (depth - s)),
                                        "log_n={log_n} top={top} depth={depth} \
                                         slab={log_slab} task={task} k={k} s={s}"
                                    );
                                }
                            }
                        }
                        assert!(seen.iter().all(|&hit| hit), "a row was left unstaged");
                    }
                }
            }
        }
        assert_eq!(shapes, 715, "the model stopped covering every shape");
    }

    /// The transform against an independent serial walk, over a range of heights.
    /// Every width and both shifts, in both directions.
    ///
    /// The walk shares no code with the transform below the field arithmetic.
    /// It recomputes `W_j(shift)` from the recurrence and every block twiddle from its index.
    /// So it pins the blocked schedule's twiddles and not merely its memory order.
    fn sweep_against_the_walk<F: ButterflyField>(heights: core::ops::RangeInclusive<usize>) {
        for width in WIDTHS {
            for log_n in heights.clone() {
                for shift_bits in SHIFTS {
                    check_walk_agrees::<F>(log_n, width, shift_bits);
                }
            }
        }
    }

    #[test]
    fn the_schedule_matches_an_independent_walk_at_32_bits() {
        sweep_against_the_walk::<BinaryField32>(LOG_HEIGHTS);
    }

    #[test]
    fn the_schedule_matches_an_independent_walk_at_64_bits() {
        sweep_against_the_walk::<BinaryField64>(LOG_HEIGHTS);
    }

    #[test]
    fn the_schedule_matches_an_independent_walk_at_128_bits() {
        sweep_against_the_walk::<BinaryField128>(LOG_HEIGHTS);
    }

    #[test]
    fn the_schedule_matches_an_independent_walk_in_the_ghash_basis() {
        sweep_against_the_walk::<Ghash128>(LOG_HEIGHTS);
    }

    #[test]
    fn the_schedule_matches_an_independent_walk_in_the_64_bit_polynomial_basis() {
        sweep_against_the_walk::<Poly64>(LOG_HEIGHTS);
    }

    #[test]
    #[ignore = "serial oracle over every width and level up to 2^14; run from heavy CI"]
    fn the_schedule_matches_an_independent_walk_at_every_blocked_shape() {
        // Invariant: the same sweep, carried to the first height that blocks at every shape.
        //
        // Narrow rows included, so no tile boundary of any level falls between two heights.
        sweep_against_the_walk::<BinaryField32>(HEAVY_LOG_HEIGHTS);
        sweep_against_the_walk::<BinaryField64>(HEAVY_LOG_HEIGHTS);
        sweep_against_the_walk::<BinaryField128>(HEAVY_LOG_HEIGHTS);
        sweep_against_the_walk::<Ghash128>(HEAVY_LOG_HEIGHTS);
    }

    /// Every width, at the three heights around its contiguous tile boundary: one budget
    /// exactly, one stage above it, and two stages above it, which a staging tile fuses.
    ///
    /// The boundary moves with the element size and the width, so it is read back out of the
    /// schedule rather than written down.
    fn sweep_across_the_tile_boundaries<F: ButterflyField>() {
        for width in BOUNDARY_WIDTHS {
            // Neither the height nor the worker count binds at a height no level can reach, so
            // this reads the raw budget.
            let tile = Schedule::for_workers::<F>(width, usize::BITS as usize, 1).log_tile_rows;
            for log_n in [tile, tile + 1, tile + 2] {
                for shift_bits in SHIFTS {
                    check_walk_agrees::<F>(log_n, width, shift_bits);
                }
            }

            // The height above the boundary again, on the blocked schedule of a named worker
            // count, since the one `transform` picks depends on the machine running the test.
            for shift_bits in SHIFTS {
                check_the_blocked_walk_agrees::<F>(tile + 2, width, shift_bits, &[1, 32]);
            }
        }
    }

    #[test]
    fn every_tile_boundary_is_crossed_at_32_bits() {
        sweep_across_the_tile_boundaries::<BinaryField32>();
    }

    #[test]
    fn every_tile_boundary_is_crossed_at_64_bits() {
        sweep_across_the_tile_boundaries::<BinaryField64>();
    }

    #[test]
    fn every_tile_boundary_is_crossed_at_128_bits() {
        sweep_across_the_tile_boundaries::<BinaryField128>();
    }

    #[test]
    fn every_tile_boundary_is_crossed_in_the_ghash_basis() {
        sweep_across_the_tile_boundaries::<Ghash128>();
    }

    #[test]
    fn every_tile_boundary_is_crossed_in_the_64_bit_polynomial_basis() {
        sweep_across_the_tile_boundaries::<Poly64>();
    }

    /// The narrow shapes whose staging tile stages a run of rows rather than a single row.
    ///
    /// A row shorter than a cache line turns one staged row into a run of matrix rows. The
    /// boundary sweep above already compares these shapes element for element, so this pins
    /// only that they really take the run branch.
    fn check_a_staged_run_of_rows<F: ButterflyField>(log_n: usize) {
        // The heights above come from the deepest tile the budget allows, which is the one a
        // lone worker takes.
        let alone = Schedule::for_workers::<F>(1, log_n, 1);
        assert_eq!(alone.log_tile_rows + MIN_FUSED_STAGES, log_n);
        assert!(alone.log_slab_rows > 0, "the staged row is a single row");

        // The clamp then moves the tile but not the run, so the group is still staged in runs.
        let shared = Schedule::for_workers::<F>(1, log_n, STAGED_WORKERS);
        assert!(
            shared.log_staged_rows >= MIN_FUSED_STAGES,
            "nothing is staged"
        );
        assert!(shared.log_slab_rows > 0, "the staged row is a single row");
    }

    #[test]
    fn narrow_rows_are_staged_in_runs() {
        check_a_staged_run_of_rows::<BinaryField32>(17);
        check_a_staged_run_of_rows::<BinaryField64>(16);
        check_a_staged_run_of_rows::<BinaryField128>(15);
        check_a_staged_run_of_rows::<Ghash128>(15);
    }

    #[test]
    fn the_transform_matches_the_reference_oracle_at_every_width() {
        // Invariant: the transform is the same map as the reference oracle, at every width.
        //
        // The oracle evaluates `sum_i d_i * X_i(x)` straight from the product definition and
        // depends on none of D8's identities, so it pins the transform to the novel basis
        // rather than to another consistent network.
        //
        // It costs `O(n^2 * width)`, which is what caps the heights below; the walk sweeps carry
        // the taller shapes, and together they reach every branch.
        for width in WIDTHS {
            for log_n in 0..=6 {
                for shift_bits in SHIFTS {
                    check_oracle_agrees::<BinaryField32>(log_n, width, shift_bits);
                    check_oracle_agrees::<BinaryField64>(log_n, width, shift_bits);
                    check_oracle_agrees::<BinaryField128>(log_n, width, shift_bits);
                    check_oracle_agrees::<Ghash128>(log_n, width, shift_bits);
                    check_oracle_agrees::<Poly64>(log_n, width, shift_bits);
                }
            }
        }
    }

    #[test]
    fn a_row_wider_than_the_budgets_degrades_to_plain_passes() {
        // Invariant: a row wider than both byte budgets leaves no room to block.
        //
        // The schedule degrades to plain passes, rather than to a tile of no rows.
        const WIDTH: usize = 16384;
        let schedule = Schedule::for_workers::<BinaryField128>(WIDTH, 4, 32);
        assert_eq!(schedule.log_tile_rows, 0);
        assert_eq!(schedule.log_slab_rows, 0);
        assert_eq!(schedule.log_staged_rows, 0);
        check_schedules_agree::<BinaryField128>(2, WIDTH, &schedule);
    }

    #[test]
    #[should_panic]
    fn shifted_ntt_batch_rejects_l_past_the_bit_width() {
        // An index of `S_l` past the bit width of `F` asks for a Cantor basis vector.
        //
        // This level does not have one.
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

        // The widest level may run its own transform in the GHASH basis, so the tower side calls the network directly.
        let mut tower = tower_coeffs.values;
        super::transform::<BinaryField128, false>(&mut tower, WIDTH, shift);
        let ghash =
            LchNtt::<Ghash128>::default().shifted_ntt_batch(ghash_coeffs, Ghash128::from(shift));

        // Converting before or after the transform must give the same values.
        for (t, g) in tower.iter().zip(&ghash.values) {
            assert_eq!(Ghash128::from(*t), *g);
        }
    }

    /// Every twiddle of a transform lies in the smallest tower subfield that holds its shift and its domain.
    ///
    /// The widest level picks the basis its network runs in from that subfield, without walking the twiddles.
    #[test]
    fn every_twiddle_lies_in_the_subfield_of_its_shift_and_domain() {
        let bit_len = |x: u128| (u128::BITS - x.leading_zeros()) as usize;

        // The tower subfield of `2^t` bits holds exactly the elements below `2^(2^t)`.
        let subfield_bits = |log_n: usize, shift: BinaryField128| {
            log_n.max(bit_len(shift.to_repr())).next_power_of_two()
        };

        // Every block twiddle of every stage, as the network seeds it.
        let check = |log_n: usize, shift: BinaryField128, bits: usize| {
            let twiddles = Twiddles::new(log_n, shift);
            for stage in 0..log_n {
                for block in 0..1usize << (log_n - 1 - stage) {
                    let t = twiddles.at(stage, block).to_repr();
                    assert!(
                        bit_len(t) <= bits,
                        "log_n={log_n} shift={:#x} stage={stage} block={block}: {t:#x} past {bits} bits",
                        shift.to_repr()
                    );
                }
            }
        };

        // Fixture state: no shift, a domain point within a nibble, a 17-bit shift, and one with bits throughout.
        let shifts = [
            BinaryField128::ZERO,
            domain_point::<BinaryField128>(5),
            BinaryField128::from_repr(0x1_2345),
            sample::<BinaryField128>(SHIFTS[1]),
        ];
        for log_n in 0..=10 {
            for shift in shifts {
                check(log_n, shift, subfield_bits(log_n, shift));
            }

            // The cosets of a padded transform start at domain points below the height, so the height bounds them.
            let bits = subfield_bits(log_n, BinaryField128::ZERO);
            for log_message in 0..=log_n {
                for c in 0..1usize << (log_n - log_message) {
                    check(log_message, domain_point(c << log_message), bits);
                }
            }
        }
    }

    /// The widest level's own transform, and its run over `Ghash128`, against its network held in the tower basis.
    ///
    /// The level takes the `Ghash128` run only for a wide twiddle on a build with a carryless multiply.
    /// So the run is also called directly, which holds it to the tower result at every size.
    #[test]
    fn the_widest_level_transforms_as_its_tower_basis_network() {
        type Transform = fn(&mut [BinaryField128], usize, BinaryField128);
        let routes: [(&str, Transform, Transform); 2] = [
            (
                "level",
                BinaryField128::lch_transform::<false>,
                BinaryField128::lch_transform::<true>,
            ),
            ("ghash", ghash_transform::<false>, ghash_transform::<true>),
        ];

        // Fixture state: one row up to sixteen rows, then 2^11 and 2^14 rows.
        //
        // - Each tall height has a shape past one tile, so the blocked schedule runs.
        // - Each also has a shape past the length at which a pool of up to 64 workers splits a change of basis.
        for log_n in [0, 1, 2, 3, 4, 5, 11, 14] {
            for width in [1usize, 3, 16] {
                for shift_bits in SHIFTS {
                    let shift = sample::<BinaryField128>(shift_bits);
                    let coeffs = matrix::<BinaryField128>(log_n, width, 21).values;

                    let mut evals = coeffs.clone();
                    super::transform::<BinaryField128, false>(&mut evals, width, shift);

                    // The inverse of arbitrary values, so it is not checked only as an undo.
                    let mut expected = coeffs.clone();
                    super::transform::<BinaryField128, true>(&mut expected, width, shift);

                    for (route, forward, inverse) in routes {
                        let label =
                            format!("{route} log_n={log_n} width={width} shift={shift_bits:#x}");

                        let mut actual = coeffs.clone();
                        forward(&mut actual, width, shift);
                        assert_eq!(actual, evals, "ntt {label}");

                        let mut actual = coeffs.clone();
                        inverse(&mut actual, width, shift);
                        assert_eq!(actual, expected, "intt {label}");

                        // And the inverse of the codeword returns the coefficients.
                        let mut actual = evals.clone();
                        inverse(&mut actual, width, shift);
                        assert_eq!(actual, coeffs, "round trip {label}");
                    }
                }
            }
        }
    }

    /// The widest level's padded transform, and its run over `Ghash128`, against its network held in the tower basis.
    ///
    /// The level takes the `Ghash128` run only for a wide domain on a build with a carryless multiply.
    /// So the run is also called directly, which holds it to the tower result at every size.
    #[test]
    fn the_widest_level_encodes_as_its_tower_basis_network() {
        type Encode = fn(&mut [BinaryField128], usize, usize);
        let routes: [(&str, Encode); 2] = [
            ("level", BinaryField128::lch_transform_cosets),
            ("ghash", ghash_transform_cosets),
        ];

        // Fixture state: one tile holds 2^9 rows of 16 columns.
        //
        //     log_message 0 ..= 4   tiny cosets, each copied from the message and transformed alone
        //     log_message 9         a coset of exactly one tile, still copied
        //     log_message 11        cosets past one tile, which share their first pass
        assert_eq!(size_of::<BinaryField128>() * (16 << 9), DEEP_TILE_BYTES);
        let shapes = (0..=4)
            .flat_map(|log_message| [1usize, 3, 16].map(|width| (width, log_message)))
            .chain([(16, 9), (16, 11)]);
        for (width, log_message) in shapes {
            for log_inv_rate in 0..=3 {
                let mut padded = matrix::<BinaryField128>(log_message, width, 73).values;
                padded.resize(padded.len() << log_inv_rate, BinaryField128::ZERO);

                let mut expected = padded.clone();
                super::transform_cosets::<BinaryField128>(&mut expected, width, log_message);

                for (route, encode) in routes {
                    let label = format!(
                        "{route} log_message={log_message} width={width} rate={log_inv_rate}"
                    );
                    let mut actual = padded.clone();
                    encode(&mut actual, width, log_message);
                    assert_eq!(actual, expected, "{label}");
                }
            }
        }
    }
}
