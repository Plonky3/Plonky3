//! The Lin-Chung-Han transform over `GF(2^128)`, carried out in its polynomial basis.

mod plan;
mod stages;

use p3_binary_field::{BinaryField128, poly_basis};
use p3_commit::zero_padded;
use p3_field::PrimeCharacteristicRing;
use p3_matrix::Matrix;
use p3_matrix::dense::{RowMajorMatrix, RowMajorMatrixView};
use p3_maybe_rayon::prelude::current_num_threads;
use p3_util::log2_strict_usize;
use plan::Plan;
use stages::{
    Fold, INTO_POLY, convert, copy_coset, for_chunks, forward, inverse, padded_sharing_first_group,
};

use crate::domain::domain_point;
use crate::lch::{BUTTERFLY_GRAIN, LchNtt};
use crate::traits::{AdditiveNtt, padded_len};

/// The additive NTT over `GF(2^128)`, with the data held in the polynomial basis throughout.
///
/// - A tower-basis product changes both operands into the polynomial basis and back, which dominates a butterfly.
/// - Changing the whole matrix once each way costs two conversions per element instead of three per stage.
/// - Every twiddle product in between is then a bare carryless multiply and a reduction.
/// - Additions are exclusive ors in both bases, so they are unaffected.
///
/// Without a carryless-multiply instruction the product is bit-serial and loses to the tower.
///
/// The transform then runs the tower-basis transform instead, and only one arm is compiled.
#[derive(Clone, Copy, Debug, Default)]
pub struct PolyBasisNtt;

/// Whether cosets of `len` elements are large enough to spread over every worker on their own.
// The serial pool exposes a `const` thread count and the parallel one does not.
#[allow(clippy::missing_const_for_fn)]
fn large_cosets(len: usize) -> bool {
    len >= 2 * BUTTERFLY_GRAIN * current_num_threads()
}

/// The depth of the first staging group a borrowed message's cosets share, where they share one.
///
/// They share it when the encoding is padded, the target multiplies carrylessly, the cosets are large, and the plan stages a group at all.
fn borrowed_first_group(plan: Plan, len: usize, log_inv_rate: usize) -> Option<usize> {
    let shared = log_inv_rate > 0 && poly_basis::HAS_HARDWARE_CLMUL && large_cosets(len);
    plan.group_sizes().next().filter(|_| shared)
}

impl AdditiveNtt<BinaryField128> for PolyBasisNtt {
    fn shifted_ntt_batch(
        &self,
        mut mat: RowMajorMatrix<BinaryField128>,
        shift: BinaryField128,
    ) -> RowMajorMatrix<BinaryField128> {
        if !poly_basis::HAS_HARDWARE_CLMUL {
            return LchNtt::default().shifted_ntt_batch(mat, shift);
        }
        let plan = Plan::new(mat.width(), log2_strict_usize(mat.height()));

        // Both basis changes ride on the transform's first and last touch of each element.
        forward(
            BinaryField128::as_repr_slice_mut(&mut mat.values),
            plan,
            shift,
            Fold::BOTH,
        );
        mat
    }

    fn shifted_intt_batch(
        &self,
        mut mat: RowMajorMatrix<BinaryField128>,
        shift: BinaryField128,
    ) -> RowMajorMatrix<BinaryField128> {
        if !poly_basis::HAS_HARDWARE_CLMUL {
            return LchNtt::default().shifted_intt_batch(mat, shift);
        }
        let plan = Plan::new(mat.width(), log2_strict_usize(mat.height()));

        // Both basis changes ride on the transform's first and last touch of each element.
        inverse(
            BinaryField128::as_repr_slice_mut(&mut mat.values),
            plan,
            shift,
            Fold::BOTH,
        );
        mat
    }

    fn ntt_batch_padded(
        &self,
        mut mat: RowMajorMatrix<BinaryField128>,
        log_inv_rate: usize,
    ) -> RowMajorMatrix<BinaryField128> {
        let log_n = log2_strict_usize(mat.height());
        assert!(log_inv_rate <= log_n, "padding exceeds matrix height");
        if log_inv_rate == 0 || !poly_basis::HAS_HARDWARE_CLMUL {
            return self.ntt_batch(mat);
        }
        let log_message = log_n - log_inv_rate;
        let plan = Plan::new(mat.width(), log_message);
        let values = BinaryField128::as_repr_slice_mut(&mut mat.values);
        let len = values.len() >> log_inv_rate;
        let large = large_cosets(len);

        // Large cosets that stage their first group gather the message once for all of them.
        if let Some(depth) = plan.group_sizes().next().filter(|_| large) {
            padded_sharing_first_group(values, None, plan, depth, log_inv_rate);
            return mat;
        }

        // Otherwise every coset starts from its own copy of the message.
        //
        // Converting the message once, before the copies, serves every coset.
        let (message, tail) = values.split_at_mut(len);
        convert(message, INTO_POLY);
        if large {
            // A large coset is transformed right after its copy, while the copy is still in cache.
            for_chunks(tail, len, log_message, |(c, chunk)| {
                copy_coset(chunk, message);
                let shift = domain_point((c + 1) << log_message);
                forward(chunk, plan, shift, Fold::EXIT);
            });
            forward(message, plan, BinaryField128::ZERO, Fold::EXIT);
        } else if log_inv_rate >= 4 && len <= 32 * 1024 / size_of::<u128>() {
            // Many cache-sized cosets repay one bounded snapshot of their common source.
            // Keep every coset, including the first, in the same parallel transform pass.
            let snapshot = message.to_vec();
            for_chunks(values, len, log_message, |(c, chunk)| {
                if c != 0 {
                    chunk.copy_from_slice(&snapshot);
                }
                forward(chunk, plan, domain_point(c << log_message), Fold::EXIT);
            });
        } else {
            // Small cosets are copied first, then transformed together.
            for_chunks(tail, len, 1, |(_, chunk)| chunk.copy_from_slice(message));
            for_chunks(values, len, log_message, |(c, chunk)| {
                forward(chunk, plan, domain_point(c << log_message), Fold::EXIT);
            });
        }
        mat
    }

    fn ntt_batch_borrowed(
        &self,
        mat: RowMajorMatrixView<'_, BinaryField128>,
        log_inv_rate: usize,
    ) -> RowMajorMatrix<BinaryField128> {
        let width = mat.width();
        let len = mat.values.len();
        let plan = Plan::new(width, log2_strict_usize(mat.height()));

        // Only cosets that share their first group gather the message themselves.
        //
        // Any other encoding transforms a padded copy of it.
        let Some(depth) = borrowed_first_group(plan, len, log_inv_rate) else {
            return self.ntt_batch_padded(zero_padded(mat, log_inv_rate), log_inv_rate);
        };

        // Every coset is written in full by its first group, the leading one included.
        let mut values = BinaryField128::zero_vec(padded_len(len, log_inv_rate));
        padded_sharing_first_group(
            BinaryField128::as_repr_slice_mut(&mut values),
            Some(BinaryField128::as_repr_slice(mat.values)),
            plan,
            depth,
            log_inv_rate,
        );
        RowMajorMatrix::new(values, width)
    }

    fn shifted_lde_batch(
        &self,
        mut mat: RowMajorMatrix<BinaryField128>,
        added_bits: usize,
        shift: BinaryField128,
    ) -> RowMajorMatrix<BinaryField128> {
        if !poly_basis::HAS_HARDWARE_CLMUL {
            return LchNtt::default().shifted_lde_batch(mat, added_bits, shift);
        }
        let width = mat.width();
        let log_n = log2_strict_usize(mat.height());
        if added_bits == 0 {
            return mat;
        }

        // The input evaluations are already the first coset of the extended codeword.
        let len = mat.values.len();
        let mut extended = BinaryField128::zero_vec(padded_len(len, added_bits));
        extended[..len].copy_from_slice(&mat.values);

        // The input's own allocation then holds the coefficients, left in the polynomial basis.
        //
        // Every further coset starts from a copy of them.
        let plan = Plan::new(width, log_n);
        let coeffs = BinaryField128::as_repr_slice_mut(&mut mat.values);
        inverse(coeffs, plan, shift, Fold::ENTRY);
        let coeffs: &[u128] = coeffs;

        // Coset c + 1 of the extended domain starts at domain point (c + 1) * 2^log_n.
        for_chunks(
            &mut BinaryField128::as_repr_slice_mut(&mut extended)[len..],
            len,
            log_n,
            |(c, chunk)| {
                chunk.copy_from_slice(coeffs);
                let coset = shift + domain_point::<BinaryField128>((c + 1) << log_n);
                forward(chunk, plan, coset, Fold::EXIT);
            },
        );
        RowMajorMatrix::new(extended, width)
    }
}

#[cfg(test)]
mod tests {
    use alloc::format;
    use alloc::vec::Vec;

    use p3_binary_field::{BinaryField128, TowerLevel, poly_basis};
    use p3_field::PrimeCharacteristicRing;
    use p3_matrix::dense::RowMajorMatrix;
    use p3_maybe_rayon::prelude::current_num_threads;
    use p3_util::log2_floor_usize;
    use proptest::prelude::*;

    use super::PolyBasisNtt;
    use super::plan::{Plan, STAGED_RUN_BYTES, STAGED_WORKERS};
    use super::stages::{
        self, Fold, INTO_POLY, INTO_TOWER, Twiddles, convert, first_group_into_cosets,
        forward_below, padded_sharing_first_group,
    };
    use crate::domain::{domain_point, subspace_polynomial};
    use crate::lch::LchNtt;
    use crate::naive::NaiveAdditiveNtt;
    use crate::traits::AdditiveNtt;

    /// Cut points small enough to keep the test matrices tiny.
    ///
    /// One `(local, log_block, depth)` triple per branch of the schedule:
    ///
    /// - `(2, 0, 3)`: full groups, plus a one-stage leftover at local + depth + 1.
    /// - `(2, 2, 3)`: the same cut with runs of four rows per strided address.
    /// - `(2, 0, 0)`: a staging tile too narrow for two runs, so no group runs at all.
    /// - `(0, 0, 3)`: no contiguous tile, so every stage is fused.
    /// - `(0, 3, 3)`: runs as long as a whole group's stride, which the group has to shorten.
    /// - `(1, 1, 1)`: a depth of one, which never pays for a tile.
    const CUTS: [(usize, usize, usize); 6] = [
        (2, 0, 3),
        (2, 2, 3),
        (2, 0, 0),
        (0, 0, 3),
        (0, 3, 3),
        (1, 1, 1),
    ];

    /// Cuts and heights whose staging groups number three or more, with the group depths
    /// each one produces:
    ///
    /// - `(0, 0, 3, 8)`: 3 + 3 + 2       a short last group, no contiguous tile.
    /// - `(0, 1, 3, 9)`: 3 + 3 + 3       three groups of full depth, runs of two rows.
    /// - `(1, 2, 2, 8)`: 2 + 2 + 2       three groups and a one-stage leftover.
    /// - `(0, 0, 2, 8)`: 2 + 2 + 2 + 2   four groups of full depth.
    ///
    /// The staging loop reseeds its twiddle walk and its conversion flag on every turn, so
    /// only a third turn shows that the reseeding is not accidentally right for two.
    const DEEP_CUTS: [(usize, usize, usize, usize); 4] =
        [(0, 0, 3, 8), (0, 1, 3, 9), (1, 2, 2, 8), (0, 0, 2, 8)];

    /// Widths that cover a single element per row, an odd row, and rows of several elements.
    const WIDTHS: [usize; 5] = [1, 2, 3, 16, 64];

    /// Production shapes and the cut each one gets, as `width, log_n, (local, log_block,
    /// depth)`, at a worker count past the staging threshold.
    ///
    /// The heights put the branch that is interesting for that width on it:
    ///
    /// - `width`: 1 @ 2^10   the tile is the whole transform.
    /// - `width`: 1 @ 2^14   one group of six stages, runs of 64 rows.
    /// - `width`: 1 @ 2^20   the production commit shape, one group of nine.
    /// - `width`: 2 @ 2^13   one group of six stages, runs of 32 rows.
    /// - `width`: 3 @ 2^12   an odd width, whose 32-row run overshoots.
    /// - `width`: 4 @ 2^14   the widest row a run still spans several of.
    /// - `width`: 16 @ 2^10   a short group, above a tile of 128 rows.
    /// - `width`: 16 @ 2^14   one full group, at a height that moves the cut.
    /// - `width`: 64 @ 2^10   one group of six stages, a run of one row.
    /// - `width`: 512 @ 2^8    two groups of three stages.
    /// - `width 1024 @ 2^7`: three groups of two stages.
    const PRODUCTION_CUTS: [(usize, usize, (usize, usize, usize)); 11] = [
        (1, 10, (10, 6, 6)),
        (1, 14, (11, 6, 6)),
        (1, 20, (11, 3, 9)),
        (2, 13, (10, 5, 6)),
        (3, 12, (9, 5, 5)),
        (4, 14, (9, 4, 6)),
        (16, 10, (7, 2, 6)),
        (16, 14, (7, 1, 7)),
        (64, 10, (5, 0, 6)),
        (512, 8, (2, 0, 3)),
        (1024, 7, (1, 0, 2)),
    ];

    /// Shapes whose cut is pinned as a triple alone, since transforming them costs too much.
    ///
    /// Each row is a run-length decision, and what decides it:
    ///
    /// - `width 48 @ 2^18`: one doubling on offer, the least a growing run can have.
    /// - `width 16 @ 2^16`: two doublings, so the added gather stays unpaid.
    /// - `width`: 8 @ 2^18   three doublings, which is exactly what a gather costs.
    /// - `width`: 4 @ 2^20   four doublings, the widest row a run still spans several of.
    /// - `width`: 1 @ 2^22   four doublings again, from a run floor above one row.
    /// - `width 16 @ 2^18`: no leftover pass, so growth rebalances (8, 3) into (6, 5).
    /// - `width 16 @ 2^20`: a deep tile whose one group takes every stage above it.
    /// - `width`: 1 @ 2^25   a deep tile whose run gives doublings up to stay one group.
    ///
    /// The first and the last two are past the shared-cache figure, so they are the rows that
    /// read the deep budget. Width 16 at `2^20` rows is `2^24` elements, which is why none of
    /// these is transformed.
    const PLAN_ONLY_CUTS: [(usize, usize, (usize, usize, usize)); 8] = [
        (48, 18, (8, 1, 7)),
        (16, 16, (7, 0, 8)),
        (8, 18, (8, 3, 6)),
        (4, 20, (9, 4, 6)),
        (1, 22, (11, 6, 6)),
        (16, 18, (7, 2, 6)),
        (16, 20, (10, 0, 10)),
        (1, 25, (14, 3, 11)),
    ];

    /// Staging groups the deep budget produces, as `width, log_block, depth`.
    ///
    /// These are the group depths and run lengths of the deep rows of the plan-only cuts,
    /// which are magnitudes no cut small enough to transform ever reaches:
    ///
    /// - `width 48`: seven stages per group, a run of two rows.
    /// - `width 16`: ten stages per group, a run of one row.
    /// - `width`: 1   eleven stages per group, a run of eight rows.
    const DEEP_BUDGET_GROUPS: [(usize, usize, usize); 3] = [(48, 1, 7), (16, 0, 10), (1, 3, 11)];

    /// Contiguous tiles the deep budget produces, as `width, local`.
    ///
    /// A deep-budget tile is `2^local` rows of the width beside it:
    ///
    /// - `width`: 1   a tile of 2^14 rows.
    /// - `width 16`: a tile of 2^10 rows.
    /// - `width 48`: a tile of 2^8 rows.
    const DEEP_BUDGET_TILES: [(usize, usize); 3] = [(1, 14), (16, 10), (48, 8)];

    /// Cuts whose staging groups run at a height the reference oracle can still reach:
    ///
    /// - `(2, 0, 3, 6)`: one group of three stages, plus a one-stage leftover.
    /// - `(2, 2, 3, 6)`: the same cut with runs of four rows per strided address.
    /// - `(1, 1, 2, 5)`: two groups of two stages, runs of two rows.
    /// - `(0, 0, 3, 6)`: no contiguous tile, so every stage is fused.
    const ORACLE_CUTS: [(usize, usize, usize, usize); 4] =
        [(2, 0, 3, 6), (2, 2, 3, 6), (1, 1, 2, 5), (0, 0, 3, 6)];

    /// Widths from one element per row up to a row that spans a gathered run on its own.
    ///
    /// A run covers several rows at every width below `64`, and one row from there up.
    /// So this covers both regimes and the boundary between them.
    const ORACLE_WIDTHS: [usize; 10] = [1, 2, 3, 4, 5, 6, 7, 8, 16, 64];

    /// The tallest matrix the width sweeps transform.
    ///
    /// The production plan of a width-1 matrix fuses its first group at `2^13`.
    /// So a sweep has to reach past that before it exercises a staged gather at all.
    const MAX_LOG_N: usize = 14;

    /// The tallest matrix one width is swept to.
    ///
    /// A row of the target run length or more gathers one row per strided address at every
    /// height, so once its first group has appeared the sweep only repeats a plan the cut
    /// tests already run - at over half the cost of the whole sweep, since these are the
    /// widest matrices in it.
    fn max_log_n(width: usize) -> usize {
        if core::mem::size_of::<u128>() * width >= STAGED_RUN_BYTES {
            10
        } else {
            MAX_LOG_N
        }
    }

    /// The tallest matrix the reference transform is asked for.
    ///
    /// The oracle costs `O(n^2 log n)` per column, so `2^7` rows is already seconds of an
    /// unoptimised test run. No production plan gathers a tile that low, which is why the
    /// staged path reaches the oracle through the synthetic oracle cuts rather
    /// than by making this sweep taller.
    const REFERENCE_LOG_N: usize = 6;

    /// Neither basis conversion rides along, so only the stage schedule is under test.
    const NONE: Fold = Fold {
        entry: false,
        exit: false,
    };

    /// A shift with bits in both halves, so no twiddle is accidentally zero.
    fn test_shift() -> BinaryField128 {
        BinaryField128::from_repr((1 << 127) | 7919)
    }

    /// Values whose bits depend on the position, so a misapplied twiddle cannot cancel out.
    fn coefficients(log_n: usize, width: usize) -> Vec<u128> {
        (0..(width << log_n))
            .map(|i| {
                let low = (i as u64)
                    .wrapping_mul(0x9e37_79b9_7f4a_7c15)
                    .wrapping_add(0x5555_5555_5555_5555);
                let high = low.rotate_left(17).wrapping_mul(0xc2b2_ae3d_27d4_eb4f);
                (u128::from(high) << 64) | u128::from(low)
            })
            .collect()
    }

    /// One full pass per stage, straight off the twiddle accessor: the schedule every cut of
    /// the stage sequence has to reproduce bit for bit.
    fn per_stage_schedule(
        values: &mut [u128],
        width: usize,
        log_n: usize,
        shift: BinaryField128,
        inverse: bool,
    ) {
        let twiddles = Twiddles::new(log_n, shift);
        for k in 0..log_n {
            // The forward direction runs the widest stage first, the inverse the narrowest.
            let j = if inverse { k } else { log_n - 1 - k };
            // Stage `j` pairs rows `2^j` apart, so a block spans `2^(j+1)` rows and the
            // block index alone picks the twiddle.
            let half = (1 << j) * width;
            for (block, rows) in values.chunks_mut(half << 1).enumerate() {
                let t = twiddles.at(j, block);
                let (lo, hi) = rows.split_at_mut(half);
                if inverse {
                    poly_basis::butterfly_inverse(lo, hi, t);
                } else {
                    poly_basis::butterfly_forward(lo, hi, t);
                }
            }
        }
    }

    /// Run the scheduled transform of one direction in place.
    fn scheduled(values: &mut [u128], plan: Plan, inverse: bool, fold: Fold) {
        if inverse {
            stages::inverse(values, plan, test_shift(), fold);
        } else {
            stages::forward(values, plan, test_shift(), fold);
        }
    }

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

    /// One plan per branch of the cut, at every width.
    ///
    /// The synthetic heights are the branch boundaries of the cut:
    ///
    /// - `local`: nothing above the tile, so the tile is the whole transform.
    /// - `local + 1`: one stage above the tile, which is left unfused.
    /// - `local + depth`: one full staging group.
    /// - `local + depth+1`: a full group and a one-stage leftover.
    /// - `2*local + depth`: several groups and several tiles.
    ///
    /// The deep cuts then carry the heights that need three or more groups.
    fn cut_plans() -> Vec<Plan> {
        let mut plans = Vec::new();
        for (local, log_block, depth) in CUTS {
            for log_n in [
                local,
                local + 1,
                local + depth,
                local + depth + 1,
                2 * local + depth,
            ] {
                for width in WIDTHS {
                    plans.push(Plan {
                        width,
                        log_n,
                        local: local.min(log_n),
                        log_block,
                        depth,
                    });
                }
            }
        }
        for (local, log_block, depth, log_n) in DEEP_CUTS {
            for width in WIDTHS {
                plans.push(Plan {
                    width,
                    log_n,
                    local,
                    log_block,
                    depth,
                });
            }
        }
        // Serial upper passes can run in pairs, with one ordinary pass left over.
        for extra in 0..=5 {
            for width in WIDTHS {
                plans.push(Plan {
                    width,
                    log_n: 4 + extra,
                    local: 4,
                    log_block: 0,
                    depth: 0,
                });
            }
        }
        plans
    }

    #[test]
    fn every_cut_of_the_stage_sequence_matches_the_per_stage_schedule() {
        // Invariant: cutting the stage sequence into staging groups and a contiguous tile is
        // a pure reordering of memory traffic, so every element must come out bit for bit
        // what one full pass per stage produces, in both directions.
        for plan in cut_plans() {
            let Plan { width, log_n, .. } = plan;
            for inverse in [false, true] {
                let mut expected = coefficients(log_n, width);
                let mut actual = expected.clone();
                per_stage_schedule(&mut expected, width, log_n, test_shift(), inverse);
                scheduled(&mut actual, plan, inverse, NONE);
                assert_eq!(actual, expected, "{plan:?} inverse={inverse}");
            }
        }

        // The production cut points, at a worker count past the threshold, so the same cut
        // runs here on a serial build and a parallel one. What each cut is, rather than what
        // it does, is pinned where the budgets are tested.
        for (width, log_n, _) in PRODUCTION_CUTS {
            let plan = Plan::for_workers(width, log_n, 32);
            for inverse in [false, true] {
                let mut expected = coefficients(log_n, width);
                let mut actual = expected.clone();
                per_stage_schedule(&mut expected, width, log_n, test_shift(), inverse);
                scheduled(&mut actual, plan, inverse, NONE);
                assert_eq!(actual, expected, "{plan:?} inverse={inverse}");
            }
        }
    }

    #[test]
    fn a_deep_budget_cut_matches_the_per_stage_schedule() {
        // Invariant: the depths and run lengths the deep budget produces reorder the memory
        // traffic like any other cut, so they too have to come out bit for bit what one full
        // pass per stage produces. A matrix whose shape reads that budget is far past the
        // height a test can transform, so the two regimes are carried onto short matrices
        // instead: the group depths over a contiguous tile small enough to leave them room,
        // and the tile depths under a shallow group.
        let matches_per_stage = |plan: Plan| {
            let Plan { width, log_n, .. } = plan;
            for inverse in [false, true] {
                let mut expected = coefficients(log_n, width);
                let mut actual = expected.clone();
                per_stage_schedule(&mut expected, width, log_n, test_shift(), inverse);
                scheduled(&mut actual, plan, inverse, NONE);
                assert_eq!(actual, expected, "{plan:?} inverse={inverse}");
            }
        };

        // A group whose stride is shorter than the run shortens the run, so the tile sits one
        // stage above the run length to leave the planned run standing.
        for (width, log_block, depth) in DEEP_BUDGET_GROUPS {
            let local = log_block + 1;
            // One full group, a group and a leftover pass, then a short group under a full one.
            for extra in [0usize, 1, 2] {
                matches_per_stage(Plan {
                    width,
                    log_n: local + depth + extra,
                    local,
                    log_block,
                    depth,
                });
            }
        }

        // A tile this deep carries most of the transform, so the group above it is shallow.
        for (width, local) in DEEP_BUDGET_TILES {
            for extra in [1usize, 2, 3] {
                matches_per_stage(Plan {
                    width,
                    log_n: local + extra,
                    local,
                    log_block: 0,
                    depth: 2,
                });
            }
        }
    }

    #[test]
    fn every_width_and_height_matches_the_per_stage_schedule() {
        // Invariant: the rows a gather moves per strided address regroup the matrix, they
        // do not change the transform. The run length a plan picks varies over the narrow
        // widths and the cut points move with the height, so the two are swept together
        // against the per-stage schedule.
        for width in ORACLE_WIDTHS {
            for log_n in 0..=max_log_n(width) {
                // One worker gathers no tile where many do, so both readings of one shape
                // are swept, not just this build's own.
                for workers in [1, 32] {
                    let plan = Plan::for_workers(width, log_n, workers);
                    for shift in [BinaryField128::ZERO, test_shift()] {
                        for inverse in [false, true] {
                            let mut expected = coefficients(log_n, width);
                            let mut actual = expected.clone();
                            per_stage_schedule(&mut expected, width, log_n, shift, inverse);
                            if inverse {
                                stages::inverse(&mut actual, plan, shift, NONE);
                            } else {
                                stages::forward(&mut actual, plan, shift, NONE);
                            }
                            assert_eq!(
                                actual, expected,
                                "{plan:?} shift={shift:?} inverse={inverse}"
                            );
                        }
                    }
                }
            }
        }
    }

    /// Full passes over the matrix a cut of `above` stages into groups of `depth` takes,
    /// counted by walking the loop the forward transform walks rather than by asking the plan.
    ///
    /// Each turn of the staging loop reads and writes the matrix once, and each stage the
    /// loop stops short of is a plain pass of its own. Nothing here calls into `Plan`, so a
    /// plan that quietly buys its run length with an extra pass comes out with a larger
    /// count than the run it is measured against.
    fn passes_above_the_tile(above: usize, depth: usize) -> usize {
        let leftover = if depth < 2 {
            above
        } else {
            usize::from(above % depth == 1)
        };
        let mut passes = 0;
        let mut remaining = above;
        while remaining > leftover {
            remaining -= depth.min(remaining - leftover);
            passes += 1;
        }
        passes + remaining
    }

    #[test]
    fn the_plan_gathers_the_longest_run_its_budgets_allow() {
        // The production cut points, as `local, log_block, depth`, past the worker threshold.
        //
        // A run length is a choice between two plans that take the same number of passes.
        // So no pass count can pin it, and these triples are what says which one is picked.
        for (width, log_n, cut) in PRODUCTION_CUTS.into_iter().chain(PLAN_ONLY_CUTS) {
            let plan = Plan::for_workers(width, log_n, 32);
            assert_eq!((plan.local, plan.log_block, plan.depth), cut, "{plan:?}");
        }

        // Invariant: growing the run never costs a pass. Gathering one row per strided
        // address is the shortest run there is, so its pass count is the ceiling, and both
        // sides are counted by walking the staging loop rather than by the rule that picked
        // the run.
        for width in [4usize, 5, 8, 16, 64, 512, 1024] {
            for log_n in 0..=24 {
                let plan = Plan::for_workers(width, log_n, 32);
                let row = core::mem::size_of::<u128>() * width;
                let one_row = log2_floor_usize((Plan::budgets(row, log_n).1 / row).max(1));
                let above = log_n - plan.local;
                assert!(
                    passes_above_the_tile(above, plan.depth)
                        <= passes_above_the_tile(above, one_row),
                    "{plan:?}"
                );
            }
        }
    }

    #[test]
    fn a_staged_cut_matches_the_reference_transform() {
        // The oracle depends on none of the identities the fast transform is built from, but
        // it costs `O(n^2 log n)` per column, so a production plan's first staging group sits
        // far above the heights it can reach. These cuts put a gather and a scatter inside
        // its range instead, which is what pins the staged path to the map rather than to the
        // stage schedule alone.
        let naive = NaiveAdditiveNtt::<BinaryField128>::default();
        for (local, log_block, depth, log_n) in ORACLE_CUTS {
            for width in [1usize, 3, 16] {
                let plan = Plan {
                    width,
                    log_n,
                    local,
                    log_block,
                    depth,
                };
                assert!(
                    plan.group_sizes().next().is_some(),
                    "{plan:?} gathers no tile"
                );
                for shift in [BinaryField128::ZERO, test_shift()] {
                    let coeffs = matrix(log_n, width, 3);
                    let evals = naive.shifted_ntt_batch(coeffs.clone(), shift);
                    let reprs = |mat: &RowMajorMatrix<BinaryField128>| -> Vec<u128> {
                        mat.values
                            .iter()
                            .copied()
                            .map(BinaryField128::to_repr)
                            .collect()
                    };

                    // Both conversions ride along, so this is what `shifted_ntt_batch` runs.
                    let mut actual = reprs(&coeffs);
                    stages::forward(&mut actual, plan, shift, Fold::BOTH);
                    assert_eq!(actual, reprs(&evals), "ntt {plan:?} shift={shift:?}");

                    let mut actual = reprs(&evals);
                    stages::inverse(&mut actual, plan, shift, Fold::BOTH);
                    assert_eq!(actual, reprs(&coeffs), "intt {plan:?} shift={shift:?}");
                }
            }
        }
    }

    #[test]
    fn a_row_that_covers_a_line_is_gathered_whatever_the_worker_count() {
        // Invariant: the worker count may only switch off a staging that one row per
        // address never reached, so a row of a line or more keeps its tile at every count.
        for width in [4usize, 16, 64, 512] {
            for log_n in [10, 14, 20] {
                let alone = Plan::for_workers(width, log_n, 1);
                let shared = Plan::for_workers(width, log_n, 32);
                assert_eq!(alone.depth, shared.depth, "{alone:?}");
                assert_eq!(alone.log_block, shared.log_block, "{alone:?}");
            }
        }
    }

    #[test]
    fn the_worker_count_decides_whether_a_narrow_row_is_gathered() {
        // The production width-1 shape, at the height whose stages a tile would fuse. One
        // worker leaves every stage above the contiguous tile as a plain pass, and the run
        // length it would have used does not matter, because no tile is gathered.
        let alone = Plan::for_workers(1, 20, 1);
        assert_eq!(alone.depth, 0, "{alone:?}");
        assert_eq!(alone.leftover(), alone.log_n - alone.local, "{alone:?}");

        // At the threshold the tile appears, and the stages above the contiguous tile fuse.
        let shared = Plan::for_workers(1, 20, STAGED_WORKERS);
        assert!(shared.depth >= 2, "{shared:?}");
        assert_eq!(shared.leftover(), 0, "{shared:?}");

        // Only the staging depth turns on the worker count; the rest of the cut does not.
        assert_eq!(alone.local, shared.local);
        assert_eq!(alone.log_block, shared.log_block);
    }

    #[test]
    fn every_width_matches_the_reference_transform() {
        // The oracle depends on none of the identities the fast transform is built from, so
        // it pins the map itself rather than the schedule. A zero shift makes the first stage
        // twiddle of every block the domain point alone, which is the one case an off-by-one
        // in the shift table survives.
        let poly = PolyBasisNtt;
        let naive = NaiveAdditiveNtt::<BinaryField128>::default();
        for width in ORACLE_WIDTHS {
            for log_n in 0..=REFERENCE_LOG_N {
                for shift in [BinaryField128::ZERO, test_shift()] {
                    let coeffs = matrix(log_n, width, 3);
                    let evals = naive.shifted_ntt_batch(coeffs.clone(), shift);
                    assert_eq!(
                        poly.shifted_ntt_batch(coeffs.clone(), shift),
                        evals,
                        "ntt width={width} log_n={log_n} shift={shift:?}"
                    );
                    assert_eq!(
                        poly.shifted_intt_batch(evals, shift),
                        coeffs,
                        "intt width={width} log_n={log_n} shift={shift:?}"
                    );
                }
            }
        }
    }

    #[test]
    fn padded_transform_matches_the_tower_at_every_width() {
        // The padded entry point plans one coset and reuses that plan for all of them, so a
        // run length suiting the full height but not the message height shows up here. The
        // tower transform stands in for the oracle, which these message heights are past, and
        // `crate::lch`'s `check_matches_naive` is what pins it to the oracle.
        let poly = PolyBasisNtt;
        let tower = LchNtt::<BinaryField128>::default();
        for width in ORACLE_WIDTHS {
            for log_message in [0, 1, 5, 9] {
                for log_inv_rate in 0..=3 {
                    let mut mat = matrix(log_message, width, 23);
                    mat.values
                        .resize(mat.values.len() << log_inv_rate, BinaryField128::ZERO);
                    let expected = tower.ntt_batch(mat.clone());
                    assert_eq!(
                        poly.ntt_batch_padded(mat, log_inv_rate),
                        expected,
                        "width={width} log_message={log_message} rate={log_inv_rate}"
                    );
                }
            }
        }
    }

    #[test]
    fn folded_conversions_match_standalone_conversion_passes() {
        // Invariant: a basis conversion carried by whichever phase first reads or last writes
        // an element is the same map as a standalone pass over the whole matrix before or
        // after the transform.
        //
        // The cut points decide which phase carries it - a staging group's gather, a
        // staging group's scatter, the contiguous tile, or a pass of its own - so the same
        // set of cuts as the schedule test runs here.
        //
        // A run of three or more groups is what shows that the entry conversion rides the
        // first group only and the exit conversion the last group only.
        for plan in cut_plans() {
            let Plan { width, log_n, .. } = plan;
            for inverse in [false, true] {
                let input = coefficients(log_n, width);

                // Conversion in, then the schedule.
                let mut expected = input.clone();
                convert(&mut expected, INTO_POLY);
                scheduled(&mut expected, plan, inverse, NONE);
                let mut actual = input.clone();
                scheduled(&mut actual, plan, inverse, Fold::ENTRY);
                assert_eq!(actual, expected, "entry {plan:?} inverse={inverse}");

                // Conversion in, the schedule, conversion out.
                convert(&mut expected, INTO_TOWER);
                let mut actual = input.clone();
                scheduled(&mut actual, plan, inverse, Fold::BOTH);
                assert_eq!(actual, expected, "both {plan:?} inverse={inverse}");

                // The schedule, then conversion out.
                let mut expected = input.clone();
                scheduled(&mut expected, plan, inverse, NONE);
                convert(&mut expected, INTO_TOWER);
                let mut actual = input;
                scheduled(&mut actual, plan, inverse, Fold::EXIT);
                assert_eq!(actual, expected, "exit {plan:?} inverse={inverse}");
            }
        }
    }

    #[test]
    fn a_first_group_shared_by_every_coset_matches_each_coset_alone() {
        // Invariant: every coset of a padded message starts from the same coefficients, so
        // gathering them once for the first group of all cosets changes the traffic alone,
        // and every coset comes out as its own transform of the message would.
        for plan in cut_plans() {
            let Plan { width, log_n, .. } = plan;
            let Some(depth) = plan.group_sizes().next() else {
                continue;
            };
            let message = coefficients(log_n, width);
            let len = message.len();
            for log_cosets in 0..=2 {
                let twiddles: Vec<_> = (0..1 << log_cosets)
                    .map(|c| Twiddles::new(log_n, domain_point(c << log_n)))
                    .collect();
                let mut actual = message.clone();
                actual.resize(len << log_cosets, 0);
                first_group_into_cosets(&mut actual, None, len, plan, depth, &twiddles);
                for (c, coset) in actual.chunks_mut(len).enumerate() {
                    forward_below(coset, plan, &twiddles[c], 1, Fold::EXIT);
                }

                // The message gathered from a buffer of its own fills the leading coset too.
                let mut separate = alloc::vec![0; len << log_cosets];
                first_group_into_cosets(
                    &mut separate,
                    Some(message.as_slice()),
                    len,
                    plan,
                    depth,
                    &twiddles,
                );
                for (c, coset) in separate.chunks_mut(len).enumerate() {
                    forward_below(coset, plan, &twiddles[c], 1, Fold::EXIT);
                }
                assert_eq!(separate, actual, "{plan:?} separate source");

                for (c, coset) in actual.chunks(len).enumerate() {
                    let mut expected = message.clone();
                    stages::forward(&mut expected, plan, domain_point(c << log_n), Fold::BOTH);
                    assert_eq!(coset, &expected[..], "{plan:?} coset={c}");
                }
            }
        }
    }

    #[test]
    fn padded_transform_matches_the_tower_where_cosets_share_their_first_group() {
        // The padded entry point shares a first group only above a length that grows with the
        // worker count, and a single column stages one only from a worker count up. Both are
        // the host's, so the shared path is driven directly, under the plan a pinned count of
        // workers gets, rather than through the entry point.
        //
        // Rows of four and sixteen elements fill a cache line, and a single column is staged
        // from `STAGED_WORKERS` up, as it commits.
        let tower = LchNtt::<BinaryField128>::default();
        for (width, log_message) in [(4, 14), (16, 12), (1, 17)] {
            let plan = Plan::for_workers(width, log_message, STAGED_WORKERS);
            let depth = plan
                .group_sizes()
                .next()
                .expect("every shape here stages a first group");
            for log_inv_rate in 1..=2 {
                let mut mat = matrix(log_message, width, 29);
                mat.values
                    .resize(mat.values.len() << log_inv_rate, BinaryField128::ZERO);
                let expected = tower.ntt_batch(mat.clone());

                let mut values: Vec<u128> = mat.values.iter().map(|v| v.to_repr()).collect();
                let message = values[..values.len() >> log_inv_rate].to_vec();
                padded_sharing_first_group(&mut values, None, plan, depth, log_inv_rate);
                let actual: Vec<_> = values.into_iter().map(BinaryField128::from_repr).collect();
                assert_eq!(
                    actual, expected.values,
                    "width={width} log_message={log_message} rate={log_inv_rate}"
                );

                // A message read from its own buffer encodes into a codeword left zero.
                let mut values = alloc::vec![0; mat.values.len()];
                padded_sharing_first_group(
                    &mut values,
                    Some(message.as_slice()),
                    plan,
                    depth,
                    log_inv_rate,
                );
                let actual: Vec<_> = values.into_iter().map(BinaryField128::from_repr).collect();
                assert_eq!(
                    actual, expected.values,
                    "borrowed width={width} log_message={log_message} rate={log_inv_rate}"
                );
            }
        }
    }

    #[test]
    fn padded_transform_matches_naive_at_wide_widths() {
        for width in [1, 4, 16, 64] {
            for added in [0, 1, 2, 3] {
                let mut mat = matrix(4, width, 13);
                mat.values
                    .resize(mat.values.len() << added, BinaryField128::ZERO);
                let expected = NaiveAdditiveNtt::default().ntt_batch(mat.clone());
                assert_eq!(PolyBasisNtt.ntt_batch_padded(mat, added), expected);
            }
        }
    }

    /// Runs a check under a fixed pool of [`STAGED_WORKERS`] workers.
    ///
    /// That many workers stage a single column's first group.
    ///
    /// Every length past `2^13` elements then counts as large, whatever the host's own count.
    #[cfg(feature = "parallel")]
    fn with_staging_workers<R: Send>(check: impl FnOnce() -> R + Send) -> R {
        rayon::ThreadPoolBuilder::new()
            .num_threads(STAGED_WORKERS)
            .build()
            .unwrap()
            .install(check)
    }

    /// Runs a check on the one worker a build without rayon has.
    #[cfg(not(feature = "parallel"))]
    fn with_staging_workers<R: Send>(check: impl FnOnce() -> R + Send) -> R {
        check()
    }

    /// The transform of a borrowed message, and the transform of its zero-padded copy.
    fn borrowed_and_padded(
        width: usize,
        log_message: usize,
        log_inv_rate: usize,
        seed: u64,
    ) -> (
        RowMajorMatrix<BinaryField128>,
        RowMajorMatrix<BinaryField128>,
    ) {
        let message = matrix(log_message, width, seed);
        let mut padded = message.clone();
        padded
            .values
            .resize(padded.values.len() << log_inv_rate, BinaryField128::ZERO);
        let borrowed = PolyBasisNtt.ntt_batch_borrowed(message.as_view(), log_inv_rate);
        let padded = PolyBasisNtt.ntt_batch_padded(padded, log_inv_rate);
        (borrowed, padded)
    }

    #[test]
    fn a_borrowed_message_encodes_as_its_padding() {
        // Small messages take the copy, and tall ones share their first group.
        //
        // A row of a whole line stages at any worker count, a single column from four workers.
        //
        // The route is asserted, so a host that would skip it fails instead of passing.
        with_staging_workers(|| {
            let workers = current_num_threads();
            for (width, log_message, routed) in [
                (1, 4, false),
                (4, 6, false),
                (4, 15, true),
                (8, 14, true),
                (1, 17, workers >= STAGED_WORKERS),
            ] {
                for log_inv_rate in 0..=2 {
                    let label =
                        format!("width={width} log_message={log_message} rate={log_inv_rate}");
                    let plan = Plan::new(width, log_message);
                    let len = width << log_message;
                    assert_eq!(
                        super::borrowed_first_group(plan, len, log_inv_rate).is_some(),
                        routed && log_inv_rate > 0 && poly_basis::HAS_HARDWARE_CLMUL,
                        "{label}"
                    );

                    let (borrowed, padded) =
                        borrowed_and_padded(width, log_message, log_inv_rate, 31);
                    assert_eq!(borrowed, padded, "{label}");
                }
            }
        });
    }

    #[test]
    fn shifted_lde_matches_naive_at_wide_widths() {
        for width in [1, 4, 16, 64] {
            for added in [0, 1, 2, 3] {
                let mat = matrix(4, width, 17);
                let shift = BinaryField128::from_repr(1 << 127);
                let expected =
                    NaiveAdditiveNtt::default().shifted_lde_batch(mat.clone(), added, shift);
                let actual = PolyBasisNtt.shifted_lde_batch(mat.clone(), added, shift);
                assert_eq!(actual, expected);
                assert_eq!(&actual.values[..mat.values.len()], &mat.values);
            }
        }
    }

    #[test]
    #[should_panic = "codeword length overflows usize"]
    fn lde_rejects_length_overflow() {
        let _ = PolyBasisNtt.lde_batch(matrix(1, 1, 0), usize::BITS as usize - 1);
    }

    #[test]
    fn incremental_twiddles_match_independent_domain_points() {
        let shift = BinaryField128::from_repr((1 << 127) | 123);
        let twiddles = Twiddles::new(usize::BITS as usize - 1, shift);
        for stage in [0, 1, 7, 15, 31]
            .into_iter()
            .filter(|&stage| stage < usize::BITS as usize - 1)
        {
            for start in [0, 1, 63, 127, (1usize << (usize::BITS - 3)) - 3] {
                let mut t = twiddles.at(stage, start);
                for block in start..start + 9 {
                    let expected = subspace_polynomial(stage, shift)
                        + domain_point::<BinaryField128>(block << 1);
                    assert_eq!(t, poly_basis::from_tower(expected));
                    t ^= twiddles.deltas[block.trailing_ones() as usize];
                }
            }
        }
    }

    #[test]
    fn transforms_cross_cache_boundaries_in_natural_order() {
        for width in [1, 3, 16, 64] {
            let mat = matrix(12, width, 29);
            let shift = BinaryField128::from_repr((1 << 127) | 7919);
            let expected = crate::LchNtt::default().shifted_ntt_batch(mat.clone(), shift);
            let actual = PolyBasisNtt.shifted_ntt_batch(mat.clone(), shift);
            assert_eq!(actual, expected);
            assert_eq!(PolyBasisNtt.shifted_intt_batch(actual, shift), mat);
        }
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(32))]

        /// A borrowed message transforms as its zero-padded copy does, at every width and rate.
        ///
        /// The pool is fixed, so the tall messages take the route that reads the message in place.
        #[test]
        fn a_borrowed_message_matches_its_padded_copy(
            width in prop::sample::select(alloc::vec![1usize, 2, 3, 4, 8]),
            log_message in 0usize..=13,
            log_inv_rate in 0usize..=3,
            seed in any::<u64>(),
        ) {
            let (borrowed, padded) = with_staging_workers(|| {
                borrowed_and_padded(width, log_message, log_inv_rate, seed)
            });
            prop_assert_eq!(borrowed, padded);
        }

        /// The polynomial-basis transform is the same map as the reference oracle.
        #[test]
        fn poly_basis_matches_naive(
            log_n in 0usize..=8,
            width in 1usize..=8,
            seed in any::<u64>(),
            shift in any::<u64>(),
        ) {
            let coeffs = matrix(log_n, width, seed);
            let shift = BinaryField128::from_le_byte_iter(
                shift.to_le_bytes().into_iter().cycle(),
            );

            let fast = PolyBasisNtt.shifted_ntt_batch(coeffs.clone(), shift);
            let slow = NaiveAdditiveNtt::<BinaryField128>::default()
                .shifted_ntt_batch(coeffs, shift);
            prop_assert_eq!(fast, slow);
        }

        #[test]
        fn poly_basis_round_trips(
            log_n in 0usize..=8,
            width in 1usize..=8,
            seed in any::<u64>(),
            shift in any::<u64>(),
        ) {
            let coeffs = matrix(log_n, width, seed);
            let shift = BinaryField128::from_le_byte_iter(
                shift.to_le_bytes().into_iter().cycle(),
            );
            let ntt = PolyBasisNtt;
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

            let lde = PolyBasisNtt.shifted_lde_batch(coeffs.clone(), added, shift);
            let naive = NaiveAdditiveNtt::<BinaryField128>::default()
                .shifted_lde_batch(coeffs.clone(), added, shift);
            prop_assert_eq!(&lde, &naive);
            prop_assert_eq!(&lde.values[..coeffs.values.len()], &coeffs.values[..]);
        }
    }

    #[test]
    fn poly_basis_matches_the_tower_across_several_tasks() {
        // These heights take more than one butterfly task per stage, so a task seeds its
        // twiddle at a block index of its own rather than at zero. Every oracle test sits
        // below them, so `LchNtt` stands in for the oracle here.
        //
        // An independent twiddle walk holds `LchNtt` itself at these heights, and it keeps its
        // data in the tower basis and derives its twiddles separately, so it shares no
        // arithmetic with the transform under test.
        let poly = PolyBasisNtt;
        let tower = LchNtt::<BinaryField128>::default();
        for width in ORACLE_WIDTHS {
            // The tall height is where the plan cuts deepest, and a tower transform of a
            // tall wide matrix is the most expensive thing here, so the narrow widths carry
            // it, where a run spans the most rows. The per-stage schedule covers every width
            // at every height regardless.
            let heights: &[usize] = if width <= 4 {
                &[7, 10, MAX_LOG_N]
            } else {
                &[7, 10]
            };
            for &log_n in heights {
                for shift_bits in [0u64, 0x1234_5678_9abc_def0] {
                    let coeffs = matrix(log_n, width, 5);
                    let shift = BinaryField128::from_le_byte_iter(
                        shift_bits.to_le_bytes().into_iter().cycle(),
                    );

                    let evals = tower.shifted_ntt_batch(coeffs.clone(), shift);
                    assert_eq!(
                        poly.shifted_ntt_batch(coeffs.clone(), shift),
                        evals,
                        "ntt width={width} log_n={log_n} shift={shift_bits:#x}"
                    );
                    assert_eq!(
                        poly.shifted_intt_batch(evals, shift),
                        coeffs,
                        "intt width={width} log_n={log_n} shift={shift_bits:#x}"
                    );
                }
            }
        }
    }

    /// A height that is not a power of two has no well-defined `l`. `l` exceeding the bit width
    /// of `BinaryField128` is covered generically by the tower transform, where the level
    /// is a type parameter and the panic is cheap to reach; at a fixed `BinaryField128` it is
    /// only reachable through a matrix of `2^129` rows, which is not a test worth writing.
    #[test]
    #[should_panic]
    fn shifted_ntt_batch_rejects_a_non_power_of_two_height() {
        let coeffs = RowMajorMatrix::new(matrix(0, 1, 0).values.repeat(3), 1);
        let _ = PolyBasisNtt.ntt_batch(coeffs);
    }

    #[test]
    fn small_high_rate_cosets_match_the_tower_transform() {
        let check = || {
            for (width, log_message, rate) in [
                (1, 0, 4),
                (3, 0, 10),
                (3, 1, 9),
                (3, 1, 10),
                (16, 3, 10),
                (16, 3, 3),
                (16, 7, 4),
                (17, 7, 4),
                (3, 9, 4),
                (3, 10, 4),
                (1, 11, 4),
                (1, 12, 4),
                (1, 13, 4),
            ] {
                let message = matrix(log_message, width, 1793);
                let mut padded = message.clone();
                padded
                    .values
                    .resize(padded.values.len() << rate, BinaryField128::ZERO);
                let expected = LchNtt::<BinaryField128>::default().ntt_batch(padded.clone());
                assert_eq!(
                    PolyBasisNtt.ntt_batch_padded(padded, rate),
                    expected,
                    "width={width} log={log_message} rate={rate}"
                );
                assert_eq!(
                    PolyBasisNtt.ntt_batch_borrowed(message.as_view(), rate),
                    expected
                );
                assert_eq!(message, matrix(log_message, width, 1793));
            }
        };
        #[cfg(feature = "parallel")]
        for workers in [1, 2, 4] {
            rayon::ThreadPoolBuilder::new()
                .num_threads(workers)
                .build()
                .unwrap()
                .install(check);
        }
        #[cfg(not(feature = "parallel"))]
        check();
    }
}
