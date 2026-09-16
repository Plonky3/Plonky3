//! The Lin–Chung–Han additive NTT.

use alloc::vec::Vec;
use core::marker::PhantomData;

use p3_binary_field::TowerLevel;
use p3_field::{PackedValue, PrimeCharacteristicRing};
use p3_matrix::Matrix;
use p3_matrix::dense::RowMajorMatrix;
use p3_maybe_rayon::prelude::*;
use p3_util::{DisjointMutPtr, log2_ceil_usize, log2_floor_usize, log2_strict_usize};

use crate::domain::{domain_point, domain_point_steps};
use crate::traits::AdditiveNtt;

/// The Lin–Chung–Han additive NTT over the Cantor-basis domain.
///
/// Twiddles are index shifts (D8): at stage `j` and butterfly block `blk` the twiddle is
/// `W_j(shift) + domain_point(blk << 1)`, so there is no twiddle table.
///
/// Consecutive blocks differ by a fixed increment, so a block takes its twiddle from the one
/// before it rather than walking its own index.
/// The `ℓ - 1` increments are shared by every stage and are the only per-size precomputation.
///
/// `W_j` is `F_2`-linear and `domain_point(0)` is zero, so over the subspace itself — the
/// coset with `shift = 0` — the first block of every stage has a zero twiddle and its
/// butterfly collapses to a single addition. The inner loop takes that as a separate case,
/// which removes one multiply in `2/ℓ` of them.
///
/// At the wide stages a pair of rows is megabytes apart, so a stage alone costs a pass over
/// memory. The schedule groups adjacent stages into row sets closed under all of them.
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
/// piece per side and instead gather `BUTTERFLY_GRAIN / half` whole blocks into one task, so a
/// task is a piece of this size on either side of the crossover.
///
/// At a few nanoseconds per butterfly a piece of this size is microseconds of work, well above
/// the cost of handing a task to another thread, while still leaving hundreds of pieces per
/// stage at the smallest useful heights.
pub(crate) const BUTTERFLY_GRAIN: usize = 1 << 10;

/// The byte budget of one contiguous row tile.
///
/// A worker holds one tile for every stage its rows are closed under, so a tile per worker has
/// to fit a core's private second-level cache.
///
/// Doubling buys one more stage inside the tile. A sweep of 32 KiB to 256 KiB flattens here.
const DEEP_TILE_BYTES: usize = 128 * 1024;

/// The byte budget of one staging tile.
///
/// Every worker holds one for the whole group, so the budget is paid per worker.
///
/// Halving it fuses one stage fewer, which is one more traversal. A sweep of 32 KiB to 1 MiB
/// is flat from here up, so this is the smallest budget at the floor.
const STAGING_TILE_BYTES: usize = 64 * 1024;

/// The smallest number of stages a staging tile has to cover to be worth gathering for.
///
/// One stage gathered and scattered moves the memory traffic one plain pass moves.
/// It then adds a copy in each direction, so grouping pays only from two stages up.
const MIN_FUSED_STAGES: usize = 2;

/// Workers a transform needs before the stages above the tile are worth staging.
///
/// Staging trades one pass per stage for a strided gather and scatter. Few workers issue those
/// passes far below what memory will serve, so the copies have nothing to pay for them.
///
/// The turn is fixed by the last-level cache, which a `no_std` crate cannot read, so the worker
/// count is the signal left. Measured, staging loses up to a third below this count and wins
/// from it up. Fewer workers give up a win rather than take a loss.
const STAGED_WORKERS: usize = 16;

/// The smallest cache line among the supported targets, in bytes.
///
/// A gather that picks up less than a line from each row it visits pays for bytes it cannot
/// use. Apple Silicon reports 128, so this is the smallest line rather than the transfer
/// granularity: past it no target wastes more than half of a line.
///
/// A staging tile therefore holds whole runs of rows wherever a row is shorter than this.
const CACHE_LINE_BYTES: usize = 64;

impl<F: TowerLevel> AdditiveNtt<F> for LchNtt<F> {
    fn shifted_ntt_batch(&self, mat: RowMajorMatrix<F>, shift: F) -> RowMajorMatrix<F> {
        transform::<F, false>(mat, shift)
    }

    fn shifted_intt_batch(&self, mat: RowMajorMatrix<F>, shift: F) -> RowMajorMatrix<F> {
        transform::<F, true>(mat, shift)
    }
}

/// Stage twiddle bases and the increments between consecutive block twiddles.
struct Twiddles<F> {
    /// `W_j(shift)`, the twiddle of the first block of stage `j`, for every stage.
    bases: Vec<F>,
    /// The difference between consecutive block twiddles, indexed by trailing-zero count.
    steps: Vec<F>,
}

impl<F: TowerLevel> Twiddles<F> {
    /// Precompute the twiddle state a transform of `2^log_n` rows over `shift + S_ℓ` needs.
    fn new(log_n: usize, shift: F) -> Self {
        // `W_0` is the identity, and `W_{j+1}(x) = W_j(x)² + W_j(x)`.
        //
        // So the bases form a chain, one squaring each rather than `j` squarings apiece.
        let mut base = shift;
        let bases = (0..log_n)
            .map(|_| {
                let current = base;
                base = base.square() + base;
                current
            })
            .collect();

        // An increment depends only on a block index's trailing-zero count, never on the stage,
        // so one table serves every stage and each stage uses the prefix it reaches.
        // A height of one runs no stage and needs no table.
        let steps = domain_point_steps::<F>(log_n.saturating_sub(1));

        Self { bases, steps }
    }

    /// The twiddle of butterfly block `block` at stage `stage`.
    #[inline]
    fn at(&self, stage: usize, block: usize) -> F {
        self.bases[stage] + domain_point::<F>(block << 1)
    }

    /// What to add to block `index - 1`'s twiddle to reach block `index`'s.
    #[inline]
    fn step(&self, index: usize) -> F {
        self.steps[index.trailing_zeros() as usize]
    }
}

/// One stage, as a single pass over every row.
fn stage_pass<F: TowerLevel, const INVERSE: bool>(
    values: &mut [F],
    width: usize,
    j: usize,
    twiddles: &Twiddles<F>,
) {
    let half = (1 << j) * width;
    let per_task = (BUTTERFLY_GRAIN / half).max(1);
    values
        .par_chunks_mut(per_task * (half << 1))
        .enumerate()
        .for_each(|(task, group)| {
            let first = task * per_task;
            let mut t = twiddles.at(j, first);
            // Invariant: blocks are visited in ascending index order.
            // Carrying the twiddle from one block to the next relies on it.
            for (i, block) in group.chunks_mut(half << 1).enumerate() {
                if i != 0 {
                    t += twiddles.step(first + i);
                }
                let (lo, hi) = block.split_at_mut(half);
                let butterfly = |lo: &mut [F], hi: &mut [F]| {
                    packed_butterfly::<F, INVERSE>(lo, hi, t);
                };
                // Pairs are independent across the block.
                //
                // So a block wider than the grain is split further, not run on one thread.
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

/// Run the `log_rows` adjacent stages a tile of `2^log_rows` rows is closed under.
///
/// Tile row `q` sits in global block `(index << (log_rows - 1 - jj)) + (q >> (jj + 1))` at the
/// stage pairing rows `2^jj` apart. The second term is what the tile-local pass already walks,
/// so only the first has to be passed in.
///
/// Forward runs its widest stage first, inverse the narrowest.
fn tile_stages<F: TowerLevel, const INVERSE: bool>(
    tile: &mut [F],
    row_len: usize,
    log_rows: usize,
    stage_base: usize,
    index: usize,
    twiddles: &Twiddles<F>,
) {
    for step in 0..log_rows {
        let jj = if INVERSE { step } else { log_rows - 1 - step };
        let half = (1 << jj) * row_len;

        // The tile's own blocks continue the global numbering from here.
        let first = index << (log_rows - 1 - jj);
        let mut t = twiddles.at(stage_base + jj, first);

        // Invariant: blocks are visited in ascending index order.
        for (i, block) in tile.chunks_mut(half << 1).enumerate() {
            if i != 0 {
                t += twiddles.step(first + i);
            }
            let (lo, hi) = block.split_at_mut(half);
            packed_butterfly::<F, INVERSE>(lo, hi, t);
        }
    }
}

/// Finish the bottom stages inside each contiguous tile of `2^log_rows` rows.
///
/// Those stages pair rows less than `2^log_rows` apart, so a tile is closed under all of them
/// and is read once and written once instead of once per stage.
fn deep_tiles<F: TowerLevel, const INVERSE: bool>(
    values: &mut [F],
    width: usize,
    log_rows: usize,
    twiddles: &Twiddles<F>,
) {
    values
        .par_chunks_mut((1 << log_rows) * width)
        .enumerate()
        .for_each(|(index, tile)| {
            tile_stages::<F, INVERSE>(tile, width, log_rows, 0, index, twiddles);
        });
}

/// The map from a staging task to the matrix rows it stages.
///
/// A task owns one butterfly block of the group's widest stage and one run of rows inside it,
/// so with `S` the narrowest stage's row distance its staged row `k` begins at
///
/// ```text
///     R(t, k) = S · ((t / runs) · 2^depth + k) + (t % runs) · 2^log_slab
/// ```
#[derive(Copy, Clone)]
struct StagedRows {
    /// Matrix rows between one staged row and the next, which is `S` above.
    stride: usize,
    /// Staged rows one tile holds.
    rows: usize,
    /// Runs of adjacent rows inside one butterfly block of the group's widest stage.
    runs: usize,
    /// Base-two log of the matrix rows one staged row holds.
    log_slab: usize,
}

impl StagedRows {
    /// The map the `depth` stages ending at stage `top` use, `2^log_slab` rows to a staged row.
    const fn new(top: usize, depth: usize, log_slab: usize) -> Self {
        let stride = 1 << (top + 1 - depth);
        Self {
            stride,
            rows: 1 << depth,
            runs: stride >> log_slab,
            log_slab,
        }
    }

    /// The butterfly block of the group's widest stage that task `task` owns.
    #[inline]
    const fn block(&self, task: usize) -> usize {
        task / self.runs
    }

    /// The first matrix row of staged row `k` of task `task`.
    #[inline]
    const fn row(&self, task: usize, k: usize) -> usize {
        self.stride * (self.block(task) * self.rows + k) + ((task % self.runs) << self.log_slab)
    }
}

/// Run the `depth` stages ending at stage `top` through a staging tile, `2^log_slab` matrix
/// rows to a staged row.
///
/// # Algorithm
///
/// Write `S = 2^(top + 1 - depth)`, the row distance of the group's narrowest stage. For a
/// block `blk` of stage `top` and an offset `r < S`, stage the rows
///
/// ```text
///     R(k) = S · (blk · 2^depth + k) + r ,     k = 0 .. 2^depth
/// ```
///
/// Stage `top - s` moves `2^(depth - 1 - s)` in `k`, so the set is closed under the group, and
/// since `r < S` the global block of `R(k)` is `(blk << s) + (k >> (depth - s))` — what a
/// contiguous tile of index `blk` sees. The pairs `(blk, r)` partition the matrix.
///
/// # Panics
/// Panics if a task's row walk reaches past the end of `values`.
fn fused_group<F: TowerLevel, const INVERSE: bool>(
    values: &mut [F],
    width: usize,
    log_n: usize,
    top: usize,
    depth: usize,
    log_slab: usize,
    twiddles: &Twiddles<F>,
) {
    // A staged row is a run of `2^log_slab` matrix rows.
    //
    // So a run has to fit inside the distance the group's narrowest stage pairs across.
    debug_assert!(depth >= 1 && log_slab + depth <= top + 1);
    debug_assert!(top < log_n);

    let map = StagedRows::new(top, depth, log_slab);
    let rows = 1 << depth;
    let row_len = (1 << log_slab) * width;
    let tasks = 1 << (log_n - depth - log_slab);
    let len = values.len();

    // Invariant: one task per run of rows, per butterfly block of the group's widest stage.
    //
    // Together they reach every row exactly once, which
    // [`tests::a_staging_group_stages_every_row_exactly_once`] walks out over every shape.
    debug_assert_eq!(map.runs << (log_n - 1 - top), tasks);
    debug_assert_eq!(
        tasks * rows * row_len,
        len,
        "tiles do not partition the matrix"
    );

    // Rayon splits a range as far as it likes, and a split is what a staging tile belongs to.
    //
    // So bound the splits to a few per thread, rather than let every task allocate one.
    let min_len = (tasks / (4 * current_num_threads())).max(1);
    let base = DisjointMutPtr::new(values);
    (0..tasks)
        .into_par_iter()
        .with_min_len(min_len)
        .for_each_init(
            // The tile is capacity only, with no initialized elements, and the gather grows it
            // from empty, so no worker zeroes a buffer it is about to overwrite in full.
            || Vec::with_capacity(rows * row_len),
            |tile: &mut Vec<F>, task| {
                // The first element of staged row `k`, from the map [`StagedRows`] defines.
                let row = |k: usize| map.row(task, k) * width;

                // The walk ascends, so bounding its last element bounds all of them, and it
                // runs once per task rather than once per row, which is what a hard check
                // costs. A staged row past the end would be a write past the end of the matrix.
                assert!(
                    row(rows - 1) + row_len <= len,
                    "staged row walk leaves the matrix"
                );

                tile.clear();
                for k in 0..rows {
                    // SAFETY: [`StagedRows::row`] is a bijection onto the matrix rows as the
                    // task and the staged row index range, and a run spans `2^log_slab` rows
                    // from each image, so no two tasks and no two iterations reach one element.
                    //
                    // The bound above puts every element of the walk inside `values`, and the
                    // exclusive borrow the base pointer came from outlives every task.
                    let source = unsafe { base.slice_mut(row(k), row_len) };
                    tile.extend_from_slice(source);
                }

                // Invariant: every element of the tile comes from the walk the gather ran.
                //
                // So nothing below reads an element the gather did not write.
                debug_assert_eq!(tile.len(), rows * row_len, "the gather left the tile short");

                tile_stages::<F, INVERSE>(
                    tile,
                    row_len,
                    depth,
                    top + 1 - depth,
                    map.block(task),
                    twiddles,
                );

                for k in 0..rows {
                    // SAFETY: the same ranges the gather above reached, for the same reason.
                    let target = unsafe { base.slice_mut(row(k), row_len) };
                    target.copy_from_slice(&tile[k * row_len..][..row_len]);
                }
            },
        );
}

/// How wide the two kinds of row tile are for a given element size and matrix width.
struct Schedule {
    /// Base-two log of the rows in one contiguous tile.
    log_tile_rows: usize,
    /// Base-two log of the staged rows one group of stages covers.
    log_staged_rows: usize,
    /// Base-two log of the matrix rows one staged row holds.
    log_slab_rows: usize,
}

impl Schedule {
    /// The tile shapes a matrix gets from the worker count the transform will run on.
    fn new<F: TowerLevel>(width: usize, log_n: usize) -> Self {
        Self::for_workers::<F>(width, log_n, current_num_threads())
    }

    /// Divide each byte budget by the bytes one row occupies, then cut the contiguous tile
    /// back until there is one per worker.
    ///
    /// A tile is also a task, so growing it past one per worker would buy depth no thread is
    /// left to run. Below [`STAGED_WORKERS`] workers nothing is staged at all.
    fn for_workers<F: TowerLevel>(width: usize, log_n: usize, workers: usize) -> Self {
        let row_bytes = core::mem::size_of::<F>() * width;

        // A tile of at least one row, even where a row alone overruns the budget.
        let capacity = log2_floor_usize((DEEP_TILE_BYTES / row_bytes).max(1));
        let log_tile_rows = capacity.min(log_n.saturating_sub(log2_ceil_usize(workers)));

        // Rows shorter than a cache line are staged in runs, so every byte a gather fetches is
        // a byte the tile uses.
        let log_slab_rows =
            log2_ceil_usize(CACHE_LINE_BYTES.div_ceil(row_bytes)).min(log_tile_rows);

        // A staged row is a whole run, so it is the run that the staging budget divides.
        let slab_bytes = row_bytes << log_slab_rows;
        let log_staged_rows = if workers >= STAGED_WORKERS {
            log2_floor_usize((STAGING_TILE_BYTES / slab_bytes).max(1))
        } else {
            0
        };

        Self {
            log_tile_rows,
            log_staged_rows,
            log_slab_rows,
        }
    }
}

/// Run the stages `low` up to, but not including, `high` as one group.
///
/// A group of a single stage is a plain pass: gathering two rows half the matrix apart moves
/// the bytes the pass moves, and copies them twice on top.
fn group_pass<F: TowerLevel, const INVERSE: bool>(
    values: &mut [F],
    width: usize,
    log_n: usize,
    low: usize,
    high: usize,
    log_slab: usize,
    twiddles: &Twiddles<F>,
) {
    if high - low == 1 {
        stage_pass::<F, INVERSE>(values, width, low, twiddles);
    } else {
        fused_group::<F, INVERSE>(
            values,
            width,
            log_n,
            high - 1,
            high - low,
            log_slab,
            twiddles,
        );
    }
}

/// Run every stage of the transform under one tile shape.
///
/// # Algorithm
///
/// Stages below the contiguous tile run inside it. The stages above it are cut into groups of
/// adjacent stages, each short enough that the rows it is closed under fit one staging tile:
///
/// ```text
///     stage   ℓ-1 ................ t+2f  t+2f-1 ..... t+f  t+f-1 ..... t  t-1 ... 0
///             \_____ staged group _____/  \_ staged group _/  \_ group _/  \_ tile _/
/// ```
///
/// The boundaries are counted up from the tile, so a short remainder falls to the top group.
/// The forward direction runs the groups from the top and the tile last, the inverse the other
/// way round.
fn run<F: TowerLevel, const INVERSE: bool>(
    values: &mut [F],
    width: usize,
    log_n: usize,
    twiddles: &Twiddles<F>,
    schedule: &Schedule,
) {
    // The schedule is the caller's, so bring it inside what this matrix can hold: a tile is at
    // most the whole height, and a staged row is a run of matrix rows that must not straddle a
    // butterfly of the narrowest stage above the tile, which pairs rows `2^tile` apart.
    let tile = schedule.log_tile_rows.min(log_n);
    let slab = schedule.log_slab_rows.min(tile);
    let group = if schedule.log_staged_rows >= MIN_FUSED_STAGES {
        schedule.log_staged_rows
    } else {
        1
    };
    let groups = (log_n - tile).div_ceil(group);
    let bounds = |g: usize| {
        let low = tile + g * group;
        (low, (low + group).min(log_n))
    };

    if INVERSE {
        if tile > 0 {
            deep_tiles::<F, INVERSE>(values, width, tile, twiddles);
        }
        for g in 0..groups {
            let (low, high) = bounds(g);
            group_pass::<F, INVERSE>(values, width, log_n, low, high, slab, twiddles);
        }
    } else {
        for g in (0..groups).rev() {
            let (low, high) = bounds(g);
            group_pass::<F, INVERSE>(values, width, log_n, low, high, slab, twiddles);
        }
        if tile > 0 {
            deep_tiles::<F, INVERSE>(values, width, tile, twiddles);
        }
    }
}

/// Transform a matrix in place, in whichever direction the flag selects.
fn transform<F: TowerLevel, const INVERSE: bool>(
    mut mat: RowMajorMatrix<F>,
    shift: F,
) -> RowMajorMatrix<F> {
    let width = mat.width();
    let log_n = log2_strict_usize(mat.height());
    let twiddles = Twiddles::new(log_n, shift);

    // A matrix no larger than a tile is cache-resident for the whole transform, so blocking it
    // would only add bookkeeping to stages already free of memory traffic.
    if core::mem::size_of_val(mat.values.as_slice()) <= DEEP_TILE_BYTES {
        for step in 0..log_n {
            let j = if INVERSE { step } else { log_n - 1 - step };
            stage_pass::<F, INVERSE>(&mut mat.values, width, j, &twiddles);
        }
        return mat;
    }

    let schedule = Schedule::new::<F>(width, log_n);
    run::<F, INVERSE>(&mut mat.values, width, log_n, &twiddles, &schedule);
    mat
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
    use alloc::{format, vec};

    use p3_binary_field::{
        BinaryField8, BinaryField16, BinaryField32, BinaryField64, BinaryField128, Ghash128,
        TowerLevel,
    };
    use p3_field::PrimeCharacteristicRing;
    use p3_matrix::Matrix;
    use p3_matrix::dense::RowMajorMatrix;
    use p3_util::log2_strict_usize;
    use proptest::prelude::*;

    use super::{
        DEEP_TILE_BYTES, LchNtt, MIN_FUSED_STAGES, STAGED_WORKERS, Schedule, StagedRows, Twiddles,
        run, stage_pass,
    };
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
    /// A shape's boundary height is `2^t`, where `2^t · row_bytes` is one tile budget.
    /// So the matrix a boundary sweep builds is four budgets whatever the width, and the cost
    /// is one budget per width rather than one per element.
    ///
    /// This subset keeps the two rows [`HEAVY_LOG_HEIGHTS`] leaves below its boundary — one
    /// element and two of them — plus an odd row, a SIMD-sized row and a wide one.
    const BOUNDARY_WIDTHS: [usize; 5] = [1, 2, 3, 16, 64];

    /// The heights every sweep covers.
    ///
    /// At production budgets only the wide end reaches the blocked schedule, and no staged run
    /// executes. The staged path is pinned by the boundary and production sweeps instead, at
    /// named worker counts.
    const LOG_HEIGHTS: core::ops::RangeInclusive<usize> = 0..=10;

    /// The heights the heavy sweep covers on top of [`LOG_HEIGHTS`].
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

    /// The inverse transform walked the same way, every twiddle from its own block index.
    /// One serial pass, with the stage order reversed and the butterfly undone.
    fn twiddle_walk_intt<F: TowerLevel>(mut mat: RowMajorMatrix<F>, shift: F) -> RowMajorMatrix<F> {
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
    fn stage_by_stage<F: TowerLevel, const INVERSE: bool>(
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
    fn check_schedules_agree<F: TowerLevel>(log_n: usize, width: usize, schedule: &Schedule) {
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
    fn check_matches_naive<F: TowerLevel>(log_n: usize, width: usize, seed: u64, shift: u64) {
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
    fn check_the_twiddle_walk<F: TowerLevel>(log_n: usize, shift_bits: u64) {
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
    fn check_walk_agrees<F: TowerLevel>(log_n: usize, width: usize, shift_bits: u64) {
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
    /// [`transform`] picks its own schedule from the shape and the thread count, so what it
    /// covers moves with the machine. This drives `run` directly instead: the walk then pins the
    /// blocked path whatever the ambient thread count is.
    fn check_the_blocked_walk_agrees<F: TowerLevel>(
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
    fn check_oracle_agrees<F: TowerLevel>(log_n: usize, width: usize, shift_bits: u64) {
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
                        let map = StagedRows::new(top, depth, log_slab);
                        let mut seen = vec![false; 1 << log_n];
                        for task in 0..1usize << (log_n - depth - log_slab) {
                            for k in 0..1usize << depth {
                                let first = map.row(task, k);

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
    fn sweep_against_the_walk<F: TowerLevel>(heights: core::ops::RangeInclusive<usize>) {
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
    fn sweep_across_the_tile_boundaries<F: TowerLevel>() {
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

    /// The narrow shapes whose staging tile stages a run of rows rather than a single row.
    ///
    /// A row shorter than a cache line turns one staged row into a run of matrix rows. The
    /// boundary sweep above already compares these shapes element for element, so this pins
    /// only that they really take the run branch.
    fn check_a_staged_run_of_rows<F: TowerLevel>(log_n: usize) {
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
        // The oracle evaluates `Σ_i d_i · X_i(x)` straight from the product definition and
        // depends on none of D8's identities, so it pins the transform to the novel basis
        // rather than to another consistent network.
        //
        // It costs `O(n² · width)`, which is what caps the heights below; the walk sweeps carry
        // the taller shapes, and together they reach every branch.
        for width in WIDTHS {
            for log_n in 0..=6 {
                for shift_bits in SHIFTS {
                    check_oracle_agrees::<BinaryField32>(log_n, width, shift_bits);
                    check_oracle_agrees::<BinaryField64>(log_n, width, shift_bits);
                    check_oracle_agrees::<BinaryField128>(log_n, width, shift_bits);
                    check_oracle_agrees::<Ghash128>(log_n, width, shift_bits);
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
        // An index of `S_ℓ` past the bit width of `F` asks for a Cantor basis vector.
        //
        // This level does not have one.
        // `BinaryField8` has `2^LOG_BITS = 8` Cantor basis vectors, indices `0..8`.
        let coeffs = matrix::<BinaryField8>((1 << BinaryField8::LOG_BITS) + 1, 1, 0);
        let _ = LchNtt::<BinaryField8>::default().ntt_batch(coeffs);
    }

    /// A height that is not a power of two has no well-defined `ℓ`.
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

        let tower = LchNtt::<BinaryField128>::default().shifted_ntt_batch(tower_coeffs, shift);
        let ghash =
            LchNtt::<Ghash128>::default().shifted_ntt_batch(ghash_coeffs, Ghash128::from(shift));

        // Converting before or after the transform must give the same values.
        for (t, g) in tower.values.iter().zip(&ghash.values) {
            assert_eq!(Ghash128::from(*t), *g);
        }
    }
}
