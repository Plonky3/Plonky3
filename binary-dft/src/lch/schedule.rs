//! How the stages of one transform are grouped into passes over memory.
//!
//! A stage pairs rows a power of two apart, so run alone it reads and writes the whole matrix.
//!
//! Two kinds of row set are closed under several adjacent stages at once:
//!
//! - The top stages pair rows far apart, which a staging tile gathers several stages at a time.
//! - The bottom stages pair adjacent rows, which one contiguous tile holds.
//!
//! Each group and the tile pass then cost one read and one write of the matrix.

use p3_maybe_rayon::prelude::*;
use p3_util::{log2_ceil_usize, log2_floor_usize};

use super::twiddles::Twiddles;
use crate::butterfly::ButterflyField;
use crate::staging::{
    Dispatch, StagedRuns, for_each_staged_tile, for_each_staged_tile_into_cosets, prefault,
};

/// Field elements one butterfly task covers on each side of a block.
///
/// - Stage `j` has blocks of `2^j * width` elements per side.
/// - Blocks wider than this are split, blocks narrower are batched, so every stage has equal tasks.
/// - A few nanoseconds per butterfly makes a task microseconds of work, well above a thread handoff.
pub(crate) const BUTTERFLY_GRAIN: usize = 1 << 10;

/// Bytes of one contiguous row tile.
///
/// A worker holds one tile for every stage it is closed under, so it must fit a private L2.
pub(super) const DEEP_TILE_BYTES: usize = 128 * 1024;

/// Bytes of one staging tile.
///
/// Each worker holds one per group, and deeper tiles stop paying past this size.
const STAGING_TILE_BYTES: usize = 64 * 1024;

/// The fewest stages a staging tile must fuse to beat plain passes.
///
/// One stage gathered and scattered moves what one plain pass moves, plus two copies.
pub(super) const MIN_FUSED_STAGES: usize = 2;

/// Workers a transform needs before staging pays.
///
/// - Staging trades a pass per stage for a strided gather and scatter.
/// - Few workers leave memory idle, so the traversals it removes were cheap.
/// - A `no_std` crate cannot read the cache sizes, so the worker count is the signal.
pub(super) const STAGED_WORKERS: usize = 16;

/// The smallest cache line among the supported targets, in bytes.
///
/// A staged row shorter than a line wastes the rest of it, so short rows are staged in runs.
const CACHE_LINE_BYTES: usize = 64;

/// The tile shapes of one matrix, as base-two logarithms.
#[derive(Copy, Clone, Debug)]
pub(super) struct Schedule {
    /// Rows in one contiguous tile.
    pub(super) log_tile_rows: usize,
    /// Staged rows in one staging tile, hence the stages one group fuses.
    pub(super) log_staged_rows: usize,
    /// Matrix rows in one staged row.
    pub(super) log_slab_rows: usize,
}

impl Schedule {
    /// The tile shapes a matrix gets on the current thread pool.
    pub(super) fn new<F>(width: usize, log_n: usize) -> Self {
        Self::for_workers::<F>(width, log_n, current_num_threads())
    }

    /// The tile shapes a matrix gets from its byte budgets and a worker count.
    ///
    /// A tile is also a task, so it is cut back until every worker has one.
    pub(super) fn for_workers<F>(width: usize, log_n: usize, workers: usize) -> Self {
        let row_bytes = size_of::<F>() * width;

        // At least one row per tile, even where a single row overruns the budget.
        let capacity = log2_floor_usize((DEEP_TILE_BYTES / row_bytes).max(1));
        let log_tile_rows = capacity.min(log_n.saturating_sub(log2_ceil_usize(workers)));

        // A row shorter than a cache line is staged in runs, so a gather uses every byte it fetches.
        let log_slab_rows =
            log2_ceil_usize(CACHE_LINE_BYTES.div_ceil(row_bytes)).min(log_tile_rows);

        // The staging budget divides by the run, since a staged row is a whole run.
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

/// A schedule clipped to one height: a contiguous tile, and groups of stages above it.
///
/// - The tile runs stages `0 .. t`.
/// - Group `g` runs stages `t + g * size .. t + (g + 1) * size`.
/// - The boundaries count up from the tile, so a short remainder lands in the top group.
#[derive(Copy, Clone, Debug)]
struct Cut {
    /// Stages the contiguous tile runs.
    tile: usize,
    /// Matrix rows in one staged row, as a base-two logarithm.
    slab: usize,
    /// Stages one full group fuses.
    group: usize,
    /// Groups above the tile.
    groups: usize,
    /// Stages in the whole transform.
    log_n: usize,
}

impl Cut {
    /// Clip a schedule to a transform of `2^log_n` rows.
    const fn new(schedule: &Schedule, log_n: usize) -> Self {
        // A tile holds at most the stages there are.
        let tile = if schedule.log_tile_rows < log_n {
            schedule.log_tile_rows
        } else {
            log_n
        };

        // A staged run must not straddle a butterfly of the narrowest stage above the tile.
        let slab = if schedule.log_slab_rows < tile {
            schedule.log_slab_rows
        } else {
            tile
        };

        // A group too shallow to pay for its gather falls back to one plain pass per stage.
        let group = if schedule.log_staged_rows >= MIN_FUSED_STAGES {
            schedule.log_staged_rows
        } else {
            1
        };

        Self {
            tile,
            slab,
            group,
            groups: (log_n - tile).div_ceil(group),
            log_n,
        }
    }

    /// The stages `low .. high` group `g` runs, counting up from the tile.
    const fn bounds(&self, g: usize) -> (usize, usize) {
        let low = self.tile + g * self.group;
        let high = if low + self.group < self.log_n {
            low + self.group
        } else {
            self.log_n
        };
        (low, high)
    }
}

/// One stage, as a single pass over every row.
pub(super) fn stage_pass<F: ButterflyField, const INVERSE: bool>(
    values: &mut [F],
    width: usize,
    j: usize,
    twiddles: &Twiddles<F>,
) {
    let half = (1 << j) * width;
    let per_task = (BUTTERFLY_GRAIN / half).max(1);
    let task_len = per_task * (half << 1);
    let task = |(task, group): (usize, &mut [F])| {
        // A task seeds its twiddle at its own first block, then steps from block to block.
        let first = task * per_task;
        let mut t = twiddles.at(j, first);

        // Invariant: blocks are visited in ascending order, which the stepping relies on.
        for (i, block) in group.chunks_mut(half << 1).enumerate() {
            if i != 0 {
                t += twiddles.step(first + i);
            }
            let (lo, hi) = block.split_at_mut(half);

            // Pairs are independent, so a block wider than the grain is split across workers.
            if half <= BUTTERFLY_GRAIN {
                F::butterfly::<INVERSE>(lo, hi, t);
            } else {
                lo.par_chunks_mut(BUTTERFLY_GRAIN)
                    .zip(hi.par_chunks_mut(BUTTERFLY_GRAIN))
                    .for_each(|(lo, hi)| F::butterfly::<INVERSE>(lo, hi, t));
            }
        }
    };

    // A pass of one task has nothing to spread, and dispatching it would cost a handoff.
    if values.len() > task_len {
        values.par_chunks_mut(task_len).enumerate().for_each(task);
    } else {
        values.chunks_mut(task_len).enumerate().for_each(task);
    }
}

/// Run the `log_rows` adjacent stages a tile of `2^log_rows` rows is closed under.
///
/// - The tile's stages are global stages `stage_base .. stage_base + log_rows`.
/// - At its stage `jj`, tile row `q` sits in global block `(index << (log_rows - 1 - jj)) + (q >> (jj + 1))`.
/// - The second term is what the local walk counts, so only the tile index is passed in.
///
/// Forward runs the widest stage first, inverse the narrowest.
fn tile_stages<F: ButterflyField, const INVERSE: bool>(
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

        // The tile's blocks continue the global block numbering from here.
        let first = index << (log_rows - 1 - jj);
        let mut t = twiddles.at(stage_base + jj, first);

        // Invariant: blocks are visited in ascending order, which the stepping relies on.
        for (i, block) in tile.chunks_mut(half << 1).enumerate() {
            if i != 0 {
                t += twiddles.step(first + i);
            }
            let (lo, hi) = block.split_at_mut(half);
            F::butterfly::<INVERSE>(lo, hi, t);
        }
    }
}

/// Run the bottom `log_rows` stages inside each contiguous tile of `2^log_rows` rows.
///
/// Those stages pair rows closer than a tile, so each tile is read once and written once.
fn deep_tiles<F: ButterflyField, const INVERSE: bool>(
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

/// The runs that the `depth` stages ending at stage `top` gather, `2^log_slab` rows to a run.
///
/// With `S = 2^(top + 1 - depth)` the row distance of the group's narrowest stage, task `t` stages
///
/// ```text
///     R(t, k) = S * ((t / runs) * 2^depth + k) + (t % runs) * 2^log_slab ,   runs = S / 2^log_slab
/// ```
pub(super) const fn staged_runs(
    width: usize,
    top: usize,
    depth: usize,
    log_slab: usize,
) -> StagedRuns {
    StagedRuns::new((1 << log_slab) * width, top + 1 - depth - log_slab, depth)
}

/// How a staging pass of `tasks` tiles spreads over the pool.
///
/// A split is what one staging buffer serves, so a few splits per thread bound the buffers.
fn staged_dispatch(tasks: usize) -> Dispatch {
    Dispatch::Parallel {
        min_len: (tasks / (4 * current_num_threads())).max(1),
    }
}

/// Run the `depth` stages ending at stage `top` through a staging tile, `2^log_slab` rows to a staged row.
///
/// # Algorithm
///
/// Write `S = 2^(top + 1 - depth)` for the row distance of the group's narrowest stage.
///
/// For a block `b` of stage `top` and an offset `r < S`, one tile stages the rows
///
/// ```text
///     R(k) = S * (b * 2^depth + k) + r ,     k = 0 .. 2^depth
/// ```
///
/// - Stage `top - s` moves `2^(depth - 1 - s)` in `k`, so the set is closed under the group.
/// - Since `r < S`, the global block of `R(k)` is `(b << s) + (k >> (depth - s))`.
/// - That is what a contiguous tile of index `b` sees, so the tile runs as an ordinary network.
/// - The pairs `(b, r)` partition the matrix.
fn fused_group<F: ButterflyField, const INVERSE: bool>(
    values: &mut [F],
    width: usize,
    top: usize,
    depth: usize,
    log_slab: usize,
    twiddles: &Twiddles<F>,
) {
    // A staged run must fit inside the distance the group's narrowest stage pairs across.
    debug_assert!(depth >= 1 && log_slab + depth <= top + 1);

    let row_len = (1 << log_slab) * width;
    let tasks = values.len() / (row_len << depth);
    for_each_staged_tile(
        values,
        staged_runs(width, top, depth, log_slab),
        staged_dispatch(tasks),
        |tile, block| {
            tile_stages::<F, INVERSE>(tile, row_len, depth, top + 1 - depth, block, twiddles);
        },
    );
}

/// Run the stages `low .. high` as one pass.
///
/// A single stage is a plain pass, since gathering two rows half the matrix apart saves nothing.
fn group_pass<F: ButterflyField, const INVERSE: bool>(
    values: &mut [F],
    width: usize,
    (low, high): (usize, usize),
    log_slab: usize,
    twiddles: &Twiddles<F>,
) {
    if high - low == 1 {
        stage_pass::<F, INVERSE>(values, width, low, twiddles);
    } else {
        fused_group::<F, INVERSE>(values, width, high - 1, high - low, log_slab, twiddles);
    }
}

/// Run every stage of a transform of `2^log_n` rows under one schedule, in place.
///
/// Forward runs the groups from the top and the tile last, inverse the other way round.
pub(super) fn run<F: ButterflyField, const INVERSE: bool>(
    values: &mut [F],
    width: usize,
    log_n: usize,
    twiddles: &Twiddles<F>,
    schedule: &Schedule,
) {
    let cut = Cut::new(schedule, log_n);
    if INVERSE {
        if cut.tile > 0 {
            deep_tiles::<F, true>(values, width, cut.tile, twiddles);
        }
        for g in 0..cut.groups {
            group_pass::<F, true>(values, width, cut.bounds(g), cut.slab, twiddles);
        }
    } else {
        forward_below(values, width, cut, cut.groups, twiddles);
    }
}

/// The forward groups below the first `groups` from the bottom, then the tile.
fn forward_below<F: ButterflyField>(
    values: &mut [F],
    width: usize,
    cut: Cut,
    groups: usize,
    twiddles: &Twiddles<F>,
) {
    for g in (0..groups).rev() {
        group_pass::<F, false>(values, width, cut.bounds(g), cut.slab, twiddles);
    }
    if cut.tile > 0 {
        deep_tiles::<F, false>(values, width, cut.tile, twiddles);
    }
}

/// Forward-transform every coset of a zero-padded message, reading the message once for all of them.
///
/// # Algorithm
///
/// `values` holds the message in its leading coset, then zeros.
///
/// Every coset starts from the message, so the first pass of all of them shares one gather:
///
/// - A tile of the message is gathered once.
/// - Each coset runs the pass on its own copy of the tile, with its own twiddles, in cache.
/// - Each copy is scattered into that coset's own rows.
///
/// No pass copies the message into the cosets before transforming them.
///
/// Each coset then finishes its remaining passes on its own rows.
pub(super) fn run_cosets<F: ButterflyField>(
    values: &mut [F],
    width: usize,
    log_message: usize,
    twiddles: &[Twiddles<F>],
    schedule: &Schedule,
) {
    let len = width << log_message;
    let cut = Cut::new(schedule, log_message);

    // The cosets past the leading one hold zeros, and the shared pass is their first write.
    //
    // A contiguous sweep faults their pages in beforehand.
    prefault(&mut values[len..], F::ZERO);

    if cut.groups == 0 {
        // The contiguous tile covers every stage, so the shared pass is the whole transform.
        //
        // A contiguous tile is a staged tile of consecutive rows.
        let runs = StagedRuns::new(width, 0, cut.tile);
        let tasks = len / (width << cut.tile);
        for_each_staged_tile_into_cosets(
            values,
            None,
            len,
            runs,
            staged_dispatch(tasks),
            |_| {},
            |tile, index, coset| {
                tile_stages::<F, false>(tile, width, cut.tile, 0, index, &twiddles[coset]);
            },
        );
        return;
    }

    // The top group is the first forward pass, and every stage it runs reads rows far apart.
    let (low, high) = cut.bounds(cut.groups - 1);
    let (top, depth) = (high - 1, high - low);
    let row_len = (1 << cut.slab) * width;
    let tasks = len / (row_len << depth);
    for_each_staged_tile_into_cosets(
        values,
        None,
        len,
        staged_runs(width, top, depth, cut.slab),
        staged_dispatch(tasks),
        |_| {},
        |tile, block, coset| {
            tile_stages::<F, false>(tile, row_len, depth, low, block, &twiddles[coset]);
        },
    );

    // Every coset finishes the passes below the shared one on its own rows.
    values
        .par_chunks_mut(len)
        .zip(twiddles.par_iter())
        .for_each(|(coset, twiddles)| {
            forward_below(coset, width, cut, cut.groups - 1, twiddles);
        });
}
