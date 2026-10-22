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

use p3_commit::{RowSink, hand_over_rows};
use p3_maybe_rayon::prelude::*;
use p3_util::{log2_ceil_usize, log2_floor_usize};

use super::twiddles::Twiddles;
use crate::butterfly::ButterflyField;
use crate::cache::CacheBudget;
use crate::staging::{
    Dispatch, SHARED_CACHE_BYTES, StagedRuns, Store, for_each_staged_tile,
    for_each_staged_tile_into_cosets, for_each_staged_tile_into_cosets_stored, prefault,
};

/// Field elements one butterfly task covers on each side of a block.
///
/// - Stage `j` has blocks of `2^j * width` elements per side.
/// - Blocks wider than this are split, blocks narrower are batched, so every stage has equal tasks.
/// - A few nanoseconds per butterfly makes a task microseconds of work, well above a thread handoff.
pub(crate) const BUTTERFLY_GRAIN: usize = 1 << 10;

/// Bytes of a matrix small enough to run as plain passes.
///
/// It stays in cache throughout, so blocking would only add bookkeeping.
pub(super) const CACHED_MATRIX_BYTES: usize = 128 * 1024;

/// The fewest stages a staging tile must fuse to beat plain passes.
///
/// One stage gathered and scattered moves what one plain pass moves, plus two copies.
pub(super) const MIN_FUSED_STAGES: usize = 2;

/// The fewest bytes one staged run covers.
///
/// - A gather pulls whole cache lines, so a run shorter than a line wastes the rest of it.
/// - The narrowest stage of a staging tile pairs single runs, so a run is also its shortest butterfly.
/// - A butterfly this long reaches the vector kernels of every target, NEON's byte kernel included.
/// - Each doubling of the run costs one fused stage, which a budget of many runs can spare.
const STAGED_RUN_BYTES: usize = 256;

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
    /// The tile shapes a matrix gets on the current thread pool and machine.
    pub(super) fn new<F>(width: usize, log_n: usize) -> Self {
        Self::for_budget::<F>(width, log_n, current_num_threads(), CacheBudget::host())
    }

    /// The tile shapes a matrix gets from a cache budget and a worker count.
    ///
    /// # Algorithm
    ///
    /// A large transform is bound by memory, so its cost is its number of sweeps.
    ///
    /// ```text
    ///     stages:  0 ......... tile ............................ log_n
    ///              |-- tile --|-- group --|-- group --| ... |-- group --|
    /// ```
    ///
    /// - A staging tile fills the private budget, which fixes the stages one group fuses.
    /// - A contiguous tile fills the private budget too.
    /// - It fills the shared budget instead only where that saves a whole sweep.
    /// - A tile is also a task, so it is cut back until every worker has one.
    /// - A lone worker over a matrix the last-level cache holds runs plain passes, which sweep no memory.
    pub(super) fn for_budget<F>(
        width: usize,
        log_n: usize,
        workers: usize,
        budget: CacheBudget,
    ) -> Self {
        let row_bytes = size_of::<F>() * width;

        // Rows one budget holds, as a log, and at least one row where a single row overruns it.
        let fit = |bytes: usize, row: usize| log2_floor_usize((bytes / row).max(1));

        // The deepest tile that still leaves one per worker.
        let parallel = log_n.saturating_sub(log2_ceil_usize(workers));

        // A row shorter than a run is staged in runs, which a tile of `tile` stages may not exceed.
        let run_rows = log2_ceil_usize(STAGED_RUN_BYTES.div_ceil(row_bytes));
        let staged = |tile: usize| {
            let slab = run_rows.min(tile);
            (slab, fit(budget.private, row_bytes << slab))
        };

        // Sweeps the stages above a tile of `tile` stages take, a plain pass each where no group fuses.
        let sweeps = |tile: usize| {
            let above = log_n - tile;
            let (_, depth) = staged(tile);
            if depth >= MIN_FUSED_STAGES {
                above.div_ceil(depth)
            } else {
                above
            }
        };

        // A tile past the private cache spills into the shared one, which only pays for a sweep saved.
        //
        //     512-byte rows, 2^21 rows, 2^10 rows private, 2^12 rows shared:
        //       private tile:  10 stages, then 2 groups (10 + 1 stages)
        //       shared tile:   12 stages, then 1 group  (9 stages)   <- one sweep fewer
        let near = fit(budget.private, row_bytes).min(parallel);
        let far = fit(budget.shared, row_bytes).min(parallel);
        let log_tile_rows = if sweeps(far) < sweeps(near) {
            far
        } else {
            near
        };

        let (log_slab_rows, staged_rows) = staged(log_tile_rows);

        // A plain pass over a matrix half the last-level cache holds moves no memory, only cache.
        //
        //     one worker:      the plain passes stream from that cache, and staging only adds copies
        //     several workers: each plain pass is a dispatch and a barrier, which staging removes
        let resident =
            budget.last_level / 2 >= row_bytes && log_n <= fit(budget.last_level / 2, row_bytes);
        let log_staged_rows = if workers == 1 && resident {
            0
        } else {
            staged_rows
        };

        // The fewest groups cover the stages above the tile, each as shallow as that count allows.
        let above = log_n.saturating_sub(log_tile_rows);
        let log_staged_rows = if log_staged_rows >= MIN_FUSED_STAGES && above > 0 {
            above.div_ceil(above.div_ceil(log_staged_rows))
        } else {
            log_staged_rows
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
///
/// A sink, with the first row the matrix starts at, receives each tile while it is still in cache.
fn deep_tiles<F: ButterflyField, const INVERSE: bool>(
    values: &mut [F],
    width: usize,
    log_rows: usize,
    twiddles: &Twiddles<F>,
    sink: Option<(&RowSink<'_, F>, usize)>,
) {
    values
        .par_chunks_mut((1 << log_rows) * width)
        .enumerate()
        .for_each(|(index, tile)| {
            tile_stages::<F, INVERSE>(tile, width, log_rows, 0, index, twiddles);
            if let Some((sink, first_row)) = sink {
                sink(first_row + (index << log_rows), tile);
            }
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
            deep_tiles::<F, true>(values, width, cut.tile, twiddles, None);
        }
        for g in 0..cut.groups {
            group_pass::<F, true>(values, width, cut.bounds(g), cut.slab, twiddles);
        }
    } else {
        forward_below(values, width, cut, cut.groups, twiddles, None);
    }
}

/// The forward groups below the first `groups` from the bottom, then the tile.
///
/// A sink, with the first row the matrix starts at, receives every finished row once.
fn forward_below<F: ButterflyField>(
    values: &mut [F],
    width: usize,
    cut: Cut,
    groups: usize,
    twiddles: &Twiddles<F>,
    sink: Option<(&RowSink<'_, F>, usize)>,
) {
    for g in (0..groups).rev() {
        group_pass::<F, false>(values, width, cut.bounds(g), cut.slab, twiddles);
    }
    if cut.tile > 0 {
        deep_tiles::<F, false>(values, width, cut.tile, twiddles, sink);
    } else if let Some((sink, first_row)) = sink {
        hand_over_rows(values, width, first_row, sink);
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
/// A separate source stands in for the leading coset, whose prior contents are then never read.
///
/// Each coset then finishes its remaining passes on its own rows.
pub(super) fn run_cosets<F: ButterflyField>(
    values: &mut [F],
    source: Option<&[F]>,
    width: usize,
    log_message: usize,
    twiddles: &[Twiddles<F>],
    schedule: &Schedule,
    sink: Option<&RowSink<'_, F>>,
) {
    let len = width << log_message;
    let cut = Cut::new(schedule, log_message);

    // Every coset the shared pass writes first is faulted in beforehand by a contiguous sweep.
    //
    // That is the leading one too when the message lies elsewhere.
    let fresh = if source.is_some() { 0 } else { len };
    prefault(&mut values[fresh..], F::ZERO);

    if cut.groups == 0 {
        // The contiguous tile covers every stage, so the shared pass is the whole transform.
        //
        // A contiguous tile is a staged tile of consecutive rows.
        let runs = StagedRuns::new(width, 0, cut.tile);
        let tasks = len / (width << cut.tile);
        for_each_staged_tile_into_cosets(
            values,
            source,
            len,
            runs,
            staged_dispatch(tasks),
            |_| {},
            |tile, index, coset| {
                tile_stages::<F, false>(tile, width, cut.tile, 0, index, &twiddles[coset]);
            },
        );
        if let Some(sink) = sink {
            hand_over_rows(values, width, 0, sink);
        }
        return;
    }

    // The top group is the first forward pass, and every stage it runs reads rows far apart.
    let (low, high) = cut.bounds(cut.groups - 1);
    let (top, depth) = (high - 1, high - low);
    let row_len = (1 << cut.slab) * width;
    let tasks = len / (row_len << depth);
    // A matrix past the shared cache is read back from memory by the next pass anyway.
    //
    // Its scatter therefore streams around the cache, which also skips reading each line first.
    let store = if size_of_val(values) > SHARED_CACHE_BYTES {
        Store::Streamed
    } else {
        Store::Cached
    };
    for_each_staged_tile_into_cosets_stored(
        values,
        source,
        len,
        staged_runs(width, top, depth, cut.slab),
        staged_dispatch(tasks),
        store,
        |_| {},
        |tile, block, coset| {
            tile_stages::<F, false>(tile, row_len, depth, low, block, &twiddles[coset]);
        },
    );

    // Every coset finishes the passes below the shared one on its own rows.
    values
        .par_chunks_mut(len)
        .zip(twiddles.par_iter())
        .enumerate()
        .for_each(|(index, (coset, twiddles))| {
            forward_below(
                coset,
                width,
                cut,
                cut.groups - 1,
                twiddles,
                sink.map(|sink| (sink, index << log_message)),
            );
        });
}
