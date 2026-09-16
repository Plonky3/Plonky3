//! The Lin–Chung–Han transform carried out in the polynomial basis of `GF(2^128)`.

use alloc::vec::Vec;

use p3_binary_field::{BinaryField128, TowerLevel, poly_basis};
use p3_field::PrimeCharacteristicRing;
use p3_matrix::Matrix;
use p3_matrix::dense::RowMajorMatrix;
use p3_maybe_rayon::prelude::*;
use p3_util::{log2_ceil_usize, log2_floor_usize, log2_strict_usize};

use crate::domain::domain_point;
use crate::lch::BUTTERFLY_GRAIN;
use crate::traits::AdditiveNtt;

/// [`LchNtt`](crate::LchNtt) over `BinaryField128`, with the data held in the polynomial basis throughout.
///
/// A tower-basis product converts both operands into the polynomial basis and the result back,
/// sixteen dependent table lookups apiece, which is most of what a butterfly costs. Converting
/// the whole matrix once on the way in and once on the way out pays `2` conversions per element
/// instead of `3ℓ/2`, and every twiddle multiply in between is a bare carryless multiply and a
/// reduction. Additions are `XOR` in both bases, so they are unaffected.
///
/// Without a carryless-multiply instruction that product is a bit-serial loop and slower than
/// the tower arithmetic it replaces, so on such a target the transform runs in the tower basis
/// instead, using typed subfield multiplication when the twiddle permits it.
/// The choice is a constant and only one arm survives compilation.
#[derive(Clone, Debug, Default)]
pub struct PolyBasisNtt {
    tower: crate::tower::TowerNtt,
}

/// Stage shifts and the XOR increments between consecutive block twiddles.
struct Twiddles {
    basis: [u128; usize::BITS as usize],
    deltas: [u128; usize::BITS as usize],
    shifts: [u128; usize::BITS as usize],
}

impl Twiddles {
    fn new(log_n: usize, shift: BinaryField128) -> Self {
        let mut result = Self {
            basis: [0; usize::BITS as usize],
            deltas: [0; usize::BITS as usize],
            shifts: [0; usize::BITS as usize],
        };
        let mut delta = 0;
        let mut base = poly_basis::from_tower(shift);
        for j in 0..log_n {
            result.basis[j] = poly_basis::from_tower(BinaryField128::cantor_basis(j + 1));
            delta ^= result.basis[j];
            result.deltas[j] = delta;
            result.shifts[j] = base;
            base = poly_basis::square(base) ^ base;
        }
        result
    }

    const fn at(&self, stage: usize, mut block: usize) -> u128 {
        let mut t = self.shifts[stage];
        while block != 0 {
            t ^= self.basis[block.trailing_zeros() as usize];
            block &= block - 1;
        }
        t
    }
}

/// Schedule enough butterfly-sized work per worker to amortize parallel dispatch.
// The serial dependency exposes a `const` thread-count query, while the parallel
// implementation cannot; keep this shared scheduler callable in both configurations.
#[allow(clippy::missing_const_for_fn)]
fn use_parallel(elements: usize) -> bool {
    let threads = p3_maybe_rayon::prelude::current_num_threads();
    threads > 1 && elements >= 2 * BUTTERFLY_GRAIN * threads
}

fn for_chunks(
    values: &mut [u128],
    chunk_len: usize,
    stages: usize,
    operation: impl Fn((usize, &mut [u128])) + Send + Sync,
) {
    if values.len() > chunk_len && use_parallel(values.len().saturating_mul(stages.max(1))) {
        values
            .par_chunks_mut(chunk_len)
            .enumerate()
            .for_each(operation);
    } else {
        values.chunks_mut(chunk_len).enumerate().for_each(operation);
    }
}

/// A change of basis applied to a whole run of elements at once.
///
/// The kernel behind it converts several elements together where the target allows.
///
/// A run therefore has to reach it unbroken, which is why this is not a per-element map.
type Conversion = fn(&mut [u128]);

/// Rewrites tower-basis bit patterns as polynomial coordinates.
const INTO_POLY: Conversion = poly_basis::from_tower_slice;

/// Rewrites polynomial coordinates as tower-basis bit patterns.
const INTO_TOWER: Conversion = poly_basis::to_tower_slice;

/// Change the basis of a whole matrix, in a pass of its own.
fn convert(values: &mut [u128], conversion: Conversion) {
    // Basis conversion does several dependent lookups per element, more work than
    // a butterfly, so it amortizes dispatch at a smaller byte volume.
    //
    // A task is a whole grain, so every task still reaches the blocked kernel.
    if use_parallel(values.len().saturating_mul(4)) {
        values.par_chunks_mut(BUTTERFLY_GRAIN).for_each(conversion);
    } else {
        conversion(values);
    }
}

/// Apply a change of basis to a tile the schedule is already holding in cache.
fn convert_tile(tile: &mut [u128], conversion: Conversion) {
    conversion(tile);
}

/// Where the two basis conversions ride, instead of taking a pass over the matrix each.
///
/// A conversion is a pure per-element map, so it commutes with every butterfly ordering.
///
/// Applied where the schedule already holds the element in cache it costs the lookups alone,
/// and no memory traffic at all.
#[derive(Copy, Clone, Debug)]
struct Fold {
    /// Map out of the tower basis where the transform first reads each element.
    entry: bool,
    /// Map back into the tower basis where the transform last writes each element.
    exit: bool,
}

impl Fold {
    /// Take tower-basis values and hand tower-basis values back.
    const BOTH: Self = Self {
        entry: true,
        exit: true,
    };
    /// Take tower-basis values and leave the result in polynomial coordinates.
    const ENTRY: Self = Self {
        entry: true,
        exit: false,
    };
    /// Take polynomial coordinates and hand tower-basis values back.
    const EXIT: Self = Self {
        entry: false,
        exit: true,
    };
}

/// A stage uses bounded chunks, each starting from its own independently indexed twiddle.
fn stage(values: &mut [u128], half: usize, j: usize, twiddles: &Twiddles, inverse: bool) {
    if !use_parallel(values.len()) {
        local_stage(values, half, j, twiddles, inverse, 0);
        return;
    }
    let blocks_per_chunk = (BUTTERFLY_GRAIN / (half << 1)).max(1);
    values
        .par_chunks_mut((half << 1) * blocks_per_chunk)
        .enumerate()
        .for_each(|(chunk_index, chunk)| {
            let first = chunk_index * blocks_per_chunk;
            let mut t = twiddles.at(j, first);
            for (index, block) in chunk.chunks_mut(half << 1).enumerate() {
                let (lo, hi) = block.split_at_mut(half);
                let butterfly = |lo: &mut [u128], hi: &mut [u128]| {
                    if inverse {
                        poly_basis::butterfly_inverse(lo, hi, t);
                    } else {
                        poly_basis::butterfly_forward(lo, hi, t);
                    }
                };
                if half <= BUTTERFLY_GRAIN {
                    butterfly(lo, hi);
                } else {
                    lo.par_chunks_mut(BUTTERFLY_GRAIN)
                        .zip(hi.par_chunks_mut(BUTTERFLY_GRAIN))
                        .for_each(|(lo, hi)| butterfly(lo, hi));
                }
                t ^= twiddles.deltas[(first + index).trailing_ones() as usize];
            }
        });
}

// A conservative tile budget; the row count scales with the element size and matrix width.
const TILE_BYTES: usize = 32 * 1024;

/// Bytes one worker's staging tile may occupy.
///
/// The tile is streamed rather than randomly addressed, so it need not fit in L1.
/// What it must not do is spill out of the private cache level below the shared one.
///
/// A sweep of the fusion depth at width 16 puts the optimum at 64 KiB, which is exactly the
/// per-core private cache size of the machine the sweep ran on. How many stages that fuses is
/// the run length divided into it, so it moves with the height rather than being fixed: six
/// stages at `2^16` rows, seven at `2^20`, eight at `2^22`.
///
/// A 1 MiB tile fuses four stages more and takes half as long again.
const STAGING_BYTES: usize = 64 * 1024;

/// Bytes a gathered run of adjacent rows must cover for the staging to be worth running.
///
/// A gather and a scatter address runs a power of two apart, so each address pulls and pushes
/// whole cache lines whatever the run length. A run shorter than a line therefore moves a
/// line's worth of traffic to carry a fraction of one:
///
/// ```text
///     16 bytes per run   one quarter of a 64-byte line is useful
///     64 bytes per run   a whole line is useful
/// ```
///
/// This is the smallest line size the supported targets have, so it is the point past which
/// no target wastes more than half of a line.
const STAGED_LINE_BYTES: usize = 64;

/// Bytes a gathered run covers where lengthening it is free.
///
/// A line is enough for the gather to use every byte it moves, but a stride that jumps every
/// line still leaves the prefetcher nothing to follow.
///
/// A sweep over runs of 16 to 4096 bytes, at equal element volume on 32 threads:
///
/// ```text
///     run bytes      16     64    256   1024   2048   4096
///     width 1      7.28   6.32   5.93   5.21   5.76   5.62   ms at 2^22 rows
///     width 4      5.71   5.74   5.23   4.40   4.88   4.94   ms at 2^20 rows
///     width 16     4.91   4.73   4.96   4.04   4.72   4.82   ms at 2^18 rows
/// ```
///
/// A run cannot be shorter than one row, so every cell left of a width's own row length holds
/// the same plan measured again: width 16's leading three are one plan three times, and their
/// spread of 4.73 to 4.96 puts the noise floor near 5%, which is what the rest of the table
/// has to be read against.
///
/// The knee is at 1024 bytes, and the rise past it is the depth the tile gives up.
/// [`Plan::new`] reads this as a target and not as a floor, because that depth is not free: a
/// run grows only while the shorter tile still takes as few traversals of the matrix.
const STAGED_RUN_BYTES: usize = 1024;

/// Workers a row below [`STAGED_LINE_BYTES`] needs before a tile is worth gathering into.
///
/// Staging trades traversals of the matrix for a strided gather and scatter.
/// One worker issues its traversals far below what memory can serve, so they cost little and
/// the copy has nothing to pay for it.
/// Several workers traverse at once against a supply that does not grow, so the traversals
/// become what the schedule waits on.
///
/// Whether a traversal reaches memory at all depends on the last-level cache, which a `no_std`
/// crate cannot read. The worker count is the available signal, and it is what multiplies
/// demand against that cache.
///
/// Width-1 transforms, as a ratio to the same transform with staging off:
///
/// ```text
///     matrix        4 MiB   8 MiB   16 MiB   32 MiB   64 MiB   128 MiB
///      1 worker      1.03    1.06     1.13     1.26     1.08      0.99
///      2 workers     0.97    1.02     1.07     1.16     0.76      0.90
///      4 workers     0.99    0.90     0.89     0.96     0.81      0.65
///      8 workers     0.88    0.87     0.79     0.79     0.78      0.49
///     32 workers     0.49    0.53     0.58     0.65     0.59      0.37
/// ```
///
/// One and two workers share a band from 8 to 32 MiB where the copy costs more than the
/// traversals it removes, and at two workers that band reaches 16%. Four is the first count
/// with no such band, and every count above it only wins by more.
///
/// # Scope
///
/// The count gates only rows too narrow for a run of one row to have reached:
///
/// ```text
///     row >= STAGED_LINE_BYTES   gathered at any worker count
///     row <  STAGED_LINE_BYTES   gathered from STAGED_WORKERS workers up
/// ```
///
/// Two workers on a 64 MiB matrix give up a quarter here, and a lone worker whose narrow
/// matrix does outgrow its cache gives up a win of its own. Those are wins forgone rather than
/// costs added, which is the safe direction for a signal this coarse.
const STAGED_WORKERS: usize = 4;

/// The shape of one transform, and the two places its stage sequence is cut.
///
/// A stage pairs rows a power of two apart, so the whole matrix has to be traversed once per
/// stage unless a set of rows closed under several stages can be brought into cache.
///
/// There are two such sets, and one cut point each:
///
/// ```text
///     stages log_n-1 .. local+leftover   rows far apart, gathered into a staging tile
///     stages local+leftover-1 .. local   rows far apart, one plain full pass each
///     stages local-1 .. 0                rows already adjacent, so a contiguous tile holds them
/// ```
///
/// The middle band holds the stages a staging tile would not pay for.
/// It is one stage wide at most, unless the staging tile cannot hold two runs.
///
/// The count of contiguous-tile stages never exceeds the count of stages there are.
#[derive(Copy, Clone, Debug)]
struct Plan {
    /// Elements per row.
    width: usize,
    /// Base-two logarithm of the row count.
    log_n: usize,
    /// Bottom stages that run to completion inside one contiguous tile of rows.
    local: usize,
    /// Base-two logarithm of the adjacent rows one staging gather moves per strided address.
    ///
    /// At least [`STAGED_LINE_BYTES`] of run, and up to [`STAGED_RUN_BYTES`] of it where the
    /// depth that costs is free.
    log_block: usize,
    /// Long-stride stages that one staging tile fuses into a single pass over the matrix.
    depth: usize,
}

impl Plan {
    /// Stages above the contiguous tile that run as plain full passes, for a cut of `above`
    /// stages into groups of `depth`.
    ///
    /// Fusing a single stage would move the same bytes the stage moves on its own, plus the
    /// copy in and out of the staging tile, so one stage is left over rather than fused.
    /// A staging tile too narrow to hold two runs leaves every stage over.
    const fn leftover_stages(above: usize, depth: usize) -> usize {
        if depth < 2 {
            above
        } else if above % depth == 1 {
            1
        } else {
            0
        }
    }

    /// The stages each staging group fuses, listed from the top stage downwards.
    ///
    /// The walk stops where the leftover stages begin, so a short group lands at the bottom
    /// of the staged band rather than the top. Both directions and the traversal budget read
    /// the cut from here, so none of the three can drift from the others.
    fn groups(above: usize, depth: usize) -> impl Iterator<Item = usize> {
        let leftover = Self::leftover_stages(above, depth);
        let mut remaining = above;
        core::iter::from_fn(move || {
            (remaining > leftover).then(|| {
                let take = depth.min(remaining - leftover);
                remaining -= take;
                take
            })
        })
    }

    /// Full traversals of the matrix the stages above the contiguous tile take.
    /// Each fused group is one traversal, and each leftover stage is one of its own.
    fn traversals(above: usize, depth: usize) -> usize {
        Self::groups(above, depth).count() + Self::leftover_stages(above, depth)
    }

    /// The cut points a matrix of this shape gets, as the schedule that will run it sees them.
    fn new(width: usize, log_n: usize) -> Self {
        Self::for_workers(width, log_n, current_num_threads())
    }

    /// The cut points a matrix of this shape gets, from the three memory budgets.
    ///
    /// The staging tile has a fixed byte budget, so doubling the run halves the rows it holds:
    /// the run length and the fused depth trade one for one. A run below a cache line is never
    /// worth gathering, so that length is taken first, and from there it grows towards
    /// [`STAGED_RUN_BYTES`] only while the shorter tile still takes as few traversals.
    ///
    /// A row under [`STAGED_LINE_BYTES`] is gathered from [`STAGED_WORKERS`] workers up, and
    /// below that every stage above the contiguous tile runs as a plain pass.
    fn for_workers(width: usize, log_n: usize, workers: usize) -> Self {
        let row = core::mem::size_of::<u128>() * width;
        let local = log2_floor_usize((TILE_BYTES / row).max(1)).min(log_n);
        let above = log_n - local;
        // Stages one tile fuses, with a run of `2^log_block` adjacent rows as its row.
        let depth_at =
            |log_block: usize| log2_floor_usize((STAGING_BYTES / (row << log_block)).max(1));
        let floor = log2_ceil_usize(STAGED_LINE_BYTES.div_ceil(row));
        let target = log2_ceil_usize(STAGED_RUN_BYTES.div_ceil(row)).max(floor);
        // The shortest admissible run sets the traversal count no longer run may exceed.
        let budget = Self::traversals(above, depth_at(floor));
        let log_block = (floor..=target)
            .rev()
            .find(|&log_block| Self::traversals(above, depth_at(log_block)) <= budget)
            .unwrap_or(floor);
        Self {
            width,
            log_n,
            local,
            log_block,
            depth: if row >= STAGED_LINE_BYTES || workers >= STAGED_WORKERS {
                depth_at(log_block)
            } else {
                0
            },
        }
    }

    /// Stages above the contiguous tile this plan runs as plain full passes.
    const fn leftover(&self) -> usize {
        Self::leftover_stages(self.log_n - self.local, self.depth)
    }

    /// The stages each of this plan's staging groups fuses, from the top stage downwards.
    fn group_sizes(&self) -> impl Iterator<Item = usize> {
        Self::groups(self.log_n - self.local, self.depth)
    }
}

/// A tile stays on one worker across its adjacent stages.
fn local_stage(
    values: &mut [u128],
    half: usize,
    j: usize,
    twiddles: &Twiddles,
    inverse: bool,
    first: usize,
) {
    let mut t = twiddles.at(j, first);
    for (index, block) in values.chunks_mut(half << 1).enumerate() {
        let (lo, hi) = block.split_at_mut(half);
        if inverse {
            poly_basis::butterfly_inverse(lo, hi, t);
        } else {
            poly_basis::butterfly_forward(lo, hi, t);
        }
        t ^= twiddles.deltas[(first + index).trailing_ones() as usize];
    }
}

/// Run the `depth` stages that a tile of `2^depth` consecutive rows of `row` elements holds.
///
/// The tile is a radix-2 network on its rows.
///
/// Sub-layer `s` pairs rows `2^(depth-1-s)` apart and splits the tile into `2^s` blocks of
/// `2^(depth-s)` rows, each block carrying one twiddle.
///
/// Globally those blocks are the blocks `block * 2^s + g`, `g = 0 .. 2^s`, of stage
/// `top - 1 - s`, so the twiddle walk starts at `block << s`.
///
/// A row of the tile is a row of the matrix for the contiguous phase and a run of adjacent
/// matrix rows for the staging phase, which is why the row length is a parameter.
fn tile_stages(
    tile: &mut [u128],
    row: usize,
    depth: usize,
    top: usize,
    twiddles: &Twiddles,
    inverse: bool,
    block: usize,
) {
    for k in 0..depth {
        // The forward direction runs the widest sub-layer first, the inverse the narrowest.
        let s = if inverse { depth - 1 - k } else { k };
        local_stage(
            tile,
            (1 << (depth - 1 - s)) * row,
            top - 1 - s,
            twiddles,
            inverse,
            block << s,
        );
    }
}

/// Complete the stages confined to one cache-sized set of rows before leaving it.
fn local_stages(values: &mut [u128], plan: Plan, twiddles: &Twiddles, inverse: bool, fold: Fold) {
    let Plan { width, local, .. } = plan;
    let tile_len = (1 << local) * width;
    for_chunks(values, tile_len, local, |(index, tile)| {
        // The tile is the first read of every element it holds when it runs before every
        // other stage, and the last write when it runs after them.
        if fold.entry {
            convert_tile(tile, INTO_POLY);
        }
        tile_stages(tile, width, local, local, twiddles, inverse, index);
        if fold.exit {
            convert_tile(tile, INTO_TOWER);
        }
    });
}

/// A raw handle to the matrix, so tasks owning runs spaced apart can run side by side.
///
/// A run is the `2^log_block` adjacent matrix rows one strided address moves, which is one row
/// of the reshaped matrix the staging reads. Slice splitters cut a slice into contiguous pieces
/// only, and the runs one staging tile gathers are a power of two apart, so the tasks share
/// this handle and address their own runs through it.
///
/// # Safety
/// The run sets two live tasks address must be disjoint, and the exclusive borrow the base
/// pointer comes from must outlive every task.
#[derive(Copy, Clone)]
struct Rows {
    /// First element of the matrix.
    base: *mut u128,
    /// Elements in one run.
    run: usize,
    /// Runs in the matrix.
    count: usize,
}

// SAFETY: the handle is a pointer and two lengths, with no interior mutability and no `Drop`,
// so sending or sharing it moves no data.
//
// The only caller derives the run index of every task from a bijection onto the run range,
// which is what makes concurrent use race-free. A run is a block of adjacent rows of one fixed
// length, so disjoint run sets are disjoint element ranges.
unsafe impl Send for Rows {}
// SAFETY: see the `Send` implementation.
unsafe impl Sync for Rows {}

impl Rows {
    /// Check that a walk of `runs` runs from `first` in steps of `stride` stays in the matrix.
    ///
    /// The walk is increasing, so bounding its last run bounds all of them.
    /// This runs once per tile rather than once per run, which is why it is a hard check and
    /// not a debug one.
    ///
    /// A run index past the end would otherwise be a write past the end of the matrix.
    ///
    /// # Panics
    /// Panics if the last run of the walk is at or beyond the run count.
    fn check(&self, first: usize, stride: usize, runs: usize) {
        assert!(
            runs == 0 || first + (runs - 1) * stride < self.count,
            "staged row walk leaves the matrix"
        );
    }

    /// Copy the runs `first`, `first + stride`, ... into consecutive rows of the tile.
    ///
    /// The tile is emptied first and then grown one run at a time.
    /// So it holds no element the walk did not write.
    ///
    /// And a worker never has to zero a tile it is about to overwrite in full.
    ///
    /// A tile whose capacity already covers the walk grows without reallocating.
    ///
    /// # Safety
    /// No other live task may address any of the runs the walk names.
    unsafe fn gather(&self, first: usize, stride: usize, runs: usize, tile: &mut Vec<u128>) {
        self.check(first, stride, runs);
        tile.clear();
        for k in 0..runs {
            // SAFETY: the bound above puts every run of the walk inside the matrix, and the
            // exclusive borrow the base pointer came from outlives the task.
            //
            // No other live task addresses this run, so nothing can write it during the read.
            let run = unsafe {
                core::slice::from_raw_parts(
                    self.base.add((first + k * stride) * self.run),
                    self.run,
                )
            };
            tile.extend_from_slice(run);
        }
    }

    /// Write consecutive rows of the tile back over the runs they were gathered from.
    ///
    /// # Safety
    /// No other live task may address any of the runs the walk names.
    unsafe fn scatter(&self, first: usize, stride: usize, tile: &[u128]) {
        let runs = tile.chunks_exact(self.run);
        self.check(first, stride, runs.len());
        for (k, run) in runs.enumerate() {
            // SAFETY: as in the gather, with the direction of the copy reversed.
            unsafe {
                core::ptr::copy_nonoverlapping(
                    run.as_ptr(),
                    self.base.add((first + k * stride) * self.run),
                    self.run,
                );
            }
        }
    }
}

/// Run stages `top - 1` down to `top - depth` through one staging tile per worker.
/// Each of those stages pairs rows far apart, so on its own it reads and writes the whole
/// matrix.
///
/// The rows they touch split into small sets that are closed under all of them, so gathering
/// one such set into a contiguous tile runs the whole group inside the cache.
///
/// The matrix is then read once and written once per group instead of once per stage.
///
/// # Algorithm
///
/// A gather addresses runs of `L = 2^log_block` adjacent rows, so one strided address moves a
/// contiguous stretch rather than a fragment of one row.
/// That is exactly the matrix reread as rows of `L * width` elements:
///
/// ```text
///     width * 2^log_n  =  (L * width) * 2^(log_n - log_block)
/// ```
///
/// Stage `j` pairs rows `2^j` apart, which is `2^(j - log_block)` reshaped rows apart.
/// So for `j >= log_block` it is stage `j - log_block` of the reshaped matrix.
///
/// A run must fit the stride the walk takes, hence `log_block <= top - depth`.
///
/// Writing `S = 2^(top - log_block - depth)` with `offset < S`, one closed set is:
///
/// ```text
///     row(k) = block * 2^(top-log_block) + offset + k * S ,     k = 0 .. 2^depth
/// ```
///
/// Stage `top-1-s` pairs those `2^(depth-1-s)` apart in `k`.
/// That stays inside the set for every `s < depth`, so the set is closed.
///
/// # Twiddles
///
/// A sub-layer needs the twiddle of the global block its pair lies in, and the two readings
/// name the same block:
///
/// ```text
///     matrix row    r                lies in block  r >> (top - s)
///     reshaped row  r >> log_block   lies in block  r >> log_block >> (top - log_block - s)
/// ```
///
/// A shift by `log_block` then by `top - log_block - s` is a shift by `top - s`.
/// So the `L` matrix rows of one run share one twiddle at every sub-layer, which is what
/// makes the reshape exact.
///
/// Sub-layer `s` then sees block index `block * 2^s + (k >> (depth - s))`.
/// Writing `k = q * 2^(depth-s) + r` leaves `offset + r * S < 2^(top-log_block-s)` below the
/// shift, so only `q` survives it.
///
/// That is what a contiguous run of `2^(depth-s)` staged rows carries.
/// So the tile runs as an ordinary radix-2 network whose twiddle walk starts at `block << s`.
fn fused_stages(
    values: &mut [u128],
    plan: Plan,
    top: usize,
    depth: usize,
    twiddles: &Twiddles,
    inverse: bool,
    convert_basis: bool,
) {
    // A group whose stride is shorter than the planned run shortens the run to match.
    let log_block = plan.log_block.min(top - depth);
    // One row of the reshaped matrix, in elements.
    let run = plan.width << log_block;
    let len = values.len();
    // Reshaped rows between two consecutive staged runs, and elements in one staging tile.
    let stride = 1 << (top - log_block - depth);
    let tile_len = run << depth;
    // One tile per `(block, offset)` pair, which is one tile per `2^depth` reshaped rows.
    let tiles = len / tile_len;
    debug_assert_eq!(
        stride << depth,
        1 << (top - log_block),
        "staged rows do not span a block"
    );
    debug_assert_eq!(tiles * tile_len, len, "tiles do not partition the matrix");

    let rows = Rows {
        base: values.as_mut_ptr(),
        run,
        count: len / run,
    };
    let task = |tile: &mut Vec<u128>, index: usize| {
        // A tile index splits into the stage-`top` block it lies in and its offset in the
        // stride, and those two together with `k` name a reshaped row:
        //
        //     index  = block * S + offset
        //     row(k) = block * 2^(top-log_block) + offset + k * S
        let block = index >> (top - log_block - depth);
        let first = (block << (top - log_block)) + (index & (stride - 1));
        // SAFETY: `index` runs over `0..tiles` and `k` over `0..2^depth`, so the map
        // `(block, offset, k) -> row(k)` decomposes `0..2^(log_n-log_block)` in mixed radix.
        // Every reshaped row is therefore inside the matrix, and every one belongs to exactly
        // one tile index, hence to exactly one task.
        //
        // A reshaped row is a fixed-length block of adjacent matrix rows, so the matrix rows
        // partition across the tasks too.
        //
        // The exclusive borrow of the matrix outlives the whole region.
        unsafe { rows.gather(first, stride, 1 << depth, tile) };
        // Invariant: every element of the tile comes from the walk the gather just ran, so
        // nothing below reads an element the gather did not write.
        debug_assert_eq!(tile.len(), tile_len, "the gather left the tile short");
        // The gather is the first read of every element when this is the first group of a
        // forward transform.
        if convert_basis && !inverse {
            convert_tile(tile, INTO_POLY);
        }
        tile_stages(tile, run, depth, top, twiddles, inverse, block);
        // The scatter is the last write of every element when this is the last group of an
        // inverse transform.
        if convert_basis && inverse {
            convert_tile(tile, INTO_TOWER);
        }
        // SAFETY: the rows are the ones the gather read, so the argument above applies
        // unchanged.
        unsafe { rows.scatter(first, stride, tile) };
    };

    // One staging tile per worker, not per task: a task is a few tens of microseconds of
    // work and the tile is tens of kilobytes.
    //
    // The tile is capacity only, with no initialized elements: the gather grows it from
    // empty, so no worker spends bandwidth zeroing tens of kilobytes it is about to
    // overwrite in full.
    let new_tile = || Vec::with_capacity(tile_len);
    if use_parallel(len.saturating_mul(depth)) {
        (0..tiles).into_par_iter().for_each_init(new_tile, task);
    } else {
        let mut tile = new_tile();
        for index in 0..tiles {
            task(&mut tile, index);
        }
    }
}

/// Forward transform of polynomial-basis values in an existing allocation.
fn forward(values: &mut [u128], plan: Plan, shift: BinaryField128, fold: Fold) {
    let Plan {
        width,
        log_n,
        local,
        ..
    } = plan;
    let twiddles = Twiddles::new(log_n, shift);

    // Peel fused groups from the top stage downwards, each replacing `take` full passes.
    //
    // The first group's gather is the first read of every element, so it carries the entry
    // conversion.
    let mut entry = fold.entry;
    let mut top = log_n;
    for take in plan.group_sizes() {
        fused_stages(values, plan, top, take, &twiddles, false, entry);
        entry = false;
        top -= take;
    }

    // A plain pass carries no per-element map, so a conversion still owed ahead of one takes
    // a pass of its own.
    if entry && top > local {
        convert(values, INTO_POLY);
        entry = false;
    }
    for j in (local..top).rev() {
        stage(values, (1 << j) * width, j, &twiddles, false);
    }

    // The contiguous tile finishes the bottom stages, and is the last write of every
    // element.
    local_stages(
        values,
        plan,
        &twiddles,
        false,
        Fold {
            entry,
            exit: fold.exit,
        },
    );
}

/// Inverse transform with the data kept in the polynomial basis.
fn inverse(values: &mut [u128], plan: Plan, shift: BinaryField128, fold: Fold) {
    let Plan {
        width,
        log_n,
        local,
        ..
    } = plan;
    let twiddles = Twiddles::new(log_n, shift);
    let leftover = plan.leftover();

    // The contiguous tile runs first, so it is the first read of every element, and the last
    // write too when no stage runs above it.
    let tile_exit = fold.exit && local == log_n;
    let mut exit = fold.exit && !tile_exit;
    local_stages(
        values,
        plan,
        &twiddles,
        true,
        Fold {
            entry: fold.entry,
            exit: tile_exit,
        },
    );

    // Stages the staging tile does not pay for go before the groups, so a group still
    // reaches the top stage and can carry the exit conversion.
    for j in local..local + leftover {
        stage(values, (1 << j) * width, j, &twiddles, true);
    }

    // The groups come off the top downwards, so running them bottom up walks the same list
    // backwards. Its length is the group count, which never exceeds the stage count.
    let mut sizes = [0; usize::BITS as usize];
    let mut count = 0;
    for take in plan.group_sizes() {
        sizes[count] = take;
        count += 1;
    }
    let mut base = local + leftover;
    for (index, take) in sizes[..count].iter().rev().enumerate() {
        base += take;
        let last = index + 1 == count;
        fused_stages(values, plan, base, *take, &twiddles, true, exit && last);
        exit &= !last;
    }

    // No group ran, so the exit conversion needs a pass of its own.
    if exit {
        convert(values, INTO_TOWER);
    }
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

        // Both conversions ride along with the transform's first and last touch of each
        // element, so neither costs a pass over the matrix.
        forward(&mut values, Plan::new(width, log_n), shift, Fold::BOTH);

        mat.values = values.into_iter().map(BinaryField128::from_repr).collect();
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
        let width = mat.width;
        let log_message = log_n - log_inv_rate;
        let len = mat.values.len() >> log_inv_rate;
        let mut values: Vec<u128> = core::mem::take(&mut mat.values)
            .into_iter()
            .map(BinaryField128::to_repr)
            .collect();
        let plan = Plan::new(width, log_message);
        let (message, tail) = values.split_at_mut(len);
        // Every coset starts from a copy of the message, so one conversion of the message
        // alone serves all `2^log_inv_rate` of them.
        //
        // Converting each copy instead would repeat the sixteen dependent lookups per element
        // once per coset, for nothing.
        convert(message, INTO_POLY);
        // Only the conversion back is left, and each coset's contiguous tile carries its own.
        if len >= 2 * BUTTERFLY_GRAIN * p3_maybe_rayon::prelude::current_num_threads() {
            // Keep large coefficient copies next to evaluation so the copied data
            // is still hot, including when only one worker is available.
            for_chunks(tail, len, log_message, |(c, chunk)| {
                chunk.copy_from_slice(message);
                forward(
                    chunk,
                    plan,
                    domain_point((c + 1) << log_message),
                    Fold::EXIT,
                );
            });
            forward(message, plan, BinaryField128::ZERO, Fold::EXIT);
        } else {
            // Small cosets can run together after all coefficient copies are made.
            for_chunks(tail, len, 1, |(_, chunk)| chunk.copy_from_slice(message));
            for_chunks(&mut values, len, log_message, |(c, chunk)| {
                forward(chunk, plan, domain_point(c << log_message), Fold::EXIT);
            });
        }
        mat.values = values.into_iter().map(BinaryField128::from_repr).collect();
        mat
    }

    fn shifted_lde_batch(
        &self,
        mut mat: RowMajorMatrix<BinaryField128>,
        added_bits: usize,
        shift: BinaryField128,
    ) -> RowMajorMatrix<BinaryField128> {
        if !poly_basis::HAS_HARDWARE_CLMUL {
            return self.tower.shifted_lde_batch(mat, added_bits, shift);
        }
        let log_n = log2_strict_usize(mat.height());
        if added_bits == 0 {
            return mat;
        }
        let width = mat.width;
        let len = mat.values.len();
        let padded_len = u32::try_from(added_bits)
            .ok()
            .and_then(|bits| len.checked_shl(bits))
            .filter(|&padded| padded >> added_bits == len)
            .expect("extended codeword length overflows usize");
        // Preserve the evaluation prefix in its final allocation, then reuse the
        // original allocation for coefficients instead of cloning it before a resize.
        let mut coeffs: Vec<u128> = core::mem::take(&mut mat.values)
            .into_iter()
            .map(BinaryField128::to_repr)
            .collect();
        let mut values = alloc::vec![0u128; padded_len];
        values[..len].copy_from_slice(&coeffs);
        // The coefficients stay in the polynomial basis: every coset transform starts from a
        // copy of them.
        let plan = Plan::new(width, log_n);
        inverse(&mut coeffs, plan, shift, Fold::ENTRY);

        // The input evaluations already are the first coset. Only new cosets need
        // evaluation and conversion back from the polynomial basis.
        for_chunks(&mut values[len..], len, log_n, |(c, chunk)| {
            chunk.copy_from_slice(&coeffs);
            forward(
                chunk,
                plan,
                shift + domain_point::<BinaryField128>((c + 1) << log_n),
                Fold::EXIT,
            );
        });
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

        // See the forward transform: neither conversion costs a pass of its own.
        inverse(&mut values, Plan::new(width, log_n), shift, Fold::BOTH);

        mat.values = values.into_iter().map(BinaryField128::from_repr).collect();
        mat
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec::Vec;

    use p3_binary_field::{BinaryField128, TowerLevel, poly_basis};
    use p3_field::PrimeCharacteristicRing;
    use p3_matrix::dense::RowMajorMatrix;
    use p3_util::log2_floor_usize;
    use proptest::prelude::*;

    use super::{Fold, Plan, PolyBasisNtt, STAGED_RUN_BYTES, STAGED_WORKERS, STAGING_BYTES};
    use crate::domain::{domain_point, subspace_polynomial};
    use crate::lch::LchNtt;
    use crate::naive::NaiveAdditiveNtt;
    use crate::traits::AdditiveNtt;

    /// Cut points small enough to keep the test matrices tiny.
    ///
    /// One `(local, log_block, depth)` triple per branch of the schedule:
    ///
    /// ```text
    ///     (2, 0, 3)  full groups, plus a one-stage leftover at local + depth + 1
    ///     (2, 2, 3)  the same cut with runs of four rows per strided address
    ///     (2, 0, 0)  a staging tile too narrow for two runs, so no group runs at all
    ///     (0, 0, 3)  no contiguous tile, so every stage is fused
    ///     (0, 3, 3)  runs as long as a whole group's stride, which the group has to shorten
    ///     (1, 1, 1)  a depth of one, which never pays for a tile
    /// ```
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
    /// ```text
    ///     (0, 0, 3, 8)   3 + 3 + 2       a short last group, no contiguous tile
    ///     (0, 1, 3, 9)   3 + 3 + 3       three groups of full depth, runs of two rows
    ///     (1, 2, 2, 8)   2 + 2 + 2       three groups and a one-stage leftover
    ///     (0, 0, 2, 8)   2 + 2 + 2 + 2   four groups of full depth
    /// ```
    ///
    /// The staging loop reseeds its twiddle walk and its conversion flag on every turn, so
    /// only a third turn shows that the reseeding is not accidentally right for two.
    const DEEP_CUTS: [(usize, usize, usize, usize); 4] =
        [(0, 0, 3, 8), (0, 1, 3, 9), (1, 2, 2, 8), (0, 0, 2, 8)];

    /// Widths that cover a single element per row, an odd row, and rows of several elements.
    const WIDTHS: [usize; 5] = [1, 2, 3, 16, 64];

    /// Production shapes and the cut each one gets, as `width, log_n, (local, log_block,
    /// depth)`, at a worker count past [`STAGED_WORKERS`].
    ///
    /// The heights put the branch that is interesting for that width on it:
    ///
    /// ```text
    ///     width    1 @ 2^10   the tile is the whole transform
    ///     width    1 @ 2^14   one group of six stages, runs of 64 rows
    ///     width    1 @ 2^20   the production commit shape, one group of nine
    ///     width    2 @ 2^13   one group of six stages, runs of 32 rows
    ///     width    3 @ 2^12   an odd width, whose 32-row run overshoots
    ///     width    4 @ 2^14   the widest row a run still spans several of
    ///     width   16 @ 2^10   a short group, above a tile of 128 rows
    ///     width   16 @ 2^14   one full group, at a height that moves the cut
    ///     width   64 @ 2^10   one group of six stages, a run of one row
    ///     width  512 @ 2^8    two groups of three stages
    ///     width 1024 @ 2^7    three groups of two stages
    /// ```
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

    /// Cuts whose staging groups run at a height the reference oracle can still reach:
    ///
    /// ```text
    ///     (2, 0, 3, 6)   one group of three stages, plus a one-stage leftover
    ///     (2, 2, 3, 6)   the same cut with runs of four rows per strided address
    ///     (1, 1, 2, 5)   two groups of two stages, runs of two rows
    ///     (0, 0, 3, 6)   no contiguous tile, so every stage is fused
    /// ```
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
    /// A row of [`STAGED_RUN_BYTES`] or more gathers one row per strided address at every
    /// height, so once its first group has appeared the sweep only repeats a plan the cut
    /// tests already run — at over half the cost of the whole sweep, since these are the
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
    /// staged path reaches the oracle through the synthetic cuts of [`ORACLE_CUTS`] rather
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
        let twiddles = super::Twiddles::new(log_n, shift);
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
            super::inverse(values, plan, test_shift(), fold);
        } else {
            super::forward(values, plan, test_shift(), fold);
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
    /// ```text
    ///     local            nothing above the tile, so the tile is the whole transform
    ///     local + 1        one stage above the tile, which is left unfused
    ///     local + depth    one full staging group
    ///     local + depth+1  a full group and a one-stage leftover
    ///     2*local + depth  several groups and several tiles
    /// ```
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
                                super::inverse(&mut actual, plan, shift, NONE);
                            } else {
                                super::forward(&mut actual, plan, shift, NONE);
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
    /// counted by walking the loop [`super::forward`] walks rather than by asking the plan.
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
    fn the_plan_gathers_the_longest_run_its_pass_budget_allows() {
        // The production cut points, as `local, log_block, depth`, at a worker count past the
        // threshold. A run length is a choice between two plans that cost the same number of
        // passes, so no pass count can pin it: these triples are what says which one the
        // budgets pick, and a change to any of the three has to come through here.
        for (width, log_n, cut) in PRODUCTION_CUTS {
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
                let one_row = log2_floor_usize((STAGING_BYTES / row).max(1));
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
                    super::forward(&mut actual, plan, shift, Fold::BOTH);
                    assert_eq!(actual, reprs(&evals), "ntt {plan:?} shift={shift:?}");

                    let mut actual = reprs(&evals);
                    super::inverse(&mut actual, plan, shift, Fold::BOTH);
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
        let poly = PolyBasisNtt::default();
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
        let poly = PolyBasisNtt::default();
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
        // The cut points decide which phase carries it — a staging group's gather, a
        // staging group's scatter, the contiguous tile, or a pass of its own — so the same
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
                super::convert(&mut expected, super::INTO_POLY);
                scheduled(&mut expected, plan, inverse, NONE);
                let mut actual = input.clone();
                scheduled(&mut actual, plan, inverse, Fold::ENTRY);
                assert_eq!(actual, expected, "entry {plan:?} inverse={inverse}");

                // Conversion in, the schedule, conversion out.
                super::convert(&mut expected, super::INTO_TOWER);
                let mut actual = input.clone();
                scheduled(&mut actual, plan, inverse, Fold::BOTH);
                assert_eq!(actual, expected, "both {plan:?} inverse={inverse}");

                // The schedule, then conversion out.
                let mut expected = input.clone();
                scheduled(&mut expected, plan, inverse, NONE);
                super::convert(&mut expected, super::INTO_TOWER);
                let mut actual = input;
                scheduled(&mut actual, plan, inverse, Fold::EXIT);
                assert_eq!(actual, expected, "exit {plan:?} inverse={inverse}");
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
                assert_eq!(
                    PolyBasisNtt::default().ntt_batch_padded(mat, added),
                    expected
                );
            }
        }
    }

    #[test]
    fn shifted_lde_matches_naive_at_wide_widths() {
        for width in [1, 4, 16, 64] {
            for added in [0, 1, 2, 3] {
                let mat = matrix(4, width, 17);
                let shift = BinaryField128::from_repr(1 << 127);
                let expected =
                    NaiveAdditiveNtt::default().shifted_lde_batch(mat.clone(), added, shift);
                let actual = PolyBasisNtt::default().shifted_lde_batch(mat.clone(), added, shift);
                assert_eq!(actual, expected);
                assert_eq!(&actual.values[..mat.values.len()], &mat.values);
            }
        }
    }

    #[test]
    #[should_panic = "extended codeword length overflows usize"]
    fn lde_rejects_length_overflow() {
        let _ = PolyBasisNtt::default().lde_batch(matrix(1, 1, 0), usize::BITS as usize - 1);
    }

    #[test]
    fn incremental_twiddles_match_independent_domain_points() {
        let shift = BinaryField128::from_repr((1 << 127) | 123);
        let twiddles = super::Twiddles::new(usize::BITS as usize - 1, shift);
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
            let actual = PolyBasisNtt::default().shifted_ntt_batch(mat.clone(), shift);
            assert_eq!(actual, expected);
            assert_eq!(
                PolyBasisNtt::default().shifted_intt_batch(actual, shift),
                mat
            );
        }
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(32))]

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

            let fast = PolyBasisNtt::default().shifted_ntt_batch(coeffs.clone(), shift);
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

    #[test]
    fn poly_basis_matches_the_tower_across_several_tasks() {
        // These heights take more than one butterfly task per stage, so a task seeds its
        // twiddle at a block index of its own rather than at zero. Every oracle test sits
        // below them, so `LchNtt` stands in for the oracle here.
        //
        // An independent twiddle walk holds `LchNtt` itself at these heights, and it keeps its
        // data in the tower basis and derives its twiddles separately, so it shares no
        // arithmetic with the transform under test.
        let poly = PolyBasisNtt::default();
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
