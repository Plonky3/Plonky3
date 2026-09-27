//! Where one polynomial-basis transform cuts its stage sequence into passes over memory.
//!
//! Every figure below was tuned by sweeps whose numbers live in the pull requests that set them.

use p3_maybe_rayon::prelude::current_num_threads;
use p3_util::{log2_ceil_usize, log2_floor_usize};

/// Bytes of one contiguous tile for a matrix the shared cache holds.
const TILE_BYTES: usize = 32 * 1024;

/// Bytes of one staging tile for a matrix the shared cache holds.
///
/// The tile is streamed, so it need not fit L1, only the private cache below the shared one.
const STAGING_BYTES: usize = 64 * 1024;

/// Bytes of both tiles for a matrix the shared cache cannot hold.
///
/// - Past the shared cache, every stage a tile fuses saves a traversal of memory itself.
/// - The figure is half of one hardware thread's share of a private L2.
/// - The other half is left to the gather and scatter streaming through the same cache.
pub(super) const DEEP_TILE_BYTES: usize = 256 * 1024;

/// Bytes of shared cache a matrix is taken to fit inside.
///
/// - A `no_std` crate cannot read the cache size, so the matrix size against a fixed figure is the signal.
/// - A smaller real cache only forgoes the deeper tiles' win, which is the safe direction.
const SHARED_CACHE_BYTES: usize = 128 * 1024 * 1024;

// The smallest matrix on the deep budget must still leave far more tiles than any machine has workers.
const _: () = assert!(SHARED_CACHE_BYTES / DEEP_TILE_BYTES >= 256);

/// The fewest bytes a gathered run of adjacent rows may cover.
///
/// A gather pulls whole cache lines, so a run shorter than the smallest line wastes the rest of it.
const STAGED_LINE_BYTES: usize = 64;

/// The run length a plan grows towards, where lengthening stops paying.
///
/// A longer run gives the prefetcher a longer stride to follow, at the cost of tile depth.
pub(super) const STAGED_RUN_BYTES: usize = 1024;

/// Doublings of the run a plan must be offered before it pays for one added gather.
///
/// One doubling loses, two break even, three win on every shape measured.
const STAGED_GATHER_DOUBLINGS: usize = 3;

/// Workers a row shorter than a cache line needs before staging it pays.
///
/// - Staging trades traversals of the matrix for a strided gather and scatter.
/// - Few workers leave memory idle, so the traversals it removes were cheap.
/// - Rows of a cache line or more are staged at any worker count.
pub(super) const STAGED_WORKERS: usize = 4;

/// The shape of one transform, and the two places its stage sequence is cut.
///
/// - The top stages pair rows far apart, and staging tiles gather them several at a time.
/// - The leftover stages below them run as one plain pass each.
/// - The bottom `local` stages pair adjacent rows, and run inside contiguous tiles.
///
/// At most one stage is left over, unless the staging tile cannot hold two runs.
#[derive(Copy, Clone, Debug)]
pub(super) struct Plan {
    /// Elements per row.
    pub(super) width: usize,
    /// Base-two logarithm of the row count.
    pub(super) log_n: usize,
    /// Bottom stages that run inside one contiguous tile of rows.
    pub(super) local: usize,
    /// Base-two logarithm of the adjacent rows one strided gather address moves.
    pub(super) log_block: usize,
    /// Long-stride stages one staging tile fuses into a single pass.
    pub(super) depth: usize,
}

impl Plan {
    /// Stages above the tile left as plain passes, for `above` stages cut into groups of `depth`.
    ///
    /// - Fusing one stage moves what a plain pass moves, plus two copies, so a lone stage is left over.
    /// - A staging tile too shallow for two stages leaves every stage over.
    pub(super) const fn leftover_stages(above: usize, depth: usize) -> usize {
        if depth < 2 {
            above
        } else if above % depth == 1 {
            1
        } else {
            0
        }
    }

    /// The stages each staging group fuses, from the top stage down.
    ///
    /// The walk stops where the leftover stages begin, so a short group lands at the bottom of the band.
    pub(super) fn groups(above: usize, depth: usize) -> impl Iterator<Item = usize> {
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

    /// Full traversals of the matrix the stages above the tile take.
    ///
    /// Each fused group is one traversal, and each leftover stage one more.
    fn traversals(above: usize, depth: usize) -> usize {
        Self::groups(above, depth).count() + Self::leftover_stages(above, depth)
    }

    /// The contiguous and staging tile budgets for rows of `row` bytes.
    ///
    /// The comparison is in rows, so no height can overflow a byte count.
    pub(super) fn budgets(row: usize, log_n: usize) -> (usize, usize) {
        if log_n > log2_floor_usize((SHARED_CACHE_BYTES / row).max(1)) {
            (DEEP_TILE_BYTES, DEEP_TILE_BYTES)
        } else {
            (TILE_BYTES, STAGING_BYTES)
        }
    }

    /// The plan a matrix gets on the current thread pool.
    pub(super) fn new(width: usize, log_n: usize) -> Self {
        Self::for_workers(width, log_n, current_num_threads())
    }

    /// The plan a matrix gets from its budgets and a worker count.
    ///
    /// # Algorithm
    ///
    /// The staging budget is fixed, so doubling the run halves the rows it holds.
    ///
    /// Run length and fused depth therefore trade one for one:
    ///
    /// - The run starts at one cache line, the shortest worth gathering.
    /// - It grows towards its target while it adds no traversal of the matrix.
    /// - It may add one gathering traversal only once it has doubled enough to pay for it.
    pub(super) fn for_workers(width: usize, log_n: usize, workers: usize) -> Self {
        let row = size_of::<u128>() * width;
        let (tile, staging) = Self::budgets(row, log_n);
        let local = log2_floor_usize((tile / row).max(1)).min(log_n);
        let above = log_n - local;

        // Stages one staging tile fuses, when each of its rows is a run of 2^log_block matrix rows.
        let depth_at = |log_block: usize| log2_floor_usize((staging / (row << log_block)).max(1));
        let floor = log2_ceil_usize(STAGED_LINE_BYTES.div_ceil(row));
        let target = log2_ceil_usize(STAGED_RUN_BYTES.div_ceil(row)).max(floor);

        // The shortest admissible run sets the two counts a longer run is held to.
        let budget = Self::traversals(above, depth_at(floor));
        let gathers = Self::groups(above, depth_at(floor)).count();
        let log_block = (floor..=target)
            .rev()
            .find(|&log_block| {
                let depth = depth_at(log_block);

                // Enough doublings of the run to cover the gather an added group costs.
                let paid = log_block >= floor + STAGED_GATHER_DOUBLINGS;
                Self::traversals(above, depth) <= budget
                    && (paid || Self::groups(above, depth).count() <= gathers)
            })
            .unwrap_or(floor);

        // A row under a cache line is staged only once enough workers contend for memory.
        let depth = if row >= STAGED_LINE_BYTES || workers >= STAGED_WORKERS {
            depth_at(log_block)
        } else {
            0
        };

        Self {
            width,
            log_n,
            local,
            log_block,
            depth,
        }
    }

    /// Stages above the tile this plan runs as plain passes.
    pub(super) const fn leftover(&self) -> usize {
        Self::leftover_stages(self.log_n - self.local, self.depth)
    }

    /// The stages each of this plan's staging groups fuses, from the top stage down.
    pub(super) fn group_sizes(&self) -> impl Iterator<Item = usize> {
        Self::groups(self.log_n - self.local, self.depth)
    }
}
