//! Strided runs of rows, staged through one contiguous tile per worker.
//!
//! A stage pairs rows a power of two apart, so a set of rows closed under several adjacent
//! stages lies spread across the matrix. Gathering the set into a contiguous tile runs all of
//! those stages inside the cache, and scattering it back writes the matrix once.

use alloc::vec::Vec;

use p3_maybe_rayon::prelude::*;
use p3_util::DisjointMutPtr;

/// How the tiles of one staging pass are spread over the workers.
#[derive(Copy, Clone, Debug)]
pub(crate) enum Dispatch {
    /// Every tile on the calling thread, through a single buffer.
    Serial,
    /// Tiles on the pool, with no split shorter than `min_len` tiles.
    ///
    /// A split is what one buffer serves, so this bounds the buffers a pass allocates.
    Parallel {
        /// The fewest consecutive tiles one split takes.
        min_len: usize,
    },
}

/// The runs a staging pass moves through its tiles.
///
/// The matrix is read as consecutive runs of `run` elements, and tile `index` holds `2^depth`
/// runs `2^log_stride` apart:
///
/// ```text
///     block  = index >> log_stride
///     offset = index mod 2^log_stride
///     run(k) = block · 2^(log_stride + depth) + offset + k · 2^log_stride ,   k = 0 .. 2^depth
/// ```
///
/// With `offset < 2^log_stride` and `k < 2^depth` that is a mixed-radix decomposition of the
/// run index, so distinct `(index, k)` name distinct runs.
#[derive(Copy, Clone, Debug)]
pub(crate) struct StagedRuns {
    /// Elements in one run.
    run: usize,
    /// Base-two log of the runs from one staged run to the next.
    log_stride: usize,
    /// Base-two log of the runs one tile holds.
    depth: usize,
}

impl StagedRuns {
    /// Runs of `run` elements, `2^depth` to a tile and `2^log_stride` runs apart.
    ///
    /// # Panics
    /// Panics if `2^(log_stride + depth)` does not fit a `usize`.
    pub(crate) const fn new(run: usize, log_stride: usize, depth: usize) -> Self {
        assert!(
            depth < usize::BITS as usize && log_stride < usize::BITS as usize - depth,
            "staging shifts exceed the word size"
        );
        Self {
            run,
            log_stride,
            depth,
        }
    }

    /// The block tile `index` lies in.
    #[inline]
    pub(crate) const fn block(&self, index: usize) -> usize {
        index >> self.log_stride
    }

    /// The run tile `index` stages as its row `k`.
    #[inline]
    pub(crate) const fn run_index(&self, index: usize, k: usize) -> usize {
        let offset = index & ((1 << self.log_stride) - 1);
        (self.block(index) << (self.log_stride + self.depth)) + offset + (k << self.log_stride)
    }
}

/// Gather every tile, hand it to `process` with its block, and scatter it back.
///
/// A tile is gathered into a buffer grown from empty, so `process` sees only elements the
/// gather wrote, and no worker zeroes a buffer it is about to overwrite in full.
///
/// # Panics
/// Panics if a tile's walk reaches past the end of `values`, or if `run == 0`, which divides
/// by zero laying out the tiles.
pub(crate) fn for_each_staged_tile<T, P>(
    values: &mut [T],
    runs: StagedRuns,
    dispatch: Dispatch,
    process: P,
) where
    T: Copy + Send + Sync,
    P: Fn(&mut [T], usize) + Send + Sync,
{
    let StagedRuns { run, depth, .. } = runs;
    let len = values.len();
    let rows = 1 << depth;
    let tile_len = run << depth;
    let tiles = len / tile_len;
    // Runs in the matrix.
    let count = len / run;
    debug_assert_eq!(tiles * tile_len, len, "tiles do not partition the matrix");

    let base = DisjointMutPtr::new(values);
    let task = move |tile: &mut Vec<T>, index: usize| {
        // The walk ascends, so bounding its last run bounds all of them. It runs once per
        // tile rather than once per run, which is what a hard check costs, and a run past the
        // end would be a write past the end of the matrix.
        assert!(
            runs.run_index(index, rows - 1) < count,
            "staged row walk leaves the matrix"
        );

        tile.clear();
        for k in 0..rows {
            // SAFETY: `run_index` is injective over `(index, k)`, as `StagedRuns` sets out, and
            // runs of one fixed length at distinct indices are disjoint element ranges. So no
            // two tasks and no two iterations of one task reach the same element.
            //
            // The assert above puts every run of this walk inside `values`. The exclusive
            // borrow the pointer came from outlives every task, since the pass returns only
            // once all of them have run.
            //
            // `index < tiles <= len / 2^depth` keeps every `run_index` below `len`, so none of
            // its shifts or sums wrap.
            let source = unsafe { base.slice_mut(runs.run_index(index, k) * run, run) };
            tile.extend_from_slice(source);
        }
        debug_assert_eq!(tile.len(), tile_len, "the gather left the tile short");

        process(tile.as_mut_slice(), runs.block(index));

        for (k, source) in tile.chunks_exact(run).enumerate() {
            // SAFETY: the runs the gather above reached, for the same reason.
            let target = unsafe { base.slice_mut(runs.run_index(index, k) * run, run) };
            target.copy_from_slice(source);
        }
    };

    // One buffer per worker, not per tile: a tile is microseconds of work and tens of
    // kilobytes of memory.
    let new_tile = || Vec::with_capacity(tile_len);
    match dispatch {
        Dispatch::Parallel { min_len } => (0..tiles)
            .into_par_iter()
            .with_min_len(min_len)
            .for_each_init(new_tile, task),
        Dispatch::Serial => {
            let mut tile = new_tile();
            for index in 0..tiles {
                task(&mut tile, index);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec::Vec;
    use alloc::{format, vec};

    use super::{Dispatch, StagedRuns, for_each_staged_tile};

    #[test]
    fn the_tiles_partition_the_runs() {
        // Invariant: over every shape that fits the matrix, the walks of all tiles together
        // name each run exactly once, and every run a tile names lies in the tile's block.
        for log_runs in 0..=8usize {
            for depth in 0..=log_runs {
                for log_stride in 0..=log_runs - depth {
                    let runs = StagedRuns::new(1, log_stride, depth);
                    let label =
                        format!("log_runs={log_runs} log_stride={log_stride} depth={depth}");
                    let mut seen = vec![false; 1 << log_runs];
                    for index in 0..1usize << (log_runs - depth) {
                        for k in 0..1usize << depth {
                            let run = runs.run_index(index, k);
                            let hit = seen
                                .get_mut(run)
                                .unwrap_or_else(|| panic!("run {run} out of range, {label}"));
                            assert!(!*hit, "run {run} staged twice, {label}");
                            *hit = true;
                            assert_eq!(
                                run >> (log_stride + depth),
                                runs.block(index),
                                "index={index} k={k} {label}"
                            );
                        }
                    }
                    assert!(seen.iter().all(|&hit| hit), "a run was left out, {label}");
                }
            }
        }
    }

    #[test]
    fn a_pass_writes_back_what_the_callback_left() {
        let dispatches = [
            Dispatch::Serial,
            Dispatch::Parallel { min_len: 1 },
            Dispatch::Parallel { min_len: 4 },
        ];
        for dispatch in dispatches {
            for run in [1usize, 3] {
                for (log_runs, log_stride, depth) in [(6, 2, 3), (6, 0, 6), (6, 6, 0), (5, 1, 2)] {
                    let len = run << log_runs;
                    let mut values: Vec<u64> = (0..len as u64).collect();
                    let shift = len as u64;
                    let label = format!(
                        "{dispatch:?} run={run} log_runs={log_runs} log_stride={log_stride} \
                         depth={depth}"
                    );

                    for_each_staged_tile(
                        &mut values,
                        StagedRuns::new(run, log_stride, depth),
                        dispatch,
                        |tile, block| {
                            for (k, staged) in tile.chunks_exact_mut(run).enumerate() {
                                // Each run arrives whole and in order, so its first element
                                // names it.
                                let r = staged[0] as usize / run;
                                for (j, &value) in staged.iter().enumerate() {
                                    assert_eq!(value, (r * run + j) as u64, "{label}");
                                }
                                assert_eq!(r >> (log_stride + depth), block, "{label}");
                                assert_eq!((r >> log_stride) & ((1 << depth) - 1), k, "{label}");

                                for value in staged.iter_mut() {
                                    *value += shift;
                                }
                            }
                        },
                    );

                    // Each element was staged once and written back where it came from.
                    for (i, &value) in values.iter().enumerate() {
                        assert_eq!(value, i as u64 + shift, "i={i} {label}");
                    }
                }
            }
        }
    }

    #[test]
    #[should_panic = "staged row walk leaves the matrix"]
    fn a_walk_past_the_matrix_is_refused() {
        // Three tiles of two runs, two runs apart: the third tile walks runs 4 and 6, and the
        // matrix holds runs 0 to 5.
        let mut values = vec![0u8; 6];
        for_each_staged_tile(
            &mut values,
            StagedRuns::new(1, 1, 1),
            Dispatch::Serial,
            |_, _| {},
        );
    }
}
