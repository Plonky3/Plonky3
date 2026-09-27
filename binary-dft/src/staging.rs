//! Strided runs of rows, staged through one contiguous tile per worker.
//!
//! - A set of rows closed under several adjacent stages lies spread across the matrix.
//! - Gathering it into a contiguous tile runs all of those stages inside the cache.
//! - Scattering it back then writes the matrix once for the whole group.

use alloc::vec::Vec;
use core::ptr;

use p3_maybe_rayon::prelude::*;
use p3_util::DisjointMutPtr;

/// Bytes in the smallest page a target maps, so a write at this stride reaches every page.
const PAGE_BYTES: usize = 1 << 12;

/// Bytes one prefault task sweeps: a transparent huge page, so one worker faults each.
const PREFAULT_BYTES: usize = 1 << 21;

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
/// The matrix is read as consecutive runs of `run` elements.
///
/// Tile `index` holds `2^depth` runs, `2^log_stride` runs apart:
///
/// ```text
///     run(k) = block * 2^(log_stride + depth) + offset + k * 2^log_stride
/// ```
///
/// - `block` is the index shifted right by `log_stride`.
/// - `offset` is the index modulo `2^log_stride`.
/// - `k` runs over `0 .. 2^depth`.
///
/// That is a mixed-radix decomposition of the run index, so distinct pairs `(index, k)` name distinct runs.
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
    ///
    /// Panics if `2^(log_stride + depth)` does not fit a word.
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
/// - Each tile is gathered into a buffer grown from empty, so no worker zeroes what it overwrites.
/// - The closure then sees only elements the gather wrote.
///
/// # Panics
///
/// - Panics if the tiles do not partition the buffer.
/// - Panics if a tile's walk reaches past its end.
/// - Panics if a run is empty, which leaves no tile layout.
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
    // A tile longer than the matrix lays out no tile at all, and would return it untransformed.
    //
    // The check runs once per pass, not once per tile.
    assert_eq!(tiles * tile_len, len, "tiles do not partition the matrix");

    let base = DisjointMutPtr::new(values);
    let task = move |tile: &mut Vec<T>, index: usize| {
        // The walk ascends, so bounding its last run bounds all of them.
        //
        // One hard check per tile is cheap, and a run past the end would be a write out of bounds.
        assert!(
            runs.run_index(index, rows - 1) < count,
            "staged row walk leaves the matrix"
        );

        tile.clear();
        for k in 0..rows {
            // SAFETY: distinct pairs (index, k) name distinct runs of one fixed length.
            //
            // - So no two tasks, and no two iterations of one task, reach the same element.
            // - The assert above keeps every run of this walk inside the buffer.
            // - The exclusive borrow behind the pointer outlives every task, since the pass joins them all.
            // - Every run index stays below the run count, so no shift or sum wraps.
            let source = unsafe { base.slice(runs.run_index(index, k) * run, run) };
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

    // One buffer per worker, not per tile, since a tile is microseconds of work and tens of kilobytes.
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

/// Fault in every page of a region a staging pass is about to overwrite, before it does.
///
/// A staging pass scatters each tile's runs across the whole region.
///
/// The first writes of every worker therefore land on the same few pages at once.
///
/// Where a fault maps and clears a huge page, every worker that loses the race to map it has
/// cleared one for nothing.
///
/// A contiguous sweep gives each huge page to a single task, which faults it once at the
/// sequential rate.
///
/// Each page takes one zero, so the region must hold zeros or be overwritten in full next.
pub(crate) fn prefault(values: &mut [u128]) {
    let page = PAGE_BYTES / size_of::<u128>();
    let touch = |chunk: &mut [u128]| {
        for element in chunk.iter_mut().step_by(page) {
            // SAFETY: `element` is an exclusive reference to an initialised element.
            //
            // It is therefore valid and aligned for a write.
            //
            // The write is volatile so that it reaches memory even where the region holds zeros.
            unsafe { ptr::write_volatile(element, 0) };
        }
    };

    // The run up to the first huge-page boundary is swept on its own.
    //
    // Every task after it then starts on a boundary and owns whole huge pages.
    let head = values
        .as_ptr()
        .align_offset(PREFAULT_BYTES)
        .min(values.len());
    let (head, body) = values.split_at_mut(head);
    touch(head);
    body.par_chunks_mut(PREFAULT_BYTES / size_of::<u128>())
        .for_each(touch);
}

/// Gather every tile of the leading coset once, and scatter one result per coset from it.
///
/// - The buffer is a sequence of equal cosets, and the tiles lay out the leading one.
/// - The tiles are gathered from the separate source when one is given, which stands in for the leading coset.
/// - Otherwise they are gathered from the leading coset itself.
/// - Each gathered tile is prepared once.
/// - Every coset then processes its own copy and scatters it to the same runs of that coset.
/// - The leading coset goes last, from the gathered tile itself, once every other coset has its copy.
///
/// # Panics
///
/// - Panics if the cosets do not partition the buffer, or the tiles one coset.
/// - Panics if a separate source is not one coset long.
/// - Panics if a tile's walk reaches past the end of a coset.
/// - Panics if a coset or a run is empty.
pub(crate) fn for_each_staged_tile_into_cosets<T, Q, P>(
    values: &mut [T],
    source: Option<&[T]>,
    coset_len: usize,
    runs: StagedRuns,
    dispatch: Dispatch,
    prepare: Q,
    process: P,
) where
    T: Copy + Send + Sync,
    Q: Fn(&mut [T]) + Send + Sync,
    P: Fn(&mut [T], usize, usize) + Send + Sync,
{
    let StagedRuns { run, depth, .. } = runs;
    let cosets = values.len() / coset_len;
    let rows = 1 << depth;
    let tile_len = run << depth;
    let tiles = coset_len / tile_len;
    // Runs in one coset.
    let count = coset_len / run;
    assert_eq!(
        cosets * coset_len,
        values.len(),
        "cosets do not partition the matrix"
    );
    assert!(
        source.is_none_or(|source| source.len() == coset_len),
        "the source is not one coset long"
    );
    assert_eq!(
        tiles * tile_len,
        coset_len,
        "tiles do not partition the coset"
    );

    let base = DisjointMutPtr::new(values);
    // Every run of every coset, `coset` cosets past the leading one.
    let slice_of = move |index: usize, k: usize, coset: usize| {
        coset * coset_len + runs.run_index(index, k) * run
    };
    let task = move |(tile, copy): &mut (Vec<T>, Vec<T>), index: usize| {
        // The walk ascends, so bounding its last run bounds all of them.
        assert!(
            runs.run_index(index, rows - 1) < count,
            "staged row walk leaves the coset"
        );

        tile.clear();
        for k in 0..rows {
            let start = slice_of(index, k, 0);
            let gathered = source.map_or_else(
                // SAFETY: distinct pairs (index, k) name distinct runs, and the assert keeps them in the leading coset.
                //
                // - So no two tasks, and no two iterations of one task, reach the same element.
                // - The exclusive borrow behind the pointer outlives every task, since the pass joins them all.
                || unsafe { base.slice(start, run) },
                |source| &source[start..start + run],
            );
            tile.extend_from_slice(gathered);
        }
        debug_assert_eq!(tile.len(), tile_len, "the gather left the tile short");
        prepare(tile.as_mut_slice());

        let block = runs.block(index);
        for coset in (0..cosets).rev() {
            let staged = if coset == 0 {
                &mut *tile
            } else {
                copy.clear();
                copy.extend_from_slice(&tile[..]);
                &mut *copy
            };
            process(staged.as_mut_slice(), block, coset);
            for (k, source) in staged.chunks_exact(run).enumerate() {
                // SAFETY: the gathered runs moved by whole cosets, so they stay disjoint.
                //
                // Every coset has the leading coset's length, so they also stay in bounds.
                let target = unsafe { base.slice_mut(slice_of(index, k, coset), run) };
                target.copy_from_slice(source);
            }
        }
    };

    // Two buffers per worker, not per tile: the gathered tile and the copy a coset runs on.
    let new_tiles = || (Vec::with_capacity(tile_len), Vec::with_capacity(tile_len));
    match dispatch {
        Dispatch::Parallel { min_len } => (0..tiles)
            .into_par_iter()
            .with_min_len(min_len)
            .for_each_init(new_tiles, task),
        Dispatch::Serial => {
            let mut buffers = new_tiles();
            for index in 0..tiles {
                task(&mut buffers, index);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec::Vec;
    use alloc::{format, vec};

    use super::{
        Dispatch, StagedRuns, for_each_staged_tile, for_each_staged_tile_into_cosets, prefault,
    };

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
    fn every_coset_receives_its_own_result_of_the_leading_one() {
        let dispatches = [
            Dispatch::Serial,
            Dispatch::Parallel { min_len: 1 },
            Dispatch::Parallel { min_len: 4 },
        ];
        for dispatch in dispatches {
            for cosets in [1usize, 2, 4] {
                for run in [1usize, 3] {
                    for (log_runs, log_stride, depth) in
                        [(6, 2, 3), (6, 0, 6), (6, 6, 0), (5, 1, 2)]
                    {
                        let coset_len = run << log_runs;
                        let label = format!(
                            "{dispatch:?} cosets={cosets} run={run} log_runs={log_runs} \
                             log_stride={log_stride} depth={depth}"
                        );
                        // The leading coset holds its own positions, and every later coset a
                        // marker the pass has to overwrite.
                        let mut values: Vec<u64> = (0..coset_len as u64)
                            .chain(core::iter::repeat_n(u64::MAX, (cosets - 1) * coset_len))
                            .collect();

                        for_each_staged_tile_into_cosets(
                            &mut values,
                            None,
                            coset_len,
                            StagedRuns::new(run, log_stride, depth),
                            dispatch,
                            |tile| tile.iter_mut().for_each(|value| *value *= 2),
                            |tile, block, coset| {
                                for (k, staged) in tile.chunks_exact_mut(run).enumerate() {
                                    // Each run arrives whole, in order and prepared once, so its
                                    // first element names it.
                                    let r = staged[0] as usize / (2 * run);
                                    for (j, &value) in staged.iter().enumerate() {
                                        assert_eq!(value, 2 * (r * run + j) as u64, "{label}");
                                    }
                                    assert_eq!(r >> (log_stride + depth), block, "{label}");
                                    assert_eq!(
                                        (r >> log_stride) & ((1 << depth) - 1),
                                        k,
                                        "{label}"
                                    );
                                    for value in staged.iter_mut() {
                                        *value += (coset * coset_len) as u64;
                                    }
                                }
                            },
                        );

                        // Element `i` of coset `c` is the doubled leading element moved `c`
                        // cosets along.
                        for (i, &value) in values.iter().enumerate() {
                            let (coset, offset) = (i / coset_len, i % coset_len);
                            let want = 2 * offset as u64 + (coset * coset_len) as u64;
                            assert_eq!(value, want, "i={i} {label}");
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn a_prefault_leaves_a_zeroed_region_zero() {
        // Regions shorter than a page, spanning several, and spanning several prefault tasks.
        //
        // A region starting past the allocation's own start has a head before its first boundary.
        for len in [
            0usize,
            1,
            255,
            256,
            257,
            3 * 256 + 5,
            (1 << 17) + 3,
            3 << 17,
        ] {
            let mut values = vec![0u128; len];
            prefault(&mut values);
            prefault(&mut values[len / 3..]);
            assert!(values.iter().all(|&value| value == 0), "len={len}");
        }
    }

    #[test]
    fn a_separate_source_stands_in_for_the_leading_coset() {
        // Invariant: gathering from a source fills every coset, the leading one included, as
        // gathering the same values out of the leading coset itself does.
        let dispatches = [Dispatch::Serial, Dispatch::Parallel { min_len: 1 }];
        let cosets = 4;
        for dispatch in dispatches {
            for (run, log_runs, log_stride, depth) in [(1, 6, 2, 3), (3, 5, 1, 2), (1, 6, 0, 6)] {
                let coset_len = run << log_runs;
                let runs = StagedRuns::new(run, log_stride, depth);
                let label = format!("{dispatch:?} run={run} log_runs={log_runs} depth={depth}");
                let source: Vec<u64> = (0..coset_len as u64).map(|v| 7 * v + 1).collect();
                let prepare = |tile: &mut [u64]| tile.iter_mut().for_each(|value| *value ^= 0x55);
                let process = |tile: &mut [u64], block: usize, coset: usize| {
                    for (k, value) in tile.iter_mut().enumerate() {
                        *value = 3 * *value + (1000 * block + 100 * coset + k) as u64;
                    }
                };

                let mut in_place: Vec<u64> = source
                    .iter()
                    .copied()
                    .chain(core::iter::repeat_n(u64::MAX, (cosets - 1) * coset_len))
                    .collect();
                for_each_staged_tile_into_cosets(
                    &mut in_place,
                    None,
                    coset_len,
                    runs,
                    dispatch,
                    prepare,
                    process,
                );

                let mut separate = vec![u64::MAX; cosets * coset_len];
                for_each_staged_tile_into_cosets(
                    &mut separate,
                    Some(source.as_slice()),
                    coset_len,
                    runs,
                    dispatch,
                    prepare,
                    process,
                );
                assert_eq!(separate, in_place, "{label}");
            }
        }
    }

    #[test]
    #[should_panic = "the source is not one coset long"]
    fn a_source_of_another_length_is_refused() {
        let mut values = vec![0u8; 16];
        for_each_staged_tile_into_cosets(
            &mut values,
            Some(&[0u8; 4][..]),
            8,
            StagedRuns::new(1, 1, 2),
            Dispatch::Serial,
            |_| {},
            |_, _, _| {},
        );
    }

    #[test]
    #[should_panic = "cosets do not partition the matrix"]
    fn cosets_that_do_not_partition_the_matrix_are_refused() {
        let mut values = vec![0u8; 12];
        for_each_staged_tile_into_cosets(
            &mut values,
            None,
            8,
            StagedRuns::new(1, 1, 2),
            Dispatch::Serial,
            |_| {},
            |_, _, _| {},
        );
    }

    #[test]
    #[should_panic = "tiles do not partition the matrix"]
    fn a_tile_longer_than_the_matrix_is_refused() {
        // One tile of four runs against a matrix of three: no tile fits, so none would run.
        let mut values = vec![0u8; 3];
        for_each_staged_tile(
            &mut values,
            StagedRuns::new(1, 0, 2),
            Dispatch::Serial,
            |_, _| {},
        );
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
