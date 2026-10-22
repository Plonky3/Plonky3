//! Deterministic parallel search for the first match in a range of candidates.
//!
//! rayon's `find_any` returns whichever match a worker reaches first, so its answer varies run to run.
//! rayon's `find_first` is deterministic, but it splits the range into a few large blocks up front.
//! On a huge range the smallest match then sits in one block that a single worker scans alone.
//!
//! The search here hands out the range in order instead:
//!
//! ```text
//!     workers claim chunks from one shared counter, lowest first
//!     a worker scans its chunk in order and stops at its first match
//!     a worker stops claiming once a match lies below the next chunk
//! ```
//!
//! Every chunk below the smallest match is claimed and scanned whole.
//! So the result is the smallest match, as a serial scan finds it.
//! Work past it is at most one chunk per worker.

use core::ops::Range;
#[cfg(feature = "parallel")]
use core::sync::atomic::{AtomicU64, Ordering};

#[cfg(feature = "parallel")]
use rayon::prelude::*;

/// Return `f(state, i)` at the smallest `i` in `candidates` where it is `Some`.
///
/// Each worker builds its own `state` with `init` and passes it to every call it makes.
/// It is scratch space only: the result must depend on `i` alone.
///
/// The answer is the one a serial scan gives, whatever the thread count.
///
/// # Performance
///
/// Chunks grow with the distance scanned, staying small against the work done so far.
/// The work past the match is then a small fraction of the search.
pub fn find_map_first<T, R>(
    candidates: Range<u64>,
    init: impl Fn() -> T + Sync,
    f: impl Fn(&mut T, u64) -> Option<R> + Sync,
) -> Option<R>
where
    R: Send,
{
    #[cfg(feature = "parallel")]
    {
        let threads = rayon::current_num_threads();
        if threads > 1 {
            return find_map_first_parallel(candidates, threads, init, f);
        }
    }

    // One worker: a plain scan in order.
    let mut state = init();
    candidates.into_iter().find_map(|i| f(&mut state, i))
}

/// The ordered search on `threads` workers.
#[cfg(feature = "parallel")]
fn find_map_first_parallel<T, R>(
    candidates: Range<u64>,
    threads: usize,
    init: impl Fn() -> T + Sync,
    f: impl Fn(&mut T, u64) -> Option<R> + Sync,
) -> Option<R>
where
    R: Send,
{
    let schedule = Schedule::new(candidates, threads);

    // The index of the next unclaimed chunk.
    let next = CacheLine(AtomicU64::new(0));
    // The smallest match found so far, or `u64::MAX`.
    let found = CacheLine(AtomicU64::new(u64::MAX));

    let worker = |_| {
        let mut state = init();
        // Claim chunks in order until the range ends.
        while let Some(chunk) = schedule.chunk(next.0.fetch_add(1, Ordering::Relaxed)) {
            // A match below this chunk is smaller than anything in it or after it.
            if chunk.start > found.0.load(Ordering::Relaxed) {
                break;
            }
            // Every chunk below this one is claimed, so this worker's first match is final for it.
            for i in chunk {
                if let Some(value) = f(&mut state, i) {
                    found.0.fetch_min(i, Ordering::Relaxed);
                    return Some((i, value));
                }
            }
        }
        None
    };

    // Each worker reports its first match; the smallest of them is the answer.
    (0..threads)
        .into_par_iter()
        .filter_map(worker)
        .min_by_key(|&(i, _)| i)
        .map(|(_, value)| value)
}

/// A value alone on its cache line.
///
/// Every worker writes the claim counter, and every worker reads the best match.
/// Apart, the reads stay cache hits between the writes.
#[cfg(feature = "parallel")]
#[repr(align(128))]
struct CacheLine<T>(T);

/// The split of a range of candidates into chunks, claimed by index.
///
/// Chunks come in levels of equal count, each level's chunks twice the size of the last:
///
/// ```text
///     level 0: per_level chunks of MIN_CHUNK candidates
///     level 1: per_level chunks of 2 * MIN_CHUNK candidates
///     ...
///     level MAX_LEVEL and after: chunks of MIN_CHUNK << MAX_LEVEL candidates
/// ```
///
/// A chunk then holds about `1 / per_level` of the candidates before it.
#[cfg(feature = "parallel")]
struct Schedule {
    /// The candidates to split.
    candidates: Range<u64>,
    /// The number of chunks in each level.
    per_level: u64,
}

#[cfg(feature = "parallel")]
impl Schedule {
    /// The fewest candidates in a chunk.
    ///
    /// A claim moves a contended cache line, so a chunk must hold several candidates.
    const MIN_CHUNK: u64 = 16;

    /// Chunks per worker in each level.
    ///
    /// Work past the smallest match is at most one chunk per worker.
    /// That is about `1 / CHUNKS_PER_WORKER` of the time the search takes.
    const CHUNKS_PER_WORKER: u64 = 128;

    /// The last level whose chunks grow: past it, the overshoot is bounded in candidates.
    const MAX_LEVEL: u64 = 12;

    /// The schedule of `candidates` for `threads` workers.
    const fn new(candidates: Range<u64>, threads: usize) -> Self {
        Self {
            candidates,
            per_level: threads as u64 * Self::CHUNKS_PER_WORKER,
        }
    }

    /// The `index`-th chunk, or `None` past the end of the range.
    fn chunk(&self, index: u64) -> Option<Range<u64>> {
        // The level, capped, and the position inside it.
        let level = (index / self.per_level).min(Self::MAX_LEVEL);
        let position = index - level * self.per_level;
        let size = Self::MIN_CHUNK << level;

        // Levels below hold `per_level * MIN_CHUNK * (2^level - 1)` candidates.
        let below = self.per_level * Self::MIN_CHUNK * ((1 << level) - 1);
        let start = position
            .checked_mul(size)
            .and_then(|offset| offset.checked_add(below))
            .and_then(|offset| offset.checked_add(self.candidates.start))
            .filter(|&start| start < self.candidates.end)?;
        Some(start..start.saturating_add(size).min(self.candidates.end))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Run `op` on four workers, so the parallel path runs on any host.
    fn on_four_workers<R: Send>(op: impl FnOnce() -> R + Send) -> R {
        #[cfg(feature = "parallel")]
        {
            rayon::ThreadPoolBuilder::new()
                .num_threads(4)
                .build()
                .unwrap()
                .install(op)
        }
        #[cfg(not(feature = "parallel"))]
        {
            op()
        }
    }

    /// A sparse pseudorandom predicate: `i` matches when its hash has its low `bits` zero.
    fn matches(i: u64, bits: u32) -> bool {
        let mut x = i.wrapping_mul(0x9e37_79b9_7f4a_7c15);
        x ^= x >> 31;
        x = x.wrapping_mul(0xbf58_476d_1ce4_e5b9);
        x ^= x >> 29;
        x.trailing_zeros() >= bits
    }

    #[test]
    fn finds_the_smallest_match() {
        // Ranges starting at 0 and elsewhere, sparse and dense predicates.
        for start in [0, 1, 1000, u64::MAX - (1 << 20)] {
            for bits in [0, 1, 4, 10, 16] {
                let range = start..start.saturating_add(1 << 22);
                let expected = range.clone().find(|&i| matches(i, bits));
                let got = on_four_workers(|| {
                    find_map_first(range, || (), |(), i| matches(i, bits).then_some(i))
                });
                assert_eq!(got, expected, "start {start}, bits {bits}");
            }
        }
    }

    #[cfg(feature = "parallel")]
    #[test]
    fn chunks_tile_the_range_in_order() {
        // Short and long ranges, one ending at the top of `u64`, on 1, 3 and 16 workers.
        for range in [0..1, 7..1000, 0..1 << 24, u64::MAX - (1 << 30)..u64::MAX] {
            for threads in [1, 3, 16] {
                let schedule = Schedule::new(range.clone(), threads);
                // Each chunk starts where the last one ended, and the last ends with the range.
                let mut end = range.start;
                let mut index = 0;
                while let Some(chunk) = schedule.chunk(index) {
                    assert_eq!(chunk.start, end);
                    assert!(chunk.start < chunk.end);
                    end = chunk.end;
                    index += 1;
                }
                assert_eq!(end, range.end);
                // Claims past the end keep returning nothing.
                assert_eq!(schedule.chunk(index + 1), None);
            }
        }
    }

    #[test]
    fn reports_no_match() {
        // An empty range, and a range with no match.
        on_four_workers(|| {
            assert_eq!(find_map_first(5..5, || (), |(), i| Some(i)), None);
            assert_eq!(find_map_first(0..100_000, || (), |(), _| None::<u64>), None);
        });
    }

    #[test]
    fn reaches_the_last_candidate() {
        // The only match is the last candidate of a range ending at the top of `u64`.
        let range = u64::MAX - 100_000..u64::MAX;
        let last = u64::MAX - 1;
        let got =
            on_four_workers(|| find_map_first(range, || (), |(), i| (i == last).then_some(i)));
        assert_eq!(got, Some(last));
    }
}
