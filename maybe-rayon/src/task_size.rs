//! Minimum task sizes for parallel loops.
//!
//! Splitting a loop across workers buys throughput and costs latency.
//! Both sides of that trade collapse into one number, the fewest items a task may hold:
//!
//! ```text
//!     floor >= total items  ->  one task, which is a serial loop
//!     floor <  total items  ->  total / floor tasks
//! ```
//!
//! A floor lets a call site write its loop body once.
//! Branching on a length writes the body twice, and can only ask for one task or for all.
//!
//! # Cost model
//!
//! One item is charged by the bytes it touches, at a fixed rate per byte.
//!
//! - Bytes track the element type, so the floor follows it without being restated.
//! - A packed extension element costs 32 times a 4-byte prime field element.
//! - A task therefore needs 32 times fewer of the wide items than of the narrow ones.
//!
//! The rate assumes a body with instruction-level parallelism, as a streaming loop has.
//! A body that waits on its own previous step charges a multiple of its width.
//!
//! Charging too little is safe for how a split loop is cut up.
//! It is not safe for whether the loop splits at all, which is why a slow body says so.
//!
//! # Environment overrides
//!
//! - `P3_MIN_PARALLEL_NS` sets the serial time a loop must be worth before it splits.
//! - `P3_MAX_TASK_NS` sets the time budget one task holds once a loop does split.
//! - Setting both to `0` restores rayon's own unbounded splitting, for an A/B run.
//!
//! Both are read once and then cached for the process.
//!
//! # Examples
//!
//! ```
//! use p3_maybe_rayon::prelude::*;
//!
//! let data = vec![1u64; 1 << 10];
//!
//! // One item is one 8-byte read, so 8 KiB of traffic in total.
//! // That is under the parallel budget, so this loop runs as a single task.
//! let sum: u64 = data.par_iter().with_min_task::<u64>().sum();
//! assert_eq!(sum, 1 << 10);
//!
//! // A call site that picks its own chunk length asks for the floor directly.
//! let chunk = min_task_len(data.len(), size_of::<u64>());
//! assert_eq!(chunk, data.len());
//! ```

use super::prelude::*;

/// Picoseconds one core spends per byte a memory-bound loop touches.
///
/// Measured on the fold kernel `a[i] += (b[i] - a[i]) * r`, over buffers past L2:
///
/// ```text
///     4-byte prime field  : 0.75 ns per item -> 190 ps per byte
///     16-byte extension   : 6.4  ns per item -> 400 ps per byte
/// ```
///
/// The midpoint keeps both within a factor of 1.6, which a floor can absorb.
const PICOS_PER_BYTE: u64 = 300;

/// Serial time a loop must be worth before splitting it pays, in picoseconds.
///
/// - Dispatching a rayon fan-out costs about 13 us on a 32-worker pool.
/// - That cost barely moves whether the loop is cut two ways or thirty-two.
/// - A loop worth less than one dispatch therefore runs faster on a single core.
///
/// Requiring 20 us asks a split to buy a clear win rather than a coin flip.
const MIN_PARALLEL_PICOS: u64 = 20_000_000;

/// Time one task holds once a loop does split, in picoseconds.
///
/// - The cap keeps a task's working set inside L1 or L2.
/// - It also leaves a long loop more tasks than workers, which work stealing then balances.
///
/// ```text
///     4 us at 300 ps per byte ~ 13 KiB per task
/// ```
///
/// Measured on the same fold kernel, tasks of 8 to 32 KiB beat both alternatives:
///
/// ```text
///     vs one task per worker  : up to 1.3x faster
///     vs unbounded splitting  : up to 2.8x faster
/// ```
const MAX_TASK_PICOS: u64 = 4_000_000;

/// The two time budgets that shape every floor.
#[derive(Clone, Copy)]
struct Budget {
    /// Serial time below which a loop stays on one core.
    min_parallel_picos: u64,
    /// Time one task holds once a loop splits.
    max_task_picos: u64,
}

impl Budget {
    /// The compiled-in budgets, used when the environment says nothing.
    const DEFAULT: Self = Self {
        min_parallel_picos: MIN_PARALLEL_PICOS,
        max_task_picos: MAX_TASK_PICOS,
    };
}

/// Reads the budgets from the environment, falling back to the compiled-in values.
///
/// The environment is read once, then cached for the process.
/// A floor is computed on entry to every parallel loop, so the read cannot be repeated.
#[cfg(feature = "parallel")]
fn budget() -> Budget {
    use std::sync::OnceLock;

    static V: OnceLock<Budget> = OnceLock::new();

    *V.get_or_init(|| {
        // Overrides are given in nanoseconds and stored in picoseconds.
        //
        // A budget is a tuning knob, so a typo falls back instead of taking the process down.
        let picos = |name: &str| {
            std::env::var(name)
                .ok()
                .and_then(|v| v.parse::<u64>().ok())
                .map(|ns| ns.saturating_mul(1_000))
        };
        // Each budget falls back on its own, so one override never disturbs the other.
        Budget {
            min_parallel_picos: picos("P3_MIN_PARALLEL_NS")
                .unwrap_or(Budget::DEFAULT.min_parallel_picos),
            max_task_picos: picos("P3_MAX_TASK_NS").unwrap_or(Budget::DEFAULT.max_task_picos),
        }
    })
}

/// The compiled-in budgets.
///
/// A serial build has no environment to read and no split to constrain.
#[cfg(not(feature = "parallel"))]
const fn budget() -> Budget {
    Budget::DEFAULT
}

/// Estimated time to touch one item, in picoseconds.
///
/// A zero-sized item is charged one byte, so that every item carries some cost.
const fn item_picos(item_bytes: usize) -> u64 {
    let bytes = if item_bytes == 0 {
        1
    } else {
        item_bytes as u64
    };
    bytes.saturating_mul(PICOS_PER_BYTE)
}

/// Minimum items per task, from budgets and a worker count given explicitly.
///
/// Split from the public entry points so tests can pin the arithmetic without a thread pool.
fn min_task_len_with(budget: Budget, threads: usize, len: usize, item_bytes: usize) -> usize {
    // Phase 1: price the whole loop as if it ran on one core.
    let per_item = item_picos(item_bytes);
    let total = per_item.saturating_mul(len as u64);

    // Phase 2: a cheap loop is floored at its own length.
    //
    // rayon cannot cut below the floor, so the loop stays a single task.
    // An empty loop still reports one, since a floor of zero would mean nothing.
    if total < budget.min_parallel_picos {
        return len.max(1);
    }

    // Phase 3: one task per worker is the coarsest split worth making.
    //
    // The cap then cuts that further, which keeps a long loop's tasks cache resident.
    let per_worker = total / threads.max(1) as u64;
    let task_picos = if per_worker < budget.max_task_picos {
        per_worker
    } else {
        budget.max_task_picos
    };

    // Phase 4: convert a task's time budget back into a count of items.
    //
    // An item costlier than a whole task cannot be subdivided, so it yields one per task.
    ((task_picos / per_item) as usize).max(1)
}

/// Minimum items one task must hold, for a loop of the given shape.
///
/// Use this where a call site picks its own chunk length, and pass the result as that length.
///
/// # Arguments
///
/// * `len` - total items the loop covers
/// * `item_bytes` - bytes one item touches, summed over every stream the body reads or writes
///
/// # Returns
///
/// The items one task must take.
/// Equal to the total when the loop should not split at all.
#[inline]
pub fn min_task_len(len: usize, item_bytes: usize) -> usize {
    // The worker count is what turns a time budget into a task count, so it is read here
    // rather than baked into the budgets.
    min_task_len_with(budget(), current_num_threads(), len, item_bytes)
}

/// Whether a loop of the given shape is worth splitting at all.
///
/// The iterator adapters express this same policy without a branch, so prefer them.
/// Reach for this only where the two paths differ in more than how the work is divided.
///
/// An in-place pass that rewrites its buffer as it walks it is the usual case.
///
/// # Arguments
///
/// * `len` - total items the loop covers
/// * `item_bytes` - bytes one item touches, summed over every stream the body reads or writes
#[inline]
pub fn should_split(len: usize, item_bytes: usize) -> bool {
    // A floor that reaches the whole length is exactly the answer "do not split".
    min_task_len(len, item_bytes) < len
}

/// Task-size adapters for parallel iterators.
///
/// An adapter reads the loop's own length and floors the split from the cost model.
/// A call site then states what one item costs, instead of hardcoding a task count.
pub trait TaskSizeExt: IndexedParallelIterator {
    /// Floors the split for a loop whose item is a single element of the named type.
    ///
    /// A body touching several streams per item charges for all of them, through the
    /// byte-counted adapter instead.
    #[inline]
    fn with_min_task<T>(self) -> impl IndexedParallelIterator<Item = Self::Item>
    where
        Self: Sized,
    {
        // One element per item is just the byte-counted form at that element's width.
        self.with_min_task_bytes(size_of::<T>())
    }

    /// Floors the split for a loop whose item spans a given number of bytes.
    ///
    /// The count sums every stream one item reads or writes.
    ///
    /// Use this when the width is known only at run time, such as a chunk of a matrix row.
    #[inline]
    fn with_min_task_bytes(
        self,
        item_bytes: usize,
    ) -> impl IndexedParallelIterator<Item = Self::Item>
    where
        Self: Sized,
    {
        // An indexed iterator already knows how long it is, so the call site never repeats it.
        let min_len = min_task_len(self.len(), item_bytes);

        // A serial build has no split to constrain, and drops the floor on the floor.
        self.with_min_len(min_len)
    }
}

impl<I: IndexedParallelIterator> TaskSizeExt for I {}

#[cfg(test)]
mod tests {
    use std::vec::Vec;

    use super::*;

    /// Budgets used by every arithmetic test, in picoseconds.
    const B: Budget = Budget {
        min_parallel_picos: 20_000_000,
        max_task_picos: 4_000_000,
    };

    #[test]
    fn short_loops_stay_on_one_core() {
        // A loop worth less than the parallel budget is floored at its own length.
        // rayon cannot split below the floor, so the loop runs as a single task.
        //
        //     1000 items * 4 B * 300 ps = 1.2 us < 20 us
        assert_eq!(min_task_len_with(B, 32, 1000, 4), 1000);

        // The same loop over a wider item crosses the budget and does split.
        //
        //     1000 items * 128 B * 300 ps = 38.4 us >= 20 us
        assert!(min_task_len_with(B, 32, 1000, 128) < 1000);

        // An empty loop still reports a usable floor rather than zero.
        assert_eq!(min_task_len_with(B, 32, 0, 4), 1);
    }

    #[test]
    fn floor_scales_inversely_with_item_width() {
        // Invariant: a task holds a fixed amount of work, whatever the item width.
        // So the item count must fall by exactly the factor the item widens.
        //
        //     1 << 22 items of 4 B is 16 MiB, far past the parallel budget
        let len = 1 << 22;
        let narrow = min_task_len_with(B, 32, len, 4);
        let wide = min_task_len_with(B, 32, len, 16);
        assert_eq!(narrow / 4, wide);

        // The cap binds at this length, so a task holds exactly the task budget.
        // Integer division truncates, which is why the ratio is read the other way above.
        //
        //     4 us / (4 B * 300 ps)  = 3333 items
        //     4 us / (16 B * 300 ps) =  833 items
        assert_eq!((narrow, wide), (3333, 833));
    }

    #[test]
    fn tasks_are_capped_below_one_per_worker() {
        // A loop long enough for the cap to bind yields far more tasks than workers.
        // That is what keeps each task cache resident and leaves slack for stealing.
        let len = 1 << 22;
        let floor = min_task_len_with(B, 32, len, 4);
        assert!(len / floor > 32);

        // Just past the parallel budget the cap does not bind, and one task per worker
        // is the coarsest split made.
        //
        //     20001 items * 4 B * 300 ps = 24.0 us, over 32 workers = 750 ns per task
        let len = 20_001;
        let floor = min_task_len_with(B, 32, len, 4);
        assert_eq!(floor, 625);
        assert!(len / floor <= 32);
    }

    #[test]
    fn costly_items_fall_back_to_one_per_task() {
        // An item worth more than a whole task cannot be subdivided further.
        //
        // One item per task is also where an unfloored rayon loop already sits.
        // So undercharging an arithmetic-bound body can never cut it more finely than that.
        //
        //     4 us budget / (64 KiB * 300 ps) = 0.2 -> floored to 1
        assert_eq!(min_task_len_with(B, 32, 1 << 20, 64 << 10), 1);
    }

    #[test]
    fn zero_sized_items_are_charged_one_byte() {
        // Charging nothing would make the total zero and leave every such loop serial.
        // One byte per item keeps a long loop over zero-sized items splittable.
        assert_eq!(item_picos(0), PICOS_PER_BYTE);
        assert!(min_task_len_with(B, 32, 1 << 30, 0) < 1 << 30);
    }

    #[test]
    fn a_single_worker_never_splits() {
        // Invariant: with one worker, one task per worker is the whole loop.
        // The cap still applies, so the floor is the task budget rather than the length.
        //
        //     4 us / (4 B * 300 ps) = 3333 items
        assert_eq!(min_task_len_with(B, 1, 1 << 20, 4), 3333);
    }

    #[test]
    fn should_split_agrees_with_the_floor() {
        // Invariant: the predicate is exactly the question the floor already answers.
        //
        //     1000 items of 4 B is 1.2 us, under the 20 us budget -> one task
        assert!(!should_split(1000, 4));

        //     1000 items of 128 B is 38.4 us, over it -> many tasks
        assert!(should_split(1000, 128));
    }

    #[test]
    fn adapters_preserve_iteration() {
        // The adapters constrain only how work is divided, never what it covers.
        //
        // Fixture state: 1000 items of 8 B, worth 2.4 us, well under the 20 us budget.
        //     -> the loop runs as a single task, and every item is still visited once.
        let data: Vec<u64> = (0..1000).collect();
        let expected = 1000 * 999 / 2;

        assert_eq!(
            data.par_iter().with_min_task::<u64>().sum::<u64>(),
            expected
        );
        assert_eq!(
            data.par_iter().with_min_task_bytes(8).sum::<u64>(),
            expected
        );

        // A width that does cross the budget splits the same loop into many tasks,
        // and the answer is unchanged.
        assert_eq!(
            data.par_iter().with_min_task_bytes(1 << 20).sum::<u64>(),
            expected
        );
    }
}
