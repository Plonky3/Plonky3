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
//! One item is charged by the bytes it moves, at a fixed rate per byte.
//!
//! Moved bytes are counted as every element the body reads plus every element it writes.
//! A read-modify-write of one element is therefore two, not one.
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
//! # Pool size
//!
//! The time a loop must be worth scales with the number of workers.
//! Waking a wider pool costs more, so a wider pool demands a longer loop.
//!
//! A pool of one worker never splits at all, which is also what a build without rayon gets.
//!
//! # Environment overrides
//!
//! - `P3_MIN_PARALLEL_NS` fixes the serial time a loop must be worth, ignoring the pool size.
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
//! // One item is one 8-byte read, so 8 KiB moved in total.
//! // That is under the parallel budget, so this loop runs as a single task.
//! let sum: u64 = data.par_iter().with_min_task::<u64>().sum();
//! assert_eq!(sum, 1 << 10);
//!
//! // A call site that picks its own chunk length asks for the floor directly.
//! let chunk = min_task_len(data.len(), size_of::<u64>());
//! assert_eq!(chunk, data.len());
//! ```

use super::prelude::*;

/// Picoseconds one core spends per byte a memory-bound loop moves.
///
/// Measured on the fold kernel `a[i] += (b[i] - a[i]) * r`, over buffers past L2.
/// One item there reads two elements and writes one, so it moves three:
///
/// ```text
///     4-byte prime field  : 0.75 ns / 12 B ->  62 ps per byte
///     16-byte extension   : 6.4  ns / 48 B -> 133 ps per byte
/// ```
///
/// The midpoint keeps both within a factor of 1.6, which a floor can absorb.
const PICOS_PER_BYTE: u64 = 100;

/// Serial time a loop must be worth per worker in the pool, before splitting pays.
///
/// Dispatching a fan-out costs roughly 0.4 us per worker:
///
/// ```text
///     16 workers :  7 us per dispatch
///     32 workers : 13 us per dispatch
/// ```
///
/// - Asking 0.625 us per worker demands a loop the model prices at 1.5 dispatches.
/// - Scaling with the pool is what keeps a 96-core host from splitting at a loss.
///
/// ```text
///     16 workers : 10 us before a split is worth it
///     32 workers : 20 us
///     96 workers : 60 us
/// ```
///
/// How much real work that is depends on how closely a body matches the rate.
/// A narrow-field fold is priced about 1.6x high, so its gate sits nearer one dispatch.
///
/// Both dispatch figures come from large pools.
///
/// The rule is extrapolated through the origin down to two workers, which is not measured.
const MIN_PARALLEL_PICOS_PER_WORKER: u64 = 625_000;

/// Time one task holds once a loop does split, in picoseconds.
///
/// - The cap keeps a task's working set inside L1 or L2.
/// - It also leaves a long loop more tasks than workers, which work stealing then balances.
///
/// ```text
///     2 us at 100 ps per byte = 20 KiB moved per task
/// ```
///
/// Measured on the same fold kernel, tasks of that size beat both alternatives:
///
/// ```text
///     vs one task per worker  : up to 1.3x faster
///     vs unbounded splitting  : up to 2.8x faster
/// ```
const MAX_TASK_PICOS: u64 = 2_000_000;

/// The two time budgets that shape every floor.
#[derive(Clone, Copy)]
struct Budget {
    /// Serial time below which a loop stays on one core.
    ///
    /// Absent means scale it with the number of workers.
    min_parallel_picos: Option<u64>,
    /// Time one task holds once a loop splits.
    max_task_picos: u64,
}

impl Budget {
    /// The compiled-in budgets, used when the environment says nothing.
    const DEFAULT: Self = Self {
        min_parallel_picos: None,
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
        // Start from the compiled-in budgets, then apply whatever the environment states.
        // Each budget is overridden on its own, so one variable never disturbs the other.
        let mut budget = Budget::DEFAULT;

        // An override fixes the gate outright, which is what makes an A/B run reproducible.
        if let Some(picos) = picos("P3_MIN_PARALLEL_NS") {
            budget.min_parallel_picos = Some(picos);
        }
        if let Some(picos) = picos("P3_MAX_TASK_NS") {
            budget.max_task_picos = picos;
        }
        budget
    })
}

/// The compiled-in budgets.
///
/// A build without rayon has no environment to read, and reports a single worker below.
#[cfg(not(feature = "parallel"))]
const fn budget() -> Budget {
    Budget::DEFAULT
}

/// Estimated time to move one item, in picoseconds.
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
///
/// # Returns
///
/// A count in `1..=max(len, 1)`, so a chunked call site can divide by it without checking.
fn min_task_len_with(budget: Budget, threads: usize, len: usize, item_bytes: usize) -> usize {
    // Phase 1: a pool that cannot hand work anywhere keeps the whole loop as one task.
    //
    // This is the answer a build without rayon gets, since it reports a single worker.
    if threads <= 1 {
        return len.max(1);
    }

    // Phase 2: price the whole loop as if it ran on one core.
    let per_item = item_picos(item_bytes);
    let total = per_item.saturating_mul(len as u64);

    // Phase 3: a cheap loop is floored at its own length.
    //
    // rayon cannot cut below the floor, so the loop stays a single task.
    // An empty loop still reports one, since a floor of zero would mean nothing.
    let min_parallel = budget
        .min_parallel_picos
        .unwrap_or_else(|| MIN_PARALLEL_PICOS_PER_WORKER.saturating_mul(threads as u64));
    if total < min_parallel {
        return len.max(1);
    }

    // Phase 4: one task per worker is the coarsest split worth making.
    //
    // The cap then cuts that further, which keeps a long loop's tasks cache resident.
    let per_worker = total / threads as u64;
    let task_picos = if per_worker < budget.max_task_picos {
        per_worker
    } else {
        budget.max_task_picos
    };

    // Phase 5: convert a task's time budget back into a count of items.
    //
    // An item costlier than a whole task cannot be subdivided, so it yields one per task.
    // The count cannot exceed the length, because a task holds at most the whole loop.
    ((task_picos / per_item) as usize).max(1).min(len.max(1))
}

/// Minimum items one task must hold, for a loop of the given shape.
///
/// Use this where a call site picks its own chunk length, and pass the result as that length.
///
/// # Arguments
///
/// * `len` - total items the loop covers
/// * `item_bytes` - bytes one item moves: every element read, plus every element written
///
/// # Returns
///
/// A count in `1..=max(len, 1)`, never zero, so it is always a usable chunk length.
/// Equal to the total when the loop should not split at all.
#[inline]
pub fn min_task_len(len: usize, item_bytes: usize) -> usize {
    // The worker count both scales the gate and turns a time budget into a task count,
    // so it is read here rather than baked into the budgets.
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
/// * `item_bytes` - bytes one item moves: every element read, plus every element written
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
    /// A body moving several elements per item charges for all of them.
    ///
    /// Such a body asks through the byte-counted adapter instead.
    #[inline]
    fn with_min_task<T>(self) -> impl IndexedParallelIterator<Item = Self::Item>
    where
        Self: Sized,
    {
        // One element per item is just the byte-counted form at that element's width.
        self.with_min_task_bytes(size_of::<T>())
    }

    /// Floors the split for a loop whose item moves a given number of bytes.
    ///
    /// The count sums every element one item reads and every element it writes.
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

    /// Budgets used by the arithmetic tests, with the gate left to scale with the pool.
    const B: Budget = Budget::DEFAULT;

    /// Worker count the arithmetic tests price against, so the gate lands at 20 us.
    const T: usize = 32;

    #[test]
    fn short_loops_stay_on_one_core() {
        // A loop worth less than the parallel budget is floored at its own length.
        // rayon cannot split below the floor, so the loop runs as a single task.
        //
        //     1000 items * 4 B * 100 ps = 0.4 us < 20 us
        assert_eq!(min_task_len_with(B, T, 1000, 4), 1000);

        // The same loop over a wider item crosses the budget and does split.
        //
        //     1000 items * 256 B * 100 ps = 25.6 us >= 20 us
        assert!(min_task_len_with(B, T, 1000, 256) < 1000);

        // An empty loop still reports a usable floor rather than zero.
        assert_eq!(min_task_len_with(B, T, 0, 4), 1);
    }

    #[test]
    fn the_gate_grows_with_the_pool() {
        // Invariant: waking a wider pool costs more, so a wider pool demands a longer loop.
        //
        // Fixture state: a loop worth 20.0 us, which is 32 workers' worth of dispatch.
        //
        //     50001 items * 4 B * 100 ps = 20.0 us
        //     gate at 32 workers = 32 * 625 ps = 20.0 us  -> splits
        //     gate at 64 workers = 64 * 625 ps = 40.0 us  -> stays whole
        let len = 50_001;
        assert!(min_task_len_with(B, 32, len, 4) < len);
        assert_eq!(min_task_len_with(B, 64, len, 4), len);
    }

    #[test]
    fn an_override_fixes_the_gate_against_the_pool() {
        // An override states the gate outright, which is what makes an A/B run reproducible.
        //
        // Fixture state: gate pinned at 1 us, against a loop worth 20 us.
        //     -> both pool sizes split, where the scaling gate would have disagreed.
        let fixed = Budget {
            min_parallel_picos: Some(1_000_000),
            max_task_picos: MAX_TASK_PICOS,
        };
        let len = 50_001;
        assert!(min_task_len_with(fixed, 32, len, 4) < len);
        assert!(min_task_len_with(fixed, 64, len, 4) < len);
    }

    #[test]
    fn floor_scales_inversely_with_item_width() {
        // Invariant: a task holds a fixed amount of work, whatever the item width.
        // So the item count must fall by exactly the factor the item widens.
        //
        //     1 << 22 items of 4 B moves 16 MiB, far past the parallel budget
        let len = 1 << 22;
        let narrow = min_task_len_with(B, T, len, 4);
        let wide = min_task_len_with(B, T, len, 16);
        assert_eq!(narrow, 4 * wide);

        // The cap binds at this length, so a task holds exactly the task budget.
        //
        //     2 us / (4 B * 100 ps)  = 5000 items
        //     2 us / (16 B * 100 ps) = 1250 items
        assert_eq!((narrow, wide), (5000, 1250));
    }

    #[test]
    fn tasks_are_capped_below_one_per_worker() {
        // A loop long enough for the cap to bind yields far more tasks than workers.
        // That is what keeps each task cache resident and leaves slack for stealing.
        let len = 1 << 22;
        let floor = min_task_len_with(B, T, len, 4);
        assert!(len / floor > T);

        // Just past the parallel budget the cap does not bind.
        //
        // One task per worker is then the coarsest split made.
        //
        //     50001 items worth 20.0 us, over 32 workers = 625 ns per task
        let len = 50_001;
        let floor = min_task_len_with(B, T, len, 4);
        assert_eq!(floor, 1562);
        assert!(len / floor <= T);
    }

    #[test]
    fn costly_items_fall_back_to_one_per_task() {
        // An item worth more than a whole task cannot be subdivided further.
        //
        // One item per task is also where an unfloored rayon loop already sits.
        // So undercharging an arithmetic-bound body can never cut it more finely than that.
        //
        //     2 us budget / (64 KiB * 100 ps) = 0.003 -> floored to 1
        assert_eq!(min_task_len_with(B, T, 1 << 20, 64 << 10), 1);
    }

    #[test]
    fn zero_sized_items_are_charged_one_byte() {
        // Charging nothing would make the total zero and leave every such loop serial.
        // One byte per item keeps a long loop over zero-sized items splittable.
        assert_eq!(item_picos(0), PICOS_PER_BYTE);
        assert!(min_task_len_with(B, T, 1 << 30, 0) < 1 << 30);
    }

    #[test]
    fn a_pool_of_one_never_splits() {
        // Invariant: a single worker has nobody to hand work to, so there is no split.
        // The whole loop is one task however expensive it is.
        assert_eq!(min_task_len_with(B, 1, 1 << 20, 4), 1 << 20);
        assert_eq!(min_task_len_with(B, 1, 1 << 20, 64 << 10), 1 << 20);

        // A pool reported as empty is treated the same way rather than dividing by zero.
        assert_eq!(min_task_len_with(B, 0, 1 << 20, 4), 1 << 20);
    }

    #[test]
    fn floor_is_always_a_usable_chunk_length() {
        // Invariant: every call site divides a length by this floor, or chunks by it.
        //
        //     lower bound 1        -> par_chunks never sees zero and panics
        //     upper bound len      -> chunk index arithmetic never runs off the end
        //
        // Swept exhaustively rather than sampled, since the grid is small and fixed.
        let fixed = Budget {
            min_parallel_picos: Some(1_000_000),
            max_task_picos: MAX_TASK_PICOS,
        };
        for budget in [B, fixed] {
            for threads in [0usize, 1, 2, 8, 32, 96, 192, 1 << 20] {
                for len in [0usize, 1, 2, 3, 5, 17, 1023, 1 << 10, 1 << 20, usize::MAX] {
                    for item_bytes in [0usize, 1, 3, 4, 8, 16, 32, 4096, 1 << 20] {
                        let floor = min_task_len_with(budget, threads, len, item_bytes);
                        assert!(
                            (1..=len.max(1)).contains(&floor),
                            "floor {floor} out of range for \
                             threads={threads} len={len} item_bytes={item_bytes}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn should_split_agrees_with_the_floor() {
        // Invariant: the predicate is exactly the question the floor already answers.
        //
        // Read through the public entry points.
        //
        // So this holds on any host pool, in either feature configuration.
        for len in [0usize, 1, 2, 1023, 1 << 10, 1 << 20] {
            for item_bytes in [1usize, 4, 16, 256, 1 << 20] {
                assert_eq!(
                    should_split(len, item_bytes),
                    min_task_len(len, item_bytes) < len
                );
            }
        }
    }

    /// A build without rayon reports one worker, so no loop is ever worth splitting.
    ///
    /// A serial build keeps every call site on its in-place, short-circuiting arm.
    ///
    /// This pins that behaviour, which those call sites rely on.
    #[cfg(not(feature = "parallel"))]
    #[test]
    fn a_serial_build_never_splits() {
        // Fixture state: a loop far past any parallel budget, over a 12-byte item.
        //
        //     1 << 19 items * 12 B * 100 ps = 629 us, which a 32-worker pool would split
        assert_eq!(current_num_threads(), 1);
        assert_eq!(min_task_len(1 << 19, 12), 1 << 19);
        assert!(!should_split(1 << 19, 12));
    }

    #[test]
    fn adapters_preserve_iteration() {
        // The adapters constrain only how work is divided, never what it covers.
        //
        // Fixture state: 1000 items of 8 B, worth 0.8 us, well under the 20 us budget.
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
