//! Dispatch of an indexed loop cut into pieces up front.
//!
//! Rayon splits a loop lazily and wakes one worker per steal.
//! A short loop pays that chain of wake-ups before every worker is busy.
//!
//! Here the loop is cut into a few pieces per worker before anyone starts.
//! Every participant, the caller included, claims pieces from one counter.
//!
//! Helpers wake each other as a tree, so no wake-up sits on the caller's path.
//! The caller waits only for the helpers that joined; a late one finds the loop closed and leaves.

use alloc::boxed::Box;
use alloc::sync::Arc;
use alloc::vec::Vec;
use core::any::Any;
use core::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::panic::AssertUnwindSafe;
use std::sync::Mutex;

use rayon::current_num_threads;
use rayon::iter::IndexedParallelIterator;
use rayon::iter::plumbing::{Producer, ProducerCallback};

/// Pieces per worker.
///
/// Enough for a straggler to be rebalanced, few enough that cutting stays cheap.
const PIECES_PER_WORKER: usize = 4;

/// A loop cut into pieces of `piece` items, the last possibly shorter.
struct Pieces<P> {
    /// Each piece, taken once by whoever claims its index.
    slots: Vec<Mutex<Option<P>>>,
    /// The next unclaimed index.
    next: AtomicUsize,
    /// Items per piece.
    piece: usize,
}

impl<P: Producer> Pieces<P> {
    /// Cuts a loop of `len` items into pieces.
    ///
    /// Hands the producer back when the loop should stay whole.
    fn cut(producer: P, len: usize) -> Result<Self, P> {
        let workers = current_num_threads();
        // A piece never goes below the producer's floor (`with_min_len`).
        let floor = producer.min_len().max(1);
        if workers <= 1 || len <= floor {
            return Err(producer);
        }
        let piece = len.div_ceil(workers * PIECES_PER_WORKER).max(floor);
        let count = len.div_ceil(piece);
        let mut slots = Vec::with_capacity(count);
        let mut rest = producer;
        for _ in 1..count {
            let (head, tail) = rest.split_at(piece);
            slots.push(Mutex::new(Some(head)));
            rest = tail;
        }
        slots.push(Mutex::new(Some(rest)));
        Ok(Self {
            slots,
            next: AtomicUsize::new(0),
            piece,
        })
    }

    /// Runs `work(index, piece)` on every piece, the caller included.
    fn run(&self, work: impl Fn(usize, P) + Sync) {
        let drain = || {
            loop {
                let i = self.next.fetch_add(1, Ordering::Relaxed);
                let Some(slot) = self.slots.get(i) else {
                    break;
                };
                // Uncontended: the counter hands index `i` to one participant.
                let piece = slot
                    .lock()
                    .unwrap()
                    .take()
                    .expect("a piece is claimed once");
                work(i, piece);
            }
        };
        let left = || self.next.load(Ordering::Relaxed) < self.slots.len();
        Gate::run(&drain, &left, self.slots.len());
    }
}

/// The handshake between a caller and the helpers it wakes.
///
/// A helper counts itself in, then runs the job only if it is still open.
///
/// The caller closes the job, then waits until no helper is inside.
///
/// Both sides use `SeqCst`, so a helper the caller does not wait for sees the job closed.
struct Gate {
    /// The job, its lifetime erased; read only while the caller waits.
    job: *const (dyn Fn() + Sync),
    /// Whether unclaimed work is left, erased like `job`.
    left: *const (dyn Fn() -> bool + Sync),
    /// Helpers still to wake.
    to_wake: AtomicUsize,
    /// Helpers inside the job.
    entered: AtomicUsize,
    /// Whether the caller has stopped waiting for newcomers.
    closed: AtomicBool,
    /// The first panic a helper hit, re-raised on the caller.
    panic: Mutex<Option<Box<dyn Any + Send>>>,
}

// SAFETY: `job` is `Sync`, and the gate's protocol bounds every dereference by the dispatch.
unsafe impl Send for Gate {}
// SAFETY: as above.
unsafe impl Sync for Gate {}

impl Gate {
    /// Runs `job` on the caller and on up to `pieces - 1` helpers.
    ///
    /// A caller outside the pool enters it first, so it never competes with the workers for a core.
    ///
    /// Helpers are woken as a tree: each one that joins wakes up to two more while work is left.
    /// The wake-ups then overlap, and none sits on the caller's path.
    fn run(job: &(dyn Fn() + Sync), left: &(dyn Fn() -> bool + Sync), pieces: usize) {
        if rayon::current_thread_index().is_none() {
            return rayon::scope(|_| Self::run(job, left, pieces));
        }
        // SAFETY: a helper reads `job` only inside an open gate, and the caller returns only once the gate is closed and empty.
        let job: *const (dyn Fn() + Sync) = unsafe { core::mem::transmute(job) };
        // SAFETY: as for `job`.
        let left: *const (dyn Fn() -> bool + Sync) = unsafe { core::mem::transmute(left) };
        let gate = Arc::new(Self {
            job,
            left,
            to_wake: AtomicUsize::new(pieces.min(current_num_threads()) - 1),
            entered: AtomicUsize::new(0),
            closed: AtomicBool::new(false),
            panic: Mutex::new(None),
        });
        Self::wake(&gate);
        // SAFETY: the job is still the caller's own borrow.
        let mine = std::panic::catch_unwind(AssertUnwindSafe(|| unsafe { (*gate.job)() }));
        gate.closed.store(true, Ordering::SeqCst);
        while gate.entered.load(Ordering::SeqCst) != 0 {
            core::hint::spin_loop();
        }
        if let Err(payload) = mine {
            std::panic::resume_unwind(payload);
        }
        let helper_panic = gate.panic.lock().unwrap().take();
        if let Some(payload) = helper_panic {
            std::panic::resume_unwind(payload);
        }
    }

    /// Wakes up to two more helpers, if any are still wanted.
    fn wake(gate: &Arc<Self>) {
        for _ in 0..2 {
            let wanted = gate
                .to_wake
                .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |n| n.checked_sub(1));
            if wanted.is_err() {
                return;
            }
            let shared = Arc::clone(gate);
            rayon::spawn(move || shared.join());
        }
    }

    /// A helper's side: wake the next helpers, then run the job if it is still open.
    fn join(self: Arc<Self>) {
        self.entered.fetch_add(1, Ordering::SeqCst);
        if !self.closed.load(Ordering::SeqCst) {
            // SAFETY: the gate is open, so the caller is still waiting.
            if unsafe { (*self.left)() } {
                Self::wake(&self);
            }
            // SAFETY: the gate was open after this helper counted in, so the caller is still waiting.
            if let Err(payload) =
                std::panic::catch_unwind(AssertUnwindSafe(|| unsafe { (*self.job)() }))
            {
                self.panic.lock().unwrap().get_or_insert(payload);
            }
        }
        self.entered.fetch_sub(1, Ordering::SeqCst);
    }
}

/// Runs `op` on every item.
pub(crate) fn for_each<I, F>(iter: I, op: F)
where
    I: IndexedParallelIterator,
    F: Fn(I::Item) + Sync,
{
    struct Callback<F>(usize, F);

    impl<T, F: Fn(T) + Sync> ProducerCallback<T> for Callback<F> {
        type Output = ();

        fn callback<P: Producer<Item = T>>(self, producer: P) {
            let Self(len, op) = self;
            match Pieces::cut(producer, len) {
                Err(whole) => whole.into_iter().for_each(op),
                Ok(pieces) => pieces.run(|_, piece| piece.into_iter().for_each(&op)),
            }
        }
    }

    let len = iter.len();
    iter.with_producer(Callback(len, op));
}

/// Maps every item, collecting the results in order.
pub(crate) fn map_collect<I, B, F>(iter: I, map_op: F) -> Vec<B>
where
    I: IndexedParallelIterator,
    B: Send,
    F: Fn(I::Item) -> B + Sync,
{
    struct Callback<F>(usize, F);

    /// The output's base pointer, carried into the workers.
    struct Out<B>(*mut B);
    // SAFETY: each piece writes only its own index range of the output.
    unsafe impl<B: Send> Sync for Out<B> {}

    impl<B> Out<B> {
        /// Slot `i` of the output.
        ///
        /// A method, so a closure captures the wrapper rather than its bare pointer.
        const fn at(&self, i: usize) -> *mut B {
            self.0.wrapping_add(i)
        }
    }

    impl<T, B: Send, F: Fn(T) -> B + Sync> ProducerCallback<T> for Callback<F> {
        type Output = Vec<B>;

        fn callback<P: Producer<Item = T>>(self, producer: P) -> Vec<B> {
            let Self(len, map_op) = self;
            match Pieces::cut(producer, len) {
                Err(whole) => whole.into_iter().map(map_op).collect(),
                Ok(pieces) => {
                    let mut out: Vec<B> = Vec::with_capacity(len);
                    let base = Out(out.as_mut_ptr());
                    pieces.run(|i, piece| {
                        let start = i * pieces.piece;
                        for (j, item) in piece.into_iter().enumerate() {
                            // SAFETY: piece `i` alone writes slots `start..start + piece.len()`, all below `len`.
                            // A panic leaks the slots written so far.
                            unsafe { base.at(start + j).write(map_op(item)) };
                        }
                    });
                    // SAFETY: the pieces tile `0..len`, and each wrote every slot of its range.
                    unsafe { out.set_len(len) };
                    out
                }
            }
        }
    }

    let len = iter.len();
    iter.with_producer(Callback(len, map_op))
}

/// Folds each piece from `identity`, then reduces the pieces in index order.
///
/// The order is the items' own, so `reduce_op` need only be associative.
pub(crate) fn fold_reduce<I, Acc, Id, F, R>(iter: I, identity: Id, fold_op: F, reduce_op: R) -> Acc
where
    I: IndexedParallelIterator,
    Acc: Send,
    Id: Fn() -> Acc + Sync,
    F: Fn(Acc, I::Item) -> Acc + Sync,
    R: Fn(Acc, Acc) -> Acc,
{
    struct Callback<Id, F, R>(usize, Id, F, R);

    impl<T, Acc, Id, F, R> ProducerCallback<T> for Callback<Id, F, R>
    where
        Acc: Send,
        Id: Fn() -> Acc + Sync,
        F: Fn(Acc, T) -> Acc + Sync,
        R: Fn(Acc, Acc) -> Acc,
    {
        type Output = Acc;

        fn callback<P: Producer<Item = T>>(self, producer: P) -> Acc {
            let Self(len, identity, fold_op, reduce_op) = self;
            match Pieces::cut(producer, len) {
                Err(whole) => whole.into_iter().fold(identity(), fold_op),
                Ok(pieces) => {
                    let partials: Vec<Mutex<Option<Acc>>> =
                        (0..pieces.slots.len()).map(|_| Mutex::new(None)).collect();
                    pieces.run(|i, piece| {
                        let acc = piece.into_iter().fold(identity(), &fold_op);
                        *partials[i].lock().unwrap() = Some(acc);
                    });
                    partials
                        .into_iter()
                        .map(|p| p.into_inner().unwrap().expect("every piece ran"))
                        .reduce(reduce_op)
                        .unwrap_or_else(identity)
                }
            }
        }
    }

    let len = iter.len();
    iter.with_producer(Callback(len, identity, fold_op, reduce_op))
}

#[cfg(test)]
mod tests {
    use alloc::vec;
    use alloc::vec::Vec;
    use core::sync::atomic::{AtomicUsize, Ordering};

    use rayon::prelude::*;

    use super::{fold_reduce, for_each, map_collect};

    const LENGTHS: [usize; 9] = [0, 1, 2, 3, 31, 128, 129, 1000, 100_003];

    #[test]
    fn every_item_runs_once() {
        for n in LENGTHS {
            let hits: Vec<AtomicUsize> = (0..n).map(|_| AtomicUsize::new(0)).collect();
            for_each((0..n).into_par_iter(), |i| {
                hits[i].fetch_add(1, Ordering::Relaxed);
            });
            assert!(
                hits.iter().all(|h| h.load(Ordering::Relaxed) == 1),
                "n = {n}"
            );
        }
    }

    #[test]
    fn map_collect_keeps_index_order() {
        for n in LENGTHS {
            let out = map_collect((0..n).into_par_iter().with_min_len(7), |i| 3 * i);
            assert_eq!(out, (0..n).map(|i| 3 * i).collect::<Vec<_>>(), "n = {n}");
        }
    }

    #[test]
    fn fold_reduce_keeps_item_order() {
        // Concatenation is associative but not commutative, so any reordering shows.
        for n in LENGTHS {
            let got = fold_reduce(
                (0..n).into_par_iter(),
                Vec::new,
                |mut acc, i| {
                    acc.push(i);
                    acc
                },
                |mut a, mut b| {
                    a.append(&mut b);
                    a
                },
            );
            assert_eq!(got, (0..n).collect::<Vec<_>>(), "n = {n}");
        }
    }

    #[test]
    fn a_panic_reaches_the_caller_and_the_pool_survives() {
        let result = std::panic::catch_unwind(|| {
            for_each((0..10_000).into_par_iter(), |i| {
                assert_ne!(i, 5_000, "boom");
            });
        });
        assert!(result.is_err());

        // The pool still runs a later loop to completion.
        let out = map_collect((0..10_000).into_par_iter(), |i| i);
        assert_eq!(out.len(), 10_000);
    }

    #[test]
    fn nested_loops_complete() {
        let count = AtomicUsize::new(0);
        for_each((0..64).into_par_iter(), |_| {
            for_each((0..64).into_par_iter(), |_| {
                count.fetch_add(1, Ordering::Relaxed);
            });
        });
        assert_eq!(count.load(Ordering::Relaxed), 64 * 64);
    }

    #[test]
    fn zipped_chunks_split_in_lockstep() {
        let mut a = vec![0u64; 10_007];
        let b: Vec<u64> = (0..10_007).collect();
        for_each(a.par_chunks_mut(13).zip(b.par_chunks(13)), |(a, b)| {
            a.copy_from_slice(b);
        });
        assert_eq!(a, b);
    }
}
