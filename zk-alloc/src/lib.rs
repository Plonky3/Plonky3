//! Bump-pointer arena allocator for ZK proving workloads.
//!
//! One mmap region split into per-thread slabs. Allocation = increment a thread-local
//! pointer; free = no-op. `begin_phase()` resets the arena: each thread's next
//! allocation starts over at the beginning of its slab, overwriting the previous
//! phase's data. Allocations that don't fit (too large, or beyond max threads) fall
//! back to the system allocator.
//!
//! Slab size defaults to 8GB per thread. Set `ZK_ALLOC_SLAB_GB` to override
//! (e.g. `ZK_ALLOC_SLAB_GB=12` for large workloads). Use `overflow_stats()`
//! to check if allocations spill to the system allocator.
//!
//! ```ignore
//! loop {
//!     begin_phase();               // arena ON; slabs reset lazily
//!     let res = heavy_work();      // fast bump increments
//!     end_phase();                 // arena OFF; new allocations go to System
//!     let copy = res.clone();      // detach from arena before next phase resets it
//! }
//! ```

use std::alloc::{GlobalAlloc, Layout};
use std::cell::Cell;
use std::sync::Once;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

mod syscall;

const DEFAULT_SLAB_GB: usize = 8;
const SLACK: usize = 4;

#[derive(Debug)]
pub struct ZkAllocator;

/// Per-thread slab size in bytes. Set once during `ensure_region()` from the
/// `ZK_ALLOC_SLAB_GB` environment variable (default: 8).
static SLAB_SIZE: AtomicUsize = AtomicUsize::new(0);

/// Incremented by `begin_phase()`. Every thread caches the last value it saw in
/// `ARENA_GEN`; when they differ, the thread resets its allocation cursor to the start
/// of its slab on the next allocation. This is how a single store on the main thread
/// "resets" every other thread's slab without any cross-thread synchronization.
static GENERATION: AtomicUsize = AtomicUsize::new(0);

/// Master switch for the arena. `true` (set by `begin_phase`) routes allocations
/// through the arena; `false` (set by `end_phase`) routes them to the system allocator.
static ARENA_ACTIVE: AtomicBool = AtomicBool::new(false);

/// Base address of the mmap'd region, or `0` before `ensure_region` runs. Read on
/// every `dealloc` to test whether a pointer belongs to us.
static REGION_BASE: AtomicUsize = AtomicUsize::new(0);

/// Total size of the mmap'd region. Set once alongside REGION_BASE.
static REGION_SIZE: AtomicUsize = AtomicUsize::new(0);

/// Synchronizes the one-time mmap so concurrent first-allocators don't race.
static REGION_INIT: Once = Once::new();

/// Monotonic counter handed out to threads to pick their slab. `fetch_add`'d once per
/// thread on its first arena allocation. Threads that get `idx >= max_threads` mark
/// themselves `ARENA_NO_SLAB` and permanently fall through to the system allocator.
static THREAD_IDX: AtomicUsize = AtomicUsize::new(0);

/// Max threads determined at init time from available_parallelism() + SLACK.
static MAX_THREADS: AtomicUsize = AtomicUsize::new(0);

static OVERFLOW_COUNT: AtomicUsize = AtomicUsize::new(0);
static OVERFLOW_BYTES: AtomicUsize = AtomicUsize::new(0);

thread_local! {
    /// Where this thread's next allocation lands. Advanced past each allocation.
    static ARENA_PTR: Cell<usize> = const { Cell::new(0) };
    /// One past the last byte of this thread's slab. An alloc fits iff
    /// `aligned + size <= ARENA_END`.
    static ARENA_END: Cell<usize> = const { Cell::new(0) };
    /// Base address of this thread's slab (`0` = not yet claimed). On reset,
    /// `ARENA_PTR` is set back to this value.
    static ARENA_BASE: Cell<usize> = const { Cell::new(0) };
    /// Last `GENERATION` value this thread observed. When the global moves past
    /// this, the next allocation resets `ARENA_PTR` to `ARENA_BASE` and updates
    /// this field.
    static ARENA_GEN: Cell<usize> = const { Cell::new(0) };
    /// `true` if this thread was created after all slabs were already claimed.
    /// Such threads skip arena logic entirely and always use the system allocator.
    static ARENA_NO_SLAB: Cell<bool> = const { Cell::new(false) };
}

/// Returns the base address of the mmap'd region, mapping it on the first call.
fn ensure_region() -> usize {
    REGION_INIT.call_once(|| {
        let slab_gb = std::env::var("ZK_ALLOC_SLAB_GB")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(DEFAULT_SLAB_GB);
        let slab_size = slab_gb << 30;
        SLAB_SIZE.store(slab_size, Ordering::Release);

        let cpus = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(8);
        let max_threads = cpus + SLACK;
        let region_size = slab_size * max_threads;

        // SAFETY: mmap_anonymous returns a page-aligned pointer or null.
        // MAP_NORESERVE means no physical memory is committed until pages are touched.
        let ptr = unsafe { syscall::mmap_anonymous(region_size) };
        if ptr.is_null() {
            std::process::abort();
        }
        unsafe { syscall::madvise(ptr, region_size, syscall::MADV_NOHUGEPAGE) };
        MAX_THREADS.store(max_threads, Ordering::Release);
        REGION_SIZE.store(region_size, Ordering::Release);
        REGION_BASE.store(ptr as usize, Ordering::Release);
    });
    REGION_BASE.load(Ordering::Acquire)
}

/// Activates the arena and resets every thread's slab. All allocations until the next
/// `end_phase()` go to the arena; the previous phase's data is overwritten in place.
pub fn begin_phase() {
    ensure_region();
    GENERATION.fetch_add(1, Ordering::Release);
    ARENA_ACTIVE.store(true, Ordering::Release);
}

/// Deactivates the arena. New allocations go to the system allocator; existing arena
/// pointers stay valid until the next `begin_phase()` resets the slabs.
pub fn end_phase() {
    ARENA_ACTIVE.store(false, Ordering::Release);
}

/// Returns (overflow_count, overflow_bytes) — allocations that fell through to System
/// because they exceeded the slab or arrived after all slabs were claimed.
pub fn overflow_stats() -> (usize, usize) {
    (
        OVERFLOW_COUNT.load(Ordering::Relaxed),
        OVERFLOW_BYTES.load(Ordering::Relaxed),
    )
}

pub fn reset_overflow_stats() {
    OVERFLOW_COUNT.store(0, Ordering::Relaxed);
    OVERFLOW_BYTES.store(0, Ordering::Relaxed);
}

/// Returns the per-thread slab size in bytes. Zero before the first `begin_phase()`.
pub fn slab_size() -> usize {
    SLAB_SIZE.load(Ordering::Relaxed)
}

#[cold]
#[inline(never)]
unsafe fn arena_alloc_cold(size: usize, align: usize) -> *mut u8 {
    let generation = GENERATION.load(Ordering::Relaxed);
    if !ARENA_NO_SLAB.get() && ARENA_GEN.get() != generation {
        let mut base = ARENA_BASE.get();
        if base == 0 {
            let region = ensure_region();
            let max = MAX_THREADS.load(Ordering::Relaxed);
            let idx = THREAD_IDX.fetch_add(1, Ordering::Relaxed);
            if idx >= max {
                ARENA_NO_SLAB.set(true);
                return unsafe {
                    std::alloc::System.alloc(Layout::from_size_align_unchecked(size, align))
                };
            }
            let slab_size = SLAB_SIZE.load(Ordering::Relaxed);
            base = region + idx * slab_size;
            ARENA_BASE.set(base);
            ARENA_END.set(base + slab_size);
        }
        ARENA_PTR.set(base);
        ARENA_GEN.set(generation);
        let aligned = base.next_multiple_of(align);
        let new_ptr = aligned + size;
        if new_ptr <= ARENA_END.get() {
            ARENA_PTR.set(new_ptr);
            return aligned as *mut u8;
        }
    }
    OVERFLOW_COUNT.fetch_add(1, Ordering::Relaxed);
    OVERFLOW_BYTES.fetch_add(size, Ordering::Relaxed);
    unsafe { std::alloc::System.alloc(Layout::from_size_align_unchecked(size, align)) }
}

// SAFETY: All pointers returned are either from our mmap'd region (valid, aligned,
// non-overlapping per thread) or from System. The arena is thread-local so no data
// races. Relaxed ordering on ARENA_ACTIVE/GENERATION is sound: worst case a thread
// sees a stale value and does one extra system-alloc before picking up the new
// generation on the next call.
unsafe impl GlobalAlloc for ZkAllocator {
    #[inline(always)]
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        if ARENA_ACTIVE.load(Ordering::Relaxed) {
            let generation = GENERATION.load(Ordering::Relaxed);
            if ARENA_GEN.get() == generation {
                let align = layout.align();
                let aligned = (ARENA_PTR.get() + align - 1) & !(align - 1);
                let new_ptr = aligned + layout.size();
                if new_ptr <= ARENA_END.get() {
                    ARENA_PTR.set(new_ptr);
                    return aligned as *mut u8;
                }
            }
            return unsafe { arena_alloc_cold(layout.size(), layout.align()) };
        }
        unsafe { std::alloc::System.alloc(layout) }
    }

    #[inline(always)]
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        let addr = ptr as usize;
        let base = REGION_BASE.load(Ordering::Relaxed);
        let region_size = REGION_SIZE.load(Ordering::Relaxed);
        if base != 0 && addr >= base && addr < base + region_size {
            return; // arena-owned pointer — free is a no-op
        }
        unsafe { std::alloc::System.dealloc(ptr, layout) };
    }

    #[inline(always)]
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        if new_size <= layout.size() {
            return ptr;
        }
        // SAFETY: new_size > layout.size() > 0, align unchanged from valid layout.
        let new_layout = unsafe { Layout::from_size_align_unchecked(new_size, layout.align()) };
        let new_ptr = unsafe { self.alloc(new_layout) };
        if !new_ptr.is_null() {
            unsafe { std::ptr::copy_nonoverlapping(ptr, new_ptr, layout.size()) };
            unsafe { self.dealloc(ptr, layout) };
        }
        new_ptr
    }
}
