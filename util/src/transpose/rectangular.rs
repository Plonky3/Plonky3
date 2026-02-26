//! High-performance matrix transpose for generic `Copy` types.
//!
//! This module provides an optimized **out-of-place** matrix transpose.
//!
//! # Overview
//!
//! Matrix transposition converts a row-major matrix into its column-major equivalent.
//! For a matrix `A` with dimensions `height × width`:
//!
//! ```text
//!     A[i][j] → A^T[j][i]
//! ```
//!
//! In memory (row-major layout), element at position `(row, col)` is stored at:
//! - **Input**: `input[row * width + col]`
//! - **Output**: `output[col * height + row]`
//!
//! # Architecture-Specific Optimizations
//!
//! On **ARM64** (aarch64), this module uses NEON SIMD intrinsics for:
//! - **4-byte elements** (typical for 32-bit field elements like `MontyField31`, `BabyBear`)
//!   using a 2-stage butterfly (`vtrn1q_u32`/`vtrn2q_u32` then `vtrn1q_u64`/`vtrn2q_u64`).
//! - **8-byte elements** (typical for 64-bit field elements like `Goldilocks`)
//!   using a simpler 1-stage butterfly (`vtrn1q_u64`/`vtrn2q_u64`) on pairs of registers.
//!
//! On other architectures or for other element sizes, it falls back to the `transpose` crate.
//!
//! # Key Optimizations
//!
//! ## NEON SIMD Registers (128-bit)
//!
//! ARM64 NEON provides 32 vector registers, each holding 128 bits.
//! - For 32-bit (4-byte) elements, each register holds exactly **`BLOCK_SIZE` elements**.
//! - A `BLOCK_SIZE`×`BLOCK_SIZE` block (`BLOCK_SIZE`^2 elements) fits perfectly in **`BLOCK_SIZE` registers**.
//!
//! ```text
//!     ┌─────────────────────────────────┐
//!     │  q0 = [ a00, a01, a02, a03 ]    │  ← 128 bits = 4 × 32-bit
//!     │  q1 = [ a10, a11, a12, a13 ]    │
//!     │  q2 = [ a20, a21, a22, a23 ]    │
//!     │  q3 = [ a30, a31, a32, a33 ]    │
//!     └─────────────────────────────────┘
//! ```
//!
//! ## In-Register Transpose (Butterfly Network)
//!
//! We transpose a `BLOCK_SIZE`×`BLOCK_SIZE` block entirely in registers using a 2-stage butterfly:
//!
//! **Stage 1**: Swap pairs of 32-bit elements using `TRN1`/`TRN2`
//! **Stage 2**: Swap pairs of 64-bit elements using `TRN1`/`TRN2` on reinterpreted u64
//!
//! ```text
//!     Input:          After Stage 1:      After Stage 2 (Output):
//!     ┌─────────────┐ ┌─────────────┐     ┌─────────────┐
//!     │ a b │ c d   │ │ a e │ c g   │     │ a e │ i m   │
//!     │ e f │ g h   │ │ b f │ d h   │     │ b f │ j n   │
//!     │─────┼───────│ │─────┼───────│     │─────┼───────│
//!     │ i j │ k l   │ │ i m │ k o   │     │ c g │ k o   │
//!     │ m n │ o p   │ │ j n │ l p   │     │ d h │ l p   │
//!     └─────────────┘ └─────────────┘     └─────────────┘
//! ```
//!
//! ## Multi-Level Tiling Strategy
//!
//! Different strategies for different matrix sizes:
//! - **Small (<`SMALL_LEN` elements)**: Scalar transpose - fits in L1, no overhead
//! - **Medium (<`MEDIUM_LEN` elements)**: `TILE_SIZE`×`TILE_SIZE` Tiled - L2-friendly tiles
//! - **Large (≥`MEDIUM_LEN` elements)**: Recursive + Tiled - Cache-oblivious

#[cfg(target_arch = "aarch64")]
use core::arch::aarch64::*;
#[cfg(target_arch = "aarch64")]
use core::mem::MaybeUninit;
#[cfg(all(target_arch = "aarch64", feature = "parallel"))]
use core::sync::atomic::{AtomicUsize, Ordering};

/// Software prefetch for write (PRFM PSTL1KEEP).
///
/// Brings the cache line containing `ptr` into the L1 data cache in exclusive
/// state, preparing for a subsequent store. This avoids Read-For-Ownership
/// (RFO) stalls when writing to memory not already in L1.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn prefetch_write(ptr: *const u8) {
    // PRFM PSTL1KEEP: Prefetch for Store, L1 cache, temporal (keep in cache).
    unsafe {
        core::arch::asm!(
            "prfm pstl1keep, [{ptr}]",
            ptr = in(reg) ptr,
            options(readonly, nostack, preserves_flags),
        );
    }
}

/// Maximum number of elements for the simple scalar transpose.
///
/// For matrices with fewer than `SMALL_LEN` elements (~1KB for 4-byte elements),
/// the overhead of tiling isn't worth it.
///
/// Direct element-by-element copy is faster because:
/// - The entire matrix fits in L1 cache (32-64KB on most CPUs)
/// - No tile boundary calculations needed
/// - Branch prediction works well for small loops
#[cfg(any(target_arch = "aarch64", test))]
const SMALL_LEN: usize = 255;

/// Maximum number of elements for the single-level tiled transpose.
///
/// For matrices up to `MEDIUM_LEN` elements (~4MB for 4-byte elements), we use
/// a simple tiled approach with `TILE_SIZE`×`TILE_SIZE` tiles.
///
/// This fits comfortably within L2 cache (256KB-512KB) with good spatial locality.
///
/// Beyond this threshold, we switch to recursive subdivision to ensure
/// cache-oblivious behavior for very large matrices.
#[cfg(any(target_arch = "aarch64", test))]
const MEDIUM_LEN: usize = 1024 * 1024;

/// Side length of a tile in elements.
///
/// We use `TILE_SIZE`×`TILE_SIZE` tiles because:
/// - `TILE_SIZE`×`TILE_SIZE` × 4 bytes = 1KB per tile, fitting in L1 cache
/// - `TILE_SIZE` is divisible by `BLOCK_SIZE`, allowing exactly (`TILE_SIZE`/`BLOCK_SIZE`)^2 NEON blocks per tile
/// - Good balance between tile overhead and cache utilization
#[cfg(any(target_arch = "aarch64", test))]
const TILE_SIZE: usize = 16;

/// Maximum dimension for recursive base case.
///
/// When recursively subdividing large matrices, we stop when both
/// dimensions are ≤ `RECURSIVE_LIMIT` elements.
///
/// At this point, the sub-matrix (up to `RECURSIVE_LIMIT`×`RECURSIVE_LIMIT` elements)
/// fits in L2 cache, so we switch to tiled transpose.
#[cfg(target_arch = "aarch64")]
const RECURSIVE_LIMIT: usize = 128;

/// Minimum number of elements before enabling parallel processing.
///
/// Parallel transpose only pays off for large matrices because:
/// - Thread spawn/join overhead (~1-10μs)
/// - Cache coherency traffic between cores
/// - Memory bandwidth becomes the bottleneck, not compute
///
/// At `PARALLEL_THRESHOLD` elements, the work per thread is large enough that
/// parallelism overhead is amortized.
#[cfg(all(target_arch = "aarch64", feature = "parallel"))]
const PARALLEL_THRESHOLD: usize = 4 * 1024 * 1024;

/// Transpose a matrix from row-major `input` to row-major `output`.
///
/// Given an input matrix with `height` rows and `width` columns, produces
/// an output matrix with `width` rows and `height` columns.
///
/// # Memory Layout
///
/// Both input and output are stored in **row-major order**.
///
/// ```text
///     Input (height=2, width=3):       Output (height=3, width=2):
///
///     Row 0: [ a, b, c ]               Row 0: [ a, d ]
///     Row 1: [ d, e, f ]               Row 1: [ b, e ]
///                                      Row 2: [ c, f ]
///
///     Memory: [a, b, c, d, e, f]       Memory: [a, d, b, e, c, f]
/// ```
///
/// # Index Transformation
///
/// - Initial element at `input[row * width + col]`,
/// - Transposed position is `output[col * height + row]`.
///
/// # Arguments
///
/// * `input` - Source matrix in row-major order
/// * `output` - Destination buffer in row-major order
/// * `width` - Number of columns in the input matrix
/// * `height` - Number of rows in the input matrix
///
/// # Panics
///
/// Panics if:
/// - `input.len() != width * height`
/// - `output.len() != width * height`
#[inline]
pub fn transpose<T: Copy + Send + Sync>(
    input: &[T],
    output: &mut [T],
    width: usize,
    height: usize,
) {
    // Input validation
    assert_eq!(
        input.len(),
        width * height,
        "Input length {} doesn't match width*height = {}",
        input.len(),
        width * height
    );
    assert_eq!(
        output.len(),
        width * height,
        "Output length {} doesn't match width*height = {}",
        output.len(),
        width * height
    );

    // Handle empty matrices
    if width == 0 || height == 0 {
        return;
    }

    // Architecture dispatch
    #[cfg(target_arch = "aarch64")]
    {
        // Use NEON-optimized path for 4-byte elements.
        //
        // This covers common field types like MontyField31.
        if core::mem::size_of::<T>() == 4 {
            // SAFETY:
            // - input/output lengths verified above
            // - T is 4 bytes, matching u32 size and alignment
            // - Pointers derived from valid slices
            unsafe {
                transpose_neon_4b(
                    input.as_ptr().cast::<u32>(),
                    output.as_mut_ptr().cast::<u32>(),
                    width,
                    height,
                );
            }
            return;
        }

        // Use NEON-optimized path for 8-byte elements.
        //
        // This covers 64-bit field types like Goldilocks.
        // A 128-bit NEON register holds 2 u64 elements, so we use
        // pairs of registers per row and a 1-stage butterfly.
        if core::mem::size_of::<T>() == 8 {
            // SAFETY:
            // - input/output lengths verified above
            // - T is 8 bytes, matching u64 size and alignment
            // - Pointers derived from valid slices
            unsafe {
                transpose_neon_8b(
                    input.as_ptr().cast::<u64>(),
                    output.as_mut_ptr().cast::<u64>(),
                    width,
                    height,
                );
            }
            return;
        }
    }

    // Fallback for non-ARM64 or unsupported element sizes.
    transpose::transpose(input, output, width, height);
}

/// Top-level NEON transpose dispatcher for 4-byte elements.
///
/// Selects the appropriate strategy based on matrix size:
///
/// ```text
///     ┌───────────────────────────────────────────────────────────────────────────────────┐
///     │                              transpose_neon_4b                                    │
///     │                                     │                                             │
///     │    ┌────────────────────────────────┼───────────────────────────────┐             │
///     │    ▼                                ▼                               ▼             │
///     │  len < SMALL_LEN     SMALL_LEN ≤ len < MEDIUM_LEN          len ≥ MEDIUM_LEN       │
///     │    │                                │                               │             │
///     │    ▼                                ▼                               ▼             │
///     │  scalar              tiled TILE_SIZE×TILE_SIZE                 recursive          │
///     │                                     │                        (→ tiled at          │
///     │                                     │                          leaves)            │
///     │                                     │                               │             │
///     │                                     └───────────────┬───────────────┘             │
///     │                                                     ▼                             │
///     │                                       parallel (if ≥ PARALLEL_THRESHOLD           │
///     │                                         and feature enabled)                      │
///     └───────────────────────────────────────────────────────────────────────────────────┘
/// ```
///
/// # Safety
///
/// Caller must ensure `input` and `output` point to valid memory regions
/// of at least `width * height` elements each.
#[cfg(target_arch = "aarch64")]
#[inline]
unsafe fn transpose_neon_4b(input: *const u32, output: *mut u32, width: usize, height: usize) {
    // Total number of elements in the matrix.
    let len = width * height;

    #[cfg(feature = "parallel")]
    {
        // Parallel path (if enabled and matrix is large enough)
        if len >= PARALLEL_THRESHOLD {
            // SAFETY: Caller guarantees valid pointers.
            unsafe {
                transpose_neon_4b_parallel(input, output, width, height);
            }
            return;
        }
    }

    // Sequential path - choose strategy based on size
    if len <= SMALL_LEN {
        // Small matrix: simple scalar transpose.
        //
        // SAFETY: Caller guarantees valid pointers.
        unsafe {
            transpose_small_4b(input, output, width, height);
        }
    } else if len <= MEDIUM_LEN {
        // Medium matrix: single-level `TILE_SIZE`×`TILE_SIZE` tiling.
        //
        // SAFETY: Caller guarantees valid pointers.
        unsafe {
            transpose_tiled_4b(input, output, width, height);
        }
    } else {
        // Large matrix: recursive subdivision then tiling.
        //
        // This is the cache-oblivious approach.
        // SAFETY: Caller guarantees valid pointers.
        unsafe {
            transpose_recursive_4b(input, output, 0, height, 0, width, width, height);
        }
    }
}

/// Parallel transpose for very large matrices (≥ `PARALLEL_THRESHOLD` elements).
///
/// Divides the matrix into horizontal stripes, one per thread.
/// Each thread processes its stripe independently using the tiled algorithm.
///
/// # Stripe Division
///
/// ```text
///     ┌─────────────────────────────────┐
///     │         Thread 0                │  rows [0, rows_per_thread)
///     ├─────────────────────────────────┤
///     │         Thread 1                │  rows [rows_per_thread, 2*rows_per_thread)
///     ├─────────────────────────────────┤
///     │         Thread 2                │  ...
///     ├─────────────────────────────────┤
///     │         ...                     │
///     └─────────────────────────────────┘
/// ```
///
/// # Data Race Safety
///
/// Each thread writes to a disjoint portion of the output:
/// - Thread processing rows `[r_start, r_end)` writes to columns `[r_start, r_end)`
///   of the transposed output.
/// - No synchronization needed beyond the initial stripe assignment.
///
/// # Safety
///
/// Caller must ensure valid pointers for `width * height` elements.
#[cfg(all(target_arch = "aarch64", feature = "parallel"))]
#[inline]
unsafe fn transpose_neon_4b_parallel(
    input: *const u32,
    output: *mut u32,
    width: usize,
    height: usize,
) {
    use rayon::prelude::*;

    // Compute stripe sizes

    // Number of available threads in the rayon thread pool.
    let num_threads = rayon::current_num_threads();

    // Divide rows as evenly as possible among threads.
    //
    // Ceiling ensures the last thread doesn't get an oversized chunk.
    let rows_per_thread = height.div_ceil(num_threads);

    // Share pointers across threads

    // We use `AtomicUsize` to pass pointer addresses to threads.
    //
    // This is safe because:
    // 1. We only read the addresses (Relaxed ordering is fine)
    // 2. Each thread writes to disjoint output regions
    let inp = AtomicUsize::new(input as usize);
    let out = AtomicUsize::new(output as usize);

    // Parallel stripe processing
    (0..num_threads).into_par_iter().for_each(|thread_idx| {
        // Compute this thread's row range.
        let row_start = thread_idx * rows_per_thread;
        let row_end = (row_start + rows_per_thread).min(height);

        // Skip if this thread has no work (can happen with more threads than rows).
        if row_start < row_end {
            // Recover pointers from atomic storage.
            let input_ptr = inp.load(Ordering::Relaxed) as *const u32;
            let output_ptr = out.load(Ordering::Relaxed) as *mut u32;

            // SAFETY:
            // - Pointers are valid (from caller)
            // - Each thread writes to disjoint output columns
            unsafe {
                transpose_region_tiled_4b(
                    input_ptr, output_ptr, row_start, row_end, 0, width, width, height,
                );
            }
        }
    });
}

/// Simple element-by-element transpose for small matrices.
///
/// For matrices with <= `SMALL_LEN` elements, the overhead of tiling isn't justified.
/// Direct copying with good cache behavior is faster.
///
/// # Algorithm
///
/// For each position `(x, y)`:
/// - Read from `input[y * width + x]`
/// - Write to `output[x * height + y]`
///
/// # Loop Order
///
/// We iterate `x` in the outer loop to improve **output locality**.
///
/// This means consecutive writes go to consecutive memory addresses,
/// which is better for the write-combining buffers.
///
/// # Safety
///
/// Caller must ensure valid pointers for `width * height` elements.
#[cfg(target_arch = "aarch64")]
#[inline]
unsafe fn transpose_small_4b(input: *const u32, output: *mut u32, width: usize, height: usize) {
    // Outer loop over columns (output rows).
    for x in 0..width {
        // Inner loop over rows (output columns).
        for y in 0..height {
            // Input index: row-major position of element (y, x).
            let input_index = x + y * width;

            // Output index: row-major position of element (x, y).
            let output_index = y + x * height;

            // SAFETY: Indices are within bounds by loop construction.
            unsafe {
                *output.add(output_index) = *input.add(input_index);
            }
        }
    }
}

/// Tiled transpose using `TILE_SIZE`×`TILE_SIZE` tiles composed of `BLOCK_SIZE`×`BLOCK_SIZE` NEON blocks.
///
/// This is important for medium-sized matrices (`SMALL_LEN` to `MEDIUM_LEN` elements).
///
/// # Tiling Strategy
///
/// - The matrix is divided into `TILE_SIZE`×`TILE_SIZE` tiles.
/// - Each tile is further divided into (`TILE_SIZE`/`BLOCK_SIZE`)^2 blocks that are transposed using NEON SIMD.
///
/// ```text
///     Matrix (e.g., 64×48):
///     ┌─────────────────────────────────────────────────────────────────────────────────────┐
///     │    Tile(0,0)       │    Tile(1,0)       │    Tile(2,0)       │    Tile(3,0)  │rem_x │
///     │ TILE_SIZE×TILE_SIZE│ TILE_SIZE×TILE_SIZE│ TILE_SIZE×TILE_SIZE│ TILE_SIZE×..  │      │
///     ├────────────────────┼────────────────────┼────────────────────┼───────────────┼──────┤
///     │    Tile(0,1)       │    Tile(1,1)       │    Tile(2,1)       │    Tile(3,1)  │rem_x │
///     │ TILE_SIZE×TILE_SIZE│ TILE_SIZE×TILE_SIZE│ TILE_SIZE×TILE_SIZE│ TILE_SIZE×..  │      │
///     ├────────────────────┼────────────────────┼────────────────────┼───────────────┼──────┤
///     │    Tile(0,2)       │    Tile(1,2)       │    Tile(2,2)       │    Tile(3,2)  │rem_x │
///     │ TILE_SIZE×TILE_SIZE│ TILE_SIZE×TILE_SIZE│ TILE_SIZE×TILE_SIZE│ TILE_SIZE×..  │      │
///     └────────────────────┴────────────────────┴────────────────────┴───────────────┴──────┘
///                                                                                     rem_y
/// ```
///
/// Remainders (`rem_x`, `rem_y`) are handled with scalar transpose.
///
/// # Safety
///
/// Caller must ensure valid pointers for `width * height` elements.
#[cfg(target_arch = "aarch64")]
#[inline]
unsafe fn transpose_tiled_4b(input: *const u32, output: *mut u32, width: usize, height: usize) {
    // Compute tile counts and remainders

    // Number of complete `TILE_SIZE`×`TILE_SIZE` tiles in each dimension.
    let x_tile_count = width / TILE_SIZE;
    let y_tile_count = height / TILE_SIZE;

    // Leftover elements that don't fit in complete tiles.
    let remainder_x = width - x_tile_count * TILE_SIZE;
    let remainder_y = height - y_tile_count * TILE_SIZE;

    // Process complete `TILE_SIZE`×`TILE_SIZE` tiles

    // Iterate over tile rows.
    for y_tile in 0..y_tile_count {
        // Iterate over tile columns.
        for x_tile in 0..x_tile_count {
            // Top-left corner of this tile.
            let x_start = x_tile * TILE_SIZE;
            let y_start = y_tile * TILE_SIZE;

            // Transpose this `TILE_SIZE`×`TILE_SIZE` tile
            //
            // SAFETY: Tile coordinates are within bounds.
            unsafe {
                transpose_tile_16x16_neon(input, output, width, height, x_start, y_start);
            }
        }

        // Handle partial column tiles (right edge)

        // Elements in columns [x_tile_count * TILE_SIZE, width) don't form a complete tile.
        // Use scalar transpose for these.
        if remainder_x > 0 {
            // SAFETY: Coordinates are within bounds.
            unsafe {
                transpose_block_scalar(
                    input,
                    output,
                    width,
                    height,
                    x_tile_count * TILE_SIZE, // x_start
                    y_tile * TILE_SIZE,       // y_start
                    remainder_x,              // block_width
                    TILE_SIZE,                // block_height
                );
            }
        }
    }

    // Handle partial row tiles (bottom edge)

    // Elements in rows [y_tile_count * TILE_SIZE, height) don't form complete tiles.
    if remainder_y > 0 {
        // Process bottom edge tiles (except corner).
        for x_tile in 0..x_tile_count {
            // SAFETY: Coordinates are within bounds.
            unsafe {
                transpose_block_scalar(
                    input,
                    output,
                    width,
                    height,
                    x_tile * TILE_SIZE,       // x_start
                    y_tile_count * TILE_SIZE, // y_start
                    TILE_SIZE,                // block_width
                    remainder_y,              // block_height
                );
            }
        }

        // Handle corner block (bottom-right)

        // The corner block is the intersection of right and bottom remainders.
        if remainder_x > 0 {
            // SAFETY: Coordinates are within bounds.
            unsafe {
                transpose_block_scalar(
                    input,
                    output,
                    width,
                    height,
                    x_tile_count * TILE_SIZE, // x_start
                    y_tile_count * TILE_SIZE, // y_start
                    remainder_x,              // block_width
                    remainder_y,              // block_height
                );
            }
        }
    }
}

/// Recursive cache-oblivious transpose for large matrices.
///
/// This algorithm recursively subdivides the matrix until sub-blocks fit
/// in cache, then uses tiled transpose on the leaves.
///
/// # Cache-Oblivious Design
///
/// The key insight is that we don't need to know cache sizes explicitly.
///
/// By recursively halving the problem, we eventually reach a size that
/// fits in any level of cache (L1, L2, or L3).
///
/// # Recursion Pattern
///
/// At each level, we split along the **longer dimension**:
///
/// ```text
///     Wide matrix (cols > rows):      Tall matrix (rows ≥ cols):
///     Split vertically                Split horizontally
///
///     ┌─────────┬─────────┐           ┌───────────────────┐
///     │         │         │           │                   │
///     │  Left   │  Right  │           │       Top         │
///     │         │         │           │                   │
///     │         │         │           ├───────────────────┤
///     │         │         │           │                   │
///     └─────────┴─────────┘           │      Bottom       │
///                                     │                   │
///                                     └───────────────────┘
/// ```
///
/// # Base Case
///
/// We stop recursing when both dimensions are ≤ `RECURSIVE_LIMIT` elements (or ≤ 2,
/// which is a degenerate case). At this point, the sub-matrix fits in
/// L2 cache (~64KB for `RECURSIVE_LIMIT`×`RECURSIVE_LIMIT`×4 bytes), so we use tiled transpose.
///
/// # Parameters
///
/// The function uses coordinate ranges rather than creating sub-arrays:
/// - `row_start..row_end`: Row range in the original matrix
/// - `col_start..col_end`: Column range in the original matrix
/// - `total_cols`, `total_rows`: Original matrix dimensions (for stride calculations)
///
/// # Safety
///
/// Caller must ensure valid pointers and that coordinate ranges are within bounds.
#[cfg(target_arch = "aarch64")]
#[allow(clippy::too_many_arguments)]
unsafe fn transpose_recursive_4b(
    input: *const u32,
    output: *mut u32,
    row_start: usize,
    row_end: usize,
    col_start: usize,
    col_end: usize,
    total_cols: usize,
    total_rows: usize,
) {
    // Compute sub-matrix dimensions
    let nbr_rows = row_end - row_start;
    let nbr_cols = col_end - col_start;

    // Base case: small enough to use tiled transpose

    // Stop recursing when:
    // 1. Both dimensions ≤ RECURSIVE_LIMIT (fits in cache), OR
    // 2. Either dimension ≤ 2 (degenerate case, no benefit from recursion)
    if (nbr_rows <= RECURSIVE_LIMIT && nbr_cols <= RECURSIVE_LIMIT)
        || nbr_rows <= 2
        || nbr_cols <= 2
    {
        // SAFETY: Caller ensures valid pointers and bounds.
        unsafe {
            transpose_region_tiled_4b(
                input, output, row_start, row_end, col_start, col_end, total_cols, total_rows,
            );
        }
        return;
    }

    // Recursive case: split along the longer dimension
    if nbr_rows >= nbr_cols {
        // Split horizontally (by rows)

        // Midpoint of the row range.
        let mid = row_start + (nbr_rows / 2);

        // Recurse on top half.
        // SAFETY: mid is within [row_start, row_end].
        unsafe {
            transpose_recursive_4b(
                input, output, row_start, mid, col_start, col_end, total_cols, total_rows,
            );
        }

        // Recurse on bottom half.
        // SAFETY: mid is within [row_start, row_end].
        unsafe {
            transpose_recursive_4b(
                input, output, mid, row_end, col_start, col_end, total_cols, total_rows,
            );
        }
    } else {
        // Split vertically (by columns)

        // Midpoint of the column range.
        let mid = col_start + (nbr_cols / 2);

        // Recurse on left half.
        // SAFETY: mid is within [col_start, col_end].
        unsafe {
            transpose_recursive_4b(
                input, output, row_start, row_end, col_start, mid, total_cols, total_rows,
            );
        }

        // Recurse on right half.
        // SAFETY: mid is within [col_start, col_end].
        unsafe {
            transpose_recursive_4b(
                input, output, row_start, row_end, mid, col_end, total_cols, total_rows,
            );
        }
    }
}

/// Tiled transpose for a rectangular region within a larger matrix.
///
/// It operates on a sub-region defined by coordinate ranges.
///
/// Used as the base case of recursive transpose and for parallel stripe processing.
///
/// # Coordinate System
///
/// ```text
///     Original matrix (total_cols × total_rows):
///     ┌─────────────────────────────────────────────────────────┐
///     │                                                         │
///     │    (col_start, row_start)                               │
///     │           ┌─────────────────────┐                       │
///     │           │                     │                       │
///     │           │       Region to     │                       │
///     │           │       transpose     │                       │
///     │           │                     │                       │
///     │           └─────────────────────┘                       │
///     │                    (col_end, row_end)                   │
///     │                                                         │
///     └─────────────────────────────────────────────────────────┘
/// ```
///
/// # Safety
///
/// Caller must ensure:
/// - Valid pointers for `total_cols * total_rows` elements
/// - `row_start < row_end <= total_rows`
/// - `col_start < col_end <= total_cols`
#[cfg(target_arch = "aarch64")]
#[inline]
#[allow(clippy::too_many_arguments)]
unsafe fn transpose_region_tiled_4b(
    input: *const u32,
    output: *mut u32,
    row_start: usize,
    row_end: usize,
    col_start: usize,
    col_end: usize,
    total_cols: usize,
    total_rows: usize,
) {
    // Compute region dimensions and tile counts

    // Dimensions of the region to transpose.
    let nbr_cols = col_end - col_start;
    let nbr_rows = row_end - row_start;

    // Number of complete `TILE_SIZE`×`TILE_SIZE` tiles in each dimension.
    let x_tile_count = nbr_cols / TILE_SIZE;
    let y_tile_count = nbr_rows / TILE_SIZE;

    // Leftover elements that don't fit in complete tiles.
    let remainder_x = nbr_cols - x_tile_count * TILE_SIZE;
    let remainder_y = nbr_rows - y_tile_count * TILE_SIZE;

    // Process complete `TILE_SIZE`×`TILE_SIZE` tiles
    for y_tile in 0..y_tile_count {
        for x_tile in 0..x_tile_count {
            // Coordinates of this tile's top-left corner in the original matrix.
            let col = col_start + x_tile * TILE_SIZE;
            let row = row_start + y_tile * TILE_SIZE;

            // SAFETY: Tile coordinates are within the region bounds.
            // Uses the buffered tile function: for large matrices the output
            // is likely in L3/RAM, so L1 buffering + write prefetching avoids
            // RFO stalls on scattered output writes.
            unsafe {
                transpose_tile_16x16_neon_buffered(input, output, total_cols, total_rows, col, row);
            }
        }

        // Handle partial column tiles (right edge of region)
        if remainder_x > 0 {
            // SAFETY: Coordinates are within region bounds.
            unsafe {
                transpose_block_scalar(
                    input,
                    output,
                    total_cols,
                    total_rows,
                    col_start + x_tile_count * TILE_SIZE, // x_start
                    row_start + y_tile * TILE_SIZE,       // y_start
                    remainder_x,                          // block_width
                    TILE_SIZE,                            // block_height
                );
            }
        }
    }

    // Handle partial row tiles (bottom edge of region)
    if remainder_y > 0 {
        for x_tile in 0..x_tile_count {
            // SAFETY: Coordinates are within region bounds.
            unsafe {
                transpose_block_scalar(
                    input,
                    output,
                    total_cols,
                    total_rows,
                    col_start + x_tile * TILE_SIZE,       // x_start
                    row_start + y_tile_count * TILE_SIZE, // y_start
                    TILE_SIZE,                            // block_width
                    remainder_y,                          // block_height
                );
            }
        }

        // Handle corner block (bottom-right of region)
        if remainder_x > 0 {
            // SAFETY: Coordinates are within region bounds.
            unsafe {
                transpose_block_scalar(
                    input,
                    output,
                    total_cols,
                    total_rows,
                    col_start + x_tile_count * TILE_SIZE, // x_start
                    row_start + y_tile_count * TILE_SIZE, // y_start
                    remainder_x,                          // block_width
                    remainder_y,                          // block_height
                );
            }
        }
    }
}

/// Transpose a complete 16×16 tile using NEON SIMD (direct-to-output).
///
/// A 16×16 tile is processed as a 4×4 grid of 4×4 NEON blocks.
/// This function is fully unrolled for maximum performance.
///
/// Used by the **medium tiled path** where the output likely fits in L2 cache
/// and the overhead of L1 buffering isn't justified.
///
/// # Safety
///
/// Caller must ensure:
/// - Valid pointers for the full matrix
/// - `x_start + 16 <= width`
/// - `y_start + 16 <= height`
#[cfg(target_arch = "aarch64")]
#[inline]
unsafe fn transpose_tile_16x16_neon(
    input: *const u32,
    output: *mut u32,
    width: usize,
    height: usize,
    x_start: usize,
    y_start: usize,
) {
    unsafe {
        // Block Row 0 (input rows y_start..y_start+4)
        let inp = input.add(y_start * width + x_start);
        let out = output.add(x_start * height + y_start);
        transpose_4x4_neon(inp, out, width, height);
        transpose_4x4_neon(inp.add(4), out.add(4 * height), width, height);
        transpose_4x4_neon(inp.add(8), out.add(8 * height), width, height);
        transpose_4x4_neon(inp.add(12), out.add(12 * height), width, height);

        // Block Row 1 (input rows y_start+4..y_start+8)
        let inp = input.add((y_start + 4) * width + x_start);
        let out = output.add(x_start * height + y_start + 4);
        transpose_4x4_neon(inp, out, width, height);
        transpose_4x4_neon(inp.add(4), out.add(4 * height), width, height);
        transpose_4x4_neon(inp.add(8), out.add(8 * height), width, height);
        transpose_4x4_neon(inp.add(12), out.add(12 * height), width, height);

        // Block Row 2 (input rows y_start+8..y_start+12)
        let inp = input.add((y_start + 8) * width + x_start);
        let out = output.add(x_start * height + y_start + 8);
        transpose_4x4_neon(inp, out, width, height);
        transpose_4x4_neon(inp.add(4), out.add(4 * height), width, height);
        transpose_4x4_neon(inp.add(8), out.add(8 * height), width, height);
        transpose_4x4_neon(inp.add(12), out.add(12 * height), width, height);

        // Block Row 3 (input rows y_start+12..y_start+16)
        let inp = input.add((y_start + 12) * width + x_start);
        let out = output.add(x_start * height + y_start + 12);
        transpose_4x4_neon(inp, out, width, height);
        transpose_4x4_neon(inp.add(4), out.add(4 * height), width, height);
        transpose_4x4_neon(inp.add(8), out.add(8 * height), width, height);
        transpose_4x4_neon(inp.add(12), out.add(12 * height), width, height);
    }
}

/// Transpose a complete 16×16 tile using NEON SIMD with L1 buffering.
///
/// Same grid of 4×4 NEON blocks, but transposed into a stack-allocated buffer
/// first, then flushed to the output with write prefetching (`PRFM PSTL1KEEP`).
///
/// Used by the **recursive/parallel path** for large matrices (≥ `MEDIUM_LEN`)
/// where the output is in L3/RAM and direct scattered writes would stall on
/// Read-For-Ownership (RFO) cache line fetches.
///
/// # Safety
///
/// Caller must ensure:
/// - Valid pointers for the full matrix
/// - `x_start + 16 <= width`
/// - `y_start + 16 <= height`
#[cfg(target_arch = "aarch64")]
#[inline]
unsafe fn transpose_tile_16x16_neon_buffered(
    input: *const u32,
    output: *mut u32,
    width: usize,
    height: usize,
    x_start: usize,
    y_start: usize,
) {
    // Stack buffer for L1-hot transpose (1 KB for u32).
    // MaybeUninit avoids unnecessary zero-initialization; every element
    // is written by the NEON blocks before the copy reads it.
    let mut buffer = MaybeUninit::<[u32; TILE_SIZE * TILE_SIZE]>::uninit();
    let buf = buffer.as_mut_ptr().cast::<u32>();

    unsafe {
        // Transpose 4×4 grid of NEON blocks into the buffer.
        // Buffer layout: buf[col * TILE_SIZE + row] = transposed element.
        // Buffer write stride is TILE_SIZE (contiguous in L1) vs. `height` (scattered).

        // Block Row 0 (input rows y_start..y_start+4)
        let inp = input.add(y_start * width + x_start);
        transpose_4x4_neon(inp, buf, width, TILE_SIZE);
        transpose_4x4_neon(inp.add(4), buf.add(4 * TILE_SIZE), width, TILE_SIZE);
        transpose_4x4_neon(inp.add(8), buf.add(8 * TILE_SIZE), width, TILE_SIZE);
        transpose_4x4_neon(inp.add(12), buf.add(12 * TILE_SIZE), width, TILE_SIZE);

        // Block Row 1 (input rows y_start+4..y_start+8)
        let inp = input.add((y_start + 4) * width + x_start);
        transpose_4x4_neon(inp, buf.add(4), width, TILE_SIZE);
        transpose_4x4_neon(inp.add(4), buf.add(4 * TILE_SIZE + 4), width, TILE_SIZE);
        transpose_4x4_neon(inp.add(8), buf.add(8 * TILE_SIZE + 4), width, TILE_SIZE);
        transpose_4x4_neon(inp.add(12), buf.add(12 * TILE_SIZE + 4), width, TILE_SIZE);

        // Block Row 2 (input rows y_start+8..y_start+12)
        let inp = input.add((y_start + 8) * width + x_start);
        transpose_4x4_neon(inp, buf.add(8), width, TILE_SIZE);
        transpose_4x4_neon(inp.add(4), buf.add(4 * TILE_SIZE + 8), width, TILE_SIZE);
        transpose_4x4_neon(inp.add(8), buf.add(8 * TILE_SIZE + 8), width, TILE_SIZE);
        transpose_4x4_neon(inp.add(12), buf.add(12 * TILE_SIZE + 8), width, TILE_SIZE);

        // Block Row 3 (input rows y_start+12..y_start+16)
        let inp = input.add((y_start + 12) * width + x_start);
        transpose_4x4_neon(inp, buf.add(12), width, TILE_SIZE);
        transpose_4x4_neon(inp.add(4), buf.add(4 * TILE_SIZE + 12), width, TILE_SIZE);
        transpose_4x4_neon(inp.add(8), buf.add(8 * TILE_SIZE + 12), width, TILE_SIZE);
        transpose_4x4_neon(inp.add(12), buf.add(12 * TILE_SIZE + 12), width, TILE_SIZE);

        // Flush buffer to output with write prefetching.
        // Each iteration copies TILE_SIZE u32s (64 bytes = 1 cache line) from the
        // L1-hot buffer to one output row. Prefetch brings the next output cache
        // line into exclusive state, avoiding RFO stalls.
        prefetch_write(output.add(x_start * height + y_start) as *const u8);
        for c in 0..TILE_SIZE {
            if c + 1 < TILE_SIZE {
                prefetch_write(output.add((x_start + c + 1) * height + y_start) as *const u8);
            }
            core::ptr::copy_nonoverlapping(
                buf.add(c * TILE_SIZE),
                output.add((x_start + c) * height + y_start),
                TILE_SIZE,
            );
        }
    }
}

/// Scalar transpose for an arbitrary rectangular block.
///
/// Used for handling edge cases where dimensions don't align to tile boundaries.
/// Falls back to simple element-by-element copying.
///
/// # When Used
///
/// - Right edge: `block_width < TILE_SIZE`
/// - Bottom edge: `block_height < TILE_SIZE`
/// - Bottom-right corner: both dimensions < `TILE_SIZE`
///
/// # Safety
///
/// Caller must ensure:
/// - Valid pointers for the full matrix
/// - `x_start + block_width <= width`
/// - `y_start + block_height <= height`
#[cfg(target_arch = "aarch64")]
#[inline]
#[allow(clippy::too_many_arguments)]
unsafe fn transpose_block_scalar(
    input: *const u32,
    output: *mut u32,
    width: usize,
    height: usize,
    x_start: usize,
    y_start: usize,
    block_width: usize,
    block_height: usize,
) {
    // Iterate over block columns (becomes output rows).
    for inner_x in 0..block_width {
        // Iterate over block rows (becomes output columns).
        for inner_y in 0..block_height {
            // Absolute coordinates in the original matrix.
            let x = x_start + inner_x;
            let y = y_start + inner_y;

            // Input index: row-major position of (y, x).
            let input_index = x + y * width;

            // Output index: row-major position of (x, y) in transposed matrix.
            let output_index = y + x * height;

            // SAFETY: Indices are within bounds by construction.
            unsafe {
                *output.add(output_index) = *input.add(input_index);
            }
        }
    }
}

/// Transpose a 4×4 block of 32-bit elements using NEON SIMD.
///
/// This is the fundamental building block of the entire transpose algorithm.
///
/// It transposes a 4×4 block entirely within NEON registers
/// using a two-stage butterfly network.
///
/// # Memory Layout
///
/// Input (4 rows, stride = `src_stride`):
/// ```text
///     src + 0*stride:  [ a00, a01, a02, a03 ]  → q0
///     src + 1*stride:  [ a10, a11, a12, a13 ]  → q1
///     src + 2*stride:  [ a20, a21, a22, a23 ]  → q2
///     src + 3*stride:  [ a30, a31, a32, a33 ]  → q3
/// ```
///
/// Output (4 rows, stride = `dst_stride`):
/// ```text
///     dst + 0*stride:  [ a00, a10, a20, a30 ]  ← r0
///     dst + 1*stride:  [ a01, a11, a21, a31 ]  ← r1
///     dst + 2*stride:  [ a02, a12, a22, a32 ]  ← r2
///     dst + 3*stride:  [ a03, a13, a23, a33 ]  ← r3
/// ```
///
/// # Butterfly Network Algorithm
///
/// The transpose is performed in two stages using `TRN1`/`TRN2` instructions:
///
/// ## Stage 1: 32-bit Transpose
///
/// - `TRN1` takes **even-indexed** elements,
/// - `TRN2` takes **odd-indexed** elements.
///
/// ```text
///     TRN1(q0, q1) = [ a00, a10, a02, a12 ]  (even indices: 0, 2)
///     TRN2(q0, q1) = [ a01, a11, a03, a13 ]  (odd indices: 1, 3)
///     TRN1(q2, q3) = [ a20, a30, a22, a32 ]
///     TRN2(q2, q3) = [ a21, a31, a23, a33 ]
/// ```
///
/// ## Stage 2: 64-bit Transpose
///
/// Reinterpret as 64-bit elements and transpose again:
///
/// ```text
///     TRN1_64(t0, t2) = [ a00, a10 | a20, a30 ]  → r0
///     TRN2_64(t0, t2) = [ a02, a12 | a22, a32 ]  → r2
///     TRN1_64(t1, t3) = [ a01, a11 | a21, a31 ]  → r1
///     TRN2_64(t1, t3) = [ a03, a13 | a23, a33 ]  → r3
/// ```
///
/// # Explanations
///
/// The butterfly network swaps elements at progressively larger distances:
/// - Stage 1: Swaps elements 1 apart (within 64-bit pairs)
/// - Stage 2: Swaps elements 2 apart (between 64-bit halves)
///
/// This is analogous to the bit-reversal pattern in FFT algorithms.
///
/// # Performance
///
/// - **4 loads** (vld1q_u32): 4 cycles
/// - **8 permutes** (vtrn): ~8 cycles (pipelined)
/// - **4 stores** (vst1q_u32): 4 cycles
/// - **Total**: ~16 cycles for 16 elements = **1 cycle/element**
///
/// # Safety
///
/// Caller must ensure:
/// - `src` is valid for reading 4 rows of `src_stride` elements each
/// - `dst` is valid for writing 4 rows of `dst_stride` elements each
/// - The first 4 elements of each row are accessible
#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn transpose_4x4_neon(src: *const u32, dst: *mut u32, src_stride: usize, dst_stride: usize) {
    unsafe {
        // Phase 1: Load 4 rows into NEON registers
        //
        // Each vld1q_u32 loads 4 consecutive u32s (16 bytes = 128 bits).
        // Total: 64 bytes = one cache line on most ARM64 CPUs.

        // Row 0: [a00, a01, a02, a03]
        let q0 = vld1q_u32(src);
        // Row 1: [a10, a11, a12, a13]
        let q1 = vld1q_u32(src.add(src_stride));
        // Row 2: [a20, a21, a22, a23]
        let q2 = vld1q_u32(src.add(2 * src_stride));
        // Row 3: [a30, a31, a32, a33]
        let q3 = vld1q_u32(src.add(3 * src_stride));

        // Phase 2: Stage 1 - Transpose 2×2 blocks of 32-bit elements
        //
        // vtrn1q_u32(a, b): Takes elements at even indices from a and b
        // - Result: [a[0], b[0], a[2], b[2]]
        //
        // vtrn2q_u32(a, b): Takes elements at odd indices from a and b
        // - Result: [a[1], b[1], a[3], b[3]]

        let t0_0 = vtrn1q_u32(q0, q1); // [a00, a10, a02, a12]
        let t0_1 = vtrn2q_u32(q0, q1); // [a01, a11, a03, a13]
        let t0_2 = vtrn1q_u32(q2, q3); // [a20, a30, a22, a32]
        let t0_3 = vtrn2q_u32(q2, q3); // [a21, a31, a23, a33]

        // Phase 3: Stage 2 - Transpose 2×2 blocks of 64-bit elements
        //
        // Reinterpret u32x4 as u64x2, then transpose.
        // This swaps the 64-bit halves of the vectors.
        //
        // vtrn1q_u64(a, b): [a.lo, b.lo]
        // vtrn2q_u64(a, b): [a.hi, b.hi]

        // r0 = [a00, a10, a20, a30] (column 0 of input → row 0 of output)
        let r0 = vreinterpretq_u32_u64(vtrn1q_u64(
            vreinterpretq_u64_u32(t0_0),
            vreinterpretq_u64_u32(t0_2),
        ));

        // r2 = [a02, a12, a22, a32] (column 2 of input → row 2 of output)
        let r2 = vreinterpretq_u32_u64(vtrn2q_u64(
            vreinterpretq_u64_u32(t0_0),
            vreinterpretq_u64_u32(t0_2),
        ));

        // r1 = [a01, a11, a21, a31] (column 1 of input → row 1 of output)
        let r1 = vreinterpretq_u32_u64(vtrn1q_u64(
            vreinterpretq_u64_u32(t0_1),
            vreinterpretq_u64_u32(t0_3),
        ));

        // r3 = [a03, a13, a23, a33] (column 3 of input → row 3 of output)
        let r3 = vreinterpretq_u32_u64(vtrn2q_u64(
            vreinterpretq_u64_u32(t0_1),
            vreinterpretq_u64_u32(t0_3),
        ));

        // Phase 4: Store 4 transposed rows
        //
        // Store row 0 of output
        vst1q_u32(dst, r0);
        // Store row 1 of output
        vst1q_u32(dst.add(dst_stride), r1);
        // Store row 2 of output
        vst1q_u32(dst.add(2 * dst_stride), r2);
        // Store row 3 of output
        vst1q_u32(dst.add(3 * dst_stride), r3);
    }
}

// ============================================================================
// 8-byte (u64) NEON transpose functions
//
// These are analogous to the 4-byte functions above, but operate on u64
// elements. Since a 128-bit NEON register holds 2 u64 elements, each row
// of a 4×4 block requires 2 registers (8 total for a block). The transpose
// uses a single-stage butterfly with vtrn1q_u64/vtrn2q_u64 on four 2×2
// sub-blocks.
// ============================================================================

/// Top-level NEON transpose dispatcher for 8-byte elements.
///
/// Selects the appropriate strategy based on matrix size, mirroring
/// `transpose_neon_4b` but for u64 elements.
///
/// # Safety
///
/// Caller must ensure `input` and `output` point to valid memory regions
/// of at least `width * height` elements each.
#[cfg(target_arch = "aarch64")]
#[inline]
unsafe fn transpose_neon_8b(input: *const u64, output: *mut u64, width: usize, height: usize) {
    let len = width * height;

    #[cfg(feature = "parallel")]
    {
        if len >= PARALLEL_THRESHOLD {
            unsafe {
                transpose_neon_8b_parallel(input, output, width, height);
            }
            return;
        }
    }

    if len <= SMALL_LEN {
        unsafe {
            transpose_small_8b(input, output, width, height);
        }
    } else if len <= MEDIUM_LEN {
        unsafe {
            transpose_tiled_8b(input, output, width, height);
        }
    } else {
        unsafe {
            transpose_recursive_8b(input, output, 0, height, 0, width, width, height);
        }
    }
}

/// Parallel transpose for very large matrices of 8-byte elements.
///
/// Divides the matrix into horizontal stripes, one per thread.
///
/// # Safety
///
/// Caller must ensure valid pointers for `width * height` elements.
#[cfg(all(target_arch = "aarch64", feature = "parallel"))]
#[inline]
unsafe fn transpose_neon_8b_parallel(
    input: *const u64,
    output: *mut u64,
    width: usize,
    height: usize,
) {
    use rayon::prelude::*;

    let num_threads = rayon::current_num_threads();
    let rows_per_thread = height.div_ceil(num_threads);

    let inp = AtomicUsize::new(input as usize);
    let out = AtomicUsize::new(output as usize);

    (0..num_threads).into_par_iter().for_each(|thread_idx| {
        let row_start = thread_idx * rows_per_thread;
        let row_end = (row_start + rows_per_thread).min(height);

        if row_start < row_end {
            let input_ptr = inp.load(Ordering::Relaxed) as *const u64;
            let output_ptr = out.load(Ordering::Relaxed) as *mut u64;

            unsafe {
                transpose_region_tiled_8b(
                    input_ptr, output_ptr, row_start, row_end, 0, width, width, height,
                );
            }
        }
    });
}

/// Simple element-by-element transpose for small matrices of 8-byte elements.
///
/// # Safety
///
/// Caller must ensure valid pointers for `width * height` elements.
#[cfg(target_arch = "aarch64")]
#[inline]
unsafe fn transpose_small_8b(input: *const u64, output: *mut u64, width: usize, height: usize) {
    for x in 0..width {
        for y in 0..height {
            let input_index = x + y * width;
            let output_index = y + x * height;

            unsafe {
                *output.add(output_index) = *input.add(input_index);
            }
        }
    }
}

/// Tiled transpose using 16×16 tiles for 8-byte elements.
///
/// # Safety
///
/// Caller must ensure valid pointers for `width * height` elements.
#[cfg(target_arch = "aarch64")]
#[inline]
unsafe fn transpose_tiled_8b(input: *const u64, output: *mut u64, width: usize, height: usize) {
    let x_tile_count = width / TILE_SIZE;
    let y_tile_count = height / TILE_SIZE;

    let remainder_x = width - x_tile_count * TILE_SIZE;
    let remainder_y = height - y_tile_count * TILE_SIZE;

    // Process complete tiles
    for y_tile in 0..y_tile_count {
        for x_tile in 0..x_tile_count {
            let x_start = x_tile * TILE_SIZE;
            let y_start = y_tile * TILE_SIZE;

            unsafe {
                transpose_tile_16x16_neon_8b(input, output, width, height, x_start, y_start);
            }
        }

        // Right edge remainder
        if remainder_x > 0 {
            unsafe {
                transpose_block_scalar_8b(
                    input,
                    output,
                    width,
                    height,
                    x_tile_count * TILE_SIZE,
                    y_tile * TILE_SIZE,
                    remainder_x,
                    TILE_SIZE,
                );
            }
        }
    }

    // Bottom edge remainder
    if remainder_y > 0 {
        for x_tile in 0..x_tile_count {
            unsafe {
                transpose_block_scalar_8b(
                    input,
                    output,
                    width,
                    height,
                    x_tile * TILE_SIZE,
                    y_tile_count * TILE_SIZE,
                    TILE_SIZE,
                    remainder_y,
                );
            }
        }

        // Bottom-right corner
        if remainder_x > 0 {
            unsafe {
                transpose_block_scalar_8b(
                    input,
                    output,
                    width,
                    height,
                    x_tile_count * TILE_SIZE,
                    y_tile_count * TILE_SIZE,
                    remainder_x,
                    remainder_y,
                );
            }
        }
    }
}

/// Recursive cache-oblivious transpose for large matrices of 8-byte elements.
///
/// # Safety
///
/// Caller must ensure valid pointers and that coordinate ranges are within bounds.
#[cfg(target_arch = "aarch64")]
#[allow(clippy::too_many_arguments)]
unsafe fn transpose_recursive_8b(
    input: *const u64,
    output: *mut u64,
    row_start: usize,
    row_end: usize,
    col_start: usize,
    col_end: usize,
    total_cols: usize,
    total_rows: usize,
) {
    let nbr_rows = row_end - row_start;
    let nbr_cols = col_end - col_start;

    if (nbr_rows <= RECURSIVE_LIMIT && nbr_cols <= RECURSIVE_LIMIT)
        || nbr_rows <= 2
        || nbr_cols <= 2
    {
        unsafe {
            transpose_region_tiled_8b(
                input, output, row_start, row_end, col_start, col_end, total_cols, total_rows,
            );
        }
        return;
    }

    if nbr_rows >= nbr_cols {
        let mid = row_start + (nbr_rows / 2);

        unsafe {
            transpose_recursive_8b(
                input, output, row_start, mid, col_start, col_end, total_cols, total_rows,
            );
        }

        unsafe {
            transpose_recursive_8b(
                input, output, mid, row_end, col_start, col_end, total_cols, total_rows,
            );
        }
    } else {
        let mid = col_start + (nbr_cols / 2);

        unsafe {
            transpose_recursive_8b(
                input, output, row_start, row_end, col_start, mid, total_cols, total_rows,
            );
        }

        unsafe {
            transpose_recursive_8b(
                input, output, row_start, row_end, mid, col_end, total_cols, total_rows,
            );
        }
    }
}

/// Tiled transpose for a rectangular region of 8-byte elements.
///
/// Used as the base case of recursive transpose and for parallel stripe processing.
///
/// # Safety
///
/// Caller must ensure:
/// - Valid pointers for `total_cols * total_rows` elements
/// - `row_start < row_end <= total_rows`
/// - `col_start < col_end <= total_cols`
#[cfg(target_arch = "aarch64")]
#[inline]
#[allow(clippy::too_many_arguments)]
unsafe fn transpose_region_tiled_8b(
    input: *const u64,
    output: *mut u64,
    row_start: usize,
    row_end: usize,
    col_start: usize,
    col_end: usize,
    total_cols: usize,
    total_rows: usize,
) {
    let nbr_cols = col_end - col_start;
    let nbr_rows = row_end - row_start;

    let x_tile_count = nbr_cols / TILE_SIZE;
    let y_tile_count = nbr_rows / TILE_SIZE;

    let remainder_x = nbr_cols - x_tile_count * TILE_SIZE;
    let remainder_y = nbr_rows - y_tile_count * TILE_SIZE;

    // Process complete tiles
    for y_tile in 0..y_tile_count {
        for x_tile in 0..x_tile_count {
            let col = col_start + x_tile * TILE_SIZE;
            let row = row_start + y_tile * TILE_SIZE;

            // Uses the buffered tile function: for large matrices the output
            // is likely in L3/RAM, so L1 buffering + write prefetching avoids
            // RFO stalls on scattered output writes.
            unsafe {
                transpose_tile_16x16_neon_8b_buffered(
                    input, output, total_cols, total_rows, col, row,
                );
            }
        }

        // Right edge remainder
        if remainder_x > 0 {
            unsafe {
                transpose_block_scalar_8b(
                    input,
                    output,
                    total_cols,
                    total_rows,
                    col_start + x_tile_count * TILE_SIZE,
                    row_start + y_tile * TILE_SIZE,
                    remainder_x,
                    TILE_SIZE,
                );
            }
        }
    }

    // Bottom edge remainder
    if remainder_y > 0 {
        for x_tile in 0..x_tile_count {
            unsafe {
                transpose_block_scalar_8b(
                    input,
                    output,
                    total_cols,
                    total_rows,
                    col_start + x_tile * TILE_SIZE,
                    row_start + y_tile_count * TILE_SIZE,
                    TILE_SIZE,
                    remainder_y,
                );
            }
        }

        // Bottom-right corner
        if remainder_x > 0 {
            unsafe {
                transpose_block_scalar_8b(
                    input,
                    output,
                    total_cols,
                    total_rows,
                    col_start + x_tile_count * TILE_SIZE,
                    row_start + y_tile_count * TILE_SIZE,
                    remainder_x,
                    remainder_y,
                );
            }
        }
    }
}

/// Transpose a complete 16×16 tile of 8-byte elements using NEON SIMD (direct-to-output).
///
/// Used by the **medium tiled path** where the output likely fits in L2 cache.
///
/// # Safety
///
/// Caller must ensure:
/// - Valid pointers for the full matrix
/// - `x_start + 16 <= width`
/// - `y_start + 16 <= height`
#[cfg(target_arch = "aarch64")]
#[inline]
unsafe fn transpose_tile_16x16_neon_8b(
    input: *const u64,
    output: *mut u64,
    width: usize,
    height: usize,
    x_start: usize,
    y_start: usize,
) {
    unsafe {
        // Block Row 0 (input rows y_start..y_start+4)
        let inp = input.add(y_start * width + x_start);
        let out = output.add(x_start * height + y_start);
        transpose_4x4_neon_8b(inp, out, width, height);
        transpose_4x4_neon_8b(inp.add(4), out.add(4 * height), width, height);
        transpose_4x4_neon_8b(inp.add(8), out.add(8 * height), width, height);
        transpose_4x4_neon_8b(inp.add(12), out.add(12 * height), width, height);

        // Block Row 1 (input rows y_start+4..y_start+8)
        let inp = input.add((y_start + 4) * width + x_start);
        let out = output.add(x_start * height + y_start + 4);
        transpose_4x4_neon_8b(inp, out, width, height);
        transpose_4x4_neon_8b(inp.add(4), out.add(4 * height), width, height);
        transpose_4x4_neon_8b(inp.add(8), out.add(8 * height), width, height);
        transpose_4x4_neon_8b(inp.add(12), out.add(12 * height), width, height);

        // Block Row 2 (input rows y_start+8..y_start+12)
        let inp = input.add((y_start + 8) * width + x_start);
        let out = output.add(x_start * height + y_start + 8);
        transpose_4x4_neon_8b(inp, out, width, height);
        transpose_4x4_neon_8b(inp.add(4), out.add(4 * height), width, height);
        transpose_4x4_neon_8b(inp.add(8), out.add(8 * height), width, height);
        transpose_4x4_neon_8b(inp.add(12), out.add(12 * height), width, height);

        // Block Row 3 (input rows y_start+12..y_start+16)
        let inp = input.add((y_start + 12) * width + x_start);
        let out = output.add(x_start * height + y_start + 12);
        transpose_4x4_neon_8b(inp, out, width, height);
        transpose_4x4_neon_8b(inp.add(4), out.add(4 * height), width, height);
        transpose_4x4_neon_8b(inp.add(8), out.add(8 * height), width, height);
        transpose_4x4_neon_8b(inp.add(12), out.add(12 * height), width, height);
    }
}

/// Transpose a complete 16×16 tile of 8-byte elements with L1 buffering.
///
/// Used by the **recursive/parallel path** for large matrices where the output
/// is in L3/RAM. L1 buffering + write prefetching avoids RFO stalls.
///
/// # Safety
///
/// Caller must ensure:
/// - Valid pointers for the full matrix
/// - `x_start + 16 <= width`
/// - `y_start + 16 <= height`
#[cfg(target_arch = "aarch64")]
#[inline]
unsafe fn transpose_tile_16x16_neon_8b_buffered(
    input: *const u64,
    output: *mut u64,
    width: usize,
    height: usize,
    x_start: usize,
    y_start: usize,
) {
    // Stack buffer for L1-hot transpose (2 KB for u64).
    let mut buffer = MaybeUninit::<[u64; TILE_SIZE * TILE_SIZE]>::uninit();
    let buf = buffer.as_mut_ptr().cast::<u64>();

    unsafe {
        // Transpose 4×4 grid of NEON blocks into the buffer.

        // Block Row 0 (input rows y_start..y_start+4)
        let inp = input.add(y_start * width + x_start);
        transpose_4x4_neon_8b(inp, buf, width, TILE_SIZE);
        transpose_4x4_neon_8b(inp.add(4), buf.add(4 * TILE_SIZE), width, TILE_SIZE);
        transpose_4x4_neon_8b(inp.add(8), buf.add(8 * TILE_SIZE), width, TILE_SIZE);
        transpose_4x4_neon_8b(inp.add(12), buf.add(12 * TILE_SIZE), width, TILE_SIZE);

        // Block Row 1 (input rows y_start+4..y_start+8)
        let inp = input.add((y_start + 4) * width + x_start);
        transpose_4x4_neon_8b(inp, buf.add(4), width, TILE_SIZE);
        transpose_4x4_neon_8b(inp.add(4), buf.add(4 * TILE_SIZE + 4), width, TILE_SIZE);
        transpose_4x4_neon_8b(inp.add(8), buf.add(8 * TILE_SIZE + 4), width, TILE_SIZE);
        transpose_4x4_neon_8b(inp.add(12), buf.add(12 * TILE_SIZE + 4), width, TILE_SIZE);

        // Block Row 2 (input rows y_start+8..y_start+12)
        let inp = input.add((y_start + 8) * width + x_start);
        transpose_4x4_neon_8b(inp, buf.add(8), width, TILE_SIZE);
        transpose_4x4_neon_8b(inp.add(4), buf.add(4 * TILE_SIZE + 8), width, TILE_SIZE);
        transpose_4x4_neon_8b(inp.add(8), buf.add(8 * TILE_SIZE + 8), width, TILE_SIZE);
        transpose_4x4_neon_8b(inp.add(12), buf.add(12 * TILE_SIZE + 8), width, TILE_SIZE);

        // Block Row 3 (input rows y_start+12..y_start+16)
        let inp = input.add((y_start + 12) * width + x_start);
        transpose_4x4_neon_8b(inp, buf.add(12), width, TILE_SIZE);
        transpose_4x4_neon_8b(inp.add(4), buf.add(4 * TILE_SIZE + 12), width, TILE_SIZE);
        transpose_4x4_neon_8b(inp.add(8), buf.add(8 * TILE_SIZE + 12), width, TILE_SIZE);
        transpose_4x4_neon_8b(inp.add(12), buf.add(12 * TILE_SIZE + 12), width, TILE_SIZE);

        // Flush buffer to output with write prefetching.
        // 16 u64 elements = 128 bytes = 2 cache lines (64 bytes each on ARM64),
        // so we prefetch both cache lines for each output column.
        let out_base = output.add(x_start * height + y_start);
        prefetch_write(out_base as *const u8);
        prefetch_write(out_base.add(8) as *const u8);

        for c in 0..TILE_SIZE {
            if c + 1 < TILE_SIZE {
                let next = output.add((x_start + c + 1) * height + y_start);
                prefetch_write(next as *const u8);
                prefetch_write(next.add(8) as *const u8);
            }
            core::ptr::copy_nonoverlapping(
                buf.add(c * TILE_SIZE),
                output.add((x_start + c) * height + y_start),
                TILE_SIZE,
            );
        }
    }
}

/// Scalar transpose for an arbitrary rectangular block of 8-byte elements.
///
/// Used for handling edge cases where dimensions don't align to tile boundaries.
///
/// # Safety
///
/// Caller must ensure:
/// - Valid pointers for the full matrix
/// - `x_start + block_width <= width`
/// - `y_start + block_height <= height`
#[cfg(target_arch = "aarch64")]
#[inline]
#[allow(clippy::too_many_arguments)]
unsafe fn transpose_block_scalar_8b(
    input: *const u64,
    output: *mut u64,
    width: usize,
    height: usize,
    x_start: usize,
    y_start: usize,
    block_width: usize,
    block_height: usize,
) {
    for inner_x in 0..block_width {
        for inner_y in 0..block_height {
            let x = x_start + inner_x;
            let y = y_start + inner_y;

            let input_index = x + y * width;
            let output_index = y + x * height;

            unsafe {
                *output.add(output_index) = *input.add(input_index);
            }
        }
    }
}

/// Transpose a 4×4 block of 64-bit elements using NEON SIMD.
///
/// This is the fundamental building block for 8-byte element transpose.
///
/// Since a 128-bit NEON register holds only 2 u64 elements, each row of 4
/// elements requires 2 registers. A 4×4 block uses 8 registers for input
/// and 8 for output (16 total, well within NEON's 32 registers).
///
/// # Algorithm
///
/// The transpose uses a single-stage butterfly on four independent 2×2
/// sub-blocks:
///
/// ```text
///     Load:  q0_lo=[a00,a01] q0_hi=[a02,a03]  (row 0)
///            q1_lo=[a10,a11] q1_hi=[a12,a13]  (row 1)
///            q2_lo=[a20,a21] q2_hi=[a22,a23]  (row 2)
///            q3_lo=[a30,a31] q3_hi=[a32,a33]  (row 3)
///
///     Transpose 2×2 sub-blocks:
///       Top-left:     trn1(q0_lo,q1_lo)=[a00,a10]  trn2(q0_lo,q1_lo)=[a01,a11]
///       Top-right:    trn1(q0_hi,q1_hi)=[a02,a12]  trn2(q0_hi,q1_hi)=[a03,a13]
///       Bottom-left:  trn1(q2_lo,q3_lo)=[a20,a30]  trn2(q2_lo,q3_lo)=[a21,a31]
///       Bottom-right: trn1(q2_hi,q3_hi)=[a22,a32]  trn2(q2_hi,q3_hi)=[a23,a33]
///
///     Store: row0=[a00,a10,a20,a30]  row1=[a01,a11,a21,a31]
///            row2=[a02,a12,a22,a32]  row3=[a03,a13,a23,a33]
/// ```
///
/// # Safety
///
/// Caller must ensure:
/// - `src` is valid for reading 4 rows of `src_stride` elements each
/// - `dst` is valid for writing 4 rows of `dst_stride` elements each
/// - The first 4 elements of each row are accessible
#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn transpose_4x4_neon_8b(
    src: *const u64,
    dst: *mut u64,
    src_stride: usize,
    dst_stride: usize,
) {
    unsafe {
        // Load 4 rows, 2 registers per row (4 u64 = 2 × 128-bit)

        // Row 0: [a00, a01] [a02, a03]
        let q0_lo = vld1q_u64(src);
        let q0_hi = vld1q_u64(src.add(2));
        // Row 1: [a10, a11] [a12, a13]
        let q1_lo = vld1q_u64(src.add(src_stride));
        let q1_hi = vld1q_u64(src.add(src_stride + 2));
        // Row 2: [a20, a21] [a22, a23]
        let q2_lo = vld1q_u64(src.add(2 * src_stride));
        let q2_hi = vld1q_u64(src.add(2 * src_stride + 2));
        // Row 3: [a30, a31] [a32, a33]
        let q3_lo = vld1q_u64(src.add(3 * src_stride));
        let q3_hi = vld1q_u64(src.add(3 * src_stride + 2));

        // Transpose four 2×2 sub-blocks using vtrn1q_u64/vtrn2q_u64

        // Top-left: rows 0,1 × columns 0,1
        let r0_lo = vtrn1q_u64(q0_lo, q1_lo); // [a00, a10]
        let r1_lo = vtrn2q_u64(q0_lo, q1_lo); // [a01, a11]
        // Top-right: rows 0,1 × columns 2,3
        let r2_lo = vtrn1q_u64(q0_hi, q1_hi); // [a02, a12]
        let r3_lo = vtrn2q_u64(q0_hi, q1_hi); // [a03, a13]
        // Bottom-left: rows 2,3 × columns 0,1
        let r0_hi = vtrn1q_u64(q2_lo, q3_lo); // [a20, a30]
        let r1_hi = vtrn2q_u64(q2_lo, q3_lo); // [a21, a31]
        // Bottom-right: rows 2,3 × columns 2,3
        let r2_hi = vtrn1q_u64(q2_hi, q3_hi); // [a22, a32]
        let r3_hi = vtrn2q_u64(q2_hi, q3_hi); // [a23, a33]

        // Store 4 transposed rows, 2 registers per row

        // Row 0: [a00, a10, a20, a30]
        vst1q_u64(dst, r0_lo);
        vst1q_u64(dst.add(2), r0_hi);
        // Row 1: [a01, a11, a21, a31]
        vst1q_u64(dst.add(dst_stride), r1_lo);
        vst1q_u64(dst.add(dst_stride + 2), r1_hi);
        // Row 2: [a02, a12, a22, a32]
        vst1q_u64(dst.add(2 * dst_stride), r2_lo);
        vst1q_u64(dst.add(2 * dst_stride + 2), r2_hi);
        // Row 3: [a03, a13, a23, a33]
        vst1q_u64(dst.add(3 * dst_stride), r3_lo);
        vst1q_u64(dst.add(3 * dst_stride + 2), r3_hi);
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec;
    use alloc::vec::Vec;

    use p3_baby_bear::BabyBear;
    use p3_field::PrimeCharacteristicRing;
    use p3_goldilocks::Goldilocks;
    use proptest::prelude::*;

    use super::*;

    /// Naive reference implementation for correctness testing.
    fn transpose_reference<T: Copy + Default>(input: &[T], width: usize, height: usize) -> Vec<T> {
        // Allocate output buffer with same size as input.
        let mut output = vec![T::default(); width * height];

        // For each position (x, y) in the input matrix:
        // - Input index: y * width + x (row-major)
        // - Output index: x * height + y (transposed row-major)
        for y in 0..height {
            for x in 0..width {
                output[x * height + y] = input[y * width + x];
            }
        }

        output
    }

    /// Strategy for generating matrix dimensions.
    fn dimension_strategy() -> impl Strategy<Value = (usize, usize)> {
        // Compute boundary dimensions from constants.
        // `small_side` is the largest square that stays in the small (scalar) path.
        let small_side = (SMALL_LEN as f64).sqrt() as usize;
        // `medium_side` is the largest square that stays in the medium (tiled) path.
        let medium_side = (MEDIUM_LEN as f64).sqrt() as usize;
        // `large_side` is the side length that triggers the large (recursive) path.
        let large_side = medium_side + 1;

        prop_oneof![
            // Edge cases: empty and degenerate matrices
            //
            // Empty matrix (0×0)
            Just((0, 0)),
            // Single row (1×n) - tests degenerate case
            (1..=100_usize).prop_map(|w| (w, 1)),
            // Single column (n×1) - tests degenerate case
            (1..=100_usize).prop_map(|h| (1, h)),
            // Small path: len < SMALL_LEN (scalar transpose)
            //
            // These dimensions exercise the scalar transpose path.

            // Tiny matrices (various shapes within small threshold)
            (1..=small_side, 1..=small_side),
            // Medium path: SMALL_LEN ≤ len < MEDIUM_LEN (tiled TILE_SIZE×TILE_SIZE)
            //
            // These dimensions exercise the tiled TILE_SIZE×TILE_SIZE path.

            // Exactly 4×4 (single NEON block)
            Just((4, 4)),
            // Exactly TILE_SIZE×TILE_SIZE (single tile)
            Just((TILE_SIZE, TILE_SIZE)),
            // Multiple complete tiles (2× and 4× TILE_SIZE)
            Just((TILE_SIZE * 2, TILE_SIZE * 2)),
            Just((TILE_SIZE * 4, TILE_SIZE * 4)),
            // Non-aligned: has remainders in both dimensions
            // Range from just above TILE_SIZE to below 4×TILE_SIZE.
            // These test the scalar fallback for tile edges.
            (
                (TILE_SIZE + 1)..=(TILE_SIZE * 4 - 1),
                (TILE_SIZE + 1)..=(TILE_SIZE * 4 - 1)
            ),
            // Wide rectangle with remainders (medium path)
            (50..=200_usize, 10..=50_usize),
            // Tall rectangle with remainders (medium path)
            (10..=50_usize, 50..=200_usize),
            // Large path: MEDIUM_LEN ≤ len < PARALLEL_THRESHOLD (recursive)
            //
            // These exercise the cache-oblivious recursive subdivision.

            // Square matrices triggering recursion (just above medium threshold)
            Just((large_side, large_side)),
            // Slightly larger square
            Just((large_side + 100, large_side + 100)),
            // Wide rectangle triggering recursion
            Just((large_side * 2, large_side / 2)),
            // Tall rectangle triggering recursion
            Just((large_side / 2, large_side * 2)),
            // Non-power-of-2 dimensions in large range
            Just((large_side + 50, large_side + 75)),
        ]
    }

    proptest! {
        #[test]
        fn proptest_transpose_babybear((width, height) in dimension_strategy()) {
            // Skip empty matrices (they're trivially correct).
            if width == 0 || height == 0 {
                // Just verify it doesn't panic.
                let input: [BabyBear; 0] = [];
                let mut output: [BabyBear; 0] = [];
                transpose(&input, &mut output, width, height);
                return Ok(());
            }

            // Create input matrix with unique values at each position.
            let input: Vec<BabyBear> = (0..width * height)
                .map(|i| BabyBear::from_u64(i as u64))
                .collect();

            // Allocate output buffer.
            let mut output = vec![BabyBear::ZERO; width * height];

            // Run optimized transpose.
            transpose(&input, &mut output, width, height);

            // Run reference transpose.
            let expected = transpose_reference(&input, width, height);

            // Verify results match.
            prop_assert_eq!(
                output,
                expected,
                "Transpose mismatch for {}×{} matrix",
                width,
                height
            );
        }

        #[test]
        fn proptest_transpose_u64((width, height) in dimension_strategy()) {
            // Skip empty and very large matrices for u64 (memory intensive).
            if width == 0 || height == 0 || width * height > 100_000 {
                return Ok(());
            }

            // Create input with unique values.
            let input: Vec<u64> = (0..width * height).map(|i| i as u64).collect();

            // Allocate output.
            let mut output = vec![0u64; width * height];

            // Run transpose.
            transpose(&input, &mut output, width, height);

            // Verify against reference.
            let expected = transpose_reference(&input, width, height);
            prop_assert_eq!(output, expected);
        }

        #[test]
        fn proptest_transpose_u8((width, height) in dimension_strategy()) {
            // Skip empty and very large matrices.
            if width == 0 || height == 0 || width * height > 100_000 {
                return Ok(());
            }

            // Create input with unique values (wrapping for u8).
            let input: Vec<u8> = (0..width * height).map(|i| i as u8).collect();

            // Allocate output.
            let mut output = vec![0u8; width * height];

            // Run transpose.
            transpose(&input, &mut output, width, height);

            // Verify against reference.
            let expected = transpose_reference(&input, width, height);
            prop_assert_eq!(output, expected);
        }

        #[test]
        fn proptest_transpose_goldilocks((width, height) in dimension_strategy()) {
            // Skip empty matrices.
            if width == 0 || height == 0 {
                let input: [Goldilocks; 0] = [];
                let mut output: [Goldilocks; 0] = [];
                transpose(&input, &mut output, width, height);
                return Ok(());
            }

            // Create input matrix with unique values at each position.
            let input: Vec<Goldilocks> = (0..width * height)
                .map(|i| Goldilocks::from_u64(i as u64))
                .collect();

            // Allocate output buffer.
            let mut output = vec![Goldilocks::ZERO; width * height];

            // Run optimized transpose.
            transpose(&input, &mut output, width, height);

            // Run reference transpose.
            let expected = transpose_reference(&input, width, height);

            // Verify results match.
            prop_assert_eq!(
                output,
                expected,
                "Transpose mismatch for {}×{} matrix",
                width,
                height
            );
        }
    }
}
