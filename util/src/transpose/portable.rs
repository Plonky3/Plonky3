//! Portable cache-oblivious matrix transpose, the fallback when no SIMD kernel applies.
//!
//! Maps `input[y * width + x]` to `output[x * height + y]` for any `Copy` type.

use core::mem::MaybeUninit;

/// Largest element count handled by the plain scalar copy.
///
/// A matrix this small fits in L1, so tiling would only add loop-setup overhead.
const SMALL_LEN: usize = 255;

/// Largest element count handled by single-level tiling before recursing.
///
/// - Below this, the working set still tiles well against L2/L3.
/// - Past it, recursive subdivision keeps each leaf cache-resident at any size.
const MEDIUM_LEN: usize = 1 << 20;

/// Side length, in elements, of one transpose tile.
///
/// - A tile is transposed as a unit, so it stays cache-hot for the whole tile.
/// - For 4-byte elements each output run is one 64-byte cache line.
const TILE: usize = 16;

/// Number of horizontal passes a full tile is split into for large matrices.
///
/// - Each pass touches fewer source rows, shrinking the active read footprint.
/// - That keeps reads inside the TLB and L1 working set when rows are far apart.
/// - The tile side must be a multiple of this value.
const NBR_SEGMENTS: usize = 4;

/// Source-row count covered by one tile segment.
const SEGMENT: usize = TILE / NBR_SEGMENTS;

/// Side length at which recursion stops and switches to tiling.
///
/// At this size a leaf sub-matrix fits in L2, so further splitting only adds overhead.
const RECURSIVE_LIMIT: usize = 128;

/// Bytes in a base memory page, the reach of one TLB entry.
const PAGE_BYTES: usize = 4096;

/// Source rows per band of a right-edge strip copied by scalar code.
///
/// A band of a few columns then stays in L1 while each of its columns is copied out.
const STRIP_BAND: usize = 256;

/// Minimum element count before the parallel band path is used.
///
/// Smaller matrices are dominated by thread fan-out and cache-coherency traffic.
#[cfg(feature = "parallel")]
const PARALLEL_THRESHOLD: usize = 4 << 20;

/// Whether whole tiles of this element type move through register tiles.
///
/// The AVX-512 tiles, for 4-byte and 8-byte words, beat the compiler's own vectorization of the scalar tile.
///
/// Narrower AVX2 tiles lose to it on cache-resident matrices, so they keep the scalar kernels here.
///
/// Every input is a constant, so the check folds away per element type.
const fn word_tiles<T>() -> bool {
    cfg!(all(target_arch = "x86_64", target_feature = "avx512f"))
        && matches!((size_of::<T>(), align_of::<T>()), (4, 4) | (8, 8))
}

/// Transpose one block through the register-tile kernel.
///
/// # Arguments
///
/// - `total_cols`: input row stride.
/// - `total_rows`: output row stride.
/// - `start_x`, `start_y`: the block's first source column and row.
/// - `out_col_start`: the output row that receives the block's first source column.
/// - `block_width`, `block_height`: the block's extent.
///
/// # Safety
///
/// Elements must support the integer-word copies required by `super::transpose_words`.
#[inline]
#[allow(clippy::too_many_arguments)]
unsafe fn transpose_block_tiled<T: Copy>(
    input: &[T],
    output: &mut [MaybeUninit<T>],
    total_cols: usize,
    total_rows: usize,
    start_x: usize,
    start_y: usize,
    out_col_start: usize,
    block_width: usize,
    block_height: usize,
) {
    // Source row i of the block, from its first column.
    let row = |i: usize| {
        let start = (start_y + i) * total_cols + start_x;
        &input[start..start + block_width]
    };

    // Output run j of the block starts one output stride after run j - 1.
    let dst = &mut output[out_col_start * total_rows + start_y..];
    // SAFETY: the word engine caller guarantees integer-word-compatible elements.
    unsafe {
        super::rows::transpose_rows_strided_words_uninit(
            block_height,
            block_width,
            row,
            dst,
            total_rows,
        );
    };
}

/// Transpose one clipped edge block through the register-tile kernel.
///
/// Edge shapes are only known at run time, so one shared copy of the masked kernel serves them all.
///
/// Keeping it out of line keeps the whole-tile loops small.
///
/// # Safety
///
/// Same contract as the block kernel.
#[inline(never)]
#[allow(clippy::too_many_arguments)]
unsafe fn transpose_edge_tiled<T: Copy>(
    input: &[T],
    output: &mut [MaybeUninit<T>],
    total_cols: usize,
    total_rows: usize,
    start_x: usize,
    start_y: usize,
    out_col_start: usize,
    block_width: usize,
    block_height: usize,
) {
    // SAFETY: forwarded block contract.
    unsafe {
        transpose_block_tiled(
            input,
            output,
            total_cols,
            total_rows,
            start_x,
            start_y,
            out_col_start,
            block_width,
            block_height,
        );
    };
}

/// Transpose a row-major matrix from `input` into `output`.
///
/// - `input` is read as `height` rows of `width` columns.
/// - `output` receives the transpose as `width` rows of `height` columns.
///
/// Output may be uninitialized. Every slot receives a valid input copy before
/// return; no existing slot is read or deinitialized, including on panic.
///
/// # Panics
///
/// Panics if `width * height` overflows or differs from either slice length.
#[inline]
pub(super) fn transpose_uninit<T: Copy + Send + Sync>(
    input: &[T],
    output: &mut [MaybeUninit<T>],
    width: usize,
    height: usize,
) {
    // SAFETY: the typed engine never reinterprets elements as integer words.
    unsafe { transpose_impl::<false, T>(input, output, width, height) };
}

/// # Safety
/// Input elements must satisfy the integer-word contract of `super::transpose_words_owned`.
#[inline]
pub(super) unsafe fn transpose_words_uninit<T: Copy + Send + Sync>(
    input: &[T],
    output: &mut [MaybeUninit<T>],
    width: usize,
    height: usize,
) {
    // SAFETY: forwarded representation guarantee.
    unsafe { transpose_impl::<true, T>(input, output, width, height) };
}

/// # Safety
/// With `WORDS`, elements must support integer-word copying.
#[inline]
unsafe fn transpose_impl<const WORDS: bool, T: Copy + Send + Sync>(
    input: &[T],
    output: &mut [MaybeUninit<T>],
    width: usize,
    height: usize,
) {
    let len = width
        .checked_mul(height)
        .expect("transpose dimensions overflow");
    assert_eq!(input.len(), len, "transpose input has the wrong length");
    assert_eq!(output.len(), len, "transpose output has the wrong length");

    // A zero-size matrix has nothing to move.
    if width == 0 || height == 0 {
        return;
    }

    // Large matrices: hand off to the parallel band split.
    #[cfg(feature = "parallel")]
    if len >= PARALLEL_THRESHOLD {
        // SAFETY: checked shape and, when enabled, the forwarded word representation contract.
        unsafe { transpose_parallel::<WORDS, T>(input, output, width, height) };
        return;
    }

    // Tiny matrices: the scalar copy beats any tiling setup.
    if len <= SMALL_LEN {
        transpose_small(input, output, width, height);
        return;
    }

    // SAFETY:
    // - Both slices hold exactly `width * height` elements (asserted above).
    // - The whole matrix is one region with output base column 0,
    //   so every index the kernels form stays in `[0, width * height)`.
    unsafe {
        if len <= MEDIUM_LEN {
            // Medium: tile the whole matrix once, no recursion overhead.
            transpose_region::<false, WORDS, T>(
                input, output, 0, height, 0, width, width, height, 0,
            );
        } else {
            // Large: recurse down to cache-resident leaves, then tile.
            transpose_recursive::<WORDS, T>(input, output, 0, height, 0, width, width, height, 0);
        }
    }
}

/// Element-by-element transpose for matrices that fit in L1.
///
/// - Iterating the source row index innermost keeps the writes contiguous.
/// - The strided reads are cheap while the whole matrix is L1-resident.
#[inline]
fn transpose_small<T: Copy>(
    input: &[T],
    output: &mut [MaybeUninit<T>],
    width: usize,
    height: usize,
) {
    // Walk output rows (source columns) one at a time.
    for x in 0..width {
        // Walk down this output row, filling the contiguous run `[x*height ..][..height]`.
        for y in 0..height {
            // SAFETY: `x < width` and `y < height`, so both indices are `< width*height`.
            unsafe {
                // Source element (row y, col x) -> output element (row x, col y).
                output
                    .get_unchecked_mut(x * height + y)
                    .write(*input.get_unchecked(x + y * width));
            }
        }
    }
}

/// Transpose an arbitrary block with contiguous writes, for a region's edges.
///
/// - Slices (not pointers) let the compiler assume no aliasing and pipeline reads.
/// - A parallel column-band renumbers output rows from zero; otherwise they match the source column.
///
/// # Safety
///
/// - `input` holds `total_cols * total_rows` elements.
/// - The largest source index read is `(start_x + block_width - 1) + (start_y + block_height - 1) * total_cols`.
/// - The largest output index written is `(out_col_start + block_width - 1) * total_rows + start_y + block_height - 1`.
/// - Callers keep both within their respective slice lengths.
#[inline]
#[allow(clippy::too_many_arguments)]
unsafe fn transpose_block<T: Copy>(
    input: &[T],
    output: &mut [MaybeUninit<T>],
    total_cols: usize,
    total_rows: usize,
    start_x: usize,
    start_y: usize,
    out_col_start: usize,
    block_width: usize,
    block_height: usize,
) {
    unsafe {
        // Each source column becomes one contiguous output run.
        for inner_x in 0..block_width {
            // Absolute source column.
            let x = start_x + inner_x;
            // Start of the matching contiguous output run.
            let out_base = (out_col_start + inner_x) * total_rows + start_y;
            // Stream down the column: contiguous stores, strided loads.
            for inner_y in 0..block_height {
                output
                    .get_unchecked_mut(out_base + inner_y)
                    .write(*input.get_unchecked(x + (start_y + inner_y) * total_cols));
            }
        }
    }
}

/// Transpose a full `TILE x TILE` block with compile-time-constant bounds.
///
/// The constant trip count lets the compiler unroll and vectorize the inner loop.
/// Fast kernel for the cache-resident medium path.
///
/// # Safety
///
/// As the generic block kernel, with both block dimensions equal to the tile side.
#[inline]
unsafe fn transpose_tile_plain<T: Copy>(
    input: &[T],
    output: &mut [MaybeUninit<T>],
    total_cols: usize,
    total_rows: usize,
    start_x: usize,
    start_y: usize,
    out_col_start: usize,
) {
    unsafe {
        // One contiguous output run per source column; bounds are the `TILE` constant.
        for inner_x in 0..TILE {
            let x = start_x + inner_x;
            let out_base = (out_col_start + inner_x) * total_rows + start_y;
            for inner_y in 0..TILE {
                output
                    .get_unchecked_mut(out_base + inner_y)
                    .write(*input.get_unchecked(x + (start_y + inner_y) * total_cols));
            }
        }
    }
}

/// Transpose a full `TILE x TILE` block in several horizontal passes.
///
/// Same result as the plain tile kernel, but each pass covers fewer source rows.
/// Used for the recursive large-matrix leaves, where rows are far apart and TLB/L1 pressure dominates.
///
/// # Safety
///
/// As the plain tile kernel.
#[inline]
unsafe fn transpose_tile_segmented<T: Copy>(
    input: &[T],
    output: &mut [MaybeUninit<T>],
    total_cols: usize,
    total_rows: usize,
    start_x: usize,
    start_y: usize,
    out_col_start: usize,
) {
    unsafe {
        // Sweep the tile top-to-bottom in `NBR_SEGMENTS` short bands.
        for segment in 0..NBR_SEGMENTS {
            // First source row of this band.
            let seg_y = start_y + segment * SEGMENT;
            for inner_x in 0..TILE {
                let x = start_x + inner_x;
                let out_base = (out_col_start + inner_x) * total_rows + seg_y;
                for inner_y in 0..SEGMENT {
                    output
                        .get_unchecked_mut(out_base + inner_y)
                        .write(*input.get_unchecked(x + (seg_y + inner_y) * total_cols));
                }
            }
        }
    }
}

/// Tiled transpose of the sub-region `[row_start, row_end) x [col_start, col_end)`.
///
/// Whole `TILE x TILE` tiles use a constant-bound kernel.
/// Partial tiles at the right and bottom edges use the generic block kernel.
///
/// Element types with a register-tile kernel take it for every whole tile, in column-stripe order.
///
/// For the others, `SEGMENTED` selects the whole-tile kernel:
/// - `false` (medium path): plain tiles stream best while the matrix is cache-resident.
/// - `true` (recursive leaves): segmented passes cut TLB/L1 pressure on large matrices.
///
/// # Arguments
///
/// - `total_cols`: input row stride (the full matrix width).
/// - `total_rows`: output row stride (the full matrix height).
/// - `col_out_base`: absolute source column mapped to output row 0 (0 for the whole matrix).
///
/// # Safety
///
/// - If `WORDS`, elements satisfy `super::transpose_words`'s representation contract.
/// - `input.len() == total_cols * total_rows`.
/// - `row_end <= total_rows` and `col_end <= total_cols`.
/// - `output` covers output rows `[col_start - col_out_base, col_end - col_out_base)`.
#[inline]
#[allow(clippy::too_many_arguments)]
unsafe fn transpose_region<const SEGMENTED: bool, const WORDS: bool, T: Copy>(
    input: &[T],
    output: &mut [MaybeUninit<T>],
    row_start: usize,
    row_end: usize,
    col_start: usize,
    col_end: usize,
    total_cols: usize,
    total_rows: usize,
    col_out_base: usize,
) {
    // Split the region into whole tiles plus the leftover edge widths.
    let nbr_rows = row_end - row_start;
    let nbr_cols = col_end - col_start;
    let x_tiles = nbr_cols / TILE;
    let y_tiles = nbr_rows / TILE;
    let remainder_x = nbr_cols - x_tiles * TILE;
    let remainder_y = nbr_rows - y_tiles * TILE;

    // Output row index of this region's first source column.
    let out_col_start = col_start - col_out_base;

    if WORDS && word_tiles::<T>() {
        // One register tile spans one cache line of elements, in both directions.
        let side = 64 / size_of::<T>();

        // Whole tiles end here.
        //
        // Past these bounds lie the clipped right and bottom strips.
        let x_end = nbr_cols - nbr_cols % side;
        let y_end = nbr_rows - nbr_rows % side;

        // Copy the block at a region offset through one of the kernels.
        macro_rules! block {
            ($kernel:ident, $x:expr, $y:expr, $width:expr, $height:expr) => {
                // SAFETY: WORDS forwards the representation guarantee; each block lies in the region.
                unsafe {
                    $kernel(
                        input,
                        output,
                        total_cols,
                        total_rows,
                        col_start + $x,
                        row_start + $y,
                        out_col_start + $x,
                        $width,
                        $height,
                    )
                }
            };
        }

        // Column stripes: every output row is written front to back.
        for x in 0..x_end / side {
            for y in 0..y_end / side {
                block!(transpose_block_tiled, x * side, y * side, side, side);
            }
        }

        // Right strip, then bottom strip.
        //
        // Source rows a page apart cost a TLB lookup per element on a scalar column walk.
        //
        // The masked kernel reads each such row once instead.
        let right = nbr_cols - x_end;
        if right > 0 && total_cols * size_of::<T>() >= PAGE_BYTES {
            block!(transpose_edge_tiled, x_end, 0, right, nbr_rows);
        } else if right > 0 {
            // Bands of rows keep the strip's source lines in L1 across its few columns.
            for y in (0..nbr_rows).step_by(STRIP_BAND) {
                block!(
                    transpose_block,
                    x_end,
                    y,
                    right,
                    STRIP_BAND.min(nbr_rows - y)
                );
            }
        }
        if y_end < nbr_rows {
            block!(transpose_edge_tiled, 0, y_end, x_end, nbr_rows - y_end);
        }
        return;
    } else {
        // SAFETY: every whole tile lies inside the region, which the caller bounds.
        unsafe {
            // Whole tiles: tight hot loop, no per-tile bookkeeping.
            // `SEGMENTED` is const, so the branch is resolved at compile time.
            for y_tile in 0..y_tiles {
                let row = row_start + y_tile * TILE;
                for x_tile in 0..x_tiles {
                    // Source column and matching output row for this tile.
                    let col = col_start + x_tile * TILE;
                    let out_col = out_col_start + x_tile * TILE;
                    if SEGMENTED {
                        transpose_tile_segmented(
                            input, output, total_cols, total_rows, col, row, out_col,
                        );
                    } else {
                        transpose_tile_plain(
                            input, output, total_cols, total_rows, col, row, out_col,
                        );
                    }
                }
            }
        }
    }

    unsafe {
        for y_tile in 0..y_tiles {
            let row = row_start + y_tile * TILE;

            // Right edge: partial-width tiles for this band of rows.
            if remainder_x > 0 {
                transpose_block(
                    input,
                    output,
                    total_cols,
                    total_rows,
                    col_start + x_tiles * TILE,
                    row,
                    out_col_start + x_tiles * TILE,
                    remainder_x,
                    TILE,
                );
            }
        }

        // Bottom edge: partial-height tiles below the last whole tile row.
        if remainder_y > 0 {
            let row = row_start + y_tiles * TILE;
            for x_tile in 0..x_tiles {
                transpose_block(
                    input,
                    output,
                    total_cols,
                    total_rows,
                    col_start + x_tile * TILE,
                    row,
                    out_col_start + x_tile * TILE,
                    TILE,
                    remainder_y,
                );
            }

            // Bottom-right corner: partial in both dimensions.
            if remainder_x > 0 {
                transpose_block(
                    input,
                    output,
                    total_cols,
                    total_rows,
                    col_start + x_tiles * TILE,
                    row,
                    out_col_start + x_tiles * TILE,
                    remainder_x,
                    remainder_y,
                );
            }
        }
    }
}

/// Cache-oblivious recursive transpose of a rectangular region.
///
/// Splits along the longer side until both sides fit in cache, then tiles each leaf.
/// Halving the longer side keeps leaves near-square, so the working set stays bounded.
///
/// ```text
///     rows >= cols          cols > rows
///     split horizontally    split vertically
///     +-----------+         +-----+-----+
///     |    top    |         |     |     |
///     +-----------+         |left |right|
///     |  bottom   |         |     |     |
///     +-----------+         +-----+-----+
/// ```
///
/// # Safety
///
/// Same contract as the tiled-region kernel.
#[inline]
#[allow(clippy::too_many_arguments)]
unsafe fn transpose_recursive<const WORDS: bool, T: Copy>(
    input: &[T],
    output: &mut [MaybeUninit<T>],
    row_start: usize,
    row_end: usize,
    col_start: usize,
    col_end: usize,
    total_cols: usize,
    total_rows: usize,
    col_out_base: usize,
) {
    let nbr_rows = row_end - row_start;
    let nbr_cols = col_end - col_start;

    unsafe {
        // Base case: a leaf that fits in cache, or a side too thin to split.
        if (nbr_rows <= RECURSIVE_LIMIT && nbr_cols <= RECURSIVE_LIMIT)
            || nbr_rows <= 2
            || nbr_cols <= 2
        {
            transpose_region::<true, WORDS, T>(
                input,
                output,
                row_start,
                row_end,
                col_start,
                col_end,
                total_cols,
                total_rows,
                col_out_base,
            );
            return;
        }

        if nbr_rows >= nbr_cols {
            // Taller than wide: cut the row range in half.
            let mid = row_start + nbr_rows / 2;
            transpose_recursive::<WORDS, T>(
                input,
                output,
                row_start,
                mid,
                col_start,
                col_end,
                total_cols,
                total_rows,
                col_out_base,
            );
            transpose_recursive::<WORDS, T>(
                input,
                output,
                mid,
                row_end,
                col_start,
                col_end,
                total_cols,
                total_rows,
                col_out_base,
            );
        } else {
            // Wider than tall: cut the column range in half.
            let mid = col_start + nbr_cols / 2;
            transpose_recursive::<WORDS, T>(
                input,
                output,
                row_start,
                row_end,
                col_start,
                mid,
                total_cols,
                total_rows,
                col_out_base,
            );
            transpose_recursive::<WORDS, T>(
                input,
                output,
                row_start,
                row_end,
                mid,
                col_end,
                total_cols,
                total_rows,
                col_out_base,
            );
        }
    }
}

/// Parallel transpose over disjoint output bands.
///
/// By default, output rows are cut into contiguous bands of whole source columns.
/// Band `k` holds the transpose of source columns starting at `k * cols_per_task`.
/// Bands never overlap, so each task gets an exclusive slice — no shared pointers.
///
/// ```text
///     source columns   ->   output rows (one contiguous band)
///     [0, c)           ->   output[0 .. c*height)
///     [c, 2c)          ->   output[c*height .. 2c*height)
///     ...
/// ```
///
/// For widths up to two that cannot fill the worker pool, each output row is
/// split again into source-row bands. The band for column `x` and source rows `[a, b)` owns
/// exactly `output[x * height + a .. x * height + b]`, obtained through safe
/// slice splitting. Distinct rows and distinct bands within a row never overlap.
///
/// # Safety
///
/// Both slices describe the checked `width * height` shape, with nonzero dimensions.
/// If `WORDS`, elements satisfy `super::transpose_words`'s representation contract.
#[cfg(feature = "parallel")]
unsafe fn transpose_parallel<const WORDS: bool, T: Copy + Send + Sync>(
    input: &[T],
    output: &mut [MaybeUninit<T>],
    width: usize,
    height: usize,
) {
    use p3_maybe_rayon::prelude::*;

    let num_threads = current_num_threads();

    // Keep wider matrices on the tiled path: column gathers amplify input traffic.
    if width <= 2 && width < num_threads {
        // Whole-column bands would expose only `width` tasks. Split each output
        // row (one source column) into contiguous source-row bands as well.
        let bands_per_col = num_threads.div_ceil(width);
        // Rounding to 64 elements keeps band boundaries at the same offset in a
        // 64-byte cache line within each output row; only the final band can be shorter.
        // Rows whose start is not line-aligned share one line between adjacent bands.
        let rows_per_task = height.div_ceil(bands_per_col).next_multiple_of(64);
        output
            .par_chunks_mut(height)
            .enumerate()
            .for_each(|(x, row)| {
                row.par_chunks_mut(rows_per_task)
                    .enumerate()
                    .for_each(|(task, band)| {
                        let row_start = task * rows_per_task;
                        // `band` exclusively owns this contiguous output run.
                        if width == 1 {
                            // SAFETY: the input run and exclusive output band have the
                            // same length, and copying T also preserves padding/provenance.
                            unsafe {
                                core::ptr::copy_nonoverlapping(
                                    input.as_ptr().add(row_start),
                                    band.as_mut_ptr().cast::<T>(),
                                    band.len(),
                                );
                            }
                        } else {
                            let src = input[row_start * width + x..].iter().step_by(width);
                            for (dst, &value) in band.iter_mut().zip(src) {
                                dst.write(value);
                            }
                        }
                    });
            });
        return;
    }

    // Whole-column bands use at most one worker each; only the last can be short.
    let cols_per_task = width.div_ceil(num_threads);

    // Each band is `cols_per_task` output rows of `height` columns.
    output
        .par_chunks_mut(cols_per_task * height)
        .enumerate()
        .for_each(|(task, band)| {
            // Absolute source columns this band owns.
            let col_start = task * cols_per_task;
            let col_end = col_start.saturating_add(cols_per_task).min(width);

            // SAFETY:
            // - `band` is exactly `(col_end - col_start) * height` long (only the last is shorter).
            // - The engine writes output rows `[0, col_end - col_start)`, all inside `band`.
            // - Reads address the shared `input`, valid for `width * height` elements.
            unsafe {
                transpose_recursive::<WORDS, T>(
                    input, band, 0, height, col_start, col_end, width, height, col_start,
                );
            }
        });
}

#[cfg(test)]
mod tests {
    extern crate alloc;

    use alloc::vec;
    use alloc::vec::Vec;

    use proptest::prelude::*;

    use super::*;
    use crate::transpose::transpose;

    #[test]
    #[should_panic(expected = "transpose dimensions overflow")]
    fn rejects_overflowing_dimensions_before_access() {
        // The unchecked product would wrap to the slice length of 2.
        transpose::<u32>(&[0; 2], &mut [0; 2], usize::MAX / 2 + 2, 2);
    }

    /// Independent textbook transpose used as the source of truth.
    fn reference<T: Copy>(input: &[T], width: usize, height: usize, zero: T) -> Vec<T> {
        let mut out = vec![zero; width * height];
        for y in 0..height {
            for x in 0..width {
                out[x * height + y] = input[y * width + x];
            }
        }
        out
    }

    /// Build a `width x height`, transpose it, and compare.
    fn check<T>(width: usize, height: usize, zero: T, make: impl Fn(usize) -> T)
    where
        T: Copy + PartialEq + core::fmt::Debug + Send + Sync,
    {
        // Distinct value per cell.
        let input: Vec<T> = (0..width * height).map(&make).collect();
        // Engine under test.
        let mut output = vec![zero; width * height];
        transpose(&input, &mut output, width, height);
        // Independent oracle.
        let expected = reference(&input, width, height, zero);
        // Every position must agree.
        assert_eq!(output, expected, "mismatch for {width}x{height}");
    }

    /// Dimensions covering every code path the engine can take.
    ///
    /// One `prop_oneof` branch per regime, so fuzzing samples them all:
    ///
    /// ```text
    ///     degenerate : empty / single element / single row / single column
    ///     scalar     : len <= SMALL_LEN
    ///     tiled      : around one tile, several tiles, both-edge remainders
    ///     recursive  : just past MEDIUM_LEN, varied aspect ratios + remainders
    /// ```
    ///
    /// Recursive shapes are fixed (not ranges) to keep allocations bounded while
    /// still exercising column splits, row splits, and edge remainders.
    fn dims() -> impl Strategy<Value = (usize, usize)> {
        prop_oneof![
            // Degenerate: empty, single element, single row, single column.
            Just((0, 0)),
            Just((1, 1)),
            (1..=200_usize).prop_map(|w| (w, 1)),
            (1..=200_usize).prop_map(|h| (1, h)),
            // Scalar path: total length below the small threshold.
            (1..=15_usize, 1..=15_usize),
            // Tile boundaries: just below, at, and just above one tile.
            ((TILE - 1)..=(TILE + 1), (TILE - 1)..=(TILE + 1)),
            // Several whole tiles, no remainder.
            Just((TILE * 4, TILE * 4)),
            // Tiled path with remainders on both edges.
            ((TILE + 1)..=(TILE * 6), (TILE + 1)..=(TILE * 6)),
            // Around the recursion-leaf side length.
            (120..=140_usize, 120..=140_usize),
            // Rectangular tiled shapes (wide and tall).
            (40..=300_usize, 10..=60_usize),
            (10..=60_usize, 40..=300_usize),
            // Exactly MEDIUM_LEN: last shape on the single-level tiled path.
            Just((1024, 1024)),
            // Recursive path: just past MEDIUM_LEN with assorted aspect ratios.
            Just((1025, 1024)), // remainder columns
            Just((1024, 1025)), // remainder rows
            Just((1100, 1000)), // remainders on both edges
            Just((2003, 757)),  // wide, prime-ish sides
            Just((757, 2003)),  // tall, prime-ish sides
        ]
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(64))]
        // 4-byte elements: the common 32-bit field width; fuzz every regime.
        #[test]
        fn portable_matches_reference_u32((width, height) in dims()) {
            check(width, height, 0u32, |i| i as u32);
        }

        // 8-byte elements: 64-bit fields like Goldilocks; fuzz every regime.
        #[test]
        fn portable_matches_reference_u64((width, height) in dims()) {
            check(width, height, 0u64, |i| i as u64);
        }
    }

    /// Dimension strategy for the wider element types.
    ///
    /// Covers the same regimes as the main strategy, with only one recursive shape.
    /// Capping the large shapes keeps the bigger element types cheap to allocate.
    fn dims_small() -> impl Strategy<Value = (usize, usize)> {
        prop_oneof![
            Just((0, 0)),
            Just((1, 1)),
            (1..=200_usize).prop_map(|w| (w, 1)),
            (1..=200_usize).prop_map(|h| (1, h)),
            (1..=15_usize, 1..=15_usize),
            ((TILE - 1)..=(TILE + 1), (TILE - 1)..=(TILE + 1)),
            ((TILE + 1)..=(TILE * 6), (TILE + 1)..=(TILE * 6)),
            (40..=300_usize, 10..=60_usize),
            (10..=60_usize, 40..=300_usize),
            // One recursive shape, so generic indexing is checked past MEDIUM_LEN too.
            Just((1025, 1024)),
        ]
    }

    proptest! {
        // 1-byte elements: smallest stride, stresses the index arithmetic.
        #[test]
        fn portable_matches_reference_u8((width, height) in dims_small()) {
            // Values wrap at 256, but the reference wraps identically, so a swap is still caught.
            check(width, height, 0u8, |i| i as u8);
        }

        // 16-byte elements: degree-2 / degree-4 extension fields.
        #[test]
        fn portable_matches_reference_u128((width, height) in dims_small()) {
            check(width, height, 0u128, |i| i as u128);
        }

        // 3-byte element: a non-power-of-two size, pins generic stride handling.
        #[test]
        fn portable_matches_reference_3byte((width, height) in dims_small()) {
            check(width, height, [0u8; 3], |i| [i as u8, (i >> 8) as u8, (i >> 16) as u8]);
        }

        // 32-byte element: a wide extension-field-sized type.
        #[test]
        fn portable_matches_reference_32byte((width, height) in dims_small()) {
            check(width, height, [0u64; 4], |i| [i as u64, 0, 0, i as u64]);
        }
    }

    #[cfg(feature = "parallel")]
    fn check_parallel<T>(width: usize, height: usize, zero: T, make: impl Fn(usize) -> T)
    where
        T: Copy + PartialEq + core::fmt::Debug + Send + Sync,
    {
        let input: Vec<T> = (0..width * height).map(make).collect();
        let mut output = vec![zero; input.len()];
        // SAFETY: both buffers have the nonzero width * height shape used by
        // these tests. The typed engine never reinterprets elements as words
        // or deinitializes output slots, so the initialized view stays valid.
        unsafe {
            transpose_parallel::<false, T>(
                &input,
                super::super::rectangular::initialized_output(&mut output),
                width,
                height,
            );
        };
        assert_eq!(output, reference(&input, width, height, zero));

        // These shapes are below the parallel threshold, so this also compares
        // the parallel decomposition with the existing serial tiled engine.
        let mut serial = vec![zero; input.len()];
        transpose(&input, &mut serial, width, height);
        assert_eq!(output, serial);
    }

    #[cfg(feature = "parallel")]
    #[test]
    fn portable_parallel_band_boundaries() {
        for threads in [1, 2, 4, 16, 32] {
            rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap()
                .install(|| {
                    for width in [1, 2, 3, 4, 8, 16, 31, 32, 256] {
                        for height in [1, 15, 16, 17, 63, 64, 65, 127, 128, 129, 257] {
                            for (w, h) in [(width, height), (height, width)] {
                                check_parallel(w, h, 0u32, |i| i as u32);
                                check_parallel(w, h, 0u64, |i| i as u64);
                                check_parallel(w, h, 0u128, |i| i as u128);
                                check_parallel(w, h, [0u64; 4], |i| [i as u64; 4]);
                            }
                        }
                    }
                });
        }
    }

    #[cfg(feature = "parallel")]
    #[test]
    fn portable_parallel_preserves_generic_copy() {
        #[derive(Clone, Copy, Debug, PartialEq)]
        #[repr(C)]
        struct Padded<'a> {
            value: u8,
            pointer: &'a u64,
        }
        let values = [11u64, 23, 47];
        for threads in [1, 2, 4, 16, 32] {
            rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap()
                .install(|| {
                    for (w, h) in [(1, 129), (2, 257), (3, 257), (8, 129), (17, 65), (65, 17)] {
                        check_parallel(w, h, [0u8; 3], |i| {
                            [i as u8, (i >> 8) as u8, (i >> 16) as u8]
                        });
                        check_parallel(w, h, &values[0], |i| &values[i % values.len()]);
                        check_parallel(
                            w,
                            h,
                            Padded {
                                value: 0,
                                pointer: &values[0],
                            },
                            |i| Padded {
                                value: i as u8,
                                pointer: &values[i % values.len()],
                            },
                        );
                        check_parallel(w, h, (), |_| ());
                    }
                });
        }
    }

    #[cfg(feature = "parallel")]
    #[test]
    fn portable_parallel_threshold_boundaries() {
        rayon::ThreadPoolBuilder::new()
            .num_threads(32)
            .build()
            .unwrap()
            .install(|| {
                for width in [1, 2, 4, 8, 16, 32, 256] {
                    let height = PARALLEL_THRESHOLD / width;
                    // Immediately below, at, and above the public dispatch threshold.
                    for height in [height - 1, height, height + 1] {
                        check(width, height, 0u32, |i| i as u32);
                    }
                }
            });
    }

    #[cfg(feature = "parallel")]
    #[test]
    fn portable_parallel_path() {
        // Invariant: a column-band split must agree with the sequential transpose.
        //
        // Each shape has >= PARALLEL_THRESHOLD elements, so the parallel path runs.
        // The renumbered output columns are the part unique to this path.
        //
        //     shape          : exercises
        //     2050 x 2050    : square, uneven band split
        //     4097 x 1025    : wide, remainder columns
        //     1025 x 4097    : tall
        //     3    x 1.4e6   : narrower than the worker count -> one column per band
        for &(w, h) in &[(2050, 2050), (4097, 1025), (1025, 4097), (3, 1_400_001)] {
            check(w, h, 0u32, |i| i as u32);
        }
    }
}
