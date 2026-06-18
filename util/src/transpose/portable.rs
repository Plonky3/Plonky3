//! Portable cache-oblivious matrix transpose, the fallback when no SIMD kernel applies.
//!
//! Maps `input[y * width + x]` to `output[x * height + y]` for any `Copy` type.

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

/// Minimum element count before the parallel column-band path is used.
///
/// Smaller matrices are dominated by thread fan-out and cache-coherency traffic.
#[cfg(feature = "parallel")]
const PARALLEL_THRESHOLD: usize = 4 << 20;

/// Transpose a row-major matrix from `input` into `output`.
///
/// - `input` is read as `height` rows of `width` columns.
/// - `output` receives the transpose as `width` rows of `height` columns.
///
/// # Panics
///
/// Panics if either slice length differs from `width * height`.
#[inline]
pub(crate) fn transpose<T: Copy + Send + Sync>(
    input: &[T],
    output: &mut [T],
    width: usize,
    height: usize,
) {
    // Both buffers must describe the same `height x width` matrix.
    assert_eq!(
        input.len(),
        width * height,
        "input length {} != width*height {}",
        input.len(),
        width * height
    );
    assert_eq!(
        output.len(),
        width * height,
        "output length {} != width*height {}",
        output.len(),
        width * height
    );

    // A zero-size matrix has nothing to move.
    if width == 0 || height == 0 {
        return;
    }

    let len = width * height;

    // Large matrices: hand off to the parallel column-band split.
    #[cfg(feature = "parallel")]
    if len >= PARALLEL_THRESHOLD {
        transpose_parallel(input, output, width, height);
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
            transpose_region::<false, T>(input, output, 0, height, 0, width, width, height, 0);
        } else {
            // Large: recurse down to cache-resident leaves, then tile.
            transpose_recursive(input, output, 0, height, 0, width, width, height, 0);
        }
    }
}

/// Element-by-element transpose for matrices that fit in L1.
///
/// - Iterating the source row index innermost keeps the writes contiguous.
/// - The strided reads are cheap while the whole matrix is L1-resident.
#[inline]
fn transpose_small<T: Copy>(input: &[T], output: &mut [T], width: usize, height: usize) {
    // Walk output rows (source columns) one at a time.
    for x in 0..width {
        // Walk down this output row, filling the contiguous run `[x*height ..][..height]`.
        for y in 0..height {
            // SAFETY: `x < width` and `y < height`, so both indices are `< width*height`.
            unsafe {
                // Source element (row y, col x) -> output element (row x, col y).
                *output.get_unchecked_mut(x * height + y) = *input.get_unchecked(x + y * width);
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
    output: &mut [T],
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
                *output.get_unchecked_mut(out_base + inner_y) =
                    *input.get_unchecked(x + (start_y + inner_y) * total_cols);
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
    output: &mut [T],
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
                *output.get_unchecked_mut(out_base + inner_y) =
                    *input.get_unchecked(x + (start_y + inner_y) * total_cols);
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
    output: &mut [T],
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
                    *output.get_unchecked_mut(out_base + inner_y) =
                        *input.get_unchecked(x + (seg_y + inner_y) * total_cols);
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
/// `SEGMENTED` selects the whole-tile kernel:
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
/// - `input.len() == total_cols * total_rows`.
/// - `row_end <= total_rows` and `col_end <= total_cols`.
/// - `output` covers output rows `[col_start - col_out_base, col_end - col_out_base)`.
#[inline]
#[allow(clippy::too_many_arguments)]
unsafe fn transpose_region<const SEGMENTED: bool, T: Copy>(
    input: &[T],
    output: &mut [T],
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
                    transpose_tile_plain(input, output, total_cols, total_rows, col, row, out_col);
                }
            }

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
unsafe fn transpose_recursive<T: Copy>(
    input: &[T],
    output: &mut [T],
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
            transpose_region::<true, T>(
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
            transpose_recursive(
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
            transpose_recursive(
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
            transpose_recursive(
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
            transpose_recursive(
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

/// Parallel transpose: one rayon task per disjoint output column-band.
///
/// Output rows are cut into contiguous bands of whole source columns.
/// Band `k` holds the transpose of source columns starting at `k * cols_per_task`.
/// Bands never overlap, so each task gets an exclusive slice — no shared pointers.
///
/// ```text
///     source columns   ->   output rows (one contiguous band)
///     [0, c)           ->   output[0 .. c*height)
///     [c, 2c)          ->   output[c*height .. 2c*height)
///     ...
/// ```
#[cfg(feature = "parallel")]
fn transpose_parallel<T: Copy + Send + Sync>(
    input: &[T],
    output: &mut [T],
    width: usize,
    height: usize,
) {
    use p3_maybe_rayon::prelude::*;

    // One band per worker; round up so the final band is the only short one.
    let num_threads = current_num_threads();
    let cols_per_task = width.div_ceil(num_threads).max(1);

    // Each band is `cols_per_task` output rows of `height` columns.
    output
        .par_chunks_mut(cols_per_task * height)
        .enumerate()
        .for_each(|(task, band)| {
            // Absolute source columns this band owns.
            let col_start = task * cols_per_task;
            let col_end = (col_start + cols_per_task).min(width);

            // SAFETY:
            // - `band` is exactly `(col_end - col_start) * height` long (only the last is shorter).
            // - The engine writes output rows `[0, col_end - col_start)`, all inside `band`.
            // - Reads address the shared `input`, valid for `width * height` elements.
            unsafe {
                transpose_recursive(
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
