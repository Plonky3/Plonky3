//! Transposes of blocks given as separate rows, through register tiles.
//!
//! A packed hasher reads one column of `W` rows as one vector.
//!
//! Rows and digests are stored one message at a time.
//!
//! Moving between the two layouts is a transpose:
//!
//! ```text
//!     rows (one message each)          columns (one vector each)
//!
//!     r_0: [ a_0  a_1  a_2 ]           c_0: [ a_0  b_0  c_0  d_0 ]
//!     r_1: [ b_0  b_1  b_2 ]    -->    c_1: [ a_1  b_1  c_1  d_1 ]
//!     r_2: [ c_0  c_1  c_2 ]           c_2: [ a_2  b_2  c_2  d_2 ]
//!     r_3: [ d_0  d_1  d_2 ]
//! ```
//!
//! A lane-by-lane gather costs one scalar load and one scalar insert per element.
//!
//! A register transpose moves a whole square tile with `log_2(L)` shuffle stages instead.
//!
//! The vector kernels cover 4-byte elements, the size of every 31-bit field.
//!
//! Every other element size, and every target without a kernel, takes the scalar reference.

#[cfg(any(
    all(target_arch = "x86_64", target_feature = "avx2"),
    all(target_arch = "aarch64", target_feature = "neon"),
))]
use core::slice;

/// Transpose a block of rows into a flat column-major buffer.
///
/// ```text
///     dst[c * rows + r] = row(r)[c]        for r < rows, c < cols
/// ```
///
/// The rows need not be adjacent in memory, so a caller can gather them from anywhere.
///
/// # Arguments
///
/// - `rows`: number of input rows.
/// - `cols`: elements read from each row; a longer row keeps its tail unread.
/// - `row`: the input row at a given index.
/// - `dst`: the output, one run of `rows` elements per input column.
///
/// # Panics
///
/// - Panics if the output does not hold exactly `rows * cols` elements.
/// - Panics if a row is shorter than the column count.
///
/// # Examples
///
/// ```
/// use p3_util::transpose::transpose_rows;
///
/// let rows = [[1u32, 2, 3], [4, 5, 6]];
/// let mut columns = [0u32; 6];
/// transpose_rows(2, 3, |r| &rows[r], &mut columns);
/// assert_eq!(columns, [1, 4, 2, 5, 3, 6]);
/// ```
#[inline]
pub fn transpose_rows<'a, T, R>(rows: usize, cols: usize, row: R, dst: &mut [T])
where
    T: Copy + 'a,
    R: Fn(usize) -> &'a [T],
{
    // The output shape is fixed by the input shape.
    assert_eq!(
        dst.len(),
        rows * cols,
        "transpose output has the wrong length"
    );
    transpose_rows_strided(rows, cols, row, dst, rows);
}

/// Transpose a block of rows into a strided column-major destination.
///
/// ```text
///     dst[c * stride + r] = row(r)[c]      for r < rows, c < cols
/// ```
///
/// Slots between the output runs are left untouched, so the block can land inside a larger matrix.
///
/// # Panics
///
/// - Panics if the stride is below the row count.
/// - Panics if the output ends before the last written slot.
/// - Panics if a row is shorter than the column count.
#[inline]
pub(crate) fn transpose_rows_strided<'a, T, R>(
    rows: usize,
    cols: usize,
    row: R,
    dst: &mut [T],
    stride: usize,
) where
    T: Copy + 'a,
    R: Fn(usize) -> &'a [T],
{
    // Output runs must not overlap, and the last run must fit.
    assert!(stride >= rows, "transpose stride is below the row count");
    if rows == 0 || cols == 0 {
        return;
    }
    assert!(
        dst.len() >= (cols - 1) * stride + rows,
        "transpose output is too short for its stride"
    );

    // Only 4-byte, 4-byte-aligned elements can be moved as 32-bit vector lanes.
    //
    // Both checks are on constants, so the untaken branch folds away per element type.
    #[cfg(any(
        all(target_arch = "x86_64", target_feature = "avx2"),
        all(target_arch = "aarch64", target_feature = "neon"),
    ))]
    if size_of::<T>() == 4 && align_of::<T>() == 4 {
        // SAFETY: `T` has the size and alignment of `u32`.
        //
        // So every row and the output reinterpret as 32-bit words over the same memory.
        //
        // The kernel only copies whole elements to whole elements.
        //
        // So every value it writes is a bitwise copy of a valid `T`.
        let dst = unsafe { slice::from_raw_parts_mut(dst.as_mut_ptr().cast(), dst.len()) };
        let row = |r: usize| -> &'a [u32] {
            let src: &'a [T] = row(r);
            unsafe { slice::from_raw_parts(src.as_ptr().cast::<u32>(), src.len()) }
        };
        simd::transpose_u32(rows, cols, row, dst, stride);
        return;
    }

    transpose_rows_scalar(rows, cols, row, dst, stride);
}

/// Scalar reference transpose, with the contract of the strided entry point.
///
/// Every vector kernel is tested against this one.
#[inline]
pub(crate) fn transpose_rows_scalar<'a, T, R>(
    rows: usize,
    cols: usize,
    row: R,
    dst: &mut [T],
    stride: usize,
) where
    T: Copy + 'a,
    R: Fn(usize) -> &'a [T],
{
    for r in 0..rows {
        // Slicing to the column count rejects a short row and drops a long row's tail.
        for (c, &x) in row(r)[..cols].iter().enumerate() {
            // Column c of the input becomes output run c.
            dst[c * stride + r] = x;
        }
    }
}

/// Visit the output of a tiled transpose one `L x L` tile at a time.
///
/// Each call receives the tile origin and its extent, clipped at the right and bottom edges:
///
/// ```text
///     rows = 20, cols = 20, L = 16
///
///               c = 0..16      c = 16..20
///     r = 0..16   16 x 16        16 x 4
///     r = 16..20   4 x 16         4 x 4
/// ```
#[cfg(any(
    all(target_arch = "x86_64", target_feature = "avx2"),
    all(target_arch = "aarch64", target_feature = "neon"),
))]
#[inline(always)]
fn for_each_tile<const L: usize>(
    rows: usize,
    cols: usize,
    mut tile: impl FnMut(usize, usize, usize, usize),
) {
    for r0 in (0..rows).step_by(L) {
        for c0 in (0..cols).step_by(L) {
            tile(r0, c0, L.min(rows - r0), L.min(cols - c0));
        }
    }
}

/// AVX-512 kernel: one 512-bit register holds one 16-element row of a tile.
#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
mod simd {
    use core::arch::x86_64::*;
    use core::array;

    use super::for_each_tile;

    /// Elements per register, and the widest tile side.
    const L: usize = 16;

    /// Transpose 32-bit words, with the contract of the strided entry point.
    ///
    /// Clipped tiles use masked loads and stores.
    ///
    /// A masked-off lane is never read or written, so no tile touches memory past a row end.
    #[inline]
    pub(super) fn transpose_u32<'a>(
        rows: usize,
        cols: usize,
        row: impl Fn(usize) -> &'a [u32],
        dst: &mut [u32],
        stride: usize,
    ) {
        // Check every row once, so the tile loads can skip per-tile bounds checks.
        for r in 0..rows {
            assert!(
                row(r).len() >= cols,
                "transpose row is shorter than the column count"
            );
        }

        for_each_tile::<L>(rows, cols, |r0, c0, h, w| {
            // A short tile needs fewer stages: eight rows take three, sixteen take four.
            //
            // SAFETY: every row holds `cols` words and the output holds every strided run, as checked.
            unsafe {
                if h <= L / 2 {
                    tile::<{ L / 2 }>(stride, &row, dst, r0, c0, h, w);
                } else {
                    tile::<L>(stride, &row, dst, r0, c0, h, w);
                }
            }
        });
    }

    /// Transpose one `h x w` tile, held as `R` registers of 16 lanes.
    ///
    /// Only the stages below `R` run, so lane bit `R` never moves:
    ///
    /// ```text
    ///     R = 16:  register i = output column c0 + i
    ///     R = 8:   register i, lanes 0..8  = output column c0 + i
    ///              register i, lanes 8..16 = output column c0 + 8 + i
    /// ```
    ///
    /// Each stage swaps one bit of the row index with the same bit of the lane index.
    ///
    /// # Safety
    ///
    /// - Rows `r0 .. r0 + h` hold at least `c0 + w` words.
    /// - The output holds every slot `(c0 + j) * stride + r0 + i` for `j < w` and `i < h`.
    #[inline(always)]
    unsafe fn tile<'a, const R: usize>(
        stride: usize,
        row: &impl Fn(usize) -> &'a [u32],
        dst: &mut [u32],
        r0: usize,
        c0: usize,
        h: usize,
        w: usize,
    ) {
        // A full tile moves with plain loads and stores.
        //
        // Masked moves cost more even with every lane on, on some cores.
        let full = h == R && w == L;

        // Phase 1: one register per input row.
        //
        // A row past the tile height loads nothing, through an all-off mask.
        let load_mask = lane_mask(w);
        let mut t: [__m512i; R] = array::from_fn(|i| {
            let (r, mask) = if i < h { (r0 + i, load_mask) } else { (r0, 0) };
            let src = row(r).as_ptr().wrapping_add(c0);

            // SAFETY: a full tile reads 16 words from column c0, inside the row.
            //
            // A clipped tile reads at most w words through its mask, also inside the row.
            unsafe {
                if full {
                    _mm512_loadu_si512(src.cast())
                } else {
                    _mm512_maskz_loadu_epi32(mask, src.cast())
                }
            }
        });

        // Phase 2: block-swap stages, largest block first.
        if R > 8 {
            stage::<8, R>(&mut t);
        }
        stage::<4, R>(&mut t);
        stage::<2, R>(&mut t);
        stage::<1, R>(&mut t);

        // Phase 3: write each output column's h words.
        //
        //     column j = b * R + i  <-  lanes b*R .. b*R + h of register i
        let base = dst.as_mut_ptr();
        let column = |j: usize| base.wrapping_add((c0 + j) * stride + r0);
        if full {
            for (i, reg) in t.iter().enumerate() {
                // SAFETY: column i and, for eight rows, column 8 + i hold R slots inside dst.
                unsafe {
                    if R == L {
                        _mm512_storeu_si512(column(i).cast(), *reg);
                    } else {
                        _mm256_storeu_si256(column(i).cast(), _mm512_castsi512_si256(*reg));
                        _mm256_storeu_si256(
                            column(R + i).cast(),
                            _mm512_extracti64x4_epi64::<1>(*reg),
                        );
                    }
                }
            }
            return;
        }

        // A clipped tile rebases each store by b * R lanes, so that lane b * R lands on the column.
        let store_mask = lane_mask(h);
        for b in 0..L / R {
            for (i, reg) in t.iter().enumerate() {
                let j = b * R + i;
                if j < w {
                    let out = column(j).wrapping_sub(b * R);

                    // SAFETY: the shifted mask writes words out + b*R .. out + b*R + h.
                    //
                    // Those are exactly the h slots of column j inside the output.
                    unsafe {
                        _mm512_mask_storeu_epi32(out.cast(), store_mask << (b * R), *reg);
                    }
                }
            }
        }
    }

    /// Mask with the low `n` of the 16 lanes set.
    #[inline(always)]
    const fn lane_mask(n: usize) -> __mmask16 {
        // Shifting a 32-bit one keeps n = 16 defined.
        ((1u32 << n) - 1) as __mmask16
    }

    /// Two-source permute indices for one block-swap stage.
    ///
    /// Indices 0..16 select from the low row, 16..32 from the high row:
    ///
    /// ```text
    ///     lane bit d clear:   low' = low[lane]          high' = low[lane + d]
    ///     lane bit d set:     low' = high[lane - d]     high' = high[lane]
    /// ```
    const fn stage_indices(d: usize, high: bool) -> [i32; L] {
        let mut idx = [0; L];
        let mut lane = 0;
        while lane < L {
            idx[lane] = match (lane & d == 0, high) {
                (true, false) => lane,
                (false, false) => L + lane - d,
                (true, true) => lane + d,
                (false, true) => L + lane,
            } as i32;
            lane += 1;
        }
        idx
    }

    /// One block-swap stage: pair row `i` with row `i + d` for every `i` with bit `d` clear.
    ///
    /// The loop bounds are constants, so the stage unrolls into straight-line permutes.
    #[inline(always)]
    fn stage<const D: usize, const R: usize>(t: &mut [__m512i; R]) {
        let low = const { stage_indices(D, false) };
        let high = const { stage_indices(D, true) };

        // SAFETY: AVX-512F is enabled at compile time for this module.
        //
        // Each index array is 16 words, exactly one 512-bit load.
        unsafe {
            let low = _mm512_loadu_si512(low.as_ptr().cast());
            let high = _mm512_loadu_si512(high.as_ptr().cast());
            for i in 0..R {
                if i & D == 0 {
                    let (a, b) = (t[i], t[i + D]);
                    t[i] = _mm512_permutex2var_epi32(a, low, b);
                    t[i + D] = _mm512_permutex2var_epi32(a, high, b);
                }
            }
        }
    }
}

/// AVX2 kernel: one 256-bit register holds one 8-element row of a tile.
#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx2",
    not(target_feature = "avx512f")
))]
mod simd {
    use core::arch::x86_64::*;
    use core::array;

    use super::for_each_tile;

    /// Elements per register, and the side of one square tile.
    const L: usize = 8;

    /// Transpose 32-bit words, with the contract of the strided entry point.
    ///
    /// Full tiles go through registers.
    ///
    /// Clipped edge tiles copy element by element.
    ///
    /// AVX2 masked moves cost more than the few elements they cover.
    #[inline]
    pub(super) fn transpose_u32<'a>(
        rows: usize,
        cols: usize,
        row: impl Fn(usize) -> &'a [u32],
        dst: &mut [u32],
        stride: usize,
    ) {
        for_each_tile::<L>(rows, cols, |r0, c0, h, w| {
            if h < L || w < L {
                // Edge tile: copy element by element with the reference indexing.
                for i in 0..h {
                    for (j, &x) in row(r0 + i)[c0..c0 + w].iter().enumerate() {
                        dst[(c0 + j) * stride + r0 + i] = x;
                    }
                }
                return;
            }

            // Phase 1: one register per input row.
            let mut tile: [__m256i; L] = array::from_fn(|i| {
                let src = &row(r0 + i)[c0..c0 + L];

                // SAFETY: the slice holds exactly the eight words loaded.
                unsafe { _mm256_loadu_si256(src.as_ptr().cast()) }
            });

            // Phase 2: transpose inside the registers.
            transpose_8x8(&mut tile);

            // Phase 3: register j holds input column c0 + j across the tile rows.
            for (j, reg) in tile.iter().enumerate() {
                let start = (c0 + j) * stride + r0;
                let out = &mut dst[start..start + L];

                // SAFETY: the slice holds exactly the eight words stored.
                unsafe { _mm256_storeu_si256(out.as_mut_ptr().cast(), *reg) };
            }
        });
    }

    /// Transpose an 8 x 8 tile of 32-bit words held one row per register.
    ///
    /// Three block-swap stages, one per block size, each with the cheapest instruction pair:
    ///
    /// ```text
    ///     d = 4   128-bit halves     permute2x128
    ///     d = 2   64-bit pairs       unpacklo / unpackhi on 64-bit lanes
    ///     d = 1   32-bit words       shift by 32 bits, then blend
    /// ```
    #[inline(always)]
    fn transpose_8x8(t: &mut [__m256i; L]) {
        // SAFETY: AVX2 is enabled at compile time for this module.
        unsafe {
            // Stage d = 4: swap the upper half of row i with the lower half of row i + 4.
            for i in 0..4 {
                let (a, b) = (t[i], t[i + 4]);
                t[i] = _mm256_permute2x128_si256::<0x20>(a, b);
                t[i + 4] = _mm256_permute2x128_si256::<0x31>(a, b);
            }

            // Stage d = 2: rows (0, 2), (1, 3), (4, 6), (5, 7).
            for i in [0, 1, 4, 5] {
                let (a, b) = (t[i], t[i + 2]);
                t[i] = _mm256_unpacklo_epi64(a, b);
                t[i + 2] = _mm256_unpackhi_epi64(a, b);
            }

            // Stage d = 1: rows (0, 1), (2, 3), (4, 5), (6, 7).
            //
            //     a' = [a_0 b_0 a_2 b_2 ...]   b shifted up one word, odd words kept
            //     b' = [a_1 b_1 a_3 b_3 ...]   a shifted down one word, odd words from b
            for i in [0, 2, 4, 6] {
                let (a, b) = (t[i], t[i + 1]);
                t[i] = _mm256_blend_epi32::<0b1010_1010>(a, _mm256_slli_epi64::<32>(b));
                t[i + 1] = _mm256_blend_epi32::<0b1010_1010>(_mm256_srli_epi64::<32>(a), b);
            }
        }
    }
}

/// NEON kernel: one 128-bit register holds one 4-element row of a tile.
#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
mod simd {
    use core::arch::aarch64::*;
    use core::array;

    use super::for_each_tile;

    /// Elements per register, and the side of one square tile.
    const L: usize = 4;

    /// Transpose 32-bit words, with the contract of the strided entry point.
    ///
    /// Full tiles go through registers.
    ///
    /// Clipped edge tiles copy element by element, since NEON has no masked moves.
    #[inline]
    pub(super) fn transpose_u32<'a>(
        rows: usize,
        cols: usize,
        row: impl Fn(usize) -> &'a [u32],
        dst: &mut [u32],
        stride: usize,
    ) {
        for_each_tile::<L>(rows, cols, |r0, c0, h, w| {
            if h < L || w < L {
                // Edge tile: copy element by element with the reference indexing.
                for i in 0..h {
                    for (j, &x) in row(r0 + i)[c0..c0 + w].iter().enumerate() {
                        dst[(c0 + j) * stride + r0 + i] = x;
                    }
                }
                return;
            }

            // Phase 1: one register per input row.
            let q: [uint32x4_t; L] = array::from_fn(|i| {
                let src = &row(r0 + i)[c0..c0 + L];

                // SAFETY: the slice holds exactly the four words loaded.
                unsafe { vld1q_u32(src.as_ptr()) }
            });

            // Phase 2: two block-swap stages.
            //
            //     d = 1   TRN1 / TRN2 on 32-bit words
            //     d = 2   TRN1 / TRN2 on 64-bit pairs
            //
            // SAFETY: NEON is enabled at compile time for this module.
            let r = unsafe {
                let t0 = vtrn1q_u32(q[0], q[1]);
                let t1 = vtrn2q_u32(q[0], q[1]);
                let t2 = vtrn1q_u32(q[2], q[3]);
                let t3 = vtrn2q_u32(q[2], q[3]);
                let pair = |a: uint32x4_t, b: uint32x4_t, high: bool| {
                    let (a, b) = (vreinterpretq_u64_u32(a), vreinterpretq_u64_u32(b));
                    vreinterpretq_u32_u64(if high {
                        vtrn2q_u64(a, b)
                    } else {
                        vtrn1q_u64(a, b)
                    })
                };
                [
                    pair(t0, t2, false),
                    pair(t1, t3, false),
                    pair(t0, t2, true),
                    pair(t1, t3, true),
                ]
            };

            // Phase 3: register j holds input column c0 + j across the tile rows.
            for (j, reg) in r.iter().enumerate() {
                let start = (c0 + j) * stride + r0;
                let out = &mut dst[start..start + L];

                // SAFETY: the slice holds exactly the four words stored.
                unsafe { vst1q_u32(out.as_mut_ptr(), *reg) };
            }
        });
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec;
    use alloc::vec::Vec;

    use p3_baby_bear::BabyBear;
    use p3_field::PrimeCharacteristicRing;
    use proptest::prelude::*;

    use super::*;

    /// A distinct word per position, so any misplaced element shows up.
    fn grid(rows: usize, row_len: usize) -> Vec<Vec<u32>> {
        (0..rows)
            .map(|r| (0..row_len).map(|c| (r * 1000 + c) as u32).collect())
            .collect()
    }

    #[test]
    fn empty_shapes_write_nothing() {
        // Zero rows or zero columns produce an empty output and read nothing.
        let rows: [&[u32]; 0] = [];
        transpose_rows(0, 5, |r| rows[r], &mut []);
        let one = [7u32; 3];
        transpose_rows(1, 0, |_| &one[..], &mut []);
    }

    #[test]
    fn long_rows_are_read_only_up_to_the_column_count() {
        // Fixture state: rows of 20 words, transposed as 17 columns.
        //
        // A 16-wide tile then a 1-wide tile, and three trailing words per row left unread.
        let src = grid(16, 20);
        let mut fast = vec![0u32; 16 * 17];
        let mut reference = vec![0u32; 16 * 17];
        transpose_rows(16, 17, |r| &src[r], &mut fast);
        transpose_rows_scalar(16, 17, |r| &src[r], &mut reference, 16);
        assert_eq!(fast, reference);
    }

    #[test]
    fn a_strided_block_leaves_the_gaps_untouched() {
        // Fixture state: a 3 x 2 block written with stride 5 into a buffer of sentinels.
        //
        //     run 0: slots 0..3     run 1: slots 5..8     slots 3, 4 and 8.. keep 9
        let src = [[1u32, 2], [3, 4], [5, 6]];
        let mut dst = [9u32; 10];
        transpose_rows_strided(3, 2, |r| &src[r], &mut dst, 5);
        assert_eq!(dst, [1, 3, 5, 9, 9, 2, 4, 6, 9, 9]);
    }

    #[test]
    fn field_elements_keep_their_values() {
        // A 31-bit field element takes the 32-bit kernel through the reinterpretation.
        //
        // Its values must come out unchanged, only moved.
        let src: Vec<Vec<BabyBear>> = (0..16)
            .map(|r| (0..9).map(|c| BabyBear::from_usize(r * 9 + c)).collect())
            .collect();
        let mut dst = vec![BabyBear::ZERO; 16 * 9];
        transpose_rows(16, 9, |r| &src[r], &mut dst);
        for (r, row) in src.iter().enumerate() {
            for (c, &x) in row.iter().enumerate() {
                assert_eq!(dst[c * 16 + r], x);
            }
        }
    }

    #[test]
    fn eight_byte_elements_take_the_reference() {
        // No kernel covers 8-byte words, so the dispatch must still produce the transpose.
        let rows = [[1u64, 2], [3, 4], [5, 6]];
        let mut dst = [0u64; 6];
        transpose_rows(3, 2, |r| &rows[r], &mut dst);
        assert_eq!(dst, [1, 3, 5, 2, 4, 6]);
    }

    #[test]
    #[should_panic(expected = "wrong length")]
    fn rejects_a_short_output() {
        // 2 x 2 needs four slots; three is one short.
        let rows = [[1u32, 2], [3, 4]];
        transpose_rows(2, 2, |r| &rows[r], &mut [0u32; 3]);
    }

    #[test]
    #[should_panic]
    fn rejects_a_short_row() {
        // The second row holds one word where two columns are requested.
        let rows: [&[u32]; 2] = [&[1, 2], &[3]];
        transpose_rows(2, 2, |r| rows[r], &mut [0u32; 4]);
    }

    #[test]
    #[should_panic(expected = "below the row count")]
    fn rejects_overlapping_runs() {
        // Stride 1 for 2 rows would write two runs over each other.
        let rows = [[1u32, 2], [3, 4]];
        transpose_rows_strided(2, 2, |r| &rows[r], &mut [0u32; 4], 1);
    }

    proptest! {
        #[test]
        fn vector_kernel_matches_reference(
            rows in 0usize..40,
            cols in 0usize..40,
            slack in 0usize..3,
            gap in 0usize..3,
            seed in any::<u32>(),
        ) {
            // Shapes on both sides of every tile width (4, 8, 16), including clipped edges.
            //
            // Rows carry up to two extra words that must never reach the output.
            //
            // A stride past the row count leaves gaps that must keep their sentinel.
            let src: Vec<Vec<u32>> = (0..rows)
                .map(|r| {
                    (0..cols + slack)
                        .map(|c| seed.wrapping_mul(r as u32 + 1).wrapping_add(c as u32 * 7919))
                        .collect()
                })
                .collect();
            let stride = rows + gap;

            let mut fast = vec![u32::MAX; stride * cols];
            let mut reference = vec![u32::MAX; stride * cols];
            transpose_rows_strided(rows, cols, |r| &src[r], &mut fast, stride);
            transpose_rows_scalar(rows, cols, |r| &src[r], &mut reference, stride);
            prop_assert_eq!(fast, reference);
        }
    }
}
