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
//! On x86 they also cover 8-byte elements, the size of a 64-bit field.
//!
//! Every other element size, and every target without a kernel, takes the scalar reference.

use core::mem::MaybeUninit;
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
/// Typed copies support arbitrary `Copy` representations.
/// Audited integer representations can use [`transpose_rows_words`] for register kernels.
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
/// - Panics if `rows * cols` overflows `usize`.
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
    let len = rows
        .checked_mul(cols)
        .expect("transpose dimensions overflow");
    assert_eq!(dst.len(), len, "transpose output has the wrong length");
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
/// - Panics if the strided output extent overflows `usize`.
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
    let extent = (cols - 1)
        .checked_mul(stride)
        .and_then(|last_run| last_run.checked_add(rows))
        .expect("transpose strided extent overflow");
    assert!(
        dst.len() >= extent,
        "transpose output is too short for its stride"
    );

    transpose_rows_scalar(rows, cols, row, dst, stride);
}

/// Transpose rows using integer-word register kernels where available.
///
/// # Safety
///
/// The input rows and destination elements must satisfy the representation contract of [`super::transpose_words`].
/// This applies to every slice returned by the callback, including when the callback changes its result.
#[inline]
pub unsafe fn transpose_rows_words<'a, T, R>(rows: usize, cols: usize, row: R, dst: &mut [T])
where
    T: Copy + 'a,
    R: Fn(usize) -> &'a [T],
{
    let len = rows
        .checked_mul(cols)
        .expect("transpose dimensions overflow");
    assert_eq!(dst.len(), len, "transpose output has the wrong length");
    // SAFETY: the caller supplies the representation guarantee.
    unsafe { transpose_rows_strided_words(rows, cols, row, dst, rows) };
}

/// Integer-word version of the strided transpose.
///
/// # Safety
///
/// All elements must satisfy [`super::transpose_words`]'s representation contract.
#[inline]
pub(crate) unsafe fn transpose_rows_strided_words<'a, T, R>(
    rows: usize,
    cols: usize,
    row: R,
    dst: &mut [T],
    stride: usize,
) where
    T: Copy + 'a,
    R: Fn(usize) -> &'a [T],
{
    // SAFETY: valid output values can be viewed as MaybeUninit; the engine
    // only writes valid copies and never deinitializes a slot, even on panic.
    unsafe {
        transpose_rows_strided_words_uninit(
            rows,
            cols,
            row,
            super::rectangular::initialized_output(dst),
            stride,
        );
    }
}

/// Word transpose into possibly uninitialized output runs; gaps are untouched.
///
/// # Safety
/// Input elements satisfy the word-copy contract. Each written slot receives a
/// valid T copied from input; no output slot is read or deinitialized.
#[inline]
pub(crate) unsafe fn transpose_rows_strided_words_uninit<'a, T, R>(
    rows: usize,
    cols: usize,
    row: R,
    dst: &mut [MaybeUninit<T>],
    stride: usize,
) where
    T: Copy + 'a,
    R: Fn(usize) -> &'a [T],
{
    assert!(stride >= rows, "transpose stride is below the row count");
    if rows == 0 || cols == 0 {
        return;
    }
    let extent = (cols - 1)
        .checked_mul(stride)
        .and_then(|last_run| last_run.checked_add(rows))
        .expect("transpose strided extent overflow");
    assert!(
        dst.len() >= extent,
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
        // Rows reinterpret as initialized words; output remains possibly uninitialized
        // word slots over the same memory.
        //
        // The caller guarantees initialized, padding-free, pointer-free inputs.
        //
        // So every word the kernel loads is an initialized integer.
        //
        // The kernel only copies whole elements to whole elements.
        //
        // So every value it writes is a bitwise copy of a valid `T`.
        let dst = unsafe {
            slice::from_raw_parts_mut(dst.as_mut_ptr().cast::<MaybeUninit<u32>>(), dst.len())
        };
        let row = |r: usize| -> &'a [u32] {
            let src: &'a [T] = row(r);
            unsafe { slice::from_raw_parts(src.as_ptr().cast::<u32>(), src.len()) }
        };
        simd::transpose_u32(rows, cols, row, dst, stride);
        return;
    }

    // The same argument, for 8-byte, 8-byte-aligned elements and 64-bit lanes.
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    if size_of::<T>() == 8 && align_of::<T>() == 8 {
        // SAFETY: as above, with `u64` in place of `u32`.
        let dst = unsafe {
            slice::from_raw_parts_mut(dst.as_mut_ptr().cast::<MaybeUninit<u64>>(), dst.len())
        };
        let row = |r: usize| -> &'a [u64] {
            let src: &'a [T] = row(r);
            unsafe { slice::from_raw_parts(src.as_ptr().cast::<u64>(), src.len()) }
        };
        simd::transpose_u64(rows, cols, row, dst, stride);
        return;
    }

    for r in 0..rows {
        for (c, &value) in row(r)[..cols].iter().enumerate() {
            dst[c * stride + r].write(value);
        }
    }
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

/// AVX-512 kernels: one 512-bit register holds one row of a tile.
#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
mod simd {
    use core::arch::x86_64::*;
    use core::array;

    use super::{MaybeUninit, for_each_tile};

    /// 32-bit elements per register, and the widest 32-bit tile side.
    const L: usize = 16;

    /// 64-bit elements per register, and the side of a 64-bit tile.
    const L64: usize = 8;

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
        dst: &mut [MaybeUninit<u32>],
        stride: usize,
    ) {
        for_each_tile::<L>(rows, cols, |r0, c0, h, w| {
            // A short tile needs fewer stages: eight rows take three, sixteen take four.
            //
            // SAFETY: each load checks its actual callback result below, and
            // the strided entry point checked the output extent.
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
    /// - `R` is 8 or 16, `h <= R`, and `w <= L`.
    /// - The output holds every slot `(c0 + j) * stride + r0 + i` for `j < w` and `i < h`.
    #[inline(always)]
    unsafe fn tile<'a, const R: usize>(
        stride: usize,
        row: &impl Fn(usize) -> &'a [u32],
        dst: &mut [MaybeUninit<u32>],
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
        // Rows past the tile height become zero registers without calling `row`.
        let load_mask = lane_mask(w);
        let mut t: [__m512i; R] = array::from_fn(|i| {
            if i >= h {
                // No source row exists for this masked-off register.
                // SAFETY: this module is compiled only with AVX-512F enabled.
                return unsafe { _mm512_setzero_si512() };
            }
            // A callback can change its result on every call. Validate the
            // exact slice used by this load, rather than an earlier result.
            let src = &row(r0 + i)[c0..c0 + w];
            let src = src.as_ptr();

            // SAFETY: a full tile reads 16 words from column c0, inside the row.
            //
            // A clipped tile reads at most w words through its mask, also inside the row.
            unsafe {
                if full {
                    _mm512_loadu_si512(src.cast())
                } else {
                    _mm512_maskz_loadu_epi32(load_mask, src.cast())
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

    /// Transpose 64-bit words, with the contract of the strided entry point.
    ///
    /// Clipped tiles use masked loads and stores, exactly as for 32-bit words.
    #[inline]
    pub(super) fn transpose_u64<'a>(
        rows: usize,
        cols: usize,
        row: impl Fn(usize) -> &'a [u64],
        dst: &mut [MaybeUninit<u64>],
        stride: usize,
    ) {
        for_each_tile::<L64>(rows, cols, |r0, c0, h, w| {
            let full = h == L64 && w == L64;

            // Phase 1: one register per input row, zero past the tile height.
            let mask = lane_mask64(w);
            let mut t: [__m512i; L64] = array::from_fn(|i| {
                if i >= h {
                    // SAFETY: this module is compiled only with AVX-512F enabled.
                    return unsafe { _mm512_setzero_si512() };
                }
                // Validate the exact slice this load reads.
                let src = row(r0 + i)[c0..c0 + w].as_ptr();

                // SAFETY: the load reads at most w words, all inside the checked slice.
                unsafe {
                    if full {
                        _mm512_loadu_si512(src.cast())
                    } else {
                        _mm512_maskz_loadu_epi64(mask, src.cast())
                    }
                }
            });

            // Phase 2: transpose inside the registers.
            transpose_8x8(&mut t);

            // Phase 3: register j now holds input column c0 + j.
            let mask = lane_mask64(h);
            for (j, reg) in t.iter().enumerate().take(w) {
                let out = dst.as_mut_ptr().wrapping_add((c0 + j) * stride + r0);

                // SAFETY: the store writes at most h words of column j, inside the output.
                unsafe {
                    if full {
                        _mm512_storeu_si512(out.cast(), *reg);
                    } else {
                        _mm512_mask_storeu_epi64(out.cast(), mask, *reg);
                    }
                }
            }
        });
    }

    #[cfg(test)]
    #[test]
    fn full_tile_uses_each_callback_result_once() {
        use core::cell::Cell;

        let src = [7u32; L];
        let calls = Cell::new(0);
        let mut dst = [0u32; L * L];
        transpose_u32(
            L,
            L,
            |_| {
                let count = calls.get();
                // Fail before any unsafe load if a preflight check re-reads rows.
                assert!(count < L, "callback was called again after preflight");
                calls.set(count + 1);
                &src
            },
            // SAFETY: the kernel only writes valid input copies, keeping
            // every previously initialized slot valid even if row() panics.
            unsafe { super::super::rectangular::initialized_output(&mut dst) },
            L,
        );
        assert_eq!(dst, [7; L * L]);
        assert_eq!(calls.get(), L);
    }

    #[cfg(test)]
    #[test]
    #[should_panic]
    fn checks_the_actual_stateful_callback_result() {
        use core::cell::Cell;

        let src = [7u32; L * 2];
        let calls = Cell::new(0);
        let mut dst = [0u32; L * L * 2];
        transpose_u32(
            L,
            L * 2,
            |_| {
                let count = calls.get();
                calls.set(count + 1);
                if count < L { &src } else { &src[..0] }
            },
            // SAFETY: the kernel only writes valid input copies, keeping
            // every previously initialized slot valid even if row() panics.
            unsafe { super::super::rectangular::initialized_output(&mut dst) },
            L,
        );
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

    /// Mask with the low `n` of 8 lanes set.
    #[inline(always)]
    const fn lane_mask64(n: usize) -> __mmask8 {
        // Shifting a 32-bit one keeps n = 8 defined.
        ((1u32 << n) - 1) as __mmask8
    }

    /// Transpose an 8 x 8 tile of 64-bit words held one row per register.
    ///
    /// A register is four 128-bit blocks, so the tile is a 4 x 4 grid of 2 x 2 blocks.
    ///
    /// - Stage 1: 64-bit unpacks transpose every 2 x 2 block.
    /// - Stages 2 and 3: block shuffles transpose the 4 x 4 grid of blocks.
    #[inline(always)]
    fn transpose_8x8(t: &mut [__m512i; L64]) {
        // SAFETY: AVX-512F is enabled at compile time for this module.
        unsafe {
            // Stage 1: rows (g, g + 1) for every even g.
            //
            // Afterwards, block k of register g + c holds rows g, g + 1 of column 2k + c.
            for g in (0..L64).step_by(2) {
                let (a, b) = (t[g], t[g + 1]);
                t[g] = _mm512_unpacklo_epi64(a, b);
                t[g + 1] = _mm512_unpackhi_epi64(a, b);
            }
        }

        // Stages 2 and 3: for each c, gather block k of registers c, 2 + c, 4 + c, 6 + c.
        let s = *t;
        for c in 0..2 {
            let cols = transpose_blocks([s[c], s[2 + c], s[4 + c], s[6 + c]]);
            for (k, col) in cols.into_iter().enumerate() {
                t[2 * k + c] = col;
            }
        }
    }

    /// Transpose a 4 x 4 grid of 128-bit blocks held one grid row per register.
    ///
    /// Output register k collects block k of every input, in input order.
    ///
    /// ```text
    ///     shuffle 0x88 keeps blocks 0, 2 of each source
    ///     shuffle 0xdd keeps blocks 1, 3 of each source
    ///
    ///     x0 = [a_0 a_2 b_0 b_2]    x1 = [a_1 a_3 b_1 b_3]
    ///     y0 = [c_0 c_2 d_0 d_2]    y1 = [c_1 c_3 d_1 d_3]
    ///
    ///     out 0 = [a_0 b_0 c_0 d_0] = 0x88 (x0, y0)
    ///     out 1 = [a_1 b_1 c_1 d_1] = 0x88 (x1, y1)
    ///     out 2 = [a_2 b_2 c_2 d_2] = 0xdd (x0, y0)
    ///     out 3 = [a_3 b_3 c_3 d_3] = 0xdd (x1, y1)
    /// ```
    #[inline(always)]
    fn transpose_blocks([a, b, c, d]: [__m512i; 4]) -> [__m512i; 4] {
        // SAFETY: AVX-512F is enabled at compile time for this module.
        unsafe {
            let x0 = _mm512_shuffle_i64x2::<0x88>(a, b);
            let x1 = _mm512_shuffle_i64x2::<0xdd>(a, b);
            let y0 = _mm512_shuffle_i64x2::<0x88>(c, d);
            let y1 = _mm512_shuffle_i64x2::<0xdd>(c, d);
            [
                _mm512_shuffle_i64x2::<0x88>(x0, y0),
                _mm512_shuffle_i64x2::<0x88>(x1, y1),
                _mm512_shuffle_i64x2::<0xdd>(x0, y0),
                _mm512_shuffle_i64x2::<0xdd>(x1, y1),
            ]
        }
    }
}

/// AVX2 kernels: one 256-bit register holds one row of a tile.
#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx2",
    not(target_feature = "avx512f")
))]
mod simd {
    use core::arch::x86_64::*;
    use core::array;

    use super::{MaybeUninit, for_each_tile};

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
        dst: &mut [MaybeUninit<u32>],
        stride: usize,
    ) {
        for_each_tile::<L>(rows, cols, |r0, c0, h, w| {
            if h < L || w < L {
                // Edge tile: copy element by element with the reference indexing.
                for i in 0..h {
                    for (j, &x) in row(r0 + i)[c0..c0 + w].iter().enumerate() {
                        dst[(c0 + j) * stride + r0 + i].write(x);
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

    /// Transpose 64-bit words, with the contract of the strided entry point.
    ///
    /// Full 4 x 4 tiles go through registers.
    ///
    /// Clipped edge tiles copy element by element.
    #[inline]
    pub(super) fn transpose_u64<'a>(
        rows: usize,
        cols: usize,
        row: impl Fn(usize) -> &'a [u64],
        dst: &mut [MaybeUninit<u64>],
        stride: usize,
    ) {
        for_each_tile::<4>(rows, cols, |r0, c0, h, w| {
            if h < 4 || w < 4 {
                // Edge tile: copy element by element with the reference indexing.
                for i in 0..h {
                    for (j, &x) in row(r0 + i)[c0..c0 + w].iter().enumerate() {
                        dst[(c0 + j) * stride + r0 + i].write(x);
                    }
                }
                return;
            }

            // Phase 1: one register per input row.
            let [a, b, c, d]: [__m256i; 4] = array::from_fn(|i| {
                let src = &row(r0 + i)[c0..c0 + 4];

                // SAFETY: the slice holds exactly the four words loaded.
                unsafe { _mm256_loadu_si256(src.as_ptr().cast()) }
            });

            // Phase 2: unpacks transpose each 2 x 2 block, then 128-bit permutes swap blocks.
            //
            //     lo(a, b) = [a_0 b_0 | a_2 b_2]    hi(a, b) = [a_1 b_1 | a_3 b_3]
            //
            // SAFETY: AVX2 is enabled at compile time for this module.
            let cols = unsafe {
                let (ab0, ab1) = (_mm256_unpacklo_epi64(a, b), _mm256_unpackhi_epi64(a, b));
                let (cd0, cd1) = (_mm256_unpacklo_epi64(c, d), _mm256_unpackhi_epi64(c, d));
                [
                    _mm256_permute2x128_si256::<0x20>(ab0, cd0),
                    _mm256_permute2x128_si256::<0x20>(ab1, cd1),
                    _mm256_permute2x128_si256::<0x31>(ab0, cd0),
                    _mm256_permute2x128_si256::<0x31>(ab1, cd1),
                ]
            };

            // Phase 3: register j holds input column c0 + j across the tile rows.
            for (j, reg) in cols.iter().enumerate() {
                let start = (c0 + j) * stride + r0;
                let out = &mut dst[start..start + 4];

                // SAFETY: the slice holds exactly the four words stored.
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

    use super::{MaybeUninit, for_each_tile};

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
        dst: &mut [MaybeUninit<u32>],
        stride: usize,
    ) {
        for_each_tile::<L>(rows, cols, |r0, c0, h, w| {
            if h < L || w < L {
                // Edge tile: copy element by element with the reference indexing.
                for i in 0..h {
                    for (j, &x) in row(r0 + i)[c0..c0 + w].iter().enumerate() {
                        dst[(c0 + j) * stride + r0 + i].write(x);
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
                unsafe { vst1q_u32(out.as_mut_ptr().cast::<u32>(), *reg) };
            }
        });
    }
}

#[cfg(test)]
mod tests {
    extern crate std;

    use alloc::vec;
    use alloc::vec::Vec;

    use p3_baby_bear::BabyBear;
    use p3_field::{Packable, PrimeCharacteristicRing};
    use p3_goldilocks::Goldilocks;
    use proptest::prelude::*;

    use super::*;

    #[test]
    fn initialized_word_output_remains_valid_when_callback_panics() {
        use core::cell::Cell;
        let calls = Cell::new(0);
        let row = [7u32; 17];
        let mut output = [91u32; 16 * 17];
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            // SAFETY: both inputs and initially valid outputs are integer words.
            unsafe {
                transpose_rows_words(
                    16,
                    17,
                    |_| {
                        calls.set(calls.get() + 1);
                        if calls.get() == 6 {
                            panic!("interrupted callback");
                        }
                        &row
                    },
                    &mut output,
                );
            }
        }));
        assert!(result.is_err());
        // Reading every slot also catches deinitialization under Miri. Different
        // kernel tile sizes may have committed different subsets before panic.
        assert!(output.iter().all(|&value| value == 7 || value == 91));
    }

    #[test]
    #[should_panic(expected = "transpose dimensions overflow")]
    fn rejects_overflowing_word_dimensions_before_access() {
        // The unchecked product and strided extent would both wrap to 2.
        // SAFETY: u32 has an initialized, padding-free, pointer-free representation.
        unsafe {
            transpose_rows_words::<u32, _>(
                usize::MAX / 2 + 2,
                2,
                |_| panic!("must not read a row"),
                &mut [0; 2],
            );
        };
    }

    #[test]
    #[should_panic(expected = "transpose dimensions overflow")]
    fn rejects_overflowing_dimensions_before_access() {
        // Rows must not be read when the product and strided extent overflow.
        transpose_rows::<u32, _>(
            usize::MAX / 2 + 2,
            2,
            |_| panic!("must not read a row"),
            &mut [0; 2],
        );
    }

    #[test]
    fn rejects_overflowing_stride_before_reading_a_row() {
        use core::cell::Cell;

        // Addition overflow and multiplication overflow must both be rejected.
        for (cols, stride) in [(2, usize::MAX), (3, usize::MAX / 2 + 1)] {
            for words in [false, true] {
                let calls = Cell::new(0);
                let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    let row = |_| -> &[u32] {
                        calls.set(calls.get() + 1);
                        // Rows must not be read once the strided extent overflows.
                        panic!("must not read a row");
                    };
                    if words {
                        // SAFETY: u32 supports integer-word copying.
                        unsafe { transpose_rows_strided_words(1, cols, row, &mut [0], stride) };
                    } else {
                        transpose_rows_strided(1, cols, row, &mut [0], stride);
                    }
                }));
                assert!(result.is_err());
                assert_eq!(
                    calls.get(),
                    0,
                    "overflow must be rejected before reading rows"
                );
            }
        }
    }

    #[test]
    fn padding_and_uninitialized_bytes_are_copied_as_typed_values() {
        use core::mem::MaybeUninit;

        #[derive(Clone, Copy, Debug, Eq, PartialEq)]
        #[repr(C, align(4))]
        struct Padded(u8);
        let src: Vec<Vec<_>> = (0..16)
            .map(|r| (0..17).map(|c| Padded((r * 17 + c) as u8)).collect())
            .collect();
        let mut dst = vec![Padded(0); 16 * 17];
        transpose_rows(16, 17, |r| &src[r], &mut dst);
        for r in 0..16 {
            for c in 0..17 {
                assert_eq!(dst[c * 16 + r], src[r][c]);
            }
        }

        let mut src = [[MaybeUninit::<u32>::uninit(); 17]; 16];
        src[1][2].write(73);
        let mut dst = [MaybeUninit::uninit(); 16 * 17];
        transpose_rows(16, 17, |r| &src[r], &mut dst);
        // SAFETY: this slot comes from the initialized src[1][2].
        assert_eq!(unsafe { dst[2 * 16 + 1].assume_init() }, 73);
    }

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
        u32::transpose_rows(16, 17, |r| &src[r], &mut fast);
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
        // 31-bit and 64-bit field elements take the 32-bit and 64-bit kernels.
        //
        // Their values must come out unchanged, only moved.
        fn check<F: Packable + PrimeCharacteristicRing + core::fmt::Debug>() {
            let src: Vec<Vec<F>> = (0..16)
                .map(|r| (0..9).map(|c| F::from_usize(r * 9 + c)).collect())
                .collect();
            let mut dst = vec![F::ZERO; 16 * 9];
            F::transpose_rows(16, 9, |r| &src[r], &mut dst);
            for (r, row) in src.iter().enumerate() {
                for (c, &x) in row.iter().enumerate() {
                    assert_eq!(dst[c * 16 + r], x);
                }
            }
        }
        check::<BabyBear>();
        check::<Goldilocks>();
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

    /// Compare the word kernels with the scalar reference on one shape.
    ///
    /// - Rows carry extra words that must never reach the output.
    /// - A stride past the row count leaves gaps that must keep their sentinel.
    fn check_words<T>(
        rows: usize,
        cols: usize,
        slack: usize,
        gap: usize,
        value: impl Fn(usize, usize) -> T,
    ) where
        T: Copy + PartialEq + core::fmt::Debug + From<u8>,
    {
        let src: Vec<Vec<T>> = (0..rows)
            .map(|r| (0..cols + slack).map(|c| value(r, c)).collect())
            .collect();
        let stride = rows + gap;
        let sentinel = T::from(u8::MAX);

        let mut fast = vec![sentinel; stride * cols];
        let mut reference = vec![sentinel; stride * cols];
        // SAFETY: integers have initialized, padding-free, pointer-free representations.
        unsafe { transpose_rows_strided_words(rows, cols, |r| &src[r], &mut fast, stride) };
        transpose_rows_scalar(rows, cols, |r| &src[r], &mut reference, stride);
        assert_eq!(fast, reference);
    }

    proptest! {
        #[test]
        fn vector_kernel_matches_reference(
            rows in 0usize..40,
            cols in 0usize..40,
            slack in 0usize..3,
            gap in 0usize..3,
            seed in any::<u64>(),
        ) {
            // Shapes on both sides of every tile side (4, 8, 16), including clipped edges.
            let value = |r: usize, c: usize| seed.wrapping_mul(r as u64 + 1).wrapping_add(c as u64 * 7919);
            check_words(rows, cols, slack, gap, |r, c| value(r, c) as u32);
            check_words(rows, cols, slack, gap, value);
        }
    }
}
