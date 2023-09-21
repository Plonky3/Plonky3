use alloc::vec;

use p3_field::Field;
use p3_matrix::dense::RowMajorMatrix;
use p3_matrix::Matrix;
use p3_maybe_rayon::{MaybeIntoParIter, ParallelIterator};
use p3_util::log2_strict_usize;

use crate::FourierTransform;

pub fn reverse_slice_index_bits<F>(vals: &mut [F]) {
    let n = vals.len();
    if n == 0 {
        return;
    }
    let log_n = log2_strict_usize(n);

    for i in 0..n {
        let j = reverse_bits_len(i, log_n);
        if i < j {
            vals.swap(i, j);
        }
    }
}

pub(crate) fn reverse_matrix_index_bits<F>(mat: &mut RowMajorMatrix<F>) {
    let w = mat.width();
    let h = mat.height();
    let log_h = log2_strict_usize(h);
    let values = mat.values.as_mut_ptr() as usize;

    (0..h).into_par_iter().for_each(|i| {
        let values = values as *mut F;
        let j = reverse_bits_len(i, log_h);
        if i < j {
            unsafe { swap_rows_raw(values, w, i, j) };
        }
    });
}

#[inline]
pub const fn reverse_bits(x: usize, n: usize) -> usize {
    reverse_bits_len(x, n.trailing_zeros() as usize)
}

#[inline]
pub const fn reverse_bits_len(x: usize, bit_len: usize) -> usize {
    // NB: The only reason we need overflowing_shr() here as opposed
    // to plain '>>' is to accommodate the case n == num_bits == 0,
    // which would become `0 >> 64`. Rust thinks that any shift of 64
    // bits causes overflow, even when the argument is zero.
    x.reverse_bits()
        .overflowing_shr(usize::BITS - bit_len as u32)
        .0
}

/// Assumes `i < j`.
pub(crate) fn swap_rows<F>(mat: &mut RowMajorMatrix<F>, i: usize, j: usize) {
    let w = mat.width();
    let (upper, lower) = mat.values.split_at_mut(j * w);
    let row_i = &mut upper[i * w..(i + 1) * w];
    let row_j = &mut lower[..w];
    row_i.swap_with_slice(row_j);
}

/// Assumes `i < j`.
///
/// SAFETY: The caller must ensure `i < j < h`, where `h` is the height of the matrix.
pub(crate) unsafe fn swap_rows_raw<F>(mat: *mut F, w: usize, i: usize, j: usize) {
    let row_i = core::slice::from_raw_parts_mut(mat.add(i * w), w);
    let row_j = core::slice::from_raw_parts_mut(mat.add(j * w), w);
    row_i.swap_with_slice(row_j);
}

/// Divide each coefficient of the given matrix by its height.
pub(crate) fn divide_by_height<F: Field>(mat: &mut RowMajorMatrix<F>) {
    let h = mat.height();
    let h_inv = F::from_canonical_usize(h).inverse();
    let (prefix, shorts, suffix) = unsafe { mat.values.align_to_mut::<F::Packing>() };
    prefix.iter_mut().for_each(|x| *x *= h_inv);
    shorts.iter_mut().for_each(|x| *x *= h_inv);
    suffix.iter_mut().for_each(|x| *x *= h_inv);
}

/// Append zeros to the "end" of the given matrix, except that the matrix is in bit-reversed order,
/// so in actuality we're interleaving zero rows.
#[inline]
pub(crate) fn bit_reversed_zero_pad<F: Field>(mat: &mut RowMajorMatrix<F>, added_bits: usize) {
    if added_bits == 0 {
        return;
    }

    // This is equivalent to:
    //     reverse_matrix_index_bits(mat);
    //     mat
    //         .values
    //         .resize(mat.values.len() << added_bits, F::ZERO);
    //     reverse_matrix_index_bits(mat);
    // But rather than implement it with bit reversals, we directly construct the resulting matrix,
    // whose rows are zero except for rows whose low `added_bits` bits are zero.

    let w = mat.width;
    let mut values = vec![F::ZERO; mat.values.len() << added_bits];
    for i in (0..mat.values.len()).step_by(w) {
        values[(i << added_bits)..((i << added_bits) + w)].copy_from_slice(&mat.values[i..i + w]);
    }
    *mat = RowMajorMatrix::new(values, w);
}

pub(crate) fn idft_batch<F: Field, Dft: FourierTransform<F, Range = F>>(
    algo: &Dft,
    mat: RowMajorMatrix<F>,
) -> RowMajorMatrix<F> {
    let mut dft = algo.dft_batch(mat);
    let h = dft.height();

    divide_by_height(&mut dft);

    for row in 1..h / 2 {
        swap_rows(&mut dft, row, h - row);
    }

    dft
}
