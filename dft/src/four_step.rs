use alloc::collections::BTreeMap;
use alloc::vec::Vec;
use core::cell::RefCell;

use itertools::Itertools;
use p3_field::{scale_slice_in_place, Field, TwoAdicField};
use p3_matrix::dense::{RowMajorMatrix, RowMajorMatrixViewMut};
use p3_matrix::util::{reverse_matrix_index_bits, reverse_matrix_index_bits_strided, swap_rows};
use p3_matrix::Matrix;
use p3_maybe_rayon::prelude::*;
use p3_util::{log2_strict_usize, MutPtr};
use tracing::{info_span, instrument};

use crate::butterflies::{Butterfly, DitButterfly, TwiddleFreeButterfly};
use crate::util::coset_shift_cols;
use crate::{divide_by_height, TwoAdicSubgroupDft};

/// The four-step NTT algorithm, a.k.a. Bailey's NTT.
///
/// Besides Bailey's paper, Remco's [NTT explainer](https://2π.com/23/ntt/) is a nice reference for
/// the algorithm.
#[derive(Default, Clone, Debug)]
pub struct FourStep<F: TwoAdicField> {
    /// Memoized twiddle factors for each length `log_n`.
    twiddles: RefCell<BTreeMap<usize, Vec<F>>>,
}

impl<F: TwoAdicField> TwoAdicSubgroupDft<F> for FourStep<F> {
    type Evaluations = RowMajorMatrix<F>;

    #[instrument(level = "info", skip_all)]
    fn dft_batch(&self, mut mat: RowMajorMatrix<F>) -> RowMajorMatrix<F> {
        let n = mat.height();
        assert_ne!(n, 0);

        let log_n = log2_strict_usize(n);
        let log_w = log_n / 2;
        let log_h = log_n - log_w;
        let w = 1 << log_w;
        let h = 1 << log_h;

        // Compute twiddle factors, or take memoized ones if already available.
        let mut twiddles_ref_mut = self.twiddles.borrow_mut();
        let twiddles = twiddles_ref_mut.entry(log_n).or_insert_with(|| {
            let root = F::two_adic_generator(log_n);
            let bases = root.powers().take(h);
            bases.flat_map(|base| base.powers().take(w)).collect()
        });

        // In-place DFT on each column.
        let mat_ptr = MutPtr(mat.values.as_mut_ptr());
        info_span!("DFT each column", w).in_scope(|| {
            (0..w).into_par_iter().for_each(|c| {
                dft_batch_in_place_strided(mat_ptr, mat.width, mat.height(), c, log_w);
            });
        });

        info_span!("apply twiddles").in_scope(|| {
            mat.par_rows_mut().zip(twiddles).for_each(|(row, twiddle)| {
                scale_slice_in_place(*twiddle, row);
            });
        });

        // In-place DFT on each row.
        info_span!("DFT each row", h).in_scope(|| {
            mat.par_row_chunks_exact_mut(w).for_each(|chunk| {
                dft_batch_in_place(chunk);
            });
        });

        mat = transpose(mat, w, h);
        mat
    }

    fn coset_lde_batch(
        &self,
        mat: RowMajorMatrix<F>,
        added_bits: usize,
        shift: F,
    ) -> Self::Evaluations {
        // Inverse.
        let mut mat = self.dft_batch(mat);
        divide_by_height(&mut mat);
        let h = mat.height();
        for row in 1..h / 2 {
            swap_rows(&mut mat, row, h - row);
        }

        // Forward.
        mat.values.resize(mat.values.len() << added_bits, F::ZERO);
        coset_shift_cols(&mut mat, shift);
        self.dft_batch(mat)
    }
}

fn dft_batch_in_place<F: TwoAdicField>(mut mat: RowMajorMatrixViewMut<F>) {
    let h = mat.height();
    let log_h = log2_strict_usize(h);

    let root = F::two_adic_generator(log_h);
    let twiddles = root.powers().take(1 << log_h).collect_vec();

    // DIT butterfly
    reverse_matrix_index_bits(&mut mat);
    for layer in 0..log_h {
        dit_layer(&mut mat.as_view_mut(), layer, &twiddles);
    }
}

/// Assumes `start < stride`.
fn dft_batch_in_place_strided<F: TwoAdicField>(
    mat: MutPtr<F>,
    w: usize,
    h: usize,
    start: usize,
    stride_bits: usize,
) {
    let h_strided = h >> stride_bits;
    let log_h_strided = log2_strict_usize(h_strided);

    let root = F::two_adic_generator(log_h_strided);
    let twiddles = root.powers().take(1 << log_h_strided).collect_vec();

    // DIT butterfly
    reverse_matrix_index_bits_strided(mat, w, h, start, stride_bits);
    for layer in 0..log_h_strided {
        dit_layer_strided(mat.0, w, h, start, stride_bits, layer, &twiddles);
    }
}

/// One layer of a DIT butterfly network.
fn dit_layer<F: Field>(mat: &mut RowMajorMatrixViewMut<'_, F>, layer: usize, twiddles: &[F]) {
    let h = mat.height();
    let log_h = log2_strict_usize(h);
    let layer_rev = log_h - 1 - layer;

    let half_block_size = 1 << layer;
    let block_size = half_block_size * 2;

    mat.row_chunks_exact_mut(block_size)
        .for_each(|mut block_chunks| {
            let (mut hi_chunks, mut lo_chunks) = block_chunks.split_rows_mut(half_block_size);
            hi_chunks
                .rows_mut()
                .zip(lo_chunks.rows_mut())
                .enumerate()
                .for_each(|(ind, (hi_chunk, lo_chunk))| {
                    if ind == 0 {
                        TwiddleFreeButterfly.apply_to_rows(hi_chunk, lo_chunk)
                    } else {
                        DitButterfly(twiddles[ind << layer_rev]).apply_to_rows(hi_chunk, lo_chunk)
                    }
                });
        });
}

/// One layer of a DIT butterfly network.
fn dit_layer_strided<F: Field>(
    mat: *mut F,
    w: usize,
    h: usize,
    start: usize,
    stride_bits: usize,
    layer: usize,
    twiddles: &[F],
) {
    let stride = 1 << stride_bits;
    let h_strided = h >> stride_bits;
    let log_h_strided = log2_strict_usize(h_strided);
    let layer_rev = log_h_strided - 1 - layer;

    let half_block_size = (1 << layer) << stride_bits;
    let block_size = half_block_size * 2;

    let mut mat =
        RowMajorMatrixViewMut::new(unsafe { core::slice::from_raw_parts_mut(mat, w * h) }, w);
    mat.row_chunks_exact_mut(block_size)
        .for_each(|mut block_chunks| {
            let (mut hi_chunks, mut lo_chunks) = block_chunks.split_rows_mut(half_block_size);
            hi_chunks
                .rows_mut_strided(start, stride)
                .zip(lo_chunks.rows_mut_strided(start, stride))
                .enumerate()
                .for_each(|(ind, (hi_chunk, lo_chunk))| {
                    if ind == 0 {
                        TwiddleFreeButterfly.apply_to_rows(hi_chunk, lo_chunk)
                    } else {
                        DitButterfly(twiddles[ind << layer_rev]).apply_to_rows(hi_chunk, lo_chunk)
                    }
                });
        });
}

/// Reinterprets the given height of the given matrix as having two inner dimensions, and transposes those two.
fn transpose<F: TwoAdicField>(
    src: RowMajorMatrix<F>,
    in_w: usize,
    in_h: usize,
) -> RowMajorMatrix<F> {
    let mut dst = Vec::with_capacity(src.values.capacity());
    for r in 0..in_w {
        for c in 0..in_h {
            dst.extend(src.row(c * in_w + r));
        }
    }
    RowMajorMatrix::new(dst, src.width())
}

#[cfg(test)]
mod tests {
    use alloc::vec::Vec;

    use p3_baby_bear::BabyBear;
    use p3_field::FieldAlgebra;
    use p3_matrix::dense::RowMajorMatrix;

    use crate::four_step::transpose;

    #[test]
    fn test_transpose() {
        // mat looks like:
        // 0 1 2
        // 3 4 5
        // 6 7 8
        // 9 10 11
        let values: Vec<_> = (0..12).map(BabyBear::from_canonical_u8).collect();
        let mat = RowMajorMatrix::new(values, 1);

        let mat_t = transpose(mat.clone(), 3, 4);
        assert_eq!(
            mat_t.values,
            [0, 3, 6, 9, 1, 4, 7, 10, 2, 5, 8, 11].map(BabyBear::from_canonical_u8)
        );
        let mat_t_t = transpose(mat_t.clone(), 4, 3);
        assert_eq!(mat, mat_t_t);
    }
}
