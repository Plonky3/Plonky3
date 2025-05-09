use core::{cell::RefCell, iter};

use alloc::{collections::btree_map::BTreeMap, vec::Vec};
use p3_field::{Field, TwoAdicField};
use p3_matrix::{
    Matrix,
    dense::{RowMajorMatrix, RowMajorMatrixViewMut},
    util::reverse_matrix_index_bits,
};
use p3_maybe_rayon::prelude::*;
use p3_util::log2_strict_usize;

use crate::{
    Butterfly, DitButterfly, DitButterflyCopiedTwiddles, TwiddleFreeButterfly, TwoAdicSubgroupDft,
    VectorPair, compute_twiddles, second_half,
};

/// A FFT algorithm which divides a butterfly network's layers into two halves.
///
/// Unlike other FFT algorithms, this algorithm is optimized for small batch sizes.
/// Hence it does not do any parallelization.
///
/// For the first half, we apply a butterfly network with smaller blocks in earlier layers,
/// i.e. either DIT or Bowers G. Then we bit-reverse, and for the second half, we continue executing
/// the same network but in bit-reversed order. This way we're always working with small blocks,
/// so within each half, we can have a certain amount of parallelism with no cross-thread
/// communication.
#[derive(Default, Clone, Debug)]
pub struct Radix2DitSmallBatch<F> {
    /// Twiddles based on roots of unity, used in the forward DFT.
    twiddles: RefCell<BTreeMap<usize, VectorPair<F>>>,
}

impl<F: TwoAdicField + Ord> TwoAdicSubgroupDft<F> for Radix2DitSmallBatch<F> {
    type Evaluations = RowMajorMatrix<F>;

    fn dft_batch(&self, mut mat: RowMajorMatrix<F>) -> Self::Evaluations {
        let h = mat.height();
        if h == 1 {
            return mat;
        }
        let log_h = log2_strict_usize(h);

        let mid = log_h >> 1;

        // Compute twiddle factors, or take memoized ones if already available.
        let mut twiddles_ref_mut = self.twiddles.borrow_mut();
        let twiddles = twiddles_ref_mut
            .entry(log_h)
            .or_insert_with(|| compute_twiddles(log_h));

        // DIF butterfly
        for layer in (mid..log_h).rev() {
            dit_layer_non_bit_reversed(&mut mat, layer, &twiddles.bitrev_twiddles);
        }
        // reverse_matrix_index_bits(&mut mat);
        // for layer in (0..mid).rev() {
        //     dit_layer_bit_reversed(
        //         &mut mat.as_view_mut(),
        //         log_h - layer - 1,
        //         &twiddles.twiddles,
        //     );
        // }
        // mat
        second_half(&mut mat, mid, &twiddles.bitrev_twiddles, None);
        reverse_matrix_index_bits(&mut mat);
        mat
    }

    // #[instrument(skip_all, fields(dims = %mat.dimensions(), added_bits = added_bits))]
    // fn coset_lde_batch(
    //     &self,
    //     mut mat: RowMajorMatrix<F>,
    //     added_bits: usize,
    //     shift: F,
    // ) -> Self::Evaluations {
    //     let w = mat.width;
    //     let h = mat.height();
    //     let log_h = log2_strict_usize(h);
    //     let mid = log_h.div_ceil(2);

    //     let mut inverse_twiddles_ref_mut = self.inverse_twiddles.borrow_mut();
    //     let inverse_twiddles = inverse_twiddles_ref_mut
    //         .entry(log_h)
    //         .or_insert_with(|| compute_inverse_twiddles(log_h));

    //     // The first half looks like a normal DIT.
    //     first_half(&mut mat, mid, &inverse_twiddles.twiddles);

    //     // For the second half, we flip the DIT, working in bit-reversed order.
    //     reverse_matrix_index_bits(&mut mat);
    //     // We'll also scale by 1/h, as per the usual inverse DFT algorithm.
    //     // If F isn't a PrimeField, (and is thus an extension field) it's much cheaper to
    //     // invert in F::PrimeSubfield.
    //     let h_inv_subfield = F::PrimeSubfield::from_int(h).try_inverse();
    //     let scale = h_inv_subfield.map(F::from_prime_subfield);
    //     second_half(&mut mat, mid, &inverse_twiddles.bitrev_twiddles, scale);
    //     // We skip the final bit-reversal, since the next FFT expects bit-reversed input.

    //     let lde_elems = w * (h << added_bits);
    //     let elems_to_add = lde_elems - w * h;
    //     debug_span!("reserve_exact").in_scope(|| mat.values.reserve_exact(elems_to_add));

    //     let g_big = F::two_adic_generator(log_h + added_bits);

    //     let mat_ptr = mat.values.as_mut_ptr();
    //     let rest_ptr = unsafe { (mat_ptr as *mut MaybeUninit<F>).add(w * h) };
    //     let first_slice: &mut [F] = unsafe { slice::from_raw_parts_mut(mat_ptr, w * h) };
    //     let rest_slice: &mut [MaybeUninit<F>] =
    //         unsafe { slice::from_raw_parts_mut(rest_ptr, lde_elems - w * h) };
    //     let mut first_coset_mat = RowMajorMatrixViewMut::new(first_slice, w);
    //     let mut rest_cosets_mat = rest_slice
    //         .chunks_exact_mut(w * h)
    //         .map(|slice| RowMajorMatrixViewMut::new(slice, w))
    //         .collect_vec();

    //     for coset_idx in 1..(1 << added_bits) {
    //         let total_shift = g_big.exp_u64(coset_idx as u64) * shift;
    //         let coset_idx = reverse_bits_len(coset_idx, added_bits);
    //         let dest = &mut rest_cosets_mat[coset_idx - 1]; // - 1 because we removed the first matrix.
    //         coset_dft_oop(self, &first_coset_mat.as_view(), dest, total_shift);
    //     }

    //     // Now run a forward DFT on the very first coset, this time in-place.
    //     coset_dft(self, &mut first_coset_mat.as_view_mut(), shift);

    //     // SAFETY: We wrote all values above.
    //     unsafe {
    //         mat.values.set_len(lde_elems);
    //     }
    //     BitReversalPerm::new_view(mat)
    // }
}

/// Applies one layer of the Radix-2 DIT FFT butterfly network.
///
/// Splits the matrix into blocks of rows and performs in-place butterfly operations
/// on each block. Uses a `TwiddleFreeButterfly` for the first pair and `DitButterfly`
/// with precomputed twiddles for the rest.
///
/// # Arguments
/// - `mat`: Mutable matrix view with height as a power of two.
/// - `layer`: Index of the current FFT layer (starting at 0).
/// - `twiddles`: Precomputed twiddle factors for this layer.
fn dit_layer_non_bit_reversed<F: Field>(mat: &mut RowMajorMatrix<F>, layer: usize, twiddles: &[F]) {
    // Each butterfly operates on 2 rows; this is the number of rows in half a block
    let half_block_size = 1 << layer;
    // Each block contains 2^layer * 2 rows; full size of the butterfly block
    let block_size = half_block_size * 2;

    // Process the matrix in blocks of rows of size `block_size`
    mat.par_row_chunks_exact_mut(block_size)
        .enumerate()
        .for_each(|(ind, mut block_chunks)| {
            // Split each block vertically into top (hi) and bottom (lo) halves
            let (hi_chunks, lo_chunks) = block_chunks.split_rows_mut(half_block_size);
            if ind == 0 {
                // The first pair doesn't require a twiddle factor
                TwiddleFreeButterfly.apply_to_rows(hi_chunks.values, lo_chunks.values)
            } else {
                // Apply DIT butterfly using the twiddle factor at index `ind << layer`
                DitButterfly(twiddles[ind]).apply_to_rows(hi_chunks.values, lo_chunks.values)
            }
        });
}

fn _dit_layer_bit_reversed<F: Field>(
    mat: &mut RowMajorMatrixViewMut<'_, F>,
    layer: usize,
    twiddles: &[F],
) {
    // Each butterfly operates on 2 rows; this is the number of rows in half a block
    let half_block_size = 1 << layer;
    // Each block contains 2^layer * 2 rows; full size of the butterfly block
    let block_size = half_block_size * 2;

    let h = mat.height();
    let log_h = log2_strict_usize(h);

    let w = mat.width();
    let extended_twiddles = twiddles
        .iter()
        .step_by(1 << (log_h - layer - 1))
        .copied()
        .flat_map(|twiddle| iter::repeat_n(twiddle, w))
        .collect::<Vec<_>>();
    assert_eq!(extended_twiddles.len(), half_block_size * w);

    let butterfly = DitButterflyCopiedTwiddles(extended_twiddles);

    // Process the matrix in blocks of rows of size `block_size`
    mat.par_row_chunks_exact_mut(block_size)
        .for_each(|mut block_chunks| {
            // Split each block vertically into top (hi) and bottom (lo) halves
            let (hi_chunks, lo_chunks) = block_chunks.split_rows_mut(half_block_size);
            (&butterfly).apply_to_rows(hi_chunks.values, lo_chunks.values);
        });
}
