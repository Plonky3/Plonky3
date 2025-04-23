use alloc::collections::BTreeMap;
use alloc::vec::Vec;
use core::cell::RefCell;

use p3_field::{Field, TwoAdicField};
use p3_matrix::Matrix;
use p3_matrix::dense::{RowMajorMatrix, RowMajorMatrixViewMut};
use p3_matrix::util::reverse_matrix_index_bits;
use p3_maybe_rayon::prelude::*;
use p3_util::log2_strict_usize;

use crate::TwoAdicSubgroupDft;
use crate::butterflies::{Butterfly, DitButterfly, TwiddleFreeButterfly};

/// Radix-2 Decimation-in-Time FFT over a two-adic subgroup.
///
/// This struct implements a fast Fourier transform (FFT) using the Radix-2
/// Decimation-in-Time (DIT) algorithm over a two-adic multiplicative subgroup of a finite field.
/// It is optimized for a batch setting where multiple FFT's are being computed simultaneously.
///
/// Internally, the implementation memoizes twiddle factors (powers of the root of unity)
/// for reuse across multiple transforms. This avoids redundant computation
/// when performing FFTs of the same size.
#[derive(Default, Clone, Debug)]
pub struct Radix2Dit<F: TwoAdicField> {
    /// Memoized twiddle factors indexed by `log2(n)`, where `n` is the DFT length.
    ///
    /// This allows fast lookup and reuse of previously computed twiddle values
    /// (powers of a two-adic generator), which are expensive to recompute.
    ///
    /// `RefCell` is used to enable interior mutability for caching purposes.
    twiddles: RefCell<BTreeMap<usize, Vec<F>>>,
}

impl<F: TwoAdicField> TwoAdicSubgroupDft<F> for Radix2Dit<F> {
    type Evaluations = RowMajorMatrix<F>;

    fn dft_batch(&self, mut mat: RowMajorMatrix<F>) -> RowMajorMatrix<F> {
        let h = mat.height();
        let log_h = log2_strict_usize(h);

        // Compute twiddle factors, or take memoized ones if already available.
        let mut twiddles_ref_mut = self.twiddles.borrow_mut();
        let twiddles = twiddles_ref_mut.entry(log_h).or_insert_with(|| {
            let root = F::two_adic_generator(log_h);
            root.powers().take(1 << log_h).collect()
        });

        // DIT butterfly
        reverse_matrix_index_bits(&mut mat);
        for layer in 0..log_h {
            dit_layer(&mut mat.as_view_mut(), layer, twiddles);
        }
        mat
    }
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
fn dit_layer<F: Field>(mat: &mut RowMajorMatrixViewMut<'_, F>, layer: usize, twiddles: &[F]) {
    // Get the number of rows in the matrix (must be a power of two)
    let h = mat.height();
    // Compute reversed layer index to access twiddle indices correctly
    let log_h = log2_strict_usize(h);
    let layer_rev = log_h - 1 - layer;

    // Each butterfly operates on 2 rows; this is the number of rows in half a block
    let half_block_size = 1 << layer;
    // Each block contains 2^layer * 2 rows; full size of the butterfly block
    let block_size = half_block_size * 2;

    // Process the matrix in blocks of rows of size `block_size`
    mat.par_row_chunks_exact_mut(block_size)
        .for_each(|mut block_chunks| {
            // Split each block vertically into top (hi) and bottom (lo) halves
            let (mut hi_chunks, mut lo_chunks) = block_chunks.split_rows_mut(half_block_size);
            // For each pair of rows (hi, lo), apply a butterfly
            hi_chunks
                .par_rows_mut()
                .zip(lo_chunks.par_rows_mut())
                .enumerate()
                .for_each(|(ind, (hi_chunk, lo_chunk))| {
                    if ind == 0 {
                        // The first pair doesn't require a twiddle factor
                        TwiddleFreeButterfly.apply_to_rows(hi_chunk, lo_chunk)
                    } else {
                        // Apply DIT butterfly using the twiddle factor at index `ind << layer_rev`
                        DitButterfly(twiddles[ind << layer_rev]).apply_to_rows(hi_chunk, lo_chunk)
                    }
                });
        });
}
