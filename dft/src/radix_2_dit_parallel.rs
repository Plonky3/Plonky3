use alloc::vec::Vec;

use itertools::{izip, Itertools};
use p3_field::{scale_slice_in_place, Field, Powers, TwoAdicField};
use p3_matrix::bitrev::{BitReversableMatrix, BitReversedMatrixView};
use p3_matrix::dense::{RowMajorMatrix, RowMajorMatrixViewMut};
use p3_matrix::util::reverse_matrix_index_bits;
use p3_matrix::Matrix;
use p3_maybe_rayon::prelude::*;
use p3_util::{log2_strict_usize, reverse_bits, reverse_slice_index_bits};
use tracing::instrument;

use crate::butterflies::{Butterfly, DitButterfly};
use crate::TwoAdicSubgroupDft;

/// A parallel FFT algorithm which divides a butterfly network's layers into two halves.
///
/// For the first half, we apply a butterfly network with smaller blocks in earlier layers,
/// i.e. either DIT or Bowers G. Then we bit-reverse, and for the second half, we continue executing
/// the same network but in bit-reversed order. This way we're always working with small blocks,
/// so within each half, we can have a certain amount of parallelism with no cross-thread
/// communication.
#[derive(Default, Clone, Debug)]
pub struct Radix2DitParallel;

impl<F: TwoAdicField> TwoAdicSubgroupDft<F> for Radix2DitParallel {
    type Evaluations = BitReversedMatrixView<RowMajorMatrix<F>>;

    fn dft_batch(&self, mut mat: RowMajorMatrix<F>) -> Self::Evaluations {
        let h = mat.height();
        let log_h = log2_strict_usize(h);

        let root = F::two_adic_generator(log_h);
        let mut twiddles: Vec<F> = root.powers().take(h / 2).collect();

        let mid = log_h / 2;

        // The first half looks like a normal DIT.
        reverse_matrix_index_bits(&mut mat);
        par_dit_layer(&mut mat, mid, &twiddles);

        // For the second half, we flip the DIT, working in bit-reversed order.
        reverse_matrix_index_bits(&mut mat);
        reverse_slice_index_bits(&mut twiddles);
        par_dit_layer_rev(&mut mat, mid, &twiddles);

        mat.bit_reverse_rows()
    }

    #[instrument(skip_all, fields(dims = %mat.dimensions(), added_bits))]
    fn coset_lde_batch(
        &self,
        mut mat: RowMajorMatrix<F>,
        added_bits: usize,
        shift: F,
    ) -> Self::Evaluations {
        let h = mat.height();
        let log_h = log2_strict_usize(h);
        let mid = log_h / 2;
        let h_inv = F::from_canonical_usize(h).inverse();

        let root = F::two_adic_generator(log_h);
        let root_inv = root.inverse();

        let mut twiddles_inv: Vec<F> = root_inv.powers().take(h / 2).collect();

        // The first half looks like a normal DIT.
        reverse_matrix_index_bits(&mut mat);
        par_dit_layer(&mut mat, mid, &twiddles_inv);

        // For the second half, we flip the DIT, working in bit-reversed order.
        reverse_matrix_index_bits(&mut mat);
        reverse_slice_index_bits(&mut twiddles_inv);
        par_dit_layer_rev(&mut mat, mid, &twiddles_inv);
        // We skip the final bit-reversal, since the next FFT expects bit-reversed input.

        // Rescale coefficients in two ways:
        // - divide by height (since we're doing an inverse DFT)
        // - multiply by powers of the coset shift (see default coset LDE impl for an explanation)
        let weights = Powers {
            base: shift,
            current: h_inv,
        }
        .take(h)
        .collect_vec();
        mat.par_rows_mut().enumerate().for_each(|(r, row)| {
            // reverse_bits because mat is encoded in bit-reversed order
            scale_slice_in_place(weights[reverse_bits(r, h)], row);
        });

        mat = mat.bit_reversed_zero_pad(added_bits);

        let h = mat.height();
        let log_h = log2_strict_usize(h);
        let mid = log_h / 2;

        let root = F::two_adic_generator(log_h);

        let mut twiddles: Vec<F> = root.powers().take(h / 2).collect();

        // The first half looks like a normal DIT.
        par_dit_layer(&mut mat, mid, &twiddles);

        // For the second half, we flip the DIT, working in bit-reversed order.
        reverse_matrix_index_bits(&mut mat);
        reverse_slice_index_bits(&mut twiddles);
        par_dit_layer_rev(&mut mat, mid, &twiddles);

        mat.bit_reverse_rows()
    }
}

/// This can be used as the first half of a parallelized butterfly network.
#[instrument(level = "debug", skip_all)]
fn par_dit_layer<F: Field>(mat: &mut RowMajorMatrix<F>, mid: usize, twiddles: &[F]) {
    let log_h = log2_strict_usize(mat.height());

    // max block size: 2^mid
    mat.par_row_chunks_exact_mut(1 << mid)
        .for_each(|mut submat| {
            for layer in 0..mid {
                dit_layer(&mut submat, log_h, layer, twiddles);
            }
        });
}

/// This can be used as the second half of a parallelized butterfly network.
#[instrument(level = "debug", skip_all)]
fn par_dit_layer_rev<F: Field>(mat: &mut RowMajorMatrix<F>, mid: usize, twiddles_rev: &[F]) {
    let log_h = log2_strict_usize(mat.height());

    // max block size: 2^(log_h - mid)
    mat.par_row_chunks_exact_mut(1 << (log_h - mid))
        .enumerate()
        .for_each(|(thread, mut submat)| {
            for layer in mid..log_h {
                let first_block = thread << (layer - mid);
                dit_layer_rev(&mut submat, log_h, layer, &twiddles_rev[first_block..]);
            }
        });
}

/// One layer of a DIT butterfly network.
fn dit_layer<F: Field>(
    submat: &mut RowMajorMatrixViewMut<'_, F>,
    log_h: usize,
    layer: usize,
    twiddles: &[F],
) {
    let layer_rev = log_h - 1 - layer;
    let layer_pow = 1 << layer_rev;

    let half_block_size = 1 << layer;
    let block_size = half_block_size * 2;
    let width = submat.width();
    debug_assert!(submat.height() >= block_size);

    for block in submat.values.chunks_mut(block_size * width) {
        let (lows, highs) = block.split_at_mut(half_block_size * width);

        for (lo, hi, &twiddle) in izip!(
            lows.chunks_mut(width),
            highs.chunks_mut(width),
            twiddles.iter().step_by(layer_pow)
        ) {
            DitButterfly(twiddle).apply_to_rows(lo, hi);
        }
    }
}

/// Like `dit_layer`, except the matrix and twiddles are encoded in bit-reversed order.
/// This can also be viewed as a layer of the Bowers G^T network.
fn dit_layer_rev<F: Field>(
    submat: &mut RowMajorMatrixViewMut<'_, F>,
    log_h: usize,
    layer: usize,
    twiddles_rev: &[F],
) {
    let layer_rev = log_h - 1 - layer;

    let half_block_size = 1 << layer_rev;
    let block_size = half_block_size * 2;
    let width = submat.width();
    debug_assert!(submat.height() >= block_size);

    for (block, &twiddle) in submat
        .values
        .chunks_mut(block_size * width)
        .zip(twiddles_rev)
    {
        let (lo, hi) = block.split_at_mut(half_block_size * width);
        DitButterfly(twiddle).apply_to_rows(lo, hi)
    }
}
