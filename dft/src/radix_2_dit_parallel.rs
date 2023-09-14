use alloc::vec::Vec;

use p3_field::{Field, Powers, TwoAdicField};
use p3_matrix::dense::{RowMajorMatrix, RowMajorMatrixViewMut};
use p3_matrix::Matrix;
use p3_maybe_rayon::{IndexedParallelIterator, ParallelIterator};
use p3_util::log2_strict_usize;

use crate::butterflies::dit_butterfly;
use crate::util::{bit_reversed_zero_pad, reverse_matrix_index_bits};
use crate::{reverse_bits, reverse_slice_index_bits, TwoAdicSubgroupDft};

/// A parallel FFT algorithm which divides a butterfly network's layers into two halves.
///
/// For the first half, we apply a butterfly network with smaller blocks in earlier layers,
/// i.e. either DIT or Bowers G. Then we bit-reverse, and for the second half, we continue executing
/// the same network but in bit-reversed order. This way we're always working with small blocks,
/// so within each half, we can have a certain amount of parallelism with no cross-thread
/// communication.
#[derive(Default, Clone)]
pub struct Radix2DitParallel;

impl<F: TwoAdicField> TwoAdicSubgroupDft<F> for Radix2DitParallel {
    fn dft_batch(&self, mut mat: RowMajorMatrix<F>) -> RowMajorMatrix<F> {
        let h = mat.height();
        let log_h = log2_strict_usize(h);

        let root = F::primitive_root_of_unity(log_h);
        let mut twiddles: Vec<F> = root.powers().take(h / 2).collect();

        let mid = log_h / 2;

        // The first half looks like a normal DIT.
        reverse_matrix_index_bits(&mut mat);
        par_dit_layer(&mut mat, mid, &twiddles);

        // For the second half, we flip the DIT, working in bit-reversed order.
        reverse_matrix_index_bits(&mut mat);
        reverse_slice_index_bits(&mut twiddles);
        par_dit_layer_rev(&mut mat, mid, &twiddles);
        reverse_matrix_index_bits(&mut mat);

        mat
    }

    fn coset_lde_batch(
        &self,
        mut mat: RowMajorMatrix<F>,
        added_bits: usize,
        shift: F,
    ) -> RowMajorMatrix<F> {
        let h = mat.height();
        let log_h = log2_strict_usize(h);
        let mid = log_h / 2;
        let h_inv = F::from_canonical_usize(h).inverse();

        let root = F::primitive_root_of_unity(log_h);
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
        .take(h);
        for (row, weight) in weights.enumerate() {
            // reverse_bits because mat is encoded in bit-reversed order
            mat.scale_row(reverse_bits(row, h), weight);
        }

        bit_reversed_zero_pad(&mut mat, added_bits);

        let h = mat.height();
        let log_h = log2_strict_usize(h);
        let mid = log_h / 2;

        let root = F::primitive_root_of_unity(log_h);

        let mut twiddles: Vec<F> = root.powers().take(h / 2).collect();

        // The first half looks like a normal DIT.
        par_dit_layer(&mut mat, mid, &twiddles);

        // For the second half, we flip the DIT, working in bit-reversed order.
        reverse_matrix_index_bits(&mut mat);
        reverse_slice_index_bits(&mut twiddles);
        par_dit_layer_rev(&mut mat, mid, &twiddles);
        reverse_matrix_index_bits(&mut mat);

        mat
    }
}

/// This can be used as the first half of a parallelized butterfly network.
fn par_dit_layer<F: Field>(mat: &mut RowMajorMatrix<F>, mid: usize, twiddles: &[F]) {
    let log_h = log2_strict_usize(mat.height());

    // max block size: 2^mid
    mat.row_chunks_mut(1 << mid).for_each(|mut submat| {
        for layer in 0..mid {
            dit_layer(&mut submat, log_h, layer, twiddles);
        }
    });
}

/// This can be used as the second half of a parallelized butterfly network.
fn par_dit_layer_rev<F: Field>(mat: &mut RowMajorMatrix<F>, mid: usize, twiddles_rev: &[F]) {
    let log_h = log2_strict_usize(mat.height());

    // max block size: 2^(log_h - mid)
    mat.row_chunks_mut(1 << (log_h - mid))
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
    submat: &mut RowMajorMatrixViewMut<F>,
    log_h: usize,
    layer: usize,
    twiddles: &[F],
) {
    let layer_rev = log_h - 1 - layer;

    let half_block_size = 1 << layer;
    let block_size = half_block_size * 2;
    debug_assert!(submat.height() >= block_size);

    for block_start in (0..submat.height()).step_by(block_size) {
        for i in 0..half_block_size {
            let hi = block_start + i;
            let lo = hi + half_block_size;
            let twiddle = twiddles[i << layer_rev];
            dit_butterfly(submat, hi, lo, twiddle);
        }
    }
}

/// Like `dit_layer`, except the matrix and twiddles are encoded in bit-reversed order.
/// This can also be viewed as a layer of the Bowers G^T network.
fn dit_layer_rev<F: Field>(
    submat: &mut RowMajorMatrixViewMut<F>,
    log_h: usize,
    layer: usize,
    twiddles_rev: &[F],
) {
    let layer_rev = log_h - 1 - layer;

    let half_block_size = 1 << layer_rev;
    let block_size = half_block_size * 2;
    debug_assert!(submat.height() >= block_size);

    for (block, block_start) in (0..submat.height()).step_by(block_size).enumerate() {
        let twiddle = twiddles_rev[block];
        for i in 0..half_block_size {
            let hi = block_start + i;
            let lo = hi + half_block_size;
            dit_butterfly(submat, hi, lo, twiddle);
        }
    }
}

#[cfg(test)]
mod tests {
    use p3_baby_bear::BabyBear;
    use p3_goldilocks::Goldilocks;

    use crate::testing::*;
    use crate::Radix2DitParallel;

    #[test]
    fn dft_matches_naive() {
        test_dft_matches_naive::<BabyBear, Radix2DitParallel>();
    }

    #[test]
    fn idft_matches_naive() {
        test_idft_matches_naive::<Goldilocks, Radix2DitParallel>();
    }

    #[test]
    fn lde_matches_naive() {
        test_lde_matches_naive::<BabyBear, Radix2DitParallel>();
    }

    #[test]
    fn coset_lde_matches_naive() {
        test_coset_lde_matches_naive::<BabyBear, Radix2DitParallel>();
    }

    #[test]
    fn dft_idft_consistency() {
        test_dft_idft_consistency::<BabyBear, Radix2DitParallel>();
    }
}
