use alloc::vec::Vec;

use p3_field::{Field, TwoAdicField};
use p3_matrix::dense::{RowMajorMatrix, RowMajorMatrixViewMut};
use p3_matrix::view::{MatrixView, RowPermutation};
use p3_matrix::{Matrix, MatrixRows};
use p3_maybe_rayon::{IndexedParallelIterator, MaybeParChunksMut, ParallelIterator};
use p3_util::log2_strict_usize;

use crate::butterflies::{dit_butterfly_on_rows, twiddle_free_butterfly_on_rows};
use crate::util::reverse_matrix_index_bits;
use crate::TwoAdicSubgroupDft;

/// The DIT FFT algorithm.
#[derive(Default, Clone)]
pub struct Radix2Dit;

impl<F: TwoAdicField> TwoAdicSubgroupDft<F> for Radix2Dit {
    fn dft_batch(&self, mat: impl MatrixRows<F>) -> MatrixView<F, RowMajorMatrix<F>> {
        let h = mat.height();
        let log_h = log2_strict_usize(h);

        let root = F::two_adic_generator(log_h);
        let twiddles: Vec<F> = root.powers().take(h / 2).collect();

        // DIT butterfly
        let mut mat = mat
            .permute_rows(RowPermutation::BitReversed)
            .to_row_major_matrix();
        for layer in 0..log_h {
            dit_layer(&mut mat.as_view_mut(), layer, &twiddles);
        }
        MatrixView::identity(mat)
    }
}

/// One layer of a DIT butterfly network.
fn dit_layer<F: Field>(mat: &mut RowMajorMatrixViewMut<F>, layer: usize, twiddles: &[F]) {
    let h = mat.height();
    let log_h = log2_strict_usize(h);
    let layer_rev = log_h - 1 - layer;

    let half_block_size = 1 << layer;
    let block_size = half_block_size * 2;

    let width = mat.width();

    mat.par_row_chunks_mut(block_size).for_each(|block_chunks| {
        let (hi_chunks, lo_chunks) = block_chunks.split_at_mut(half_block_size * width);
        hi_chunks
            .par_chunks_exact_mut(width)
            .zip(lo_chunks.par_chunks_exact_mut(width))
            .enumerate()
            .for_each(|(ind, (hi_chunk, lo_chunk))| {
                if ind == 0 {
                    twiddle_free_butterfly_on_rows(hi_chunk, lo_chunk)
                } else {
                    let twiddle = twiddles[ind << layer_rev];
                    dit_butterfly_on_rows(hi_chunk, lo_chunk, twiddle)
                }
            });
    });
}

#[cfg(test)]
mod tests {
    use p3_baby_bear::BabyBear;
    use p3_goldilocks::Goldilocks;

    use crate::testing::*;
    use crate::Radix2Dit;

    #[test]
    fn dft_matches_naive() {
        test_dft_matches_naive::<BabyBear, Radix2Dit>();
    }

    #[test]
    fn coset_dft_matches_naive() {
        test_coset_dft_matches_naive::<BabyBear, Radix2Dit>();
    }

    #[test]
    fn idft_matches_naive() {
        test_idft_matches_naive::<Goldilocks, Radix2Dit>();
    }

    #[test]
    fn lde_matches_naive() {
        test_lde_matches_naive::<BabyBear, Radix2Dit>();
    }

    #[test]
    fn coset_lde_matches_naive() {
        test_coset_lde_matches_naive::<BabyBear, Radix2Dit>();
    }

    #[test]
    fn dft_idft_consistency() {
        test_dft_idft_consistency::<BabyBear, Radix2Dit>();
    }
}
