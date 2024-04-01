use alloc::collections::BTreeMap;
use alloc::vec::Vec;
use core::cell::RefCell;

use p3_field::{Field, TwoAdicField};
use p3_matrix::dense::{RowMajorMatrix, RowMajorMatrixViewMut};
use p3_matrix::util::reverse_matrix_index_bits;
use p3_matrix::Matrix;
use p3_maybe_rayon::prelude::*;
use p3_util::log2_strict_usize;

use crate::butterflies::{Butterfly, DitButterfly, TwiddleFreeButterfly};
use crate::TwoAdicSubgroupDft;

/// The DIT FFT algorithm.
#[derive(Default, Clone)]
pub struct Radix2Dit<F: TwoAdicField> {
    /// Memoized twiddle factors for each length log_n.
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

/// One layer of a DIT butterfly network.
fn dit_layer<F: Field>(mat: &mut RowMajorMatrixViewMut<F>, layer: usize, twiddles: &[F]) {
    let h = mat.height();
    let log_h = log2_strict_usize(h);
    let layer_rev = log_h - 1 - layer;

    let half_block_size = 1 << layer;
    let block_size = half_block_size * 2;

    let _width = mat.width();

    mat.par_row_chunks_mut(block_size)
        .for_each(|mut block_chunks| {
            let (mut hi_chunks, mut lo_chunks) = block_chunks.split_rows_mut(half_block_size);
            hi_chunks
                .par_rows_mut()
                .zip(lo_chunks.par_rows_mut())
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

#[cfg(test)]
mod tests {
    use p3_baby_bear::BabyBear;
    use p3_goldilocks::Goldilocks;

    use crate::testing::*;
    use crate::Radix2Dit;

    #[test]
    fn dft_matches_naive() {
        test_dft_matches_naive::<BabyBear, Radix2Dit<_>>();
    }

    #[test]
    fn coset_dft_matches_naive() {
        test_coset_dft_matches_naive::<BabyBear, Radix2Dit<_>>();
    }

    #[test]
    fn idft_matches_naive() {
        test_idft_matches_naive::<Goldilocks, Radix2Dit<_>>();
    }

    #[test]
    fn coset_idft_matches_naive() {
        test_coset_idft_matches_naive::<BabyBear, Radix2Dit<_>>();
        test_coset_idft_matches_naive::<Goldilocks, Radix2Dit<_>>();
    }

    #[test]
    fn lde_matches_naive() {
        test_lde_matches_naive::<BabyBear, Radix2Dit<_>>();
    }

    #[test]
    fn coset_lde_matches_naive() {
        test_coset_lde_matches_naive::<BabyBear, Radix2Dit<_>>();
    }

    #[test]
    fn dft_idft_consistency() {
        test_dft_idft_consistency::<BabyBear, Radix2Dit<_>>();
    }
}
