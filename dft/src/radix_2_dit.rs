use alloc::vec::Vec;

use p3_field::{Field, TwoAdicField};
use p3_matrix::dense::{RowMajorMatrix, RowMajorMatrixViewMut};
use p3_matrix::Matrix;
use p3_util::log2_strict_usize;

use crate::butterflies::{dit_butterfly, twiddle_free_butterfly};
use crate::util::{self, reverse_matrix_index_bits};
use crate::FourierTransform;

/// The DIT FFT algorithm.
#[derive(Default, Clone)]
pub struct Radix2Dit;

impl<F: TwoAdicField> FourierTransform<F> for Radix2Dit {
    type Range = F;

    fn dft_batch(&self, mut mat: RowMajorMatrix<F>) -> RowMajorMatrix<F> {
        let h = mat.height();
        let log_h = log2_strict_usize(h);

        let root = F::primitive_root_of_unity(log_h);
        let twiddles: Vec<F> = root.powers().take(h / 2).collect();

        // DIT butterfly
        reverse_matrix_index_bits(&mut mat);
        for layer in 0..log_h {
            dit_layer(&mut mat.as_view_mut(), layer, &twiddles);
        }
        mat
    }

    fn idft_batch(&self, mat: RowMajorMatrix<F>) -> RowMajorMatrix<F> {
        util::idft_batch(self, mat)
    }
}

/// One layer of a DIT butterfly network.
fn dit_layer<F: Field>(mat: &mut RowMajorMatrixViewMut<F>, layer: usize, twiddles: &[F]) {
    let h = mat.height();
    let log_h = log2_strict_usize(h);
    let layer_rev = log_h - 1 - layer;

    let half_block_size = 1 << layer;
    let block_size = half_block_size * 2;

    for j in (0..h).step_by(block_size) {
        // Unroll i=0 case
        let butterfly_hi = j;
        let butterfly_lo = butterfly_hi + half_block_size;
        twiddle_free_butterfly(mat, butterfly_hi, butterfly_lo);

        for i in 1..half_block_size {
            let butterfly_hi = j + i;
            let butterfly_lo = butterfly_hi + half_block_size;
            let twiddle = twiddles[i << layer_rev];
            dit_butterfly(mat, butterfly_hi, butterfly_lo, twiddle);
        }
    }
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
