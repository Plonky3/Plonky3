use alloc::vec::Vec;

use p3_field::{Field, Powers, TwoAdicField};
use p3_matrix::dense::{RowMajorMatrix, RowMajorMatrixViewMut};
use p3_matrix::Matrix;
use p3_util::log2_strict_usize;

use crate::butterflies::{dif_butterfly, dit_butterfly, twiddle_free_butterfly};
use crate::util::{
    bit_reversed_zero_pad, divide_by_height, reverse_bits, reverse_matrix_index_bits,
    reverse_slice_index_bits,
};
use crate::TwoAdicSubgroupDft;

/// The Bowers G FFT algorithm.
/// See: "Improved Twiddle Access for Fast Fourier Transforms"
#[derive(Default, Clone)]
pub struct Radix2Bowers;

impl<F: TwoAdicField> TwoAdicSubgroupDft<F> for Radix2Bowers {
    fn dft_batch(&self, mut mat: RowMajorMatrix<F>) -> RowMajorMatrix<F> {
        reverse_matrix_index_bits(&mut mat);
        bowers_g(&mut mat.as_view_mut());
        mat
    }

    /// Compute the inverse DFT of each column in `mat`.
    fn idft_batch(&self, mut mat: RowMajorMatrix<F>) -> RowMajorMatrix<F> {
        bowers_g_t(&mut mat.as_view_mut());
        divide_by_height(&mut mat);
        reverse_matrix_index_bits(&mut mat);
        mat
    }

    fn lde_batch(&self, mut mat: RowMajorMatrix<F>, added_bits: usize) -> RowMajorMatrix<F> {
        bowers_g_t(&mut mat.as_view_mut());
        divide_by_height(&mut mat);
        bit_reversed_zero_pad(&mut mat, added_bits);
        bowers_g(&mut mat.as_view_mut());
        mat
    }

    fn coset_lde_batch(
        &self,
        mut mat: RowMajorMatrix<F>,
        added_bits: usize,
        shift: F,
    ) -> RowMajorMatrix<F> {
        let h = mat.height();
        let h_inv = F::from_canonical_usize(h).inverse();

        bowers_g_t(&mut mat.as_view_mut());

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

        bowers_g(&mut mat.as_view_mut());

        mat
    }
}

/// Executes the Bowers G network. This is like a DFT, except it assumes the input is in
/// bit-reversed order.
fn bowers_g<F: TwoAdicField>(mat: &mut RowMajorMatrixViewMut<F>) {
    let h = mat.height();
    let log_h = log2_strict_usize(h);

    let root = F::primitive_root_of_unity(log_h);
    let mut twiddles: Vec<F> = root.powers().take(h / 2).collect();
    reverse_slice_index_bits(&mut twiddles);

    let log_h = log2_strict_usize(mat.height());
    for log_half_block_size in 0..log_h {
        bowers_g_layer(mat, log_half_block_size, &twiddles);
    }
}

/// Executes the Bowers G^T network. This is like an inverse DFT, except we skip rescaling by
/// 1/height, and the output is bit-reversed.
fn bowers_g_t<F: TwoAdicField>(mat: &mut RowMajorMatrixViewMut<F>) {
    let h = mat.height();
    let log_h = log2_strict_usize(h);

    let root_inv = F::primitive_root_of_unity(log_h).inverse();
    let mut twiddles: Vec<F> = root_inv.powers().take(h / 2).collect();
    reverse_slice_index_bits(&mut twiddles);

    let log_h = log2_strict_usize(mat.height());
    for log_half_block_size in (0..log_h).rev() {
        bowers_g_t_layer(mat, log_half_block_size, &twiddles);
    }
}

/// One layer of a Bowers G network. Equivalent to `bowers_g_t_layer` except for the butterfly.
fn bowers_g_layer<F: Field>(
    mat: &mut RowMajorMatrixViewMut<F>,
    log_half_block_size: usize,
    twiddles: &[F],
) {
    let h = mat.height();
    let log_block_size = log_half_block_size + 1;
    let half_block_size = 1 << log_half_block_size;
    let num_blocks = h >> log_block_size;

    // Unroll first iteration with a twiddle factor of 1.
    for hi in 0..half_block_size {
        let lo = hi + half_block_size;
        twiddle_free_butterfly(mat, hi, lo);
    }

    for (block, &twiddle) in (1..num_blocks).zip(&twiddles[1..]) {
        let block_start = block << log_block_size;
        for i in 0..half_block_size {
            let hi = block_start + i;
            let lo = hi + half_block_size;
            dif_butterfly(mat, hi, lo, twiddle);
        }
    }
}

/// One layer of a Bowers G^T network. Equivalent to `bowers_g_layer` except for the butterfly.
fn bowers_g_t_layer<F: Field>(
    mat: &mut RowMajorMatrixViewMut<F>,
    log_half_block_size: usize,
    twiddles: &[F],
) {
    let h = mat.height();
    let log_block_size = log_half_block_size + 1;
    let half_block_size = 1 << log_half_block_size;
    let num_blocks = h >> log_block_size;

    // Unroll first iteration with a twiddle factor of 1.
    for hi in 0..half_block_size {
        let lo = hi + half_block_size;
        twiddle_free_butterfly(mat, hi, lo);
    }

    for (block, &twiddle) in (1..num_blocks).zip(&twiddles[1..]) {
        let block_start = block << log_block_size;
        for i in 0..half_block_size {
            let hi = block_start + i;
            let lo = hi + half_block_size;
            dit_butterfly(mat, hi, lo, twiddle);
        }
    }
}

#[cfg(test)]
mod tests {
    use p3_baby_bear::BabyBear;
    use p3_goldilocks::Goldilocks;

    use crate::radix_2_bowers::Radix2Bowers;
    use crate::testing::*;

    #[test]
    fn dft_matches_naive() {
        test_dft_matches_naive::<BabyBear, Radix2Bowers>();
    }

    #[test]
    fn coset_dft_matches_naive() {
        test_coset_dft_matches_naive::<BabyBear, Radix2Bowers>();
    }

    #[test]
    fn idft_matches_naive() {
        test_idft_matches_naive::<Goldilocks, Radix2Bowers>();
    }

    #[test]
    fn lde_matches_naive() {
        test_lde_matches_naive::<BabyBear, Radix2Bowers>();
    }

    #[test]
    fn coset_lde_matches_naive() {
        test_coset_lde_matches_naive::<BabyBear, Radix2Bowers>();
    }

    #[test]
    fn dft_idft_consistency() {
        test_dft_idft_consistency::<BabyBear, Radix2Bowers>();
    }
}
