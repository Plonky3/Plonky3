use alloc::vec::Vec;

use p3_field::{Field, Powers, TwoAdicField};
use p3_matrix::dense::{RowMajorMatrix, RowMajorMatrixViewMut};
use p3_matrix::util::reverse_matrix_index_bits;
use p3_matrix::Matrix;
use p3_maybe_rayon::prelude::*;
use p3_util::{log2_strict_usize, reverse_bits, reverse_slice_index_bits};

use crate::butterflies::{Butterfly, DifButterfly, DitButterfly, TwiddleFreeButterfly};
use crate::util::{bit_reversed_zero_pad, divide_by_height};
use crate::TwoAdicSubgroupDft;

/// The Bowers G FFT algorithm.
/// See: "Improved Twiddle Access for Fast Fourier Transforms"
#[derive(Default, Clone)]
pub struct Radix2Bowers;

impl<F: TwoAdicField> TwoAdicSubgroupDft<F> for Radix2Bowers {
    type Evaluations = RowMajorMatrix<F>;

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

    let root = F::two_adic_generator(log_h);
    let mut twiddles: Vec<_> = root.powers().take(h / 2).map(DifButterfly).collect();
    reverse_slice_index_bits(&mut twiddles);

    let log_h = log2_strict_usize(mat.height());
    for log_half_block_size in 0..log_h {
        butterfly_layer(mat, 1 << log_half_block_size, &twiddles)
    }
}

/// Executes the Bowers G^T network. This is like an inverse DFT, except we skip rescaling by
/// 1/height, and the output is bit-reversed.
fn bowers_g_t<F: TwoAdicField>(mat: &mut RowMajorMatrixViewMut<F>) {
    let h = mat.height();
    let log_h = log2_strict_usize(h);

    let root_inv = F::two_adic_generator(log_h).inverse();
    let mut twiddles: Vec<_> = root_inv.powers().take(h / 2).map(DitButterfly).collect();
    reverse_slice_index_bits(&mut twiddles);

    let log_h = log2_strict_usize(mat.height());
    for log_half_block_size in (0..log_h).rev() {
        butterfly_layer(mat, 1 << log_half_block_size, &twiddles)
    }
}

fn butterfly_layer<F: Field, B: Butterfly<F>>(
    mat: &mut RowMajorMatrixViewMut<F>,
    half_block_size: usize,
    twiddles: &[B],
) {
    mat.par_row_chunks_exact_mut(2 * half_block_size)
        .enumerate()
        .for_each(|(block, mut chunks)| {
            let (mut hi_chunks, mut lo_chunks) = chunks.split_rows_mut(half_block_size);
            hi_chunks
                .par_rows_mut()
                .zip(lo_chunks.par_rows_mut())
                .for_each(|(hi_chunk, lo_chunk)| {
                    if block == 0 {
                        TwiddleFreeButterfly.apply_to_rows(hi_chunk, lo_chunk)
                    } else {
                        twiddles[block].apply_to_rows(hi_chunk, lo_chunk);
                    }
                });
        });
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
    fn coset_idft_matches_naive() {
        test_coset_idft_matches_naive::<BabyBear, Radix2Bowers>();
        test_coset_idft_matches_naive::<Goldilocks, Radix2Bowers>();
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
