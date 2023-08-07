use alloc::vec::Vec;

use p3_field::{Field, TwoAdicField};
use p3_matrix::dense::RowMajorMatrix;
use p3_matrix::Matrix;
use p3_util::log2_strict_usize;

use crate::util::{reverse_bits, reverse_matrix_index_bits};
use crate::TwoAdicSubgroupDFT;

/// The Bowers G^T FFT algorithm.
/// See: "Improved Twiddle Access for Fast Fourier Transforms"
#[derive(Default)]
pub struct Radix2BowersFft;

impl<F: TwoAdicField> TwoAdicSubgroupDFT<F, F> for Radix2BowersFft {
    fn dft_batch(&self, mut mat: RowMajorMatrix<F>) -> RowMajorMatrix<F> {
        let h = mat.height();
        let log_h = log2_strict_usize(h);

        let root = F::primitive_root_of_unity(log_h);
        let twiddles: Vec<F> = root.powers().take(h / 2).collect();

        for log_half_block_size in (0..log_h).rev() {
            bowers_layer(&mut mat, log_half_block_size, &twiddles);
        }
        reverse_matrix_index_bits(&mut mat);
        mat
    }
}

/// One layer of a Bowers G^T network.
fn bowers_layer<F: Field>(mat: &mut RowMajorMatrix<F>, log_half_block_size: usize, twiddles: &[F]) {
    let h = mat.height();
    let log_block_size = log_half_block_size + 1;
    let half_block_size = 1 << log_half_block_size;
    let block_size = 1 << log_block_size;
    let num_blocks = h >> log_block_size;

    for block in 0..num_blocks {
        let block_start = block * block_size;
        let twiddle = twiddles[reverse_bits(block, num_blocks) * half_block_size];
        for butterfly_hi in block_start..block_start + half_block_size {
            let butterfly_lo = butterfly_hi + half_block_size;
            bowers_butterfly(mat, butterfly_hi, butterfly_lo, twiddle);
        }
    }
}

#[inline]
fn bowers_butterfly<F: Field>(mat: &mut RowMajorMatrix<F>, row_1: usize, row_2: usize, twiddle: F) {
    let RowMajorMatrix { values, width } = mat;
    for col in 0..*width {
        let idx_1 = row_1 * *width + col;
        let idx_2 = row_2 * *width + col;
        let val_1 = values[idx_1];
        let val_2 = values[idx_2] * twiddle;
        values[idx_1] = val_1 + val_2;
        values[idx_2] = val_1 - val_2;
    }
}

#[cfg(test)]
mod tests {
    use p3_baby_bear::BabyBear;
    use p3_matrix::dense::RowMajorMatrix;
    use rand::thread_rng;

    use crate::radix_2_bowers::Radix2BowersFft;
    use crate::{NaiveDFT, TwoAdicSubgroupDFT};

    #[test]
    fn consistency() {
        type F = BabyBear;
        let mut rng = thread_rng();
        let mat = RowMajorMatrix::<F>::rand(&mut rng, 64, 3);
        let dft_naive = NaiveDFT.dft_batch(mat.clone());
        let dft_radix_2_bowers = Radix2BowersFft.dft_batch(mat);
        assert_eq!(dft_naive, dft_radix_2_bowers);
    }
}
