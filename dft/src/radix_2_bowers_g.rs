use alloc::vec::Vec;

use p3_field::{Field, TwoAdicField};
use p3_matrix::dense::RowMajorMatrix;
use p3_matrix::Matrix;
use p3_util::log2_strict_usize;

use crate::util::{reverse_bits, reverse_matrix_index_bits};
use crate::{dif_butterfly, TwoAdicSubgroupDft};

/// The Bowers G FFT algorithm.
/// See: "Improved Twiddle Access for Fast Fourier Transforms"
#[derive(Default, Clone)]
pub struct Radix2BowersG;

impl<F: TwoAdicField> TwoAdicSubgroupDft<F> for Radix2BowersG {
    fn dft_batch(&self, mut mat: RowMajorMatrix<F>) -> RowMajorMatrix<F> {
        let h = mat.height();
        let log_h = log2_strict_usize(h);

        let root = F::primitive_root_of_unity(log_h);
        let twiddles: Vec<F> = root.powers().take(h / 2).collect();

        reverse_matrix_index_bits(&mut mat);
        for log_half_block_size in 0..log_h {
            bowers_layer(&mut mat, log_half_block_size, &twiddles);
        }
        mat
    }
}

/// One layer of a Bowers G network.
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
            dif_butterfly(mat, butterfly_hi, butterfly_lo, twiddle);
        }
    }
}

#[cfg(test)]
mod tests {
    use p3_baby_bear::BabyBear;
    use p3_goldilocks::Goldilocks;
    use p3_matrix::dense::RowMajorMatrix;
    use rand::thread_rng;

    use crate::radix_2_bowers_g::Radix2BowersG;
    use crate::{NaiveDft, TwoAdicSubgroupDft};

    #[test]
    fn matches_naive() {
        type F = BabyBear;
        let mut rng = thread_rng();
        let mat = RowMajorMatrix::<F>::rand(&mut rng, 64, 3);
        let dft_naive = NaiveDft.dft_batch(mat.clone());
        let dft_bowers_g = Radix2BowersG.dft_batch(mat);
        assert_eq!(dft_naive, dft_bowers_g);
    }

    #[test]
    fn dft_idft_consistency() {
        type F = Goldilocks;
        let mut rng = thread_rng();
        let original = RowMajorMatrix::<F>::rand(&mut rng, 64, 3);
        let dft = Radix2BowersG.dft_batch(original.clone());
        let idft = Radix2BowersG.idft_batch(dft);
        assert_eq!(original, idft);
    }
}
