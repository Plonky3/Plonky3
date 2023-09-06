use alloc::vec::Vec;

use p3_field::{Field, TwoAdicField};
use p3_matrix::dense::RowMajorMatrix;
use p3_matrix::Matrix;
use p3_util::log2_strict_usize;

use crate::butterflies::{dit_butterfly, twiddle_free_butterfly};
use crate::util::reverse_matrix_index_bits;
use crate::TwoAdicSubgroupDft;

/// The DIT FFT algorithm.
#[derive(Default, Clone)]
pub struct Radix2Dit;

impl<F: TwoAdicField> TwoAdicSubgroupDft<F> for Radix2Dit {
    fn dft_batch(&self, mut mat: RowMajorMatrix<F>) -> RowMajorMatrix<F> {
        let h = mat.height();
        let log_h = log2_strict_usize(h);

        let root = F::primitive_root_of_unity(log_h);
        let twiddles: Vec<F> = root.powers().take(h / 2).collect();

        // DIT butterfly
        reverse_matrix_index_bits(&mut mat);
        for layer in 0..log_h {
            dit_layer(&mut mat, layer, &twiddles);
        }
        mat
    }
}

/// One layer of a DIT butterfly network.
fn dit_layer<F: Field>(mat: &mut RowMajorMatrix<F>, layer: usize, twiddles: &[F]) {
    let h = mat.height();
    let log_h = log2_strict_usize(h);
    let layer_rev = log_h - 1 - layer;

    let half_block_size = 1 << layer;
    let block_size = half_block_size * 2;

    for j in (0..h).step_by(block_size) {
        // Unroll i=0 case
        let hi = j;
        let lo = hi + half_block_size;
        twiddle_free_butterfly(mat, hi, lo);

        for i in 1..half_block_size {
            let hi = j + i;
            let lo = hi + half_block_size;
            let twiddle = twiddles[i << layer_rev];
            dit_butterfly(mat, hi, lo, twiddle);
        }
    }
}

#[cfg(test)]
mod tests {
    use p3_baby_bear::BabyBear;
    use p3_goldilocks::Goldilocks;
    use p3_matrix::dense::RowMajorMatrix;
    use rand::thread_rng;

    use crate::{NaiveDft, Radix2Dit, TwoAdicSubgroupDft};

    #[test]
    fn matches_naive() {
        type F = BabyBear;
        let mut rng = thread_rng();
        let mat = RowMajorMatrix::<F>::rand(&mut rng, 64, 3);
        let dft_naive = NaiveDft.dft_batch(mat.clone());
        let dft_radix_2_dit = Radix2Dit.dft_batch(mat);
        assert_eq!(dft_naive, dft_radix_2_dit);
    }

    #[test]
    fn dft_idft_consistency() {
        type F = Goldilocks;
        let mut rng = thread_rng();
        let original = RowMajorMatrix::<F>::rand(&mut rng, 64, 3);
        let dft = Radix2Dit.dft_batch(original.clone());
        let idft = Radix2Dit.idft_batch(dft);
        assert_eq!(original, idft);
    }
}
