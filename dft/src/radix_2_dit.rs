use alloc::vec::Vec;

use p3_field::{Field, TwoAdicField};
use p3_matrix::dense::RowMajorMatrix;
use p3_matrix::Matrix;
use p3_util::log2_strict_usize;

use crate::util::reverse_matrix_index_bits;
use crate::TwoAdicSubgroupDft;

/// The DIT FFT algorithm.
#[derive(Default, Clone)]
pub struct Radix2DitFft;

impl<F: TwoAdicField> TwoAdicSubgroupDft<F> for Radix2DitFft {
    fn dft_batch(&self, mut mat: RowMajorMatrix<F>) -> RowMajorMatrix<F> {
        let h = mat.height();
        let log_h = log2_strict_usize(h);

        // roots: sequence of root squares from {2^-1, 2^-2, 2^-4, ..., root=2^-N}
        let root = F::primitive_root_of_unity(log_h);
        let roots: Vec<F> = (0..log_h)
            .scan(root, |root_i, _| {
                let ret = *root_i;
                *root_i = root_i.square();
                Some(ret)
            })
            .collect();

        // DIT butterfly
        reverse_matrix_index_bits(&mut mat);
        for (layer, root) in roots.iter().rev().enumerate() {
            dit_layer(&mut mat, layer, *root);
        }
        mat
    }
}

/// One layer of a DIT butterfly.
fn dit_layer<F: Field>(mat: &mut RowMajorMatrix<F>, layer: usize, root: F) {
    let h = mat.height();
    let half_block_size = 1 << layer;
    let block_size = half_block_size * 2;

    for j in (0..h).step_by(block_size) {
        for (k, root_power) in (j..j + half_block_size).zip(root.powers()) {
            let neighbor = k + half_block_size;
            dit_butterfly(mat, k, neighbor, root_power);
        }
    }
}

#[inline]
fn dit_butterfly<F: Field>(mat: &mut RowMajorMatrix<F>, row_1: usize, row_2: usize, root_power: F) {
    let RowMajorMatrix { values, width } = mat;
    for col in 0..*width {
        let idx_1 = row_1 * *width + col;
        let idx_2 = row_2 * *width + col;
        let val_1 = values[idx_1];
        let val_2 = values[idx_2] * root_power;
        values[idx_1] = val_1 + val_2;
        values[idx_2] = val_1 - val_2;
    }
}

#[cfg(test)]
mod tests {
    use p3_baby_bear::BabyBear;
    use p3_goldilocks::Goldilocks;
    use p3_matrix::dense::RowMajorMatrix;
    use rand::thread_rng;

    use crate::{NaiveDft, Radix2DitFft, TwoAdicSubgroupDft};

    #[test]
    fn matches_naive() {
        type F = BabyBear;
        let mut rng = thread_rng();
        let mat = RowMajorMatrix::<F>::rand(&mut rng, 64, 3);
        let dft_naive = NaiveDft.dft_batch(mat.clone());
        let dft_radix_2_dit = Radix2DitFft.dft_batch(mat);
        assert_eq!(dft_naive, dft_radix_2_dit);
    }

    #[test]
    fn dft_idft_consistency() {
        type F = Goldilocks;
        let mut rng = thread_rng();
        let original = RowMajorMatrix::<F>::rand(&mut rng, 64, 3);
        let dft = Radix2DitFft.dft_batch(original.clone());
        let idft = Radix2DitFft.idft_batch(dft);
        assert_eq!(original, idft);
    }
}
