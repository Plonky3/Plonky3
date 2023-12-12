use alloc::vec;

use p3_field::TwoAdicField;
use p3_matrix::dense::RowMajorMatrix;
use p3_matrix::Matrix;
use p3_util::log2_strict_usize;

use crate::{reverse_matrix_index_bits, TwoAdicSubgroupDft};

#[derive(Default, Clone)]
pub struct NaiveDft;

impl<F: TwoAdicField> TwoAdicSubgroupDft<F> for NaiveDft {
    fn dft_batch_bitrev<const IN_BITREV: bool, const OUT_BITREV: bool>(
        &self,
        mut mat: RowMajorMatrix<F>,
    ) -> RowMajorMatrix<F> {
        let w = mat.width();
        let h = mat.height();
        let log_h = log2_strict_usize(h);
        let g = F::two_adic_generator(log_h);

        if IN_BITREV {
            reverse_matrix_index_bits(&mut mat);
        }

        let mut res = RowMajorMatrix::new(vec![F::zero(); w * h], w);
        for (res_r, point) in g.powers().take(h).enumerate() {
            for (src_r, point_power) in point.powers().take(h).enumerate() {
                for c in 0..w {
                    res.values[res_r * w + c] += point_power * mat.values[src_r * w + c]
                }
            }
        }

        if OUT_BITREV {
            reverse_matrix_index_bits(&mut res);
        }
        res
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec;

    use p3_baby_bear::BabyBear;
    use p3_field::AbstractField;
    use p3_goldilocks::Goldilocks;
    use p3_matrix::dense::RowMajorMatrix;
    use rand::thread_rng;

    use crate::{NaiveDft, TwoAdicSubgroupDft};

    #[test]
    fn basic() {
        type F = BabyBear;

        // A few polynomials:
        // 5 + 4x
        // 2 + 3x
        // 0
        let mat = RowMajorMatrix::new(
            vec![
                F::from_canonical_u8(5),
                F::from_canonical_u8(2),
                F::zero(),
                F::from_canonical_u8(4),
                F::from_canonical_u8(3),
                F::zero(),
            ],
            3,
        );

        let dft = NaiveDft.dft_batch(mat);
        // Expected evaluations on {1, -1}:
        // 9, 1
        // 5, -1
        // 0, 0
        assert_eq!(
            dft,
            RowMajorMatrix {
                values: vec![
                    F::from_canonical_u8(9),
                    F::from_canonical_u8(5),
                    F::zero(),
                    F::one(),
                    F::neg_one(),
                    F::zero(),
                ],
                width: 3,
            }
        )
    }

    #[test]
    fn dft_idft_consistency() {
        type F = Goldilocks;
        let mut rng = thread_rng();
        let original = RowMajorMatrix::<F>::rand(&mut rng, 8, 3);
        let dft = NaiveDft.dft_batch(original.clone());
        let idft = NaiveDft.idft_batch(dft);
        assert_eq!(original, idft);
    }
}
