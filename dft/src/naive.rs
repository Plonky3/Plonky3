use alloc::vec;

use p3_field::TwoAdicField;
use p3_matrix::dense::RowMajorMatrix;
use p3_matrix::Matrix;
use p3_util::log2_strict_usize;

use crate::util::{divide_by_height, swap_rows};
use crate::FourierTransform;

#[derive(Default, Clone)]
pub struct NaiveDft;

impl<F: TwoAdicField> FourierTransform<F> for NaiveDft {
    type Range = F;

    fn dft_batch(&self, mat: RowMajorMatrix<F>) -> RowMajorMatrix<F> {
        let w = mat.width();
        let h = mat.height();
        let log_h = log2_strict_usize(h);
        let g = F::primitive_root_of_unity(log_h);

        let mut res = RowMajorMatrix::new(vec![F::ZERO; w * h], w);
        for (res_r, point) in g.powers().take(h).enumerate() {
            for (src_r, point_power) in point.powers().take(h).enumerate() {
                for c in 0..w {
                    res.values[res_r * w + c] += point_power * mat.values[src_r * w + c]
                }
            }
        }
        res
    }

    fn idft_batch(&self, mat: RowMajorMatrix<F>) -> RowMajorMatrix<F> {
        let mut dft = self.dft_batch(mat);
        let h = dft.height();

        divide_by_height(&mut dft);

        for row in 1..h / 2 {
            swap_rows(&mut dft, row, h - row);
        }

        dft
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

    use crate::{FourierTransform, NaiveDft};

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
                F::ZERO,
                F::from_canonical_u8(4),
                F::from_canonical_u8(3),
                F::ZERO,
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
                    F::ZERO,
                    F::ONE,
                    F::NEG_ONE,
                    F::ZERO,
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
