use alloc::vec;

use p3_field::{ExtensionField, Field, TwoAdicField};
use p3_matrix::dense::RowMajorMatrix;
use p3_matrix::Matrix;
use p3_util::log2_strict_usize;

use crate::TwoAdicSubgroupDFT;

pub struct NaiveDFT;

impl<Val, Dom> TwoAdicSubgroupDFT<Val, Dom> for NaiveDFT
where
    Val: Field,
    Dom: ExtensionField<Val> + TwoAdicField,
{
    fn dft_batch(&self, mat: RowMajorMatrix<Val>) -> RowMajorMatrix<Dom> {
        let w = mat.width();
        let h = mat.height();
        let log_h = log2_strict_usize(h);
        let g = Dom::primitive_root_of_unity(log_h);

        let mut res = RowMajorMatrix::new(vec![Dom::ZERO; w * h], w);
        for (res_r, point) in g.powers().take(h).enumerate() {
            for (src_r, point_power) in point.powers().take(h).enumerate() {
                for c in 0..w {
                    res.values[res_r * w + c] += point_power * mat.values[src_r * w + c]
                }
            }
        }
        res
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec;

    use p3_baby_bear::BabyBear;
    use p3_field::AbstractField;
    use p3_matrix::dense::RowMajorMatrix;

    use crate::{NaiveDFT, TwoAdicSubgroupDFT};

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

        let dft = NaiveDFT.dft_batch(mat);
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
}
