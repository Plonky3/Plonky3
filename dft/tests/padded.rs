//! Zero-tail coefficient transforms must retain the full DFT's roots and row ordering.

use p3_baby_bear::BabyBear;
use p3_dft::{Radix2Bowers, Radix2DitParallel, TwoAdicSubgroupDft};
use p3_field::extension::BinomialExtensionField;
use p3_field::{PrimeCharacteristicRing, TwoAdicField};
use p3_goldilocks::Goldilocks;
use p3_matrix::Matrix;
use p3_matrix::dense::RowMajorMatrix;
use rand::SeedableRng;
use rand::distr::{Distribution, StandardUniform};
use rand::rngs::SmallRng;

fn check_padded<F, D>(dft: D)
where
    F: TwoAdicField,
    D: TwoAdicSubgroupDft<F>,
    StandardUniform: Distribution<F>,
{
    let mut rng = SmallRng::seed_from_u64(0xabcdef);
    for log_height in 0..=8 {
        for log_inv_rate in 0..=log_height {
            for width in [1, 3, 4, 17] {
                let mut mat =
                    RowMajorMatrix::<F>::rand(&mut rng, 1 << (log_height - log_inv_rate), width);
                mat.values.resize((1 << log_height) * width, F::ZERO);
                let expected = dft.dft_batch(mat.clone()).to_row_major_matrix();
                let actual = dft
                    .dft_batch_padded(mat, log_inv_rate)
                    .to_row_major_matrix();
                assert_eq!(
                    actual, expected,
                    "height={log_height}, rate={log_inv_rate}, width={width}"
                );
            }
        }
    }
}

#[test]
fn padded_dit_matches_full_dft_across_fields_and_layer_boundaries() {
    check_padded(Radix2DitParallel::<BabyBear>::default());
    check_padded(Radix2DitParallel::<Goldilocks>::default());
    check_padded(Radix2DitParallel::<BinomialExtensionField<BabyBear, 4>>::default());
}

#[test]
fn padded_default_preserves_backend_layout() {
    check_padded::<BabyBear, _>(Radix2Bowers);
}

// A downstream backend may use the inverse primitive root as its forward root.
// The default hook must call that backend, rather than impose another convention.
#[derive(Clone, Default)]
struct InverseRootDft;

impl TwoAdicSubgroupDft<BabyBear> for InverseRootDft {
    type Evaluations = RowMajorMatrix<BabyBear>;

    fn dft_batch(&self, mat: RowMajorMatrix<BabyBear>) -> Self::Evaluations {
        let dft = Radix2DitParallel::default();
        let mut result = dft.dft_batch(mat).to_row_major_matrix();
        let height = result.height();
        for row in 1..height.div_ceil(2) {
            p3_matrix::util::swap_rows(&mut result, row, height - row);
        }
        result
    }
}

#[test]
fn padded_default_preserves_custom_root_convention() {
    check_padded::<BabyBear, _>(InverseRootDft);
}

#[test]
#[should_panic(expected = "padding exceeds matrix height")]
fn padded_dit_rejects_excess_padding() {
    let mat = RowMajorMatrix::new_col(vec![BabyBear::ONE; 4]);
    let _ = Radix2DitParallel::default().dft_batch_padded(mat, 3);
}

#[test]
#[should_panic(expected = "padding exceeds matrix height")]
fn padded_default_rejects_shift_overflow() {
    let mat = RowMajorMatrix::new_col(vec![BabyBear::ONE; 4]);
    let _ = InverseRootDft.dft_batch_padded(mat, usize::MAX);
}

#[test]
#[should_panic]
fn padded_dit_rejects_non_power_of_two_height() {
    let mat = RowMajorMatrix::new_col(vec![BabyBear::ONE; 3]);
    let _ = Radix2DitParallel::default().dft_batch_padded(mat, 1);
}

#[test]
fn padded_dit_evaluates_sparse_coefficients_at_declared_roots() {
    let dft = Radix2DitParallel::<BabyBear>::default();
    let mut values = vec![BabyBear::ZERO; 128 * 2];
    for (row, col, coefficient) in [(0, 0, 2), (2, 0, 5), (1, 1, 7), (5, 1, 3)] {
        values[row * 2 + col] = BabyBear::from_u8(coefficient);
    }
    let evaluations = dft.dft_batch_padded(RowMajorMatrix::new(values, 2), 4);
    for (row, x) in BabyBear::two_adic_generator(7)
        .powers()
        .take(128)
        .enumerate()
    {
        assert_eq!(
            evaluations.get(row, 0).unwrap(),
            BabyBear::from_u8(2) + BabyBear::from_u8(5) * x.square()
        );
        assert_eq!(
            evaluations.get(row, 1).unwrap(),
            BabyBear::from_u8(7) * x + BabyBear::from_u8(3) * x.exp_u64(5)
        );
    }

    let zero = RowMajorMatrix::new(vec![BabyBear::ZERO; 64 * 3], 3);
    assert_eq!(
        dft.dft_batch_padded(zero.clone(), 3).to_row_major_matrix(),
        zero
    );
}
