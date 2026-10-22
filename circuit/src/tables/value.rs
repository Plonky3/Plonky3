use alloc::vec::Vec;

use p3_air::{Air, BaseAir, WindowAccess};
use p3_field::Field;
use p3_lookup::InteractionBuilder;
use p3_matrix::dense::RowMajorMatrix;

use super::{send, write};

/// Constants and prover-chosen values, one wire per row.
///
/// Preprocessed columns: `id, reads, is_base, is_fixed, fixed_0 .. fixed_{D-1}`.
///
/// Main columns: the limbs of the value.
///
/// A fixed row holds its preprocessed value, and a base row has zero high limbs.
#[derive(Clone, Debug)]
pub struct ValueAir<F> {
    /// Limbs of a challenge-field value.
    pub(crate) dimension: usize,
    /// The preprocessed columns.
    pub(crate) preprocessed: RowMajorMatrix<F>,
}

impl<F: Field> BaseAir<F> for ValueAir<F> {
    fn width(&self) -> usize {
        self.dimension
    }

    fn preprocessed_trace(&self) -> Option<RowMajorMatrix<F>> {
        Some(self.preprocessed.clone())
    }

    fn preprocessed_width(&self) -> usize {
        4 + self.dimension
    }

    fn main_next_row_columns(&self) -> Vec<usize> {
        Vec::new()
    }

    fn preprocessed_next_row_columns(&self) -> Vec<usize> {
        Vec::new()
    }
}

impl<AB: InteractionBuilder<F: Field>> Air<AB> for ValueAir<AB::F> {
    fn eval(&self, builder: &mut AB) {
        let preprocessed = builder.preprocessed().current_slice().to_vec();
        let limbs = builder.main().current_slice().to_vec();
        let [id, reads, is_base, is_fixed] = [0, 1, 2, 3].map(|i| preprocessed[i]);

        // A base-field wire carries zero in every limb after the first.
        for &limb in &limbs[1..] {
            builder.assert_zero(is_base * limb);
        }

        // A fixed wire carries the value the circuit fixes.
        for (&limb, &fixed) in limbs.iter().zip(&preprocessed[4..]) {
            builder.assert_zero(is_fixed * (limb - fixed));
        }

        send(
            builder,
            id.into(),
            limbs.iter().map(|&limb| limb.into()),
            self.dimension,
            write(reads.into()),
        );
    }
}
