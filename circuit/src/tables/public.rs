use alloc::vec::Vec;

use p3_air::{Air, AirBuilder, BaseAir, WindowAccess};
use p3_field::Field;
use p3_lookup::InteractionBuilder;
use p3_matrix::dense::RowMajorMatrix;

use super::{send, write};

/// The public inputs, all in the first row.
///
/// Preprocessed columns: `id_i, reads_i` for every input `i`.
///
/// Main columns: the inputs, each bound to its public value on the first row.
#[derive(Clone, Debug)]
pub struct PublicAir<F> {
    /// Limbs of a challenge-field value.
    pub(crate) dimension: usize,
    /// Number of public inputs.
    pub(crate) count: usize,
    /// The preprocessed columns.
    pub(crate) preprocessed: RowMajorMatrix<F>,
}

impl<F: Field> BaseAir<F> for PublicAir<F> {
    fn width(&self) -> usize {
        self.count
    }

    fn preprocessed_trace(&self) -> Option<RowMajorMatrix<F>> {
        Some(self.preprocessed.clone())
    }

    fn preprocessed_width(&self) -> usize {
        2 * self.count
    }

    fn num_public_values(&self) -> usize {
        self.count
    }

    fn main_next_row_columns(&self) -> Vec<usize> {
        Vec::new()
    }

    fn preprocessed_next_row_columns(&self) -> Vec<usize> {
        Vec::new()
    }
}

impl<AB: InteractionBuilder<F: Field>> Air<AB> for PublicAir<AB::F> {
    fn eval(&self, builder: &mut AB) {
        let preprocessed = builder.preprocessed().current_slice().to_vec();
        let values = builder.main().current_slice().to_vec();
        let publics = builder.public_values().to_vec();

        for (i, (&value, &public)) in values.iter().zip(&publics).enumerate() {
            builder.when_first_row().assert_eq(value, public);
            send(
                builder,
                preprocessed[2 * i].into(),
                [value.into()],
                self.dimension,
                write(preprocessed[2 * i + 1].into()),
            );
        }
    }
}
