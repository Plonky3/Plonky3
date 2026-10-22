use alloc::vec::Vec;

use p3_air::{Air, BaseAir, WindowAccess};
use p3_field::Field;
use p3_lookup::InteractionBuilder;
use p3_matrix::dense::RowMajorMatrix;

use super::{read, send, write};
use crate::PermutationChip;

/// Calls of the circuit's permutation, one per row.
///
/// Preprocessed columns: `active`, then the input wires, output wires and output write counts.
///
/// Main columns: the chip's own trace, which reads its inputs and outputs in place.
#[derive(Clone, Debug)]
pub struct PermuteAir<F, P> {
    /// The permutation and its AIR.
    pub(crate) chip: P,
    /// Limbs of a challenge-field value.
    pub(crate) dimension: usize,
    /// The preprocessed columns.
    pub(crate) preprocessed: RowMajorMatrix<F>,
}

impl<F: Field, P: PermutationChip<F>> BaseAir<F> for PermuteAir<F, P> {
    fn width(&self) -> usize {
        self.chip.air().width()
    }

    fn preprocessed_trace(&self) -> Option<RowMajorMatrix<F>> {
        Some(self.preprocessed.clone())
    }

    fn preprocessed_width(&self) -> usize {
        1 + 3 * P::WIDTH
    }

    fn max_constraint_degree(&self) -> Option<usize> {
        self.chip.air().max_constraint_degree()
    }

    fn main_next_row_columns(&self) -> Vec<usize> {
        Vec::new()
    }

    fn preprocessed_next_row_columns(&self) -> Vec<usize> {
        Vec::new()
    }
}

impl<AB, P> Air<AB> for PermuteAir<AB::F, P>
where
    AB: InteractionBuilder<F: Field>,
    P: PermutationChip<AB::F>,
    P::Air: Air<AB>,
{
    fn eval(&self, builder: &mut AB) {
        self.chip.air().eval(builder);

        let preprocessed = builder.preprocessed().current_slice().to_vec();
        let main = builder.main().current_slice().to_vec();
        let width = P::WIDTH;
        let active = preprocessed[0];
        for i in 0..width {
            let input = main[self.chip.input_column(i)];
            let output = main[self.chip.output_column(i)];
            send(
                builder,
                preprocessed[1 + i].into(),
                [input.into()],
                self.dimension,
                read(active.into()),
            );
            send(
                builder,
                preprocessed[1 + width + i].into(),
                [output.into()],
                self.dimension,
                write(preprocessed[1 + 2 * width + i].into()),
            );
        }
    }
}
