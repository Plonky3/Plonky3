use alloc::vec::Vec;

use p3_air::{Air, BaseAir, WindowAccess};
use p3_field::{Field, PrimeCharacteristicRing};
use p3_lookup::InteractionBuilder;
use p3_matrix::dense::RowMajorMatrix;
use p3_uni_stark::SubAirBuilder;

use super::{WIRE_BUS, message, read, send, write};
use crate::PermutationChip;

/// Calls of the circuit's permutation, one per row.
///
/// Preprocessed columns: `active`, `merkle`, `conditional` and the bit wire.
///
/// Then the input wires, the output wires and the output write counts.
///
/// Main columns: the chip's own trace, which reads its inputs and outputs in place.
///
/// Then the bit, and the flag `keep = conditional * (1 - bit)`.
///
/// A Merkle row keeps the sibling in its own inputs.
///
/// It reads the running digest from the side the bit names.
///
/// A conditional row writes its inputs instead of its outputs where `keep` is set.
#[derive(Clone, Debug)]
pub struct PermuteAir<F, P> {
    /// The permutation and its AIR.
    pub(crate) chip: P,
    /// Limbs of a challenge-field value.
    pub(crate) dimension: usize,
    /// Digest size of a Merkle row, or zero when the table has none.
    pub(crate) digest: usize,
    /// The preprocessed columns.
    pub(crate) preprocessed: RowMajorMatrix<F>,
}

impl<F: Field, P: PermutationChip<F>> BaseAir<F> for PermuteAir<F, P> {
    fn width(&self) -> usize {
        self.chip.air().width() + 2
    }

    fn preprocessed_trace(&self) -> Option<RowMajorMatrix<F>> {
        Some(self.preprocessed.clone())
    }

    fn preprocessed_width(&self) -> usize {
        4 + 3 * P::WIDTH
    }

    fn max_constraint_degree(&self) -> Option<usize> {
        self.chip
            .air()
            .max_constraint_degree()
            .map(|degree| degree.max(3))
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
    P::Air: for<'b> Air<SubAirBuilder<'b, AB, P::Air, AB::Var>>,
{
    fn eval(&self, builder: &mut AB) {
        let chip_width = self.chip.air().width();
        self.chip
            .air()
            .eval(&mut SubAirBuilder::<AB, P::Air, AB::Var>::new(
                builder,
                0..chip_width,
            ));

        let preprocessed = builder.preprocessed().current_slice().to_vec();
        let main = builder.main().current_slice().to_vec();
        let (width, size) = (P::WIDTH, self.digest);
        let (active, merkle, conditional, bit_id) = (
            preprocessed[0],
            preprocessed[1],
            preprocessed[2],
            preprocessed[3],
        );
        let (bit, keep) = (main[chip_width], main[chip_width + 1]);
        let input = |i: usize| -> AB::Expr { main[self.chip.input_column(i)].into() };

        // A Merkle or conditional row reads its bit, which is boolean.
        let reads_bit: AB::Expr = merkle + conditional;
        builder.assert_zero(reads_bit.clone() * bit * (bit - AB::Expr::ONE));
        send(
            builder,
            bit_id.into(),
            [bit.into()],
            self.dimension,
            read(reads_bit),
        );
        builder.assert_eq(keep, conditional * (AB::Expr::ONE - bit));

        // The running digest is the left child where the bit is zero, and the right child otherwise.
        let right: AB::Expr = merkle * bit;
        for i in 0..size {
            let id: AB::Expr = preprocessed[4 + i].into();
            builder.push_exclusive_interaction(
                WIRE_BUS,
                [
                    (
                        active - right.clone(),
                        read(AB::Expr::ONE),
                        message(id.clone(), [input(i)], self.dimension),
                    ),
                    (
                        right.clone(),
                        read(AB::Expr::ONE),
                        message(id, [input(size + i)], self.dimension),
                    ),
                ],
            );
        }

        // The other inputs are read on plain rows; on a Merkle row they hold the sibling, then zeros.
        let plain: AB::Expr = active - merkle;
        for i in size..width {
            if i >= 2 * size {
                builder.assert_zero(merkle * input(i));
            }
            send(
                builder,
                preprocessed[4 + i].into(),
                [input(i)],
                self.dimension,
                read(plain.clone()),
            );
        }

        // Each output wire carries the permuted lane, or the input lane where `keep` is set.
        for i in 0..width {
            let output = main[self.chip.output_column(i)];
            let id: AB::Expr = preprocessed[4 + width + i].into();
            let count: AB::Expr = preprocessed[4 + 2 * width + i].into();
            builder.push_exclusive_interaction(
                WIRE_BUS,
                [
                    (
                        AB::Expr::ONE - keep,
                        write(count.clone()),
                        message(id.clone(), [output.into()], self.dimension),
                    ),
                    (
                        keep.into(),
                        write(count),
                        message(id, [input(i)], self.dimension),
                    ),
                ],
            );
        }
    }
}
