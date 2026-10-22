use alloc::vec::Vec;
use core::marker::PhantomData;

use p3_air::{Air, BaseAir, WindowAccess};
use p3_field::Field;
use p3_lookup::InteractionBuilder;
use p3_matrix::dense::RowMajorMatrix;

use super::{CircuitExtension, read, send, write};

/// `out = a * b + c`, one operation per row.
///
/// Preprocessed columns: `active, a, b, c, out, out_writes, out_read`.
///
/// Main columns: the limbs of `a`, `b`, `c` and `out`, one limb each over the base field.
///
/// A row whose `out_read` is set reads `out` instead of writing it, which asserts the equation.
#[derive(Clone, Debug)]
pub struct AluAir<F, EF> {
    /// Whether the operands are challenge-field values.
    pub(crate) ext: bool,
    /// Limbs of a challenge-field value.
    pub(crate) dimension: usize,
    /// The preprocessed columns.
    pub(crate) preprocessed: RowMajorMatrix<F>,
    /// Marker for the challenge field.
    pub(crate) _ef: PhantomData<EF>,
}

impl<F, EF> AluAir<F, EF> {
    /// Limbs of each operand.
    pub(crate) const fn limbs(&self) -> usize {
        if self.ext { self.dimension } else { 1 }
    }
}

impl<F: Field, EF: Sync> BaseAir<F> for AluAir<F, EF> {
    fn width(&self) -> usize {
        4 * self.limbs()
    }

    fn preprocessed_trace(&self) -> Option<RowMajorMatrix<F>> {
        Some(self.preprocessed.clone())
    }

    fn preprocessed_width(&self) -> usize {
        7
    }

    fn main_next_row_columns(&self) -> Vec<usize> {
        Vec::new()
    }

    fn preprocessed_next_row_columns(&self) -> Vec<usize> {
        Vec::new()
    }
}

impl<AB, EF> Air<AB> for AluAir<AB::F, EF>
where
    AB: InteractionBuilder<F: Field>,
    EF: CircuitExtension<AB::F>,
{
    fn eval(&self, builder: &mut AB) {
        let preprocessed = builder.preprocessed().current_slice().to_vec();
        let main = builder.main().current_slice().to_vec();
        let n = self.limbs();
        let operand = |k: usize| -> Vec<AB::Expr> {
            main[k * n..(k + 1) * n].iter().map(|&x| x.into()).collect()
        };
        let (a, b, c, out) = (operand(0), operand(1), operand(2), operand(3));

        // out = a * b + c, limb by limb in the operands' field.
        let product = if self.ext {
            EF::mul_limbs(&a, &b)
        } else {
            alloc::vec![a[0].clone() * b[0].clone()]
        };
        for ((out, product), c) in out.iter().zip(product).zip(&c) {
            builder.assert_eq(out.clone(), product + c.clone());
        }

        let [active, a_id, b_id, c_id, out_id, out_writes, out_read] =
            core::array::from_fn(|i| preprocessed[i].into());
        for (id, limbs) in [(a_id, a), (b_id, b), (c_id, c)] {
            send(builder, id, limbs, self.dimension, read(active.clone()));
        }
        send(
            builder,
            out_id.clone(),
            out.clone(),
            self.dimension,
            write(out_writes),
        );
        send(builder, out_id, out, self.dimension, read(out_read));
    }
}
