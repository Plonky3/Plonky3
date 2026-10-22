use alloc::vec::Vec;

use p3_air::{Air, BaseAir, WindowAccess};
use p3_field::Field;
use p3_lookup::InteractionBuilder;
use p3_matrix::dense::RowMajorMatrix;

use super::{read, send, write};

/// A challenge-field wire and the base-field wires of its limbs, one value per row.
///
/// Composing rows read the limbs and write the value.
///
/// Preprocessed columns: `active, ext, ext_writes, limb_0 .. limb_{D-1}`.
///
/// Decomposing rows read the value and write the limbs.
///
/// Preprocessed columns: `active, ext, limb_0 .. limb_{D-1}, limb_writes_0 .. limb_writes_{D-1}`.
///
/// Main columns: the limbs, so both sides carry the same numbers and no constraint is needed.
#[derive(Clone, Debug)]
pub struct RecomposeAir<F> {
    /// Whether rows write the challenge-field wire.
    pub(crate) compose: bool,
    /// Limbs of a challenge-field value.
    pub(crate) dimension: usize,
    /// The preprocessed columns.
    pub(crate) preprocessed: RowMajorMatrix<F>,
}

impl<F: Field> BaseAir<F> for RecomposeAir<F> {
    fn width(&self) -> usize {
        self.dimension
    }

    fn preprocessed_trace(&self) -> Option<RowMajorMatrix<F>> {
        Some(self.preprocessed.clone())
    }

    fn preprocessed_width(&self) -> usize {
        if self.compose {
            3 + self.dimension
        } else {
            2 + 2 * self.dimension
        }
    }

    fn main_next_row_columns(&self) -> Vec<usize> {
        Vec::new()
    }

    fn preprocessed_next_row_columns(&self) -> Vec<usize> {
        Vec::new()
    }
}

impl<AB: InteractionBuilder<F: Field>> Air<AB> for RecomposeAir<AB::F> {
    fn eval(&self, builder: &mut AB) {
        let preprocessed = builder.preprocessed().current_slice().to_vec();
        let limbs = builder.main().current_slice().to_vec();
        let d = self.dimension;
        let (active, ext) = (preprocessed[0], preprocessed[1]);
        let limb_expr = || limbs.iter().map(|&limb| limb.into());

        if self.compose {
            send(
                builder,
                ext.into(),
                limb_expr(),
                d,
                write(preprocessed[2].into()),
            );
            for (j, &limb) in limbs.iter().enumerate() {
                send(
                    builder,
                    preprocessed[3 + j].into(),
                    [limb.into()],
                    d,
                    read(active.into()),
                );
            }
        } else {
            send(builder, ext.into(), limb_expr(), d, read(active.into()));
            for (j, &limb) in limbs.iter().enumerate() {
                let (id, writes) = (preprocessed[2 + j], preprocessed[2 + d + j]);
                send(builder, id.into(), [limb.into()], d, write(writes.into()));
            }
        }
    }
}
