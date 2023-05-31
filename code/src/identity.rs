use crate::{LinearCode, SystematicCode, SystematicLinearCode};
use p3_field::Field;
use p3_matrix::dense::{RowMajorMatrixView, RowMajorMatrixViewMut};

/// The trivial code whose encoder is the identity function.
pub struct IdentityCode {
    pub len: usize,
}

impl<F: Field> SystematicCode<F> for IdentityCode {
    fn systematic_len(&self) -> usize {
        self.len
    }

    fn parity_len(&self) -> usize {
        0
    }

    fn write_parity(
        &self,
        _systematic: RowMajorMatrixView<F>,
        _parity: &mut RowMajorMatrixViewMut<F>,
    ) {
        // All done! There are no parity bits.
    }
}

impl<F: Field> LinearCode<F> for IdentityCode {}

impl<F: Field> SystematicLinearCode<F> for IdentityCode {}
