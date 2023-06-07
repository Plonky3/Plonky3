use crate::{Code, CodeFamily, CodeOrFamily, LinearCode};
use p3_field::Field;
use p3_matrix::dense::{RowMajorMatrixView, RowMajorMatrixViewMut};
use p3_matrix::Matrix;

/// A systematic code, or a family thereof.
pub trait SystematicCodeOrFamily<F: Field>: CodeOrFamily<F> {
    /// Encode a batch of messages, stored in a matrix with a message in each column.
    fn write_parity(
        &self,
        systematic: RowMajorMatrixView<F>,
        parity: &mut RowMajorMatrixViewMut<F>,
    );
}

/// A systematic code.
pub trait SystematicCode<F: Field>: SystematicCodeOrFamily<F> + Code<F> {
    fn systematic_len(&self) -> usize;

    fn parity_len(&self) -> usize;
}

pub trait SystematicLinearCode<F: Field>: SystematicCode<F> + LinearCode<F> {}

/// A family of systematic codes.
pub trait SystematicCodeFamily<F: Field>: SystematicCodeOrFamily<F> + CodeFamily<F> {}

impl<F: Field, S: SystematicCodeOrFamily<F>> CodeOrFamily<F> for S {
    fn encode_batch(
        &self,
        messages: RowMajorMatrixView<F>,
        mut codewords: RowMajorMatrixViewMut<F>,
    ) {
        let (systematic, mut parity) = codewords.split_rows(messages.height());
        systematic.values.copy_from_slice(messages.values);
        self.write_parity(messages, &mut parity);
    }
}

impl<F: Field, S: SystematicCode<F>> Code<F> for S {
    fn message_len(&self) -> usize {
        self.systematic_len()
    }

    fn codeword_len(&self) -> usize {
        self.systematic_len() + self.parity_len()
    }
}
