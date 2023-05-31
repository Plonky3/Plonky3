use p3_field::Field;
use p3_matrix::dense::{RowMajorMatrixView, RowMajorMatrixViewMut};

/// A code (in the coding theory sense).
pub trait Code<F: Field> {
    fn encode_batch(&self, messages: RowMajorMatrixView<F>, codewords: RowMajorMatrixViewMut<F>);

    fn message_len(&self) -> usize;

    fn codeword_len(&self) -> usize;
}

/// A linear code.
pub trait LinearCode<F: Field>: Code<F> {}
