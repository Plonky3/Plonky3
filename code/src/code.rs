use p3_field::Field;
use p3_matrix::dense::{RowMajorMatrixView, RowMajorMatrixViewMut};

/// A code (in the coding theory sense), or a family thereof.
pub trait CodeOrFamily<F: Field> {
    // TODO: Return codewords instead of mutating? Could return a "chained" matrix.
    fn encode_batch(&self, messages: RowMajorMatrixView<F>, codewords: RowMajorMatrixViewMut<F>);
}

/// A code (in the coding theory sense).
pub trait Code<F: Field>: CodeOrFamily<F> {
    /// The input length of this code's encoder. In other words, the dimension of the code.
    fn message_len(&self) -> usize;

    fn codeword_len(&self) -> usize;
}

/// A family of codes (in the coding theory sense).
pub trait CodeFamily<F: Field>: CodeOrFamily<F> {
    fn next_message_len(&self, len: usize) -> Option<usize>;

    fn codeword_len(&self, len: usize) -> Option<usize>;
}

/// A linear code.
pub trait LinearCode<F: Field>: Code<F> {}

/// A family of linear codes.
pub trait LinearCodeFamily<F: Field>: CodeFamily<F> {}
