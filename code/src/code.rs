use p3_field::Field;
use p3_matrix::MatrixRows;

/// A code (in the coding theory sense), or a family thereof.
pub trait CodeOrFamily<F: Field, In: MatrixRows<F>> {
    type Out: MatrixRows<F>;

    fn encode_batch(&self, messages: In) -> Self::Out;
}

/// A code (in the coding theory sense).
pub trait Code<F: Field, In: MatrixRows<F>>: CodeOrFamily<F, In> {
    /// The input length of this code's encoder. In other words, the dimension of the code.
    fn message_len(&self) -> usize;

    fn codeword_len(&self) -> usize;
}

/// A family of codes (in the coding theory sense).
pub trait CodeFamily<F: Field, In: MatrixRows<F>>: CodeOrFamily<F, In> {
    fn next_message_len(&self, len: usize) -> Option<usize>;

    fn codeword_len(&self, len: usize) -> Option<usize>;
}

/// A linear code.
pub trait LinearCode<F: Field, In: MatrixRows<F>>: Code<F, In> {}

/// A family of linear codes.
pub trait LinearCodeFamily<F: Field, In: MatrixRows<F>>: CodeFamily<F, In> {}
