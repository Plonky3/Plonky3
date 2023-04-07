//! A framework for codes (in the coding theory sense).

#![no_std]

use p3_field::field::Field;
use p3_matrix::dense::{DenseMatrix, DenseMatrixView, DenseMatrixViewMut};
use p3_matrix::Matrix;

/// A code (in the coding theory sense).
pub trait Code<F: Field> {
    fn encode_batch(&self, messages: DenseMatrixView<F>, codewords: DenseMatrixViewMut<F>);

    fn message_len(&self) -> usize;

    fn codeword_len(&self) -> usize;
}

/// A linear code.
pub trait LinearCode<F: Field>: Code<F> {}

/// A systematic code.
pub trait SystematicCode<F: Field>: Code<F> {
    fn systematic_len(&self) -> usize;

    fn parity_len(&self) -> usize;

    /// Encode a batch of messages, stored in a matrix with a message in each column.
    ///
    /// Since this is a systemic code, this method extends the input matrix to avoid copying.
    fn append_parity(&self, messages: &mut DenseMatrix<F>) {
        assert_eq!(
            messages.height(),
            self.systematic_len(),
            "Wrong message height"
        );
        messages.expand_to_height(self.codeword_len());
        let mut messages_view = messages.as_view_mut();
        let (systematic, mut parity) = messages_view.split_rows(self.systematic_len());
        self.write_parity(systematic.as_view(), &mut parity);
    }

    fn write_parity(&self, systematic: DenseMatrixView<F>, parity: &mut DenseMatrixViewMut<F>);
}

impl<F: Field, S: SystematicCode<F>> Code<F> for S {
    fn encode_batch(&self, messages: DenseMatrixView<F>, mut codewords: DenseMatrixViewMut<F>) {
        let (systematic, mut parity) = codewords.split_rows(self.systematic_len());
        systematic.values.copy_from_slice(messages.values);
        self.write_parity(messages, &mut parity);
    }

    fn message_len(&self) -> usize {
        self.systematic_len()
    }

    fn codeword_len(&self) -> usize {
        self.systematic_len() + self.parity_len()
    }
}

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

    fn write_parity(&self, _systematic: DenseMatrixView<F>, _parity: &mut DenseMatrixViewMut<F>) {
        // All done! There are no parity bits.
    }
}

impl<F: Field> LinearCode<F> for IdentityCode {}
