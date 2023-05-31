use crate::SystematicLinearCode;
use alloc::boxed::Box;
use alloc::vec::Vec;
use p3_field::Field;
use p3_matrix::dense::RowMajorMatrix;
use p3_matrix::Matrix;

/// A registry of systematic, linear codes for various message sizes.
pub struct SLCodeRegistry<F: Field> {
    codes: Vec<Box<dyn SystematicLinearCode<F>>>,
}

impl<F: Field> SLCodeRegistry<F> {
    pub fn new(codes: Vec<Box<dyn SystematicLinearCode<F>>>) -> Self {
        Self { codes }
    }

    pub fn for_message_len(&self, message_len: usize) -> &dyn SystematicLinearCode<F> {
        for c in &self.codes {
            if c.message_len() == message_len {
                return &**c;
            }
        }
        panic!("No code found for message length {}", message_len);
    }

    pub fn append_parity(&self, messages: &mut RowMajorMatrix<F>) {
        self.for_message_len(messages.height())
            .append_parity(messages);
    }
}
