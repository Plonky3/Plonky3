use crate::{
    CodeFamily, CodeOrFamily, LinearCodeFamily, SystematicCodeFamily, SystematicCodeOrFamily,
    SystematicLinearCode,
};
use alloc::boxed::Box;
use alloc::vec::Vec;
use p3_field::Field;
use p3_matrix::Matrix;

/// A registry of systematic, linear codes for various message sizes.
pub struct SLCodeRegistry<F: Field, In: Matrix<F>, Out: Matrix<F>> {
    /// Ordered by message length, ascending.
    codes: Vec<Box<dyn SystematicLinearCode<F, In, Out = Out>>>,
}

impl<F: Field, In: Matrix<F>, Out: Matrix<F>> SLCodeRegistry<F, In, Out> {
    pub fn new(mut codes: Vec<Box<dyn SystematicLinearCode<F, In, Out = Out>>>) -> Self {
        codes.sort_by_key(|c| c.message_len());
        Self { codes }
    }

    pub fn for_message_len(
        &self,
        message_len: usize,
    ) -> &dyn SystematicLinearCode<F, In, Out = Out> {
        for c in &self.codes {
            if c.message_len() == message_len {
                return &**c;
            }
        }
        panic!("No code found for message length {}", message_len);
    }
}

impl<F: Field, In: Matrix<F>, Out: Matrix<F>> CodeOrFamily<F, In> for SLCodeRegistry<F, In, Out> {
    type Out = Out;

    fn encode_batch(&self, messages: In) -> Self::Out {
        self.for_message_len(messages.height())
            .encode_batch(messages)
    }
}

impl<F: Field, In: Matrix<F>, Out: Matrix<F>> CodeFamily<F, In> for SLCodeRegistry<F, In, Out> {
    /// The next supported message length that is at least `min`.
    fn next_message_len(&self, min: usize) -> Option<usize> {
        for c in &self.codes {
            if c.message_len() >= min {
                return Some(c.message_len());
            }
        }
        None
    }

    fn codeword_len(&self, message_len: usize) -> Option<usize> {
        for c in &self.codes {
            if c.message_len() == message_len {
                return Some(c.message_len());
            }
        }
        None
    }
}

impl<F: Field, In: Matrix<F>, Out: Matrix<F>> SystematicCodeOrFamily<F, In>
    for SLCodeRegistry<F, In, Out>
{
}

impl<F: Field, In: Matrix<F>, Out: Matrix<F>> SystematicCodeFamily<F, In>
    for SLCodeRegistry<F, In, Out>
{
}

impl<F: Field, In: Matrix<F>, Out: Matrix<F>> LinearCodeFamily<F, In>
    for SLCodeRegistry<F, In, Out>
{
}
