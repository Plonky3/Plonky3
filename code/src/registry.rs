use crate::{
    CodeFamily, CodeOrFamily, LinearCodeFamily, SystematicCodeFamily, SystematicCodeOrFamily,
    SystematicLinearCode,
};
use alloc::boxed::Box;
use alloc::vec::Vec;
use p3_field::Field;
use p3_matrix::{Matrix, MatrixRows};

/// A registry of systematic, linear codes for various message sizes.
pub struct SLCodeRegistry<F: Field, In: Matrix<F>, Out: Matrix<F>> {
    /// Ordered by message length, ascending.
    codes: Vec<Box<dyn SystematicLinearCode<F, In, Out = Out>>>,
}

impl<F, In, Out> SLCodeRegistry<F, In, Out>
where
    F: Field,
    In: for<'a> MatrixRows<'a, F>,
    Out: for<'a> MatrixRows<'a, F>,
{
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

impl<F, In, Out> CodeOrFamily<F, In> for SLCodeRegistry<F, In, Out>
where
    F: Field,
    In: for<'a> MatrixRows<'a, F>,
    Out: for<'a> MatrixRows<'a, F>,
{
    type Out = Out;

    fn encode_batch(&self, messages: In) -> Self::Out {
        self.for_message_len(messages.height())
            .encode_batch(messages)
    }
}

impl<F, In, Out> CodeFamily<F, In> for SLCodeRegistry<F, In, Out>
where
    F: Field,
    In: for<'a> MatrixRows<'a, F>,
    Out: for<'a> MatrixRows<'a, F>,
{
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

impl<F, In, Out> SystematicCodeOrFamily<F, In> for SLCodeRegistry<F, In, Out>
where
    F: Field,
    In: for<'a> MatrixRows<'a, F>,
    Out: for<'a> MatrixRows<'a, F>,
{
}

impl<F, In, Out> SystematicCodeFamily<F, In> for SLCodeRegistry<F, In, Out>
where
    F: Field,
    In: for<'a> MatrixRows<'a, F>,
    Out: for<'a> MatrixRows<'a, F>,
{
}

impl<F, In, Out> LinearCodeFamily<F, In> for SLCodeRegistry<F, In, Out>
where
    F: Field,
    In: for<'a> MatrixRows<'a, F>,
    Out: for<'a> MatrixRows<'a, F>,
{
}
