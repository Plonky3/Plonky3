use crate::{
    Code, CodeOrFamily, LinearCode, SystematicCode, SystematicCodeOrFamily, SystematicLinearCode,
};
use p3_field::Field;
use p3_matrix::Matrix;

/// The trivial code whose encoder is the identity function.
pub struct IdentityCode {
    pub len: usize,
}

impl<F: Field, In: Matrix<F>> CodeOrFamily<F, In> for IdentityCode {
    type Out = In;

    fn encode_batch(&self, messages: In) -> Self::Out {
        messages
    }
}

impl<F: Field, In: Matrix<F>> Code<F, In> for IdentityCode {
    fn message_len(&self) -> usize {
        self.len
    }

    fn codeword_len(&self) -> usize {
        self.len
    }
}

impl<F: Field, In: Matrix<F>> SystematicCodeOrFamily<F, In> for IdentityCode {}

impl<F: Field, In: Matrix<F>> SystematicCode<F, In> for IdentityCode {}

impl<F: Field, In: Matrix<F>> LinearCode<F, In> for IdentityCode {}

impl<F: Field, In: Matrix<F>> SystematicLinearCode<F, In> for IdentityCode {}
