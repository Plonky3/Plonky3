use p3_field::Field;
use p3_matrix::Matrix;

use crate::{
    Code, CodeOrFamily, LinearCode, SystematicCode, SystematicCodeOrFamily, SystematicLinearCode,
};

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

#[cfg(test)]
mod tests {
    use alloc::vec::Vec;

    use p3_field::AbstractField;
    use p3_matrix::dense::RowMajorMatrix;
    use p3_mersenne_31::Mersenne31;

    use super::*;

    const CODE_LEN: usize = 2;
    type F = Mersenne31;
    type In = RowMajorMatrix<F>;

    #[test]
    fn test_encode_batch() {
        const WIDTH: usize = 5;
        let in_messages = [1_u16, 2, 3, 4, 5, 6, 7, 8, 9, 10]
            .iter()
            .map(|t| F::from_canonical_u16(*t))
            .collect::<Vec<_>>();
        let identity_code = IdentityCode { len: CODE_LEN };
        let messages = RowMajorMatrix::new(in_messages, WIDTH);
        let encoded_batch =
            <IdentityCode as CodeOrFamily<F, In>>::encode_batch(&identity_code, messages.clone());
        assert_eq!(encoded_batch, messages);

        assert_eq!(
            <IdentityCode as Code<F, In>>::message_len(&identity_code),
            CODE_LEN
        );
        assert_eq!(
            <IdentityCode as Code<F, In>>::codeword_len(&identity_code),
            CODE_LEN
        );
    }
}
