use crate::{Code, CodeFamily, CodeOrFamily, LinearCode};
use p3_field::Field;
use p3_matrix::Matrix;

/// A systematic code, or a family thereof.
pub trait SystematicCodeOrFamily<F: Field, In: Matrix<F>>: CodeOrFamily<F, In> {}

/// A systematic code.
pub trait SystematicCode<F: Field, In: Matrix<F>>:
    SystematicCodeOrFamily<F, In> + Code<F, In>
{
    fn parity_len(&self) -> usize {
        self.codeword_len() - self.message_len()
    }
}

pub trait SystematicLinearCode<F: Field, In: Matrix<F>>:
    SystematicCode<F, In> + LinearCode<F, In>
{
}

/// A family of systematic codes.
pub trait SystematicCodeFamily<F: Field, In: Matrix<F>>:
    SystematicCodeOrFamily<F, In> + CodeFamily<F, In>
{
}
