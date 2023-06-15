//! This crate contains an implementation of the Spielman-based code described in the Brakedown paper.

#![no_std]

extern crate alloc;

use alloc::boxed::Box;
use p3_code::{
    Code, CodeOrFamily, LinearCode, SystematicCode, SystematicCodeOrFamily, SystematicLinearCode,
};
use p3_field::Field;
use p3_matrix::dense::RowMajorMatrix;
use p3_matrix::mul::mul_csr_dense;
use p3_matrix::sparse::CsrMatrix;
use p3_matrix::stack::VerticalPair;
use p3_matrix::Matrix;

/// The Spielman-based code described in the Brakedown paper.
pub struct BrakedownCode<F, IC>
where
    F: Field,
    IC: SystematicCode<F, RowMajorMatrix<F>>,
{
    pub a: CsrMatrix<F>,
    pub b: CsrMatrix<F>,
    pub inner_code: Box<IC>,
}

impl<F, IC> BrakedownCode<F, IC>
where
    F: Field,
    IC: SystematicCode<F, RowMajorMatrix<F>>,
{
    fn y_len(&self) -> usize {
        self.a.height()
    }

    fn z_parity_len(&self) -> usize {
        self.inner_code.parity_len()
    }

    fn v_len(&self) -> usize {
        self.b.height()
    }
}

impl<F, IC> CodeOrFamily<F, RowMajorMatrix<F>> for BrakedownCode<F, IC>
where
    F: Field,
    IC: SystematicCode<F, RowMajorMatrix<F>>,
{
    type Out = VerticalPair<F, RowMajorMatrix<F>, VerticalPair<F, IC::Out, RowMajorMatrix<F>>>;

    fn encode_batch(&self, x: RowMajorMatrix<F>) -> Self::Out {
        let y = mul_csr_dense(&self.a, &x);
        let z = self.inner_code.encode_batch(y);
        let v = mul_csr_dense(&self.b, &z);

        let parity = VerticalPair::new(z, v);
        VerticalPair::new(x, parity)
    }
}

impl<F, IC> Code<F, RowMajorMatrix<F>> for BrakedownCode<F, IC>
where
    F: Field,
    IC: SystematicCode<F, RowMajorMatrix<F>>,
{
    fn message_len(&self) -> usize {
        self.a.width()
    }

    fn codeword_len(&self) -> usize {
        self.message_len() + self.parity_len()
    }
}

impl<F, IC> SystematicCodeOrFamily<F, RowMajorMatrix<F>> for BrakedownCode<F, IC>
where
    F: Field,
    IC: SystematicCode<F, RowMajorMatrix<F>>,
{
}

impl<F, IC> SystematicCode<F, RowMajorMatrix<F>> for BrakedownCode<F, IC>
where
    F: Field,
    IC: SystematicCode<F, RowMajorMatrix<F>>,
{
    fn parity_len(&self) -> usize {
        self.y_len() + self.z_parity_len() + self.v_len()
    }
}

impl<F, IC> LinearCode<F, RowMajorMatrix<F>> for BrakedownCode<F, IC>
where
    F: Field,
    IC: SystematicCode<F, RowMajorMatrix<F>>,
{
}

impl<F, IC> SystematicLinearCode<F, RowMajorMatrix<F>> for BrakedownCode<F, IC>
where
    F: Field,
    IC: SystematicCode<F, RowMajorMatrix<F>>,
{
}
