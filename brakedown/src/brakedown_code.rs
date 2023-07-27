use alloc::boxed::Box;

use p3_code::{
    Code, CodeOrFamily, LinearCode, SystematicCode, SystematicCodeOrFamily, SystematicLinearCode,
};
use p3_field::Field;
use p3_matrix::dense::RowMajorMatrix;
use p3_matrix::mul::mul_csr_dense;
use p3_matrix::sparse::CsrMatrix;
use p3_matrix::stack::VerticalPair;
use p3_matrix::{Matrix, MatrixRows};

/// The Spielman-based code described in the Brakedown paper.
pub struct BrakedownCode<F>
where
    F: Field,
{
    pub a: CsrMatrix<F>,
    pub b: CsrMatrix<F>,
    pub inner_code: Box<dyn SystematicLinearCode<F, RowMajorMatrix<F>, Out = RowMajorMatrix<F>>>,
}

impl<F> BrakedownCode<F>
where
    F: Field,
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

impl<F> CodeOrFamily<F, RowMajorMatrix<F>> for BrakedownCode<F>
where
    F: Field,
{
    type Out = RowMajorMatrix<F>;

    fn encode_batch(&self, x: RowMajorMatrix<F>) -> Self::Out {
        let y = mul_csr_dense(&self.a, &x);
        let z = self.inner_code.encode_batch(y);
        let v = mul_csr_dense(&self.b, &z);

        let parity = VerticalPair::new(z, v);
        VerticalPair::new(x, parity).to_row_major_matrix()
    }
}

impl<F> Code<F, RowMajorMatrix<F>> for BrakedownCode<F>
where
    F: Field,
{
    fn message_len(&self) -> usize {
        self.a.width()
    }

    fn codeword_len(&self) -> usize {
        <BrakedownCode<F> as Code<F, RowMajorMatrix<F>>>::message_len(self)
            + <BrakedownCode<F> as SystematicCode<F, RowMajorMatrix<F>>>::parity_len(self)
    }
}

impl<F> SystematicCodeOrFamily<F, RowMajorMatrix<F>> for BrakedownCode<F> where F: Field {}

impl<F> SystematicCode<F, RowMajorMatrix<F>> for BrakedownCode<F>
where
    F: Field,
{
    fn parity_len(&self) -> usize {
        self.y_len() + self.z_parity_len() + self.v_len()
    }
}

impl<F> LinearCode<F, RowMajorMatrix<F>> for BrakedownCode<F> where F: Field {}

impl<F> SystematicLinearCode<F, RowMajorMatrix<F>> for BrakedownCode<F> where F: Field {}
