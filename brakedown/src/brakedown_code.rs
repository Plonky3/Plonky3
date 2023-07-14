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

impl<F, IC, In> CodeOrFamily<F, In> for BrakedownCode<F, IC>
where
    F: Field,
    IC: SystematicCode<F, RowMajorMatrix<F>>,
    IC::Out: Sync,
    In: for<'a> MatrixRows<'a, F> + Sync,
{
    type Out = VerticalPair<F, In, VerticalPair<F, IC::Out, RowMajorMatrix<F>>>;

    fn encode_batch(&self, x: In) -> Self::Out {
        let y = mul_csr_dense(&self.a, &x);
        let z = self.inner_code.encode_batch(y);
        let v = mul_csr_dense(&self.b, &z);

        let parity = VerticalPair::new(z, v);
        VerticalPair::new(x, parity)
    }
}

impl<F, IC, In> Code<F, In> for BrakedownCode<F, IC>
where
    F: Field,
    IC: SystematicCode<F, RowMajorMatrix<F>>,
    IC::Out: Sync,
    In: for<'a> MatrixRows<'a, F> + Sync,
{
    fn message_len(&self) -> usize {
        self.a.width()
    }

    fn codeword_len(&self) -> usize {
        <BrakedownCode<F, IC> as Code<F, In>>::message_len(self)
            + <BrakedownCode<F, IC> as SystematicCode<F, In>>::parity_len(self)
    }
}

impl<F, IC, In> SystematicCodeOrFamily<F, In> for BrakedownCode<F, IC>
where
    F: Field,
    IC: SystematicCode<F, RowMajorMatrix<F>>,
    IC::Out: Sync,
    In: for<'a> MatrixRows<'a, F> + Sync,
{
}

impl<F, IC, In> SystematicCode<F, In> for BrakedownCode<F, IC>
where
    F: Field,
    IC: SystematicCode<F, RowMajorMatrix<F>>,
    IC::Out: Sync,
    In: for<'a> MatrixRows<'a, F> + Sync,
{
    fn parity_len(&self) -> usize {
        self.y_len() + self.z_parity_len() + self.v_len()
    }
}

impl<F, IC, In> LinearCode<F, In> for BrakedownCode<F, IC>
where
    F: Field,
    IC: SystematicCode<F, RowMajorMatrix<F>>,
    IC::Out: Sync,
    In: for<'a> MatrixRows<'a, F> + Sync,
{
}

impl<F, IC, In> SystematicLinearCode<F, In> for BrakedownCode<F, IC>
where
    F: Field,
    IC: SystematicCode<F, RowMajorMatrix<F>>,
    IC::Out: Sync,
    In: for<'a> MatrixRows<'a, F> + Sync,
{
}
