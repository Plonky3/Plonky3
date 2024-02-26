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
    In: MatrixRows<F> + Sync,
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
    In: MatrixRows<F> + Sync,
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
    In: MatrixRows<F> + Sync,
{
}

impl<F, IC, In> SystematicCode<F, In> for BrakedownCode<F, IC>
where
    F: Field,
    IC: SystematicCode<F, RowMajorMatrix<F>>,
    IC::Out: Sync,
    In: MatrixRows<F> + Sync,
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
    In: MatrixRows<F> + Sync,
{
}

impl<F, IC, In> SystematicLinearCode<F, In> for BrakedownCode<F, IC>
where
    F: Field,
    IC: SystematicCode<F, RowMajorMatrix<F>>,
    IC::Out: Sync,
    In: MatrixRows<F> + Sync,
{
}

#[cfg(test)]
#[allow(deprecated)] // TODO: remove when `p3_lde::NaiveUndefinedLde` is gone
mod tests {
    use p3_mersenne_31::Mersenne31;
    use rand::SeedableRng;
    use rand_chacha::ChaCha20Rng;

    use super::*;
    use crate::macros::{brakedown, brakedown_to_rs};

    type F = Mersenne31;
    type Mat = RowMajorMatrix<F>;

    #[test]
    fn test_brakedown_methods() {
        let brakedown = brakedown!(237, 29, 11, 41, 60, 15, brakedown_to_rs!(29, 4, 0, 5, 7, 0));

        assert_eq!(brakedown.y_len(), 29);
        assert_eq!(
            brakedown.z_parity_len(),
            <BrakedownCode<F, _> as Code<F, Mat>>::codeword_len(&brakedown.inner_code)
                - <BrakedownCode<F, _> as Code<F, Mat>>::message_len(&brakedown.inner_code)
        );
        assert_eq!(brakedown.v_len(), 60);
    }

    #[test]
    fn test_brakedown_systematic_code_impl() {
        let brakedown = brakedown!(237, 29, 11, 41, 60, 15, brakedown_to_rs!(29, 4, 0, 5, 7, 0));
        assert_eq!(
            <BrakedownCode<F, _> as SystematicCode<F, Mat>>::parity_len(&brakedown),
            brakedown.y_len() + brakedown.z_parity_len() + brakedown.v_len()
        );
    }

    #[test]
    fn test_brakedown_code_impl() {
        let brakedown = brakedown!(237, 29, 11, 41, 60, 15, brakedown_to_rs!(29, 4, 0, 5, 7, 0));
        assert_eq!(
            <BrakedownCode<F, _> as Code<F, Mat>>::message_len(&brakedown),
            brakedown.a.width()
        );
        assert_eq!(
            <BrakedownCode<F, _> as Code<F, Mat>>::codeword_len(&brakedown),
            <BrakedownCode<F, _> as Code<F, Mat>>::message_len(&brakedown)
                + <BrakedownCode<F, _> as SystematicCode<F, Mat>>::parity_len(&brakedown)
        );
    }
}
