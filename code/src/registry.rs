use alloc::boxed::Box;
use alloc::vec::Vec;

use p3_field::Field;
use p3_matrix::{Matrix, MatrixRows};

use crate::{
    CodeFamily, CodeOrFamily, LinearCodeFamily, SystematicCodeFamily, SystematicCodeOrFamily,
    SystematicLinearCode,
};

/// A registry of systematic, linear codes for various message sizes.
pub struct SLCodeRegistry<F: Field, In: Matrix<F>, Out: Matrix<F>> {
    /// Ordered by message length, ascending.
    codes: Vec<Box<dyn SystematicLinearCode<F, In, Out = Out>>>,
}

impl<F, In, Out> SLCodeRegistry<F, In, Out>
where
    F: Field,
    In: MatrixRows<F>,
    Out: MatrixRows<F>,
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
    In: MatrixRows<F>,
    Out: MatrixRows<F>,
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
    In: MatrixRows<F>,
    Out: MatrixRows<F>,
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
    In: MatrixRows<F>,
    Out: MatrixRows<F>,
{
}

impl<F, In, Out> SystematicCodeFamily<F, In> for SLCodeRegistry<F, In, Out>
where
    F: Field,
    In: MatrixRows<F>,
    Out: MatrixRows<F>,
{
}

impl<F, In, Out> LinearCodeFamily<F, In> for SLCodeRegistry<F, In, Out>
where
    F: Field,
    In: MatrixRows<F>,
    Out: MatrixRows<F>,
{
}

#[cfg(test)]
mod tests {
    use p3_field::AbstractField;
    use p3_matrix::dense::RowMajorMatrix;
    use p3_mersenne_31::Mersenne31;

    use super::*;
    use crate::{Code, LinearCode, SystematicCode};

    type F = Mersenne31;
    type In = RowMajorMatrix<F>;
    type Out = RowMajorMatrix<F>;

    struct TestSystematicLinearCode {
        len: usize,
    }

    impl CodeOrFamily<F, In> for TestSystematicLinearCode {
        type Out = In;
        fn encode_batch(&self, messages: In) -> Self::Out {
            // Multiplies every row in messages by `self.len`
            RowMajorMatrix::new(
                messages
                    .values
                    .iter()
                    .map(|x| F::from_canonical_usize(self.len) * (*x))
                    .collect(),
                messages.width(),
            )
        }
    }

    impl CodeFamily<F, In> for TestSystematicLinearCode {
        fn codeword_len(&self, len: usize) -> Option<usize> {
            Some(len)
        }

        fn next_message_len(&self, len: usize) -> Option<usize> {
            Some(len)
        }
    }

    impl Code<F, In> for TestSystematicLinearCode {
        fn codeword_len(&self) -> usize {
            self.len
        }

        fn message_len(&self) -> usize {
            self.len
        }
    }

    impl SystematicCodeOrFamily<F, In> for TestSystematicLinearCode {}

    impl SystematicCodeFamily<F, In> for TestSystematicLinearCode {}

    impl SystematicCode<F, In> for TestSystematicLinearCode {}

    impl LinearCodeFamily<F, In> for TestSystematicLinearCode {}

    impl LinearCode<F, In> for TestSystematicLinearCode {}

    impl SystematicLinearCode<F, In> for TestSystematicLinearCode {}

    macro_rules! create_sl_code_registry {
        ($($len_const:ident),* $(,)?) => {
            {
                let codes = [
                    $(
                        Box::new(TestSystematicLinearCode { len: $len_const })
                            as Box<dyn SystematicLinearCode<F, In, Out = Out>>,
                    )*
                ]
                .into_iter()
                .collect::<Vec<_>>();

                SLCodeRegistry::new(codes)
            }
        };
    }

    macro_rules! should_be_codes {
        ($($len_const:ident),* $(,)?) => {{
            [$(
                Box::new(TestSystematicLinearCode { len: $len_const })
                    as Box<dyn SystematicLinearCode<F, In, Out = Out>>,
            )*]
            .into_iter()
            .collect::<Vec<_>>()
        }};
    }

    fn get_row_major_matrix(width: usize, height: usize, multiplier: usize) -> RowMajorMatrix<F> {
        let row_major_mat_values = (0..width * height)
            .map(|i| F::from_canonical_usize(multiplier) * F::from_canonical_usize(i))
            .collect::<Vec<_>>();

        RowMajorMatrix::new(row_major_mat_values, width)
    }

    #[test]
    fn test_sl_code_registry_initialization() {
        const LEN_1: usize = 5;
        const LEN_2: usize = 3;
        const LEN_3: usize = 1;

        let sl_code_registry = create_sl_code_registry!(LEN_1, LEN_2, LEN_3);
        let should_be_codes = should_be_codes!(LEN_3, LEN_2, LEN_1);

        assert_eq!(
            sl_code_registry
                .codes
                .iter()
                .map(|x| x.as_ref().codeword_len())
                .collect::<Vec<_>>(),
            should_be_codes
                .iter()
                .map(|c| c.as_ref().codeword_len())
                .collect::<Vec<_>>()
        );

        // assert  that `for_message_len`
        assert_eq!(sl_code_registry.for_message_len(LEN_1).message_len(), LEN_1);
        assert_eq!(sl_code_registry.for_message_len(LEN_2).message_len(), LEN_2);
        assert_eq!(sl_code_registry.for_message_len(LEN_3).message_len(), LEN_3);
    }

    #[test]
    #[should_panic]
    fn test_panic_for_message_len() {
        const LEN: usize = 3;
        const NON_EXISTING_MESSAGE_LEN: usize = 1;
        let sl_code_registry = create_sl_code_registry!(LEN);
        // should panic:
        sl_code_registry.for_message_len(NON_EXISTING_MESSAGE_LEN);
    }

    #[test]
    fn test_sl_registry_encode_batch() {
        const REGISTRY_LEN_1: usize = 3;
        const REGISTRY_LEN_2: usize = 5;
        const WIDTH: usize = 3;
        const HEIGHT: usize = 3;
        const MULTIPLIER_1: usize = 1;
        const MULTIPLIER_2: usize = 3;

        let sl_code_registry = create_sl_code_registry!(REGISTRY_LEN_1, REGISTRY_LEN_2);
        let row_major_matrix = get_row_major_matrix(WIDTH, HEIGHT, MULTIPLIER_1);
        let should_be_row_major_matrix = get_row_major_matrix(WIDTH, HEIGHT, MULTIPLIER_2);

        assert_eq!(
            sl_code_registry.encode_batch(row_major_matrix),
            should_be_row_major_matrix
        );
    }

    #[test]
    fn test_sl_registry_code_family_impl() {
        const REGISTRY_LEN_1: usize = 3;
        const REGISTRY_LEN_2: usize = 5;

        let sl_code_registry = create_sl_code_registry!(REGISTRY_LEN_1, REGISTRY_LEN_2);

        // test `next_message_len` method
        assert_eq!(sl_code_registry.next_message_len(1), Some(REGISTRY_LEN_1));
        assert_eq!(sl_code_registry.next_message_len(3), Some(REGISTRY_LEN_1));
        assert_eq!(sl_code_registry.next_message_len(4), Some(REGISTRY_LEN_2));
        assert_eq!(sl_code_registry.next_message_len(5), Some(REGISTRY_LEN_2));
        assert_eq!(sl_code_registry.next_message_len(6), None);

        // test `codeword_len` method
        assert_eq!(sl_code_registry.codeword_len(3), Some(REGISTRY_LEN_1));
        assert_eq!(sl_code_registry.codeword_len(4), None);
        assert_eq!(sl_code_registry.codeword_len(5), Some(REGISTRY_LEN_2));
        assert_eq!(sl_code_registry.codeword_len(10), None);
    }
}
