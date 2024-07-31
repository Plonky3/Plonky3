use core::marker::PhantomData;

use p3_challenger::FieldChallenger;
use p3_field::{ExtensionField, Field};
use p3_matrix::MatrixRows;

use crate::pcs::MultivariatePcs;

#[allow(dead_code)]
pub struct UniFromMultiPcs<Val, EF, In, M, Challenger>
where
    Val: Field,
    EF: ExtensionField<Val>,
    In: MatrixRows<Val>,
    M: MultivariatePcs<Val, EF, In, Challenger>,
    Challenger: FieldChallenger<Val>,
{
    _multi: M,
    _phantom: PhantomData<(Val, EF, In, Challenger)>,
}

// impl<F: Field, M: MultivariatePcs<F>> UnivariatePcs<F> for UniFromMultiPcs<F> {}
