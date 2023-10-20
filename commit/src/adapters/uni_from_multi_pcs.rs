use core::marker::PhantomData;

use p3_challenger::FieldChallenger;
use p3_field::{ExtensionField, Field, TwoAdicField};
use p3_matrix::MatrixRows;

use crate::pcs::MultivariatePcs;

pub struct UniFromMultiPcs<Val, Domain, EF, In, M, Challenger>
where
    Val: Field,
    Domain: ExtensionField<Val> + TwoAdicField,
    EF: ExtensionField<Domain>,
    In: MatrixRows<Val>,
    M: MultivariatePcs<Val, Domain, EF, In, Challenger>,
    Challenger: FieldChallenger<Val>,
{
    _multi: M,
    _phantom: PhantomData<(Val, Domain, EF, In, Challenger)>,
}

// impl<F: Field, M: MultivariatePcs<F>> UnivariatePcs<F> for UniFromMultiPcs<F> {}
