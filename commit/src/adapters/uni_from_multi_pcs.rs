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
    _phantom_val: PhantomData<Val>,
    _phantom_domain: PhantomData<Domain>,
    _phantom_ef: PhantomData<EF>,
    _phantom_in: PhantomData<In>,
    _phantom_chal: PhantomData<Challenger>,
}

// impl<F: Field, M: MultivariatePcs<F>> UnivariatePcs<F> for UniFromMultiPcs<F> {}
