use core::marker::PhantomData;

use p3_challenger::FieldChallenger;
use p3_field::{ExtensionField, Field, TwoAdicField};
use p3_matrix::MatrixRows;

use crate::pcs::UnivariatePcs;

pub struct MultiFromUniPcs<Val, Domain, EF, In, U, Challenger>
where
    Val: Field,
    Domain: ExtensionField<Val> + TwoAdicField,
    EF: ExtensionField<Domain>,
    In: MatrixRows<Val>,
    U: UnivariatePcs<Val, Domain, EF, In, Challenger>,
    Challenger: FieldChallenger<Val>,
{
    _uni: U,
    _phantom_val: PhantomData<Val>,
    _phantom_dom: PhantomData<Domain>,
    _phantom_ef: PhantomData<EF>,
    _phantom_in: PhantomData<In>,
    _phantom_chal: PhantomData<Challenger>,
}

// TODO: Impl PCS, MultivariatePcs
