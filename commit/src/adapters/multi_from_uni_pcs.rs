use core::marker::PhantomData;

use p3_challenger::FieldChallenger;
use p3_field::{ExtensionField, Field, TwoAdicField};
use p3_matrix::MatrixRows;

use crate::pcs::UnivariatePcs;

pub struct MultiFromUniPcs<Val, Domain, In, U, Challenger>
where
    Val: Field,
    Domain: ExtensionField<Val> + TwoAdicField,
    In: MatrixRows<Val>,
    U: UnivariatePcs<Val, Domain, In, Challenger>,
    Challenger: FieldChallenger<Val>,
{
    _uni: U,
    _phantom_val: PhantomData<Val>,
    _phantom_dom: PhantomData<Domain>,
    _phantom_in: PhantomData<In>,
    _phantom_chal: PhantomData<Challenger>,
}

// TODO: Impl PCS, MultivariatePcs
