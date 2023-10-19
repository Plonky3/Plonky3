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
    _phantom: PhantomData<(Val, Domain, EF, In, Challenger)>,
}

// TODO: Impl PCS, MultivariatePcs
