use core::marker::PhantomData;

use p3_challenger::FieldChallenger;
use p3_field::{ExtensionField, Field};
use p3_matrix::MatrixRows;

use crate::pcs::UnivariatePcs;

#[allow(dead_code)]
pub struct MultiFromUniPcs<Val, EF, In, U, Challenger>
where
    Val: Field,
    EF: ExtensionField<Val>,
    In: MatrixRows<Val>,
    U: UnivariatePcs<Val, EF, In, Challenger>,
    Challenger: FieldChallenger<Val>,
{
    _uni: U,
    _phantom: PhantomData<(Val, EF, In, Challenger)>,
}

// TODO: Impl PCS, MultivariatePcs
