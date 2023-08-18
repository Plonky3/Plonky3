use core::marker::PhantomData;

use p3_challenger::FieldChallenger;
use p3_field::Field;
use p3_matrix::MatrixRows;

use crate::pcs::MultivariatePcs;

pub struct UniFromMultiPcs<F, In, M, Challenger>
where
    F: Field,
    In: MatrixRows<F>,
    M: MultivariatePcs<F, In, Challenger>,
    Challenger: FieldChallenger<F>,
{
    _multi: M,
    _phantom_f: PhantomData<F>,
    _phantom_in: PhantomData<In>,
    _phantom_chal: PhantomData<Challenger>,
}

// impl<F: Field, M: MultivariatePcs<F>> UnivariatePcs<F> for UniFromMultiPcs<F> {}
