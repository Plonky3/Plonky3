use core::marker::PhantomData;

use p3_challenger::FieldChallenger;
use p3_field::Field;
use p3_matrix::MatrixRows;

use crate::pcs::MultivariatePCS;

pub struct UniFromMultiPCS<F, In, M, Chal>
where
    F: Field,
    In: for<'a> MatrixRows<'a, F>,
    M: MultivariatePCS<F, In, Chal>,
    Chal: FieldChallenger<F>,
{
    _multi: M,
    _phantom_f: PhantomData<F>,
    _phantom_in: PhantomData<In>,
    _phantom_chal: PhantomData<Chal>,
}

// impl<F: Field, M: MultivariatePCS<F>> UnivariatePCS<F> for UniFromMultiPCS<F> {}
