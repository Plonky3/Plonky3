use core::marker::PhantomData;

use p3_field::Field;
use p3_matrix::MatrixRows;

use crate::pcs::MultilinearPCS;

pub struct UniFromMultiPCS<F, In, M>
where
    F: Field,
    In: for<'a> MatrixRows<'a, F>,
    M: MultilinearPCS<F, In>,
{
    _multi: M,
    _phantom_f: PhantomData<F>,
    _phantom_in: PhantomData<In>,
}

// impl<F: Field, M: MultilinearPCS<F>> UnivariatePCS<F> for UniFromMultiPCS<F> {}
