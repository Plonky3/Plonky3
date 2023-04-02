use core::marker::PhantomData;
use p3_field::field::Field;
use crate::pcs::MultivariatePCS;

pub struct UniFromMultiPCS<F: Field, M: MultivariatePCS<F>> {
    _multi: M,
    _phantom_f: PhantomData<F>,
}

// impl<F: Field, M: MultivariatePCS<F>> UnivariatePCS<F> for UniFromMultiPCS<F> {}
