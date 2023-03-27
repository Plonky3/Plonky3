use crate::MultivariatePCS;
use core::marker::PhantomData;
use hyperfield::field::Field;

pub struct UniFromMultiPCS<F: Field, M: MultivariatePCS<F>> {
    multi: M,
    _phantom_f: PhantomData<F>,
}

// impl<F: Field, M: MultivariatePCS<F>> UnivariatePCS<F> for UniFromMultiPCS<F> {}
