//! A PCS using degree 2 tensor codes, based on BCG20 (https://eprint.iacr.org/2020/1426).

use hypercode::SystematicCode;
use hyperfield::field::Field;
use std::marker::PhantomData;

pub struct TensorPcs<F: Field, C: SystematicCode<F>> {
    _code: C,
    _phantom: PhantomData<F>,
}

// impl<F: Field, C: SystematicCode<F>> PCS<F> for TensorPcs<F, C> {}
