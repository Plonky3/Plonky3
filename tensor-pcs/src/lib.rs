//! A PCS using degree 2 tensor codes, based on BCG20 <https://eprint.iacr.org/2020/1426>.

#![no_std]

use core::marker::PhantomData;
use p3_code::SystematicCode;
use p3_field::Field;

pub struct TensorPCS<F: Field, C: SystematicCode<F>> {
    _code: C,
    _phantom: PhantomData<F>,
}

// impl<F: Field, C: SystematicCode<F>> PCS<F> for TensorPCS<F, C> {}
