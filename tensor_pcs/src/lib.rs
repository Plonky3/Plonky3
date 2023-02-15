//! A PCS using degree 2 tensor codes, based on BCG20 (https://eprint.iacr.org/2020/1426).

use hypercode::SystematicCode;
use hyperfield::field::Field;
use hyperfield::matrix::dense::DenseMatrix;
use std::marker::PhantomData;

pub struct TensorPcs<F: Field, C: SystematicCode<F>> {
    code: C,
    _phantom: PhantomData<F>,
}

impl<F: Field, C: SystematicCode<F>> TensorPcs<F, C> {
    // TODO: Figure out right matrix API for this.
    pub fn commit(&self, _polynomials: DenseMatrix<F>) {
    }
}
