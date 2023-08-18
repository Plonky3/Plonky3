//! Reed-Solomon codes.

use std::marker::PhantomData;

use p3_code::{
    Code, CodeOrFamily, LinearCode, SystematicCode, SystematicCodeOrFamily, SystematicLinearCode,
};
use p3_field::Field;
use p3_lde::UndefinedLde;
use p3_matrix::MatrixRows;

/// A Reed-Solomon code based on an `UndefinedLde`.
pub struct UndefinedReedSolomonCode<F, L, In>
where
    F: Field,
    L: UndefinedLde<F, F, In>,
    In: MatrixRows<F>,
{
    lde: L,
    n: usize,
    k: usize,
    _phantom_f: PhantomData<F>,
    _phantom_l: PhantomData<L>,
    _phantom_in: PhantomData<In>,
}

impl<F, L, In> UndefinedReedSolomonCode<F, L, In>
where
    F: Field,
    L: UndefinedLde<F, F, In>,
    In: MatrixRows<F>,
{
    pub fn new(lde: L, n: usize, k: usize) -> Self {
        Self {
            lde,
            n,
            k,
            _phantom_f: PhantomData,
            _phantom_l: PhantomData,
            _phantom_in: PhantomData,
        }
    }
}

impl<F, L, In> CodeOrFamily<F, In> for UndefinedReedSolomonCode<F, L, In>
where
    F: Field,
    In: MatrixRows<F>,
    L: UndefinedLde<F, F, In>,
{
    type Out = L::Out;

    fn encode_batch(&self, messages: In) -> Self::Out {
        self.lde.lde_batch(messages, self.n)
    }
}

impl<F, L, In> Code<F, In> for UndefinedReedSolomonCode<F, L, In>
where
    F: Field,
    L: UndefinedLde<F, F, In>,
    In: MatrixRows<F>,
{
    fn message_len(&self) -> usize {
        self.k
    }

    fn codeword_len(&self) -> usize {
        self.n
    }
}

impl<F, L, In> LinearCode<F, In> for UndefinedReedSolomonCode<F, L, In>
where
    F: Field,
    L: UndefinedLde<F, F, In>,
    In: MatrixRows<F>,
{
}

impl<F, L, In> SystematicCodeOrFamily<F, In> for UndefinedReedSolomonCode<F, L, In>
where
    F: Field,
    L: UndefinedLde<F, F, In>,
    In: MatrixRows<F>,
{
}

impl<F, L, In> SystematicCode<F, In> for UndefinedReedSolomonCode<F, L, In>
where
    F: Field,
    L: UndefinedLde<F, F, In>,
    In: MatrixRows<F>,
{
}

impl<F, L, In> SystematicLinearCode<F, In> for UndefinedReedSolomonCode<F, L, In>
where
    F: Field,
    L: UndefinedLde<F, F, In>,
    In: MatrixRows<F>,
{
}
