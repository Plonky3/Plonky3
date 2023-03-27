use alloc::vec::Vec;
use core::marker::PhantomData;
use hyperfield::field::Field;
use p3_symmetric::hash::AlgebraicHash;
use crate::Challenger;

pub struct HashChallenger<F: Field, H: AlgebraicHash<F, OUT_WIDTH>, const OUT_WIDTH: usize> {
    hash: H,
    input_buffer: Vec<F>,
    _phantom_f: PhantomData<F>,
}

impl<F: Field, H: AlgebraicHash<F, OUT_WIDTH>, const OUT_WIDTH: usize> Challenger<F>
for HashChallenger<F, H, OUT_WIDTH>
{
    fn observe_element(&mut self, element: F) {
        self.input_buffer.push(element);
    }
}
