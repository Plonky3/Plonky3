#![no_std]

extern crate alloc;

use core::marker::PhantomData;
use hyperfield::field::{Field, FieldExtension};
use p3_symmetric::hash::AlgebraicHash;
use p3_symmetric::permutation::AlgebraicPermutation;

pub trait Challenger<F: Field> {
    fn observe_element(&mut self, element: F);

    fn observe_extension_element<FE: FieldExtension<Base = F>>(&mut self, _element: FE) {
        todo!()
    }
}

pub struct HashChallenger<F: Field, H: AlgebraicHash<F, OUT_WIDTH>, const OUT_WIDTH: usize> {
    hash: H,
    _phantom_f: PhantomData<F>,
}

pub struct DuplexChallenger<F: Field, P: AlgebraicPermutation<F, WIDTH>, const WIDTH: usize> {
    permutation: P,
    _phantom_f: PhantomData<F>,
}
