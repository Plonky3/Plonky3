use core::marker::PhantomData;
use hyperfield::field::Field;
use p3_symmetric::permutation::AlgebraicPermutation;

pub struct DuplexChallenger<F: Field, P: AlgebraicPermutation<F, WIDTH>, const WIDTH: usize> {
    _permutation: P,
    _phantom_f: PhantomData<F>,
}
