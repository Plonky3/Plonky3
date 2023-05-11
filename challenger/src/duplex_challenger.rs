use core::marker::PhantomData;
use p3_field::Field;
use p3_symmetric::permutation::ArrayPermutation;

pub struct DuplexChallenger<F: Field, P: ArrayPermutation<F, WIDTH>, const WIDTH: usize> {
    _permutation: P,
    _phantom_f: PhantomData<F>,
}
