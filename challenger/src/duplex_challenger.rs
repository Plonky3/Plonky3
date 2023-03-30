use core::marker::PhantomData;
use hyperfield::field::Field;
use p3_symmetric::permutation::CryptographicPermutation;

pub struct DuplexChallenger<F: Field, P: CryptographicPermutation<F, WIDTH>, const WIDTH: usize> {
    _permutation: P,
    _phantom_f: PhantomData<F>,
}
