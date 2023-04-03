use alloc::vec::Vec;
use core::marker::PhantomData;

pub trait CryptographicHasher<I, O> {
    fn hash(&self, input: I) -> O;
}

pub trait VecToArrHasher<T, const OUT_LEN: usize>:
    CryptographicHasher<Vec<T>, [T; OUT_LEN]>
{
}

/// Converts a hash from [I1] -> O to a hash [I2] -> O.
pub struct HashAdapter<I1, I2, O, Inner, F>
where
    Inner: CryptographicHasher<I1, O>,
    F: Fn(I2) -> I1,
{
    inner: Inner,
    map: F,
    // Grr... https://github.com/rust-lang/rust/issues/23246
    _phantom_i1: PhantomData<I1>,
    _phantom_i2: PhantomData<I2>,
    _phantom_o: PhantomData<O>,
}

impl<I1, I2, O, Inner, F> CryptographicHasher<I2, O> for HashAdapter<I1, I2, O, Inner, F>
where
    Inner: CryptographicHasher<I1, O>,
    F: Fn(I2) -> I1,
{
    fn hash(&self, input: I2) -> O {
        self.inner.hash((self.map)(input))
    }
}
