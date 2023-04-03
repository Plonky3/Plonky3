use alloc::vec::Vec;

pub trait CryptographicHasher<I, O> {
    fn hash(input: &I) -> O;
}

pub trait VecToArrHasher<T, const OUT_LEN: usize>:
    CryptographicHasher<Vec<T>, [T; OUT_LEN]>
{
}
