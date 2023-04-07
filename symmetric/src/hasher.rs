use alloc::vec::Vec;

pub trait CryptographicHasher<In, Out> {
    fn hash(input: &In) -> Out;
}

pub trait VecToArrHasher<T, const OUT_LEN: usize>:
    CryptographicHasher<Vec<T>, [T; OUT_LEN]>
{
}

pub trait IterHasher<Item, Out> {
    fn hash_iter<I>(input: I) -> Out
    where
        I: IntoIterator<Item = Item>;
}
