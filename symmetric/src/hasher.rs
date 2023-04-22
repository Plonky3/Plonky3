use alloc::vec::Vec;

pub trait CryptographicHasher<In, Out> {
    fn hash(&self, input: &In) -> Out;
}

#[deprecated] // Just use IterHasher?
pub trait VecToArrHasher<T, const OUT_LEN: usize>:
    CryptographicHasher<Vec<T>, [T; OUT_LEN]>
{
}

pub trait IterHasher<Item, Out> {
    fn hash_iter<I>(&self, input: I) -> Out
    where
        I: IntoIterator<Item = Item>;

    // fn hash_iter_slices<'a, I>(input: I) -> Out
    //     where I: IntoIterator<Item = &'a [Item]>, Item: 'a {
    //     Self::hash_iter(input.into_iter().flatten())
    // }
}
