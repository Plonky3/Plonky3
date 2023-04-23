use alloc::vec::Vec;
use core::marker::PhantomData;

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

pub struct TruncatingIterHasher<
    InnerH,
    InItem,
    OutItem,
    const ORIGINAL: usize,
    const TRUNCATED: usize,
> {
    _phantom_in: PhantomData<InItem>,
    _phantom_out: PhantomData<OutItem>,
    inner: InnerH,
}

impl<InnerH, InItem, OutItem, const ORIGINAL: usize, const TRUNCATED: usize>
    IterHasher<InItem, [OutItem; TRUNCATED]>
    for TruncatingIterHasher<InnerH, InItem, OutItem, ORIGINAL, TRUNCATED>
where
    OutItem: Copy,
    InnerH: IterHasher<InItem, [OutItem; ORIGINAL]>,
{
    fn hash_iter<I>(&self, input: I) -> [OutItem; TRUNCATED]
    where
        I: IntoIterator<Item = InItem>,
    {
        self.inner.hash_iter(input)[..TRUNCATED].try_into().unwrap()
    }
}
