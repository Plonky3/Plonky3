use p3_field::PackedValue;
use p3_symmetric::CryptographicHasher;
use p3_symmetric::PseudoCompressionFunction;

// TODO add to doc: closely mimics CryptographicHasher but
pub trait HybridCryptographicHasher<Item: Clone, Out>: Clone {
    fn hash_iter<I>(&self, input: I, sizes: &[usize], current_size: usize) -> Out
    where
        I: IntoIterator<Item = Item>;

    fn hash_iter_slices<'a, I>(&self, input: I, sizes: &[usize], current_size: usize) -> Out
    where
        I: IntoIterator<Item = &'a [Item]>,
        Item: 'a,
    {
        self.hash_iter(input.into_iter().flatten().cloned(), sizes, current_size)
    }

    fn hash_slice(&self, input: &[Item], sizes: &[usize], current_size: usize) -> Out {
        self.hash_iter_slices(core::iter::once(input), sizes, current_size)
    }

    fn hash_item(&self, input: Item, sizes: &[usize], current_size: usize) -> Out {
        self.hash_slice(&[input], sizes, current_size)
    }
}

pub trait HybridPseudoCompressionFunction<T, const N: usize>: Clone {
    fn compress(&self, input: [T; N], sizes: &[usize], current_size: usize) -> T;
}

// Hybrid functions with a single hasher/compressor - functionally equivalent to
// CryptographicHasher/CryptographicPermutation

// TODO this breaks because a downstream crate could implement
// CryptographicHasher for HybridCryptographicHasher (due to the generics)

// impl<T, Item, Out> HybridCryptographicHasher<Item, Out> for T
// where
//     Item: Clone,
//     T: CryptographicHasher<Item, Out>,
// {
//     fn hash_iter<I>(&self, input: I, sizes: &[usize], current_size: usize) -> Out
//     where
//         I: IntoIterator<Item = Item>,
//     {
//         self.hash_iter(input)
//     }
// }

// Hybrid hasher/compressor with two functions, the first of which is chosen
// only at the lowest level and the second of which is chosen elsewhere
#[derive(Clone)]
pub struct SimpleHybridHasher<H1, H2>
where
    H1: Clone,
    H2: Clone,
{
    h1: H1,
    h2: H2,
}

impl<H1, H2> SimpleHybridHasher<H1, H2>
where
    H1: Clone,
    H2: Clone,
{
    pub fn new<F, W, P, PW, const DIGEST_ELEMS: usize>(h1: H1, h2: H2) -> Self
    where
        F: Clone,
        P: PackedValue<Value = F>,
        PW: PackedValue<Value = W>,
        H1: CryptographicHasher<F, [W; DIGEST_ELEMS]> + CryptographicHasher<P, [PW; DIGEST_ELEMS]>,
        H2: CryptographicHasher<F, [W; DIGEST_ELEMS]> + CryptographicHasher<P, [PW; DIGEST_ELEMS]>,
    {
        Self { h1, h2 }
    }
}

#[derive(Clone)]
pub struct SimpleHybridCompressor<C1, C2>
where
    C1: Clone,
    C2: Clone,
{
    c1: C1,
    c2: C2,
}

impl<C1, C2> SimpleHybridCompressor<C1, C2>
where
    C1: Clone,
    C2: Clone,
{
    pub fn new<W, P, PW, const DIGEST_ELEMS: usize>(c1: C1, c2: C2) -> Self
    where
        PW: PackedValue<Value = W>,
        C1: PseudoCompressionFunction<[W; DIGEST_ELEMS], 2>
            + PseudoCompressionFunction<[PW; DIGEST_ELEMS], 2>,
        C2: PseudoCompressionFunction<[W; DIGEST_ELEMS], 2>
            + PseudoCompressionFunction<[PW; DIGEST_ELEMS], 2>,
    {
        Self { c1, c2 }
    }
}

impl<Item, Out, H1, H2> HybridCryptographicHasher<Item, Out> for SimpleHybridHasher<H1, H2>
where
    Item: Clone,
    H1: CryptographicHasher<Item, Out>,
    H2: CryptographicHasher<Item, Out>,
{
    fn hash_iter<I>(&self, input: I, sizes: &[usize], current_size: usize) -> Out
    where
        I: IntoIterator<Item = Item>,
    {
        // TODO
        if sizes.len() == 1 {
            self.h1.hash_iter(input)
        } else {
            self.h2.hash_iter(input)
        }
    }
}

impl<T, const N: usize, C1, C2> HybridPseudoCompressionFunction<T, N>
    for SimpleHybridCompressor<C1, C2>
where
    C1: PseudoCompressionFunction<T, N>,
    C2: PseudoCompressionFunction<T, N>,
{
    fn compress(&self, input: [T; N], sizes: &[usize], current_size: usize) -> T {
        // TODO
        if sizes.len() == 1 {
            self.c1.compress(input)
        } else {
            self.c2.compress(input)
        }
    }
}
