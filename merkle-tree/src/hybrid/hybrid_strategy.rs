use core::marker::PhantomData;
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
pub struct SimpleHybridHasher<
    H1,
    H2,
    F,
    W1,
    W2,
    const DIGEST_ELEMS_1: usize,
    const DIGEST_ELEMS_2: usize,
> where
    F: Clone,
    H1: CryptographicHasher<F, [W1; DIGEST_ELEMS_1]> + Clone,
    H2: CryptographicHasher<F, [W2; DIGEST_ELEMS_2]> + Clone,
{
    h1: H1,
    h2: H2,
    _marker: PhantomData<(F, W1, W2)>,
}

impl<H1, H2, F, W1, W2, const DIGEST_ELEMS_1: usize, const DIGEST_ELEMS_2: usize>
    SimpleHybridHasher<H1, H2, F, W1, W2, DIGEST_ELEMS_1, DIGEST_ELEMS_2>
where
    F: Clone,
    H1: CryptographicHasher<F, [W1; DIGEST_ELEMS_1]> + Clone,
    H2: CryptographicHasher<F, [W2; DIGEST_ELEMS_2]> + Clone,
{
    pub fn new<P, PW>(h1: H1, h2: H2) -> Self
    where
        F: Clone,
        P: PackedValue<Value = F>,
        PW: PackedValue<Value = W1>,
        H1: CryptographicHasher<F, [W1; DIGEST_ELEMS_1]>, // + CryptographicHasher<P, [PW; DIGEST_ELEMS]>,
        H2: CryptographicHasher<F, [W2; DIGEST_ELEMS_2]>, //+ CryptographicHasher<P, [PW; DIGEST_ELEMS]>,
    {
        Self {
            h1,
            h2,
            _marker: PhantomData,
        }
    }
}

#[derive(Clone)]
pub struct SimpleHybridCompressor<
    C1,
    C2,
    W1,
    W2,
    const DIGEST_ELEMS_1: usize,
    const DIGEST_ELEMS_2: usize,
> where
    C1: Clone,
    C2: Clone,
    C1: PseudoCompressionFunction<[W1; DIGEST_ELEMS_1], 2> + Clone,
    C2: PseudoCompressionFunction<[W2; DIGEST_ELEMS_2], 2> + Clone,
{
    c1: C1,
    c2: C2,
    _marker: PhantomData<(W1, W2)>,
}

impl<C1, C2, W1, W2, const DIGEST_ELEMS_1: usize, const DIGEST_ELEMS_2: usize>
    SimpleHybridCompressor<C1, C2, W1, W2, DIGEST_ELEMS_1, DIGEST_ELEMS_2>
where
    C1: Clone,
    C2: Clone,
    C1: PseudoCompressionFunction<[W1; DIGEST_ELEMS_1], 2> + Clone,
    C2: PseudoCompressionFunction<[W2; DIGEST_ELEMS_2], 2> + Clone,
{
    pub fn new(c1: C1, c2: C2) -> Self {
        Self {
            c1,
            c2,
            _marker: PhantomData,
        }
    }
}

impl<F, W1, W2, H1, H2, const DIGEST_ELEMS_1: usize, const DIGEST_ELEMS_2: usize>
    HybridCryptographicHasher<F, [W1; DIGEST_ELEMS_1]>
    for SimpleHybridHasher<H1, H2, F, W1, W2, DIGEST_ELEMS_1, DIGEST_ELEMS_2>
where
    F: Clone,
    H1: CryptographicHasher<F, [W1; DIGEST_ELEMS_1]>,
    H2: CryptographicHasher<F, [W2; DIGEST_ELEMS_2]>,
    W1: Clone,
    W2: Clone,
    [W1; DIGEST_ELEMS_1]: PackedValue<Value = W2>,
{
    fn hash_iter<I>(&self, input: I, sizes: &[usize], current_size: usize) -> [W1; DIGEST_ELEMS_1]
    where
        I: IntoIterator<Item = F>,
    {
        // TODO
        if sizes.len() == 1 {
            self.h1.hash_iter(input)
        } else {
            // TODO fix
            <[W1; DIGEST_ELEMS_1] as PackedValue>::from_slice(&self.h2.hash_iter(input)).clone()
        }
    }
}

impl<C1, C2, W1, W2, const DIGEST_ELEMS_1: usize, const DIGEST_ELEMS_2: usize>
    HybridPseudoCompressionFunction<[W1; DIGEST_ELEMS_1], 2>
    for SimpleHybridCompressor<C1, C2, W1, W2, DIGEST_ELEMS_1, DIGEST_ELEMS_2>
where
    C1: PseudoCompressionFunction<[W1; DIGEST_ELEMS_1], 2> + Clone,
    C2: PseudoCompressionFunction<[W2; DIGEST_ELEMS_2], 2> + Clone,
    W1: Clone,
    W2: Clone,
    [W1; DIGEST_ELEMS_1]: PackedValue<Value = W2>,
    [W2; DIGEST_ELEMS_2]: PackedValue<Value = W1>,
{
    fn compress(
        &self,
        input: [[W1; DIGEST_ELEMS_1]; 2],
        sizes: &[usize],
        current_size: usize,
    ) -> [W1; DIGEST_ELEMS_1] {
        // TODO
        if sizes.len() == 1 {
            self.c1.compress(input)
        } else {
            let input_w2 = [
                <[W2; DIGEST_ELEMS_2] as PackedValue>::from_slice(&input[0]).clone(),
                <[W2; DIGEST_ELEMS_2] as PackedValue>::from_slice(&input[1]).clone(),
            ];

            <[W1; DIGEST_ELEMS_1] as PackedValue>::from_slice(&self.c2.compress(input_w2)).clone()
        }
    }
}
