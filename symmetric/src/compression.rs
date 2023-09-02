use core::marker::PhantomData;

use crate::hasher::CryptographicHasher;
use crate::permutation::CryptographicPermutation;

/// An `n`-to-1 compression function, like `CompressionFunction`, except that it need only be
/// collision-resistant in a hash tree setting, where the preimage of a non-leaf node must consist
/// of compression outputs.
pub trait PseudoCompressionFunction<T, const N: usize>: Clone {
    fn compress(&self, input: [T; N]) -> T;
}

/// An `N`-to-1 compression function.
pub trait CompressionFunction<T, const N: usize>: PseudoCompressionFunction<T, N> {}

#[derive(Clone)]
pub struct TruncatedPermutation<T, InnerP, const N: usize, const CHUNK: usize, const WIDTH: usize> {
    inner_permutation: InnerP,
    _phantom_t: PhantomData<T>,
}

impl<T, InnerP, const N: usize, const CHUNK: usize, const WIDTH: usize>
    TruncatedPermutation<T, InnerP, N, CHUNK, WIDTH>
{
    pub fn new(inner_permutation: InnerP) -> Self {
        Self {
            inner_permutation,
            _phantom_t: PhantomData,
        }
    }
}

impl<T, InnerP, const N: usize, const CHUNK: usize, const WIDTH: usize>
    PseudoCompressionFunction<[T; CHUNK], N> for TruncatedPermutation<T, InnerP, N, CHUNK, WIDTH>
where
    T: Copy + Default,
    InnerP: CryptographicPermutation<[T; WIDTH]>,
{
    fn compress(&self, input: [[T; CHUNK]; N]) -> [T; CHUNK] {
        debug_assert!(CHUNK * N <= WIDTH);
        let mut pre = [T::default(); WIDTH];
        for i in 0..N {
            pre[i * CHUNK..(i + 1) * CHUNK].copy_from_slice(&input[i]);
        }
        let post = self.inner_permutation.permute(pre);
        post[..CHUNK].try_into().unwrap()
    }
}

#[derive(Clone)]
pub struct CompressionFunctionFromIterHasher<T, H, const N: usize, const CHUNK: usize>
where
    T: Clone,
    H: CryptographicHasher<T, [T; CHUNK]>,
{
    _phantom_t: PhantomData<T>,
    hasher: H,
}

impl<T, H, const N: usize, const CHUNK: usize> PseudoCompressionFunction<[T; CHUNK], N>
    for CompressionFunctionFromIterHasher<T, H, N, CHUNK>
where
    T: Clone,
    H: CryptographicHasher<T, [T; CHUNK]>,
{
    fn compress(&self, input: [[T; CHUNK]; N]) -> [T; CHUNK] {
        self.hasher.hash_iter(input.into_iter().flatten())
    }
}

impl<T, H, const N: usize, const CHUNK: usize> CompressionFunction<[T; CHUNK], N>
    for CompressionFunctionFromIterHasher<T, H, N, CHUNK>
where
    T: Clone,
    H: CryptographicHasher<T, [T; CHUNK]>,
{
}
