use crate::hasher::CryptographicHasher;
use crate::permutation::CryptographicPermutation;
use core::marker::PhantomData;
use itertools::Itertools;

/// An `n`-to-1 compression function, like `CompressionFunction`, except that it need only be
/// collision-resistant in a hash tree setting, where the preimage of a non-leaf node must consist
/// of compression outputs.
pub trait PseudoCompressionFunction<T, const N: usize> {
    fn compress(&self, input: [T; N]) -> T;
}

/// An `n`-to-1 compression function.
pub trait CompressionFunction<T, const N: usize>: PseudoCompressionFunction<T, N> {}

pub struct TruncatedPermutation<T, InnerP, const N: usize, const CHUNK: usize, const PROD: usize> {
    inner_permutation: InnerP,
    _phantom_t: PhantomData<T>,
}

impl<T, InnerP, const N: usize, const CHUNK: usize, const PROD: usize>
    TruncatedPermutation<T, InnerP, N, CHUNK, PROD>
{
    pub fn new(inner_permutation: InnerP) -> Self {
        Self {
            inner_permutation,
            _phantom_t: PhantomData,
        }
    }
}

impl<T, InnerP, const N: usize, const CHUNK: usize, const PROD: usize>
    PseudoCompressionFunction<[T; CHUNK], N> for TruncatedPermutation<T, InnerP, N, CHUNK, PROD>
where
    T: Copy,
    InnerP: CryptographicPermutation<[T; PROD]>,
{
    fn compress(&self, input: [[T; CHUNK]; N]) -> [T; CHUNK] {
        let flat_input = input
            .into_iter()
            .flatten()
            .collect_vec()
            .try_into()
            .unwrap_or_else(|_| panic!("Impossible!"));
        let perm_out = self.inner_permutation.permute(flat_input);
        perm_out[..CHUNK].try_into().unwrap()
    }
}

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
