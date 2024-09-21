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

#[derive(Clone, Debug)]
pub struct TruncatedPermutation<InnerP, const N: usize, const CHUNK: usize, const WIDTH: usize> {
    inner_permutation: InnerP,
}

impl<InnerP, const N: usize, const CHUNK: usize, const WIDTH: usize>
    TruncatedPermutation<InnerP, N, CHUNK, WIDTH>
{
    pub const fn new(inner_permutation: InnerP) -> Self {
        Self { inner_permutation }
    }
}

impl<T, InnerP, const N: usize, const CHUNK: usize, const WIDTH: usize>
    PseudoCompressionFunction<[T; CHUNK], N> for TruncatedPermutation<InnerP, N, CHUNK, WIDTH>
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

#[derive(Clone, Debug)]
pub struct CompressionFunctionFromHasher<H, const N: usize, const CHUNK: usize> {
    hasher: H,
}

impl<H, const N: usize, const CHUNK: usize> CompressionFunctionFromHasher<H, N, CHUNK> {
    pub const fn new(hasher: H) -> Self {
        Self { hasher }
    }
}

impl<T, H, const N: usize, const CHUNK: usize> PseudoCompressionFunction<[T; CHUNK], N>
    for CompressionFunctionFromHasher<H, N, CHUNK>
where
    T: Clone,
    H: CryptographicHasher<T, [T; CHUNK]>,
{
    fn compress(&self, input: [[T; CHUNK]; N]) -> [T; CHUNK] {
        self.hasher.hash_iter(input.into_iter().flatten())
    }
}

impl<T, H, const N: usize, const CHUNK: usize> CompressionFunction<[T; CHUNK], N>
    for CompressionFunctionFromHasher<H, N, CHUNK>
where
    T: Clone,
    H: CryptographicHasher<T, [T; CHUNK]>,
{
}
