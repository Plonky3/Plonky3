use p3_field::Field;

use crate::CryptographicHasher;

/// Converts a hasher which can hash bytes, u32's or u64's into a hasher which can hash field elements.
///
/// Supports two types of hashing.
/// - Hashing a a sequence of field elements.
/// - Hashing a sequence of arrays of `N` field elements as if we are hashing `N` sequences of field elements in parallel.
///   This is useful when the inner hash is able to use vectorized instructions to compute multiple hashes at once.
#[derive(Copy, Clone, Debug)]
pub struct SerializingHasher<Inner> {
    inner: Inner,
}

impl<Inner> SerializingHasher<Inner> {
    pub const fn new(inner: Inner) -> Self {
        Self { inner }
    }
}

impl<F, Inner, const N: usize> CryptographicHasher<F, [u8; N]> for SerializingHasher<Inner>
where
    F: Field,
    Inner: CryptographicHasher<u8, [u8; N]>,
{
    fn hash_iter<I>(&self, input: I) -> [u8; N]
    where
        I: IntoIterator<Item = F>,
    {
        self.inner.hash_iter(F::into_byte_stream(input))
    }
}

impl<F, Inner, const N: usize> CryptographicHasher<F, [u32; N]> for SerializingHasher<Inner>
where
    F: Field,
    Inner: CryptographicHasher<u32, [u32; N]>,
{
    fn hash_iter<I>(&self, input: I) -> [u32; N]
    where
        I: IntoIterator<Item = F>,
    {
        self.inner.hash_iter(F::into_u32_stream(input))
    }
}

impl<F, Inner, const N: usize> CryptographicHasher<F, [u64; N]> for SerializingHasher<Inner>
where
    F: Field,
    Inner: CryptographicHasher<u64, [u64; N]>,
{
    fn hash_iter<I>(&self, input: I) -> [u64; N]
    where
        I: IntoIterator<Item = F>,
    {
        self.inner.hash_iter(F::into_u64_stream(input))
    }
}

impl<F, Inner, const N: usize, const M: usize> CryptographicHasher<[F; M], [[u8; M]; N]>
    for SerializingHasher<Inner>
where
    F: Field,
    Inner: CryptographicHasher<[u8; M], [[u8; M]; N]>,
{
    fn hash_iter<I>(&self, input: I) -> [[u8; M]; N]
    where
        I: IntoIterator<Item = [F; M]>,
    {
        self.inner.hash_iter(F::into_parallel_byte_streams(input))
    }
}

impl<F, Inner, const N: usize, const M: usize> CryptographicHasher<[F; M], [[u32; M]; N]>
    for SerializingHasher<Inner>
where
    F: Field,
    Inner: CryptographicHasher<[u32; M], [[u32; M]; N]>,
{
    fn hash_iter<I>(&self, input: I) -> [[u32; M]; N]
    where
        I: IntoIterator<Item = [F; M]>,
    {
        self.inner.hash_iter(F::into_parallel_u32_streams(input))
    }
}

impl<F, Inner, const N: usize, const M: usize> CryptographicHasher<[F; M], [[u64; M]; N]>
    for SerializingHasher<Inner>
where
    F: Field,
    Inner: CryptographicHasher<[u64; M], [[u64; M]; N]>,
{
    fn hash_iter<I>(&self, input: I) -> [[u64; M]; N]
    where
        I: IntoIterator<Item = [F; M]>,
    {
        self.inner.hash_iter(F::into_parallel_u64_streams(input))
    }
}
