use core::iter;

use itertools::Itertools;
use p3_field::{Field, PackedValue};

use crate::CryptographicHasher;

/// Serializes field elements to bytes then hashes those bytes using some inner hasher.
#[derive(Copy, Clone, Debug)]
pub struct SerializingHasher<Inner> {
    inner: Inner,
}

impl<Inner> SerializingHasher<Inner> {
    pub const fn new(inner: Inner) -> Self {
        Self { inner }
    }
}

impl<F, Inner> CryptographicHasher<F, [u8; 32]> for SerializingHasher<Inner>
where
    F: Field,
    Inner: CryptographicHasher<u8, [u8; 32]>,
{
    fn hash_iter<I>(&self, input: I) -> [u8; 32]
    where
        I: IntoIterator<Item = F>,
    {
        self.inner.hash_iter(
            input
                .into_iter()
                .flat_map(|elem| elem.to_bytes().into_iter()),
        )
    }
}

impl<P, PW, Inner> CryptographicHasher<P, [PW; 8]> for SerializingHasher<Inner>
where
    P: PackedValue,
    P::Value: Field,
    PW: PackedValue<Value = u32>,
    Inner: CryptographicHasher<PW, [PW; 8]>,
{
    fn hash_iter<I>(&self, input: I) -> [PW; 8]
    where
        I: IntoIterator<Item = P>,
    {
        // We assume that the number of bytes which P::Value maps into is a multiple of 4.
        // This should be true for all our fields.
        debug_assert_eq!(size_of::<P>() % 4, 0);

        // We start by converting our iterator into a stream of u32s.
        let mut u32_stream = input
            .into_iter()
            .flat_map(|elem| {
                elem.as_slice()
                    .iter()
                    .flat_map(|&y| y.to_u32s())
                    .collect_vec() // I suspect there should be a way to avoid this allocation.
            })
            .peekable();

        // Now we combine the u32's into PW's. If lengths do not align we pad by zeros.
        self.inner.hash_iter(iter::from_fn(
            #[inline]
            || {
                u32_stream
                    .peek()
                    .is_some()
                    .then(|| PW::from_fn(|_| u32_stream.next().unwrap_or_default()))
            },
        ))
    }
}

impl<P, PW, Inner> CryptographicHasher<P, [PW; 4]> for SerializingHasher<Inner>
where
    P: PackedValue,
    P::Value: Field,
    PW: PackedValue<Value = u64>,
    Inner: CryptographicHasher<PW, [PW; 4]>,
{
    fn hash_iter<I>(&self, input: I) -> [PW; 4]
    where
        I: IntoIterator<Item = P>,
    {
        // We assume that the number of bytes which P::Value maps into is a multiple of 4.
        // This should be true for all our fields.
        debug_assert_eq!(size_of::<P>() % 4, 0);

        // We start by converting our iterator into a stream of u32s.
        let mut u32_stream = input
            .into_iter()
            .flat_map(|elem| {
                elem.as_slice()
                    .iter()
                    .flat_map(|&y| y.to_u32s())
                    .collect_vec() // I suspect there should be a way to avoid this allocation.
            })
            .peekable();

        // Now we combine the u32's into PW's. If lengths do not align we pad by zeros.
        self.inner.hash_iter(iter::from_fn(
            #[inline]
            || {
                u32_stream.peek().is_some().then(|| {
                    PW::from_fn(|_| {
                        let a = u32_stream.next().unwrap_or_default() as u64;
                        let b = u32_stream.next().unwrap_or_default() as u64;
                        a << 32 | b // Combine the two u32's into a u64.
                    })
                })
            },
        ))
    }
}
