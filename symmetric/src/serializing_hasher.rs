use core::marker::PhantomData;

use p3_field::{PrimeField32, PrimeField64};

use crate::CryptographicHasher;

/// Maps input field elements to their 4-byte little-endian encodings, and maps output of the form
/// `[u8; 32]` to `[F; 8]`.
#[derive(Copy, Clone)]
pub struct SerializingHasher32<F, Inner> {
    inner: Inner,
    _phantom_f: PhantomData<F>,
}

/// Maps input field elements to their 8-byte little-endian encodings, and maps output of the form
/// `[u8; 32]` to `[F; 4]`.
#[derive(Copy, Clone)]
pub struct SerializingHasher64<F, Inner> {
    inner: Inner,
    _phantom_f: PhantomData<F>,
}

impl<F, Inner> SerializingHasher32<F, Inner> {
    pub fn new(inner: Inner) -> Self {
        Self {
            inner,
            _phantom_f: PhantomData,
        }
    }
}

impl<F, Inner> SerializingHasher64<F, Inner> {
    pub fn new(inner: Inner) -> Self {
        Self {
            inner,
            _phantom_f: PhantomData,
        }
    }
}

impl<F, Inner> CryptographicHasher<F, [F; 8]> for SerializingHasher32<F, Inner>
where
    F: PrimeField32,
    Inner: CryptographicHasher<u8, [u8; 32]>,
{
    fn hash_iter<I>(&self, input: I) -> [F; 8]
    where
        I: IntoIterator<Item = F>,
    {
        let inner_out = self.inner.hash_iter(
            input
                .into_iter()
                .flat_map(|x| x.as_canonical_u32().to_le_bytes()),
        );

        core::array::from_fn(|i| {
            let inner_out_chunk: [u8; 4] = inner_out[i * 4..(i + 1) * 4].try_into().unwrap();
            F::from_wrapped_u32(u32::from_le_bytes(inner_out_chunk))
        })
    }
}

impl<F, Inner> CryptographicHasher<F, [F; 4]> for SerializingHasher64<F, Inner>
where
    F: PrimeField64,
    Inner: CryptographicHasher<u8, [u8; 32]>,
{
    fn hash_iter<I>(&self, input: I) -> [F; 4]
    where
        I: IntoIterator<Item = F>,
    {
        let inner_out = self.inner.hash_iter(
            input
                .into_iter()
                .flat_map(|x| x.as_canonical_u64().to_le_bytes()),
        );

        core::array::from_fn(|i| {
            let inner_out_chunk: [u8; 8] = inner_out[i * 8..(i + 1) * 8].try_into().unwrap();
            F::from_wrapped_u64(u64::from_le_bytes(inner_out_chunk))
        })
    }
}
