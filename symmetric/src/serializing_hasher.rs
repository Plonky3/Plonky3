use p3_field::{PackedField, PackedValue, PrimeField32, PrimeField64};

use crate::CryptographicHasher;

/// Maps input field elements to their 4-byte little-endian encodings, outputs `[u8; 32]`.
#[derive(Copy, Clone)]
pub struct SerializingHasher32<Inner> {
    inner: Inner,
}

/// Maps input field elements to their 8-byte little-endian encodings, outputs `[u8; 32]`.
#[derive(Copy, Clone)]
pub struct SerializingHasher64<Inner> {
    inner: Inner,
}

impl<Inner> SerializingHasher32<Inner> {
    pub fn new(inner: Inner) -> Self {
        Self { inner }
    }
}

impl<Inner> SerializingHasher64<Inner> {
    pub fn new(inner: Inner) -> Self {
        Self { inner }
    }
}

impl<F, Inner> CryptographicHasher<F, [u8; 32]> for SerializingHasher32<Inner>
where
    F: PrimeField32,
    Inner: CryptographicHasher<u8, [u8; 32]>,
{
    fn hash_iter<I>(&self, input: I) -> [u8; 32]
    where
        I: IntoIterator<Item = F>,
    {
        self.inner.hash_iter(
            input
                .into_iter()
                .flat_map(|x| x.as_canonical_u32().to_le_bytes()),
        )
    }
}

impl<P, PW, Inner> CryptographicHasher<P, [PW; 8]> for SerializingHasher32<Inner>
where
    P: PackedField,
    P::F: PrimeField32,
    PW: PackedValue<Value = u32>,
    Inner: CryptographicHasher<PW, [PW; 8]>,
{
    fn hash_iter<I>(&self, input: I) -> [PW; 8]
    where
        I: IntoIterator<Item = P>,
    {
        self.inner.hash_iter(
            input
                .into_iter()
                .map(|x| PW::from_fn(|i| x.as_slice()[i].as_canonical_u32())),
        )
    }
}

impl<F, Inner> CryptographicHasher<F, [u8; 32]> for SerializingHasher64<Inner>
where
    F: PrimeField64,
    Inner: CryptographicHasher<u8, [u8; 32]>,
{
    fn hash_iter<I>(&self, input: I) -> [u8; 32]
    where
        I: IntoIterator<Item = F>,
    {
        self.inner.hash_iter(
            input
                .into_iter()
                .flat_map(|x| x.as_canonical_u64().to_le_bytes()),
        )
    }
}

impl<P, PW, Inner> CryptographicHasher<P, [PW; 4]> for SerializingHasher64<Inner>
where
    P: PackedField,
    P::F: PrimeField64,
    PW: PackedValue<Value = u64>,
    Inner: CryptographicHasher<PW, [PW; 4]>,
{
    fn hash_iter<I>(&self, input: I) -> [PW; 4]
    where
        I: IntoIterator<Item = P>,
    {
        self.inner.hash_iter(
            input
                .into_iter()
                .map(|x| PW::from_fn(|i| x.as_slice()[i].as_canonical_u64())),
        )
    }
}
