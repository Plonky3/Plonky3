use core::borrow::Borrow;
use core::marker::PhantomData;

use p3_field::{PrimeField32, PrimeField64};
use serde::{Deserialize, Serialize};

use crate::CryptographicHasher;

/// Maps input field elements to their 4-byte little-endian encodings, outputs `[u8; 32]`.
#[derive(Copy, Clone)]
pub struct SerializingHasher32<Inner> {
    inner: Inner,
}

/// A wrapper around an array digest, with a phantom type parameter to ensure that the digest is
/// associated with a particular field.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(bound(serialize = "[W; DIGEST_ELEMS]: Serialize"))]
#[serde(bound(deserialize = "[W; DIGEST_ELEMS]: Deserialize<'de>"))]
pub struct Hash<F, W, const DIGEST_ELEMS: usize> {
    value: [W; DIGEST_ELEMS],
    _marker: PhantomData<F>,
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

impl<F, W, const DIGEST_ELEMS: usize> From<[W; DIGEST_ELEMS]> for Hash<F, W, DIGEST_ELEMS> {
    fn from(value: [W; DIGEST_ELEMS]) -> Self {
        Self {
            value,
            _marker: PhantomData,
        }
    }
}

impl<F, W, const DIGEST_ELEMS: usize> From<Hash<F, W, DIGEST_ELEMS>> for [W; DIGEST_ELEMS] {
    fn from(value: Hash<F, W, DIGEST_ELEMS>) -> [W; DIGEST_ELEMS] {
        value.value
    }
}

impl<F, W: PartialEq, const DIGEST_ELEMS: usize> PartialEq<[W; DIGEST_ELEMS]>
    for Hash<F, W, DIGEST_ELEMS>
{
    fn eq(&self, other: &[W; DIGEST_ELEMS]) -> bool {
        self.value == *other
    }
}

impl<F, W, const DIGEST_ELEMS: usize> IntoIterator for Hash<F, W, DIGEST_ELEMS> {
    type Item = W;
    type IntoIter = core::array::IntoIter<W, DIGEST_ELEMS>;

    fn into_iter(self) -> Self::IntoIter {
        self.value.into_iter()
    }
}

impl<F, W, const DIGEST_ELEMS: usize> Borrow<[W; DIGEST_ELEMS]> for Hash<F, W, DIGEST_ELEMS> {
    fn borrow(&self) -> &[W; DIGEST_ELEMS] {
        &self.value
    }
}

impl<F, W, const DIGEST_ELEMS: usize> AsRef<[W; DIGEST_ELEMS]> for Hash<F, W, DIGEST_ELEMS> {
    fn as_ref(&self) -> &[W; DIGEST_ELEMS] {
        &self.value
    }
}
