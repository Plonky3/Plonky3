use alloc::vec::Vec;
use core::borrow::Borrow;
use core::marker::PhantomData;

use p3_util::log2_strict_usize;
use serde::{Deserialize, Serialize};

/// A wrapper around an array digest, with a phantom type parameter to ensure that the digest is
/// associated with a particular field.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(bound(serialize = "[W; DIGEST_ELEMS]: Serialize"))]
#[serde(bound(deserialize = "[W; DIGEST_ELEMS]: Deserialize<'de>"))]
pub struct Hash<F, W, const DIGEST_ELEMS: usize> {
    value: [W; DIGEST_ELEMS],
    _marker: PhantomData<F>,
}

/// The Merkle cap of height `h` of a Merkle tree is the `h`-th layer (from the root) of the tree.
/// It can be used in place of the root to verify Merkle paths, which are `h` elements shorter.
///
/// A cap of height 0 contains a single element (the root), while a cap of height `h` contains
/// `2^h` elements.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(bound(serialize = "[W; DIGEST_ELEMS]: Serialize"))]
#[serde(bound(deserialize = "[W; DIGEST_ELEMS]: Deserialize<'de>"))]
pub struct MerkleCap<F, W, const DIGEST_ELEMS: usize> {
    cap: Vec<[W; DIGEST_ELEMS]>,
    _marker: PhantomData<F>,
}

impl<F, W, const DIGEST_ELEMS: usize> Default for MerkleCap<F, W, DIGEST_ELEMS> {
    fn default() -> Self {
        Self {
            cap: Vec::new(),
            _marker: PhantomData,
        }
    }
}

impl<F, W, const DIGEST_ELEMS: usize> MerkleCap<F, W, DIGEST_ELEMS> {
    /// Create a new `MerkleCap` from a vector of digests.
    pub fn new(cap: Vec<[W; DIGEST_ELEMS]>) -> Self {
        debug_assert!(cap.len().is_power_of_two() || cap.is_empty());
        Self {
            cap,
            _marker: PhantomData,
        }
    }

    /// Returns the number of digests in the cap.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.cap.len()
    }

    /// Returns true if the cap contains no digests.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.cap.is_empty()
    }

    /// Returns the height of the cap (log2 of the number of elements).
    /// A cap with 1 element has height 0, a cap with 2 elements has height 1, etc.
    #[must_use]
    pub const fn height(&self) -> usize {
        if self.cap.is_empty() {
            0
        } else {
            log2_strict_usize(self.cap.len())
        }
    }

    /// Flattens the cap into a single vector of digest words.
    pub fn flatten(&self) -> Vec<W>
    where
        W: Clone,
    {
        self.cap.iter().flat_map(|h| h.iter().cloned()).collect()
    }

    /// Returns a reference to the underlying vector of digests.
    #[must_use]
    pub fn as_slice(&self) -> &[[W; DIGEST_ELEMS]] {
        &self.cap
    }
}

impl<F, W, const DIGEST_ELEMS: usize> From<Vec<[W; DIGEST_ELEMS]>>
    for MerkleCap<F, W, DIGEST_ELEMS>
{
    fn from(cap: Vec<[W; DIGEST_ELEMS]>) -> Self {
        Self::new(cap)
    }
}

impl<F, W: Copy, const DIGEST_ELEMS: usize> From<Hash<F, W, DIGEST_ELEMS>>
    for MerkleCap<F, W, DIGEST_ELEMS>
{
    fn from(hash: Hash<F, W, DIGEST_ELEMS>) -> Self {
        Self::new(alloc::vec![hash.into()])
    }
}

impl<F, W, const DIGEST_ELEMS: usize> Borrow<[[W; DIGEST_ELEMS]]>
    for MerkleCap<F, W, DIGEST_ELEMS>
{
    fn borrow(&self) -> &[[W; DIGEST_ELEMS]] {
        &self.cap
    }
}

impl<F, W, const DIGEST_ELEMS: usize> AsRef<[[W; DIGEST_ELEMS]]> for MerkleCap<F, W, DIGEST_ELEMS> {
    fn as_ref(&self) -> &[[W; DIGEST_ELEMS]] {
        &self.cap
    }
}

impl<F, W, const DIGEST_ELEMS: usize> core::ops::Index<usize> for MerkleCap<F, W, DIGEST_ELEMS> {
    type Output = [W; DIGEST_ELEMS];

    fn index(&self, index: usize) -> &Self::Output {
        &self.cap[index]
    }
}

impl<F, W, const DIGEST_ELEMS: usize> IntoIterator for MerkleCap<F, W, DIGEST_ELEMS> {
    type Item = [W; DIGEST_ELEMS];
    type IntoIter = alloc::vec::IntoIter<[W; DIGEST_ELEMS]>;

    fn into_iter(self) -> Self::IntoIter {
        self.cap.into_iter()
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
