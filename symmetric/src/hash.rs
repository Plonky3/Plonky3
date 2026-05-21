use alloc::vec;
use alloc::vec::{IntoIter, Vec};
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
/// `2^h` elements. The `Digest` type is the full digest (e.g. `[W; DIGEST_ELEMS]`).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(bound(serialize = "Digest: Serialize"))]
#[serde(bound(deserialize = "Digest: Deserialize<'de>"))]
pub struct MerkleCap<F, Digest> {
    cap: Vec<Digest>,
    _marker: PhantomData<F>,
}

impl<F, Digest> MerkleCap<F, Digest> {
    /// Create a new `MerkleCap` from a vector of digests.
    pub fn new(cap: Vec<Digest>) -> Self {
        assert!(cap.len().is_power_of_two());
        Self {
            cap,
            _marker: PhantomData,
        }
    }

    /// Returns the number of digests in the cap.
    #[must_use]
    pub const fn num_roots(&self) -> usize {
        self.cap.len()
    }

    /// Returns the height of the cap (log2 of the number of elements).
    /// A cap with 1 element has height 0, a cap with 2 elements has height 1, etc.
    #[must_use]
    pub const fn height(&self) -> usize {
        log2_strict_usize(self.num_roots())
    }

    /// Returns a reference to the underlying slice of digests.
    #[must_use]
    pub fn roots(&self) -> &[Digest] {
        &self.cap
    }

    /// Flattens the cap into a single vector of digest words.
    pub fn into_roots(self) -> Vec<Digest> {
        self.cap.into_iter().collect()
    }
}

impl<F, Digest> From<Vec<Digest>> for MerkleCap<F, Digest> {
    fn from(cap: Vec<Digest>) -> Self {
        Self::new(cap)
    }
}

impl<F, W, const N: usize> From<Hash<F, W, N>> for MerkleCap<F, [W; N]> {
    fn from(hash: Hash<F, W, N>) -> Self {
        Self::new(vec![hash.into()])
    }
}

impl<F, Digest> Borrow<[Digest]> for MerkleCap<F, Digest> {
    fn borrow(&self) -> &[Digest] {
        &self.cap
    }
}

impl<F, Digest> AsRef<[Digest]> for MerkleCap<F, Digest> {
    fn as_ref(&self) -> &[Digest] {
        &self.cap
    }
}

impl<F, Digest> core::ops::Index<usize> for MerkleCap<F, Digest> {
    type Output = Digest;

    fn index(&self, index: usize) -> &Self::Output {
        &self.cap[index]
    }
}

impl<F, Digest> IntoIterator for MerkleCap<F, Digest> {
    type Item = Digest;
    type IntoIter = IntoIter<Digest>;

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

#[cfg(test)]
mod tests {
    use alloc::vec;

    use p3_goldilocks::Goldilocks;

    use super::*;

    type F = Goldilocks;
    type Digest = [u8; 4];

    #[test]
    fn test_merkle_cap_new_with_power_of_two_sizes() {
        let cap = MerkleCap::<F, Digest>::new(vec![[7u8; 4]]);
        assert_eq!(cap.num_roots(), 1);
        assert_eq!(cap.height(), 0);

        let cap = MerkleCap::<F, Digest>::new(vec![[0u8; 4]; 2]);
        assert_eq!(cap.num_roots(), 2);
        assert_eq!(cap.height(), 1);

        let cap = MerkleCap::<F, Digest>::new(vec![[0u8; 4]; 8]);
        assert_eq!(cap.num_roots(), 8);
        assert_eq!(cap.height(), 3);
    }

    #[test]
    #[should_panic]
    fn test_merkle_cap_new_panics_on_empty() {
        let _ = MerkleCap::<F, Digest>::new(vec![]);
    }

    #[test]
    #[should_panic]
    fn test_merkle_cap_new_panics_on_three() {
        let _ = MerkleCap::<F, Digest>::new(vec![[0u8; 4]; 3]);
    }

    #[test]
    #[should_panic]
    fn test_merkle_cap_from_vec_panics_on_empty() {
        let _: MerkleCap<F, Digest> = vec![].into();
    }

    #[test]
    fn test_merkle_cap_from_hash() {
        let hash = Hash::<F, u8, 4>::from([1u8, 2, 3, 4]);
        let cap: MerkleCap<F, [u8; 4]> = hash.into();
        assert_eq!(cap.num_roots(), 1);
        assert_eq!(cap.height(), 0);
    }
}
