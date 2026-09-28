use alloc::vec;
use alloc::vec::{IntoIter, Vec};
use core::borrow::Borrow;
use core::marker::PhantomData;

use p3_util::log2_strict_usize;
use serde::de::Error;
use serde::{Deserialize, Deserializer, Serialize};

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
/// A cap of height 0 contains a single element (the root).
///
/// A cap of height `h` holds the product of the arities of the `h` levels above it.
///
/// That is `2^h` digests in a binary tree and `5^h` in a tree of T5 nodes.
///
/// The `Digest` type is the full digest (e.g. `[W; DIGEST_ELEMS]`).
///
/// The root count is never zero, since every tree layer holds at least one digest.
///
/// Every path that builds a cap enforces this, including the one that reads it off the wire.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
#[serde(bound(serialize = "Digest: Serialize"))]
pub struct MerkleCap<F, Digest> {
    cap: Vec<Digest>,
    _marker: PhantomData<F>,
}

/// Mirror of the cap layer that lets the derive produce the decoder.
///
/// The cap itself exposes no unchecked constructor, so the derive cannot target it.
///
/// Field names, field order and the container name reproduce the derived encoding exactly.
///
/// Changing any of the three moves the bytes of every stored proof.
#[derive(Deserialize)]
#[serde(rename = "MerkleCap")]
#[serde(bound(deserialize = "Digest: Deserialize<'de>"))]
struct MerkleCapRepr<F, Digest> {
    /// The digests of the layer, left to right.
    cap: Vec<Digest>,
    /// Pins the layer to one field without carrying any data of its own.
    _marker: PhantomData<F>,
}

impl<'de, F, Digest: Deserialize<'de>> Deserialize<'de> for MerkleCap<F, Digest> {
    /// # Errors
    ///
    /// Returns an error when the cap holds no root.
    ///
    /// Untrusted bytes are the only route by which such a cap could reach a verifier.
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        // Read the wire shape first, so the check runs on a fully decoded layer.
        let MerkleCapRepr { cap, _marker } = MerkleCapRepr::<F, Digest>::deserialize(deserializer)?;

        // Invariant: a cap is a full tree layer, and every layer holds at least one digest.
        //
        // Its exact size depends on the arity of the tree, which only the verifier knows.
        if cap.is_empty() {
            return Err(D::Error::invalid_length(0, &"at least one Merkle cap root"));
        }

        Ok(Self { cap, _marker })
    }
}

impl<F, Digest> MerkleCap<F, Digest> {
    /// Wrap one full layer of digests as a cap.
    ///
    /// # Panics
    ///
    /// Panics when there are no digests.
    pub fn new(cap: Vec<Digest>) -> Self {
        // Invariant: a cap is a full tree layer, and every layer holds at least one digest.
        assert!(!cap.is_empty(), "a Merkle cap holds at least one root");
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

    /// Returns the height of a binary cap (log2 of the number of elements).
    ///
    /// A cap with 1 element has height 0, a cap with 2 elements has height 1, etc.
    ///
    /// # Panics
    ///
    /// Panics when the root count is not a power of two, as in a cap of a T5 tree.
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
    use alloc::string::ToString;
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
    fn test_merkle_cap_wire_shape_is_stable() {
        // Fixture state: four roots, each a digest of four repeated bytes.
        let cap = MerkleCap::<F, Digest>::new(vec![[1u8; 4], [2u8; 4], [3u8; 4], [4u8; 4]]);

        // Compact encoding: a varint root count, the roots in order, nothing for the marker.
        //
        //     04 | 01010101 | 02020202 | 03030303 | 04040404
        //
        // Every stored proof replays these bytes, so they are pinned rather than round-tripped.
        let bytes = postcard::to_allocvec(&cap).unwrap();
        assert_eq!(bytes, [4, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4]);
        let back: MerkleCap<F, Digest> = postcard::from_bytes(&bytes).unwrap();
        assert_eq!(back, cap);
        assert_eq!(back.height(), 2);

        // Self-describing encoding: a map whose two keys pin the field names and their order.
        let json = serde_json::to_string(&cap).unwrap();
        assert_eq!(
            json,
            r#"{"cap":[[1,1,1,1],[2,2,2,2],[3,3,3,3],[4,4,4,4]],"_marker":null}"#
        );
        assert_eq!(
            serde_json::from_str::<MerkleCap<F, Digest>>(&json).unwrap(),
            cap
        );
    }

    #[test]
    fn test_merkle_cap_deserialize_rejects_an_empty_cap() {
        // Zero roots is the degenerate count a truncating encoder would produce.
        //
        // The self-describing decoder reports the count and the requirement verbatim.
        let err = serde_json::from_str::<MerkleCap<F, Digest>>(r#"{"cap":[],"_marker":null}"#)
            .expect_err("a cap with no root must not deserialize");
        assert_eq!(
            err.to_string(),
            "invalid length 0, expected at least one Merkle cap root"
        );

        // The compact decoder refuses the empty count and accepts a single root.
        postcard::from_bytes::<MerkleCap<F, Digest>>(&[0u8])
            .expect_err("a cap with no root must not deserialize");
        let cap = postcard::from_bytes::<MerkleCap<F, Digest>>(&[1u8, 1, 1, 1, 1])
            .expect("one root is accepted");
        assert_eq!(cap.num_roots(), 1);
    }

    #[test]
    fn test_merkle_cap_of_a_five_ary_layer_round_trips() {
        // A T5 tree's cap of height 1 is the five children of the root.
        let cap = MerkleCap::<F, Digest>::new(vec![[9u8; 4]; 5]);
        let bytes = postcard::to_allocvec(&cap).unwrap();
        let back: MerkleCap<F, Digest> = postcard::from_bytes(&bytes).unwrap();
        assert_eq!(back.num_roots(), 5);
        assert_eq!(back, cap);
    }

    #[test]
    #[should_panic]
    fn test_merkle_cap_new_panics_on_empty() {
        let _ = MerkleCap::<F, Digest>::new(vec![]);
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
