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
/// A cap of height 0 contains a single element (the root), while a cap of height `h` contains
/// `2^h` elements. The `Digest` type is the full digest (e.g. `[W; DIGEST_ELEMS]`).
///
/// The root count is always a power of two.
///
/// A cap is one full layer of a binary tree, so a layer at depth `h` holds `2^h` digests.
///
/// The height is recovered as `log_2` of the root count, which exists only for a power of two.
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
    /// Returns an error when the root count is not a power of two.
    ///
    /// Such a count has no base-two logarithm, so the cap would have no height.
    ///
    /// Untrusted bytes are the only route by which such a cap could reach a verifier.
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        // Read the wire shape first, so the check runs on a fully decoded layer.
        let MerkleCapRepr { cap, _marker } = MerkleCapRepr::<F, Digest>::deserialize(deserializer)?;

        // Invariant: a cap is a full tree layer, so it holds 2^h digests for its depth h.
        //
        //     4 roots -> depth 2, accepted
        //     3 roots -> no layer has that size, rejected
        //     0 roots -> no layer has that size, rejected
        if !cap.len().is_power_of_two() {
            return Err(D::Error::invalid_length(
                cap.len(),
                &"a power-of-two number of Merkle cap roots",
            ));
        }

        Ok(Self { cap, _marker })
    }
}

impl<F, Digest> MerkleCap<F, Digest> {
    /// Wrap one full layer of digests as a cap.
    ///
    /// # Panics
    ///
    /// Panics when the number of digests is not a power of two.
    pub fn new(cap: Vec<Digest>) -> Self {
        // Invariant: a cap is a full tree layer, so it holds 2^h digests for its depth h.
        assert!(
            cap.len().is_power_of_two(),
            "a Merkle cap holds a power-of-two number of roots, got {}",
            cap.len()
        );
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
    fn test_merkle_cap_deserialize_rejects_non_power_of_two_root_count() {
        // Three roots is the smallest count no tree layer can have.
        //
        // Zero roots is the degenerate count a truncating encoder would produce.
        //
        // Each count is fed through both decoding paths:
        //
        //     compact         -> the fields arrive as a sequence
        //     self-describing -> the fields arrive as a map
        for (bytes, json) in [
            (
                &[3u8, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3][..],
                r#"{"cap":[[1,1,1,1],[2,2,2,2],[3,3,3,3]],"_marker":null}"#,
            ),
            (&[0u8][..], r#"{"cap":[],"_marker":null}"#),
        ] {
            postcard::from_bytes::<MerkleCap<F, Digest>>(bytes)
                .expect_err("a cap with a non-power-of-two root count must not deserialize");
            serde_json::from_str::<MerkleCap<F, Digest>>(json)
                .expect_err("a cap with a non-power-of-two root count must not deserialize");
        }
    }

    #[test]
    fn test_merkle_cap_deserialize_error_names_the_requirement() {
        // The rejection reaches an operator as text, so it has to state what was wrong.
        let err = serde_json::from_str::<MerkleCap<F, Digest>>(
            r#"{"cap":[[1,1,1,1],[2,2,2,2],[3,3,3,3]],"_marker":null}"#,
        )
        .expect_err("three roots is not a power of two");
        assert!(
            err.to_string().contains("power-of-two"),
            "unexpected error: {err}"
        );
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
