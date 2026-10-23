//! Accounting for the initialized payload of retained digest layers.

use core::cmp::Reverse;

use itertools::Itertools;
use p3_field::PackedValue;

use crate::builder::layer_shapes;
use crate::{MerkleTreeError, MerkleTreeMmcs};

/// Initialized payload retained in a Merkle tree's digest-layer buffers.
///
/// Counts every layer from leaves through the root, including padding and layers above the cap.
/// This excludes matrices, hiding salts, copied cap digests, planner and vector metadata,
/// allocator overhead, spare capacity, and hashing scratch space. In particular, prehashed leaf
/// vectors supplied by a caller may have more capacity than the initialized payload counted here.
/// This is not a bound on the total memory needed to build or retain a commitment.
#[must_use]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub struct MerkleDigestStorage {
    /// Number of retained digest layers, including the leaf layer and root.
    pub layer_count: usize,
    /// Number of initialized digests across all retained layers, including padding.
    pub digest_count: usize,
    /// Payload bytes: `digest_count * size_of::<[PW::Value; DIGEST_ELEMS]>()`.
    pub digest_bytes: usize,
}

impl<P, PW, H, C, const N: usize, const DIGEST_ELEMS: usize>
    MerkleTreeMmcs<P, PW, H, C, N, DIGEST_ELEMS>
where
    PW: PackedValue,
{
    /// Calculate retained digest-buffer payload before allocating matrices or hashing them.
    ///
    /// Heights may be supplied in any order, with repeats. They must satisfy the same height
    /// ladder and progressing-bridge requirements as commitment construction. The calculation
    /// shares the builder's layer plan and does not depend on matrix widths or the cap height.
    /// It allocates only sorting and layer-planning metadata, never digest buffers.
    ///
    /// See [`MerkleDigestStorage`] for what is included and excluded.
    ///
    /// # Errors
    /// Returns the commitment geometry error for an invalid batch, or
    /// [`MerkleTreeError::DigestStorageOverflow`] if height rounding, padding, the total digest
    /// count, or its byte count cannot be represented by `usize`.
    pub fn digest_storage(
        &self,
        heights: &[usize],
    ) -> Result<MerkleDigestStorage, MerkleTreeError> {
        let shapes = layer_shapes::<N>(heights.iter().copied().sorted_by_key(|&h| Reverse(h)))?;
        let digest_count = shapes.iter().try_fold(0usize, |count, layer| {
            count
                .checked_add(layer.padded)
                .ok_or(MerkleTreeError::DigestStorageOverflow)
        })?;
        let digest_bytes = digest_count
            .checked_mul(size_of::<[PW::Value; DIGEST_ELEMS]>())
            .ok_or(MerkleTreeError::DigestStorageOverflow)?;
        Ok(MerkleDigestStorage {
            layer_count: shapes.len(),
            digest_count,
            digest_bytes,
        })
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec;
    use alloc::vec::Vec;

    use p3_baby_bear::BabyBear;
    use p3_commit::Mmcs;
    use p3_field::Field;
    use p3_keccak::Keccak256Hash;
    use p3_matrix::dense::RowMajorMatrix;
    use p3_symmetric::CompressionFunctionFromHasher;

    use crate::{MerkleTreeError, MerkleTreeMmcs};

    fn check_storage<const N: usize>(heights: &[usize]) {
        for cap_height in [0, 1, usize::MAX] {
            let mmcs = MerkleTreeMmcs::<u8, u8, _, _, N, 32>::new(
                Keccak256Hash,
                CompressionFunctionFromHasher::new(Keccak256Hash),
                cap_height,
            );
            let storage = mmcs.digest_storage(heights).unwrap();
            let matrices = heights
                .iter()
                .enumerate()
                .map(|(i, &height)| RowMajorMatrix::new(vec![i as u8; height * (i + 1)], i + 1))
                .collect();
            let (_, tree) = mmcs.commit(matrices);
            let actual_count: usize = tree.digest_layers.iter().map(Vec::len).sum();
            assert_eq!(
                storage.layer_count,
                tree.digest_layers.len(),
                "N={N}, {heights:?}"
            );
            assert_eq!(storage.digest_count, actual_count, "N={N}, {heights:?}");
            assert_eq!(
                storage.digest_bytes,
                actual_count * 32,
                "N={N}, {heights:?}"
            );
        }
    }

    #[test]
    fn digest_storage_matches_retained_layers_with_caps_and_injections() {
        // Omitting padding, a bridge layer, or layers above the cap breaks this oracle.
        for heights in [
            &[1][..],
            &[3],
            &[17],
            &[64],
            &[64, 32],
            &[32, 64, 32],
            &[17, 9, 5, 3],
            &[5, 5, 3],
            &[64, 16, 1],
        ] {
            check_storage::<2>(heights);
            check_storage::<4>(heights);
            check_storage::<8>(heights);
        }
    }

    #[test]
    fn digest_storage_counts_bridge_padding_and_digest_word_bytes() {
        type Packing = <BabyBear as Field>::Packing;
        let mmcs = MerkleTreeMmcs::<Packing, Packing, _, _, 4, 4>::new((), (), 0);
        let single = mmcs.digest_storage(&[64]).unwrap();
        assert_eq!(
            (single.layer_count, single.digest_count, single.digest_bytes),
            (4, 85, 1360)
        );
        // The bridge retains 32 nodes; later a full step produces 2 nodes padded to 4.
        // Retained layer lengths: 64, 32, 8, 4, 1.
        let mixed = mmcs.digest_storage(&[64, 32]).unwrap();
        assert_eq!(
            (mixed.layer_count, mixed.digest_count, mixed.digest_bytes),
            (5, 109, 1744)
        );
    }

    #[test]
    fn digest_storage_rejects_invalid_heights_without_building() {
        let mmcs = MerkleTreeMmcs::<u8, u8, _, _, 8, 32>::new((), (), 0);
        assert!(matches!(
            mmcs.digest_storage(&[]),
            Err(MerkleTreeError::EmptyBatch)
        ));
        assert!(matches!(
            mmcs.digest_storage(&[0, 0]),
            Err(MerkleTreeError::EmptyBatch)
        ));
        for heights in [&[8, 0][..], &[8, 3], &[5, 4]] {
            assert!(matches!(
                mmcs.digest_storage(heights),
                Err(MerkleTreeError::IncompatibleHeights { .. })
            ));
        }
        for heights in [&[4, 2][..], &[32, 2], &[256, 2], &[8, 4, 2]] {
            assert!(matches!(
                mmcs.digest_storage(heights),
                Err(MerkleTreeError::NonShrinkingLayer { .. })
            ));
        }
    }

    #[test]
    fn digest_storage_checks_overflow_before_any_digest_allocation() {
        let mmcs = MerkleTreeMmcs::<u8, u8, _, _, 2, 32>::new((), (), 0);
        for heights in [
            &[usize::MAX][..],
            &[usize::MAX, 0],
            &[1usize << (usize::BITS - 1)],
        ] {
            assert!(matches!(
                mmcs.digest_storage(heights),
                Err(MerkleTreeError::DigestStorageOverflow)
            ));
        }
        // A bridge can retain two minimum-width layers whose combined count overflows.
        let wide =
            MerkleTreeMmcs::<u8, u8, _, _, { 1usize << (usize::BITS - 1) }, 1>::new((), (), 0);
        assert!(matches!(
            wide.digest_storage(&[1usize << (usize::BITS - 1), 1usize << (usize::BITS - 2)]),
            Err(MerkleTreeError::DigestStorageOverflow)
        ));
    }
}
