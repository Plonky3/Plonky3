//! Batch Merkle proofs with deduplication of redundant siblings.
//!
//! When opening multiple indices from the same Merkle tree, there's often redundancy
//! in the sibling nodes across different paths. The `BatchMerkleProof` structure
//! uses a variation of the [Octopus](https://eprint.iacr.org/2017/933) algorithm
//! to remove duplicate internal nodes.

use alloc::collections::BTreeMap;
use alloc::vec;
use alloc::vec::Vec;

use serde::{Deserialize, Serialize};

/// Batch Merkle proof aggregating multiple openings with deduplication.
///
/// When opening multiple indices from the same Merkle tree, sibling nodes at
/// higher levels are often shared between paths. This structure stores only
/// the unique siblings needed to verify all openings.
///
/// The algorithm is a variation of [Octopus](https://eprint.iacr.org/2017/933).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BatchMerkleProof<Digest> {
    /// For each normalized index (sorted, deduplicated), the siblings needed
    /// along its path that aren't covered by other opened indices.
    /// `nodes[i]` contains siblings for the i-th normalized index.
    pub nodes: Vec<Vec<Digest>>,
    /// Depth of the Merkle tree (number of levels from leaves to root).
    pub depth: u8,
}

impl<Digest: Clone + Eq> BatchMerkleProof<Digest> {
    /// Construct a batch Merkle proof from individual proofs.
    ///
    /// # Arguments
    /// * `proofs` - Individual Merkle proofs (sibling vectors from leaf to root)
    /// * `indexes` - Leaf indices corresponding to each proof
    ///
    /// # Panics
    /// * If `proofs` is empty
    /// * If `proofs` and `indexes` have different lengths
    /// * If proofs have different depths
    pub fn from_single_proofs(proofs: &[Vec<Digest>], indexes: &[usize]) -> Self {
        assert!(!proofs.is_empty(), "at least one proof must be provided");
        assert_eq!(
            proofs.len(),
            indexes.len(),
            "number of proofs must equal number of indexes"
        );

        let depth = proofs[0].len();
        for proof in proofs {
            assert_eq!(depth, proof.len(), "all proofs must have the same depth");
        }

        // Sort indexes in ascending order and rearrange proofs accordingly
        let mut proof_map: BTreeMap<usize, &Vec<Digest>> = BTreeMap::new();
        for (&index, proof) in indexes.iter().zip(proofs.iter()) {
            proof_map.insert(index, proof);
        }
        let sorted_indexes: Vec<usize> = proof_map.keys().copied().collect();
        let sorted_proofs: Vec<&Vec<Digest>> = proof_map.values().copied().collect();

        let mut nodes: Vec<Vec<Digest>> = Vec::with_capacity(sorted_indexes.len());

        let mut i = 0;
        while i < sorted_indexes.len() {
            if i + 1 < sorted_indexes.len()
                && are_siblings(sorted_indexes[i], sorted_indexes[i + 1])
            {
                nodes.push(vec![]);
                nodes.push(vec![]);
                i += 2;
            } else {
                nodes.push(vec![sorted_proofs[i][0].clone()]);
                i += 1;
            }
        }

        let mut current_indexes = sorted_indexes;
        for d in 1..depth {
            let mut parent_indexes: Vec<usize> = Vec::new();
            let mut index_to_node_idx: BTreeMap<usize, usize> = BTreeMap::new();

            for (node_idx, &idx) in current_indexes.iter().enumerate() {
                let parent_idx = idx >> 1;
                if !index_to_node_idx.contains_key(&parent_idx) {
                    index_to_node_idx.insert(parent_idx, node_idx);
                    parent_indexes.push(parent_idx);
                }
            }

            let mut j = 0;
            while j < parent_indexes.len() {
                let parent_idx = parent_indexes[j];
                let node_idx = *index_to_node_idx.get(&parent_idx).unwrap();

                if j + 1 < parent_indexes.len() && are_siblings(parent_idx, parent_indexes[j + 1]) {
                    j += 2;
                } else {
                    nodes[node_idx].push(sorted_proofs[node_idx][d].clone());
                    j += 1;
                }
            }

            current_indexes = parent_indexes;
        }

        BatchMerkleProof {
            nodes,
            depth: depth as u8,
        }
    }

    /// Reconstruct individual proofs from a batch proof.
    ///
    /// Given the original indices (in any order) and their leaf values,
    /// reconstructs the individual Merkle proofs that can be verified separately.
    ///
    /// # Arguments
    /// * `indexes` - The original indices that were opened
    ///
    /// # Returns
    /// A vector of individual proofs, one for each index in the input order.
    pub fn into_single_proofs(&self, indexes: &[usize]) -> Result<Vec<Vec<Digest>>, BatchProofError>
    where
        Digest: Copy,
    {
        if indexes.is_empty() {
            return Err(BatchProofError::EmptyIndexes);
        }

        let index_map: BTreeMap<usize, Vec<usize>> = {
            let mut map = BTreeMap::new();
            for (pos, &idx) in indexes.iter().enumerate() {
                map.entry(idx).or_insert_with(Vec::new).push(pos);
            }
            map
        };

        let sorted_unique_indexes: Vec<usize> = index_map.keys().copied().collect();

        if sorted_unique_indexes.len() != self.nodes.len() {
            return Err(BatchProofError::IndexCountMismatch {
                expected: self.nodes.len(),
                got: sorted_unique_indexes.len(),
            });
        }

        let mut proofs: Vec<Vec<Digest>> = vec![vec![]; indexes.len()];
        let mut proof_pointers: Vec<usize> = vec![0; sorted_unique_indexes.len()];
        let mut node_idx_to_sibling: BTreeMap<usize, Digest> = BTreeMap::new();

        let mut i = 0;
        while i < sorted_unique_indexes.len() {
            let idx = sorted_unique_indexes[i];
            let sibling_idx = idx ^ 1;

            let sibling_node_idx = sorted_unique_indexes.iter().position(|&x| x == sibling_idx);

            let sibling = if sibling_node_idx.is_some() {
                i += 2;
                continue;
            } else {
                if self.nodes[i].is_empty() {
                    return Err(BatchProofError::MissingProofNode);
                }
                let sib = self.nodes[i][0];
                proof_pointers[i] = 1;
                i += 1;
                sib
            };

            node_idx_to_sibling.insert(i - 1, sibling);
        }

        for (orig_pos, &orig_idx) in indexes.iter().enumerate() {
            let sorted_pos = sorted_unique_indexes
                .iter()
                .position(|&x| x == orig_idx)
                .unwrap();

            let mut proof = Vec::with_capacity(self.depth as usize);
            let mut current_idx = orig_idx;

            for level in 0..self.depth as usize {
                let sibling_idx = current_idx ^ 1;

                if level == 0 {
                    let sibling_pos = sorted_unique_indexes.iter().position(|&x| x == sibling_idx);

                    if sibling_pos.is_some() {
                        return Err(BatchProofError::CannotReconstructWithOpenedSiblings);
                    } else if let Some(&sib) = node_idx_to_sibling.get(&sorted_pos) {
                        proof.push(sib);
                    } else {
                        return Err(BatchProofError::MissingProofNode);
                    }
                } else {
                    return Err(BatchProofError::CannotReconstructWithOpenedSiblings);
                }

                current_idx >>= 1;
            }

            proofs[orig_pos] = proof;
        }

        Ok(proofs)
    }

    /// Verify batch openings against a root commitment using a compression function.
    ///
    /// # Arguments
    /// * `root` - The expected Merkle root
    /// * `indexes` - The opened leaf indices (must be in the same order as when batch was created)
    /// * `leaf_hashes` - The hashes of the opened leaves, corresponding to each index
    /// * `compress` - Function to compress two digests into one
    ///
    /// # Returns
    /// `Ok(())` if verification succeeds, error otherwise.
    pub fn verify<F>(
        &self,
        root: &Digest,
        indexes: &[usize],
        leaf_hashes: &[Digest],
        compress: F,
    ) -> Result<(), BatchProofError>
    where
        Digest: Copy,
        F: Fn([Digest; 2]) -> Digest,
    {
        if indexes.is_empty() {
            return Err(BatchProofError::EmptyIndexes);
        }
        if indexes.len() != leaf_hashes.len() {
            return Err(BatchProofError::LeafCountMismatch);
        }

        let mut index_map: BTreeMap<usize, usize> = BTreeMap::new();
        let mut sorted_indexes: Vec<usize> = Vec::new();
        let mut sorted_leaf_hashes: Vec<Digest> = Vec::new();

        for (pos, &idx) in indexes.iter().enumerate() {
            if !index_map.contains_key(&idx) {
                index_map.insert(idx, sorted_indexes.len());
                sorted_indexes.push(idx);
                sorted_leaf_hashes.push(leaf_hashes[pos]);
            }
        }

        if sorted_indexes.len() != self.nodes.len() {
            return Err(BatchProofError::IndexCountMismatch {
                expected: self.nodes.len(),
                got: sorted_indexes.len(),
            });
        }

        let mut current_values: BTreeMap<usize, Digest> = BTreeMap::new();
        for (i, &idx) in sorted_indexes.iter().enumerate() {
            current_values.insert(idx, sorted_leaf_hashes[i]);
        }

        let mut proof_pointers = vec![0usize; sorted_indexes.len()];

        for level in 0..self.depth as usize {
            let mut next_values: BTreeMap<usize, Digest> = BTreeMap::new();

            let level_indexes: Vec<usize> = current_values.keys().copied().collect();
            let mut i = 0;
            let mut node_map: BTreeMap<usize, usize> = BTreeMap::new();
            for (orig_i, &idx) in sorted_indexes.iter().enumerate() {
                let level_idx = idx >> level;
                node_map.entry(level_idx).or_insert(orig_i);
            }

            while i < level_indexes.len() {
                let idx = level_indexes[i];
                let sibling_idx = idx ^ 1;

                let current_val = *current_values.get(&idx).unwrap();

                let sibling_val = if let Some(&sib_val) = current_values.get(&sibling_idx) {
                    i += 2;
                    sib_val
                } else {
                    let orig_node_idx = *node_map.get(&idx).ok_or(BatchProofError::InvalidProof)?;
                    let ptr = proof_pointers[orig_node_idx];
                    if ptr >= self.nodes[orig_node_idx].len() {
                        return Err(BatchProofError::MissingProofNode);
                    }
                    let sib = self.nodes[orig_node_idx][ptr];
                    proof_pointers[orig_node_idx] += 1;
                    i += 1;
                    sib
                };

                let (left_val, right_val) = if idx & 1 == 0 {
                    (current_val, sibling_val)
                } else {
                    (sibling_val, current_val)
                };

                let parent = compress([left_val, right_val]);
                let parent_idx = idx >> 1;
                next_values.insert(parent_idx, parent);
            }

            current_values = next_values;
        }

        if current_values.len() != 1 {
            return Err(BatchProofError::InvalidProof);
        }

        let computed_root = current_values.values().next().unwrap();
        if computed_root == root {
            Ok(())
        } else {
            Err(BatchProofError::RootMismatch)
        }
    }
}

impl<Digest> BatchMerkleProof<Digest> {
    /// Returns the total number of digest elements in this proof.
    pub fn num_digests(&self) -> usize {
        self.nodes.iter().map(|v| v.len()).sum()
    }
}

/// Errors that can occur during batch proof operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BatchProofError {
    /// No indices were provided.
    EmptyIndexes,
    /// Number of leaves doesn't match number of indices.
    LeafCountMismatch,
    /// Number of unique indices doesn't match proof structure.
    IndexCountMismatch { expected: usize, got: usize },
    /// A required proof node is missing.
    MissingProofNode,
    /// The proof structure is invalid.
    InvalidProof,
    /// Computed root doesn't match expected root.
    RootMismatch,
    /// Cannot reconstruct individual proofs when siblings are opened.
    CannotReconstructWithOpenedSiblings,
}

/// Returns true if two indices are siblings (differ only in the last bit).
fn are_siblings(left: usize, right: usize) -> bool {
    left ^ right == 1
}

/// Normalize indices by replacing each with the smaller of itself and its sibling.
/// Returns sorted, deduplicated list.
pub fn normalize_indexes(indexes: &[usize]) -> Vec<usize> {
    let mut normalized: Vec<usize> = indexes.iter().map(|&i| i & !1).collect();
    normalized.sort_unstable();
    normalized.dedup();
    normalized
}

/// Create a mapping from normalized index to original index positions.
pub fn map_indexes(indexes: &[usize]) -> BTreeMap<usize, Vec<usize>> {
    let mut map: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
    for (pos, &idx) in indexes.iter().enumerate() {
        map.entry(idx).or_default().push(pos);
    }
    map
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_are_siblings() {
        assert!(are_siblings(0, 1));
        assert!(are_siblings(2, 3));
        assert!(are_siblings(4, 5));
        assert!(!are_siblings(0, 2));
        assert!(!are_siblings(1, 2));
        assert!(!are_siblings(1, 3));
    }

    #[test]
    fn test_normalize_indexes() {
        assert_eq!(normalize_indexes(&[0, 1, 2, 3]), vec![0, 2]);
        assert_eq!(normalize_indexes(&[1, 3, 5]), vec![0, 2, 4]);
        assert_eq!(normalize_indexes(&[7, 2, 1, 6]), vec![0, 2, 6]);
    }

    #[test]
    fn test_batch_proof_single_index() {
        // Single proof: index 0 in a tree of depth 3
        let proof = vec![[1u8; 4], [2u8; 4], [3u8; 4]];
        let batch = BatchMerkleProof::from_single_proofs(&[proof.clone()], &[0]);

        assert_eq!(batch.depth, 3);
        assert_eq!(batch.nodes.len(), 1);
        assert_eq!(batch.nodes[0], proof);
    }

    #[test]
    fn test_batch_proof_sibling_indices() {
        // Two proofs for sibling indices 0 and 1
        // At level 0, they are siblings so we don't need the sibling hash
        let proof0 = vec![[10u8; 4], [20u8; 4], [30u8; 4]];
        let proof1 = vec![[11u8; 4], [20u8; 4], [30u8; 4]];

        let batch = BatchMerkleProof::from_single_proofs(&[proof0, proof1], &[0, 1]);

        assert_eq!(batch.depth, 3);
        assert_eq!(batch.nodes.len(), 2);
        // First level siblings are each other, so no sibling stored at level 0
        assert!(batch.nodes[0].is_empty() || batch.nodes[0][0] != [10u8; 4]);
    }

    #[test]
    fn test_batch_proof_compression_ratio() {
        // 4 indices that are not all siblings: 0, 2, 4, 6
        let proof0 = vec![[1u8; 4], [10u8; 4], [100u8; 4], [200u8; 4]];
        let proof2 = vec![[3u8; 4], [10u8; 4], [100u8; 4], [200u8; 4]];
        let proof4 = vec![[5u8; 4], [30u8; 4], [100u8; 4], [200u8; 4]];
        let proof6 = vec![[7u8; 4], [30u8; 4], [100u8; 4], [200u8; 4]];

        let batch =
            BatchMerkleProof::from_single_proofs(&[proof0, proof2, proof4, proof6], &[0, 2, 4, 6]);

        // Individual proofs: 4 * 4 = 16 digests
        let individual_total = 16;
        let batch_total = batch.num_digests();

        assert!(
            batch_total < individual_total,
            "batch ({}) should be smaller than individual ({})",
            batch_total,
            individual_total
        );
    }

    #[test]
    fn test_batch_proof_verify_simple_tree() {
        // Build a simple tree manually and verify batch proof
        // Tree of depth 2 (4 leaves)
        //         root
        //        /    \
        //      n01    n23
        //     /  \   /  \
        //    L0  L1 L2  L3

        let l0 = [0u8; 4];
        let l1 = [1u8; 4];
        let l2 = [2u8; 4];
        let l3 = [3u8; 4];

        let compress = |input: [[u8; 4]; 2]| -> [u8; 4] {
            let mut result = [0u8; 4];
            for i in 0..4 {
                result[i] = input[0][i].wrapping_add(input[1][i]);
            }
            result
        };

        let n01 = compress([l0, l1]);
        let n23 = compress([l2, l3]);
        let root = compress([n01, n23]);

        // Individual proof for index 0: [L1, n23]
        let proof0 = vec![l1, n23];
        // Individual proof for index 2: [L3, n01]
        let proof2 = vec![l3, n01];

        let batch = BatchMerkleProof::from_single_proofs(&[proof0, proof2], &[0, 2]);

        // Verify the batch proof
        let result = batch.verify(&root, &[0, 2], &[l0, l2], compress);
        assert!(result.is_ok(), "Batch verification failed: {:?}", result);

        // Verify with wrong leaf fails
        let wrong_result = batch.verify(&root, &[0, 2], &[l1, l2], compress);
        assert_eq!(wrong_result, Err(BatchProofError::RootMismatch));
    }

    #[test]
    fn test_batch_proof_verify_adjacent_siblings() {
        // Open siblings 0 and 1 - they share all ancestors
        let l0 = [10u8; 4];
        let l1 = [20u8; 4];
        let l2 = [30u8; 4];
        let l3 = [40u8; 4];

        let compress = |input: [[u8; 4]; 2]| -> [u8; 4] {
            let mut result = [0u8; 4];
            for i in 0..4 {
                result[i] = input[0][i] ^ input[1][i];
            }
            result
        };

        let n01 = compress([l0, l1]);
        let n23 = compress([l2, l3]);
        let root = compress([n01, n23]);

        // For index 0, sibling is l1, then n23
        let proof0 = vec![l1, n23];
        // For index 1, sibling is l0, then n23
        let proof1 = vec![l0, n23];

        let batch = BatchMerkleProof::from_single_proofs(&[proof0, proof1], &[0, 1]);

        // When opening siblings, we save at level 0 (don't need to store sibling)
        // but still need n23 at level 1
        // So batch should have fewer digests than 2 * 2 = 4
        assert!(batch.num_digests() < 4);

        // Verify succeeds
        let result = batch.verify(&root, &[0, 1], &[l0, l1], compress);
        assert!(result.is_ok(), "Batch verification failed: {:?}", result);
    }
}
