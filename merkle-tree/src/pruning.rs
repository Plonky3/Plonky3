//! Merkle path pruning: compact multi-opening proofs via shared-ancestor deduplication.
//!
//! # Problem
//!
//! When a STARK prover opens k leaf indices in a Merkle tree of height h,
//! each opening produces h levels of sibling digests for verification — a total of k * h digest-levels.
//!
//! Many of these are **redundant**: leaves that are close in the tree share
//! ancestors, and their authentication paths overlap above the Lowest Common Ancestor (LCA).
//!
//! # Solution
//!
//! This module eliminates redundant sibling digests by exploiting tree structure:
//!
//! 1. Sort all opened paths by leaf index.
//! 2. For each consecutive pair, compute their LCA level using the per-level
//!    arity (stored as log_2 shifts).
//! 3. Each path only emits siblings **below** its LCA with the previous path.
//! 4. During restoration, siblings above the LCA are copied from the previous path.
//!
//! # Visual example (binary tree)
//!
//! Height-3 tree with 8 leaves, opening leaves 1, 2, and 5:
//!
//! ```text
//!                         [root]                      Level 3
//!                       /        \
//!                    [A]          [B]                 Level 2
//!                   /   \        /   \
//!                 [C]   [D]    [E]   [F]              Level 1
//!                 / \   / \    / \   / \
//!                0   1 2   3  4   5 6   7             Level 0 (leaves)
//!                    ^ ^          ^
//!                    opened leaves
//! ```
//!
//! **Without pruning** (3 paths x 3 levels = 9 sibling digests):
//! - Leaf 1: siblings `[0, D, B]`       (3 digests)
//! - Leaf 2: siblings `[3, C, B]`       (3 digests)
//! - Leaf 5: siblings `[4, F, A]`       (3 digests)
//!
//! **With pruning** (only 7 sibling digests):
//! - Leaf 1: siblings `[0, D, B]`       (3 digests — first path, no sharing)
//! - Leaf 2: siblings `[3, C]`          (2 digests — LCA at level 2, share level 2)
//! - Leaf 5: siblings `[4, F, A]`       (3 digests — LCA at root, no sharing)

use alloc::vec;
use alloc::vec::Vec;

use serde::{Deserialize, Serialize};

/// A full (unpruned) Merkle authentication path for a single leaf.
///
/// All sibling digests are stored in a **single contiguous allocation**.
///
/// The number of siblings at each tree level depends on the compression
/// arity at that level: 2^shift - 1 digests per level.
/// Level boundaries can be recovered from the per-level shift schedule.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(bound(serialize = "[D; DIGEST_ELEMS]: Serialize"))]
#[serde(bound(deserialize = "[D; DIGEST_ELEMS]: serde::de::DeserializeOwned"))]
pub(crate) struct MerkleAuthPath<D, const DIGEST_ELEMS: usize> {
    /// Index of this leaf in the tree (0-based).
    pub leaf_index: usize,

    /// All sibling digests concatenated: level 0 first, then level 1, etc.
    ///
    /// The count at each level is 2^shift - 1, where shift is the log_2
    /// of the compression arity at that level.
    pub siblings: Vec<[D; DIGEST_ELEMS]>,
}

/// Compact representation of multiple Merkle authentication paths with
/// redundant sibling digests removed.
///
/// Paths are sorted by leaf index and deduplicated.
///
/// Each path stores only the siblings below its LCA with the previous path;
/// the rest are reconstructed during restoration by copying from that previous path.
///
/// # Wire format
///
/// - Tree depth and per-level arity are not stored.
/// - The verifier rederives both from the committed matrix dimensions.
/// - Doing so shrinks the proof and removes a DoS surface where a malicious
///   schedule could dictate per-level sibling counts.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(bound(serialize = "[D; DIGEST_ELEMS]: Serialize"))]
#[serde(bound(deserialize = "[D; DIGEST_ELEMS]: serde::de::DeserializeOwned"))]
pub struct PrunedMerklePaths<D, const DIGEST_ELEMS: usize> {
    /// Permutation mapping original input order to sorted/deduplicated index.
    ///
    /// Entry i holds the index into the sorted path array for the i-th
    /// original query.
    pub original_order: Vec<u32>,

    /// Pruned authentication paths, sorted by leaf index.
    pub paths: Vec<PrunedPath<D, DIGEST_ELEMS>>,
}

/// A single pruned Merkle authentication path.
///
/// Contains only the siblings below this path's LCA with the previous sorted path.
///
/// Siblings at and above the LCA are shared with the neighbor and omitted.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(bound(serialize = "[D; DIGEST_ELEMS]: Serialize"))]
#[serde(bound(deserialize = "[D; DIGEST_ELEMS]: serde::de::DeserializeOwned"))]
pub struct PrunedPath<D, const DIGEST_ELEMS: usize> {
    /// Leaf index in the original tree.
    pub leaf_index: usize,
    /// Flat sibling digests for the first few levels only (below LCA).
    pub siblings: Vec<[D; DIGEST_ELEMS]>,
}

/// Total number of sibling digests for the first few levels.
#[inline]
fn total_siblings_for_levels(num_levels: usize, shift_schedule: &[u32]) -> usize {
    // Sum 2^shift - 1 for each of the first few levels.
    //
    //   Example: shift_schedule = [1, 2, 1], num_levels = 2
    //     level 0: 2^1 - 1 = 1
    //     level 1: 2^2 - 1 = 3
    //     → total = 4
    shift_schedule[..num_levels]
        .iter()
        .map(|&s| (1usize << s) - 1)
        .sum()
}

/// O(1) LCA for purely binary trees.
///
/// XOR the two leaf indices and count the bit-width of the result.
/// The highest differing bit position is exactly the tree level where
/// the two paths diverge.
///
/// For identical leaves, returns 1 (they share the parent at level 1).
///
/// # Returns
///
/// The smallest level at which both leaves share the same ancestor,
/// clamped to the total number of levels.
#[inline]
fn first_shared_level_binary(a: usize, b: usize, num_levels: usize) -> usize {
    let xor = a ^ b;
    if xor == 0 {
        // Same leaf — they trivially share the same parent.
        return 1;
    }
    // Bit-width of the XOR = position of the highest differing bit + 1.
    let level = (usize::BITS - xor.leading_zeros()) as usize;
    level.min(num_levels)
}

/// Generic LCA via per-level shift loop.
///
/// Right-shifts both indices by the arity at each level.
/// The first level where they become equal is the LCA.
///
/// For binary trees the caller should use the O(1) fast path instead;
/// that check must be hoisted outside hot loops.
#[inline]
fn first_shared_level_generic(a: usize, b: usize, shift_schedule: &[u32]) -> usize {
    let mut a_idx = a;
    let mut b_idx = b;
    for (level, &shift) in shift_schedule.iter().enumerate() {
        // Divide both indices by the arity at this level (via shift).
        a_idx >>= shift;
        b_idx >>= shift;
        // If both land in the same group, they share this ancestor.
        if a_idx == b_idx {
            return level + 1;
        }
    }
    // No convergence within the schedule — share only the root.
    shift_schedule.len()
}

/// Eliminates redundant sibling levels shared between adjacent sorted paths.
///
/// # Algorithm
///
/// 1. Sort a lightweight index array by leaf index (avoids moving large structs).
/// 2. Deduplicate consecutive entries with the same leaf index.
/// 3. For each path, compute the first shared level with the previous path
///    and emit only the siblings below that level.
///
/// # Performance
///
/// - Binary trees: O(1) LCA via bitwise XOR (checked once, hoisted).
/// - N-ary trees: O(h) LCA via shift loop per consecutive pair.
/// - Sorting: lightweight u32 index array, not the full path structs.
pub(crate) fn prune_paths<D, const DIGEST_ELEMS: usize>(
    num_levels: usize,
    shift_schedule: &[u32],
    paths: &[MerkleAuthPath<D, DIGEST_ELEMS>],
) -> PrunedMerklePaths<D, DIGEST_ELEMS>
where
    D: Clone + PartialEq,
{
    debug_assert_eq!(shift_schedule.len(), num_levels);

    // Empty input → empty output.
    if paths.is_empty() {
        return PrunedMerklePaths {
            original_order: vec![],
            paths: vec![],
        };
    }

    // Phase 1: Sort by leaf index.
    //
    // We sort a lightweight u32 index array instead of moving the full
    // auth-path structs (which can be hundreds of bytes each).
    // This keeps the sort entirely within L1 cache.
    let mut order: Vec<u32> = (0..paths.len() as u32).collect();
    order.sort_unstable_by_key(|&i| paths[i as usize].leaf_index);

    // Phase 2: Deduplicate consecutive entries with the same leaf index.
    //
    // When the same leaf is queried multiple times, we store it once
    // in the pruned output and record which original queries map to it.
    //
    //   Example: input queries [5, 1, 3, 1]
    //     sorted order:    [1, 1, 3, 5]
    //     after dedup:     [1, 3, 5]       (3 unique paths)
    //     original_order:  [2, 0, 1, 0]    (query 0→slot 2, query 1→slot 0, ...)
    let mut original_order = vec![0u32; paths.len()];
    let mut deduped_indices: Vec<u32> = Vec::with_capacity(paths.len());

    for &sorted_idx in &order {
        let leaf = paths[sorted_idx as usize].leaf_index;
        // Start a new deduped entry only when the leaf differs from the last one.
        if deduped_indices
            .last()
            .is_none_or(|&prev| paths[prev as usize].leaf_index != leaf)
        {
            deduped_indices.push(sorted_idx);
        }
        // Record which deduped slot this original query maps to.
        original_order[sorted_idx as usize] = (deduped_indices.len() - 1) as u32;
    }

    // Phase 3: Build pruned paths.
    //
    // For each unique leaf (in sorted order), compute how many levels
    // of siblings to keep. The first path keeps everything. Subsequent
    // paths keep only the levels below their LCA with the predecessor —
    // the rest are identical and will be copied during restoration.
    //
    //   Example: height-3 binary tree, sorted leaves [0, 1, 4]
    //     path 0 (leaf 0): first → keep all 3 levels
    //     path 1 (leaf 1): LCA with leaf 0 at level 1 → keep 1 level
    //     path 2 (leaf 4): LCA with leaf 1 at level 3 → keep 3 levels
    let n = deduped_indices.len();
    let mut pruned_paths = Vec::with_capacity(n);

    // Hoist the binary-tree check: O(h) once, not O(h) per pair.
    // When all shifts are 1, the O(1) XOR fast path applies.
    let is_purely_binary = shift_schedule.iter().all(|&s| s == 1);

    for j in 0..n {
        let path = &paths[deduped_indices[j] as usize];

        // First path: keep all levels (no predecessor to share with).
        // Later paths: keep only levels below the LCA with the predecessor.
        let keep_levels = if j == 0 {
            num_levels
        } else {
            let prev_leaf = paths[deduped_indices[j - 1] as usize].leaf_index;
            if is_purely_binary {
                // O(1): XOR the two leaf indices, count bit-width.
                first_shared_level_binary(prev_leaf, path.leaf_index, num_levels)
            } else {
                // O(h): right-shift both indices level by level until they match.
                first_shared_level_generic(prev_leaf, path.leaf_index, shift_schedule)
            }
        };

        // Clamp to the actual number of levels (safety guard).
        let keep = keep_levels.min(num_levels);
        // Convert level count → flat-buffer element count.
        //
        //   Example: keep = 2, shift_schedule = [1, 2, 1]
        //     level 0: 1 sibling, level 1: 3 siblings → 4 elements total
        let sibling_count = total_siblings_for_levels(keep, shift_schedule);

        pruned_paths.push(PrunedPath {
            leaf_index: path.leaf_index,
            // Slice the flat buffer up to the computed boundary.
            siblings: path.siblings[..sibling_count].to_vec(),
        });
    }

    PrunedMerklePaths {
        original_order,
        paths: pruned_paths,
    }
}

/// Restores full Merkle authentication paths from a pruned representation.
///
/// Processes paths in sorted order.
/// Each path takes its own stored siblings (below LCA) and copies the
/// remaining siblings (at and above LCA) from the previous restored path.
///
/// # Trust model
///
/// - The full-sibling count is supplied by the caller, never read from the proof.
/// - A malicious prover cannot use it to inflate per-path allocations.
///
/// # Returns
///
/// - Full authentication paths in the **original input order**.
/// - `None` if the proof data is malformed.
pub(crate) fn restore_paths<D, const DIGEST_ELEMS: usize>(
    pruned: &PrunedMerklePaths<D, DIGEST_ELEMS>,
    full_sibling_count: usize,
) -> Option<Vec<MerkleAuthPath<D, DIGEST_ELEMS>>>
where
    D: Clone + Default,
{
    let n = pruned.paths.len();

    // Zero paths with zero queries is valid (empty proof).
    // Zero paths with nonzero queries is malformed.
    if n == 0 {
        return if pruned.original_order.is_empty() {
            Some(vec![])
        } else {
            None
        };
    }

    let mut restored: Vec<MerkleAuthPath<D, DIGEST_ELEMS>> = Vec::with_capacity(n);

    for i in 0..n {
        let pruned_path = &pruned.paths[i];
        let kept_siblings = pruned_path.siblings.len();

        // Allocate the full-size buffer once (no resizing).
        let mut siblings = Vec::with_capacity(full_sibling_count);

        // Lower portion: the unique siblings this path stored (below LCA).
        // These differ from the predecessor and were kept during pruning.
        siblings.extend_from_slice(&pruned_path.siblings);

        // Upper portion: shared siblings copied from the previous path.
        // Both paths traverse the same nodes above the LCA, so their
        // siblings there are identical — just memcpy the tail.
        //
        //   Example: full = 5 siblings, this path stored 2
        //     this path's buffer:  [s0, s1, ?, ?, ?]
        //     previous path:       [_, _, s2, s3, s4]   (shared above LCA)
        //     after copy:          [s0, s1, s2, s3, s4]
        if let Some(prev) = restored.last()
            && kept_siblings < full_sibling_count
        {
            siblings.extend_from_slice(&prev.siblings[kept_siblings..]);
        }

        // Sanity check: restored path must be exactly the right size.
        // If not, the pruned proof is malformed.
        if siblings.len() != full_sibling_count {
            return None;
        }

        restored.push(MerkleAuthPath {
            leaf_index: pruned_path.leaf_index,
            siblings,
        });
    }

    // Reorder from sorted/deduped order back to the caller's original query
    // order. Each entry in the order mapping points to the deduped path slot.
    pruned
        .original_order
        .iter()
        .map(|&idx| restored.get(idx as usize).cloned())
        .collect()
}

// Tests

#[cfg(test)]
mod tests {
    use alloc::vec;
    use alloc::vec::Vec;

    use proptest::prelude::*;

    use super::*;

    /// Total number of individual sibling digests across all pruned paths.
    ///
    /// Compare against the unpruned total (paths * full sibling count)
    /// to measure the compression ratio.
    fn pruned_sibling_count<D, const DIGEST_ELEMS: usize>(
        pruned: &PrunedMerklePaths<D, DIGEST_ELEMS>,
    ) -> usize {
        pruned.paths.iter().map(|p| p.siblings.len()).sum()
    }

    fn binary_shifts(h: usize) -> Vec<u32> {
        // All levels binary (arity 2 → shift 1).
        vec![1; h]
    }

    fn quad_shifts(h: usize) -> Vec<u32> {
        // All levels 4-ary (arity 4 → shift 2).
        vec![2; h]
    }

    fn mock_path<const DE: usize>(leaf_index: usize, h: usize) -> MerkleAuthPath<u32, DE> {
        // Deterministic siblings: value encodes (leaf_index, level) for easy debugging.
        MerkleAuthPath {
            leaf_index,
            siblings: (0..h)
                .map(|lvl| [(leaf_index * 100 + lvl) as u32; DE])
                .collect(),
        }
    }

    fn realistic_binary_path<const DE: usize>(
        leaf_index: usize,
        h: usize,
    ) -> MerkleAuthPath<u32, DE> {
        // Binary tree mock where siblings above the LCA are structurally
        // identical for leaves sharing that ancestor.
        // Sibling at level l = the "other child" index at that level.
        MerkleAuthPath {
            leaf_index,
            siblings: (0..h)
                .map(|lvl| [((leaf_index >> lvl) ^ 1) as u32; DE])
                .collect(),
        }
    }

    fn realistic_quad_path<const DE: usize>(
        leaf_index: usize,
        num_levels: usize,
    ) -> MerkleAuthPath<u32, DE> {
        // 4-ary tree mock: 3 siblings per level, shared above LCA.
        // Group index at each level determines the sibling values.
        let mut siblings = Vec::new();
        for lvl in 0..num_levels {
            let group_idx = leaf_index / 4usize.pow(lvl as u32 + 1);
            for s in 0..3u32 {
                siblings.push([(group_idx * 100 + lvl * 10 + s as usize) as u32; DE]);
            }
        }
        MerkleAuthPath {
            leaf_index,
            siblings,
        }
    }

    // LCA tests

    #[test]
    fn test_lca_binary_fast_path() {
        // Binary tree, height 3 (8 leaves).
        //
        //          [root]            Level 3
        //        /        \
        //     [A]          [B]       Level 2
        //    /   \        /   \
        //  [C]   [D]    [E]   [F]    Level 1
        //  / \   / \    / \   / \
        // 0   1 2   3  4   5 6   7   Level 0

        // Adjacent siblings share at level 1.
        assert_eq!(first_shared_level_binary(0, 1, 3), 1);
        assert_eq!(first_shared_level_binary(4, 5, 3), 1);

        // Leaves in the same quadrant share at level 2.
        assert_eq!(first_shared_level_binary(0, 3, 3), 2);
        assert_eq!(first_shared_level_binary(4, 6, 3), 2);

        // Leaves in opposite halves share only at the root.
        assert_eq!(first_shared_level_binary(0, 7, 3), 3);

        // Same leaf: trivially share at level 1.
        assert_eq!(first_shared_level_binary(5, 5, 3), 1);
    }

    #[test]
    fn test_lca_generic_binary() {
        // Generic loop path must produce identical results for binary trees.
        let s = binary_shifts(3);
        assert_eq!(first_shared_level_generic(0, 1, &s), 1);
        assert_eq!(first_shared_level_generic(0, 3, &s), 2);
        assert_eq!(first_shared_level_generic(0, 7, &s), 3);
        assert_eq!(first_shared_level_generic(4, 5, &s), 1);
    }

    #[test]
    fn test_lca_4ary() {
        // 4-ary tree, height 3 (64 leaves).
        // Leaves 0-3 share the same parent → LCA at level 1.
        let s = quad_shifts(3);
        assert_eq!(first_shared_level_generic(0, 1, &s), 1);
        assert_eq!(first_shared_level_generic(0, 3, &s), 1);

        // Leaves 0 and 4 are in different level-0 groups, share at level 2.
        assert_eq!(first_shared_level_generic(0, 4, &s), 2);
        assert_eq!(first_shared_level_generic(0, 15, &s), 2);

        // Leaves in different top quadrants share at level 3 (root).
        assert_eq!(first_shared_level_generic(0, 16, &s), 3);
        assert_eq!(first_shared_level_generic(0, 63, &s), 3);
    }

    #[test]
    fn test_lca_mixed_schedule() {
        // Mixed arity: 4-ary, binary, 4-ary → 4 * 2 * 4 = 32 leaves.
        let s = vec![2u32, 1, 2];

        // Leaves 0 and 3 are in the same 4-group at level 0.
        assert_eq!(first_shared_level_generic(0, 3, &s), 1);

        // Leaves 0 and 4 span different 4-groups but the same binary parent.
        assert_eq!(first_shared_level_generic(0, 4, &s), 2);

        // Leaves 0 and 8 span different top-level groups.
        assert_eq!(first_shared_level_generic(0, 8, &s), 3);
    }

    #[test]
    fn test_lca_symmetry() {
        // LCA must be symmetric: swapping the two leaves must not change the result.
        let s = vec![2u32, 1, 2];
        for a in 0..32 {
            for b in 0..32 {
                assert_eq!(
                    first_shared_level_generic(a, b, &s),
                    first_shared_level_generic(b, a, &s)
                );
            }
        }
    }

    #[test]
    fn test_binary_fast_path_matches_generic() {
        // The O(1) binary fast path must produce identical results to the
        // generic O(h) loop for all pairs in a binary tree.
        let s = binary_shifts(5);
        for a in 0..32 {
            for b in 0..32 {
                assert_eq!(
                    first_shared_level_generic(a, b, &s),
                    first_shared_level_binary(a, b, 5),
                    "mismatch for a={a}, b={b}"
                );
            }
        }
    }

    // Total siblings tests

    #[test]
    fn test_total_siblings_binary() {
        // Binary tree: 1 sibling per level.
        //   1 level  → 1
        //   3 levels → 1 + 1 + 1 = 3
        //   5 levels → 5
        let s = binary_shifts(5);
        assert_eq!(total_siblings_for_levels(0, &s), 0);
        assert_eq!(total_siblings_for_levels(1, &s), 1);
        assert_eq!(total_siblings_for_levels(3, &s), 3);
        assert_eq!(total_siblings_for_levels(5, &s), 5);
    }

    #[test]
    fn test_total_siblings_4ary() {
        // 4-ary tree: 3 siblings per level.
        //   1 level  → 3
        //   2 levels → 3 + 3 = 6
        //   3 levels → 9
        let s = quad_shifts(3);
        assert_eq!(total_siblings_for_levels(0, &s), 0);
        assert_eq!(total_siblings_for_levels(1, &s), 3);
        assert_eq!(total_siblings_for_levels(2, &s), 6);
        assert_eq!(total_siblings_for_levels(3, &s), 9);
    }

    #[test]
    fn test_total_siblings_mixed() {
        // Mixed: binary (1), 4-ary (3), 8-ary (7) → cumulative 1, 4, 11.
        let s = vec![1u32, 2, 3];
        assert_eq!(total_siblings_for_levels(0, &s), 0);
        assert_eq!(total_siblings_for_levels(1, &s), 1);
        assert_eq!(total_siblings_for_levels(2, &s), 4);
        assert_eq!(total_siblings_for_levels(3, &s), 11);
    }

    // Pruning tests

    #[test]
    fn test_prune_empty() {
        // No paths → empty pruned output.
        let pruned = prune_paths::<u32, 2>(3, &binary_shifts(3), &[]);
        assert!(pruned.paths.is_empty());
    }

    #[test]
    fn test_prune_single_path() {
        // A single path has no predecessor to share with → all levels kept.
        // Binary: 1 sibling per level × 3 levels = 3 total.
        let path = mock_path::<2>(5, 3);
        let pruned = prune_paths(3, &binary_shifts(3), &[path]);
        assert_eq!(pruned.paths.len(), 1);
        assert_eq!(pruned.paths[0].siblings.len(), 3);
    }

    #[test]
    fn test_prune_adjacent_binary() {
        // Leaves 4 and 5 are binary siblings → LCA at level 1.
        // First path: all 3 levels. Second path: only level 0.
        let paths = [mock_path::<2>(4, 3), mock_path::<2>(5, 3)];
        let pruned = prune_paths(3, &binary_shifts(3), &paths);
        assert_eq!(pruned.paths[0].siblings.len(), 3);
        assert_eq!(pruned.paths[1].siblings.len(), 1);
    }

    #[test]
    fn test_prune_all_leaves_binary_height3() {
        // All 8 leaves in a height-3 binary tree.
        //
        // Expected sibling counts per sorted path (1 sibling per binary level):
        //   [3, 1, 2, 1, 3, 1, 2, 1]
        //
        // Total pruned siblings: 14 vs unpruned 8 * 3 = 24.
        let h = 3;
        let paths: Vec<_> = (0..8).map(|i| mock_path::<2>(i, h)).collect();
        let pruned = prune_paths(h, &binary_shifts(h), &paths);
        let counts: Vec<usize> = pruned.paths.iter().map(|p| p.siblings.len()).collect();
        assert_eq!(counts, vec![3, 1, 2, 1, 3, 1, 2, 1]);
        assert_eq!(pruned_sibling_count(&pruned), 14);
    }

    #[test]
    fn test_prune_4ary_adjacent() {
        // Leaves 0-3 are siblings in a 4-ary tree → all share level 1.
        // First path: both levels (6 siblings). Remaining paths: only level 0 (3 siblings).
        let s = quad_shifts(2);
        let paths: Vec<_> = (0..4).map(|i| realistic_quad_path::<2>(i, 2)).collect();
        let pruned = prune_paths(2, &s, &paths);
        assert_eq!(pruned.paths[0].siblings.len(), 6);
        assert_eq!(pruned.paths[1].siblings.len(), 3);
        assert_eq!(pruned.paths[2].siblings.len(), 3);
        assert_eq!(pruned.paths[3].siblings.len(), 3);
    }

    #[test]
    fn test_prune_duplicate_indices() {
        // Two queries for the same leaf → deduplicated to one path.
        // Both original queries map to the same deduped slot.
        let paths = [mock_path::<2>(3, 3), mock_path::<2>(3, 3)];
        let pruned = prune_paths(3, &binary_shifts(3), &paths);
        assert_eq!(pruned.paths.len(), 1);
        assert_eq!(pruned.original_order, vec![0, 0]);
    }

    #[test]
    fn test_original_order_preservation() {
        // Input order [5, 1, 3] → sorted [1, 3, 5] at indices [0, 1, 2].
        // Original query 0 (leaf 5) maps to sorted index 2, etc.
        let paths = [
            mock_path::<2>(5, 3),
            mock_path::<2>(1, 3),
            mock_path::<2>(3, 3),
        ];
        let pruned = prune_paths(3, &binary_shifts(3), &paths);
        assert_eq!(pruned.original_order, vec![2, 0, 1]);
    }

    #[test]
    fn test_proof_size_smaller_for_overlapping() {
        // Clustered leaves must produce a strictly smaller pruned proof.
        let h = 5;
        let paths: Vec<_> = [2, 3, 6, 7].iter().map(|&i| mock_path::<2>(i, h)).collect();
        let unpruned: usize = paths.iter().map(|p| p.siblings.len()).sum();
        let pruned = prune_paths(h, &binary_shifts(h), &paths);
        assert!(pruned_sibling_count(&pruned) < unpruned);
    }

    // Roundtrip tests

    #[test]
    fn test_roundtrip_single() {
        // Single path: prune → restore must be identity.
        let h = 4;
        let s = binary_shifts(h);
        let full = total_siblings_for_levels(h, &s);
        let paths = [realistic_binary_path::<2>(3, h)];
        let pruned = prune_paths(h, &s, &paths);
        let restored = restore_paths(&pruned, full).unwrap();
        assert_eq!(restored, paths);
    }

    #[test]
    fn test_roundtrip_binary_all_leaves() {
        // All 8 leaves in a height-3 binary tree: full roundtrip.
        let h = 3;
        let s = binary_shifts(h);
        let full = total_siblings_for_levels(h, &s);
        let paths: Vec<_> = (0..8).map(|i| realistic_binary_path::<2>(i, h)).collect();
        let pruned = prune_paths(h, &s, &paths);
        let restored = restore_paths(&pruned, full).unwrap();
        assert_eq!(restored, paths);
    }

    #[test]
    fn test_roundtrip_unordered() {
        // Unsorted input [5, 1, 3] must be restored in the original order.
        let h = 3;
        let s = binary_shifts(h);
        let full = total_siblings_for_levels(h, &s);
        let paths = [
            realistic_binary_path::<2>(5, h),
            realistic_binary_path::<2>(1, h),
            realistic_binary_path::<2>(3, h),
        ];
        let pruned = prune_paths(h, &s, &paths);
        let restored = restore_paths(&pruned, full).unwrap();
        assert_eq!(restored.as_slice(), &paths);
    }

    #[test]
    fn test_roundtrip_duplicates() {
        // Duplicate queries must roundtrip correctly.
        // Both copies must match their respective originals.
        let h = 3;
        let s = binary_shifts(h);
        let full = total_siblings_for_levels(h, &s);
        let paths = [
            realistic_binary_path::<2>(2, h),
            realistic_binary_path::<2>(5, h),
            realistic_binary_path::<2>(2, h),
        ];
        let pruned = prune_paths(h, &s, &paths);
        let restored = restore_paths(&pruned, full).unwrap();
        assert_eq!(restored.as_slice(), &paths);
    }

    #[test]
    fn test_roundtrip_4ary() {
        // All 16 leaves of a 2-level 4-ary tree.
        let s = quad_shifts(2);
        let full = total_siblings_for_levels(2, &s);
        let paths: Vec<_> = (0..16).map(|i| realistic_quad_path::<2>(i, 2)).collect();
        let pruned = prune_paths(2, &s, &paths);
        let restored = restore_paths(&pruned, full).unwrap();
        assert_eq!(restored, paths);
    }

    #[test]
    fn test_roundtrip_height5() {
        // Sparse query set across a height-5 binary tree.
        let h = 5;
        let s = binary_shifts(h);
        let full = total_siblings_for_levels(h, &s);
        let indices = [0, 3, 7, 12, 15, 16, 20, 31];
        let paths: Vec<_> = indices
            .iter()
            .map(|&i| realistic_binary_path::<4>(i, h))
            .collect();
        let pruned = prune_paths(h, &s, &paths);
        let restored = restore_paths(&pruned, full).unwrap();
        assert_eq!(restored, paths);
    }

    #[test]
    fn test_roundtrip_mixed_arity() {
        // Mixed schedule [1, 2, 1] (binary, 4-ary, binary) → 16 leaves.
        // Exercises both the LCA loop and the per-level sibling counts.
        let shifts = vec![1u32, 2, 1];
        let full = total_siblings_for_levels(shifts.len(), &shifts);
        let indices = [0, 1, 4, 7, 8, 11, 15];
        let paths: Vec<_> = indices
            .iter()
            .map(|&i| realistic_nary_path::<2>(i, &shifts))
            .collect();
        let pruned = prune_paths(shifts.len(), &shifts, &paths);
        let restored = restore_paths(&pruned, full).unwrap();
        assert_eq!(restored, paths);
    }

    // Error / bounds tests

    #[test]
    fn test_restore_empty() {
        // Zero paths with zero queries → valid empty result, regardless of the
        // sibling count the verifier expected.
        let pruned = PrunedMerklePaths::<u32, 2> {
            original_order: vec![],
            paths: vec![],
        };
        assert!(restore_paths(&pruned, 3).unwrap().is_empty());
    }

    #[test]
    fn test_restore_rejects_inconsistent_siblings() {
        // First path stores only 1 level of siblings but the verifier expects 3.
        // With no predecessor to copy from, restoration must fail.
        let pruned = PrunedMerklePaths::<u32, 2> {
            original_order: vec![0],
            paths: vec![PrunedPath {
                leaf_index: 0,
                siblings: vec![[0; 2]],
            }],
        };
        assert!(restore_paths(&pruned, 3).is_none());
    }

    #[test]
    fn test_restore_rejects_oversized_first_path() {
        // Invariant: the first path's stored siblings must equal the
        // verifier-derived full count. Anything bigger is malformed.
        //
        // Fixture state: verifier expects 3 siblings per full path.
        //
        // Mutation: first path stores 10 siblings (forged).
        //
        //     stored:    [s0, s1, s2, s3, s4, s5, s6, s7, s8, s9]   (len 10)
        //     expected:  [s0, s1, s2]                               (len 3)
        //     → 10 != 3 → reject
        //
        // Why this matters: trusting the prover-supplied length would let
        // a malicious proof inflate per-path allocations arbitrarily.
        let pruned = PrunedMerklePaths::<u32, 2> {
            original_order: vec![0],
            paths: vec![PrunedPath {
                leaf_index: 0,
                siblings: vec![[0; 2]; 10],
            }],
        };
        assert!(restore_paths(&pruned, 3).is_none());
    }

    // Size analysis

    #[test]
    fn test_pruned_size_leq_original() {
        // Invariant: pruning must never increase the total sibling count.
        for h in 2..6 {
            let n_leaves = 1 << h;
            for subset in 1..=n_leaves.min(16) {
                let paths: Vec<_> = (0..subset).map(|i| mock_path::<2>(i, h)).collect();
                let unpruned: usize = paths.iter().map(|p| p.siblings.len()).sum();
                let pruned = prune_paths(h, &binary_shifts(h), &paths);
                assert!(pruned_sibling_count(&pruned) <= unpruned);
            }
        }
    }

    // Proptests

    fn binary_tree_queries() -> impl Strategy<Value = (usize, Vec<usize>)> {
        // Random binary tree: height 1..10, up to 32 random leaf queries.
        (1..10usize).prop_flat_map(|h| {
            let n_leaves = 1usize << h;
            let indices = proptest::collection::vec(0..n_leaves, 1..=n_leaves.min(32));
            (Just(h), indices)
        })
    }

    fn quad_tree_queries() -> impl Strategy<Value = (usize, Vec<usize>)> {
        // Random 4-ary tree: height 1..5, up to 32 random leaf queries.
        (1..5usize).prop_flat_map(|h| {
            let n_leaves = 4usize.pow(h as u32);
            let indices = proptest::collection::vec(0..n_leaves, 1..=n_leaves.min(32));
            (Just(h), indices)
        })
    }

    /// Build a realistic mock path for any power-of-two arity schedule.
    ///
    /// For each level, generates (2^shift - 1) siblings whose values are
    /// determined by the ancestor group index at that level. This means
    /// leaves sharing an ancestor will have identical siblings above the LCA.
    fn realistic_nary_path<const DE: usize>(
        leaf_index: usize,
        shift_schedule: &[u32],
    ) -> MerkleAuthPath<u32, DE> {
        let mut siblings = Vec::new();
        let mut group_size = 1usize;
        for &shift in shift_schedule {
            // The arity at this level.
            let arity = 1usize << shift;
            // Group size below this level (product of all arities so far + this one).
            group_size *= arity;
            // Which group does this leaf belong to at this level?
            let group_idx = leaf_index / group_size;
            // Generate (arity - 1) sibling digests for this level.
            for s in 0..(arity - 1) {
                siblings.push([(group_idx * 1000 + s) as u32; DE]);
            }
        }
        MerkleAuthPath {
            leaf_index,
            siblings,
        }
    }

    /// Strategy for a tree with random per-level arities (2, 4, 8, or 16).
    /// Returns the shift schedule, total leaf count, and random query indices.
    fn mixed_arity_tree_queries() -> impl Strategy<Value = (Vec<u32>, usize, Vec<usize>)> {
        // 1..6 levels, each with a random power-of-two arity (shift 1..=4).
        proptest::collection::vec(1..=4u32, 1..6).prop_flat_map(|shifts| {
            // Total leaves = product of arities = 2^(sum of shifts).
            let total_shift: u32 = shifts.iter().sum();
            // Cap at 2^16 = 65536 leaves to keep tests fast.
            if total_shift > 16 {
                // Truncate the schedule to stay within bounds.
                let mut cumulative = 0u32;
                let truncated: Vec<u32> = shifts
                    .into_iter()
                    .take_while(|&s| {
                        cumulative += s;
                        cumulative <= 16
                    })
                    .collect();
                let n_leaves = 1usize << truncated.iter().sum::<u32>();
                let indices = proptest::collection::vec(0..n_leaves, 1..=n_leaves.min(32));
                (Just(truncated), Just(n_leaves), indices)
            } else {
                let n_leaves = 1usize << total_shift;
                let indices = proptest::collection::vec(0..n_leaves, 1..=n_leaves.min(32));
                (Just(shifts), Just(n_leaves), indices)
            }
        })
    }

    proptest! {
        #[test]
        fn proptest_binary_roundtrip((h, indices) in binary_tree_queries()) {
            // Prune → restore must recover the exact original paths.
            let s = binary_shifts(h);
            let full = total_siblings_for_levels(h, &s);
            let paths: Vec<_> = indices.iter().map(|&i| realistic_binary_path::<2>(i, h)).collect();
            let pruned = prune_paths(h, &s, &paths);
            let restored = restore_paths(&pruned, full).unwrap();
            prop_assert_eq!(&restored, &paths);
        }

        #[test]
        fn proptest_quad_roundtrip((h, indices) in quad_tree_queries()) {
            // Same roundtrip invariant for 4-ary trees.
            let s = quad_shifts(h);
            let full = total_siblings_for_levels(h, &s);
            let paths: Vec<_> = indices.iter().map(|&i| realistic_quad_path::<2>(i, h)).collect();
            let pruned = prune_paths(h, &s, &paths);
            let restored = restore_paths(&pruned, full).unwrap();
            prop_assert_eq!(&restored, &paths);
        }

        #[test]
        fn proptest_pruned_size_leq_original((h, indices) in binary_tree_queries()) {
            // Pruning must never increase total sibling count.
            let s = binary_shifts(h);
            let paths: Vec<_> = indices.iter().map(|&i| mock_path::<2>(i, h)).collect();
            let unpruned: usize = paths.iter().map(|p| p.siblings.len()).sum();
            let pruned = prune_paths(h, &s, &paths);
            prop_assert!(pruned_sibling_count(&pruned) <= unpruned);
        }

        #[test]
        fn proptest_binary_fast_path_matches_generic(a in 0..1024usize, b in 0..1024usize) {
            // The O(1) binary fast path must match the O(h) generic loop.
            let h = 10;
            let s = binary_shifts(h);
            prop_assert_eq!(
                first_shared_level_generic(a, b, &s),
                first_shared_level_binary(a, b, h)
            );
        }

        #[test]
        fn proptest_dedup_preserves_identity((h, mut indices) in binary_tree_queries()) {
            // Duplicating every query must produce identical restored paths
            // for the original and duplicated halves.
            let original_len = indices.len();
            indices.extend_from_slice(&indices.clone());
            let s = binary_shifts(h);
            let full = total_siblings_for_levels(h, &s);
            let paths: Vec<_> = indices.iter().map(|&i| realistic_binary_path::<2>(i, h)).collect();
            let pruned = prune_paths(h, &s, &paths);
            let restored = restore_paths(&pruned, full).unwrap();
            prop_assert_eq!(restored.len(), original_len * 2);
            for i in 0..original_len {
                prop_assert_eq!(&restored[i], &restored[i + original_len]);
            }
        }

        #[test]
        fn proptest_mixed_arity_roundtrip((shifts, _n_leaves, indices) in mixed_arity_tree_queries()) {
            // Roundtrip with random per-level arities (2, 4, 8, or 16).
            // Covers mixed schedules like [1, 3, 2] (binary, 8-ary, 4-ary).
            let h = shifts.len();
            let full = total_siblings_for_levels(h, &shifts);
            let paths: Vec<_> = indices.iter().map(|&i| realistic_nary_path::<2>(i, &shifts)).collect();
            let pruned = prune_paths(h, &shifts, &paths);
            let restored = restore_paths(&pruned, full).unwrap();
            prop_assert_eq!(&restored, &paths);
        }

        #[test]
        fn proptest_mixed_arity_size_leq_original((shifts, _n_leaves, indices) in mixed_arity_tree_queries()) {
            // Pruning must never increase sibling count, regardless of arity.
            let h = shifts.len();
            let paths: Vec<_> = indices.iter().map(|&i| realistic_nary_path::<2>(i, &shifts)).collect();
            let unpruned: usize = paths.iter().map(|p| p.siblings.len()).sum();
            let pruned = prune_paths(h, &shifts, &paths);
            prop_assert!(pruned_sibling_count(&pruned) <= unpruned);
        }
    }
}
