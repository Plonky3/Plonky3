//! Merkle path pruning: minimal multi-opening proofs via a level-by-level frontier.
//!
//! # Problem
//!
//! Opening `k` leaves in a height-`h` tree costs `k * h` sibling digests unpruned.
//! Many are redundant: an ancestor of another queried leaf is recomputable.
//! Such a sibling need not be sent.
//!
//! # Solution — optimal frontier
//!
//! Walk the tree bottom-up over the set of queried nodes (the "frontier").
//! At each level the frontier is grouped by parent.
//! For a group, a child digest is sent only when no query lies under it:
//!
//! - child covers a queried leaf  -> recomputable, send nothing.
//! - child covers no queried leaf -> boundary digest, must be sent.
//!
//! This is the strictly minimal set: every remaining sibling is forced by the tree.
//!
//! The queried indices come from the verifier's transcript, not the proof.
//! So the proof is just one flat list of boundary digests in frontier order.
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
//! Frontier walk (sent digests only):
//!
//! - Level 0: nodes {1,2,5}  -> send siblings 0, 3, 4       (their pairs are unqueried)
//! - Level 1: nodes {C,D,E}  -> C,D pair into A for free; send F   (E's pair)
//! - Level 2: nodes {A,B}    -> pair into root for free; send nothing
//!
//! Total: 4 digests (0, 3, 4, F) versus 9 unpruned.

use alloc::vec;
use alloc::vec::Vec;

use serde::{Deserialize, Serialize};

use crate::PrunedProofError;

/// A full (unpruned) Merkle authentication path for a single leaf.
///
/// All sibling digests live in one contiguous allocation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(bound(serialize = "[D; DIGEST_ELEMS]: Serialize"))]
#[serde(bound(deserialize = "[D; DIGEST_ELEMS]: serde::de::DeserializeOwned"))]
pub(crate) struct MerkleAuthPath<D, const DIGEST_ELEMS: usize> {
    /// Index of this leaf in the tree (0-based).
    pub leaf_index: usize,

    /// Sibling digests concatenated level by level, level 0 first.
    ///
    /// A level of arity `a` contributes `a - 1` entries.
    /// They are the group's children with this path's own child removed.
    pub siblings: Vec<[D; DIGEST_ELEMS]>,
}

/// Compact multi-opening proof: the minimal set of boundary sibling digests.
///
/// Leaf indices, tree depth, and per-level arity are not stored.
/// The verifier supplies the queried indices from its own transcript.
/// It rederives the tree geometry from the committed matrix dimensions.
///
/// Storing nothing but digests buys two guarantees:
/// - No leaf index in the proof means none a forger could substitute.
/// - No depth in the proof means none that could inflate the digest count.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(bound(serialize = "[D; DIGEST_ELEMS]: Serialize"))]
#[serde(bound(deserialize = "[D; DIGEST_ELEMS]: serde::de::DeserializeOwned"))]
pub struct PrunedMerklePaths<D, const DIGEST_ELEMS: usize> {
    /// Boundary sibling digests in frontier order:
    /// - level 0 first, then level 1, and so on,
    /// - within a level, groups by ascending parent index,
    /// - within a group, missing child positions ascending.
    pub sibling_hashes: Vec<[D; DIGEST_ELEMS]>,
}

/// Number of siblings a full path holds across the given number of levels.
///
/// Each level contributes `arity - 1` siblings (one child is the path's own).
#[inline]
fn total_siblings_for_levels(num_levels: usize, arity_schedule: &[usize]) -> usize {
    //   Example: arity_schedule = [2, 4, 2], num_levels = 2
    //     level 0: 2 - 1 = 1
    //     level 1: 4 - 1 = 3
    //     → total = 4
    arity_schedule[..num_levels].iter().map(|&a| a - 1).sum()
}

/// One node of the level-by-level frontier during a walk.
#[derive(Clone, Copy)]
struct FrontierNode {
    /// Node index at the current level.
    index: usize,

    /// Smallest queried-leaf slot beneath this node.
    ///
    /// Every leaf under a shared ancestor carries identical siblings above it.
    /// So this one slot's buffer stands in for the whole subtree.
    lead: usize,
}

/// Runs the callback once per boundary child of the frontier, in wire order.
///
/// The frontier starts at the sorted-unique leaf nodes.
/// It folds up one level per step, grouping nodes that share a parent.
/// A child covered by a frontier node is recomputable and skipped.
/// Every other child is a boundary and fires one callback.
///
/// The callback receives, in order:
/// - the current level, indexing the arity schedule,
/// - the slot whose buffer owns this group,
/// - the lead node's position within the group,
/// - the boundary child's position within the group.
///
/// Pruning and restoration share this walk, so they agree on wire order.
fn walk_frontier(
    sorted_unique: &[usize],
    arity_schedule: &[usize],
    mut visit: impl FnMut(usize, usize, usize, usize),
) {
    if sorted_unique.is_empty() {
        return;
    }

    // Seed the frontier with one node per queried leaf, its own slot as lead.
    let mut nodes: Vec<FrontierNode> = sorted_unique
        .iter()
        .enumerate()
        .map(|(slot, &index)| FrontierNode { index, lead: slot })
        .collect();
    let mut parents: Vec<FrontierNode> = Vec::with_capacity(nodes.len());

    for (level, &arity) in arity_schedule.iter().enumerate() {
        parents.clear();

        let mut i = 0;
        while i < nodes.len() {
            let group = nodes[i].index / arity;
            let group_start = group * arity;
            // The first node in a group is its lead (smallest index → smallest slot).
            let lead = nodes[i].lead;
            let lead_pos = nodes[i].index - group_start;

            // Walk child positions 0..arity in ascending order.
            // A position matched by a frontier node is recomputable, so skip it.
            // Every other position is a boundary child, reported here.
            let mut member = i;
            for k in 0..arity {
                if member < nodes.len() && nodes[member].index == group_start + k {
                    member += 1;
                } else {
                    visit(level, lead, lead_pos, k);
                }
            }

            parents.push(FrontierNode { index: group, lead });
            i = member;
        }

        core::mem::swap(&mut nodes, &mut parents);
    }
}

/// Offset of a child inside a lead path's flat sibling chunk for one level.
///
/// The chunk lists the group's children except the lead's own position, ascending.
/// So a child sits at its position, shifted down by one once past the skipped slot.
#[inline]
const fn sibling_offset(k: usize, lead_pos: usize) -> usize {
    if k < lead_pos { k } else { k - 1 }
}

/// Emits the minimal set of boundary sibling digests for a batch of paths.
///
/// - Sorts the queried leaves and drops duplicates.
/// - Folds the frontier up the tree, level by level.
/// - Reads each boundary digest from a group member's full path.
///
/// The output order matches what restoration consumes.
pub(crate) fn prune_paths<D, const DIGEST_ELEMS: usize>(
    arity_schedule: &[usize],
    paths: &[MerkleAuthPath<D, DIGEST_ELEMS>],
) -> PrunedMerklePaths<D, DIGEST_ELEMS>
where
    D: Clone,
{
    // Sort a lightweight index array by leaf index, so duplicates sit adjacent.
    // Sorting indices instead of the heavy path structs keeps the pass cache-friendly.
    let mut order: Vec<u32> = (0..paths.len() as u32).collect();
    order.sort_unstable_by_key(|&i| paths[i as usize].leaf_index);
    // Collapse repeated leaves to one entry each.
    order.dedup_by_key(|&mut i| paths[i as usize].leaf_index);

    let sorted_unique: Vec<usize> = order
        .iter()
        .map(|&i| paths[i as usize].leaf_index)
        .collect();

    // Prefix sums of per-level chunk sizes: the level-l chunk of a path starts here.
    let chunk_base: Vec<usize> = (0..=arity_schedule.len())
        .map(|l| total_siblings_for_levels(l, arity_schedule))
        .collect();

    // Each boundary child's digest is copied from the lead path's stored siblings.
    // The lead's chunk for a level lists its group's other children.
    // The boundary child at a given position lands at that chunk's matching offset.
    let mut sibling_hashes = Vec::new();
    walk_frontier(
        &sorted_unique,
        arity_schedule,
        |level, lead, lead_pos, k| {
            let src = &paths[order[lead] as usize].siblings;
            sibling_hashes.push(src[chunk_base[level] + sibling_offset(k, lead_pos)].clone());
        },
    );

    PrunedMerklePaths { sibling_hashes }
}

/// Rebuilds full authentication paths from the pruned frontier.
///
/// - Indices come from the verifier, never the proof, so the walk is bound to them.
/// - Grouping is replayed by index arithmetic alone, with no hashing.
/// - Each boundary digest is scattered into its group lead's full sibling buffer.
///
/// The amortized verifier reads only these boundary positions.
/// It recomputes every other child itself, so untouched positions stay at default.
///
/// # Trust model
///
/// - The full sibling count and the arity schedule come from verifier-known data.
/// - A digest count differing from the frontier is rejected, whether short or long.
///
/// # Returns
///
/// - Full authentication paths in sorted-unique order on success.
/// - An error describing the count mismatch otherwise.
pub(crate) fn restore_paths<D, const DIGEST_ELEMS: usize>(
    pruned: &PrunedMerklePaths<D, DIGEST_ELEMS>,
    sorted_unique: &[usize],
    arity_schedule: &[usize],
    full_sibling_count: usize,
) -> Result<Vec<MerkleAuthPath<D, DIGEST_ELEMS>>, PrunedProofError>
where
    D: Clone + Default,
{
    let supplied = &pruned.sibling_hashes;

    // Empty query set: the proof must carry no digests.
    if sorted_unique.is_empty() {
        return if supplied.is_empty() {
            Ok(vec![])
        } else {
            Err(PrunedProofError::SiblingCountMismatch {
                expected: 0,
                got: supplied.len(),
            })
        };
    }

    let chunk_base: Vec<usize> = (0..=arity_schedule.len())
        .map(|l| total_siblings_for_levels(l, arity_schedule))
        .collect();

    // One full-size, default-filled buffer per unique path.
    // The walk overwrites only the lead-path boundary slots it touches.
    // Building the zero digest by hand avoids requiring the digest type to be `Copy`.
    let default_digest: [D; DIGEST_ELEMS] = core::array::from_fn(|_| D::default());
    let mut restored: Vec<MerkleAuthPath<D, DIGEST_ELEMS>> = sorted_unique
        .iter()
        .map(|&leaf_index| MerkleAuthPath {
            leaf_index,
            siblings: vec![default_digest.clone(); full_sibling_count],
        })
        .collect();

    // Scatter supplied digests into boundary slots, following the prover's wire order.
    // Running out of supplied digests part-way means the proof is malformed.
    let mut cursor = 0;
    let mut overrun = false;
    walk_frontier(sorted_unique, arity_schedule, |level, lead, lead_pos, k| {
        if let Some(d) = supplied.get(cursor) {
            restored[lead].siblings[chunk_base[level] + sibling_offset(k, lead_pos)] = d.clone();
        } else {
            overrun = true;
        }
        cursor += 1;
    });

    // `cursor` is the exact count the frontier requires.
    // Reject a proof that supplied a different number of digests.
    if overrun || cursor != supplied.len() {
        return Err(PrunedProofError::SiblingCountMismatch {
            expected: cursor,
            got: supplied.len(),
        });
    }

    Ok(restored)
}

// Tests

#[cfg(test)]
mod tests {
    use alloc::vec;
    use alloc::vec::Vec;

    use proptest::prelude::*;

    use super::*;

    // Digest width used throughout the unit tests.
    const DE: usize = 2;

    /// Deterministic N-ary compression of `arity` child digests into a parent.
    ///
    /// Order-sensitive and position-weighted, like a real compressor.
    /// A wrong sibling order or a swapped child changes the output.
    fn combine(children: &[[u32; DE]]) -> [u32; DE] {
        let mut out = [0u32; DE];
        for (i, c) in children.iter().enumerate() {
            for e in 0..DE {
                out[e] = out[e]
                    .wrapping_mul(31)
                    .wrapping_add(c[e])
                    .wrapping_add(i as u32);
            }
        }
        out
    }

    /// Build a full N-ary tree over `leaves` and return the digests per level.
    ///
    /// Level 0 is the leaves.
    /// Each subsequent level compresses groups of one arity into their parent.
    /// The leaf count must equal the product of the arity schedule.
    fn build_tree(leaves: &[[u32; DE]], arity_schedule: &[usize]) -> Vec<Vec<[u32; DE]>> {
        let mut levels = vec![leaves.to_vec()];
        for &arity in arity_schedule {
            let prev = levels.last().unwrap();
            let cur: Vec<[u32; DE]> = prev.chunks(arity).map(combine).collect();
            levels.push(cur);
        }
        levels
    }

    /// A leaf's full authentication path: every sibling at every level, flat.
    ///
    /// Matches the layout the prover produces.
    /// Level 0 first; within a level the group's other children ascending, own position skipped.
    fn full_path(
        levels: &[Vec<[u32; DE]>],
        leaf_index: usize,
        arity_schedule: &[usize],
    ) -> MerkleAuthPath<u32, DE> {
        let mut siblings = Vec::new();
        let mut idx = leaf_index;
        for (lvl, &arity) in arity_schedule.iter().enumerate() {
            let group_start = (idx / arity) * arity;
            let pos = idx % arity;
            for k in 0..arity {
                if k != pos {
                    siblings.push(levels[lvl][group_start + k]);
                }
            }
            idx /= arity;
        }
        MerkleAuthPath {
            leaf_index,
            siblings,
        }
    }

    /// Recompute the root from restored paths, mirroring the amortized verifier.
    ///
    /// Member positions come from the running group digests.
    /// Boundary positions are read from the lead path's restored siblings.
    /// This is the reference the real verifier walk must match (no cap, no injection).
    fn root_from_restored(
        restored: &[MerkleAuthPath<u32, DE>],
        sorted_unique: &[usize],
        leaf_digests: &[[u32; DE]],
        arity_schedule: &[usize],
    ) -> [u32; DE] {
        let mut digests = leaf_digests.to_vec();
        let mut node_idx = sorted_unique.to_vec();
        let mut lead: Vec<usize> = (0..sorted_unique.len()).collect();
        let mut cursor = vec![0usize; sorted_unique.len()];

        for &arity in arity_schedule {
            let mut nd = Vec::new();
            let mut ni = Vec::new();
            let mut nl = Vec::new();
            let mut nc = Vec::new();
            let mut i = 0;
            while i < digests.len() {
                let parent = node_idx[i] / arity;
                let group_start = parent * arity;
                let mut j = i + 1;
                while j < digests.len() && node_idx[j] / arity == parent {
                    j += 1;
                }
                let lead_path = lead[i];
                let lead_pos = node_idx[i] - group_start;
                let lead_sib = &restored[lead_path].siblings;
                let cursor_start = cursor[i];

                let mut inputs = vec![[0u32; DE]; arity];
                let mut filled = vec![false; arity];
                for k in i..j {
                    let pos = node_idx[k] - group_start;
                    inputs[pos] = digests[k];
                    filled[pos] = true;
                }
                let mut sib = 0;
                for k in 0..arity {
                    if k == lead_pos {
                        continue;
                    }
                    if !filled[k] {
                        inputs[k] = lead_sib[cursor_start + sib];
                    }
                    sib += 1;
                }

                nd.push(combine(&inputs));
                ni.push(parent);
                nl.push(lead_path);
                nc.push(cursor_start + (arity - 1));
                i = j;
            }
            digests = nd;
            node_idx = ni;
            lead = nl;
            cursor = nc;
        }
        digests[0]
    }

    /// Distinct pseudo-random leaf digests for a tree of `n` leaves.
    fn leaf_digests(n: usize) -> Vec<[u32; DE]> {
        (0..n)
            .map(|i| [(i as u32) * 2 + 1, (i as u32) * 7 + 3])
            .collect()
    }

    /// End-to-end oracle: prune then restore must reconstruct the true root.
    ///
    /// The queries may be unsorted and contain duplicates.
    /// The prover feeds full paths in that order.
    /// The verifier restores from the sorted-unique set.
    fn assert_roundtrip(arity_schedule: &[usize], query_indices: &[usize]) {
        let n_leaves: usize = arity_schedule.iter().product();
        let leaves = leaf_digests(n_leaves);
        let tree = build_tree(&leaves, arity_schedule);
        let root = *tree.last().unwrap().first().unwrap();

        // Prover: full paths in caller order.
        let paths: Vec<_> = query_indices
            .iter()
            .map(|&i| full_path(&tree, i, arity_schedule))
            .collect();
        let pruned = prune_paths(arity_schedule, &paths);

        // Verifier: sorted-unique query set drives restoration.
        let mut sorted_unique = query_indices.to_vec();
        sorted_unique.sort_unstable();
        sorted_unique.dedup();
        let full = total_siblings_for_levels(arity_schedule.len(), arity_schedule);
        let restored = restore_paths(&pruned, &sorted_unique, arity_schedule, full).unwrap();

        // The reconstructed root must equal the true root.
        let leaf_ds: Vec<[u32; DE]> = sorted_unique.iter().map(|&i| leaves[i]).collect();
        let got = root_from_restored(&restored, &sorted_unique, &leaf_ds, arity_schedule);
        assert_eq!(
            got, root,
            "root mismatch for schedule {arity_schedule:?}, queries {query_indices:?}"
        );
    }

    /// Pruned digest count for a query set (sorted-unique, arity schedule).
    fn pruned_count(arity_schedule: &[usize], sorted_unique: &[usize]) -> usize {
        let mut c = 0;
        walk_frontier(sorted_unique, arity_schedule, |_, _, _, _| c += 1);
        c
    }

    // Frontier count (minimality) tests

    #[test]
    fn count_full_binary_tree_is_zero() {
        // Every leaf queried → every sibling is recomputable → nothing sent.
        assert_eq!(pruned_count(&[2, 2, 2], &(0..8).collect::<Vec<_>>()), 0);
    }

    #[test]
    fn count_scattered_binary_matches_doc_example() {
        // Leaves {1,2,5} in a height-3 binary tree: 0,3,4 at level 0, F at level 1.
        assert_eq!(pruned_count(&[2, 2, 2], &[1, 2, 5]), 4);
    }

    #[test]
    fn count_single_leaf_is_full_path() {
        // One query shares nothing → one sibling per level.
        assert_eq!(pruned_count(&[2, 2, 2], &[5]), 3);
        assert_eq!(pruned_count(&[4, 4], &[9]), 6);
    }

    #[test]
    fn count_full_4ary_tree_is_zero() {
        assert_eq!(pruned_count(&[4, 4], &(0..16).collect::<Vec<_>>()), 0);
    }

    #[test]
    fn count_never_exceeds_unpruned() {
        // Pruning must never send more than the naive per-path total.
        let schedule = [2, 2, 2, 2, 2];
        let full = total_siblings_for_levels(schedule.len(), &schedule);
        for subset_end in 1..=32usize {
            let indices: Vec<usize> = (0..subset_end).map(|i| (i * 13) % 32).collect();
            let mut su = indices.clone();
            su.sort_unstable();
            su.dedup();
            assert!(pruned_count(&schedule, &su) <= su.len() * full);
        }
    }

    // Round-trip tests

    #[test]
    fn roundtrip_binary_all_leaves() {
        assert_roundtrip(&[2, 2, 2], &(0..8).collect::<Vec<_>>());
    }

    #[test]
    fn roundtrip_binary_scattered() {
        assert_roundtrip(&[2, 2, 2, 2, 2], &[0, 3, 7, 12, 15, 20, 31]);
    }

    #[test]
    fn roundtrip_single() {
        assert_roundtrip(&[2, 2, 2, 2], &[6]);
    }

    #[test]
    fn roundtrip_unordered_with_duplicates() {
        // Unsorted input with a repeated leaf must still reconstruct the root.
        assert_roundtrip(&[2, 2, 2], &[5, 1, 3, 1]);
    }

    #[test]
    fn roundtrip_4ary_all_leaves() {
        assert_roundtrip(&[4, 4], &(0..16).collect::<Vec<_>>());
    }

    #[test]
    fn roundtrip_4ary_scattered() {
        assert_roundtrip(&[4, 4, 4], &[0, 5, 17, 33, 50, 63]);
    }

    #[test]
    fn roundtrip_mixed_arity() {
        // Mixed schedule [2,4,2] → 16 leaves.
        // Exercises a per-level arity that varies down the tree.
        assert_roundtrip(&[2, 4, 2], &[0, 1, 4, 7, 8, 11, 15]);
    }

    // Rejection / bounds tests

    #[test]
    fn restore_empty_query_set() {
        let pruned = PrunedMerklePaths::<u32, DE> {
            sibling_hashes: vec![],
        };
        assert!(
            restore_paths(&pruned, &[], &[2, 2, 2], 3)
                .unwrap()
                .is_empty()
        );
    }

    #[test]
    fn restore_rejects_nonempty_proof_for_empty_query_set() {
        let pruned = PrunedMerklePaths::<u32, DE> {
            sibling_hashes: vec![[0; DE]],
        };
        assert!(matches!(
            restore_paths(&pruned, &[], &[2, 2, 2], 3),
            Err(PrunedProofError::SiblingCountMismatch {
                expected: 0,
                got: 1
            })
        ));
    }

    #[test]
    fn restore_rejects_too_few_siblings() {
        // Frontier for {1,2,5} needs 4 digests.
        // Supply two fewer than that.
        let schedule = [2, 2, 2];
        let sorted_unique = [1, 2, 5];
        let need = pruned_count(&schedule, &sorted_unique);
        let pruned = PrunedMerklePaths::<u32, DE> {
            sibling_hashes: vec![[0; DE]; need - 2],
        };
        let full = total_siblings_for_levels(schedule.len(), &schedule);
        assert!(matches!(
            restore_paths(&pruned, &sorted_unique, &schedule, full),
            Err(PrunedProofError::SiblingCountMismatch { .. })
        ));
    }

    #[test]
    fn restore_rejects_too_many_siblings() {
        // Same frontier, but supply one extra digest.
        let schedule = [2, 2, 2];
        let sorted_unique = [1, 2, 5];
        let need = pruned_count(&schedule, &sorted_unique);
        let pruned = PrunedMerklePaths::<u32, DE> {
            sibling_hashes: vec![[0; DE]; need + 1],
        };
        let full = total_siblings_for_levels(schedule.len(), &schedule);
        assert!(matches!(
            restore_paths(&pruned, &sorted_unique, &schedule, full),
            Err(PrunedProofError::SiblingCountMismatch { expected, got })
                if expected == need && got == need + 1
        ));
    }

    // total_siblings_for_levels

    #[test]
    fn total_siblings_accumulates_per_level() {
        // binary(1) + 4ary(3) + 8ary(7) → cumulative 0,1,4,11.
        let s = [2, 4, 8];
        assert_eq!(total_siblings_for_levels(0, &s), 0);
        assert_eq!(total_siblings_for_levels(1, &s), 1);
        assert_eq!(total_siblings_for_levels(2, &s), 4);
        assert_eq!(total_siblings_for_levels(3, &s), 11);
    }

    // Proptests

    fn binary_case() -> impl Strategy<Value = (usize, Vec<usize>)> {
        (1..8usize).prop_flat_map(|h| {
            let n = 1usize << h;
            (Just(h), proptest::collection::vec(0..n, 1..=n.min(32)))
        })
    }

    fn nary_case() -> impl Strategy<Value = (Vec<usize>, Vec<usize>)> {
        // 1..5 levels, each arity 2 or 4, capped at 4096 leaves.
        proptest::collection::vec(prop_oneof![Just(2usize), Just(4usize)], 1..5).prop_flat_map(
            |schedule| {
                let n: usize = schedule.iter().product();
                let indices = proptest::collection::vec(0..n, 1..=n.min(32));
                (Just(schedule), indices)
            },
        )
    }

    proptest! {
        #[test]
        fn proptest_binary_roundtrip((h, indices) in binary_case()) {
            // prune → restore → reconstruct must recover the true root.
            let schedule = vec![2usize; h];
            assert_roundtrip(&schedule, &indices);
        }

        #[test]
        fn proptest_nary_roundtrip((schedule, indices) in nary_case()) {
            assert_roundtrip(&schedule, &indices);
        }

        #[test]
        fn proptest_count_leq_unpruned((h, indices) in binary_case()) {
            let schedule = vec![2usize; h];
            let full = total_siblings_for_levels(schedule.len(), &schedule);
            let mut su = indices;
            su.sort_unstable();
            su.dedup();
            prop_assert!(pruned_count(&schedule, &su) <= su.len() * full);
        }
    }
}
