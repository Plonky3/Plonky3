//! A MerkleTreeMmcs is a generalization of the standard MerkleTree commitment scheme which supports
//! committing to several matrices of different dimensions, with arbitrary arity N (binary by default).
//!
//! Say we wish to commit to 2 matrices M and N with dimensions (8, i) and (2, j) respectively.
//! Let H denote the hash function and C the compression function for our tree.
//! Then MerkleTreeMmcs produces a commitment to M and N using the following tree structure:
//!
//! ```rust,ignore
//! ///
//! ///                                      root = c00 = C(c10, c11)
//! ///                       /                                                \
//! ///         c10 = C(C(c20, c21), H(N[0]))                     c11 = C(C(c22, c23), H(N[1]))
//! ///           /                      \                          /                      \
//! ///      c20 = C(L, R)            c21 = C(L, R)            c22 = C(L, R)            c23 = C(L, R)
//! ///   L/             \R        L/             \R        L/             \R        L/             \R
//! /// H(M[0])         H(M[1])  H(M[2])         H(M[3])  H(M[4])         H(M[5])  H(M[6])         H(M[7])
//! ```
//! E.g. we start by making a standard MerkleTree commitment for each row of M and then add in the rows of N when we
//! get to the correct level. A proof for the values of say `M[5]` and `N[1]` consists of the siblings `H(M[4]), c23, c10`.
//!
//! Every shorter matrix height must equal `ceil(max_height / 2^k)` for some `k`.
//! This guarantees that each global leaf index maps to a row in every committed matrix.
//!

use alloc::vec;
use alloc::vec::Vec;
use core::cmp::Reverse;
use core::marker::PhantomData;

use geometry::ArityAndPositions;
use itertools::Itertools;
use p3_commit::Mmcs;
use p3_field::PackedValue;
use p3_matrix::{Dimensions, Matrix};
use p3_symmetric::{CryptographicHasher, MerkleCap, PseudoCompressionFunction};
use p3_util::log2_ceil_usize;
use serde::{Deserialize, Serialize};

use crate::MerkleTree;
use crate::MerkleTreeError::{
    CapMismatch, IndexOutOfBounds, WrongBatchSize, WrongCapSize, WrongHeight,
};
use crate::merkle_tree::{padded_len, select_arity_step};
use crate::pruning::{MerkleAuthPath, PrunedMerklePaths, restore_paths, walk_frontier};

mod batch;
mod error;
mod geometry;
mod pruned;
mod verify;

pub use error::{MerkleTreeError, PrunedProofError};
pub(crate) use geometry::{check_widths, validate_commit_reachable_heights};
pub use pruned::PrunedBatchOpening;

/// Materialize a row whose width is known even when its iterator has no size hint.
fn collect_opening_row<T>(row: impl IntoIterator<Item = T>, width: usize) -> Vec<T> {
    let row = row.into_iter();
    // Keep `collect` specializations, including taking ownership of an existing Vec buffer.
    if row.size_hint().0 == width {
        row.collect()
    } else {
        let mut values = Vec::with_capacity(width);
        values.extend(row);
        values
    }
}

/// A Merkle Tree-based commitment scheme for multiple matrices of potentially differing heights.
///
/// `MerkleTreeMmcs` generalizes a classical Merkle Tree to support committing to a list of
/// matrices by arranging their rows into a unified binary tree. The tallest matrix defines
/// the maximum height, and smaller matrices are integrated at appropriate depths.
///
/// The commitment is a [`MerkleCap`], at most `cap_height` layers below the root.
/// A `cap_height` of 0 means the commitment is just the root (a single hash).
/// The effective cap height is reduced when necessary so every matrix is injected at or below
/// the cap. A cap must not expose a layer that omits shorter matrices.
///
/// Type Parameters:
/// - `P`: Packed leaf value (e.g. a field element or vector of elements)
/// - `PW`: Packed digest element (used in the hash and compression output)
/// - `H`: Cryptographic hash function (leaf hash)
/// - `C`: Pseudo-compression function (internal node compression)
/// - `N`: Arity of the compression function
/// - `DIGEST_ELEMS`: Number of elements in a single digest
#[derive(Copy, Clone, Debug)]
pub struct MerkleTreeMmcs<P, PW, H, C, const N: usize, const DIGEST_ELEMS: usize> {
    /// The hash function used to hash individual matrix rows (leaf level).
    hash: H,

    /// The compression function used to hash internal tree nodes.
    compress: C,

    /// The maximum height of the Merkle cap, measured in layers below the root.
    /// It is reduced when necessary to include every matrix injection.
    cap_height: usize,

    /// Phantom type to associate `P` and `PW` without storing values.
    _phantom: PhantomData<(P, PW)>,
}

/// The outcome of one amortized pruned-frontier walk.
///
/// Produced by [`MerkleTreeMmcs::walk_pruned_frontier`] and consumed by both of its callers:
/// [`MerkleTreeMmcs::verify_batch_pruned`] checks `digests` against the commitment, while
/// [`MerkleTreeMmcs::restore_and_recompute_paths`] hands `restored` back to the caller.
struct PrunedFrontier<W, const DIGEST_ELEMS: usize> {
    /// Final digest of every group still standing at the cap layer.
    digests: Vec<[W; DIGEST_ELEMS]>,

    /// Layer-local (cap) index of each surviving group, parallel to `digests`.
    node_indices: Vec<usize>,

    /// Ascending distinct queried leaves: the slot order `restored` is indexed by.
    sorted_unique: Vec<usize>,

    /// One authentication path per sorted-unique slot. Complete only when the walk ran with
    /// recording enabled; empty during verification.
    restored: Vec<MerkleAuthPath<W, DIGEST_ELEMS>>,
}

impl<W, const DIGEST_ELEMS: usize> Default for PrunedFrontier<W, DIGEST_ELEMS> {
    /// The walk of an empty query set: nothing to authenticate, nothing to restore.
    fn default() -> Self {
        Self {
            digests: vec![],
            node_indices: vec![],
            sorted_unique: vec![],
            restored: vec![],
        }
    }
}

impl<P, PW, H, C, const N: usize, const DIGEST_ELEMS: usize>
    MerkleTreeMmcs<P, PW, H, C, N, DIGEST_ELEMS>
{
    /// Create a new `MerkleTreeMmcs` with the given hash and compression functions.
    ///
    /// # Arguments
    /// * `hash` - The hash function used to hash individual matrix rows (leaf level).
    /// * `compress` - The compression function used to hash internal tree nodes.
    /// * `cap_height` - The maximum number of layers below the root for the cap.
    ///   A height of 0 uses only the root. Mixed-height batches may use a smaller
    ///   height so the cap includes every matrix injection.
    pub const fn new(hash: H, compress: C, cap_height: usize) -> Self {
        const {
            assert!(N >= 2, "Arity N must be at least 2");
            assert!(N.is_power_of_two(), "Arity N must be a power of two");
        }
        Self {
            hash,
            compress,
            cap_height,
            _phantom: PhantomData,
        }
    }

    /// The configured maximum cap height. Mixed-height batches may use a smaller cap height
    /// to keep every matrix bound to the commitment.
    pub const fn cap_height(&self) -> usize {
        self.cap_height
    }

    // Verifier-side arity geometry, derived without hashing.

    /// Replay the arity schedule for a given Merkle path and recover, for each
    /// tree level, both:
    ///
    /// - the compression arity `step` used at that level (either `N` or `2`), and
    /// - the position of the queried child within its group, `pos_in_group`, in
    ///   the range `0..step`.
    ///
    /// This helper mirrors the arity logic used in [`Mmcs::verify_batch`] but
    /// omits all hashing.
    pub fn replay_arity_and_positions(
        &self,
        dimensions: &[Dimensions],
        mut index: usize,
        num_opening_proofs: usize,
    ) -> Result<ArityAndPositions, MerkleTreeError>
    where
        P: PackedValue,
        PW: PackedValue,
    {
        // Geometry gate: the claimed heights must form a tree the commitment can build.
        // The tallest height bounds the index below.
        let max_height =
            validate_commit_reachable_heights(dimensions.iter().map(|dims| dims.height))?;

        if index >= max_height {
            return Err(IndexOutOfBounds { max_height, index });
        }

        // The authoritative schedule, derived the same way `verify_batch` derives
        // it: it already stops at the root and excludes cap layers, so there is
        // no level left to fabricate once it is exhausted.
        let arity_schedule = self.proof_arity_schedule(dimensions)?;
        let expected_proof_len: usize = arity_schedule.iter().map(|step| step - 1).sum();
        if num_opening_proofs != expected_proof_len {
            return Err(WrongHeight {
                expected_proof_len,
                num_siblings: num_opening_proofs,
            });
        }

        let mut positions = Vec::with_capacity(arity_schedule.len());
        for &step in &arity_schedule {
            positions.push(index % step);
            index /= step;
        }

        Ok(ArityAndPositions {
            arity_schedule,
            query_positions: positions,
        })
    }

    /// Per-level compression arity for an unpruned proof, derived without hashing.
    ///
    /// One entry per proof level, each either `N` or `2`.
    ///
    /// # Returns
    ///
    /// - One entry per proof level, from leaves up to just above the cap.
    /// - Cap layers are excluded — the proof never traverses them.
    ///
    /// # Trust model
    ///
    /// - Built from verifier-known dimensions and cap height only.
    /// - Nothing is read from the proof.
    /// - Safe to feed into pruned-path restoration: a malicious proof cannot
    ///   inflate the verifier's allocation budget.
    pub fn proof_arity_schedule(
        &self,
        dimensions: &[Dimensions],
    ) -> Result<Vec<usize>, MerkleTreeError>
    where
        P: PackedValue,
        PW: PackedValue,
    {
        let (mut schedule, proof_levels) = self.arity_schedule_and_proof_levels(dimensions)?;
        schedule.truncate(proof_levels);
        Ok(schedule)
    }

    /// Reject a commitment that does not hold exactly the roots the tree's top layers produce.
    ///
    /// The cap sits at most the configured number of layers below the root,
    /// and never below the final matrix injection.
    ///
    /// Its length is the product of the arities above it, fixed by the dimensions alone.
    ///
    /// Openings never reach roots past that length, so nothing else would reject them.
    pub(crate) fn check_cap_size(
        &self,
        commit: &MerkleCap<P::Value, [PW::Value; DIGEST_ELEMS]>,
        dimensions: &[Dimensions],
    ) -> Result<(), MerkleTreeError>
    where
        P: PackedValue,
        PW: PackedValue,
    {
        let (schedule, proof_levels) = self.arity_schedule_and_proof_levels(dimensions)?;
        let expected = schedule[proof_levels..].iter().product();
        if commit.num_roots() != expected {
            return Err(WrongCapSize {
                expected,
                got: commit.num_roots(),
            });
        }
        Ok(())
    }

    /// The full arity schedule, leaves to root, and how many of its levels the proof walks.
    ///
    /// The remaining top levels are inside the cap.
    fn arity_schedule_and_proof_levels(
        &self,
        dimensions: &[Dimensions],
    ) -> Result<(Vec<usize>, usize), MerkleTreeError>
    where
        P: PackedValue,
        PW: PackedValue,
    {
        // Geometry gate: the claimed heights must form a tree the commitment can build.
        // The tallest height also seeds the walk below.
        let max_height =
            validate_commit_reachable_heights(dimensions.iter().map(|dims| dims.height))?;

        // Phase 1: order matrices tallest-first so the walk can peek ahead
        // and consume each one at its injection layer.
        let mut heights_tallest_first = dimensions
            .iter()
            .enumerate()
            .sorted_by_key(|(_, dims)| Reverse(dims.height))
            .peekable();

        // Phase 2: walk from leaves toward the root.
        //
        // Seed length is the leaf layer width rounded up to a full N-ary group —
        // the same padding the prover applies when building the tree.
        let mut curr_height_padded = padded_len(max_height, N);

        let leaf_height_npt = max_height.next_power_of_two();

        // Drop matrices already hashed into the leaf layer.
        //
        // Only shorter ones still need an injection slot.
        heights_tallest_first
            .peeking_take_while(|(_, dims)| dims.height.next_power_of_two() == leaf_height_npt)
            .for_each(|_| {});

        // One schedule entry per non-root layer.
        //
        //     layer_len:  L_0 → L_1 → L_2 → ... → 1   (root)
        //     step:        s_0   s_1   s_2  ...
        //     schedule  = [s_0,  s_1,  s_2,  ...]
        let mut schedule = Vec::new();
        let mut last_injection_level = 0;
        while curr_height_padded > 1 {
            // Arity at this layer:
            //   N → full N-ary compression
            //   2 → binary bridge inserted to land on the next injection point
            let step = select_arity_step::<N>(
                curr_height_padded,
                leaf_height_npt,
                heights_tallest_first.clone().map(|(_, dims)| dims.height),
            )?;
            schedule.push(step);

            // - Shrink by `step`,
            // - Re-pad so the next layer can form complete N-ary groups for the compression after it.
            let logical_next = curr_height_padded / step;
            curr_height_padded = padded_len(logical_next, N);

            // - Inject any matrix whose rounded height matches the next layer's pre-pad width,
            // - Consume it so it stops driving arity below.
            let logical_next_npt = logical_next.next_power_of_two();
            let next_height = heights_tallest_first
                .peek()
                .map(|(_, dims)| dims.height)
                .filter(|h| h.next_power_of_two() == logical_next_npt);
            if let Some(next_height) = next_height {
                heights_tallest_first
                    .peeking_take_while(|(_, dims)| dims.height == next_height)
                    .for_each(|_| {});
                last_injection_level = schedule.len();
            }
        }

        // The proof must reach every injection, including one at the cap layer itself.
        // Cutting below the last injection would leave that matrix outside the commitment.
        let total_levels = schedule.len();
        let proof_levels =
            (total_levels - self.cap_height.min(total_levels)).max(last_injection_level);

        Ok((schedule, proof_levels))
    }

    /// Derive the same cap boundary for committing and both opening paths as for verification.
    fn cap_height_and_proof_levels<M>(
        &self,
        prover_data: &MerkleTree<P::Value, PW::Value, M, N, DIGEST_ELEMS>,
    ) -> (usize, usize)
    where
        P: PackedValue,
        PW: PackedValue,
        M: Matrix<P::Value>,
    {
        let dimensions: Vec<_> = prover_data.leaves.iter().map(Matrix::dimensions).collect();
        let (schedule, proof_levels) = self
            .arity_schedule_and_proof_levels(&dimensions)
            .expect("committed matrices have valid heights");
        (schedule.len() - proof_levels, proof_levels)
    }

    // Amortized multi-query openings with pruned authentication paths.

    /// Opens multiple leaf indices at once and returns a pruned proof.
    ///
    /// Equivalent to opening each index individually and then pruning the
    /// resulting authentication paths, but avoids redundant allocations.
    ///
    /// The returned value contains:
    /// - The opened matrix rows for each query (in input order).
    /// - A compact pruned proof with shared siblings deduplicated.
    ///
    /// Use the pruned-verification method on the verifier side.
    pub fn open_batch_pruned<M: Matrix<P::Value>>(
        &self,
        indices: &[usize],
        prover_data: &MerkleTree<P::Value, PW::Value, M, N, DIGEST_ELEMS>,
    ) -> PrunedBatchOpening<P::Value, PW::Value, DIGEST_ELEMS>
    where
        P: PackedValue,
        PW: PackedValue,
        H: CryptographicHasher<P::Value, [PW::Value; DIGEST_ELEMS]>
            + CryptographicHasher<P, [PW; DIGEST_ELEMS]>,
        C: PseudoCompressionFunction<[PW::Value; DIGEST_ELEMS], N>
            + PseudoCompressionFunction<[PW; DIGEST_ELEMS], N>,
        PW::Value: Eq + Clone,
        [PW::Value; DIGEST_ELEMS]: Serialize + for<'de> Deserialize<'de>,
    {
        // Phase 1: Derive tree geometry from the committed data.
        //
        // The tree has multiple digest layers. The "cap" is a configurable
        // number of layers cut from the top — the commitment is those top
        // digests rather than just the single root.
        // The proof only needs to cover levels below the cap.
        //
        //   Example: 5 digest layers, cap_height = 1
        //     layers:          [0] [1] [2] [3] [4]
        //     cap covers:                      [4]   (1 layer from the top)
        //     proof covers:    [0] [1] [2] [3]       (4 proof levels)
        let max_height = prover_data
            .leaves
            .iter()
            .map(|m| m.height())
            .max()
            .expect("No committed matrices?");
        let (_, proof_levels) = self.cap_height_and_proof_levels(prover_data);
        // Used to compute the bit-shift from global leaf index to a shorter
        // matrix's row index. Smaller matrices have fewer rows, so their
        // row index is the global index right-shifted by the height difference.
        let log_max_height = log2_ceil_usize(max_height);

        let all_opened_values = indices
            .iter()
            .map(|&index| {
                assert!(
                    index < max_height,
                    "index {index} out of bounds for height {max_height}"
                );
                prover_data
                    .leaves
                    .iter()
                    .map(|matrix| {
                        let bits_reduced = log_max_height - log2_ceil_usize(matrix.height());
                        collect_opening_row(
                            matrix.row(index >> bits_reduced).unwrap(),
                            matrix.width(),
                        )
                    })
                    .collect()
            })
            .collect();

        // Emit boundary children directly, in the same level/group/child order as pruning
        // full paths. No per-query authentication paths are needed on this path.
        let mut sorted_unique = indices.to_vec();
        sorted_unique.sort_unstable();
        sorted_unique.dedup();
        let mut sibling_hashes = Vec::new();
        walk_frontier(
            &sorted_unique,
            &prover_data.arity_schedule[..proof_levels],
            |level, _, _, _, index| sibling_hashes.push(prover_data.digest_layers[level][index]),
        );
        let pruned = PrunedMerklePaths { sibling_hashes };

        PrunedBatchOpening {
            opened_values: all_opened_values,
            pruned_proof: pruned,
        }
    }

    /// Walks the pruned frontier once for all queries together, the shared engine behind
    /// [`verify_batch_pruned`](Self::verify_batch_pruned) and
    /// [`restore_and_recompute_paths`](Self::restore_and_recompute_paths).
    ///
    /// Paths that share a parent at some level reuse a single compression instead of
    /// recomputing it once per query. Mirrors the per-path
    /// [`Mmcs::verify_batch`] algorithm but fans out into
    /// level-by-level groups.
    ///
    /// Every shape and consistency check the pruned-proof soundness argument relies on lives
    /// here — the geometry gate, the sorted-unique derivation from verifier-side indices,
    /// duplicate-opening agreement, width checks, and the group-opening agreement at each
    /// injection point — so both entry points inherit exactly the same ones. What they do
    /// with the result differs: one checks the surviving digests against the commitment, the
    /// other hands back the recorded paths.
    ///
    /// With `RECORD`, each level additionally scatters its "other children" into *every*
    /// query's own path rather than only the group lead's. `RECORD = false` compiles that
    /// block out entirely, leaving the verification walk exactly as it would be written by
    /// hand.
    ///
    /// # Index binding
    ///
    /// `indices` must come from verifier-side data (e.g. the transcript).
    /// The walk is pinned entirely to those indices, so a proof cannot silently
    /// substitute a different leaf.
    fn walk_pruned_frontier<R, const RECORD: bool>(
        &self,
        dimensions: &[Dimensions],
        indices: &[usize],
        opened_values: &[Vec<R>],
        proof: &PrunedMerklePaths<PW::Value, DIGEST_ELEMS>,
    ) -> Result<PrunedFrontier<PW::Value, DIGEST_ELEMS>, MerkleTreeError>
    where
        R: AsRef<[P::Value]> + PartialEq,
        P: PackedValue,
        PW: PackedValue,
        H: CryptographicHasher<P::Value, [PW::Value; DIGEST_ELEMS]>
            + CryptographicHasher<P, [PW; DIGEST_ELEMS]>
            + Sync,
        C: PseudoCompressionFunction<[PW::Value; DIGEST_ELEMS], N>
            + PseudoCompressionFunction<[PW; DIGEST_ELEMS], N>
            + Sync,
        PW::Value: Eq + Clone,
        [PW::Value; DIGEST_ELEMS]: Serialize + for<'de> Deserialize<'de>,
    {
        // Phase 1: Derive the verifier-known arity schedule and full-path sibling count.
        //
        // A level of arity `a` contributes `a - 1` siblings: every child but the queried one.
        //
        //   schedule = [4, 2]   → 3 + 1 = 4 expected siblings
        //   schedule = [2, 2, 2] → 1 + 1 + 1 = 3 expected siblings
        //
        // The geometry gate also returns the tallest height.
        // That height bounds the leaf indices and seeds the amortized walk below.
        let max_height =
            validate_commit_reachable_heights(dimensions.iter().map(|dims| dims.height))?;
        let arity_schedule = self.proof_arity_schedule(dimensions)?;
        let full_sibling_count: usize = arity_schedule.iter().map(|step| step - 1).sum();

        // Phase 2: Length gate. One opened row set per query.
        let n_originals = indices.len();
        if opened_values.len() != n_originals {
            return Err(WrongBatchSize {
                expected: n_originals,
                got: opened_values.len(),
            });
        }

        // Empty query set: there is nothing to authenticate.
        // A proof carrying any sibling for it is malformed.
        if n_originals == 0 {
            return if proof.sibling_hashes.is_empty() {
                Ok(PrunedFrontier::default())
            } else {
                Err(PrunedProofError::SiblingCountMismatch {
                    expected: 0,
                    got: proof.sibling_hashes.len(),
                }
                .into())
            };
        }

        // Phase 3: Derive the sorted-unique query set from the verifier's own indices.
        // The proof stores no indices, so there is none a forger could substitute.
        // The frontier is pinned entirely to what the verifier asked for.
        //
        //   sorted_unique : ascending distinct leaves (the frontier's leaf layer)
        let mut sorted_unique: Vec<usize> = indices.to_vec();
        sorted_unique.sort_unstable();
        sorted_unique.dedup();
        let n_unique = sorted_unique.len();

        // Every queried leaf must live in the tree.
        let &highest = sorted_unique.last().expect("nonempty query set");
        if highest >= max_height {
            return Err(IndexOutOfBounds {
                max_height,
                index: highest,
            });
        }

        // Phase 4: Pick one representative query per unique slot.
        //   - The representative supplies the opened rows for leaf hashing and injection.
        //   - Duplicate queries on one leaf must agree on their opened rows.
        //   - Otherwise deduplication would silently erase the disagreement.
        let mut reps: Vec<Option<usize>> = vec![None; n_unique];
        for (orig_idx, &leaf) in indices.iter().enumerate() {
            let slot = sorted_unique
                .binary_search(&leaf)
                .expect("leaf came from the query set");
            match reps[slot] {
                None => reps[slot] = Some(orig_idx),
                Some(rep_idx) => {
                    if opened_values[rep_idx] != opened_values[orig_idx] {
                        return Err(PrunedProofError::InconsistentDuplicateOpenings { slot }.into());
                    }
                }
            }
        }
        // Each slot is reached by the query it came from, so none is missing.
        let reps: Vec<usize> = reps
            .into_iter()
            .map(|r| r.expect("every unique slot is one of the queries"))
            .collect();

        // Reject malformed boundary counts before any hashing. Only restoration callers
        // allocate full paths; verification counts the frontier and consumes the wire stream.
        let mut restored = if RECORD {
            restore_paths(proof, &sorted_unique, &arity_schedule, full_sibling_count)?
        } else {
            let mut expected = 0;
            walk_frontier(&sorted_unique, &arity_schedule, |_, _, _, _, _| {
                expected += 1;
            });
            if expected != proof.sibling_hashes.len() {
                return Err(PrunedProofError::SiblingCountMismatch {
                    expected,
                    got: proof.sibling_hashes.len(),
                }
                .into());
            }
            Vec::new()
        };
        let mut boundary_cursor = 0;

        // Every representative opening must expose one row per committed matrix, each at its
        // matrix's width. Ascending, distinct leaf order already holds by construction.
        for &rep in &reps {
            check_widths(dimensions, &opened_values[rep])?;
        }

        // Phase 6: Set up the tallest-first matrix iterator. The amortized
        // walk consumes matrices at the same layer boundaries as the per-path
        // verifier does — this mirrors `verify_batch` line-for-line so the
        // algebraic checks stay identical.
        let mut heights_tallest_first = dimensions
            .iter()
            .enumerate()
            .sorted_by_key(|(_, dims)| Reverse(dims.height))
            .peekable();

        // Leaf layer width before the walk starts, padded to a full N-ary group.
        let mut curr_height_padded = padded_len(max_height, N);

        let leaf_height_npt = max_height.next_power_of_two();

        // Phase 7: Initial leaf hashes for every unique path.
        //
        // The leaf layer covers all matrices whose padded height matches
        // `leaf_height_npt`. Each unique path hashes its rows from those
        // matrices into the starting digest. Cost: one hash per unique path
        // (vs `n_originals` in the per-path verifier).
        let leaf_matrix_indices: Vec<usize> = heights_tallest_first
            .peeking_take_while(|(_, dims)| dims.height.next_power_of_two() == leaf_height_npt)
            .map(|(i, _)| i)
            .collect();

        let mut digests = self.hash_opened_rows(&leaf_matrix_indices, &reps, opened_values);

        // Per-group state during the walk:
        // - `node_indices[g]` : layer-local node index for group g
        // - `lead[g]`         : sorted path whose siblings buffer we read from
        //                       (any path in the group works; we keep the first)
        let mut node_indices: Vec<usize> = sorted_unique.clone();
        let mut lead: Vec<usize> = (0..n_unique).collect();

        // Sorted-path-slot range `[group_lo[g], group_hi[g])` merged into
        // group `g` so far. Sorted paths only ever merge with contiguous
        // neighbors, so a group's original members always form a contiguous
        // slot range — no need to track a full member list.
        let mut group_lo: Vec<usize> = (0..n_unique).collect();
        let mut group_hi: Vec<usize> = (1..=n_unique).collect();

        let default_digest = [PW::Value::default(); DIGEST_ELEMS];

        // Reusable scratch — refilled inside the inner loop, never read across iterations.
        let mut new_digests: Vec<[PW::Value; DIGEST_ELEMS]> = Vec::with_capacity(n_unique);
        let mut compression_inputs = Vec::with_capacity(n_unique);
        let mut new_indices: Vec<usize> = Vec::with_capacity(n_unique);
        let mut new_lead: Vec<usize> = Vec::with_capacity(n_unique);
        let mut new_group_lo: Vec<usize> = Vec::with_capacity(n_unique);
        let mut new_group_hi: Vec<usize> = Vec::with_capacity(n_unique);

        // Offset into recorded full paths; verification consumes the boundary stream instead.
        let mut chunk_start = 0;

        // Phase 8: Walk up the tree. At each layer we collapse contiguous
        // groups of unique paths that share a parent, replacing each group
        // with a single combined digest.
        for &step in &arity_schedule {
            new_digests.clear();
            compression_inputs.clear();
            new_indices.clear();
            new_lead.clear();
            new_group_lo.clear();
            new_group_hi.clear();

            let mut i = 0;
            while i < digests.len() {
                let parent_idx = node_indices[i] / step;
                let group_start = parent_idx * step;

                // Find the contiguous range of unique paths in this group.
                // The sorted-index invariant guarantees the group is contiguous.
                let mut j = i + 1;
                while j < digests.len() && node_indices[j] / step == parent_idx {
                    j += 1;
                }

                let lead_path = lead[i];

                // Frontier members already supply their digests. Missing positions consume
                // boundary children sequentially, exactly in the prover's wire order.
                let mut inputs = [default_digest; N];
                let mut member = i;
                for (pos, input) in inputs.iter_mut().enumerate().take(step) {
                    if member < j && node_indices[member] == group_start + pos {
                        *input = digests[member];
                        member += 1;
                    } else {
                        *input = proof.sibling_hashes[boundary_cursor];
                        boundary_cursor += 1;
                    }
                }

                if RECORD {
                    // Every pre-merge entry in this group gives its own "other children" view —
                    // at this level's own position, not the group lead's — to every original
                    // query slot it represents (a range of one, unless it merged earlier).
                    for k in i..j {
                        let own_pos = node_indices[k] - group_start;
                        for (pos, &child) in inputs.iter().enumerate().take(step) {
                            if pos == own_pos {
                                continue;
                            }
                            let offset = if pos < own_pos { pos } else { pos - 1 };
                            for item in restored.iter_mut().take(group_hi[k]).skip(group_lo[k]) {
                                item.siblings[chunk_start + offset] = child;
                            }
                        }
                    }
                }

                compression_inputs.push(inputs);
                new_indices.push(parent_idx);
                new_lead.push(lead_path);
                new_group_lo.push(group_lo[i]);
                new_group_hi.push(group_hi[j - 1]);

                i = j;
            }

            new_digests.resize(compression_inputs.len(), default_digest);
            self.compress_opened_groups(&compression_inputs, &mut new_digests);
            chunk_start += step - 1;
            core::mem::swap(&mut digests, &mut new_digests);
            core::mem::swap(&mut node_indices, &mut new_indices);
            core::mem::swap(&mut lead, &mut new_lead);
            core::mem::swap(&mut group_lo, &mut new_group_lo);
            core::mem::swap(&mut group_hi, &mut new_group_hi);

            // Layer geometry update mirrors `verify_batch`.
            let logical_next = curr_height_padded / step;
            curr_height_padded = padded_len(logical_next, N);

            // Inject any shorter matrices whose padded height matches the
            // new layer. Paths in the same group share the row index, so we
            // hash the matrix rows once per group.
            let logical_next_npt = logical_next.next_power_of_two();
            let next_height = heights_tallest_first
                .peek()
                .map(|(_, dims)| dims.height)
                .filter(|h| h.next_power_of_two() == logical_next_npt);
            if let Some(next_height) = next_height {
                let inject_matrix_indices: Vec<usize> = heights_tallest_first
                    .peeking_take_while(|(_, dims)| dims.height == next_height)
                    .map(|(i, _)| i)
                    .collect();

                for g in 0..digests.len() {
                    let rep_orig = reps[lead[g]];

                    // Every sorted-path slot merged into this group shares the
                    // same reduced row index into the injected matrices, so an
                    // honest prover reports the same row for all of them. Only
                    // the lead's row is hashed below, so every other member
                    // must be checked explicitly against it here.
                    for (slot, &member_orig) in
                        reps.iter().enumerate().take(group_hi[g]).skip(group_lo[g])
                    {
                        if slot == lead[g] {
                            continue;
                        }
                        for &mi in &inject_matrix_indices {
                            if opened_values[member_orig][mi] != opened_values[rep_orig][mi] {
                                return Err(PrunedProofError::InconsistentGroupOpening {
                                    slot,
                                    matrix: mi,
                                }
                                .into());
                            }
                        }
                    }
                }
                let inject_reps: Vec<_> = lead.iter().map(|&slot| reps[slot]).collect();
                let injected =
                    self.hash_opened_rows(&inject_matrix_indices, &inject_reps, opened_values);
                compression_inputs.clear();
                compression_inputs.extend(digests.iter().zip(injected).map(|(&digest, row)| {
                    let mut inputs = [default_digest; N];
                    inputs[0] = digest;
                    inputs[1] = row;
                    inputs
                }));
                self.compress_opened_groups(&compression_inputs, &mut digests);
            }
        }

        // Unreachable for validated dimensions: the schedule reaches every injection.
        // Fail closed if this walk ever disagrees with the schedule.
        if heights_tallest_first.peek().is_some() {
            return Err(CapMismatch);
        }

        Ok(PrunedFrontier {
            digests,
            node_indices,
            sorted_unique,
            restored,
        })
    }

    /// Verifies a pruned batch opening against the commitment.
    ///
    /// Walks the tree once for all queries together via
    /// `walk_pruned_frontier`: paths that share a parent at
    /// some level reuse a single compression instead of recomputing it once per query.
    ///
    /// # Index binding
    ///
    /// `indices` must come from verifier-side data (e.g. the transcript).
    /// Each opening is checked to authenticate exactly its supplied index,
    /// so a proof cannot silently substitute a different leaf.
    ///
    /// # Arguments
    ///
    /// - `commit`         — the Merkle cap the openings must land in.
    /// - `dimensions`     — verifier-known dimensions of the committed matrices.
    /// - `indices`        — leaf index for each query, in query order.
    /// - `opened_values`  — `opened_values[q][m]` is the claimed row of matrix `m` at `indices[q]`.
    /// - `proof`          — the pruned authentication paths.
    pub fn verify_batch_pruned<R>(
        &self,
        commit: &<Self as Mmcs<P::Value>>::Commitment,
        dimensions: &[Dimensions],
        indices: &[usize],
        opened_values: &[Vec<R>],
        proof: &PrunedMerklePaths<PW::Value, DIGEST_ELEMS>,
    ) -> Result<(), MerkleTreeError>
    where
        R: AsRef<[P::Value]> + PartialEq,
        P: PackedValue,
        PW: PackedValue,
        H: CryptographicHasher<P::Value, [PW::Value; DIGEST_ELEMS]>
            + CryptographicHasher<P, [PW; DIGEST_ELEMS]>
            + Sync,
        C: PseudoCompressionFunction<[PW::Value; DIGEST_ELEMS], N>
            + PseudoCompressionFunction<[PW; DIGEST_ELEMS], N>
            + Sync,
        PW::Value: Eq + Clone,
        [PW::Value; DIGEST_ELEMS]: Serialize + for<'de> Deserialize<'de>,
    {
        self.check_cap_size(commit, dimensions)?;

        // `RECORD = false`: nothing here reads the per-query paths, so the recording branch
        // compiles out and this walk is exactly the amortized verification walk.
        let frontier =
            self.walk_pruned_frontier::<R, false>(dimensions, indices, opened_values, proof)?;

        // Each surviving digest must land inside the cap at its layer-local
        // index. A single mismatch rejects the whole batch.
        for (&cap_idx, digest) in frontier.node_indices.iter().zip(&frontier.digests) {
            if cap_idx >= commit.num_roots() || commit[cap_idx] != *digest {
                return Err(CapMismatch);
            }
        }

        Ok(())
    }

    /// Reconstructs one full, self-sufficient Merkle authentication path per query from a
    /// pruned multiproof.
    ///
    /// [`verify_batch_pruned`](Self::verify_batch_pruned) only checks the final digest of
    /// each amortized group against the commitment; it never needs a usable per-query path,
    /// so it never records one. Some callers do need that — e.g. an in-circuit verifier built
    /// around a single-path-per-query gadget, with no notion of a shared/amortized proof.
    /// This runs `walk_pruned_frontier` — literally the same
    /// walk verification uses, so it costs no more hashing — with recording enabled, which
    /// additionally scatters each level's "other children" into *every* query's own path,
    /// not just the group lead's, including query pairs that only merge partway up the tree,
    /// whose later-level siblings are otherwise nowhere in the pruned proof at all.
    ///
    /// Arguments match [`verify_batch_pruned`](Self::verify_batch_pruned) minus `commit`: this
    /// function does not check anything against a commitment. A caller that also wants that
    /// check should call `verify_batch_pruned` (or verify each returned path against the
    /// commitment itself); the two do the same walk and either can be skipped once the other
    /// has run.
    ///
    /// # Returns
    ///
    /// One [`MerkleAuthPath`] per entry of `indices`, in the same order (duplicates included) —
    /// the same shape a pre-pruning, per-query proof would have supplied directly.
    pub fn restore_and_recompute_paths<R>(
        &self,
        dimensions: &[Dimensions],
        indices: &[usize],
        opened_values: &[Vec<R>],
        proof: &PrunedMerklePaths<PW::Value, DIGEST_ELEMS>,
    ) -> Result<Vec<MerkleAuthPath<PW::Value, DIGEST_ELEMS>>, MerkleTreeError>
    where
        R: AsRef<[P::Value]> + PartialEq,
        P: PackedValue,
        PW: PackedValue,
        H: CryptographicHasher<P::Value, [PW::Value; DIGEST_ELEMS]>
            + CryptographicHasher<P, [PW; DIGEST_ELEMS]>
            + Sync,
        C: PseudoCompressionFunction<[PW::Value; DIGEST_ELEMS], N>
            + PseudoCompressionFunction<[PW; DIGEST_ELEMS], N>
            + Sync,
        PW::Value: Eq + Clone,
        [PW::Value; DIGEST_ELEMS]: Serialize + for<'de> Deserialize<'de>,
    {
        let frontier =
            self.walk_pruned_frontier::<R, true>(dimensions, indices, opened_values, proof)?;

        // Expand back to the caller's original (possibly duplicated) query order.
        indices
            .iter()
            .map(|&leaf| {
                let slot = frontier
                    .sorted_unique
                    .binary_search(&leaf)
                    .expect("leaf came from the query set");
                Ok(frontier.restored[slot].clone())
            })
            .collect()
    }
}
