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
use p3_symmetric::{CryptographicHasher, PseudoCompressionFunction};
use p3_util::log2_ceil_usize;
use serde::{Deserialize, Serialize};

use crate::MerkleTree;
use crate::MerkleTreeError::{CapMismatch, IndexOutOfBounds, WrongBatchSize, WrongHeight};
use crate::merkle_tree::{padded_len, select_arity_step};
use crate::pruning::{MerkleAuthPath, prune_paths, restore_paths};

mod batch;
mod error;
mod geometry;
mod pruned;

pub use error::{MerkleTreeError, PrunedProofError};
pub(crate) use geometry::{check_widths, validate_commit_reachable_heights};
pub use pruned::PrunedBatchOpening;

/// A Merkle Tree-based commitment scheme for multiple matrices of potentially differing heights.
///
/// `MerkleTreeMmcs` generalizes a classical Merkle Tree to support committing to a list of
/// matrices by arranging their rows into a unified binary tree. The tallest matrix defines
/// the maximum height, and smaller matrices are integrated at appropriate depths.
///
/// The commitment is a [`MerkleCap`](crate::MerkleCap), which is the `cap_height`-th layer from the root.
/// A `cap_height` of 0 means the commitment is just the root (a single hash).
/// A `cap_height` of h means the commitment contains `2^h` hashes and proofs are `h` elements shorter.
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

    /// The height of the Merkle cap. A cap_height of 0 uses only the root,
    /// while a cap_height of h uses 2^h hashes from h levels below the root.
    cap_height: usize,

    /// Phantom type to associate `P` and `PW` without storing values.
    _phantom: PhantomData<(P, PW)>,
}

impl<P, PW, H, C, const N: usize, const DIGEST_ELEMS: usize>
    MerkleTreeMmcs<P, PW, H, C, N, DIGEST_ELEMS>
{
    /// Create a new `MerkleTreeMmcs` with the given hash and compression functions.
    ///
    /// # Arguments
    /// * `hash` - The hash function used to hash individual matrix rows (leaf level).
    /// * `compress` - The compression function used to hash internal tree nodes.
    /// * `cap_height` - The height of the Merkle cap. A cap_height of 0 uses only the root,
    ///   while a cap_height of h uses 2^h hashes from h levels below the root.
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
    /// This helper mirrors the arity logic used in [`Mmcs::verify_batch`](p3_commit::Mmcs::verify_batch) but
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
        // The tallest height anchors the index bound and the leaf layer width below.
        let max_height =
            validate_commit_reachable_heights(dimensions.iter().map(|dims| dims.height))?;

        let mut heights_tallest_first = dimensions
            .iter()
            .enumerate()
            .sorted_by_key(|(_, dims)| Reverse(dims.height))
            .peekable();

        // Leaf layer width before the walk starts, padded to a full N-ary group.
        let mut curr_height_padded = padded_len(max_height, N);

        if index >= max_height {
            return Err(IndexOutOfBounds { max_height, index });
        }

        let leaf_height_npt = max_height.next_power_of_two();

        // Consume tallest matrices (mirrors verify_batch's initial hash step).
        heights_tallest_first
            .peeking_take_while(|(_, dims)| dims.height.next_power_of_two() == leaf_height_npt)
            .for_each(|_| {});

        let mut proof_pos: usize = 0;
        let mut steps = Vec::new();
        let mut positions = Vec::new();

        while proof_pos < num_opening_proofs {
            let step = select_arity_step::<N>(
                curr_height_padded,
                leaf_height_npt,
                heights_tallest_first.clone().map(|(_, dims)| dims.height),
            );

            let num_siblings = step - 1;
            if proof_pos + num_siblings > num_opening_proofs {
                return Err(WrongHeight {
                    expected_proof_len: proof_pos + num_siblings,
                    num_siblings: num_opening_proofs,
                });
            }
            proof_pos += num_siblings;

            let pos_in_group = index % step;
            steps.push(step);
            positions.push(pos_in_group);

            index /= step;
            let logical_next = curr_height_padded / step;
            curr_height_padded = padded_len(logical_next, N);

            // Mimic `verify_batch` but skip hashing values
            let logical_next_npt = logical_next.next_power_of_two();
            let next_height = heights_tallest_first
                .peek()
                .map(|(_, dims)| dims.height)
                .filter(|h| h.next_power_of_two() == logical_next_npt);
            if let Some(next_height) = next_height {
                heights_tallest_first
                    .peeking_take_while(|(_, dims)| dims.height == next_height)
                    .for_each(|_| {});
            }
        }

        Ok(ArityAndPositions {
            arity_schedule: steps,
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
        while curr_height_padded > 1 {
            // Arity at this layer:
            //   N → full N-ary compression
            //   2 → binary bridge inserted to land on the next injection point
            let step = select_arity_step::<N>(
                curr_height_padded,
                leaf_height_npt,
                heights_tallest_first.clone().map(|(_, dims)| dims.height),
            );
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
            }
        }

        // Phase 3: strip the top `cap_height` entries
        //
        // Those layers live inside the verifier's cap, not in the proof.
        let total_levels = schedule.len();
        let effective_cap_height = self.cap_height.min(total_levels);
        schedule.truncate(total_levels - effective_cap_height);

        Ok(schedule)
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
        let num_layers = prover_data.digest_layers.len();
        let effective_cap_height = self.cap_height.min(num_layers.saturating_sub(1));
        let proof_levels = num_layers
            .saturating_sub(1)
            .saturating_sub(effective_cap_height);
        // Used to compute the bit-shift from global leaf index to a shorter
        // matrix's row index. Smaller matrices have fewer rows, so their
        // row index is the global index right-shifted by the height difference.
        let log_max_height = log2_ceil_usize(max_height);

        // Phase 2: Build the shift schedule for the pruning algorithm.
        //
        // The arity schedule stores the branching factor at each level
        // (2 for binary, 4 for quad, etc.). The pruning LCA computation
        // needs log_2(arity) at each level so it can use bit-shifts instead
        // of integer division. trailing_zeros gives log_2 for powers of two.
        let shift_schedule: Vec<u32> = prover_data.arity_schedule[..proof_levels]
            .iter()
            .map(|&step| step.trailing_zeros())
            .collect();

        // Pre-compute the exact total number of sibling digests in a full
        // (unpruned) proof. At each level, the sibling count is arity - 1
        // (one child is ours, the rest are siblings).
        // This lets us allocate the flat sibling buffer exactly once per
        // query with no dynamic resizing.
        //
        //   Example: 3 binary levels → 1 + 1 + 1 = 3 siblings total
        //   Example: 2 quad levels   → 3 + 3     = 6 siblings total
        let expected_siblings: usize = prover_data.arity_schedule[..proof_levels]
            .iter()
            .map(|&step| step - 1)
            .sum();

        // Phase 3: For each queried leaf index, collect its opened rows
        // and full authentication path.
        //
        // Each query is independent: it reads one row from each committed
        // matrix (at the appropriate bit-shifted index) and walks down the
        // digest layers to collect all sibling digests.
        //
        // The result is unzipped into two parallel vectors:
        //   - one with the opened matrix rows (for the verifier)
        //   - one with the flat sibling arrays (for pruning)
        let (all_opened_values, auth_paths): (Vec<_>, Vec<_>) = indices
            .iter()
            .map(|&index| {
                assert!(
                    index < max_height,
                    "index {index} out of bounds for height {max_height}"
                );

                // Gather the row from each committed matrix at this leaf index.
                //
                // Shorter matrices cover fewer rows. The bit-shift
                // (log_max_height - log_height) maps the global leaf index
                // down to the row index in that shorter matrix.
                //
                //   Example: max_height = 32 (5 bits), matrix height = 8 (3 bits)
                //     bits_reduced = 5 - 3 = 2
                //     leaf index 20 → row 20 >> 2 = 5
                let openings: Vec<Vec<P::Value>> = prover_data
                    .leaves
                    .iter()
                    .map(|matrix| {
                        let log2_height = log2_ceil_usize(matrix.height());
                        let bits_reduced = log_max_height - log2_height;
                        let reduced_index = index >> bits_reduced;
                        matrix.row(reduced_index).unwrap().into_iter().collect()
                    })
                    .collect();

                // Walk up the digest layers collecting sibling digests.
                //
                // At each level, the queried leaf belongs to a group of
                // `step` children under one parent. We emit all children
                // in that group except the queried one — those are the
                // siblings the verifier needs to recompute the parent hash.
                //
                //   Example (binary, step = 2):
                //     group = [child_0, child_1], queried = child_0
                //     → emit child_1 (1 sibling)
                //
                //   Example (4-ary, step = 4):
                //     group = [c0, c1, c2, c3], queried = c2
                //     → emit c0, c1, c3 (3 siblings)
                //
                // After collecting siblings at this level, divide the index
                // by the step to move up to the parent level.
                let mut siblings = Vec::with_capacity(expected_siblings);
                let mut idx = index;
                for layer_idx in 0..proof_levels {
                    let step = prover_data.arity_schedule[layer_idx];
                    // Start of the N-child group containing this index.
                    let group_start = (idx / step) * step;
                    // Position of the queried child within the group.
                    let pos_in_group = idx % step;
                    // Emit every child in the group except the queried one.
                    for k in 0..step {
                        if k != pos_in_group {
                            siblings.push(prover_data.digest_layers[layer_idx][group_start + k]);
                        }
                    }
                    // Move to the parent index for the next level.
                    idx /= step;
                }

                (
                    openings,
                    MerkleAuthPath {
                        leaf_index: index,
                        siblings,
                    },
                )
            })
            .unzip();

        // Phase 4: Prune the authentication paths.
        //
        // This sorts paths by leaf index, deduplicates, computes the LCA
        // between each consecutive pair, and strips the shared upper siblings.
        // The result is a compact proof that can be restored on the verifier
        // side with zero information loss.
        let pruned = prune_paths(proof_levels, &shift_schedule, &auth_paths);

        PrunedBatchOpening {
            opened_values: all_opened_values,
            pruned_proof: pruned,
        }
    }

    /// Verifies a pruned batch opening against the commitment.
    ///
    /// Walks the tree once for all queries together: paths that share a parent
    /// at some level reuse a single compression instead of recomputing it once
    /// per query. Mirrors the per-path [`Mmcs::verify_batch`](p3_commit::Mmcs::verify_batch) algorithm but
    /// fans out into level-by-level groups.
    ///
    /// Takes the opening **by value** to avoid deep-cloning the three-level
    /// nested opened-values structure on return.
    ///
    /// # Returns
    ///
    /// On success, the opened matrix rows for each query (moved, not cloned).
    /// On failure, a verification error.
    pub fn verify_batch_pruned(
        &self,
        commit: &<Self as Mmcs<P::Value>>::Commitment,
        dimensions: &[Dimensions],
        pruned_opening: PrunedBatchOpening<P::Value, PW::Value, DIGEST_ELEMS>,
    ) -> Result<Vec<Vec<Vec<P::Value>>>, MerkleTreeError>
    where
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
        // Phase 1: Derive the verifier-known arity schedule and the expected
        // full-path sibling count.
        //
        // Each schedule entry contributes `step - 1` siblings: one of the
        // step children is the queried node itself, the rest are siblings.
        //
        //   schedule = [4, 2]   → 3 + 1 = 4 expected siblings
        //   schedule = [2, 2, 2] → 1 + 1 + 1 = 3 expected siblings
        //
        // The geometry gate also returns the tallest height,
        // which bounds the leaf indices and seeds the walk in phase 5.
        let max_height =
            validate_commit_reachable_heights(dimensions.iter().map(|dims| dims.height))?;
        let arity_schedule = self.proof_arity_schedule(dimensions)?;
        let full_sibling_count: usize = arity_schedule.iter().map(|step| step - 1).sum();

        // Invariant: leaf indices are strictly ascending and distinct in `[0, max_height)`.
        //   => a valid proof has at most `max_height` unique paths.
        // Reject an oversized count here, using verifier-known `max_height`.
        // Otherwise `restore_paths` expands `n_unique * full_sibling_count` digests first.
        if pruned_opening.pruned_proof.paths.len() > max_height {
            return Err(PrunedProofError::TooManyUniquePaths {
                got: pruned_opening.pruned_proof.paths.len(),
                max_height,
            }
            .into());
        }

        // Phase 2: Restore the full (unpruned) authentication paths from the
        // compact representation. We index siblings level by level during the
        // amortized walk, so we materialize the per-path buffers once and
        // borrow them throughout.
        let restored = restore_paths(&pruned_opening.pruned_proof, full_sibling_count)?;

        let original_order = &pruned_opening.pruned_proof.original_order;
        let sorted_paths = &pruned_opening.pruned_proof.paths;
        let n_originals = original_order.len();
        let n_unique = sorted_paths.len();

        // Restored length is one entry per original query, by construction.
        if restored.len() != n_originals || pruned_opening.opened_values.len() != n_originals {
            return Err(WrongBatchSize);
        }

        // Empty proof is valid only when no queries were submitted.
        if n_unique == 0 {
            return if n_originals == 0 {
                Ok(pruned_opening.opened_values)
            } else {
                Err(PrunedProofError::PathQueryCountMismatch {
                    num_paths: 0,
                    num_queries: n_originals,
                }
                .into())
            };
        }

        // Phase 3: Pick a representative original query for each unique path.
        // Verify that every original maps to a valid sorted slot and that
        // duplicates carry identical opened values — otherwise a malicious
        // prover could collapse two distinct openings into one verified slot.
        let mut reps: Vec<Option<usize>> = vec![None; n_unique];
        for (orig_idx, &sorted_idx) in original_order.iter().enumerate() {
            let sorted_idx = sorted_idx as usize;
            if sorted_idx >= n_unique {
                return Err(PrunedProofError::OriginalOrderOutOfRange {
                    slot: sorted_idx,
                    num_paths: n_unique,
                }
                .into());
            }
            match reps[sorted_idx] {
                None => reps[sorted_idx] = Some(orig_idx),
                Some(rep_idx) => {
                    if pruned_opening.opened_values[rep_idx]
                        != pruned_opening.opened_values[orig_idx]
                    {
                        return Err(PrunedProofError::InconsistentDuplicateOpenings {
                            slot: sorted_idx,
                        }
                        .into());
                    }
                }
            }
        }
        // Every unique sorted slot must be reached by at least one original.
        let reps: Vec<usize> = reps
            .into_iter()
            .enumerate()
            .map(|(slot, r)| r.ok_or(PrunedProofError::UnreferencedPath { slot }))
            .collect::<Result<Vec<usize>, PrunedProofError>>()?;

        // Phase 4: Shape and ordering invariants for the unique paths.
        // - Each opening's per-matrix count must match `dimensions`.
        // - Sorted leaf indices must be strictly ascending (the pruning
        //   contract). A malformed proof reaching this point would otherwise
        //   produce silently wrong groupings later.
        for &rep in &reps {
            if pruned_opening.opened_values[rep].len() != dimensions.len() {
                return Err(WrongBatchSize);
            }
            // Row boundaries within a flattened leaf hash are only pinned by the widths.
            check_widths(dimensions, &pruned_opening.opened_values[rep])?;
        }
        for (position, window) in sorted_paths.windows(2).enumerate() {
            if window[0].leaf_index >= window[1].leaf_index {
                return Err(PrunedProofError::NonAscendingLeaves {
                    position: position + 1,
                    index: window[1].leaf_index,
                }
                .into());
            }
        }

        // Phase 5: Set up the tallest-first matrix iterator. The amortized
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

        for path in sorted_paths {
            if path.leaf_index >= max_height {
                return Err(IndexOutOfBounds {
                    max_height,
                    index: path.leaf_index,
                });
            }
        }

        let leaf_height_npt = max_height.next_power_of_two();

        // Phase 6: Initial leaf hashes for every unique path.
        //
        // The leaf layer covers all matrices whose padded height matches
        // `leaf_height_npt`. Each unique path hashes its rows from those
        // matrices into the starting digest. Cost: one hash per unique path
        // (vs `n_originals` in the per-path verifier).
        let leaf_matrix_indices: Vec<usize> = heights_tallest_first
            .peeking_take_while(|(_, dims)| dims.height.next_power_of_two() == leaf_height_npt)
            .map(|(i, _)| i)
            .collect();

        let mut digests: Vec<[PW::Value; DIGEST_ELEMS]> = reps
            .iter()
            .map(|&rep| {
                self.hash.hash_iter_slices(
                    leaf_matrix_indices
                        .iter()
                        .map(|&mi| pruned_opening.opened_values[rep][mi].as_slice()),
                )
            })
            .collect();

        // Per-group state during the walk:
        // - `indices[g]`     : layer-local node index for group g
        // - `lead[g]`        : sorted path whose siblings buffer we read from
        //                      (any path in the group works; we keep the first)
        // - `sibling_cursor` : how many sibling slots of `lead[g]` we've used
        let mut indices: Vec<usize> = sorted_paths.iter().map(|p| p.leaf_index).collect();
        let mut lead: Vec<usize> = (0..n_unique).collect();
        let mut sibling_cursor: Vec<usize> = vec![0usize; n_unique];

        let default_digest = [PW::Value::default(); DIGEST_ELEMS];

        // Reusable scratch — refilled inside the inner loop, never read across iterations.
        let mut new_digests: Vec<[PW::Value; DIGEST_ELEMS]> = Vec::with_capacity(n_unique);
        let mut new_indices: Vec<usize> = Vec::with_capacity(n_unique);
        let mut new_lead: Vec<usize> = Vec::with_capacity(n_unique);
        let mut new_cursor: Vec<usize> = Vec::with_capacity(n_unique);

        // Phase 7: Walk up the tree. At each layer we collapse contiguous
        // groups of unique paths that share a parent, replacing each group
        // with a single combined digest.
        for &step in &arity_schedule {
            let num_siblings_per_path = step - 1;

            new_digests.clear();
            new_indices.clear();
            new_lead.clear();
            new_cursor.clear();

            let mut i = 0;
            while i < digests.len() {
                let parent_idx = indices[i] / step;
                let group_start = parent_idx * step;

                // Find the contiguous range of unique paths in this group.
                // The sorted-index invariant guarantees the group is contiguous.
                let mut j = i + 1;
                while j < digests.len() && indices[j] / step == parent_idx {
                    j += 1;
                }

                let lead_path = lead[i];
                let lead_pos = indices[i] - group_start;
                // `restored` is indexed by ORIGINAL query, not sorted slot — map
                // through `reps` to find any original mapping to this sorted path.
                // Both originals (when duplicates exist) have identical buffers,
                // so the first representative is canonical.
                let lead_siblings = &restored[reps[lead_path]].siblings;
                let cursor_start = sibling_cursor[i];

                // Build the N-ary input array:
                // - positions hit by group members → their current digest
                // - other in-range positions → siblings of the lead path
                // - out-of-range positions (k >= step) → default padding
                let mut inputs = [default_digest; N];
                let mut filled = [false; N];
                for k in i..j {
                    let pos = indices[k] - group_start;
                    inputs[pos] = digests[k];
                    filled[pos] = true;
                }

                // The lead path stored siblings for every non-lead position
                // in order: index 0, 1, …, step-1 with `lead_pos` skipped.
                // Walk the same order so the cursor advances by exactly
                // `num_siblings_per_path`, matching the per-path verifier.
                let mut sib_idx = 0;
                for k in 0..step {
                    if k == lead_pos {
                        continue;
                    }
                    if !filled[k] {
                        inputs[k] = lead_siblings[cursor_start + sib_idx];
                    }
                    sib_idx += 1;
                }

                new_digests.push(self.compress.compress(inputs));
                new_indices.push(parent_idx);
                new_lead.push(lead_path);
                new_cursor.push(cursor_start + num_siblings_per_path);

                i = j;
            }

            core::mem::swap(&mut digests, &mut new_digests);
            core::mem::swap(&mut indices, &mut new_indices);
            core::mem::swap(&mut lead, &mut new_lead);
            core::mem::swap(&mut sibling_cursor, &mut new_cursor);

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
                    let next_height_digest = self.hash.hash_iter_slices(
                        inject_matrix_indices
                            .iter()
                            .map(|&mi| pruned_opening.opened_values[rep_orig][mi].as_slice()),
                    );

                    let inject_inputs: [_; N] = core::array::from_fn(|k| {
                        if k == 0 {
                            digests[g]
                        } else if k == 1 {
                            next_height_digest
                        } else {
                            default_digest
                        }
                    });
                    digests[g] = self.compress.compress(inject_inputs);
                }
            }
        }

        // Phase 8: Each surviving digest must land inside the cap at its
        // layer-local index. A single mismatch rejects the whole batch.
        for g in 0..digests.len() {
            let cap_idx = indices[g];
            if cap_idx >= commit.num_roots() || commit[cap_idx] != digests[g] {
                return Err(CapMismatch);
            }
        }

        Ok(pruned_opening.opened_values)
    }
}
