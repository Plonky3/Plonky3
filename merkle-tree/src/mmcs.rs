//! A MerkleTreeMmcs is a generalization of the standard MerkleTree commitment scheme which supports
//! committing to several matrices of different dimensions.
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

use alloc::vec::Vec;
use core::cmp::Reverse;
use core::marker::PhantomData;

use itertools::Itertools;
use p3_commit::{BatchOpening, BatchOpeningRef, Mmcs};
use p3_field::PackedValue;
use p3_matrix::{Dimensions, Matrix};
use p3_symmetric::{CryptographicHasher, PseudoCompressionFunction};
use p3_util::log2_ceil_usize;
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::MerkleTreeError::{
    CapMismatch, EmptyBatch, IncompatibleHeights, IndexOutOfBounds, MalformedPrunedProof,
    WrongBatchSize, WrongHeight,
};
use crate::merkle_tree::{padded_len, select_arity_step};
use crate::pruning::{MerkleAuthPath, prune_paths, restore_paths};
use crate::{MerkleCap, MerkleTree};

/// A Merkle Tree-based commitment scheme for multiple matrices of potentially differing heights.
///
/// `MerkleTreeMmcs` generalizes a classical Merkle Tree to support committing to a list of
/// matrices by arranging their rows into a unified binary tree. The tallest matrix defines
/// the maximum height, and smaller matrices are integrated at appropriate depths.
///
/// The commitment is a [`MerkleCap`], which is the `cap_height`-th layer from the root.
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

/// Errors that may arise during Merkle tree commitment, opening, or verification.
#[derive(Debug, Error)]
pub enum MerkleTreeError {
    /// The number of openings provided does not match the expected number.
    #[error("wrong batch size: number of openings does not match expected")]
    WrongBatchSize,

    /// A matrix has a different width than expected.
    #[error("wrong width: matrix has a different width than expected")]
    WrongWidth,

    /// The number of proof nodes does not match the expected tree height.
    #[error("wrong height: expected {expected_proof_len} siblings, got {num_siblings}")]
    WrongHeight {
        /// Expected number of sibling hashes in the proof.
        expected_proof_len: usize,

        /// Actual number of sibling hashes provided in the proof.
        num_siblings: usize,
    },

    /// Matrix heights are incompatible; they cannot share a common binary Merkle tree.
    #[error("incompatible heights: matrices cannot share a common binary Merkle tree")]
    IncompatibleHeights,

    /// The queried row index exceeds the maximum height.
    #[error("index out of bounds: index {index} exceeds max height {max_height}")]
    IndexOutOfBounds {
        /// Maximum admissible height.
        max_height: usize,
        /// Row index that was provided.
        index: usize,
    },

    /// Attempted to open an empty batch (no committed matrices).
    #[error("empty batch: attempted to open an empty batch with no committed matrices")]
    EmptyBatch,

    /// The computed Merkle digest does not match any entry in the cap.
    #[error("cap mismatch: computed digest does not match any entry in the Merkle cap")]
    CapMismatch,

    /// The cap height is too large for the tree.
    #[error("cap height {cap_height} exceeds tree depth {tree_depth}")]
    WrongCapHeight {
        /// The configured cap height.
        cap_height: usize,
        /// The actual tree depth.
        tree_depth: usize,
    },

    /// A pruned batch opening could not be restored (malformed proof).
    #[error("malformed pruned proof: cannot restore full authentication paths")]
    MalformedPrunedProof,
}

/// The arity schedule and query positions for a given Merkle path.
/// This is used to replay the arity schedule for a given Merkle path and recover, for each
/// tree level, both:
/// - the compression arity `step` used at that level (either `N` or `2`), and
/// - the position of the queried child within its group, `pos_in_group`, in
///   the range `0..step`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArityAndPositions {
    /// The arity schedule for the Merkle path.
    pub arity_schedule: Vec<usize>,
    /// The query positions for the Merkle path.
    pub query_positions: Vec<usize>,
}

impl<P, PW, H, C, const N: usize, const DIGEST_ELEMS: usize>
    MerkleTreeMmcs<P, PW, H, C, N, DIGEST_ELEMS>
{
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
        if dimensions.is_empty() {
            return Err(EmptyBatch);
        }

        let mut heights_tallest_first = dimensions
            .iter()
            .enumerate()
            .sorted_by_key(|(_, dims)| Reverse(dims.height))
            .peekable();

        // Matrix heights that round up to the same power of two must be equal.
        if !heights_tallest_first
            .clone()
            .map(|(_, dims)| dims.height)
            .tuple_windows()
            .all(|(curr, next)| {
                curr == next || curr.next_power_of_two() != next.next_power_of_two()
            })
        {
            return Err(IncompatibleHeights);
        }

        // Initial padded height and bounds check, identical to verify_batch.
        let (max_height, mut curr_height_padded) = match heights_tallest_first.peek() {
            Some((_, dims)) => {
                let max_height = dims.height;
                let curr_height_padded = padded_len(max_height, N);
                (max_height, curr_height_padded)
            }
            None => return Err(EmptyBatch),
        };

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
        // No commitments → no tree → no schedule.
        if dimensions.is_empty() {
            return Err(EmptyBatch);
        }

        // Phase 1: order matrices tallest-first so the walk can peek ahead
        // and consume each one at its injection layer.
        let mut heights_tallest_first = dimensions
            .iter()
            .enumerate()
            .sorted_by_key(|(_, dims)| Reverse(dims.height))
            .peekable();

        // Heights sharing the same next-power-of-two must match exactly.
        //
        //     [8, 5]   → both round to 8, differ → reject
        //     [8, 8]   → identical at the same slot → OK
        //     [8, 4]   → distinct slots → OK
        if !heights_tallest_first
            .clone()
            .map(|(_, dims)| dims.height)
            .tuple_windows()
            .all(|(curr, next)| {
                curr == next || curr.next_power_of_two() != next.next_power_of_two()
            })
        {
            return Err(IncompatibleHeights);
        }

        // Phase 2: walk from leaves toward the root.
        //
        // Seed length is `padded_len(max_height, N)` — same padding the
        // prover uses to round the leaf layer up to a full N-ary group.
        let (max_height, mut curr_height_padded) = match heights_tallest_first.peek() {
            Some((_, dims)) => {
                let max_height = dims.height;
                let curr_height_padded = padded_len(max_height, N);
                (max_height, curr_height_padded)
            }
            None => return Err(EmptyBatch),
        };

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
    /// Restores full authentication paths from the pruned proof, then verifies
    /// each one individually using the standard single-index verification logic.
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
        // Phase 1: Derive the expected full-path sibling count from the
        // verifier-known dimensions, *not* from the proof.
        //
        // Each schedule entry contributes `step - 1` siblings: one of the
        // step children is the queried node itself, the rest are siblings.
        //
        //   schedule = [4, 2]   → 3 + 1 = 4 expected siblings
        //   schedule = [2, 2, 2] → 1 + 1 + 1 = 3 expected siblings
        let proof_arity = self.proof_arity_schedule(dimensions)?;
        let full_sibling_count: usize = proof_arity.iter().map(|step| step - 1).sum();

        // Phase 2: Restore the full (unpruned) authentication paths from
        // the compact representation. This copies shared upper siblings
        // from each predecessor in a single forward pass.
        let restored = restore_paths(&pruned_opening.pruned_proof, full_sibling_count)
            .ok_or(MalformedPrunedProof)?;

        // The number of restored paths must match the number of query openings.
        if restored.len() != pruned_opening.opened_values.len() {
            return Err(WrongBatchSize);
        }

        // Phase 3: Verify each restored path individually against the commitment.
        //
        // Each path's flat sibling buffer is borrowed directly — no cloning.
        // The standard single-index verification hashes the opened values,
        // replays the arity schedule, and checks the result against the cap.
        //
        // TODO: amortize verifier cost across paths the same way the proof
        // is amortized — shared upper-level compressions are recomputed once
        // per path instead of once per shared subtree.
        for (auth_path, opened) in restored.iter().zip(&pruned_opening.opened_values) {
            let batch_ref = BatchOpeningRef::new(opened.as_slice(), &auth_path.siblings);
            self.verify_batch(commit, dimensions, auth_path.leaf_index, batch_ref)?;
        }

        // Move the opened values out of the consumed struct — zero-copy return.
        Ok(pruned_opening.opened_values)
    }
}

/// A batch opening with pruned Merkle authentication paths.
///
/// The pruned equivalent of opening multiple indices individually.
/// Shared sibling digests between queries are removed.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(bound(serialize = "T: Serialize, [D; DIGEST_ELEMS]: Serialize"))]
#[serde(bound(
    deserialize = "T: serde::de::DeserializeOwned, [D; DIGEST_ELEMS]: serde::de::DeserializeOwned"
))]
pub struct PrunedBatchOpening<T, D, const DIGEST_ELEMS: usize> {
    /// Opened matrix rows for each query, in original input order.
    /// Outer index = query, middle index = matrix, inner = row elements.
    pub opened_values: Vec<Vec<Vec<T>>>,

    /// Compact authentication paths with redundant siblings removed.
    pub pruned_proof: crate::pruning::PrunedMerklePaths<D, DIGEST_ELEMS>,
}

impl<P, PW, H, C, const N: usize, const DIGEST_ELEMS: usize> Mmcs<P::Value>
    for MerkleTreeMmcs<P, PW, H, C, N, DIGEST_ELEMS>
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
    type ProverData<M> = MerkleTree<P::Value, PW::Value, M, N, DIGEST_ELEMS>;
    type Commitment = MerkleCap<P::Value, [PW::Value; DIGEST_ELEMS]>;
    type Proof = Vec<[PW::Value; DIGEST_ELEMS]>;
    type Error = MerkleTreeError;

    fn commit<M: Matrix<P::Value>>(
        &self,
        inputs: Vec<M>,
    ) -> (Self::Commitment, Self::ProverData<M>) {
        if let Some(max_height) = inputs.iter().map(|m| m.height()).max()
            && max_height > 0
        {
            let log_max_height = log2_ceil_usize(max_height);
            for matrix in &inputs {
                let height = matrix.height();
                assert!(height > 0, "matrix height 0 not supported");

                let log_height = log2_ceil_usize(height);
                let bits_reduced = log_max_height - log_height;
                // ceil(max / 2^{bits_reduced}) without risk of shift overflow
                let expected_height = ((max_height - 1) >> bits_reduced) + 1;

                assert!(
                    height == expected_height,
                    "matrix height {height} incompatible with tallest height {max_height}; \
                         expected ceil_div({max_height}, 2^{bits_reduced}) = {expected_height} \
                         so every global index maps to a row at depth {bits_reduced}"
                );
            }
        } else {
            panic!("all matrices have height 0");
        }

        let tree = MerkleTree::new::<P, PW, H, C>(&self.hash, &self.compress, inputs);

        // Make cap_height fit this tree (small trees during FRI folding may have fewer layers)
        let num_layers = tree.num_layers();
        let effective_cap_height = self.cap_height.min(num_layers.saturating_sub(1));

        let cap = tree.cap(effective_cap_height);
        (cap, tree)
    }

    /// Opens a batch of rows from committed matrices.
    ///
    /// Returns `(openings, proof)` where `openings` is a vector whose `i`th element is
    /// the `j`th row of the ith matrix `M[i]`, and `proof` is the vector of sibling
    /// Merkle tree nodes allowing the verifier to reconstruct the committed cap entry.
    ///
    /// At each tree level the number of siblings is `step - 1`, where `step` is either
    /// `N` (full N-ary compression) or `2` (binary) at levels that sit between N-ary layers).
    /// For binary levels the remaining `N - 2` inputs are padded with the default digest.
    fn open_batch<M: Matrix<P::Value>>(
        &self,
        index: usize,
        prover_data: &MerkleTree<P::Value, PW::Value, M, N, DIGEST_ELEMS>,
    ) -> BatchOpening<P::Value, Self> {
        let max_height = self.get_max_height(prover_data);
        assert!(
            index < max_height,
            "index {index} out of bounds for height {max_height}"
        );
        let log_max_height = log2_ceil_usize(max_height);

        // Get the matrix rows encountered along the path from the cap to the given leaf index.
        let openings = prover_data
            .leaves
            .iter()
            .map(|matrix| {
                let log2_height = log2_ceil_usize(matrix.height());
                let bits_reduced = log_max_height - log2_height;
                let reduced_index = index >> bits_reduced;
                matrix.row(reduced_index).unwrap().into_iter().collect()
            })
            .collect_vec();

        // Only include siblings up to (but not including) the cap layer.
        let num_layers = prover_data.digest_layers.len();
        let effective_cap_height = self.cap_height.min(num_layers.saturating_sub(1));
        let proof_levels = num_layers
            .saturating_sub(1)
            .saturating_sub(effective_cap_height);

        let mut proof = Vec::new();
        let mut idx = index;
        for layer_idx in 0..proof_levels {
            let step = prover_data.arity_schedule[layer_idx];
            let group_start = (idx / step) * step;
            let pos_in_group = idx % step;
            for k in 0..step {
                if k != pos_in_group {
                    proof.push(prover_data.digest_layers[layer_idx][group_start + k]);
                }
            }
            idx /= step;
        }

        BatchOpening::new(openings, proof)
    }

    fn get_matrices<'a, M: Matrix<P::Value>>(
        &self,
        prover_data: &'a Self::ProverData<M>,
    ) -> Vec<&'a M> {
        prover_data.leaves.iter().collect()
    }

    /// Verifies an opened batch of rows with respect to a given commitment (Merkle cap).
    ///
    /// At each tree level, the verifier determines whether this level used a full
    /// N-ary step or a binary step (when a matrix injection sits between N-ary
    /// layers). Binary steps carry `1` sibling in the proof; N-ary steps carry
    /// `N-1` siblings. Both cases use the same N-to-1 compression function —
    /// binary steps pad the remaining `N-2` slots with the default digest.
    ///
    /// # Arguments
    /// - `commit`: The Merkle cap of the tree.
    /// - `dimensions`: A vector of the dimensions of the matrices committed to.
    /// - `index`: The index of a leaf in the tree.
    /// - `batch_proof`: A reference to a batched opening proof, containing:
    ///   - `opened_values`: A vector of matrix rows. Assume that the tallest matrix committed
    ///     to has height `2^n >= M_tall.height() > 2^{n - 1}` and the `j`th matrix has height
    ///     `2^m >= Mj.height() > 2^{m - 1}`. Then `j`'th value of opened values must be the row `Mj[index >> (m - n)]`.
    ///   - `opening_proof`: A vector of sibling nodes. The `i`th element should be the node at level `i`
    ///     with index `(index << i) ^ 1`.
    ///
    /// # Returns
    /// `Ok(())` if the verification is successful; otherwise returns a verification error.
    fn verify_batch(
        &self,
        commit: &Self::Commitment,
        dimensions: &[Dimensions],
        mut index: usize,
        batch_proof: BatchOpeningRef<'_, P::Value, Self>,
    ) -> Result<(), Self::Error> {
        let (opened_values, opening_proof) = batch_proof.unpack();
        // Check that the openings have the correct shape.
        if dimensions.len() != opened_values.len() {
            return Err(WrongBatchSize);
        }

        let mut heights_tallest_first = dimensions
            .iter()
            .enumerate()
            .sorted_by_key(|(_, dims)| Reverse(dims.height))
            .peekable();

        // Matrix heights that round up to the same power of two must be equal
        if !heights_tallest_first
            .clone()
            .map(|(_, dims)| dims.height)
            .tuple_windows()
            .all(|(curr, next)| {
                curr == next || curr.next_power_of_two() != next.next_power_of_two()
            })
        {
            return Err(IncompatibleHeights);
        }

        // Get the initial height padded to a multiple of N. As heights_tallest_first is sorted,
        // the initial height will be the maximum height.
        let (max_height, mut curr_height_padded) = match heights_tallest_first.peek() {
            Some((_, dims)) => {
                let max_height = dims.height;
                let curr_height_padded = padded_len(max_height, N);
                (max_height, curr_height_padded)
            }
            None => return Err(EmptyBatch),
        };

        if index >= max_height {
            return Err(IndexOutOfBounds { max_height, index });
        }

        // Hash all matrix openings at the current height.
        let leaf_height_npt = max_height.next_power_of_two();
        let mut digest: [PW::Value; DIGEST_ELEMS] = self.hash.hash_iter_slices(
            heights_tallest_first
                .peeking_take_while(|(_, dims)| dims.height.next_power_of_two() == leaf_height_npt)
                .map(|(i, _)| opened_values[i].as_slice()),
        );

        let default_digest = [PW::Value::default(); DIGEST_ELEMS];

        // Replay the arity schedule that the prover computed during tree
        // construction. We use the remaining matrix heights to decide at
        // each level whether this was an N-ary or a binary step.
        let mut proof_pos: usize = 0;

        while proof_pos < opening_proof.len() {
            let step = select_arity_step::<N>(
                curr_height_padded,
                leaf_height_npt,
                heights_tallest_first.clone().map(|(_, dims)| dims.height),
            );

            let num_siblings = step - 1;
            if proof_pos + num_siblings > opening_proof.len() {
                return Err(WrongHeight {
                    expected_proof_len: proof_pos + num_siblings,
                    num_siblings: opening_proof.len(),
                });
            }
            let siblings = &opening_proof[proof_pos..proof_pos + num_siblings];
            proof_pos += num_siblings;

            let pos_in_group = index % step;
            let mut sibling_idx = 0;
            let inputs: [_; N] = core::array::from_fn(|k| {
                if k < step {
                    if k == pos_in_group {
                        digest
                    } else {
                        let s = siblings[sibling_idx];
                        sibling_idx += 1;
                        s
                    }
                } else {
                    default_digest
                }
            });

            digest = self.compress.compress(inputs);
            index /= step;
            let logical_next = curr_height_padded / step;
            curr_height_padded = padded_len(logical_next, N);

            // Check if there are any new matrix rows to inject at the next height.
            let logical_next_npt = logical_next.next_power_of_two();
            let next_height = heights_tallest_first
                .peek()
                .map(|(_, dims)| dims.height)
                .filter(|h| h.next_power_of_two() == logical_next_npt);
            if let Some(next_height) = next_height {
                let next_height_openings_digest = self.hash.hash_iter_slices(
                    heights_tallest_first
                        .peeking_take_while(|(_, dims)| dims.height == next_height)
                        .map(|(i, _)| opened_values[i].as_slice()),
                );

                let inject_inputs: [_; N] = core::array::from_fn(|k| {
                    if k == 0 {
                        digest
                    } else if k == 1 {
                        next_height_openings_digest
                    } else {
                        default_digest
                    }
                });
                digest = self.compress.compress(inject_inputs);
            }
        }

        // After processing the proof, `index` has been shifted by the proof length.
        // This index now points into the cap layer.
        let cap_index = index;
        if cap_index < commit.num_roots() && commit[cap_index] == digest {
            Ok(())
        } else {
            Err(CapMismatch)
        }
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec::Vec;
    use alloc::{format, vec};

    use itertools::Itertools;
    use p3_baby_bear::{BabyBear, Poseidon2BabyBear};
    use p3_commit::{BatchOpeningRef, Mmcs};
    use p3_field::{Field, PrimeCharacteristicRing};
    use p3_matrix::dense::RowMajorMatrix;
    use p3_matrix::{Dimensions, Matrix};
    use p3_symmetric::{
        CryptographicHasher, PaddingFreeSponge, PseudoCompressionFunction, TruncatedPermutation,
    };
    use proptest::prelude::*;
    use rand::SeedableRng;
    use rand::rngs::SmallRng;

    use super::MerkleTreeMmcs;

    type F = BabyBear;

    type Perm = Poseidon2BabyBear<16>;
    type MyHash = PaddingFreeSponge<Perm, 16, 8, 8>;
    type MyCompress = TruncatedPermutation<Perm, 2, 8, 16>;
    type MyMmcs =
        MerkleTreeMmcs<<F as Field>::Packing, <F as Field>::Packing, MyHash, MyCompress, 2, 8>;

    // 4-ary Poseidon2-based MMCS:
    //   - width-16 permutation for leaves
    //   - width-32 (4-to-1) permutation for internal compression
    type PermWide = Poseidon2BabyBear<32>;
    type MyCompress4 = TruncatedPermutation<PermWide, 4, 8, 32>;
    type MyMmcs4 =
        MerkleTreeMmcs<<F as Field>::Packing, <F as Field>::Packing, MyHash, MyCompress4, 4, 8>;

    #[test]
    fn commit_single_1x8() {
        let mut rng = SmallRng::seed_from_u64(1);
        let perm = Perm::new_from_rng_128(&mut rng);
        let hash = MyHash::new(perm.clone());
        let compress = MyCompress::new(perm);
        let mmcs = MyMmcs::new(hash.clone(), compress.clone(), 0);

        // v = [2, 1, 2, 2, 0, 0, 1, 0]
        let v = vec![
            F::TWO,
            F::ONE,
            F::TWO,
            F::TWO,
            F::ZERO,
            F::ZERO,
            F::ONE,
            F::ZERO,
        ];
        let (commit, _) = mmcs.commit_vec(v.clone());

        let expected_result = compress.compress([
            compress.compress([
                compress.compress([hash.hash_item(v[0]), hash.hash_item(v[1])]),
                compress.compress([hash.hash_item(v[2]), hash.hash_item(v[3])]),
            ]),
            compress.compress([
                compress.compress([hash.hash_item(v[4]), hash.hash_item(v[5])]),
                compress.compress([hash.hash_item(v[6]), hash.hash_item(v[7])]),
            ]),
        ]);
        assert_eq!(commit[0], expected_result);
    }

    #[test]
    fn poseidon2_4ary_single_matrix_roundtrip() {
        let mut rng = SmallRng::seed_from_u64(42);

        // Leaf hasher: width-16
        let perm16 = Perm::new_from_rng_128(&mut rng);
        let hash = MyHash::new(perm16);

        // Internal compression: width-32, 4-to-1.
        let perm32 = PermWide::new_from_rng_128(&mut rng);
        let compress4 = MyCompress4::new(perm32);

        let mmcs4 = MyMmcs4::new(hash, compress4, 0);

        let mat = RowMajorMatrix::<F>::rand(&mut rng, 64, 8);
        let dims = vec![mat.dimensions()];
        let (commit, prover_data) = mmcs4.commit(vec![mat]);

        let index = 17;
        let opening = mmcs4.open_batch(index, &prover_data);
        mmcs4
            .verify_batch(&commit, &dims, index, (&opening).into())
            .expect("4-ary Poseidon2 MMCS roundtrip should verify");
    }

    #[test]
    fn small_height_4ary_roundtrip() {
        let mut rng = SmallRng::seed_from_u64(99);
        let perm16 = Perm::new_from_rng_128(&mut rng);
        let hash = MyHash::new(perm16);
        let perm32 = PermWide::new_from_rng_128(&mut rng);
        let compress4 = MyCompress4::new(perm32);
        let mmcs4 = MyMmcs4::new(hash, compress4, 0);

        // max_height=2 with N=4: edge case where max_height < N.
        let mat = RowMajorMatrix::<F>::rand(&mut rng, 2, 8);
        let dims = vec![mat.dimensions()];
        let (commit, prover_data) = mmcs4.commit(vec![mat]);

        for index in 0..2 {
            let opening = mmcs4.open_batch(index, &prover_data);
            mmcs4
                .verify_batch(&commit, &dims, index, (&opening).into())
                .expect("small-height 4-ary roundtrip should verify");
        }
    }

    #[test]
    fn commit_single_8x1() {
        let mut rng = SmallRng::seed_from_u64(1);
        let perm = Perm::new_from_rng_128(&mut rng);
        let hash = MyHash::new(perm.clone());
        let compress = MyCompress::new(perm);
        let mmcs = MyMmcs::new(hash.clone(), compress, 0);

        let mat = RowMajorMatrix::<F>::rand(&mut rng, 1, 8);
        let (commit, _) = mmcs.commit(vec![mat.clone()]);

        let expected_result = hash.hash_iter(mat.vertically_packed_row(0));
        assert_eq!(commit[0], expected_result);
    }

    #[test]
    fn commit_single_2x2() {
        let mut rng = SmallRng::seed_from_u64(1);
        let perm = Perm::new_from_rng_128(&mut rng);
        let hash = MyHash::new(perm.clone());
        let compress = MyCompress::new(perm);
        let mmcs = MyMmcs::new(hash.clone(), compress.clone(), 0);

        // mat = [
        //   0 1
        //   2 1
        // ]
        let mat = RowMajorMatrix::new(vec![F::ZERO, F::ONE, F::TWO, F::ONE], 2);

        let (commit, _) = mmcs.commit(vec![mat]);

        let expected_result = compress.compress([
            hash.hash_slice(&[F::ZERO, F::ONE]),
            hash.hash_slice(&[F::TWO, F::ONE]),
        ]);
        assert_eq!(commit[0], expected_result);
    }

    #[test]
    fn commit_single_2x3() {
        let mut rng = SmallRng::seed_from_u64(1);
        let perm = Perm::new_from_rng_128(&mut rng);
        let hash = MyHash::new(perm.clone());
        let compress = MyCompress::new(perm);
        let mmcs = MyMmcs::new(hash.clone(), compress.clone(), 0);
        let default_digest = [F::ZERO; 8];

        // mat = [
        //   0 1
        //   2 1
        //   2 2
        // ]
        let mat = RowMajorMatrix::new(vec![F::ZERO, F::ONE, F::TWO, F::ONE, F::TWO, F::TWO], 2);

        let (commit, _) = mmcs.commit(vec![mat]);

        let expected_result = compress.compress([
            compress.compress([
                hash.hash_slice(&[F::ZERO, F::ONE]),
                hash.hash_slice(&[F::TWO, F::ONE]),
            ]),
            compress.compress([hash.hash_slice(&[F::TWO, F::TWO]), default_digest]),
        ]);
        assert_eq!(commit[0], expected_result);
    }

    #[test]
    fn commit_mixed() {
        let mut rng = SmallRng::seed_from_u64(1);
        let perm = Perm::new_from_rng_128(&mut rng);
        let hash = MyHash::new(perm.clone());
        let compress = MyCompress::new(perm);
        let mmcs = MyMmcs::new(hash.clone(), compress.clone(), 0);
        let default_digest = [F::ZERO; 8];

        // mat_1 = [
        //   0 1
        //   2 1
        //   2 2
        //   2 1
        //   2 2
        // ]
        let mat_1 = RowMajorMatrix::new(
            vec![
                F::ZERO,
                F::ONE,
                F::TWO,
                F::ONE,
                F::TWO,
                F::TWO,
                F::TWO,
                F::ONE,
                F::TWO,
                F::TWO,
            ],
            2,
        );
        // mat_2 = [
        //   1 2 1
        //   0 2 2
        //   1 2 1
        // ]
        let mat_2 = RowMajorMatrix::new(
            vec![
                F::ONE,
                F::TWO,
                F::ONE,
                F::ZERO,
                F::TWO,
                F::TWO,
                F::ONE,
                F::TWO,
                F::ONE,
            ],
            3,
        );

        let (commit, prover_data) = mmcs.commit(vec![mat_1, mat_2]);

        let mat_1_leaf_hashes = [
            hash.hash_slice(&[F::ZERO, F::ONE]),
            hash.hash_slice(&[F::TWO, F::ONE]),
            hash.hash_slice(&[F::TWO, F::TWO]),
            hash.hash_slice(&[F::TWO, F::ONE]),
            hash.hash_slice(&[F::TWO, F::TWO]),
        ];
        let mat_2_leaf_hashes = [
            hash.hash_slice(&[F::ONE, F::TWO, F::ONE]),
            hash.hash_slice(&[F::ZERO, F::TWO, F::TWO]),
            hash.hash_slice(&[F::ONE, F::TWO, F::ONE]),
        ];

        let expected_result = compress.compress([
            compress.compress([
                compress.compress([
                    compress.compress([mat_1_leaf_hashes[0], mat_1_leaf_hashes[1]]),
                    mat_2_leaf_hashes[0],
                ]),
                compress.compress([
                    compress.compress([mat_1_leaf_hashes[2], mat_1_leaf_hashes[3]]),
                    mat_2_leaf_hashes[1],
                ]),
            ]),
            compress.compress([
                compress.compress([
                    compress.compress([mat_1_leaf_hashes[4], default_digest]),
                    mat_2_leaf_hashes[2],
                ]),
                default_digest,
            ]),
        ]);

        assert_eq!(commit[0], expected_result);

        let (opened_values, _) = mmcs.open_batch(2, &prover_data).unpack();
        assert_eq!(
            opened_values,
            vec![vec![F::TWO, F::TWO], vec![F::ZERO, F::TWO, F::TWO]]
        );
    }

    #[test]
    fn commit_either_order() {
        let mut rng = SmallRng::seed_from_u64(1);
        let perm = Perm::new_from_rng_128(&mut rng);
        let hash = MyHash::new(perm.clone());
        let compress = MyCompress::new(perm);
        let mmcs = MyMmcs::new(hash, compress, 0);

        let input_1 = RowMajorMatrix::<F>::rand(&mut rng, 5, 8);
        let input_2 = RowMajorMatrix::<F>::rand(&mut rng, 3, 16);

        let (commit_1_2, _) = mmcs.commit(vec![input_1.clone(), input_2.clone()]);
        let (commit_2_1, _) = mmcs.commit(vec![input_2, input_1]);
        assert_eq!(commit_1_2, commit_2_1);
    }

    #[test]
    #[should_panic]
    fn mismatched_heights() {
        let mut rng = SmallRng::seed_from_u64(1);
        let perm = Perm::new_from_rng_128(&mut rng);
        let hash = MyHash::new(perm.clone());
        let compress = MyCompress::new(perm);
        let mmcs = MyMmcs::new(hash, compress, 0);

        // attempt to commit to a mat with 8 rows and a mat with 7 rows. this should panic.
        let large_mat = RowMajorMatrix::new([1, 2, 3, 4, 5, 6, 7, 8].map(F::from_u8).to_vec(), 1);
        let small_mat = RowMajorMatrix::new([1, 2, 3, 4, 5, 6, 7].map(F::from_u8).to_vec(), 1);
        let _ = mmcs.commit(vec![large_mat, small_mat]);
    }

    #[test]
    fn verify_tampered_proof_fails() {
        let mut rng = SmallRng::seed_from_u64(1);
        let perm = Perm::new_from_rng_128(&mut rng);
        let hash = MyHash::new(perm.clone());
        let compress = MyCompress::new(perm);
        let mmcs = MyMmcs::new(hash, compress, 0);

        // 4 8x1 matrixes, 4 8x2 matrixes
        let mut mats = (0..4)
            .map(|_| RowMajorMatrix::<F>::rand(&mut rng, 8, 1))
            .collect_vec();
        let large_mat_dims = (0..4).map(|_| Dimensions {
            height: 8,
            width: 1,
        });
        mats.extend((0..4).map(|_| RowMajorMatrix::<F>::rand(&mut rng, 8, 2)));
        let small_mat_dims = (0..4).map(|_| Dimensions {
            height: 8,
            width: 2,
        });

        let (commit, prover_data) = mmcs.commit(mats);

        // open the 3rd row of each matrix, mess with proof, and verify
        let mut batch_opening = mmcs.open_batch(3, &prover_data);
        batch_opening.opening_proof[0][0] += F::ONE;
        mmcs.verify_batch(
            &commit,
            &large_mat_dims.chain(small_mat_dims).collect_vec(),
            3,
            (&batch_opening).into(),
        )
        .expect_err("expected verification to fail");
    }

    #[test]
    fn size_gaps() {
        let mut rng = SmallRng::seed_from_u64(1);
        let perm = Perm::new_from_rng_128(&mut rng);
        let hash = MyHash::new(perm.clone());
        let compress = MyCompress::new(perm);
        let mmcs = MyMmcs::new(hash, compress, 0);

        // 4 mats with 1000 rows, 8 columns
        let mut mats = (0..4)
            .map(|_| RowMajorMatrix::<F>::rand(&mut rng, 1000, 8))
            .collect_vec();
        let large_mat_dims = (0..4).map(|_| Dimensions {
            height: 1000,
            width: 8,
        });

        // 5 mats with 125 rows, 8 columns
        mats.extend((0..5).map(|_| RowMajorMatrix::<F>::rand(&mut rng, 125, 8)));
        let medium_mat_dims = (0..5).map(|_| Dimensions {
            height: 125,
            width: 8,
        });

        // 6 mats with 8 rows, 8 columns
        mats.extend((0..6).map(|_| RowMajorMatrix::<F>::rand(&mut rng, 8, 8)));
        let small_mat_dims = (0..6).map(|_| Dimensions {
            height: 8,
            width: 8,
        });

        // 7 tiny mat with 1 row, 8 columns
        mats.extend((0..7).map(|_| RowMajorMatrix::<F>::rand(&mut rng, 1, 8)));
        let tiny_mat_dims = (0..7).map(|_| Dimensions {
            height: 1,
            width: 8,
        });

        let dims = large_mat_dims
            .chain(medium_mat_dims)
            .chain(small_mat_dims)
            .chain(tiny_mat_dims)
            .collect_vec();

        let (commit, prover_data) = mmcs.commit(mats);

        for &index in &[0, 6, 124, 999] {
            let batch_opening = mmcs.open_batch(index, &prover_data);
            mmcs.verify_batch(&commit, &dims, index, (&batch_opening).into())
                .expect("expected verification to succeed");
        }
    }

    #[test]
    fn different_widths() {
        let mut rng = SmallRng::seed_from_u64(1);
        let perm = Perm::new_from_rng_128(&mut rng);
        let hash = MyHash::new(perm.clone());
        let compress = MyCompress::new(perm);
        let mmcs = MyMmcs::new(hash, compress, 0);

        // 10 mats with 32 rows where the ith mat has i + 1 cols
        let mats = (0..10)
            .map(|i| RowMajorMatrix::<F>::rand(&mut rng, 32, i + 1))
            .collect_vec();
        let dims = mats.iter().map(|m| m.dimensions()).collect_vec();

        let (commit, prover_data) = mmcs.commit(mats);
        let batch_opening = mmcs.open_batch(17, &prover_data);
        mmcs.verify_batch(&commit, &dims, 17, (&batch_opening).into())
            .expect("expected verification to succeed");
    }

    #[test]
    #[should_panic(expected = "matrix height 5 incompatible")]
    fn commit_rejects_missing_leaf_coverage() {
        let mut rng = SmallRng::seed_from_u64(9);
        let perm = Perm::new_from_rng_128(&mut rng);
        let hash = MyHash::new(perm.clone());
        let compress = MyCompress::new(perm);
        let mmcs = MyMmcs::new(hash, compress, 0);

        let tallest = RowMajorMatrix::new(vec![F::ONE; 11], 1);
        let invalid = RowMajorMatrix::new(vec![F::ONE; 5], 1);

        // We expect a panic because the smaller matrix needs ceil(11 / 2) == 6 rows;
        // height 5 would leave the global index 5 unmapped at that layer.
        let _ = mmcs.commit(vec![tallest, invalid]);
    }

    #[test]
    fn cap_height_produces_shorter_proofs() {
        let mut rng = SmallRng::seed_from_u64(2);
        let perm = Perm::new_from_rng_128(&mut rng);
        let hash = MyHash::new(perm.clone());
        let compress = MyCompress::new(perm);

        // Tree with 32 leaves -> log2(32) = 5 layers
        let mat = RowMajorMatrix::<F>::rand(&mut rng, 32, 4);
        let dims = vec![Dimensions {
            height: 32,
            width: 4,
        }];

        for cap_height in 0..5 {
            let mmcs = MyMmcs::new(hash.clone(), compress.clone(), cap_height);
            let (commit, data) = mmcs.commit(vec![mat.clone()]);

            // Cap should have 2^cap_height elements
            assert_eq!(commit.num_roots(), 1 << cap_height);

            // Proof should have (log2(32) - cap_height) = 5 - cap_height elements
            let opening = mmcs.open_batch(17, &data);
            assert_eq!(opening.opening_proof.len(), 5 - cap_height);

            // Verification should succeed
            mmcs.verify_batch(&commit, &dims, 17, (&opening).into())
                .expect("verification should succeed");
        }
    }

    #[test]
    fn cap_verification_with_various_indices() {
        let mut rng = SmallRng::seed_from_u64(3);
        let perm = Perm::new_from_rng_128(&mut rng);
        let hash = MyHash::new(perm.clone());
        let compress = MyCompress::new(perm);

        let mmcs = MyMmcs::new(hash, compress, 2); // cap_height = 2 -> 4 cap elements

        // 64 rows -> 6 layers, proofs have 6 - 2 = 4 siblings
        let mat = RowMajorMatrix::<F>::rand(&mut rng, 64, 8);
        let dims = vec![Dimensions {
            height: 64,
            width: 8,
        }];

        let (commit, data) = mmcs.commit(vec![mat]);

        assert_eq!(commit.num_roots(), 4);

        // Test various indices that map to different cap entries
        for index in [0, 15, 16, 31, 32, 47, 48, 63] {
            let opening = mmcs.open_batch(index, &data);
            assert_eq!(opening.opening_proof.len(), 4);
            mmcs.verify_batch(&commit, &dims, index, (&opening).into())
                .unwrap_or_else(|_| panic!("verification at index {index} should succeed"));
        }
    }

    #[test]
    fn cap_tampered_proof_fails() {
        let mut rng = SmallRng::seed_from_u64(4);
        let perm = Perm::new_from_rng_128(&mut rng);
        let hash = MyHash::new(perm.clone());
        let compress = MyCompress::new(perm);

        let mmcs = MyMmcs::new(hash, compress, 1);

        let mat = RowMajorMatrix::<F>::rand(&mut rng, 16, 4);
        let dims = vec![Dimensions {
            height: 16,
            width: 4,
        }];

        let (commit, data) = mmcs.commit(vec![mat]);

        let mut opening = mmcs.open_batch(7, &data);
        opening.opening_proof[0][0] += F::ONE;

        mmcs.verify_batch(&commit, &dims, 7, (&opening).into())
            .expect_err("tampered proof should fail verification");
    }

    #[test]
    fn cap_with_mixed_height_matrices() {
        let mut rng = SmallRng::seed_from_u64(5);
        let perm = Perm::new_from_rng_128(&mut rng);
        let hash = MyHash::new(perm.clone());
        let compress = MyCompress::new(perm);

        let mmcs = MyMmcs::new(hash, compress, 2);

        // 64 rows and 8 rows (8 = 64 / 8, valid ratio)
        let mat_large = RowMajorMatrix::<F>::rand(&mut rng, 64, 4);
        let mat_small = RowMajorMatrix::<F>::rand(&mut rng, 8, 6);

        let dims = vec![
            Dimensions {
                height: 64,
                width: 4,
            },
            Dimensions {
                height: 8,
                width: 6,
            },
        ];

        let (commit, data) = mmcs.commit(vec![mat_large, mat_small]);
        assert_eq!(commit.num_roots(), 4);

        // Verify various indices
        for index in [0, 7, 8, 15, 32, 63] {
            let opening = mmcs.open_batch(index, &data);
            mmcs.verify_batch(&commit, &dims, index, (&opening).into())
                .unwrap_or_else(|_| panic!("verification at index {index} should succeed"));
        }
    }

    #[test]
    fn cap_height_shortened_for_small_trees() {
        let mut rng = SmallRng::seed_from_u64(6);
        let perm = Perm::new_from_rng_128(&mut rng);
        let hash = MyHash::new(perm.clone());
        let compress = MyCompress::new(perm);

        // For 4 rows: digest_layers has [4 row hashes, 2, 1] = 3 layers.
        // max cap_height = 3 - 1 = 2, giving 2^2 = 4 cap entries.
        // With cap_height = 10 (too large), it should be shortened to 2.
        let mmcs = MyMmcs::new(hash, compress, 10);
        let mat = RowMajorMatrix::<F>::rand(&mut rng, 4, 4);
        let (cap, prover_data) = mmcs.commit(vec![mat]);

        // Cap should have 4 entries (shortened to height 2)
        assert_eq!(cap.num_roots(), 4);
        assert_eq!(cap.height(), 2);

        // Verify that opening and verification still work
        let (opening, proof) = mmcs.open_batch(0, &prover_data).unpack();
        let dims = vec![Dimensions {
            width: 4,
            height: 4,
        }];
        mmcs.verify_batch(&cap, &dims, 0, BatchOpeningRef::new(&opening, &proof))
            .unwrap();
    }

    #[test]
    fn single_row_matrix_with_cap_height() {
        let mut rng = SmallRng::seed_from_u64(7);
        let perm = Perm::new_from_rng_128(&mut rng);
        let hash = MyHash::new(perm.clone());
        let compress = MyCompress::new(perm);

        // Single-row matrix: digest_layers has [1 row hash] = 1 layer.
        // max cap_height = 1 - 1 = 0, so any cap_height > 0 gets shortened.
        let mmcs = MyMmcs::new(hash, compress, 5);
        let mat = RowMajorMatrix::<F>::rand(&mut rng, 1, 8);
        let (cap, prover_data) = mmcs.commit(vec![mat]);

        // Cap should have 1 entry (shortened to height 0 = root)
        assert_eq!(cap.num_roots(), 1);
        assert_eq!(cap.height(), 0);

        // Proof length should be 0 (we're already at the cap)
        let (opening, proof) = mmcs.open_batch(0, &prover_data).unpack();
        assert!(
            proof.is_empty(),
            "proof should be empty for single-row tree"
        );

        let dims = vec![Dimensions {
            width: 8,
            height: 1,
        }];
        mmcs.verify_batch(&cap, &dims, 0, BatchOpeningRef::new(&opening, &proof))
            .unwrap();
    }

    #[test]
    fn proof_length_zero_at_max_cap_height() {
        let mut rng = SmallRng::seed_from_u64(8);
        let perm = Perm::new_from_rng_128(&mut rng);
        let hash = MyHash::new(perm.clone());
        let compress = MyCompress::new(perm);

        // 8 rows: digest_layers has [8, 4, 2, 1] = 4 layers, log_max_height = 3.
        // max cap_height = 4 - 1 = 3, giving 2^3 = 8 cap entries.
        // With cap_height = 3, proof length = 3 - 3 = 0.
        let mmcs = MyMmcs::new(hash, compress, 3);
        let mat = RowMajorMatrix::<F>::rand(&mut rng, 8, 4);
        let (cap, prover_data) = mmcs.commit(vec![mat]);

        // Cap should have 8 entries (all leaf hashes)
        assert_eq!(cap.num_roots(), 8);
        assert_eq!(cap.height(), 3);

        // Proof length should be 0 since cap is at leaf level
        for index in 0..8 {
            let (opening, proof) = mmcs.open_batch(index, &prover_data).unpack();
            assert!(
                proof.is_empty(),
                "proof should be empty when cap is at leaf level"
            );

            let dims = vec![Dimensions {
                width: 4,
                height: 8,
            }];
            mmcs.verify_batch(&cap, &dims, index, BatchOpeningRef::new(&opening, &proof))
                .unwrap();
        }
    }

    #[test]
    fn cap_height_exact_boundary() {
        let mut rng = SmallRng::seed_from_u64(9);
        let perm = Perm::new_from_rng_128(&mut rng);
        let hash = MyHash::new(perm.clone());
        let compress = MyCompress::new(perm);

        // 16 rows: digest_layers has [16, 8, 4, 2, 1] = 5 layers, log_max_height = 4.
        // max cap_height = 5 - 1 = 4.
        // Test cap_height = 4 (exact boundary, no shortening).
        let mmcs = MyMmcs::new(hash, compress, 4);
        let mat = RowMajorMatrix::<F>::rand(&mut rng, 16, 4);
        let (cap, prover_data) = mmcs.commit(vec![mat]);

        // Cap should have 16 entries
        assert_eq!(cap.num_roots(), 16);
        assert_eq!(cap.height(), 4);

        // Proof length should be 0
        let (opening, proof) = mmcs.open_batch(0, &prover_data).unpack();
        assert!(proof.is_empty());

        let dims = vec![Dimensions {
            width: 4,
            height: 16,
        }];
        mmcs.verify_batch(&cap, &dims, 0, BatchOpeningRef::new(&opening, &proof))
            .unwrap();

        // Also test cap_height = 3 (one below boundary, proof_len = 1)
        let mmcs2 = MyMmcs::new(
            MyHash::new(Perm::new_from_rng_128(&mut rng)),
            MyCompress::new(Perm::new_from_rng_128(&mut rng)),
            3,
        );
        let mat2 = RowMajorMatrix::<F>::rand(&mut rng, 16, 4);
        let (cap2, prover_data2) = mmcs2.commit(vec![mat2]);

        assert_eq!(cap2.num_roots(), 8);
        assert_eq!(cap2.height(), 3);

        let (opening2, proof2) = mmcs2.open_batch(0, &prover_data2).unpack();
        assert_eq!(proof2.len(), 1, "proof should have 1 sibling");

        mmcs2
            .verify_batch(&cap2, &dims, 0, BatchOpeningRef::new(&opening2, &proof2))
            .unwrap();
    }

    #[test]
    fn replay_arity_and_positions_binary() {
        let mut rng = SmallRng::seed_from_u64(123);
        let perm = Perm::new_from_rng_128(&mut rng);
        let hash = MyHash::new(perm.clone());
        let compress = MyCompress::new(perm);
        let mmcs = MyMmcs::new(hash, compress, 0);

        // Use a moderately sized tree to ensure multiple levels.
        let mat = RowMajorMatrix::<F>::rand(&mut rng, 32, 4);
        let dims = vec![mat.dimensions()];
        let (_commit, prover_data) = mmcs.commit(vec![mat]);

        // For a range of indices, the helper's schedule should match the tree's
        // actual arity_schedule and the naive index-based positions.
        for index in [0usize, 1, 5, 16, 31] {
            let opening = mmcs.open_batch(index, &prover_data);
            let (_opened_values, proof) = opening.unpack();

            let arity_and_positions = mmcs
                .replay_arity_and_positions(&dims, index, proof.len())
                .expect("schedule replay should succeed");
            let steps = arity_and_positions.arity_schedule;
            let positions = arity_and_positions.query_positions;

            // With cap_height = 0, we expect one step per non-root layer.
            let expected_levels = prover_data.digest_layers.len().saturating_sub(1);
            assert_eq!(steps.len(), expected_levels);
            assert_eq!(positions.len(), expected_levels);

            // Steps must match the concrete arity_schedule stored in the tree.
            for (i, &step) in steps.iter().enumerate() {
                assert_eq!(
                    step, prover_data.arity_schedule[i],
                    "step mismatch at level {i} for index {index}"
                );
            }

            // Positions must agree with repeatedly dividing the index by the step.
            let mut idx = index;
            for (level, (&step, &pos_in_group)) in steps.iter().zip(&positions).enumerate() {
                let expected_pos = idx % step;
                assert_eq!(
                    pos_in_group, expected_pos,
                    "pos_in_group mismatch at level {level} for index {index}"
                );
                idx /= step;
            }
        }
    }

    #[test]
    fn replay_arity_and_positions_4ary() {
        let mut rng = SmallRng::seed_from_u64(456);
        let perm16 = Perm::new_from_rng_128(&mut rng);
        let hash = MyHash::new(perm16);
        let perm32 = PermWide::new_from_rng_128(&mut rng);
        let compress4 = MyCompress4::new(perm32);
        let mmcs4 = MyMmcs4::new(hash, compress4, 0);

        // Height chosen so that both N-ary and possible binary bridge steps can appear.
        let mat = RowMajorMatrix::<F>::rand(&mut rng, 64, 8);
        let dims = vec![mat.dimensions()];
        let (_commit, prover_data) = mmcs4.commit(vec![mat]);

        for index in [0usize, 3, 7, 17, 42, 63] {
            let opening = mmcs4.open_batch(index, &prover_data);
            let (_opened_values, proof) = opening.unpack();

            let arity_and_positions = mmcs4
                .replay_arity_and_positions(&dims, index, proof.len())
                .expect("schedule replay should succeed");
            let steps = arity_and_positions.arity_schedule;
            let positions = arity_and_positions.query_positions;

            let expected_levels = prover_data.digest_layers.len().saturating_sub(1);
            assert_eq!(steps.len(), expected_levels);
            assert_eq!(positions.len(), expected_levels);

            // Each step must be either 2 (binary) or 4 (full 4-ary) and match the
            // concrete arity_schedule stored in the Merkle tree.
            for (i, &step) in steps.iter().enumerate() {
                assert!(
                    step == 2 || step == 4,
                    "unexpected step {step} at level {i} for index {index}"
                );
                assert_eq!(
                    step, prover_data.arity_schedule[i],
                    "step mismatch at level {i} for index {index}"
                );
            }

            // Positions must be consistent with index reduction at each level.
            let mut idx = index;
            for (level, (&step, &pos_in_group)) in steps.iter().zip(&positions).enumerate() {
                let expected_pos = idx % step;
                assert_eq!(
                    pos_in_group, expected_pos,
                    "pos_in_group mismatch at level {level} for index {index}"
                );
                idx /= step;
            }
        }
    }

    #[test]
    fn replay_arity_and_positions_4ary_mixed_heights() {
        let mut rng = SmallRng::seed_from_u64(789);
        let perm16 = Perm::new_from_rng_128(&mut rng);
        let hash = MyHash::new(perm16);
        let perm32 = PermWide::new_from_rng_128(&mut rng);
        let compress4 = MyCompress4::new(perm32);
        let mmcs4 = MyMmcs4::new(hash, compress4, 0);

        let mat64 = RowMajorMatrix::<F>::rand(&mut rng, 64, 8);
        let mat16 = RowMajorMatrix::<F>::rand(&mut rng, 16, 6);
        let mat4 = RowMajorMatrix::<F>::rand(&mut rng, 4, 5);
        let dims = vec![mat64.dimensions(), mat16.dimensions(), mat4.dimensions()];
        let (_commit, prover_data) = mmcs4.commit(vec![mat64, mat16, mat4]);

        for index in [0usize, 3, 7, 17, 42, 63] {
            let opening = mmcs4.open_batch(index, &prover_data);
            let (_opened_values, proof) = opening.unpack();

            let arity_and_positions = mmcs4
                .replay_arity_and_positions(&dims, index, proof.len())
                .expect("schedule replay should succeed");
            let steps = arity_and_positions.arity_schedule;
            let positions = arity_and_positions.query_positions;

            let expected_levels = prover_data.digest_layers.len().saturating_sub(1);
            assert_eq!(steps.len(), expected_levels);
            assert_eq!(positions.len(), expected_levels);

            for (i, &step) in steps.iter().enumerate() {
                assert_eq!(
                    step, prover_data.arity_schedule[i],
                    "step mismatch at level {i} for index {index}"
                );
            }

            let mut idx = index;
            for (level, (&step, &pos_in_group)) in steps.iter().zip(&positions).enumerate() {
                let expected_pos = idx % step;
                assert_eq!(
                    pos_in_group, expected_pos,
                    "pos_in_group mismatch at level {level} for index {index}"
                );
                idx /= step;
            }
        }
    }

    #[test]
    fn replay_arity_and_positions_4ary_non_power_of_two_target() {
        let mut rng = SmallRng::seed_from_u64(790);
        let perm16 = Perm::new_from_rng_128(&mut rng);
        let hash = MyHash::new(perm16);
        let perm32 = PermWide::new_from_rng_128(&mut rng);
        let compress4 = MyCompress4::new(perm32);
        let mmcs4 = MyMmcs4::new(hash, compress4, 0);

        // This shape forces an intermediate padded height of 12, where 12 / 4 = 3
        // is not a power of two. The prover and verifier must still pick the same
        // step schedule.
        let mat33 = RowMajorMatrix::<F>::rand(&mut rng, 33, 8);
        let mat3 = RowMajorMatrix::<F>::rand(&mut rng, 3, 5);
        let dims = vec![mat33.dimensions(), mat3.dimensions()];
        let (commit, prover_data) = mmcs4.commit(vec![mat33, mat3]);

        assert!(
            prover_data.arity_schedule.iter().all(|&step| step == 4),
            "expected full 4-ary schedule for this shape"
        );

        for index in [0usize, 1, 16, 32] {
            let opening = mmcs4.open_batch(index, &prover_data);
            let (opened_values, proof) = opening.unpack();

            mmcs4
                .verify_batch(
                    &commit,
                    &dims,
                    index,
                    BatchOpeningRef::new(&opened_values, &proof),
                )
                .expect("verification should succeed for non-power-of-two target case");

            let arity_and_positions = mmcs4
                .replay_arity_and_positions(&dims, index, proof.len())
                .expect("schedule replay should succeed");
            assert_eq!(
                arity_and_positions.arity_schedule, prover_data.arity_schedule,
                "replayed schedule must match prover schedule at index {index}"
            );
        }
    }

    mod proptests {
        use alloc::vec::Vec;

        use proptest::prelude::*;

        use super::*;

        type PermWide = Poseidon2BabyBear<32>;
        type MyCompress4 = TruncatedPermutation<PermWide, 4, 8, 32>;
        type MyMmcs4 =
            MerkleTreeMmcs<<F as Field>::Packing, <F as Field>::Packing, MyHash, MyCompress4, 4, 8>;

        fn make_binary_mmcs(seed: u64) -> MyMmcs {
            let mut rng = SmallRng::seed_from_u64(seed);
            let perm = Perm::new_from_rng_128(&mut rng);
            let hash = MyHash::new(perm.clone());
            let compress = MyCompress::new(perm);
            MyMmcs::new(hash, compress, 0)
        }

        fn make_4ary_mmcs(seed: u64) -> MyMmcs4 {
            let mut rng = SmallRng::seed_from_u64(seed);
            let perm16 = Perm::new_from_rng_128(&mut rng);
            let perm32 = PermWide::new_from_rng_128(&mut rng);
            let hash = MyHash::new(perm16);
            let compress = MyCompress4::new(perm32);
            MyMmcs4::new(hash, compress, 0)
        }

        fn matrix_strategy() -> impl Strategy<Value = (usize, usize, u64)> {
            (1..=256_usize, 1..=32_usize, 0u64..=u64::MAX)
        }

        proptest! {
            #[test]
            fn proptest_binary_merkle_roundtrip((height, width, seed) in matrix_strategy()) {
                let mut rng = SmallRng::seed_from_u64(seed);
                let mat = RowMajorMatrix::<F>::rand(&mut rng, height, width);
                let dims = vec![mat.dimensions()];
                let mmcs = make_binary_mmcs(seed.wrapping_add(1));

                let (commit, prover_data) = mmcs.commit(vec![mat.clone()]);

                let index = (seed as usize) % height;
                let opening = mmcs.open_batch(index, &prover_data);

                let (opened_values, proof) = opening.unpack();
                mmcs.verify_batch(&commit, &dims, index, BatchOpeningRef::new(&opened_values, &proof))
                    .expect("binary MerkleTreeMmcs verify should succeed");

                let expected_row: Vec<F> = mat.row(index).unwrap().into_iter().collect();
                prop_assert_eq!(&opened_values[0], &expected_row, "opened row should match");
            }

            #[test]
            fn proptest_4ary_merkle_roundtrip((height, width, seed) in matrix_strategy()) {
                let mut rng = SmallRng::seed_from_u64(seed);
                let mat = RowMajorMatrix::<F>::rand(&mut rng, height, width);
                let dims = vec![mat.dimensions()];
                let mmcs = make_4ary_mmcs(seed.wrapping_add(1));

                let (commit, prover_data) = mmcs.commit(vec![mat.clone()]);

                let index = (seed as usize) % height;
                let opening = mmcs.open_batch(index, &prover_data);

                let (opened_values, proof) = opening.unpack();
                mmcs.verify_batch(&commit, &dims, index, BatchOpeningRef::new(&opened_values, &proof))
                    .expect("4-ary MerkleTreeMmcs verify should succeed");

                let expected_row: Vec<F> = mat.row(index).unwrap().into_iter().collect();
                prop_assert_eq!(&opened_values[0], &expected_row, "opened row should match");
            }

            #[test]
            fn proptest_4ary_all_indices((height, seed) in (1..=64_usize, 0u64..=u64::MAX)) {
                let mut rng = SmallRng::seed_from_u64(seed);
                let mat = RowMajorMatrix::<F>::rand(&mut rng, height, 8);
                let dims = vec![mat.dimensions()];
                let mmcs = make_4ary_mmcs(seed.wrapping_add(1));
                let (commit, prover_data) = mmcs.commit(vec![mat.clone()]);

                for index in 0..height {
                    let opening = mmcs.open_batch(index, &prover_data);
                    let (opened_values, proof) = opening.unpack();
                    mmcs.verify_batch(&commit, &dims, index, BatchOpeningRef::new(&opened_values, &proof))
                        .expect("4-ary verify at each index should succeed");
                    let expected: Vec<F> = mat.row(index).unwrap().into_iter().collect();
                    prop_assert_eq!(&opened_values[0], &expected);
                }
            }
        }
    }

    // ----------------------------------------------------------------
    // Pruned opening integration tests
    // ----------------------------------------------------------------

    fn make_binary_mmcs(seed: u64) -> MyMmcs {
        let mut rng = SmallRng::seed_from_u64(seed);
        let perm = Perm::new_from_rng_128(&mut rng);
        let hash = MyHash::new(perm.clone());
        let compress = MyCompress::new(perm);
        MyMmcs::new(hash, compress, 0)
    }

    fn make_4ary_mmcs_pruning(seed: u64) -> MyMmcs4 {
        let mut rng = SmallRng::seed_from_u64(seed);
        let perm16 = Perm::new_from_rng_128(&mut rng);
        let perm32 = PermWide::new_from_rng_128(&mut rng);
        let hash = MyHash::new(perm16);
        let compress = MyCompress4::new(perm32);
        MyMmcs4::new(hash, compress, 0)
    }

    #[test]
    fn pruned_opening_matches_individual_binary() {
        let seed = 42u64;
        let mmcs = make_binary_mmcs(seed);

        let mut rng = SmallRng::seed_from_u64(seed);
        let mat = RowMajorMatrix::<F>::rand(&mut rng, 32, 4);
        let dims = vec![mat.dimensions()];
        let (commit, prover_data) = mmcs.commit(vec![mat]);

        let indices: Vec<usize> = vec![0, 5, 7, 12, 15, 20, 31];

        // Open individually.
        let individual_openings: Vec<_> = indices
            .iter()
            .map(|&i| mmcs.open_batch(i, &prover_data))
            .collect();

        // Open pruned.
        let pruned_opening = mmcs.open_batch_pruned(&indices, &prover_data);

        // Check opened values match.
        for (i, individual) in individual_openings.iter().enumerate() {
            assert_eq!(
                pruned_opening.opened_values[i], individual.opened_values,
                "opened values mismatch at query {i}"
            );
        }

        // Verify pruned opening (consumed by value — zero deep-clone).
        let result = mmcs.verify_batch_pruned(&commit, &dims, pruned_opening);
        assert!(result.is_ok(), "pruned verification should succeed");
    }

    #[test]
    fn pruned_opening_4ary_roundtrip() {
        let seed = 99u64;
        let mmcs = make_4ary_mmcs_pruning(seed);

        let mut rng = SmallRng::seed_from_u64(seed);
        let mat = RowMajorMatrix::<F>::rand(&mut rng, 64, 8);
        let dims = vec![mat.dimensions()];
        let (commit, prover_data) = mmcs.commit(vec![mat]);

        let indices: Vec<usize> = vec![0, 1, 10, 20, 30, 40, 50, 63];
        let pruned_opening = mmcs.open_batch_pruned(&indices, &prover_data);

        let result = mmcs.verify_batch_pruned(&commit, &dims, pruned_opening);
        assert!(result.is_ok(), "4-ary pruned verification should succeed");
    }

    #[test]
    fn pruned_opening_rejects_tampered_proof() {
        let seed = 77u64;
        let mmcs = make_binary_mmcs(seed);

        let mut rng = SmallRng::seed_from_u64(seed);
        let mat = RowMajorMatrix::<F>::rand(&mut rng, 16, 4);
        let dims = vec![mat.dimensions()];
        let (commit, prover_data) = mmcs.commit(vec![mat]);

        let indices: Vec<usize> = vec![0, 3, 7, 15];
        let mut pruned_opening = mmcs.open_batch_pruned(&indices, &prover_data);

        // Tamper with a sibling digest in the first path.
        if let Some(first_path) = pruned_opening.pruned_proof.paths.first_mut()
            && let Some(sibling) = first_path.siblings.first_mut()
        {
            sibling[0] = F::from_u32(999999);
        }

        let result = mmcs.verify_batch_pruned(&commit, &dims, pruned_opening);
        assert!(result.is_err(), "tampered proof should fail verification");
    }

    #[test]
    fn pruned_proof_is_smaller_than_individual() {
        let seed = 55u64;
        let mmcs = make_binary_mmcs(seed);

        let mut rng = SmallRng::seed_from_u64(seed);
        let mat = RowMajorMatrix::<F>::rand(&mut rng, 256, 4);
        let (_, prover_data) = mmcs.commit(vec![mat]);

        let indices: Vec<usize> = vec![0, 1, 2, 3, 10, 11, 100, 101, 200, 201, 254, 255];

        // Count total siblings in individual proofs.
        let individual_total: usize = indices
            .iter()
            .map(|&i| mmcs.open_batch(i, &prover_data).opening_proof.len())
            .sum();

        // Count total siblings in pruned proof.
        let pruned_opening = mmcs.open_batch_pruned(&indices, &prover_data);
        let pruned_total: usize = pruned_opening
            .pruned_proof
            .paths
            .iter()
            .map(|p| p.siblings.len())
            .sum();

        assert!(
            pruned_total < individual_total,
            "pruned {pruned_total} should be < individual {individual_total}"
        );
    }

    #[test]
    fn pruned_opening_with_duplicate_indices() {
        let seed = 33u64;
        let mmcs = make_binary_mmcs(seed);

        let mut rng = SmallRng::seed_from_u64(seed);
        let mat = RowMajorMatrix::<F>::rand(&mut rng, 16, 4);
        let dims = vec![mat.dimensions()];
        let (commit, prover_data) = mmcs.commit(vec![mat]);

        // Duplicate index 5.
        let indices: Vec<usize> = vec![5, 10, 5];
        let pruned_opening = mmcs.open_batch_pruned(&indices, &prover_data);

        // Should have 3 entries in opened_values (preserving duplicates).
        assert_eq!(pruned_opening.opened_values.len(), 3);
        assert_eq!(
            pruned_opening.opened_values[0], pruned_opening.opened_values[2],
            "duplicate queries should have identical values"
        );

        let result = mmcs.verify_batch_pruned(&commit, &dims, pruned_opening);
        assert!(result.is_ok());
    }

    #[test]
    fn pruned_opening_mixed_heights() {
        let seed = 44u64;
        let mmcs = make_binary_mmcs(seed);

        let mut rng = SmallRng::seed_from_u64(seed);
        let mat1 = RowMajorMatrix::<F>::rand(&mut rng, 32, 4);
        let mat2 = RowMajorMatrix::<F>::rand(&mut rng, 8, 6);
        let dims = vec![mat1.dimensions(), mat2.dimensions()];
        let (commit, prover_data) = mmcs.commit(vec![mat1, mat2]);

        let indices: Vec<usize> = vec![0, 7, 15, 31];
        let pruned_opening = mmcs.open_batch_pruned(&indices, &prover_data);

        let result = mmcs.verify_batch_pruned(&commit, &dims, pruned_opening);
        assert!(
            result.is_ok(),
            "mixed-height pruned verification should succeed"
        );
    }

    #[test]
    fn pruned_opening_4ary_mixed_heights() {
        // 4-ary MMCS with matrices at heights that don't align to the same
        // power of 4 — the schedule mixes 4-ary steps with binary bridges.
        let seed = 88u64;
        let mmcs = make_4ary_mmcs_pruning(seed);

        let mut rng = SmallRng::seed_from_u64(seed);
        let mat1 = RowMajorMatrix::<F>::rand(&mut rng, 64, 4);
        let mat2 = RowMajorMatrix::<F>::rand(&mut rng, 8, 6);
        let dims = vec![mat1.dimensions(), mat2.dimensions()];
        let (commit, prover_data) = mmcs.commit(vec![mat1, mat2]);

        let indices: Vec<usize> = vec![0, 5, 17, 33, 50, 63];
        let pruned_opening = mmcs.open_batch_pruned(&indices, &prover_data);

        let result = mmcs.verify_batch_pruned(&commit, &dims, pruned_opening);
        assert!(
            result.is_ok(),
            "4-ary mixed-height pruned verification should succeed"
        );
    }

    // Pruned-opening proptests

    proptest! {
        #[test]
        fn proptest_pruned_binary_roundtrip(
            height in 2..128usize,
            width in 1..16usize,
            seed in 0u64..=u64::MAX,
            num_queries in 1..20usize,
        ) {
            // Invariant: pruned opening must produce the same opened values
            // and pass verification identically to individual openings.
            //
            // Fixture state: random binary tree (height x width) committed
            // with a seeded Poseidon2 permutation.

            // Build a binary MMCS and commit a random matrix.
            let mmcs = make_binary_mmcs(seed);
            let mut rng = SmallRng::seed_from_u64(seed);
            let mat = RowMajorMatrix::<F>::rand(&mut rng, height, width);
            let dims = vec![mat.dimensions()];
            let (commit, prover_data) = mmcs.commit(vec![mat]);

            // Deterministic pseudo-random query indices (mod height).
            // Uses a large prime stride (7919) to spread queries across the tree.
            let indices: Vec<usize> = (0..num_queries)
                .map(|i| (seed as usize).wrapping_add(i * 7919) % height)
                .collect();

            // Reference: open each index individually (standard unpruned path).
            let individual: Vec<_> = indices
                .iter()
                .map(|&i| mmcs.open_batch(i, &prover_data))
                .collect();

            // Test subject: open all indices at once via pruning.
            let pruned_opening = mmcs.open_batch_pruned(&indices, &prover_data);

            // Check: opened values must be identical for every query.
            for (i, ind) in individual.iter().enumerate() {
                prop_assert_eq!(
                    &pruned_opening.opened_values[i],
                    &ind.opened_values,
                );
            }

            // Check: the pruned proof must verify against the commitment.
            mmcs.verify_batch_pruned(&commit, &dims, pruned_opening)
                .map_err(|e| TestCaseError::fail(format!("{e:?}")))?;
        }

        #[test]
        fn proptest_pruned_4ary_roundtrip(
            height in 2..128usize,
            width in 1..16usize,
            seed in 0u64..=u64::MAX,
            num_queries in 1..20usize,
        ) {
            // Same invariant as the binary test, but for 4-ary compression.
            // The arity schedule may include both 4-ary and binary bridge
            // steps depending on the matrix height.

            let mmcs = make_4ary_mmcs_pruning(seed);
            let mut rng = SmallRng::seed_from_u64(seed);
            let mat = RowMajorMatrix::<F>::rand(&mut rng, height, width);
            let dims = vec![mat.dimensions()];
            let (commit, prover_data) = mmcs.commit(vec![mat]);

            let indices: Vec<usize> = (0..num_queries)
                .map(|i| (seed as usize).wrapping_add(i * 7919) % height)
                .collect();

            // Reference: individual openings.
            let individual: Vec<_> = indices
                .iter()
                .map(|&i| mmcs.open_batch(i, &prover_data))
                .collect();

            // Test subject: pruned opening.
            let pruned_opening = mmcs.open_batch_pruned(&indices, &prover_data);

            // Check: identical opened values.
            for (i, ind) in individual.iter().enumerate() {
                prop_assert_eq!(
                    &pruned_opening.opened_values[i],
                    &ind.opened_values,
                );
            }

            // Check: pruned proof verifies.
            mmcs.verify_batch_pruned(&commit, &dims, pruned_opening)
                .map_err(|e| TestCaseError::fail(format!("{e:?}")))?;
        }

        #[test]
        fn proptest_pruned_proof_size_leq_individual(
            height in 2..256usize,
            seed in 0u64..=u64::MAX,
            num_queries in 2..30usize,
        ) {
            // Invariant: the pruned proof must never contain more sibling
            // digests than the sum of all individual proofs.
            //
            // This should hold for any query pattern — clustered queries
            // save more, spread queries save less, but never go negative.

            let mmcs = make_binary_mmcs(seed);
            let mut rng = SmallRng::seed_from_u64(seed);
            let mat = RowMajorMatrix::<F>::rand(&mut rng, height, 4);
            let (_, prover_data) = mmcs.commit(vec![mat]);

            let indices: Vec<usize> = (0..num_queries)
                .map(|i| (seed as usize).wrapping_add(i * 7919) % height)
                .collect();

            // Sum of individual proof sizes (unpruned baseline).
            let individual_total: usize = indices
                .iter()
                .map(|&i| mmcs.open_batch(i, &prover_data).opening_proof.len())
                .sum();

            // Pruned proof total.
            let pruned_opening = mmcs.open_batch_pruned(&indices, &prover_data);
            let pruned_total: usize = pruned_opening
                .pruned_proof
                .paths
                .iter()
                .map(|p| p.siblings.len())
                .sum();

            prop_assert!(
                pruned_total <= individual_total,
                "pruned {} > individual {}", pruned_total, individual_total
            );
        }
    }
}
