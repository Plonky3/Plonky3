use alloc::vec;
use alloc::vec::Vec;
use core::array;
use core::cmp::Reverse;
use core::marker::PhantomData;

use itertools::Itertools;
use p3_field::PackedValue;
use p3_matrix::Matrix;
use p3_maybe_rayon::prelude::*;
use p3_symmetric::{CryptographicHasher, Hash, MerkleCap, PseudoCompressionFunction};
use serde::{Deserialize, Serialize};
use tracing::instrument;

/// An N-ary Merkle tree whose leaves are vectors of matrix rows.
///
/// * `F` – scalar element type inside each matrix row.
/// * `W` – scalar element type of every digest word.
/// * `M` – matrix type. Must implement [`Matrix<F>`].
/// * `N` – arity of the compression function.
/// * `DIGEST_ELEMS` – number of `W` words in one digest.
///
/// The tree is **balanced only at the digest layer**.
/// Leaf matrices may have arbitrary heights, but every height must sit on the
/// `ceil(max_height / 2^k)` ladder anchored at the tallest matrix — the same
/// requirement `Mmcs::commit` enforces before building a tree.
///
/// Use [`Self::root`] to fetch the final digest once the tree is built.
///
/// This generally shouldn't be used directly. If you're using a Merkle tree as an MMCS,
/// see `MerkleTreeMmcs`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MerkleTree<F, W, M, const N: usize, const DIGEST_ELEMS: usize> {
    /// All leaf matrices in insertion order.
    ///
    /// Each matrix contributes rows to one or more digest layers, depending on its height.
    /// Specifically, only the tallest matrices are included in the first digest layer,
    /// while shorter matrices are injected into higher digest layers at positions determined
    /// by their padded heights.
    ///
    /// This vector is retained only for inspection or re-opening of the tree; it is not used
    /// after construction time.
    pub(crate) leaves: Vec<M>,

    /// All intermediate digest layers, index 0 being the first layer above
    /// the leaves and the last layer containing exactly one root digest.
    ///
    /// Every inner vector holds contiguous digests `[left₀, right₀, left₁,
    /// right₁, …]`; higher layers refer to these by index.
    ///
    /// Serialization requires that `[W; DIGEST_ELEMS]` implements `Serialize` and
    /// `Deserialize`. This is automatically satisfied when `W` is a fixed-size type.
    #[serde(
        bound(serialize = "[W; DIGEST_ELEMS]: Serialize"),
        bound(deserialize = "[W; DIGEST_ELEMS]: Deserialize<'de>")
    )]
    pub(crate) digest_layers: Vec<Vec<[W; DIGEST_ELEMS]>>,

    /// The compression arity used at each tree level (transition from
    /// `digest_layers[i]` to `digest_layers[i+1]`).
    ///
    /// Each entry is either `N` (full N-ary step) as specified by the `N`
    /// parameter associated to the compression function, or `2` (binary step)
    /// when a matrix injection falls between N-ary levels.
    pub(crate) arity_schedule: Vec<usize>,

    /// Zero-sized marker that binds the generic `F` but occupies no space.
    _phantom: PhantomData<F>,
}

impl<F: Clone + Send + Sync, W: Clone, M: Matrix<F>, const N: usize, const DIGEST_ELEMS: usize>
    MerkleTree<F, W, M, N, DIGEST_ELEMS>
{
    /// Build a tree from **one or more matrices**.
    ///
    /// * `h` – hashing function used on raw rows.
    /// * `c` – N-to-1 compression function used on digests.
    /// * `leaves` – matrices to commit to. Must be non-empty.
    ///
    /// Matrices do **not** need to have power-of-two heights. However, every height must sit
    /// on the `ceil(max_height / 2^k)` ladder anchored at the tallest matrix — i.e. at `k`
    /// halvings above the leaves, the only admissible height is `ceil(max_height / 2^k)`. This
    /// ensures proper balancing when folding digests layer-by-layer, and that every global leaf
    /// index maps to a row in every committed matrix.
    ///
    /// All matrices are hashed row-by-row with `h`. The resulting digests are
    /// then folded upwards with `c` until a single root remains.
    ///
    /// # Panics
    /// * If `leaves` is empty, or every leaf has height 0.
    /// * If the packing widths of `P` and `PW` differ.
    /// * If any leaf height is off the `ceil(max_height / 2^k)` ladder.
    #[instrument(name = "build merkle tree", level = "debug", skip_all,
                 fields(dimensions = alloc::format!("{:?}", leaves.iter().map(|l| l.dimensions()).collect::<Vec<_>>())))]
    pub fn new<P, PW, H, C>(h: &H, c: &C, leaves: Vec<M>) -> Self
    where
        P: PackedValue<Value = F>,
        PW: PackedValue<Value = W>,
        H: CryptographicHasher<F, [W; DIGEST_ELEMS]>
            + CryptographicHasher<P, [PW; DIGEST_ELEMS]>
            + Sync,
        C: PseudoCompressionFunction<[W; DIGEST_ELEMS], N>
            + PseudoCompressionFunction<[PW; DIGEST_ELEMS], N>
            + Sync,
    {
        assert!(!leaves.is_empty(), "No matrices given?");
        const {
            assert!(N >= 2, "Arity N must be at least 2");
            assert!(N.is_power_of_two(), "Arity N must be a power of two");
            assert!(P::WIDTH == PW::WIDTH, "Packing widths must match");
        }

        // Geometry gate: every height must sit on the `ceil(max_height / 2^k)`
        // ladder anchored at the tallest matrix, or no tree can be built from
        // them. `Mmcs::commit` enforces the same gate; this constructor is
        // public, so it must enforce it too rather than fail later with an
        // out-of-bounds panic deep inside layer construction.
        if let Err(err) =
            crate::mmcs::validate_commit_reachable_heights(leaves.iter().map(|l| l.height()))
        {
            panic!("{err}");
        }

        let mut leaves_largest_first = leaves
            .iter()
            .sorted_by_key(|l| Reverse(l.height()))
            .peekable();

        let max_height = leaves_largest_first.peek().unwrap().height();
        let leaf_height_npt = max_height.next_power_of_two();
        let tallest_matrices = leaves_largest_first
            .peeking_take_while(|m| m.height() == max_height)
            .collect_vec();

        let mut digest_layers = vec![first_digest_layer::<P, _, _, _, N, DIGEST_ELEMS>(
            h,
            &tallest_matrices,
        )];
        let mut arity_schedule = Vec::new();

        loop {
            let prev_layer = digest_layers.last().unwrap().as_slice();
            if prev_layer.len() <= 1 {
                break;
            }

            // Decide whether this level is a full N-ary step or a binary step.
            let step = select_arity_step::<N>(
                prev_layer.len(),
                leaf_height_npt,
                leaves_largest_first.clone().map(|m| m.height()),
            );

            let next_layer_len = (prev_layer.len() / step).next_power_of_two();

            // The matrices that get injected at this layer.
            let matrices_to_inject = leaves_largest_first
                .peeking_take_while(|m| m.height().next_power_of_two() == next_layer_len)
                .collect_vec();

            let next_digests = compress_and_inject::<P, _, _, _, _, N, DIGEST_ELEMS>(
                prev_layer,
                step,
                &matrices_to_inject,
                h,
                c,
            );
            arity_schedule.push(step);
            digest_layers.push(next_digests);
        }

        Self {
            leaves,
            digest_layers,
            arity_schedule,
            _phantom: PhantomData,
        }
    }

    /// Return the root digest of the tree.
    #[must_use]
    pub fn root(&self) -> Hash<F, W, DIGEST_ELEMS>
    where
        W: Copy,
    {
        self.digest_layers.last().unwrap()[0].into()
    }

    /// Return the Merkle cap at the specified height from the root.
    ///
    /// A cap height of 0 returns just the root (1 element).
    /// A cap height of h returns `product(arity_schedule[layer_idx..])` elements,
    /// where each arity is either N or 2 depending on the tree layout.
    ///
    /// # Panics
    /// Panics if `cap_height` exceeds the tree depth.
    #[must_use]
    pub fn cap(&self, cap_height: usize) -> MerkleCap<F, [W; DIGEST_ELEMS]>
    where
        W: Clone,
    {
        let num_layers = self.digest_layers.len();
        assert!(
            cap_height < num_layers,
            "cap_height {} exceeds tree depth {}",
            cap_height,
            num_layers
        );

        let layer_idx = num_layers - 1 - cap_height;
        let layer = &self.digest_layers[layer_idx];

        let cap_len: usize = self.arity_schedule[layer_idx..].iter().product();
        let cap_len = cap_len.min(layer.len());

        MerkleCap::new(layer[..cap_len].to_vec())
    }

    #[must_use]
    pub const fn num_layers(&self) -> usize {
        self.digest_layers.len()
    }
}

/// Select the compression arity for the current layer.
///
/// Returns `N` for a full N-ary step, or `2` for a binary bridge step when a
/// matrix injection must happen before the next N-ary target level.
pub(crate) fn select_arity_step<const N: usize>(
    curr_height_padded: usize,
    leaf_height_npt: usize,
    remaining_heights_tallest_first: impl Iterator<Item = usize>,
) -> usize {
    if curr_height_padded < N {
        return 2;
    }

    let n_ary_target = (curr_height_padded / N).next_power_of_two();
    let has_intermediate = remaining_heights_tallest_first
        .filter(|height| height.next_power_of_two() != leaf_height_npt)
        .any(|height| height.next_power_of_two() > n_ary_target);

    if has_intermediate { 2 } else { N }
}

/// Allocate a digest layer of `len` slots, every slot set to the default digest.
///
/// Every slot of a layer is overwritten by the level that produces it, apart from a short
/// padding tail, so the fill is pure overhead and worth getting for free from the allocator.
///
/// A zero-filled allocation is free: the operating system hands back pages that are already
/// zero and only faults them in when the hashing threads write them.
///
/// The standard vector constructor reaches that path only when it can see at run time that the
/// element is all zero bits.
///
/// It gives up on that check for arrays longer than sixteen elements, which is exactly the
/// shape of a thirty-two-byte digest.
///
/// Building the layer as one flat run of digest words restores the check, because a word is a
/// primitive the constructor still inspects.
///
/// The flat run and the run of digests have the very same allocation layout, so viewing one as
/// the other costs nothing:
///
/// ```text
///     flat:    [ w_0 w_1 ... w_{D-1} | w_D ... w_{2D-1} | ... ]   len * D words
///     digests: [       digest_0      |     digest_1     | ... ]   len digests
/// ```
///
/// # Panics
///
/// Panics if the layer is too large for the address space.
fn default_digest_layer<W, const DIGEST_ELEMS: usize>(len: usize) -> Vec<[W; DIGEST_ELEMS]>
where
    W: Copy + Default,
{
    // A layer with no slots, or a digest of no words, owns no allocation worth reinterpreting.
    if len == 0 || DIGEST_ELEMS == 0 {
        return vec![[W::default(); DIGEST_ELEMS]; len];
    }

    // Guard the multiply so a wrapped word count can never under-allocate the layer.
    let words = len
        .checked_mul(DIGEST_ELEMS)
        .expect("digest layer length overflows");

    let mut flat: Vec<W> = vec![W::default(); words];

    // A vector may reserve more than requested, and only an exact allocation converts.
    if flat.capacity() != words {
        return vec![[W::default(); DIGEST_ELEMS]; len];
    }

    let ptr = flat.as_mut_ptr().cast::<[W; DIGEST_ELEMS]>();
    core::mem::forget(flat);

    // SAFETY: an array of `DIGEST_ELEMS` words has no padding, so `len` digests occupy exactly
    // the `len * DIGEST_ELEMS` words allocated above, with the same alignment as one word.
    //
    // The requested length equals the reserved capacity, so the layout handed back to the
    // allocator on drop is byte for byte the layout it handed out.
    //
    // Every word is initialized, so every digest slot reads as the default digest.
    unsafe { Vec::from_raw_parts(ptr, len, len) }
}

/// Output nodes below which a level is hashed on the calling thread.
///
/// A parallel dispatch costs a fixed amount per level, and the levels near the root hold so few
/// nodes that the dispatch outweighs the hashing itself.
///
/// Grouping messages is independent of threading, so a small level keeps the full vector width
/// even while it runs on one thread.
///
/// 1024 is the point where the two costs balance in practice.
///
/// It still leaves tens of nodes per thread on a many-core machine, and every wider level fans
/// out.
const SERIAL_LEVEL_NODES: usize = 1024;

/// Output nodes handed to one parallel task.
///
/// Matching the serial threshold keeps a task large enough to amortize the dispatch and small
/// enough that its scratch buffers stay in the mid-level cache.
const TASK_NODES: usize = 1024;

/// Target size in bytes of the buffer holding one group of copied rows.
///
/// A batched hasher wants its messages back to back in memory, and a matrix only promises access
/// one row at a time, so a group of rows is copied into a buffer first.
///
/// 16 KiB keeps that buffer inside the first-level cache, so the hasher reads the rows back
/// while they are still hot.
const ROW_SCRATCH_BYTES: usize = 16 * 1024;

/// Hash a run of rows from a set of equal-height matrices, several messages per hash call.
///
/// The message for a row is the concatenation of that row across every matrix, in matrix order,
/// which is exactly what the unbatched path feeds its hasher one row at a time.
///
/// # Arguments
///
/// - `h`: hasher applied to each row message.
/// - `matrices`: matrices of equal height whose rows are concatenated.
///
/// - `first_row`: index of the row whose digest lands in the first output slot.
/// - `out`: one slot per consecutive row starting at that index.
///
/// # Panics
///
/// Panics if any matrix is shorter than the requested row range.
fn hash_rows_batched<F, W, H, M, const DIGEST_ELEMS: usize>(
    h: &H,
    matrices: &[&M],
    first_row: usize,
    out: &mut [[W; DIGEST_ELEMS]],
) where
    F: Clone + Send + Sync,
    W: Send + Sync,
    H: CryptographicHasher<F, [W; DIGEST_ELEMS]> + Sync,
    M: Matrix<F>,
{
    // Bound the row range up front, since the copy loop below skips per-row bounds checks.
    assert!(
        matrices.iter().all(|m| m.height() >= first_row + out.len()),
        "row range {}..{} exceeds a matrix height",
        first_row,
        first_row + out.len()
    );

    // A message spans one row of every matrix.
    let total_width: usize = matrices.iter().map(|m| m.width()).sum();
    let row_bytes = (total_width * size_of::<F>()).max(1);

    // Fill as many whole lane groups as the scratch budget allows, and never fewer than one,
    // so every hash call keeps the hasher's vector width busy.
    let lanes = H::LANES;
    let rows_per_call = (ROW_SCRATCH_BYTES / row_bytes / lanes).max(1) * lanes;

    let hash_chunk = |base: usize, digests: &mut [[W; DIGEST_ELEMS]]| {
        // One buffer per task, reused by every group inside it.
        let mut scratch: Vec<F> = Vec::with_capacity(rows_per_call * total_width);

        for (group, group_digests) in digests.chunks_mut(rows_per_call).enumerate() {
            let first = base + group * rows_per_call;

            // Lay the group's messages back to back: row by row, matrix by matrix.
            scratch.clear();
            for row in first..first + group_digests.len() {
                for m in matrices {
                    // SAFETY: the assertion above bounds every requested row by every height.
                    let slice = unsafe { m.row_slice_unchecked(row) };
                    scratch.extend_from_slice(&slice);
                }
            }

            h.hash_many(&scratch, group_digests);
        }
    };

    if out.len() <= SERIAL_LEVEL_NODES {
        // Small level: one task would be shared by nobody, so skip the dispatch.
        hash_chunk(first_row, out);
    } else {
        // Keep each task a whole number of groups so only the final group is ever short.
        let task = rows_per_call * TASK_NODES.div_ceil(rows_per_call);
        out.par_chunks_mut(task)
            .enumerate()
            .for_each(|(task_index, digests)| hash_chunk(first_row + task_index * task, digests));
    }
}

/// Compress a run of already-grouped children, several groups per compression call.
///
/// # Arguments
///
/// - `c`: compression function applied to each group.
/// - `groups`: one array of `N` children per output node.
///
/// - `out`: one slot per group.
fn compress_groups_batched<T, C, const N: usize>(c: &C, groups: &[[T; N]], out: &mut [T])
where
    T: Clone + Send + Sync,
    C: PseudoCompressionFunction<T, N> + Sync,
{
    if out.len() <= SERIAL_LEVEL_NODES {
        // Small level: one task would be shared by nobody, so skip the dispatch.
        c.compress_many(groups, out);
    } else {
        groups
            .par_chunks(TASK_NODES)
            .zip(out.par_chunks_mut(TASK_NODES))
            .for_each(|(group_chunk, digest_chunk)| c.compress_many(group_chunk, digest_chunk));
    }
}

/// Build the first digest layer with a hasher that hashes several messages per call.
fn first_digest_layer_batched<F, W, H, M, const N: usize, const DIGEST_ELEMS: usize>(
    h: &H,
    tallest_matrices: &[&M],
) -> Vec<[W; DIGEST_ELEMS]>
where
    F: Clone + Send + Sync,
    W: Copy + Default + Send + Sync,
    H: CryptographicHasher<F, [W; DIGEST_ELEMS]> + Sync,
    M: Matrix<F>,
{
    // All of the tallest matrices share one height by construction.
    let max_height = tallest_matrices[0].height();
    let max_height_padded = padded_len(max_height, N);

    // Slots past the real rows exist only so the next level can form whole groups, and the
    // allocation already leaves them at the default digest.
    let mut digests = default_digest_layer::<W, DIGEST_ELEMS>(max_height_padded);

    hash_rows_batched(h, tallest_matrices, 0, &mut digests[..max_height]);

    digests
}

/// Fold one digest layer into the next with a compression function that compresses several
/// groups per call.
///
/// A node's children sit next to each other in the layer below, so a run of `N` children is
/// already the contiguous preimage of one parent and a run of parents is a contiguous run of
/// those preimages:
///
/// ```text
///     prev_layer: [ c_0 c_1 | c_2 c_3 | c_4 c_5 | ... ]     N = 2
///                   \-----/   \-----/   \-----/
///     out:            d_0       d_1       d_2       ...
/// ```
///
/// Reading the layer as groups is therefore a reinterpretation of the same memory, with no
/// transposition and no gathering.
fn compress_batched<W, C, const N: usize, const DIGEST_ELEMS: usize>(
    prev_layer: &[[W; DIGEST_ELEMS]],
    c: &C,
) -> Vec<[W; DIGEST_ELEMS]>
where
    W: Copy + Default + Send + Sync,
    C: PseudoCompressionFunction<[W; DIGEST_ELEMS], N> + Sync,
{
    let next_len = prev_layer.len() / N;
    let next_len_padded = padded_len(next_len, N);

    let mut next_digests = default_digest_layer::<W, DIGEST_ELEMS>(next_len_padded);

    // Any trailing child that cannot complete a group takes no part in this level.
    let (groups, _) = prev_layer[..next_len * N].as_chunks::<N>();
    compress_groups_batched(c, groups, &mut next_digests[..next_len]);

    next_digests
}

/// Fold one digest layer into the next and mix in rows of smaller matrices, batching both the
/// compressions and the row hashes.
///
/// Each output node is built in three passes over the same chunk, so the intermediate buffers
/// stay proportional to a task rather than to the whole level:
///
/// ```text
///     pass 1: compress N children            -> folded
///     pass 2: hash one row per output node    -> injected
///     pass 3: compress [folded, injected]    -> out
/// ```
///
/// Output nodes past the injected matrices' height get the default digest in place of a row
/// digest, matching the unbatched path.
fn compress_and_inject_batched<F, W, H, C, M, const N: usize, const DIGEST_ELEMS: usize>(
    prev_layer: &[[W; DIGEST_ELEMS]],
    matrices_to_inject: &[&M],
    h: &H,
    c: &C,
) -> Vec<[W; DIGEST_ELEMS]>
where
    F: Clone + Send + Sync,
    W: Copy + Default + Send + Sync,
    H: CryptographicHasher<F, [W; DIGEST_ELEMS]> + Sync,
    C: PseudoCompressionFunction<[W; DIGEST_ELEMS], N> + Sync,
    M: Matrix<F>,
{
    // Rows are injected for the leading nodes only.
    // The nodes past the injected matrices' height are pure compressions.
    let inject_len = matrices_to_inject[0].height();
    let raw_next = prev_layer.len() / N;
    let next_len_padded = padded_len(raw_next, N);

    let default_digest = [W::default(); DIGEST_ELEMS];
    let mut next_digests = default_digest_layer::<W, DIGEST_ELEMS>(next_len_padded);

    let (groups, _) = prev_layer[..raw_next * N].as_chunks::<N>();

    let build = |base: usize,
                 group_chunk: &[[[W; DIGEST_ELEMS]; N]],
                 digest_chunk: &mut [[W; DIGEST_ELEMS]]| {
        let count = digest_chunk.len();

        // Pass 1: fold the children of every node in this chunk.
        let mut folded = vec![default_digest; count];
        c.compress_many(group_chunk, &mut folded);

        // Pass 2: hash one row per node, for as many nodes as the matrices are tall.
        let injected_count = count.min(inject_len.saturating_sub(base));
        let mut injected = vec![default_digest; injected_count];
        hash_rows_batched(h, matrices_to_inject, base, &mut injected);

        // Pass 3: pair each folded digest with its row digest, padding the group to `N`.
        let mut pairs = vec![[default_digest; N]; count];
        for (node, pair) in pairs.iter_mut().enumerate() {
            *pair = array::from_fn(|slot| match slot {
                0 => folded[node],
                1 if node < injected_count => injected[node],
                _ => default_digest,
            });
        }
        c.compress_many(&pairs, digest_chunk);
    };

    if raw_next <= SERIAL_LEVEL_NODES {
        // Small level: one task would be shared by nobody, so skip the dispatch.
        build(0, groups, &mut next_digests[..raw_next]);
    } else {
        groups
            .par_chunks(TASK_NODES)
            .zip(next_digests[..raw_next].par_chunks_mut(TASK_NODES))
            .enumerate()
            .for_each(|(task_index, (group_chunk, digest_chunk))| {
                build(task_index * TASK_NODES, group_chunk, digest_chunk);
            });
    }

    next_digests
}

/// Hash every row of the tallest matrices and build the first digest layer.
///
/// This function is responsible for creating the first layer of Merkle digests,
/// starting from raw rows of the tallest matrices. Each row is hashed using the
/// provided cryptographic hasher `h`. The result is a vector of digests that serve
/// as the base (leaf-level) nodes for the rest of the Merkle tree.
///
/// # Details
/// - We always return an *even number of digests* (except when height is 1), to
///   ensure even pairing at higher layers.
/// - Matrices are "vertically packed" to allow SIMD-friendly parallel hashing,
///   meaning rows can be processed in batches.
/// - If the total number of rows isn't a multiple of the SIMD packing width,
///   the final few rows are handled using a fallback scalar path.
///
/// # Arguments
/// - `h`: Reference to the cryptographic hasher.
/// - `tallest_matrices`: References to the tallest matrices (all must have same height).
///
/// # Returns
/// A vector of `[PW::Value; DIGEST_ELEMS]`, containing the digests of each row.
#[instrument(name = "first digest layer", level = "debug", skip_all)]
fn first_digest_layer<P, PW, H, M, const N: usize, const DIGEST_ELEMS: usize>(
    h: &H,
    tallest_matrices: &[&M],
) -> Vec<[PW::Value; DIGEST_ELEMS]>
where
    P: PackedValue,
    PW: PackedValue,
    H: CryptographicHasher<P::Value, [PW::Value; DIGEST_ELEMS]>
        + CryptographicHasher<P, [PW; DIGEST_ELEMS]>
        + Sync,
    M: Matrix<P::Value>,
{
    // A hasher that hashes several messages per call wants whole rows, not packed columns,
    // so it takes a separate driver.
    if <H as CryptographicHasher<P::Value, [PW::Value; DIGEST_ELEMS]>>::LANES > 1 {
        return first_digest_layer_batched::<P::Value, PW::Value, H, M, N, DIGEST_ELEMS>(
            h,
            tallest_matrices,
        );
    }

    // The number of rows to pack and hash together in one SIMD batch.
    let width = PW::WIDTH;

    // Get the height of the tallest matrices (they are guaranteed to be equal).
    let max_height = tallest_matrices[0].height();

    let max_height_padded = padded_len(max_height, N);

    // Allocate the digest vector with padded size, every slot at the default digest so the
    // padding tail past the real rows is already correct.
    let mut digests = default_digest_layer::<PW::Value, DIGEST_ELEMS>(max_height_padded);

    // Parallel loop: process complete batches of `width` rows at a time.
    digests[0..max_height]
        .par_chunks_exact_mut(width)
        .enumerate()
        .for_each(|(i, digests_chunk)| {
            // Compute the starting row index for this chunk.
            let first_row = i * width;

            // Collect all vertically packed rows from each matrix at `first_row`.
            // These packed rows are then hashed together using `h`.
            //
            // The single-matrix case feeds `h` the row iterator directly: going
            // through `flat_map` hands the hasher a compound iterator whose
            // `next()` defeats the optimizer's vectorization of the absorb loop,
            // which is worth ~40% of the leaf-hashing time on wide matrices.
            let packed_digest: [PW; DIGEST_ELEMS] = if let [m] = tallest_matrices {
                h.hash_iter(m.vertically_packed_row(first_row))
            } else {
                h.hash_iter(
                    tallest_matrices
                        .iter()
                        .flat_map(|m| m.vertically_packed_row(first_row)),
                )
            };

            // Unpack the resulting packed digest into individual scalar digests.
            PW::unpack_into(&packed_digest, digests_chunk);
        });

    // Handle leftover rows that do not form a full SIMD batch (if any).
    // `digests` is padded to `max_height_padded`, so cap the slice at `max_height`
    // to leave the padding tail untouched.
    let leftover_start = (max_height / width) * width;
    for (offset, digest) in digests[leftover_start..max_height].iter_mut().enumerate() {
        let i = leftover_start + offset;
        unsafe {
            // Safety: i < max_height == matrix height.
            // Use `row_unchecked` to avoid bounds checks for performance.
            *digest = h.hash_iter(tallest_matrices.iter().flat_map(|m| m.row_unchecked(i)));
        }
    }

    // Return the final digest vector (now fully populated).
    digests
}

/// Fold one digest layer into the next and, when present, mix in rows
/// taken from smaller matrices.
///
/// `step` is the grouping size for this level (either `N` for a full N-ary
/// step or `2` for a binary step when a matrix injection falls between
/// N-ary layers). Groups of `step` children are taken from `prev_layer`,
/// padded to `N` inputs with the default digest, then compressed with the
/// N-to-1 compression function.
fn compress_and_inject<P, PW, H, C, M, const N: usize, const DIGEST_ELEMS: usize>(
    prev_layer: &[[PW::Value; DIGEST_ELEMS]],
    step: usize,
    matrices_to_inject: &[&M],
    h: &H,
    c: &C,
) -> Vec<[PW::Value; DIGEST_ELEMS]>
where
    P: PackedValue,
    PW: PackedValue,
    H: CryptographicHasher<P::Value, [PW::Value; DIGEST_ELEMS]>
        + CryptographicHasher<P, [PW; DIGEST_ELEMS]>
        + Sync,
    C: PseudoCompressionFunction<[PW::Value; DIGEST_ELEMS], N>
        + PseudoCompressionFunction<[PW; DIGEST_ELEMS], N>
        + Sync,
    M: Matrix<P::Value>,
{
    if matrices_to_inject.is_empty() {
        return compress::<PW, _, N, DIGEST_ELEMS>(prev_layer, step, c);
    }

    // The batched arm reads a node's children as one contiguous group, which only lines up
    // when the level takes a full `N`-ary step.
    // A binary bridge step keeps the packed arm.
    if step == N
        && <H as CryptographicHasher<P::Value, [PW::Value; DIGEST_ELEMS]>>::LANES > 1
        && <C as PseudoCompressionFunction<[PW::Value; DIGEST_ELEMS], N>>::LANES > 1
    {
        return compress_and_inject_batched::<P::Value, PW::Value, H, C, M, N, DIGEST_ELEMS>(
            prev_layer,
            matrices_to_inject,
            h,
            c,
        );
    }

    let width = PW::WIDTH;
    let next_len = matrices_to_inject[0].height();
    let raw_next = prev_layer.len() / step;
    let next_len_padded = padded_len(raw_next, N);

    let default_digest = [PW::Value::default(); DIGEST_ELEMS];
    let mut next_digests = default_digest_layer::<PW::Value, DIGEST_ELEMS>(next_len_padded);

    let default_packed: [PW; DIGEST_ELEMS] =
        array::from_fn(|_| PW::broadcast(PW::Value::default()));

    next_digests[0..next_len]
        .par_chunks_exact_mut(width)
        .enumerate()
        .for_each(|(i, digests_chunk)| {
            let first_row = i * width;
            let children: [[PW; DIGEST_ELEMS]; N] = array::from_fn(|n| {
                if n < step {
                    PW::pack_columns_fn(|lane| prev_layer[step * (first_row + lane) + n])
                } else {
                    default_packed
                }
            });
            let mut packed_digest = c.compress(children);

            // As in `first_digest_layer`, the single-matrix case feeds `h` the
            // row iterator directly: a `flat_map` compound iterator defeats the
            // optimizer's vectorization of the absorb loop.
            let tallest_digest: [PW; DIGEST_ELEMS] = if let [m] = matrices_to_inject {
                h.hash_iter(m.vertically_packed_row(first_row))
            } else {
                h.hash_iter(
                    matrices_to_inject
                        .iter()
                        .flat_map(|m| m.vertically_packed_row(first_row)),
                )
            };
            let inject_inputs: [[PW; DIGEST_ELEMS]; N] = array::from_fn(|n| {
                if n == 0 {
                    packed_digest
                } else if n == 1 {
                    tallest_digest
                } else {
                    default_packed
                }
            });
            packed_digest = c.compress(inject_inputs);
            PW::unpack_into(&packed_digest, digests_chunk);
        });

    for i in (next_len / width * width)..next_len {
        let children: [_; N] = array::from_fn(|n| {
            if n < step {
                prev_layer[step * i + n]
            } else {
                default_digest
            }
        });
        let digest = c.compress(children);
        let rows_digest = unsafe {
            // Safety: i < next_len == matrices_to_inject height.
            h.hash_iter(matrices_to_inject.iter().flat_map(|m| m.row_unchecked(i)))
        };
        let inject_inputs: [_; N] = array::from_fn(|n| {
            if n == 0 {
                digest
            } else if n == 1 {
                rows_digest
            } else {
                default_digest
            }
        });
        next_digests[i] = c.compress(inject_inputs);
    }

    for i in next_len..raw_next {
        let children: [_; N] = array::from_fn(|n| {
            if n < step {
                prev_layer[step * i + n]
            } else {
                default_digest
            }
        });
        let digest = c.compress(children);
        let inject_inputs: [_; N] =
            array::from_fn(|n| if n == 0 { digest } else { default_digest });
        next_digests[i] = c.compress(inject_inputs);
    }

    next_digests
}

/// Compute the padded output length for a compression step.
///
/// The output layer must be large enough for the *next* compression step
/// to form complete groups. There are three cases:
///
/// - `raw_len <= 1`: this is the root, no padding needed.
/// - `raw_len >= n`: pad up to the next multiple of `n`.
/// - `1 < raw_len < n`: pad to exactly `n` so that the next step can do a
///   single full N-to-1 compression to produce the root. This is safe
///   because the extra slots are filled with the default digest — the same
///   value that `compress` would use as padding internally.
pub(crate) const fn padded_len(raw_len: usize, n: usize) -> usize {
    if raw_len <= 1 {
        raw_len
    } else if raw_len >= n {
        raw_len.div_ceil(n) * n
    } else {
        n
    }
}

/// Pure compression step used when no extra rows are injected.
///
/// Takes groups of digests from `prev_layer`, feeds them to `c`,
/// and writes the results in order.
///
/// Groups `step` consecutive digests from `prev_layer`, pads each group
/// to `N` inputs with the default digest, then compresses N-to-1.
fn compress<P, C, const N: usize, const DIGEST_ELEMS: usize>(
    prev_layer: &[[P::Value; DIGEST_ELEMS]],
    step: usize,
    c: &C,
) -> Vec<[P::Value; DIGEST_ELEMS]>
where
    P: PackedValue,
    C: PseudoCompressionFunction<[P::Value; DIGEST_ELEMS], N>
        + PseudoCompressionFunction<[P; DIGEST_ELEMS], N>
        + Sync,
{
    // Same reinterpretation rule as the injecting level.
    // A full `N`-ary step makes a node's children one contiguous group, a binary bridge step
    // does not.
    if step == N && <C as PseudoCompressionFunction<[P::Value; DIGEST_ELEMS], N>>::LANES > 1 {
        return compress_batched::<P::Value, C, N, DIGEST_ELEMS>(prev_layer, c);
    }

    let width = P::WIDTH;
    let next_len = prev_layer.len() / step;
    let next_len_padded = padded_len(next_len, N);

    let default_digest = [P::Value::default(); DIGEST_ELEMS];
    let mut next_digests = default_digest_layer::<P::Value, DIGEST_ELEMS>(next_len_padded);

    let default_packed: [P; DIGEST_ELEMS] = array::from_fn(|_| P::broadcast(P::Value::default()));

    next_digests[0..next_len]
        .par_chunks_exact_mut(width)
        .enumerate()
        .for_each(|(i, digests_chunk)| {
            let first_row = i * width;
            let children: [[P; DIGEST_ELEMS]; N] = array::from_fn(|n| {
                if n < step {
                    P::pack_columns_fn(|lane| prev_layer[step * (first_row + lane) + n])
                } else {
                    default_packed
                }
            });
            let packed_digest = c.compress(children);
            P::unpack_into(&packed_digest, digests_chunk);
        });

    for i in (next_len / width * width)..next_len {
        let children: [_; N] = array::from_fn(|n| {
            if n < step {
                prev_layer[step * i + n]
            } else {
                default_digest
            }
        });
        next_digests[i] = c.compress(children);
    }

    next_digests
}

#[cfg(test)]
mod tests {
    use p3_baby_bear::{BabyBear, Poseidon2BabyBear};
    use p3_field::Field;
    use p3_keccak::Keccak256Hash;
    use p3_matrix::dense::RowMajorMatrix;
    use p3_symmetric::{
        CompressionFunctionFromHasher, PaddingFreeSponge, PseudoCompressionFunction,
        SerializingHasher, TruncatedPermutation,
    };
    use rand::rngs::SmallRng;
    use rand::{RngExt, SeedableRng};

    use super::*;

    type F = BabyBear;
    type Perm = Poseidon2BabyBear<16>;
    type MyHash = PaddingFreeSponge<Perm, 16, 8, 8>;
    type MyCompress = TruncatedPermutation<Perm, 2, 8, 16>;

    #[derive(Clone, Copy)]
    struct DummyCompressionFunction;

    impl PseudoCompressionFunction<[u8; 32], 2> for DummyCompressionFunction {
        fn compress(&self, input: [[u8; 32]; 2]) -> [u8; 32] {
            let mut output = [0u8; 32];
            for (i, o) in output.iter_mut().enumerate() {
                // Simple XOR-based compression
                *o = input[0][i] ^ input[1][i];
            }
            output
        }
    }

    /// Tree shapes the batched and unbatched drivers must agree on, as matrix heights and width.
    ///
    /// Between them they cover every boundary the batching introduces:
    ///
    /// - A tree of a single row, which has no level above the leaves at all.
    /// - Odd heights, which leave a padding tail at the top of a level.
    /// - Node counts that leave a partial final group in a hash call.
    ///
    /// - Several matrices sharing the tallest height, so a leaf message spans two rows.
    /// - A ragged height ladder, which forces injection levels.
    ///
    /// - Levels wide enough to fan out across threads instead of staying serial.
    /// - Rows long enough to make the sponge absorb more than one block.
    const SHAPES: &[(&[usize], usize)] = &[
        (&[1], 1),
        (&[3], 4),
        (&[13], 1),
        (&[8, 8, 4], 3),
        (&[17, 9, 5, 3], 2),
        (&[1100], 1),
        (&[2049, 1025, 513], 5),
        (&[64], 135),
    ];

    /// A hasher and compressor pair that reports one lane, forcing the unbatched driver.
    ///
    /// Every digest it produces is the wrapped primitive's, so the two drivers are compared on
    /// the same hash function and any difference is the driver's alone.
    #[derive(Clone, Copy, Debug)]
    struct Unbatched<T>(T);

    impl<Item, Out, T> CryptographicHasher<Item, Out> for Unbatched<T>
    where
        Item: Clone,
        T: CryptographicHasher<Item, Out>,
    {
        const LANES: usize = 1;

        fn hash_iter<I>(&self, input: I) -> Out
        where
            I: IntoIterator<Item = Item>,
        {
            self.0.hash_iter(input)
        }

        fn hash_iter_slices<'a, I>(&self, input: I) -> Out
        where
            I: IntoIterator<Item = &'a [Item]>,
            Item: 'a,
        {
            self.0.hash_iter_slices(input)
        }
    }

    impl<T, Inner, const N: usize> PseudoCompressionFunction<T, N> for Unbatched<Inner>
    where
        Inner: PseudoCompressionFunction<T, N>,
    {
        const LANES: usize = 1;

        fn compress(&self, input: [T; N]) -> T {
            self.0.compress(input)
        }
    }

    /// A byte hasher reporting three lanes, with no batched implementation of its own.
    ///
    /// Three is deliberately neither a power of two nor a divisor of any test height, so every
    /// group the driver forms ends in a short remainder.
    ///
    /// Leaving the batched hash at its default also isolates the driver: any disagreement comes
    /// from how the driver assembles messages, not from a vectorized sponge.
    #[derive(Clone, Copy, Debug)]
    struct ThreeLaneMix;

    impl CryptographicHasher<u8, [u8; 32]> for ThreeLaneMix {
        const LANES: usize = 3;

        fn hash_iter<I>(&self, input: I) -> [u8; 32]
        where
            I: IntoIterator<Item = u8>,
        {
            // A four-word state absorbing one byte per step, mixed with an odd multiplier and a
            // rotation so that byte order and message length both change the result.
            let mut state = [0x243f_6a88_85a3_08d3u64; 4];
            let mut count = 0u64;
            for byte in input {
                let word = &mut state[(count % 4) as usize];
                *word = word.rotate_left(11).wrapping_mul(0x9e37_79b9_7f4a_7c15) ^ u64::from(byte);
                count += 1;
            }

            // Fold the length in so a truncated message cannot collide with a longer one.
            state[0] ^= count.wrapping_mul(0xc2b2_ae3d_27d4_eb4f);

            let mut digest = [0u8; 32];
            for (word, slot) in state.iter().zip(digest.as_chunks_mut::<8>().0) {
                *slot = word.to_le_bytes();
            }
            digest
        }
    }

    /// Build the same tree with both drivers and compare every node of every layer.
    ///
    /// Comparing whole layers rather than just the root pins where a divergence begins, and a
    /// root match alone could hide two compensating errors at a lower level.
    fn assert_drivers_agree<H, C, const N: usize>(h: &H, c: &C, heights: &[usize], width: usize)
    where
        H: CryptographicHasher<F, [u8; 32]> + Sync,
        C: PseudoCompressionFunction<[u8; 32], N> + Sync,
    {
        // Fixture: one random matrix per requested height, all at the same width.
        let mut rng = SmallRng::seed_from_u64(heights[0] as u64 * 1_000_003 + width as u64);
        let leaves: Vec<RowMajorMatrix<F>> = heights
            .iter()
            .map(|&height| RowMajorMatrix::rand(&mut rng, height, width))
            .collect();

        let batched =
            MerkleTree::<F, u8, RowMajorMatrix<F>, N, 32>::new::<F, u8, H, C>(h, c, leaves.clone());

        let unbatched_h = Unbatched(h.clone());
        let unbatched_c = Unbatched(c.clone());
        let unbatched = MerkleTree::<F, u8, RowMajorMatrix<F>, N, 32>::new::<
            F,
            u8,
            Unbatched<H>,
            Unbatched<C>,
        >(&unbatched_h, &unbatched_c, leaves);

        assert_eq!(
            batched.arity_schedule, unbatched.arity_schedule,
            "arity schedule differs for heights {heights:?} width {width}"
        );
        assert_eq!(
            batched.digest_layers.len(),
            unbatched.digest_layers.len(),
            "layer count differs for heights {heights:?} width {width}"
        );
        for (level, (left, right)) in batched
            .digest_layers
            .iter()
            .zip(&unbatched.digest_layers)
            .enumerate()
        {
            assert_eq!(
                left, right,
                "layer {level} differs for heights {heights:?} width {width}"
            );
        }
        assert_eq!(batched.root(), unbatched.root());
    }

    #[test]
    fn default_digest_layer_is_all_default() {
        // Byte digests: the word type is a primitive, so the allocation takes the zeroed path
        // and the reinterpretation back to digests must still read as the default digest.
        let bytes = default_digest_layer::<u8, 32>(5);
        assert_eq!(bytes, vec![[0u8; 32]; 5]);

        // A field word type whose default is not a primitive zero follows the plain fill path.
        let words = default_digest_layer::<F, 8>(3);
        assert_eq!(words, vec![[F::default(); 8]; 3]);

        // An empty layer is legal: a height-zero tree allocates nothing.
        assert!(default_digest_layer::<u8, 32>(0).is_empty());

        // A digest width of one exercises the case where a digest is a single word.
        assert_eq!(default_digest_layer::<u8, 1>(7), vec![[0u8; 1]; 7]);
    }

    #[test]
    fn keccak_batched_tree_matches_unbatched_binary() {
        // Binary arity is the configuration every byte-digest scheme in the workspace uses.
        let h = SerializingHasher::new(Keccak256Hash);
        let c = CompressionFunctionFromHasher::<_, 2, 32>::new(Keccak256Hash);

        for &(heights, width) in SHAPES {
            assert_drivers_agree::<_, _, 2>(&h, &c, heights, width);
        }
    }

    #[test]
    fn keccak_batched_tree_matches_unbatched_quaternary() {
        // At arity four a level takes either a full four-to-one step, which the batched arm
        // handles, or a binary bridge step before an injection, which falls back to the
        // unbatched arm.
        //
        // Both must land on the same digests.
        let h = SerializingHasher::new(Keccak256Hash);
        let c = CompressionFunctionFromHasher::<_, 4, 32>::new(Keccak256Hash);

        for &(heights, width) in SHAPES {
            assert_drivers_agree::<_, _, 4>(&h, &c, heights, width);
        }
    }

    #[test]
    fn three_lane_batched_tree_matches_unbatched() {
        // A lane count of three exercises the driver's group arithmetic away from the powers of
        // two the vectorized sponges use.
        let h = SerializingHasher::new(ThreeLaneMix);
        let c = CompressionFunctionFromHasher::<_, 2, 32>::new(ThreeLaneMix);

        for &(heights, width) in SHAPES {
            assert_drivers_agree::<_, _, 2>(&h, &c, heights, width);
        }
    }

    #[test]
    fn keccak_tree_is_deterministic_across_builds() {
        // Padding slots take part in the next level's compression, so an unwritten slot would
        // show up as a root that changes between two builds of the same input.
        let h = SerializingHasher::new(Keccak256Hash);
        let c = CompressionFunctionFromHasher::<_, 2, 32>::new(Keccak256Hash);

        let mut rng = SmallRng::seed_from_u64(7);
        // Height 13 pads the leaf layer to 14 and every level above it to an even count.
        let leaves = vec![RowMajorMatrix::<F>::rand(&mut rng, 13, 3)];

        let first = MerkleTree::<F, u8, RowMajorMatrix<F>, 2, 32>::new::<F, u8, _, _>(
            &h,
            &c,
            leaves.clone(),
        );
        let second =
            MerkleTree::<F, u8, RowMajorMatrix<F>, 2, 32>::new::<F, u8, _, _>(&h, &c, leaves);

        assert_eq!(first.digest_layers, second.digest_layers);
    }

    #[test]
    fn test_compress_even_length() {
        let prev_layer = [[0x01; 32], [0x02; 32], [0x03; 32], [0x04; 32]];
        let compressor = DummyCompressionFunction;
        let expected = vec![
            [0x03; 32], // 0x01 ^ 0x02
            [0x07; 32], // 0x03 ^ 0x04
        ];
        let result = compress::<u8, DummyCompressionFunction, 2, 32>(&prev_layer, 2, &compressor);
        assert_eq!(result, expected);
    }

    #[test]
    fn test_compress_odd_length() {
        let prev_layer = [[0x05; 32], [0x06; 32], [0x07; 32]];
        let compressor = DummyCompressionFunction;
        let expected = vec![
            [0x03; 32], // 0x05 ^ 0x06
        ];
        let result = compress::<u8, DummyCompressionFunction, 2, 32>(&prev_layer, 2, &compressor);
        assert_eq!(result, expected);
    }

    #[test]
    fn test_compress_random_values() {
        let mut rng = SmallRng::seed_from_u64(1);
        let prev_layer: Vec<[u8; 32]> = (0..8).map(|_| rng.random()).collect();
        let compressor = DummyCompressionFunction;
        let expected: Vec<[u8; 32]> = prev_layer
            .as_chunks::<2>()
            .0
            .iter()
            .map(|pair| {
                let mut result = [0u8; 32];
                for (i, r) in result.iter_mut().enumerate() {
                    *r = pair[0][i] ^ pair[1][i];
                }
                result
            })
            .collect();
        let result = compress::<u8, DummyCompressionFunction, 2, 32>(&prev_layer, 2, &compressor);
        assert_eq!(result, expected);
    }

    #[test]
    fn test_compress_root_case_single_pair() {
        // When `prev_layer.len() == 2` we are at the “root-formation” case and
        // the function must return exactly one digest.
        //
        // 0xAA ^ 0x55 = 0xFF
        let prev_layer = [[0xAA; 32], [0x55; 32]];
        let compressor = DummyCompressionFunction;
        let expected = vec![[0xFF; 32]];
        let result = compress::<u8, DummyCompressionFunction, 2, 32>(&prev_layer, 2, &compressor);
        assert_eq!(result, expected);
    }

    #[test]
    fn test_compress_non_power_of_two_with_padding() {
        // The code intentionally pads to the next even length unless the output
        // would become the root.  With `len() == 6` the output length must be 4
        // (three real digests plus one zero digest).

        let prev_layer = [
            [0x01; 32], [0x02; 32], [0x03; 32], [0x04; 32], [0x05; 32], [0x06; 32],
        ];
        let compressor = DummyCompressionFunction;

        let mut expected = vec![
            [0x03; 32], // 01 ^ 02
            [0x07; 32], // 03 ^ 04
            [0x03; 32], // 05 ^ 06
        ];
        // extra padded digest filled with 0
        expected.push([0x00; 32]);

        let result = compress::<u8, DummyCompressionFunction, 2, 32>(&prev_layer, 2, &compressor);
        assert_eq!(result, expected);
        assert_eq!(result.len(), 4);
    }

    #[test]
    fn test_padded_len_n2() {
        assert_eq!(padded_len(0, 2), 0);
        assert_eq!(padded_len(1, 2), 1);
        assert_eq!(padded_len(2, 2), 2);
        assert_eq!(padded_len(3, 2), 4);
        assert_eq!(padded_len(4, 2), 4);
        assert_eq!(padded_len(5, 2), 6);
        assert_eq!(padded_len(7, 2), 8);
        assert_eq!(padded_len(8, 2), 8);
        assert_eq!(padded_len(9, 2), 10);
        assert_eq!(padded_len(15, 2), 16);
        assert_eq!(padded_len(16, 2), 16);
    }

    #[test]
    fn test_padded_len_n4() {
        assert_eq!(padded_len(0, 4), 0);
        assert_eq!(padded_len(1, 4), 1);
        // Below-arity case: pad to exactly N
        assert_eq!(padded_len(2, 4), 4);
        assert_eq!(padded_len(3, 4), 4);
        // At or above arity: pad to next multiple of N
        assert_eq!(padded_len(4, 4), 4);
        assert_eq!(padded_len(5, 4), 8);
        assert_eq!(padded_len(7, 4), 8);
        assert_eq!(padded_len(8, 4), 8);
        assert_eq!(padded_len(9, 4), 12);
    }

    #[test]
    fn test_padded_len_n8() {
        assert_eq!(padded_len(0, 8), 0);
        assert_eq!(padded_len(1, 8), 1);
        // Below-arity: all pad to exactly N=8
        assert_eq!(padded_len(2, 8), 8);
        assert_eq!(padded_len(3, 8), 8);
        assert_eq!(padded_len(5, 8), 8);
        assert_eq!(padded_len(7, 8), 8);
        // At or above arity: next multiple of 8
        assert_eq!(padded_len(8, 8), 8);
        assert_eq!(padded_len(9, 8), 16);
        assert_eq!(padded_len(15, 8), 16);
        assert_eq!(padded_len(16, 8), 16);
    }

    #[test]
    fn test_padded_len_always_admits_full_groups() {
        // For any N in {2, 4, 8} and any raw_len > 1,
        // padded_len must be >= N and divisible by N (so a full compression
        // group is always possible), OR padded_len == raw_len <= 1 (root).
        for n in [2, 4, 8] {
            for raw_len in 2..=128 {
                let pl = padded_len(raw_len, n);
                assert!(
                    pl >= n && pl.is_multiple_of(n),
                    "padded_len({raw_len}, {n}) = {pl} is not a valid multiple of {n}",
                );
            }
        }
    }

    #[test]
    #[should_panic(expected = "matrix height 4 incompatible with tallest height 6")]
    fn new_rejects_heights_off_ladder() {
        // `MerkleTree::new` is a public constructor that bypasses `Mmcs::commit`'s
        // geometry gate. Heights 6 and 4 pass the weaker "equal within the same
        // power-of-two bucket" rule (next_power_of_two(6) = 8, next_power_of_two(4) = 4
        // — different buckets), but 4 is off the ceil(6 / 2^k) ladder: at k = 1 the
        // only admissible height is ceil(6 / 2) = 3. Building a tree from these would
        // panic later, out of bounds, inside a rayon closure — the constructor must
        // reject it up front instead.
        let mut rng = SmallRng::seed_from_u64(0);
        let perm = Perm::new_from_rng_128(&mut rng);
        let hash = MyHash::new(perm.clone());
        let compress = MyCompress::new(perm);

        let mat6 = RowMajorMatrix::<F>::rand(&mut rng, 6, 1);
        let mat4 = RowMajorMatrix::<F>::rand(&mut rng, 4, 1);

        let _ = MerkleTree::new::<<F as Field>::Packing, <F as Field>::Packing, MyHash, MyCompress>(
            &hash,
            &compress,
            vec![mat6, mat4],
        );
    }

    #[test]
    #[should_panic]
    fn new_rejects_single_zero_height_matrix() {
        // A single height-0 matrix used to build `digest_layers == [[]]`,
        // and `root()` would panic later. The constructor must reject it directly.
        let mut rng = SmallRng::seed_from_u64(0);
        let perm = Perm::new_from_rng_128(&mut rng);
        let hash = MyHash::new(perm.clone());
        let compress = MyCompress::new(perm);

        let empty_mat = RowMajorMatrix::<F>::new(Vec::new(), 1);

        let _ = MerkleTree::new::<<F as Field>::Packing, <F as Field>::Packing, MyHash, MyCompress>(
            &hash,
            &compress,
            vec![empty_mat],
        );
    }
}
