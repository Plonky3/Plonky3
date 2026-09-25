//! Planning and parallel construction of the digest layers of a Merkle tree.
//!
//! A build runs in two steps.
//!
//! - Plan: walk the matrix heights once and fix every layer's arity, injected matrices and size.
//! - Fill: hash the layers bottom up, whole subtrees at a time.
//!
//! A node depends only on a contiguous run of nodes one layer down.
//!
//! So a run of nodes at some layer depends on one contiguous run at every layer below it:
//!
//! ```text
//!     layer 2    [   block 0   |   block 1   ]
//!     layer 1    [ b0  |  b0   |  b1  |  b1  ]
//!     layer 0    [b0|b0|b0|b0  |b1|b1|b1|b1  ]
//! ```
//!
//! One task builds one such block from the leaves up, with no barrier between its layers.
//!
//! Its digests stay in the private caches from the layer that writes them to the one reading them.
//!
//! A pass stops at the highest layer where every block still holds a few full vector groups.
//!
//! The layers above it are narrow, so the next pass splits them into fewer blocks.

use alloc::vec;
use alloc::vec::Vec;
use core::cmp::Reverse;
use core::marker::PhantomData;
use core::ops::RangeInclusive;
use core::{array, mem};

use itertools::Itertools;
use p3_field::PackedValue;
use p3_matrix::Matrix;
use p3_maybe_rayon::prelude::*;
use p3_symmetric::{CryptographicHasher, PseudoCompressionFunction};
use p3_util::transpose::transpose_rows;

use crate::merkle_tree::{padded_len, select_arity_step};

/// Blocks per worker thread in the first pass.
///
/// A pass ends when its last block does, so a block is the unit of load imbalance.
///
/// Sixteen blocks per thread bound that tail to about a sixteenth of a thread's share.
///
/// That keeps a slow permutation, where one block is milliseconds of work, from idling the rest.
const BLOCKS_PER_THREAD: usize = 16;

/// Vector groups a block must hold at the top layer of its pass.
///
/// One full group per block keeps every vector call at full width.
const MIN_BLOCK_GROUPS: usize = 1;

/// Nodes of an injecting layer staged at once by the batched arm.
///
/// Three buffers of this many digests fit in the mid-level cache on every current core.
const INJECT_CHUNK_NODES: usize = 1024;

/// Target size in bytes of the buffer holding one group of copied rows.
///
/// A batched hasher wants its messages back to back in memory.
/// A matrix only promises access one row at a time.
/// So a group of rows is copied in first.
///
/// 16 KiB keeps that copy in the first-level cache until the pass that reads it back.
///
/// This is a target rather than a cap.
///
/// A group is rounded up to a whole lane group.
///
/// One row is staged however wide it is.
///
/// So the buffer reaches `max(this, lanes * row bytes)`, which at eight lanes is:
///
/// ```text
///     row bytes   staged           the batching consumer
///        2 KiB     16 KiB   (1x)   groups rows, every lane busy
///        8 KiB     64 KiB   (4x)   groups rows, every lane busy
///       64 KiB    512 KiB  (32x)   groups rows, every lane busy
/// ```
///
/// An idle lane costs a whole permutation, far more than the cache level given up.
pub(crate) const ROW_SCRATCH_BYTES: usize = 16 * 1024;

/// Rows one hash call stages, from the byte target and the hasher's lane count.
///
/// # Returns
///
/// A count of at least one, always a whole number of lane groups.
///
/// It stages at most the target, or one lane group of rows when that is larger.
///
/// Zero on either argument is read as one, so every input has an answer.
pub(crate) fn rows_per_call(row_bytes: usize, lanes: usize) -> usize {
    // Both divisors are normalized here, so no caller reaches a division by zero.
    let row_bytes = row_bytes.max(1);
    let lanes = lanes.max(1);

    // Whole lane groups inside the target, and one group when none fits.
    (ROW_SCRATCH_BYTES / row_bytes / lanes).max(1) * lanes
}

/// Allocate a digest layer with every slot set to the default digest.
///
/// Hashing overwrites every slot but a short padding tail, so the fill is pure overhead.
///
/// A zero-filled allocation makes it free.
///
/// The operating system hands back zero pages and faults them in only when hashing writes them.
///
/// The standard vector constructor reaches that path only for primitive element types.
///
/// It gives up for arrays longer than sixteen elements, the shape of a thirty-two-byte digest.
///
/// Building the layer as one flat run of digest words restores it:
///
/// ```text
///     flat:    [ w_0 w_1 ... w_{D-1} | w_D ... w_{2D-1} | ... ]   len * D words
///     digests: [       digest_0      |     digest_1     | ... ]   len digests
/// ```
///
/// # Panics
///
/// Panics if the layer is too large for the address space.
pub(crate) fn default_digest_layer<W, const DIGEST_ELEMS: usize>(
    len: usize,
) -> Vec<[W; DIGEST_ELEMS]>
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
    mem::forget(flat);

    // SAFETY: an array of words has no padding.
    //
    // So `len` digests occupy exactly the words allocated above, with the alignment of one word.
    //
    // The length equals the reserved capacity.
    //
    // So the layout handed back to the allocator on drop is the layout it handed out.
    //
    // Every word is initialized, so every digest slot reads as the default digest.
    unsafe { Vec::from_raw_parts(ptr, len, len) }
}

/// One digest layer, as fixed by the plan before any hashing.
///
/// A node at an upper layer is, with `a` its arity and the default digest `0` as filler:
///
/// ```text
///     folded = C(child_{a*i}, ..., child_{a*i + a - 1}, 0, ..., 0)
///     node_i = folded                                 no injected matrices
///     node_i = C(folded, H(row_i), 0, ..., 0)         injected matrices, i < their height
///     node_i = C(folded, 0, 0, ..., 0)                injected matrices, i >= their height
/// ```
///
/// A leaf is the hash of row `i` concatenated across the tallest matrices.
#[derive(Debug)]
pub(crate) struct Layer<'a, M> {
    /// Children folded into each node: zero at the leaves, else two or the full arity.
    pub(crate) arity: usize,

    /// Matrices whose rows are hashed into this layer, all of one height, in insertion order.
    pub(crate) matrices: Vec<&'a M>,

    /// Nodes hashed at this layer.
    ///
    /// The allocated layer is padded past them with default digests.
    pub(crate) computed: usize,
}

/// Builds every digest layer of one tree.
///
/// It holds the plan and the two hash functions, and is shared read-only by every task.
pub(crate) struct TreeBuilder<'a, P, PW, H, C, M, const N: usize, const DIGEST_ELEMS: usize> {
    /// Hasher applied to matrix rows.
    h: &'a H,

    /// Compression function applied to groups of `N` digests.
    c: &'a C,

    /// Every layer, from the leaves up to the root.
    layers: Vec<Layer<'a, M>>,

    /// Binds the packed input and digest types used by the vector arms.
    _packing: PhantomData<(P, PW)>,
}

impl<'a, P, PW, H, C, M, const N: usize, const DIGEST_ELEMS: usize>
    TreeBuilder<'a, P, PW, H, C, M, N, DIGEST_ELEMS>
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
    /// Messages the row hasher takes per call.
    const HASH_LANES: usize =
        <H as CryptographicHasher<P::Value, [PW::Value; DIGEST_ELEMS]>>::LANES;

    /// Groups the compression function takes per call.
    const COMPRESS_LANES: usize =
        <C as PseudoCompressionFunction<[PW::Value; DIGEST_ELEMS], N>>::LANES;

    /// Plan the layers of a tree over a set of matrices.
    ///
    /// Heights are assumed to sit on the `ceil(max_height / 2^k)` ladder.
    ///
    /// # Panics
    ///
    /// Panics if there are no matrices.
    pub(crate) fn new(h: &'a H, c: &'a C, leaves: &'a [M]) -> Self {
        // Matrices are consumed tallest first, each at the layer matching its height.
        let mut by_height = leaves
            .iter()
            .sorted_by_key(|m| Reverse(m.height()))
            .peekable();

        let max_height = by_height.peek().expect("no matrices to commit").height();
        let leaf_height_npt = max_height.next_power_of_two();

        // The leaf layer hashes every matrix of the tallest height.
        let tallest = by_height
            .peeking_take_while(|m| m.height() == max_height)
            .collect_vec();
        let mut layers = vec![Layer {
            arity: 0,
            matrices: tallest,
            computed: max_height,
        }];

        // Each further layer folds the padded layer below it, until one digest remains.
        let mut len = padded_len(max_height, N);
        while len > 1 {
            // A binary bridge step lands a shorter matrix on its own layer.
            let arity =
                select_arity_step::<N>(len, leaf_height_npt, by_height.clone().map(|m| m.height()));
            let computed = len / arity;

            // Matrices whose padded height matches this layer are injected into it.
            let target = computed.next_power_of_two();
            let matrices = by_height
                .peeking_take_while(|m| m.height().next_power_of_two() == target)
                .collect_vec();

            layers.push(Layer {
                arity,
                matrices,
                computed,
            });
            len = padded_len(computed, N);
        }

        Self {
            h,
            c,
            layers,
            _packing: PhantomData,
        }
    }

    /// The compression arity of every layer above the leaves, bottom up.
    pub(crate) fn arity_schedule(&self) -> Vec<usize> {
        self.layers[1..].iter().map(|layer| layer.arity).collect()
    }

    /// Hash every layer and return them, the leaf digests first and the root last.
    pub(crate) fn build(&self) -> Vec<Vec<[PW::Value; DIGEST_ELEMS]>> {
        // Every layer is allocated up front, so each pass writes into disjoint pieces of them.
        let mut digests: Vec<_> = self
            .layers
            .iter()
            .map(|layer| default_digest_layer(padded_len(layer.computed, N)))
            .collect();

        // Each pass covers a contiguous band of layers, lowest first.
        let mut lo = 0;
        while lo < self.layers.len() {
            let (blocks, hi) = self.pass_shape(lo);
            self.run_pass(&mut digests, lo..=hi, blocks);
            lo = hi + 1;
        }

        digests
    }

    /// Nodes one vector call hashes together.
    ///
    /// Blocks are cut on multiples of it, so that only the last block runs a partial group.
    fn group() -> usize {
        P::WIDTH.max(Self::HASH_LANES).max(Self::COMPRESS_LANES)
    }

    /// Choose the block count and the top layer of a pass.
    ///
    /// # Arguments
    ///
    /// - `lo`: the lowest layer of the pass, the first one not yet built.
    ///
    /// # Returns
    ///
    /// The block count, and the highest layer where every block holds its minimum of groups.
    ///
    /// Once some thread would get no block, one block takes every remaining layer.
    fn pass_shape(&self, lo: usize) -> (usize, usize) {
        let group = Self::group();
        let min_block = group * MIN_BLOCK_GROUPS;

        // The first pass carries almost all the work, so it takes many blocks for balance.
        //
        // A later pass carries little, so one block per thread lets it climb the highest.
        let threads = current_num_threads();
        let per_thread = if lo == 0 { BLOCKS_PER_THREAD } else { 1 };

        // No more blocks than the base layer can fill.
        let blocks = (self.layers[lo].computed / min_block)
            .min(per_thread * threads)
            .max(1);

        // Once some thread would get no block, the layers left are too narrow to share.
        //
        // One block then finishes them, cheaper than a dispatch per remaining layer.
        if blocks == 1 || blocks < threads {
            return (1, self.layers.len() - 1);
        }

        // Climb while the layer still gives every block its minimum.
        let hi = (lo..self.layers.len())
            .take_while(|&l| self.layers[l].computed >= blocks * min_block)
            .last()
            .unwrap_or(lo);
        (blocks, hi)
    }

    /// Build a band of layers as independent subtree blocks, one task each.
    ///
    /// Block `j` owns nodes from `j * size_l` of layer `l`, where sizes grow by each arity:
    ///
    /// ```text
    ///     size_hi     = group * ceil(computed_hi / (blocks * group))
    ///     size_{l-1}  = size_l * arity_l
    /// ```
    ///
    /// Its nodes at layer `l` read only its own nodes at layer `l - 1`.
    ///
    /// The last block's pieces also hold the padding tails, which stay at the default digest.
    ///
    /// # Arguments
    ///
    /// - `digests`: every layer, with the ones below the band already built.
    /// - `band`: the layers this pass builds, bottom to top.
    /// - `blocks`: the requested block count, before rounding to whole vector groups.
    fn run_pass(
        &self,
        digests: &mut [Vec<[PW::Value; DIGEST_ELEMS]>],
        band: RangeInclusive<usize>,
        blocks: usize,
    ) {
        let (lo, hi) = (*band.start(), *band.end());
        let group = Self::group();

        // Block sizes at every layer of the band, top down from the rounded top size.
        let mut sizes = vec![0; hi - lo + 1];
        sizes[hi - lo] = group * self.layers[hi].computed.div_ceil(blocks * group);
        for l in (lo..hi).rev() {
            sizes[l - lo] = sizes[l + 1 - lo] * self.layers[l + 1].arity;
        }

        // Layers below the band are finished and read-only; the band is split into pieces.
        let (below, band_layers) = digests.split_at_mut(lo);
        let children = below.last().map(Vec::as_slice);
        let band_layers = &mut band_layers[..=hi - lo];

        // Rounding the top size up can leave fewer non-empty blocks than requested.
        let count = self.layers[hi].computed.div_ceil(sizes[hi - lo]);

        // Cut every layer into one piece per block, block-major.
        let mut pieces: Vec<Vec<&mut [[PW::Value; DIGEST_ELEMS]]>> = (0..count)
            .map(|_| Vec::with_capacity(band_layers.len()))
            .collect();
        for (layer, &size) in band_layers.iter_mut().zip(&sizes) {
            let mut rest = layer.as_mut_slice();
            for (j, block) in pieces.iter_mut().enumerate() {
                // The last block also takes the padding tail of every layer below the top.
                let take = if j + 1 == count {
                    rest.len()
                } else {
                    size.min(rest.len())
                };
                let (piece, tail) = mem::take(&mut rest).split_at_mut(take);
                block.push(piece);
                rest = tail;
            }
        }

        // One set of staging buffers per worker, reused by every block it runs.
        let fill =
            |scratch: &mut Scratch<P::Value, PW::Value>,
             (j, mut block): (usize, Vec<&mut [[PW::Value; DIGEST_ELEMS]]>)| {
                for (offset, l) in (lo..=hi).enumerate() {
                    let layer = &self.layers[l];
                    let first = j * sizes[offset];

                    // The children of this piece start at `arity * first` one layer down.
                    let (done, rest) = block.split_at_mut(offset);
                    let kids: &[_] = match (done.last(), children) {
                        (Some(piece), _) => piece,
                        (None, Some(below)) => &below[layer.arity * first..],
                        (None, None) => &[],
                    };

                    // Only nodes below the computed count are hashed; the rest is padding.
                    let out = &mut *rest[0];
                    let nodes = layer.computed.saturating_sub(first).min(out.len());
                    self.fill_layer(layer, kids, &mut out[..nodes], first, scratch);
                }
            };

        if count == 1 {
            // One block needs no dispatch.
            let mut scratch = Scratch::default();
            pieces
                .into_iter()
                .enumerate()
                .for_each(|block| fill(&mut scratch, block));
        } else {
            pieces
                .into_par_iter()
                .enumerate()
                .for_each_init(Scratch::default, fill);
        }
    }

    /// Hash a contiguous run of nodes of one layer.
    ///
    /// # Arguments
    ///
    /// - `layer`: the plan of the layer.
    /// - `kids`: the layer below, from the first child of the run; empty at the leaves.
    /// - `out`: one slot per node of the run.
    /// - `first`: the index in the layer of the run's first node.
    /// - `scratch`: staging buffers owned by the calling task.
    fn fill_layer(
        &self,
        layer: &Layer<'a, M>,
        kids: &[[PW::Value; DIGEST_ELEMS]],
        out: &mut [[PW::Value; DIGEST_ELEMS]],
        first: usize,
        scratch: &mut Scratch<P::Value, PW::Value>,
    ) {
        let (hash_lanes, compress_lanes) = (Self::HASH_LANES, Self::COMPRESS_LANES);

        // A batched arm reads a node's children as one contiguous group of `N`.
        //
        // A binary bridge step below a larger arity breaks that, and takes the packed arm.
        let full_groups = layer.arity == N;
        match (layer.arity, layer.matrices.is_empty()) {
            (0, _) if hash_lanes > 1 => {
                self.hash_rows_batched(&layer.matrices, first, out, scratch);
            }
            (0, _) => self.hash_rows_packed(&layer.matrices, first, out, scratch),
            (_, true) if full_groups && compress_lanes > 1 => {
                let (groups, _) = kids[..out.len() * N].as_chunks::<N>();
                self.c.compress_many(groups, out);
            }
            (_, false) if full_groups && compress_lanes > 1 && hash_lanes > 1 => {
                self.inject_batched(layer, kids, out, first, scratch);
            }
            _ => self.fold_packed(layer, kids, out, first, scratch),
        }
    }

    /// Hash one leaf per row with a hasher that hashes several messages per call.
    ///
    /// The message of a row is that row concatenated across every matrix, in order.
    fn hash_rows_batched(
        &self,
        matrices: &[&M],
        first: usize,
        out: &mut [[PW::Value; DIGEST_ELEMS]],
        scratch: &mut Scratch<P::Value, PW::Value>,
    ) {
        // Bound the row range once, since the copy loop below skips per-row bounds checks.
        assert!(
            matrices.iter().all(|m| m.height() >= first + out.len()),
            "row range {first}..{} exceeds a matrix height",
            first + out.len()
        );

        let width: usize = matrices.iter().map(|m| m.width()).sum();
        let rows_per_group = rows_per_call(width * size_of::<P::Value>(), Self::HASH_LANES);

        for (group, digests) in out.chunks_mut(rows_per_group).enumerate() {
            let start = first + group * rows_per_group;

            // Lay the group's messages back to back: row by row, matrix by matrix.
            //
            // Extending from the row iterator keeps this free of allocation for every matrix type.
            let rows = &mut scratch.rows;
            rows.clear();
            for row in start..start + digests.len() {
                for m in matrices {
                    // SAFETY: the assertion above bounds every requested row by every height.
                    rows.extend(unsafe { m.row_unchecked(row) });
                }
            }

            self.h.hash_many(rows, digests);
        }
    }

    /// Hash one leaf per row, one row per lane of a packed hasher.
    fn hash_rows_packed(
        &self,
        matrices: &[&M],
        first: usize,
        out: &mut [[PW::Value; DIGEST_ELEMS]],
        scratch: &mut Scratch<P::Value, PW::Value>,
    ) {
        let lanes = P::WIDTH;
        let (groups, tail) = out.split_at_mut(Self::vector_nodes(out.len()));

        // Whole lane groups: one packed hash per group of rows.
        for (g, digests) in groups.chunks_exact_mut(lanes).enumerate() {
            // SAFETY: the group ends at or before the last requested row, which every matrix holds.
            let packed = unsafe { self.hash_packed_rows(matrices, first + g * lanes, scratch) };
            unpack_digests(&packed, digests);
        }

        // Tail rows: hash each one alone.
        let tail_first = first + groups.len();
        for (offset, digest) in tail.iter_mut().enumerate() {
            *digest = self.hash_row(matrices, tail_first + offset);
        }
    }

    /// Hash one packed group of rows, one row message per lane.
    ///
    /// Lane `l` hashes row `l` of the group, concatenated across every matrix in order.
    ///
    /// The group is staged column-major so that each packed input is one whole vector:
    ///
    /// ```text
    ///     staged: [ col_0 x W | col_1 x W | ... ]   one block of W lanes per column
    ///     packed: [   p_0     |   p_1     | ... ]   the same memory, one vector per column
    /// ```
    ///
    /// # Safety
    ///
    /// Every matrix must hold every row of the group.
    unsafe fn hash_packed_rows(
        &self,
        matrices: &[&M],
        first: usize,
        scratch: &mut Scratch<P::Value, PW::Value>,
    ) -> [PW; DIGEST_ELEMS] {
        let lanes = P::WIDTH;
        let width: usize = matrices.iter().map(|m| m.width()).sum();

        // Every group has the same shape, so after the first call this resize is a no-op.
        //
        // The transposes overwrite every slot, so stale contents never reach the hasher.
        let staged = &mut scratch.rows;
        staged.resize(width * lanes, P::Value::default());

        // Each matrix owns a contiguous run of columns in the staged message.
        let mut start = 0;
        for m in matrices {
            let cols = m.width();

            // Hold the lane rows while the transpose reads them.
            //
            // SAFETY: the caller guarantees rows first .. first + lanes exist.
            let rows: Vec<_> = (0..lanes)
                .map(|lane| unsafe { m.row_slice_unchecked(first + lane) })
                .collect();

            let block = &mut staged[start * lanes..(start + cols) * lanes];
            transpose_rows(lanes, cols, |lane| &*rows[lane], block);
            start += cols;
        }

        self.h.hash_iter(P::pack_slice(staged).iter().copied())
    }

    /// Fold children into nodes and inject rows, one node per lane of a packed compression.
    ///
    /// A lane group straddling the injected height mixes both kinds of node.
    ///
    /// Such a group takes the scalar path.
    fn fold_packed(
        &self,
        layer: &Layer<'a, M>,
        kids: &[[PW::Value; DIGEST_ELEMS]],
        out: &mut [[PW::Value; DIGEST_ELEMS]],
        first: usize,
        scratch: &mut Scratch<P::Value, PW::Value>,
    ) {
        let lanes = PW::WIDTH;
        let rows = Self::injected_height(layer);
        let default = [PW::broadcast(PW::Value::default()); DIGEST_ELEMS];

        let (groups, tail) = out.split_at_mut(Self::vector_nodes(out.len()));
        for (g, digests) in groups.chunks_exact_mut(lanes).enumerate() {
            let local = g * lanes;
            let node = first + local;

            // A group is uniform when it lies wholly below or wholly above the injected rows.
            let injected = !layer.matrices.is_empty() && node < rows;
            if injected && node + lanes > rows {
                for (i, digest) in digests.iter_mut().enumerate() {
                    *digest = self.node(layer, kids, local + i, node + i);
                }
                continue;
            }

            let children = pack_children::<PW, N, DIGEST_ELEMS>(
                &kids[layer.arity * local..],
                layer.arity,
                &mut scratch.digests,
            );
            let mut packed = self.c.compress(children);

            // Mix in the row digest, or the default digest past the injected rows.
            if !layer.matrices.is_empty() {
                let row = if injected {
                    // SAFETY: the uniform group lies wholly below the injected height.
                    unsafe { self.hash_packed_rows(&layer.matrices, node, scratch) }
                } else {
                    default
                };
                packed = self.c.compress(array::from_fn(|slot| match slot {
                    0 => packed,
                    1 => row,
                    _ => default,
                }));
            }

            unpack_digests(&packed, digests);
        }

        // Tail nodes past the last whole group.
        let tail_local = groups.len();
        for (i, digest) in tail.iter_mut().enumerate() {
            *digest = self.node(layer, kids, tail_local + i, first + tail_local + i);
        }
    }

    /// Fold children and inject rows with batched hashing and batched compression.
    ///
    /// Each chunk of nodes goes through three passes over small buffers:
    ///
    /// ```text
    ///     pass 1: compress N children            -> folded
    ///     pass 2: hash one row per output node    -> injected
    ///     pass 3: compress [folded, injected]    -> out
    /// ```
    ///
    /// Nodes past the injected height mix in the default digest in place of a row digest.
    fn inject_batched(
        &self,
        layer: &Layer<'a, M>,
        kids: &[[PW::Value; DIGEST_ELEMS]],
        out: &mut [[PW::Value; DIGEST_ELEMS]],
        first: usize,
        scratch: &mut Scratch<P::Value, PW::Value>,
    ) {
        let rows = Self::injected_height(layer);
        let default = [PW::Value::default(); DIGEST_ELEMS];
        let (groups, _) = kids[..out.len() * N].as_chunks::<N>();

        for (k, (groups, out)) in groups
            .chunks(INJECT_CHUNK_NODES)
            .zip(out.chunks_mut(INJECT_CHUNK_NODES))
            .enumerate()
        {
            let base = first + k * INJECT_CHUNK_NODES;
            let count = out.len();

            // Pass 1: fold the children of every node in this chunk.
            let mut folded = vec![default; count];
            self.c.compress_many(groups, &mut folded);

            // Pass 2: hash one row per node, for as many nodes as the matrices are tall.
            let injected_count = count.min(rows.saturating_sub(base));
            let mut injected = vec![default; injected_count];
            self.hash_rows_batched(&layer.matrices, base, &mut injected, scratch);

            // Pass 3: pair each folded digest with its row digest, padding the group to `N`.
            let pairs: Vec<[_; N]> = (0..count)
                .map(|node| {
                    array::from_fn(|slot| match slot {
                        0 => folded[node],
                        1 if node < injected_count => injected[node],
                        _ => default,
                    })
                })
                .collect();
            self.c.compress_many(&pairs, out);
        }
    }

    /// Leading nodes of a run that go through whole vector groups; the rest take the scalar path.
    ///
    /// A one-lane packing gains nothing from staging, so all of its nodes take the scalar path.
    const fn vector_nodes(len: usize) -> usize {
        if P::WIDTH > 1 {
            len / P::WIDTH * P::WIDTH
        } else {
            0
        }
    }

    /// Height of the matrices injected into a layer, or zero when there are none.
    fn injected_height(layer: &Layer<'a, M>) -> usize {
        layer.matrices.first().map_or(0, |m| m.height())
    }

    /// Hash one leaf: one row concatenated across every matrix, in order.
    fn hash_row(&self, matrices: &[&M], row: usize) -> [PW::Value; DIGEST_ELEMS] {
        // SAFETY: every caller passes a row below the height of every matrix.
        unsafe {
            // One matrix: its row iterator alone, free of the flattening adapter's checks.
            if let [m] = matrices {
                return self.h.hash_iter(m.row_unchecked(row));
            }
            self.h
                .hash_iter(matrices.iter().flat_map(|m| m.row_unchecked(row)))
        }
    }

    /// Hash one upper-layer node on the scalar path.
    ///
    /// # Arguments
    ///
    /// - `layer`: the plan of the layer.
    /// - `kids`: the layer below, from the first child of the calling run.
    /// - `local`: the node's position in that run, which locates its children.
    /// - `node`: the node's index in the layer, which locates its injected row.
    fn node(
        &self,
        layer: &Layer<'a, M>,
        kids: &[[PW::Value; DIGEST_ELEMS]],
        local: usize,
        node: usize,
    ) -> [PW::Value; DIGEST_ELEMS] {
        let default = [PW::Value::default(); DIGEST_ELEMS];
        let arity = layer.arity;

        // Fold the node's children, padded to the full arity.
        let folded = self.c.compress(array::from_fn(|slot| {
            if slot < arity {
                kids[arity * local + slot]
            } else {
                default
            }
        }));
        if layer.matrices.is_empty() {
            return folded;
        }

        // Mix in the row digest, or the default digest past the injected rows.
        let row = if node < Self::injected_height(layer) {
            self.hash_row(&layer.matrices, node)
        } else {
            default
        };
        self.c.compress(array::from_fn(|slot| match slot {
            0 => folded,
            1 => row,
            _ => default,
        }))
    }
}

/// Buffers one task reuses across the layers it builds.
#[derive(Debug, Default)]
struct Scratch<F, W> {
    /// Staged row messages, row-major for batched hashing or column-major for packed hashing.
    rows: Vec<F>,

    /// Staged children, column-major for packed compression.
    digests: Vec<W>,
}

/// Pack the children of `P::WIDTH` consecutive nodes into one vector per child word.
///
/// Node `l` of the group takes its `a` children from position `a * l` of the layer below.
///
/// Those children are contiguous, so each lane reads one run of `arity * D` words:
///
/// ```text
///     lane l reads:   [ child_0 word_0 .. word_{D-1} | child_1 ... ]      D = digest words
///     packed[n][k]:   word k of child n, across the lanes
/// ```
///
/// Slots `arity .. N` take the default digest, the padding of a binary bridge step.
fn pack_children<P, const N: usize, const DIGEST_ELEMS: usize>(
    kids: &[[P::Value; DIGEST_ELEMS]],
    arity: usize,
    scratch: &mut Vec<P::Value>,
) -> [[P; DIGEST_ELEMS]; N]
where
    P: PackedValue,
{
    let lanes = P::WIDTH;
    let words = arity * DIGEST_ELEMS;

    // Same shape on every call, so the resize allocates once per task.
    scratch.resize(lanes * words, P::Value::default());
    transpose_rows(
        lanes,
        words,
        |lane| kids[arity * lane..arity * (lane + 1)].as_flattened(),
        scratch,
    );

    // Block n * D + k of the staged buffer is word k of child n.
    let packed = P::pack_slice(scratch);
    let default = P::broadcast(P::Value::default());
    array::from_fn(|n| {
        array::from_fn(|k| {
            if n < arity {
                packed[n * DIGEST_ELEMS + k]
            } else {
                default
            }
        })
    })
}

/// Unpack one digest per lane into consecutive output slots.
///
/// Word `k` of lane `l` lands in word `k` of output digest `l`, the transpose of the packing.
///
/// # Panics
///
/// Panics if the output does not hold exactly one digest per lane.
fn unpack_digests<P, const DIGEST_ELEMS: usize>(
    packed: &[P; DIGEST_ELEMS],
    out: &mut [[P::Value; DIGEST_ELEMS]],
) where
    P: PackedValue,
{
    let lanes = P::WIDTH;

    // Row k of the source is packed word k, one entry per lane.
    let words = P::unpack_slice(packed);
    transpose_rows(
        DIGEST_ELEMS,
        lanes,
        |k| &words[k * lanes..(k + 1) * lanes],
        out.as_flattened_mut(),
    );
}

#[cfg(test)]
mod tests {
    use alloc::vec::Vec;

    use p3_baby_bear::{BabyBear, Poseidon2BabyBear};
    use p3_field::{Field, PrimeCharacteristicRing, PrimeField64};
    use p3_keccak::{Keccak256Hash, KeccakF, VECTOR_LEN};
    use p3_matrix::dense::RowMajorMatrix;
    use p3_symmetric::{
        CompressionFunctionFromHasher, PaddingFreeSponge, SerializingHasher, TruncatedPermutation,
    };
    use proptest::collection;
    use proptest::prelude::*;
    use rand::SeedableRng;
    use rand::rngs::SmallRng;

    use super::*;
    use crate::MerkleTree;

    type F = BabyBear;
    type Packed = <F as Field>::Packing;

    /// Heights and width of each tree the reference comparison builds.
    ///
    /// - One row, and odd heights that leave padding tails.
    /// - A ragged height ladder, which injects shorter matrices on upper layers.
    /// - Several matrices sharing the tallest height, so a leaf spans more than one row.
    /// - Trees tall enough to split into many blocks and to need more than one pass.
    /// - Rows long enough to take more than one register tile.
    const SHAPES: &[(&[usize], usize)] = &[
        (&[1], 1),
        (&[3], 4),
        (&[13], 1),
        (&[8, 8, 4], 3),
        (&[17, 9, 5, 3], 2),
        (&[1100], 1),
        (&[2049, 1025, 513], 5),
        (&[64], 135),
        (&[4099, 4099, 2050, 1025, 17], 2),
        (&[8192, 256], 1),
        (&[6001], 20),
    ];

    /// Build every digest layer straight from the node definitions, one node at a time.
    ///
    /// Only the planning helpers are shared with the builder.
    ///
    /// Hashing, packing, batching and blocking are all independent of it.
    fn reference_layers<H, C, const N: usize, const D: usize>(
        h: &H,
        c: &C,
        leaves: &[RowMajorMatrix<F>],
    ) -> (Vec<Vec<[u64; D]>>, Vec<usize>)
    where
        H: CryptographicHasher<F, [u64; D]>,
        C: PseudoCompressionFunction<[u64; D], N>,
    {
        let zero = [0u64; D];
        let hash_row = |ms: &[&RowMajorMatrix<F>], r: usize| {
            h.hash_iter(ms.iter().flat_map(|m| m.row_slice(r).unwrap().to_vec()))
        };

        // Tallest first, ties in insertion order.
        let mut sorted: Vec<&RowMajorMatrix<F>> = leaves.iter().collect();
        sorted.sort_by_key(|m| Reverse(m.height()));
        let max_height = sorted[0].height();
        let split = sorted
            .iter()
            .take_while(|m| m.height() == max_height)
            .count();
        let (tallest, mut rest) = sorted.split_at(split);

        // Leaf layer: one digest per row, then default padding.
        let mut layer: Vec<_> = (0..max_height).map(|r| hash_row(tallest, r)).collect();
        layer.resize(padded_len(max_height, N), zero);
        let mut layers = vec![layer];
        let mut arities = Vec::new();

        while layers.last().unwrap().len() > 1 {
            let below = layers.last().unwrap();
            let arity = select_arity_step::<N>(
                below.len(),
                max_height.next_power_of_two(),
                rest.iter().map(|m| m.height()),
            );
            let computed = below.len() / arity;

            // Matrices whose padded height matches this layer are injected into it.
            let take = rest
                .iter()
                .take_while(|m| m.height().next_power_of_two() == computed.next_power_of_two())
                .count();
            let (inject, remaining) = rest.split_at(take);
            rest = remaining;

            let mut layer: Vec<_> = (0..computed)
                .map(|i| {
                    let folded = c.compress(array::from_fn(|n| {
                        if n < arity {
                            below[arity * i + n]
                        } else {
                            zero
                        }
                    }));
                    if inject.is_empty() {
                        return folded;
                    }
                    let row = if i < inject[0].height() {
                        hash_row(inject, i)
                    } else {
                        zero
                    };
                    c.compress(array::from_fn(|n| match n {
                        0 => folded,
                        1 => row,
                        _ => zero,
                    }))
                })
                .collect();
            layer.resize(padded_len(computed, N), zero);
            layers.push(layer);
            arities.push(arity);
        }
        (layers, arities)
    }

    /// Random matrices with the given heights, all of one width.
    fn matrices(heights: &[usize], width: usize) -> Vec<RowMajorMatrix<F>> {
        let mut rng =
            SmallRng::seed_from_u64(heights.iter().sum::<usize>() as u64 * 31 + width as u64);
        heights
            .iter()
            .map(|&height| RowMajorMatrix::rand(&mut rng, height, width))
            .collect()
    }

    /// Build with the tree and with the reference model, and compare every layer.
    ///
    /// Comparing whole layers pins where a divergence starts.
    fn assert_matches_reference<P, PW, H, C, const N: usize, const D: usize>(h: &H, c: &C)
    where
        P: PackedValue<Value = F>,
        PW: PackedValue<Value = u64>,
        H: CryptographicHasher<F, [u64; D]> + CryptographicHasher<P, [PW; D]> + Sync,
        C: PseudoCompressionFunction<[u64; D], N> + PseudoCompressionFunction<[PW; D], N> + Sync,
    {
        for &(heights, width) in SHAPES {
            let leaves = matrices(heights, width);
            let (expected, arities) = reference_layers::<H, C, N, D>(h, c, &leaves);
            let tree = MerkleTree::<F, u64, _, N, D>::new::<P, PW, H, C>(h, c, leaves);

            assert_eq!(tree.arity_schedule, arities, "heights {heights:?}");
            assert_eq!(
                tree.digest_layers.len(),
                expected.len(),
                "heights {heights:?}"
            );
            for (level, (got, want)) in tree.digest_layers.iter().zip(&expected).enumerate() {
                assert!(
                    got == want,
                    "layer {level} differs for heights {heights:?} width {width}"
                );
            }
        }
    }

    /// The same comparison over byte digests, for the byte-oriented hashers.
    fn assert_bytes_match_reference<H, C, const N: usize>(h: &H, c: &C)
    where
        H: CryptographicHasher<F, [u8; 32]> + Sync,
        C: PseudoCompressionFunction<[u8; 32], N> + Sync,
    {
        // Reuse the word model by viewing each digest through a byte-to-word adapter.
        let hw = Words(h);
        let cw = Words(c);
        for &(heights, width) in SHAPES {
            let leaves = matrices(heights, width);
            let (expected, arities) = reference_layers::<_, _, N, 4>(&hw, &cw, &leaves);
            let tree = MerkleTree::<F, u8, _, N, 32>::new::<F, u8, H, C>(h, c, leaves);

            assert_eq!(tree.arity_schedule, arities, "heights {heights:?}");
            for (level, (got, want)) in tree.digest_layers.iter().zip(&expected).enumerate() {
                let got: Vec<[u64; 4]> = got.iter().map(to_words).collect();
                assert!(
                    &got == want,
                    "layer {level} differs for heights {heights:?} width {width}"
                );
            }
        }
    }

    /// Adapter exposing a byte-digest primitive as a four-word digest primitive.
    struct Words<'a, T>(&'a T);

    impl<T> Clone for Words<'_, T> {
        fn clone(&self) -> Self {
            Self(self.0)
        }
    }

    /// A 32-byte digest read as four little-endian words.
    fn to_words(bytes: &[u8; 32]) -> [u64; 4] {
        let (words, _) = bytes.as_chunks::<8>();
        array::from_fn(|i| u64::from_le_bytes(words[i]))
    }

    /// Four little-endian words written back as a 32-byte digest.
    fn to_bytes(words: [u64; 4]) -> [u8; 32] {
        let mut bytes = [0u8; 32];
        for (chunk, word) in bytes.as_chunks_mut::<8>().0.iter_mut().zip(words) {
            *chunk = word.to_le_bytes();
        }
        bytes
    }

    impl<T: CryptographicHasher<F, [u8; 32]>> CryptographicHasher<F, [u64; 4]> for Words<'_, T> {
        fn hash_iter<I: IntoIterator<Item = F>>(&self, input: I) -> [u64; 4] {
            to_words(&self.0.hash_iter(input))
        }
    }

    impl<T: PseudoCompressionFunction<[u8; 32], N>, const N: usize>
        PseudoCompressionFunction<[u64; 4], N> for Words<'_, T>
    {
        fn compress(&self, input: [[u64; 4]; N]) -> [u64; 4] {
            to_words(&self.0.compress(input.map(to_bytes)))
        }
    }

    #[test]
    fn packed_poseidon2_matches_reference() {
        // Packed arm: one tree node per vector lane, sixteen lanes under AVX-512.
        //
        // The reference hashes the same sponge one row at a time.
        let mut rng = SmallRng::seed_from_u64(1);
        let perm16 = Poseidon2BabyBear::<16>::new_from_rng_128(&mut rng);
        let h = PaddingFreeSponge::<_, 16, 8, 8>::new(perm16.clone());

        // Binary tree: every step folds two children.
        let c2 = TruncatedPermutation::<_, 2, 8, 16>::new(perm16);
        assert_field_matches_reference::<_, _, 2>(&h, &c2);

        // Quaternary tree: full steps fold four, bridge steps fold two and pad the rest.
        let perm32 = Poseidon2BabyBear::<32>::new_from_rng_128(&mut rng);
        let c4 = TruncatedPermutation::<_, 4, 8, 32>::new(perm32);
        assert_field_matches_reference::<_, _, 4>(&h, &c4);
    }

    /// The reference comparison over field digests, for the packed field sponges.
    fn assert_field_matches_reference<H, C, const N: usize>(h: &H, c: &C)
    where
        H: CryptographicHasher<F, [F; 8]> + CryptographicHasher<Packed, [Packed; 8]> + Sync,
        C: PseudoCompressionFunction<[F; 8], N> + PseudoCompressionFunction<[Packed; 8], N> + Sync,
    {
        for &(heights, width) in SHAPES {
            let leaves = matrices(heights, width);
            let tree =
                MerkleTree::<F, F, _, N, 8>::new::<Packed, Packed, H, C>(h, c, leaves.clone());

            // One row per call on the scalar sponge, with the digest words widened to u64.
            let hw = FieldWords(h);
            let cw = FieldWords(c);
            let (expected, arities) = reference_layers::<_, _, N, 8>(&hw, &cw, &leaves);

            assert_eq!(tree.arity_schedule, arities, "heights {heights:?}");
            for (level, (got, want)) in tree.digest_layers.iter().zip(&expected).enumerate() {
                let got: Vec<[u64; 8]> = got
                    .iter()
                    .map(|d| d.map(|x| x.as_canonical_u64()))
                    .collect();
                assert!(
                    &got == want,
                    "layer {level} differs for heights {heights:?} width {width}"
                );
            }
        }
    }

    /// Adapter exposing a field-digest primitive as a word-digest primitive.
    struct FieldWords<'a, T>(&'a T);

    impl<T> Clone for FieldWords<'_, T> {
        fn clone(&self) -> Self {
            Self(self.0)
        }
    }

    impl<T: CryptographicHasher<F, [F; 8]>> CryptographicHasher<F, [u64; 8]> for FieldWords<'_, T> {
        fn hash_iter<I: IntoIterator<Item = F>>(&self, input: I) -> [u64; 8] {
            self.0.hash_iter(input).map(|x| x.as_canonical_u64())
        }
    }

    impl<T: PseudoCompressionFunction<[F; 8], N>, const N: usize>
        PseudoCompressionFunction<[u64; 8], N> for FieldWords<'_, T>
    {
        fn compress(&self, input: [[u64; 8]; N]) -> [u64; 8] {
            let input = input.map(|d| d.map(F::from_u64));
            self.0.compress(input).map(|x| x.as_canonical_u64())
        }
    }

    #[test]
    fn batched_keccak_matches_reference() {
        // Batched arm: several whole-row messages per hash call.
        let h = SerializingHasher::new(Keccak256Hash);
        assert_bytes_match_reference::<_, _, 2>(
            &h,
            &CompressionFunctionFromHasher::<_, 2, 32>::new(Keccak256Hash),
        );
        assert_bytes_match_reference::<_, _, 4>(
            &h,
            &CompressionFunctionFromHasher::<_, 4, 32>::new(Keccak256Hash),
        );
    }

    #[test]
    fn packed_keccak_words_match_reference() {
        // Packed arm over u64 lanes: a Keccak state per lane, BabyBear rows serialized in pairs.
        type Sponge = PaddingFreeSponge<KeccakF, 25, 17, 4>;
        let sponge = Sponge::new(KeccakF);
        let h = SerializingHasher::new(sponge);
        let c = CompressionFunctionFromHasher::<_, 2, 4>::new(sponge);
        assert_matches_reference::<[F; VECTOR_LEN], [u64; VECTOR_LEN], _, _, 2, 4>(&h, &c);
    }

    #[test]
    fn pass_shape_keeps_every_block_busy() {
        // Invariant: every pass but the last single-block one gives each block its minimum.
        //
        // Invariant: consecutive passes tile the layers bottom to top, with no gap.
        let mut rng = SmallRng::seed_from_u64(5);
        let perm = Poseidon2BabyBear::<16>::new_from_rng_128(&mut rng);
        type Sponge = PaddingFreeSponge<Poseidon2BabyBear<16>, 16, 8, 8>;
        type Compress = TruncatedPermutation<Poseidon2BabyBear<16>, 2, 8, 16>;
        type Builder<'a> =
            TreeBuilder<'a, Packed, Packed, Sponge, Compress, RowMajorMatrix<F>, 2, 8>;
        let h = Sponge::new(perm.clone());
        let c = Compress::new(perm);
        let group = Builder::group();

        for height in [1usize, 2, 63, 64, 65, 1000, 4096, 100_000] {
            let leaves = [RowMajorMatrix::<F>::new(vec![F::ZERO; height], 1)];
            let builder = Builder::new(&h, &c, &leaves);

            let mut lo = 0;
            while lo < builder.layers.len() {
                let (blocks, hi) = builder.pass_shape(lo);
                assert!((lo..builder.layers.len()).contains(&hi), "height {height}");
                if blocks > 1 {
                    assert!(blocks <= BLOCKS_PER_THREAD * current_num_threads());
                    assert!(blocks >= current_num_threads(), "height {height}");
                    assert!(builder.layers[hi].computed >= blocks * group * MIN_BLOCK_GROUPS);
                } else {
                    // A single block always finishes the tree.
                    assert_eq!(hi, builder.layers.len() - 1, "height {height}");
                }
                lo = hi + 1;
            }
            assert_eq!(lo, builder.layers.len());
        }
    }

    proptest! {
        #[test]
        fn random_ladders_match_reference(
            max_height in 1usize..3000,
            depths in collection::vec(0u32..12, 0..4),
            width in 1usize..12,
        ) {
            // Any heights on the ceil(max / 2^k) ladder form a valid tree.
            //
            // Depth 0 repeats the tallest height, so a leaf can span several matrices.
            let mut heights = vec![max_height];
            heights.extend(depths.iter().map(|&k| max_height.div_ceil(1 << k)));
            let leaves = matrices(&heights, width);

            let h = SerializingHasher::new(Keccak256Hash);
            let c = CompressionFunctionFromHasher::<_, 2, 32>::new(Keccak256Hash);
            let (expected, _) = reference_layers::<_, _, 2, 4>(&Words(&h), &Words(&c), &leaves);
            let tree = MerkleTree::<F, u8, _, 2, 32>::new::<F, u8, _, _>(&h, &c, leaves);

            // Compare every layer, viewed as words like the reference.
            let got: Vec<Vec<[u64; 4]>> = tree
                .digest_layers
                .iter()
                .map(|layer| layer.iter().map(to_words).collect())
                .collect();
            prop_assert!(got == expected);
        }
    }

    #[test]
    fn default_digest_layer_is_all_default() {
        // Byte digests: the word type is a primitive, so the allocation takes the zeroed path.
        //
        // The reinterpretation back to digests must still read as the default digest.
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
    fn a_staged_group_is_lane_aligned_and_bounded() {
        // Invariant: a group is a whole number of lane groups.
        //
        // Invariant: it stages at most `max(target, lanes * row_bytes)` bytes.
        //
        // The second bound is the price of the first.
        // A row at or past the target still needs one full lane group staged.
        //
        // Both assertions have teeth, checked by mutating the sizing:
        //
        //     drop the lane rounding   ->  lanes 2, 16383 B: 1 row, and 1 % 2 != 0
        //     round the target up      ->  one lane, 8191 B: 3 rows, 24573 bytes staged
        //
        // The second is why the sweep carries a width just under half the target.
        for lanes in [0usize, 1, 2, 4, 8, 16] {
            for row_bytes in [
                0usize,
                1,
                64,
                ROW_SCRATCH_BYTES / 8,
                ROW_SCRATCH_BYTES / 2 - 1,
                ROW_SCRATCH_BYTES - 1,
                ROW_SCRATCH_BYTES,
                ROW_SCRATCH_BYTES + 1,
                4 * ROW_SCRATCH_BYTES,
            ] {
                let rows = rows_per_call(row_bytes, lanes);

                // Zero on either argument is read as one, so both divisions stay defined.
                let (effective_lanes, effective_bytes) = (lanes.max(1), row_bytes.max(1));

                assert!(rows >= 1, "lanes={lanes} row_bytes={row_bytes}");
                assert_eq!(
                    rows % effective_lanes,
                    0,
                    "lanes={lanes} row_bytes={row_bytes} rows={rows}"
                );
                assert!(
                    rows * effective_bytes
                        <= ROW_SCRATCH_BYTES.max(effective_lanes * effective_bytes),
                    "lanes={lanes} row_bytes={row_bytes} staged {} bytes",
                    rows * effective_bytes
                );
            }
        }

        // Narrow rows: many whole lane groups fit and the target holds.
        //
        //     16 KiB / 64 B = 256 rows, over 8 lanes = 32 whole groups
        assert_eq!(rows_per_call(64, 8), 256);
        assert!(rows_per_call(64, 8) * 64 <= ROW_SCRATCH_BYTES);

        // A row past the target still stages one whole lane group.
        assert_eq!(rows_per_call(ROW_SCRATCH_BYTES + 1, 8), 8);
    }

    #[test]
    fn the_staged_buffer_meets_the_bound_the_target_states() {
        // Invariant: the bound is stated in bytes, and the buffer is sized in field elements.
        //
        // The batched driver composes both from the same width, so the two must line up:
        //
        //     row bytes    = total width * element bytes
        //     buffer bytes = rows per group * total width * element bytes
        //
        // Checking the helper alone would miss a call site that sized its buffer some other way.
        const ELEMENT: usize = size_of::<BabyBear>();

        for lanes in [1usize, 2, 4, 8] {
            // Widths straddling the target: 4096 four-byte columns are exactly 16 KiB.
            for total_width in [0usize, 1, 16, 1024, 4095, 4096, 4097, 16384] {
                let row_bytes = total_width * ELEMENT;
                let rows = rows_per_call(row_bytes, lanes);

                // Exactly what the batched driver reserves, converted to bytes.
                let staged = rows * total_width * ELEMENT;

                // Zero-width matrices stage nothing, whatever count the sizing hands back.
                if total_width == 0 {
                    assert_eq!(staged, 0, "lanes={lanes}");
                    continue;
                }

                assert!(
                    staged <= ROW_SCRATCH_BYTES.max(lanes * row_bytes),
                    "lanes={lanes} total_width={total_width} staged {staged} bytes"
                );
            }
        }
    }
}
