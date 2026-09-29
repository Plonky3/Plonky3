//! Hashing many equal-length messages at once.
//!
//! Equal lengths give every message the same chunks, blocks, counters and flags.
//!
//! So one compression advances a whole group, one vector lane per message.
//!
//! The tree over the chunks has the same shape in every lane too.
//!
//! Parent nodes therefore run in lockstep as well, straight from the registers.
//!
//! Messages short of a full group can instead spread their chunks across the lanes.

mod compress;
mod lanes;
mod spread;

use blake3::{BLOCK_LEN, CHUNK_LEN, OUT_LEN};

pub(crate) use self::compress::IV;
use self::compress::{BLOCK_WORDS, CHUNK_END, CHUNK_START, PARENT, ROOT, STATE_WORDS, compress};
use self::lanes::{Backend, Word};
#[cfg(test)]
pub(crate) use self::lanes::{Kernel, supported};
pub(crate) use self::lanes::{LANES, detect};

/// Chaining values of every lane of `G` register groups of `V`.
type State<V, const G: usize> = [[V; STATE_WORDS]; G];

/// Message words of every lane of `G` register groups of `V`.
type Block<V, const G: usize> = [[V; BLOCK_WORDS]; G];

/// Where the message of every lane of `G` register groups of `W` lanes starts in the batch.
struct Lanes<'a, const W: usize, const G: usize> {
    /// Every message of the batch, back to back.
    batch: &'a [u8],
    /// Byte offset of each lane's message in `batch`.
    starts: [[usize; W]; G],
    /// The largest start, so one bounds check covers the same read in every lane.
    last: usize,
}

impl<const W: usize, const G: usize> Lanes<'_, W, G> {
    /// Load the block at byte `offset` of every lane's message.
    ///
    /// # Panics
    ///
    /// Panics if a lane has fewer than `offset + BLOCK_LEN` bytes left in the batch.
    #[inline(always)]
    fn load<V: Backend<W>>(&self, offset: usize) -> Block<V, G> {
        // Every lane starts at or before the last one, so this check covers them all.
        assert!(self.last + offset + BLOCK_LEN <= self.batch.len());
        let mut block = [[V::splat(0); BLOCK_WORDS]; G];
        for (words, starts) in block.iter_mut().zip(&self.starts) {
            let rows = core::array::from_fn(|l| {
                let at = starts[l] + offset;
                // SAFETY: `starts[l] <= last`, so the assertion bounds this read too.
                unsafe { &*self.batch.as_ptr().add(at).cast::<[u8; BLOCK_LEN]>() }
            });
            *words = V::load_block(&rows);
        }
        block
    }
}

/// The key and domain flags every compression of a hash carries.
///
/// Plain hashing is the initialization vector with no flags.
///
/// The keyed and key-derivation modes only change these two values.
#[derive(Clone, Copy)]
pub(crate) struct Mode {
    /// The key words, which every chunk and parent starts from.
    pub(crate) key: [u32; STATE_WORDS],
    /// Flags added to every compression.
    pub(crate) flags: u32,
}

impl Mode {
    /// Plain unkeyed hashing.
    pub(crate) const HASH: Self = Self { key: IV, flags: 0 };

    /// The key in every lane, where every chunk and parent starts.
    #[inline(always)]
    fn state<V: Word, const G: usize>(self) -> State<V, G> {
        [self.key.map(V::splat); G]
    }
}

/// Hash equal-length messages with backend `V`, in groups of `G` registers of `W` lanes.
///
/// The caller guarantees `input.len() == len * out.len()`.
///
/// # Safety
///
/// The running CPU has the target features of `V`.
unsafe fn hash_many_with<V: Backend<W>, const W: usize, const G: usize>(
    mode: Mode,
    input: &[u8],
    len: usize,
    out: &mut [[u8; OUT_LEN]],
) {
    debug_assert_eq!(input.len(), len * out.len());

    // Full register groups first, one lane per message.
    let full = out.len() / (W * G) * (W * G);
    let (grouped, out) = out.split_at_mut(full);
    let (groups, _) = grouped.as_chunks_mut::<W>().0.as_chunks_mut::<G>();
    for (index, digests) in groups.iter_mut().enumerate() {
        let first = index * W * G;
        let starts = core::array::from_fn(|g| core::array::from_fn(|l| (first + g * W + l) * len));
        let lanes = Lanes {
            batch: input,
            starts,
            last: (first + W * G - 1) * len,
        };
        // SAFETY: the caller runs this on a CPU with the features of `V`.
        unsafe { V::hash_group::<G>(mode, &lanes, len, digests) };
    }

    // The messages short of a group leave lanes idle in lockstep.
    //
    // Long enough messages fill those lanes with their own chunks instead.
    if spread::pays::<W, G>(out.len(), len, V::LONE_REGISTER_COST) {
        // SAFETY: the caller runs this on a CPU with the features of `V`.
        unsafe { V::spread::<G>(mode, &input[full * len..], len, out) };
        return;
    }

    // Otherwise single registers, then one short register.
    let (singles, rest) = out.as_chunks_mut::<W>();
    let mut first = full;
    for digests in singles {
        let lanes = Lanes {
            batch: input,
            starts: [core::array::from_fn(|l| (first + l) * len)],
            last: (first + W - 1) * len,
        };
        // SAFETY: the caller runs this on a CPU with the features of `V`.
        unsafe { V::hash_group::<1>(mode, &lanes, len, core::array::from_mut(digests)) };
        first += W;
    }

    // A short final register repeats its messages to fill the spare lanes.
    //
    // Those lanes compute digests that are never written out.
    if !rest.is_empty() {
        let lanes = Lanes {
            batch: input,
            starts: [core::array::from_fn(|l| (first + l % rest.len()) * len)],
            last: (first + rest.len() - 1) * len,
        };
        let mut digests = [[[0u8; OUT_LEN]; W]];
        // SAFETY: the caller runs this on a CPU with the features of `V`.
        unsafe { V::hash_group::<1>(mode, &lanes, len, &mut digests) };
        rest.copy_from_slice(&digests[0][..rest.len()]);
    }
}

/// Hash the `len`-byte message of every lane.
///
/// # Safety
///
/// The running CPU has the target features of `V`.
#[inline(always)]
unsafe fn hash_group<V: Backend<W>, const W: usize, const G: usize>(
    mode: Mode,
    lanes: &Lanes<'_, W, G>,
    len: usize,
    out: &mut [[[u8; OUT_LEN]; W]; G],
) {
    // One chunk is its own root, and more chunks fold into a tree.
    let state = if len <= CHUNK_LEN {
        chunk::<V, W, G>(mode, lanes, len, 0, ROOT)
    } else {
        // SAFETY: the caller runs this on a CPU with the features of `V`.
        unsafe { V::subtree(mode, lanes, len, 0, len.div_ceil(CHUNK_LEN), ROOT) }
    };

    for (state, out) in state.iter().zip(out) {
        V::store_digests(state, out);
    }
}

/// The chaining value of `chunks` consecutive chunks, starting at chunk `first`.
///
/// The left child takes the largest power of two chunks that leaves the right one non-empty.
///
/// For 5 chunks, the root joins chunks 0 to 3 with chunk 4.
///
/// `root` is `ROOT` for the top of the tree and zero for every node below it.
///
/// # Safety
///
/// The running CPU has the target features of `V`.
#[inline(always)]
unsafe fn subtree<V: Backend<W>, const W: usize, const G: usize>(
    mode: Mode,
    lanes: &Lanes<'_, W, G>,
    len: usize,
    first: usize,
    chunks: usize,
    root: u32,
) -> State<V, G> {
    // SAFETY (every call below): the caller runs this on a CPU with the features of `V`.
    if chunks == 1 {
        return unsafe { V::leaf(mode, lanes, len, first, root) };
    }

    // Largest power of two strictly below `chunks`.
    let left_chunks = 1 << (usize::BITS - 1 - (chunks - 1).leading_zeros());
    let left = unsafe { V::subtree(mode, lanes, len, first, left_chunks, 0) };
    let right = unsafe {
        V::subtree(
            mode,
            lanes,
            len,
            first + left_chunks,
            chunks - left_chunks,
            0,
        )
    };

    unsafe { V::parent(mode, &left, &right, root) }
}

/// The chaining value of a parent node over two children.
#[inline(always)]
fn parent<V: Word, const G: usize>(
    mode: Mode,
    left: &State<V, G>,
    right: &State<V, G>,
    root: u32,
) -> State<V, G> {
    // A parent block is the left chaining value, then the right one.
    //
    // Both already sit in lane order, so no transpose is needed.
    let mut block = [[V::splat(0); BLOCK_WORDS]; G];
    for ((block, left), right) in block.iter_mut().zip(left).zip(right) {
        block[..STATE_WORDS].copy_from_slice(left);
        block[STATE_WORDS..].copy_from_slice(right);
    }

    // Parents always have a full block and a zero counter.
    let mut state = mode.state();
    compress(
        &mut state,
        &block,
        0,
        BLOCK_LEN as u32,
        mode.flags | PARENT | root,
    );
    state
}

/// The chaining value of chunk `index` in every lane.
///
/// `root` is `ROOT` when this chunk is the whole message.
#[inline(always)]
fn chunk<V: Backend<W>, const W: usize, const G: usize>(
    mode: Mode,
    lanes: &Lanes<'_, W, G>,
    len: usize,
    index: usize,
    root: u32,
) -> State<V, G> {
    let start = index * CHUNK_LEN;
    let end = len.min(start + CHUNK_LEN);

    // The last block is flagged, so it is held back even when full.
    //
    // - 130 bytes: two plain blocks, then a last block of 2 bytes.
    // - 128 bytes: one plain block, then a last block of 64 bytes.
    // - 0 bytes: one last block of no bytes at all.
    let plain_blocks = (end - start).saturating_sub(1) / BLOCK_LEN;

    // Every block of a chunk carries the chunk index as its counter.
    let counter = index as u64;
    let mut state = mode.state();
    for block in 0..plain_blocks {
        let words = lanes.load::<V>(start + block * BLOCK_LEN);
        let flags = if block == 0 { CHUNK_START } else { 0 };
        compress(
            &mut state,
            &words,
            counter,
            BLOCK_LEN as u32,
            mode.flags | flags,
        );
    }

    // The last block holds up to 64 message bytes, zero padded.
    let offset = start + plain_blocks * BLOCK_LEN;
    let tail = end - offset;
    let words = last_block::<V, W, G>(lanes, offset, tail);

    let first = if plain_blocks == 0 { CHUNK_START } else { 0 };
    let flags = mode.flags | first | CHUNK_END | root;
    compress(&mut state, &words, counter, tail as u32, flags);
    state
}

/// Load the last block of a chunk: `tail` bytes at `offset`, zero padded to a full block.
#[inline(always)]
fn last_block<V: Backend<W>, const W: usize, const G: usize>(
    lanes: &Lanes<'_, W, G>,
    offset: usize,
    tail: usize,
) -> Block<V, G> {
    // Read 64 bytes in place when every lane reaches that far, then mask.
    //
    // Only the last group of a batch falls short, and copies into a zero-padded block.
    let mut words = if lanes.last + offset + BLOCK_LEN <= lanes.batch.len() {
        lanes.load::<V>(offset)
    } else {
        spilled_block::<V, W, G>(lanes, offset, tail)
    };

    // Zero every byte past the message end.
    //
    // All lanes share the tail length, so one mask per word serves every lane.
    //
    // For a tail of 6 bytes, word 0 keeps 4 bytes, word 1 keeps 2, and words 2 to 15 keep none.
    if tail < BLOCK_LEN {
        for w in tail / 4..BLOCK_WORDS {
            let kept_bytes = tail.saturating_sub(4 * w);
            let mask = V::splat(!(u32::MAX << (8 * kept_bytes)));
            for group in &mut words {
                group[w] = group[w].and(mask);
            }
        }
    }
    words
}

/// Load `tail` bytes at `offset` of every lane through a zero-padded copy.
///
/// At most a few blocks of a batch come this way, so it runs without the backend's target features.
#[cold]
#[inline(never)]
fn spilled_block<V: Backend<W>, const W: usize, const G: usize>(
    lanes: &Lanes<'_, W, G>,
    offset: usize,
    tail: usize,
) -> Block<V, G> {
    let mut block = [[V::splat(0); BLOCK_WORDS]; G];
    for (words, starts) in block.iter_mut().zip(&lanes.starts) {
        let mut spill = [[0u8; BLOCK_LEN]; W];
        for (spill, &start) in spill.iter_mut().zip(starts) {
            spill[..tail].copy_from_slice(&lanes.batch[start + offset..][..tail]);
        }
        *words = V::load_block(&spill.each_ref());
    }
    block
}
