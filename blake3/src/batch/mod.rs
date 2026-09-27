//! Hashing many equal-length messages at once.
//!
//! Equal lengths give every message the same chunks, blocks, counters and flags.
//!
//! So one compression advances a whole group, one vector lane per message.
//!
//! The tree over the chunks has the same shape in every lane too.
//!
//! Parent nodes therefore run in lockstep as well, straight from the registers.

mod compress;
mod lanes;

use blake3::{BLOCK_LEN, CHUNK_LEN, OUT_LEN};

pub(crate) use self::compress::IV;
use self::compress::{BLOCK_WORDS, CHUNK_END, CHUNK_START, PARENT, ROOT, STATE_WORDS, compress};
use self::lanes::{GROUPS, Vector, Word, load_block, store_digests};
pub(crate) use self::lanes::{LANES, WIDTH};

/// Chaining values of every lane of `G` register groups.
type State<const G: usize> = [[Vector; STATE_WORDS]; G];

/// Message words of every lane of `G` register groups.
type Block<const G: usize> = [[Vector; BLOCK_WORDS]; G];

/// Where the message of every lane of `G` register groups starts in the batch.
struct Lanes<'a, const G: usize> {
    /// Every message of the batch, back to back.
    batch: &'a [u8],
    /// Byte offset of each lane's message in `batch`.
    starts: [[usize; WIDTH]; G],
    /// The largest start, so one bounds check covers the same read in every lane.
    last: usize,
}

impl<const G: usize> Lanes<'_, G> {
    /// Load the block at byte `offset` of every lane's message.
    ///
    /// # Panics
    ///
    /// Panics if a lane has fewer than `offset + BLOCK_LEN` bytes left in the batch.
    #[inline(always)]
    fn load(&self, offset: usize) -> Block<G> {
        // Every lane starts at or before the last one, so this check covers them all.
        assert!(self.last + offset + BLOCK_LEN <= self.batch.len());
        let mut block = [[Vector::splat(0); BLOCK_WORDS]; G];
        for (words, starts) in block.iter_mut().zip(&self.starts) {
            let rows = core::array::from_fn(|l| {
                let at = starts[l] + offset;
                // SAFETY: `starts[l] <= last`, so the assertion bounds this read too.
                unsafe { &*self.batch.as_ptr().add(at).cast::<[u8; BLOCK_LEN]>() }
            });
            *words = load_block(&rows);
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
    fn state<const G: usize>(self) -> State<G> {
        [self.key.map(Vector::splat); G]
    }
}

/// Hash equal-length messages of `len` bytes laid end to end in `input`.
///
/// The caller guarantees `input.len() == len * out.len()`.
pub(crate) fn hash_many(mode: Mode, input: &[u8], len: usize, out: &mut [[u8; OUT_LEN]]) {
    debug_assert_eq!(input.len(), len * out.len());

    // Full register groups first, then the single registers left over.
    let (registers, rest) = out.as_chunks_mut::<WIDTH>();
    let (groups, singles) = registers.as_chunks_mut::<GROUPS>();
    for (index, digests) in groups.iter_mut().enumerate() {
        let first = index * LANES;
        let starts =
            core::array::from_fn(|g| core::array::from_fn(|l| (first + g * WIDTH + l) * len));
        let lanes = Lanes {
            batch: input,
            starts,
            last: (first + LANES - 1) * len,
        };
        hash_group::<GROUPS>(mode, &lanes, len, digests);
    }

    let mut first = groups.len() * LANES;
    for digests in singles {
        let lanes = Lanes {
            batch: input,
            starts: [core::array::from_fn(|l| (first + l) * len)],
            last: (first + WIDTH - 1) * len,
        };
        hash_group::<1>(mode, &lanes, len, core::array::from_mut(digests));
        first += WIDTH;
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
        let mut digests = [[[0u8; OUT_LEN]; WIDTH]];
        hash_group::<1>(mode, &lanes, len, &mut digests);
        rest.copy_from_slice(&digests[0][..rest.len()]);
    }
}

/// Hash the `len`-byte message of every lane.
fn hash_group<const G: usize>(
    mode: Mode,
    lanes: &Lanes<'_, G>,
    len: usize,
    out: &mut [[[u8; OUT_LEN]; WIDTH]; G],
) {
    // One chunk is its own root, and more chunks fold into a tree.
    let state = if len <= CHUNK_LEN {
        chunk(mode, lanes, len, 0, ROOT)
    } else {
        subtree(mode, lanes, len, 0, len.div_ceil(CHUNK_LEN), ROOT)
    };

    for (state, out) in state.iter().zip(out) {
        store_digests(state, out);
    }
}

/// The chaining value of `chunks` consecutive chunks, starting at chunk `first`.
///
/// The left child takes the largest power of two chunks that leaves the right one non-empty.
///
/// For 5 chunks, the root joins chunks 0 to 3 with chunk 4.
///
/// `root` is `ROOT` for the top of the tree and zero for every node below it.
fn subtree<const G: usize>(
    mode: Mode,
    lanes: &Lanes<'_, G>,
    len: usize,
    first: usize,
    chunks: usize,
    root: u32,
) -> State<G> {
    if chunks == 1 {
        return leaf(mode, lanes, len, first, root);
    }

    // Largest power of two strictly below `chunks`.
    let left_chunks = 1 << (usize::BITS - 1 - (chunks - 1).leading_zeros());
    let left = subtree(mode, lanes, len, first, left_chunks, 0);
    let right = subtree(
        mode,
        lanes,
        len,
        first + left_chunks,
        chunks - left_chunks,
        0,
    );

    parent(mode, &left, &right, root)
}

/// A chunk of the tree, kept out of line so the recursion in [`subtree`] carries small frames.
#[inline(never)]
fn leaf<const G: usize>(
    mode: Mode,
    lanes: &Lanes<'_, G>,
    len: usize,
    index: usize,
    root: u32,
) -> State<G> {
    chunk(mode, lanes, len, index, root)
}

/// The chaining value of a parent node over two children.
///
/// Kept out of line, like [`leaf`], so the recursion in [`subtree`] carries small frames.
#[inline(never)]
fn parent<const G: usize>(mode: Mode, left: &State<G>, right: &State<G>, root: u32) -> State<G> {
    // A parent block is the left chaining value, then the right one.
    //
    // Both already sit in lane order, so no transpose is needed.
    let mut block = [[Vector::splat(0); BLOCK_WORDS]; G];
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
fn chunk<const G: usize>(
    mode: Mode,
    lanes: &Lanes<'_, G>,
    len: usize,
    index: usize,
    root: u32,
) -> State<G> {
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
        let words = lanes.load(start + block * BLOCK_LEN);
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
    let words = last_block(lanes, offset, tail);

    let first = if plain_blocks == 0 { CHUNK_START } else { 0 };
    let flags = mode.flags | first | CHUNK_END | root;
    compress(&mut state, &words, counter, tail as u32, flags);
    state
}

/// Load the last block of a chunk: `tail` bytes at `offset`, zero padded to a full block.
#[inline(always)]
fn last_block<const G: usize>(lanes: &Lanes<'_, G>, offset: usize, tail: usize) -> Block<G> {
    // Read 64 bytes in place when every lane reaches that far, then mask.
    //
    // Only the last group of a batch falls short, and copies into a zero-padded block.
    let mut words = if lanes.last + offset + BLOCK_LEN <= lanes.batch.len() {
        lanes.load(offset)
    } else {
        spilled_block(lanes, offset, tail)
    };

    // Zero every byte past the message end.
    //
    // All lanes share the tail length, so one mask per word serves every lane.
    //
    // For a tail of 6 bytes, word 0 keeps 4 bytes, word 1 keeps 2, and words 2 to 15 keep none.
    if tail < BLOCK_LEN {
        for w in tail / 4..BLOCK_WORDS {
            let kept_bytes = tail.saturating_sub(4 * w);
            let mask = Vector::splat(!(u32::MAX << (8 * kept_bytes)));
            for group in &mut words {
                group[w] = group[w].and(mask);
            }
        }
    }
    words
}

/// Load `tail` bytes at `offset` of every lane through a zero-padded copy.
#[cold]
#[inline(never)]
fn spilled_block<const G: usize>(lanes: &Lanes<'_, G>, offset: usize, tail: usize) -> Block<G> {
    let mut block = [[Vector::splat(0); BLOCK_WORDS]; G];
    for (words, starts) in block.iter_mut().zip(&lanes.starts) {
        let mut spill = [[0u8; BLOCK_LEN]; WIDTH];
        for (spill, &start) in spill.iter_mut().zip(starts) {
            spill[..tail].copy_from_slice(&lanes.batch[start + offset..][..tail]);
        }
        *words = load_block(&spill.each_ref());
    }
    block
}
