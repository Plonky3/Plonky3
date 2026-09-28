//! Hashing a few long messages by spreading their chunks across the lanes.
//!
//! Lockstep hashing gives every message one lane.
//!
//! With fewer messages than lanes, the spare lanes compute nothing useful.
//!
//! The chunks of one message are independent, so they can fill those lanes instead:
//!
//! ```text
//!     2 messages of 64 chunks, 32 lanes
//!
//!     lockstep:   2 lanes busy, 30 idle, for 64 chunks in a row
//!     spread:    32 lanes busy,  0 idle, for  4 chunks in a row
//! ```
//!
//! The chunk values then fold into each message's tree, many nodes per compression again.

use core::marker::PhantomData;

use blake3::{BLOCK_LEN, CHUNK_LEN, OUT_LEN};

use super::compress::{CHUNK_END, CHUNK_START, PARENT, ROOT, compress, compress_counters};
use super::lanes::{Backend, from_lanes};
use super::{Lanes, Mode, State, chunk};

/// Blocks in one chunk.
const CHUNK_BLOCKS: usize = CHUNK_LEN / BLOCK_LEN;

/// Chunk values one window holds.
///
/// 256 values take 8 KiB, which stays in the first-level cache next to the message blocks.
const WINDOW: usize = 256;

/// The most messages spread at once.
///
/// Spreading only takes the messages short of a full group, and no backend has more than 32 lanes.
const MAX_MESSAGES: usize = 32;

/// Pieces one message splits into: one per bit of a chunk count.
const MAX_PIECES: usize = usize::BITS as usize;

/// How a piece joins the fold that ends at the root.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Fold {
    /// The smallest subtree, whose value starts the fold.
    Seed,
    /// A subtree folded in below the root.
    Inner,
    /// The largest subtree, whose fold step is the root.
    Root,
    /// The whole message, whose top parent is the root.
    Whole,
}

/// A complete subtree of one message: `size` chunks from chunk `base`, a power of two.
#[derive(Clone, Copy)]
struct Piece {
    /// Index of the first chunk.
    base: usize,
    /// Chunks in the subtree.
    size: usize,
    /// How its value joins the root.
    fold: Fold,
}

impl Piece {
    /// The widest subtree a window builds for this piece.
    ///
    /// The top parent of a whole message is the root, so it runs with the fold.
    const fn reach(self) -> usize {
        match self.fold {
            Fold::Whole => self.size / 2,
            _ => self.size,
        }
    }
}

/// One piece of one message, placed in a window.
#[derive(Clone, Copy)]
struct Unit {
    /// Index of the message in the batch.
    message: usize,
    /// The piece of that message.
    piece: Piece,
}

/// The pieces every message of the batch splits into, smallest first.
///
/// The tree of the specification gives its left child the largest power of two of chunks below the whole.
///
/// So it is a fold, from the right, of complete subtrees that follow the set bits of the chunk count:
///
/// ```text
///     6 full chunks = 4 + 2:        root = P( T(c0 c1 c2 c3), T(c4 c5) )
///     6 chunks, the last one short: root = P( T(c0 c1 c2 c3), P( c4, c5 ) )
/// ```
///
/// Full chunks share one shape, so all of them join the subtrees.
///
/// A short last chunk has a shape of its own.
///
/// It is hashed apart and starts the fold, and the other `n - 1` chunks split by their own bits.
struct Plan {
    /// The pieces, smallest first, which is the order they fold in.
    pieces: [Piece; MAX_PIECES],
    /// Number of pieces.
    len: usize,
}

impl Plan {
    /// Split a message of `chunks` chunks, the last of them full when `last_full`.
    fn new(chunks: usize, last_full: bool) -> Self {
        // The chunks that split into subtrees.
        let tree = if last_full { chunks } else { chunks - 1 };
        let top = usize::BITS - 1 - tree.leading_zeros();
        let whole = last_full && tree.is_power_of_two();

        let mut pieces = [Piece {
            base: 0,
            size: 0,
            fold: Fold::Seed,
        }; MAX_PIECES];
        let mut len = 0;

        // One subtree per set bit, smallest first.
        for bit in (0..=top).filter(|&b| tree >> b & 1 == 1) {
            // The larger subtrees sit before this one, so it starts where they end.
            //
            // For 5 = 101b: the 1-chunk subtree starts at chunk 4, the 4-chunk one at chunk 0.
            let base = tree >> bit >> 1 << bit << 1;

            // Without a short last chunk, the smallest subtree starts the fold.
            let fold = match () {
                () if whole => Fold::Whole,
                () if bit == top => Fold::Root,
                () if last_full && len == 0 => Fold::Seed,
                () => Fold::Inner,
            };
            pieces[len] = Piece {
                base,
                size: 1 << bit,
                fold,
            };
            len += 1;
        }
        Self { pieces, len }
    }

    /// The pieces, smallest first.
    fn pieces(&self) -> &[Piece] {
        &self.pieces[..self.len]
    }
}

/// Whether spreading `count` messages of `len` bytes beats hashing them in lockstep.
///
/// Both costs count compressions of one register, the unit both strategies share.
///
/// Lockstep runs every block of a message once per register of messages.
///
/// Spreading runs every chunk once per lane, then the parents level by level, then the fold.
pub(super) fn pays<const W: usize, const G: usize>(count: usize, len: usize) -> bool {
    let chunks = len.div_ceil(CHUNK_LEN);
    if chunks < 2 || count == 0 || count > MAX_MESSAGES {
        return false;
    }

    // Registers one pass over `jobs` lanes takes: a single register, or whole padded groups.
    let registers = |jobs: usize| {
        if jobs <= W {
            jobs.min(1)
        } else {
            jobs.div_ceil(W * G) * G
        }
    };

    // The last chunk holds 1 to 1024 bytes, and at least one block.
    let head = chunks - 1;
    let tail_blocks = (len - head * CHUNK_LEN).div_ceil(BLOCK_LEN);
    let last_full = len.is_multiple_of(CHUNK_LEN);

    // Lockstep: every block of a message, plus one parent per chunk but one.
    let lockstep = count.div_ceil(W) * (head * CHUNK_BLOCKS + tail_blocks + head);

    // Spreading, phase 1: the chunks of the subtrees in one pass, a short last chunk apart.
    let tree = if last_full { chunks } else { head };
    let mut spread = registers(count * tree) * CHUNK_BLOCKS;
    if !last_full {
        spread += registers(count) * tail_blocks;
    }

    // Phase 2: level `l` of the subtrees holds `tree >> l` parents per message.
    for level in 1..usize::BITS {
        let jobs = count * (tree >> level);
        if jobs == 0 {
            break;
        }
        spread += registers(jobs);
    }

    // Phase 3: one fold step per subtree but the seed, all messages together.
    let steps = tree.count_ones() as usize - usize::from(last_full);
    spread += steps * registers(count);

    spread < lockstep
}

/// Hash equal-length messages of `len` bytes, spreading their chunks across the lanes.
///
/// The caller guarantees:
///
/// - `input.len() == len * out.len()`;
/// - every message spans at least two chunks;
/// - there are at most 32 messages.
///
/// # Safety
///
/// The running CPU has the target features of `V`.
#[inline(always)]
pub(super) unsafe fn hash<V: Backend<W>, const W: usize, const G: usize>(
    mode: Mode,
    input: &[u8],
    len: usize,
    out: &mut [[u8; OUT_LEN]],
) {
    let count = out.len();
    let chunks = len.div_ceil(CHUNK_LEN);
    let last_full = len.is_multiple_of(CHUNK_LEN);
    assert!(chunks >= 2 && count <= MAX_MESSAGES);
    debug_assert_eq!(input.len(), len * count);

    let plan = Plan::new(chunks, last_full);
    let mut spreader = Spreader::<V, W, G> {
        mode,
        input,
        len,
        count,
        scratch: [[0; OUT_LEN]; WINDOW],
        fold: [[0; OUT_LEN]; MAX_MESSAGES],
        blocks: [[0; 2 * OUT_LEN]; MAX_MESSAGES],
        _backend: PhantomData,
    };

    // Phase 1: a short last chunk has a shape of its own, so it runs apart, one lane per message.
    if !last_full {
        spreader.last_chunks(chunks - 1);
    }

    // Phase 2: pieces that fit a window, packed together so their chunks fill the lanes.
    //
    // Unit u is piece u / count of message u % count: every message of a piece, then the next piece.
    let pieces = plan.pieces();
    let small = pieces.partition_point(|p| p.size <= WINDOW);
    let unit = |u: usize| Unit {
        message: u % count,
        piece: pieces[u / count],
    };
    let units = small * count;
    let mut first = 0;
    while first < units {
        // Take units while their chunks fit the window.
        let mut end = first;
        let mut filled = 0;
        while end < units && filled + unit(end).piece.size <= WINDOW {
            filled += unit(end).piece.size;
            end += 1;
        }
        let window = (first..end).map(unit);
        spreader.window(&window);
        spreader.fold(window, out);
        first = end;
    }

    // Phase 3: pieces larger than a window, one message at a time, a window at a time.
    for &piece in &pieces[small..] {
        for message in 0..count {
            spreader.fold_wide(message, piece, out);
        }
    }
}

/// The state of one spread batch, shared by its passes.
struct Spreader<'a, V, const W: usize, const G: usize> {
    /// Key and domain flags of the hash.
    mode: Mode,
    /// Every message of the batch, back to back.
    input: &'a [u8],
    /// Bytes per message.
    len: usize,
    /// Messages in the batch.
    count: usize,
    /// Chunk values of the current window, then its parents in place.
    scratch: [[u8; OUT_LEN]; WINDOW],
    /// Running fold of each message, from its last chunk outwards.
    fold: [[u8; OUT_LEN]; MAX_MESSAGES],
    /// Blocks of one pass of fold steps: a subtree value, then the fold so far.
    blocks: [[u8; 2 * OUT_LEN]; MAX_MESSAGES],
    /// The backend these passes run on.
    _backend: PhantomData<V>,
}

/// Up to one pass of independent compressions, one per lane.
struct Wave<const W: usize, const G: usize> {
    /// Byte offset of each lane's first block in its source.
    starts: [[usize; W]; G],
    /// Chunk counter of each lane.
    counters: [[u64; W]; G],
    /// Where each lane's result goes.
    targets: [[usize; W]; G],
    /// Lanes filled so far.
    len: usize,
}

impl<const W: usize, const G: usize> Wave<W, G> {
    /// An empty pass.
    const fn new() -> Self {
        Self {
            starts: [[0; W]; G],
            counters: [[0; W]; G],
            targets: [[0; W]; G],
            len: 0,
        }
    }

    /// Fill the next lane, and report whether every lane is now taken.
    #[inline(always)]
    const fn push(&mut self, start: usize, counter: u64, target: usize) -> bool {
        let (g, l) = (self.len / W, self.len % W);
        self.starts[g][l] = start;
        self.counters[g][l] = counter;
        self.targets[g][l] = target;
        self.len += 1;
        self.len == W * G
    }

    /// Copy the first lane into every free lane, so each lane reads valid input.
    ///
    /// Their results are never written out.
    #[inline(always)]
    fn pad(&mut self) {
        for i in self.len..W * G {
            let (g, l) = (i / W, i % W);
            self.starts[g][l] = self.starts[0][0];
            self.counters[g][l] = self.counters[0][0];
        }
    }

    /// The targets of the filled lanes, in lane order.
    #[inline(always)]
    fn targets(&self) -> &[usize] {
        &self.targets.as_flattened()[..self.len]
    }
}

/// Digests of every lane of every group of one pass.
type Values<const W: usize, const G: usize> = [[[u8; OUT_LEN]; W]; G];

/// One kind of pass: the compressions every lane runs, from its start in a shared source.
pub(super) trait Pass<V: Backend<W>, const W: usize> {
    /// The chaining value of every lane, on `G` groups.
    fn run<const G: usize>(&self, lanes: &Lanes<'_, W, G>, counters: &[[u64; W]; G])
    -> State<V, G>;
}

/// Whole chunks, each lane with its own counter.
struct WholeChunks(Mode);

/// The short last chunk of each lane's message, all at chunk `index`.
struct LastChunks {
    /// Key and domain flags of the hash.
    mode: Mode,
    /// Bytes per message.
    len: usize,
    /// Index of the last chunk.
    index: usize,
}

/// Parents, each lane's block holding both children.
struct Parents {
    /// Key and domain flags of the hash.
    mode: Mode,
    /// The root flag when these parents are the roots, else zero.
    root: u32,
}

impl<V: Backend<W>, const W: usize> Pass<V, W> for WholeChunks {
    #[inline(always)]
    fn run<const G: usize>(
        &self,
        lanes: &Lanes<'_, W, G>,
        counters: &[[u64; W]; G],
    ) -> State<V, G> {
        whole_chunks(self.0, lanes, counters)
    }
}

impl<V: Backend<W>, const W: usize> Pass<V, W> for LastChunks {
    #[inline(always)]
    fn run<const G: usize>(&self, lanes: &Lanes<'_, W, G>, _: &[[u64; W]; G]) -> State<V, G> {
        chunk(self.mode, lanes, self.len, self.index, 0)
    }
}

impl<V: Backend<W>, const W: usize> Pass<V, W> for Parents {
    #[inline(always)]
    fn run<const G: usize>(&self, lanes: &Lanes<'_, W, G>, _: &[[u64; W]; G]) -> State<V, G> {
        parents(self.mode, lanes, self.root)
    }
}

impl<const W: usize, const G: usize> Wave<W, G> {
    /// Run the filled lanes, on one register when they fit in one, else on every group.
    ///
    /// Lanes read their blocks from `source`, at their starts.
    #[inline(always)]
    fn run<V: Backend<W>, P: Pass<V, W>>(&mut self, pass: &P, source: &[u8]) -> Values<W, G> {
        self.pad();
        let mut values = [[[0u8; OUT_LEN]; W]; G];
        // SAFETY: spreading only runs inside a driver compiled for the backend's features.
        unsafe {
            if self.len <= W {
                let lanes = lanes_at([self.starts[0]], source);
                let one: &mut [_; 1] = (&mut values[..1]).try_into().unwrap();
                V::pass::<P, 1>(pass, &lanes, &[self.counters[0]], one);
            } else {
                let lanes = lanes_at(self.starts, source);
                V::pass::<P, G>(pass, &lanes, &self.counters, &mut values);
            }
        }
        values
    }

    /// Write the value of each filled lane to its target slot, and empty the pass.
    #[inline(always)]
    fn scatter(&mut self, values: &Values<W, G>, slots: &mut [[u8; OUT_LEN]]) {
        for (value, &target) in values.as_flattened().iter().zip(self.targets()) {
            slots[target] = *value;
        }
        self.len = 0;
    }
}

impl<V: Backend<W>, const W: usize, const G: usize> Spreader<'_, V, W, G> {
    /// Hash the chunks of every unit of a window, then build each unit's subtree in place.
    ///
    /// Each unit takes the scratch slots right after the previous one.
    ///
    /// Its subtree value ends in its first slot.
    fn window(&mut self, units: &(impl Iterator<Item = Unit> + Clone)) {
        // Level 0: one lane per chunk, each with its own counter.
        let mut wave = Wave::<W, G>::new();
        let mut at = 0;
        for unit in units.clone() {
            for offset in 0..unit.piece.size {
                let chunk = unit.piece.base + offset;
                let message = unit.message * self.len;
                if wave.push(message + chunk * CHUNK_LEN, chunk as u64, at + offset) {
                    self.run_chunks(&mut wave);
                }
            }
            at += unit.piece.size;
        }
        self.run_chunks(&mut wave);

        // Level l joins pairs of level l - 1 values, in place.
        //
        // Pair p of a unit reads slots 2p and 2p + 1 of that unit and writes slot p:
        //
        //     level 0: [a b c d]  ->  level 1: [ab cd . .]  ->  level 2: [abcd . . .]
        //
        // A pass loads every block before it writes, and later pairs p' read slots 2p' > p.
        //
        // So no write lands on a slot still to be read.
        let mut span = 2;
        while units.clone().any(|u| u.piece.reach() >= span) {
            let mut at = 0;
            for unit in units.clone() {
                let pairs = if unit.piece.reach() >= span {
                    unit.piece.size / span
                } else {
                    0
                };
                for pair in 0..pairs {
                    if wave.push(OUT_LEN * (at + 2 * pair), 0, at + pair) {
                        self.run_parents(&mut wave);
                    }
                }
                at += unit.piece.size;
            }
            self.run_parents(&mut wave);
            span *= 2;
        }
    }

    /// Fold the subtree value of every unit of a window into its message.
    ///
    /// Units of one piece are consecutive messages, which fold together in one pass.
    ///
    /// The next piece of a message folds only once the previous one has.
    fn fold(&mut self, units: impl Iterator<Item = Unit>, out: &mut [[u8; OUT_LEN]]) {
        let mut wave = Wave::<W, G>::new();
        let mut current: Option<Piece> = None;
        let mut at = 0;
        for unit in units {
            // A new piece starts: the pending steps of the previous one run first.
            if let Some(piece) = current.filter(|p| p.base != unit.piece.base) {
                self.run_fold(&mut wave, piece.fold, out);
            }
            current = Some(unit.piece);

            let left = self.scratch[at];
            let slot = at;
            at += unit.piece.size;
            let block = &mut self.blocks[wave.len];
            match unit.piece.fold {
                // The smallest subtree is the whole fold so far.
                Fold::Seed => {
                    self.fold[unit.message] = left;
                    continue;
                }
                // A whole message ends in the parent of its two halves.
                Fold::Whole => {
                    block[..OUT_LEN].copy_from_slice(&left);
                    block[OUT_LEN..].copy_from_slice(&self.scratch[slot + 1]);
                }
                // Any other subtree sits to the left of everything folded so far.
                Fold::Inner | Fold::Root => {
                    block[..OUT_LEN].copy_from_slice(&left);
                    block[OUT_LEN..].copy_from_slice(&self.fold[unit.message]);
                }
            }
            if wave.push(2 * OUT_LEN * wave.len, 0, unit.message) {
                self.run_fold(&mut wave, unit.piece.fold, out);
            }
        }
        if let Some(piece) = current {
            self.run_fold(&mut wave, piece.fold, out);
        }
    }

    /// Fold a subtree larger than a window into one message.
    fn fold_wide(&mut self, message: usize, piece: Piece, out: &mut [[u8; OUT_LEN]]) {
        // The parent block: both halves of a whole message, or the subtree then the fold so far.
        let (left, right) = match piece.fold {
            Fold::Whole => {
                let half = piece.size / 2;
                let left = self.subtree(message, piece.base, half);
                (left, self.subtree(message, piece.base + half, half))
            }
            _ => (
                self.subtree(message, piece.base, piece.size),
                self.fold[message],
            ),
        };
        if piece.fold == Fold::Seed {
            self.fold[message] = left;
            return;
        }
        self.blocks[0][..OUT_LEN].copy_from_slice(&left);
        self.blocks[0][OUT_LEN..].copy_from_slice(&right);
        let mut wave = Wave::<W, G>::new();
        wave.push(0, 0, message);
        self.run_fold(&mut wave, piece.fold, out);
    }

    /// The value of the complete subtree of `size` chunks of one message, from chunk `base`.
    ///
    /// A window at a time, the window values merge on a stack like a binary counter:
    ///
    /// ```text
    ///     after window 1: [w0 w1]  ->  [w01]
    ///     after window 3: [w01 w2 w3]  ->  [w01 w23]  ->  [w0123]
    /// ```
    fn subtree(&mut self, message: usize, base: usize, size: usize) -> [u8; OUT_LEN] {
        let pass = Parents {
            mode: self.mode,
            root: 0,
        };
        let mut stack = [[0u8; 2 * OUT_LEN]; usize::BITS as usize];
        let mut depth = 0;
        for window in 0..size / WINDOW {
            let piece = Piece {
                base: base + window * WINDOW,
                size: WINDOW,
                fold: Fold::Inner,
            };
            self.window(&core::iter::once(Unit { message, piece }));

            // A window value is a right child when the stack holds its left sibling.
            let mut value = self.scratch[0];
            let mut done = window + 1;
            while done.is_multiple_of(2) {
                depth -= 1;
                stack[depth][OUT_LEN..].copy_from_slice(&value);
                let mut wave = Wave::<W, G>::new();
                wave.push(0, 0, 0);
                value = wave.run::<V, _>(&pass, &stack[depth])[0][0];
                done /= 2;
            }
            stack[depth][..OUT_LEN].copy_from_slice(&value);
            depth += 1;
        }
        stack[0][..OUT_LEN].try_into().unwrap()
    }

    /// Hash the short last chunk of every message, one lane per message.
    fn last_chunks(&mut self, index: usize) {
        let pass = LastChunks {
            mode: self.mode,
            len: self.len,
            index,
        };
        let mut wave = Wave::<W, G>::new();
        for message in 0..self.count {
            let full = wave.push(message * self.len, 0, message);
            if full || message + 1 == self.count {
                let values = wave.run::<V, _>(&pass, self.input);
                wave.scatter(&values, &mut self.fold);
            }
        }
    }

    /// Run a pass of whole chunks into the scratch slots.
    fn run_chunks(&mut self, wave: &mut Wave<W, G>) {
        if wave.len > 0 {
            let values = wave.run::<V, _>(&WholeChunks(self.mode), self.input);
            wave.scatter(&values, &mut self.scratch);
        }
    }

    /// Run a pass of parents over pairs of scratch slots, back into the scratch slots.
    ///
    /// Every lane loads its block before any result is written.
    fn run_parents(&mut self, wave: &mut Wave<W, G>) {
        if wave.len > 0 {
            let pass = Parents {
                mode: self.mode,
                root: 0,
            };
            let values = wave.run::<V, _>(&pass, self.scratch.as_flattened());
            wave.scatter(&values, &mut self.scratch);
        }
    }

    /// Run a pass of fold steps, into the running folds, or into the digests at the root.
    fn run_fold(&mut self, wave: &mut Wave<W, G>, fold: Fold, out: &mut [[u8; OUT_LEN]]) {
        if wave.len == 0 {
            return;
        }
        let root = matches!(fold, Fold::Root | Fold::Whole);
        let pass = Parents {
            mode: self.mode,
            root: if root { ROOT } else { 0 },
        };
        let values = wave.run::<V, _>(&pass, self.blocks.as_flattened());
        wave.scatter(&values, if root { out } else { &mut self.fold });
    }
}

/// Lanes reading from `batch` at the given byte starts.
#[inline(always)]
fn lanes_at<const W: usize, const G: usize>(
    starts: [[usize; W]; G],
    batch: &[u8],
) -> Lanes<'_, W, G> {
    let last = starts.iter().flatten().copied().max().unwrap_or(0);
    Lanes {
        batch,
        starts,
        last,
    }
}

/// The chaining values of whole chunks, one per lane, each lane with its own counter.
#[inline(always)]
fn whole_chunks<V: Backend<W>, const W: usize, const G: usize>(
    mode: Mode,
    lanes: &Lanes<'_, W, G>,
    counters: &[[u64; W]; G],
) -> State<V, G> {
    // The counter of each lane, split into its low and high words.
    let counters: [[V; 2]; G] = core::array::from_fn(|g| {
        [
            from_lanes(counters[g].map(|c| c as u32)),
            from_lanes(counters[g].map(|c| (c >> 32) as u32)),
        ]
    });

    // Sixteen full blocks, the first and the last flagged.
    let mut state = mode.state();
    for block in 0..CHUNK_BLOCKS {
        let words = lanes.load::<V>(block * BLOCK_LEN);
        let flags = match block {
            0 => CHUNK_START,
            b if b == CHUNK_BLOCKS - 1 => CHUNK_END,
            _ => 0,
        };
        compress_counters(
            &mut state,
            &words,
            &counters,
            BLOCK_LEN as u32,
            mode.flags | flags,
        );
    }
    state
}

/// The chaining values of parents, one per lane, each lane's block holding both children.
#[inline(always)]
fn parents<V: Backend<W>, const W: usize, const G: usize>(
    mode: Mode,
    lanes: &Lanes<'_, W, G>,
    root: u32,
) -> State<V, G> {
    // Parents always have a full block and a zero counter.
    let words = lanes.load::<V>(0);
    let mut state = mode.state();
    compress(
        &mut state,
        &words,
        0,
        BLOCK_LEN as u32,
        mode.flags | PARENT | root,
    );
    state
}
