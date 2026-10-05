//! 32-way SHA-256 for x86-64 AVX-512.
//!
//! A register holds the same word of sixteen messages, one per 32-bit lane.
//! Two groups of registers run side by side, so one compression advances 32 messages.
//!
//! Equal-length messages share their block count and their padding.
//! So every lane runs the same rounds, and a block of padding only is the same in every lane.
//!
//! The end of a batch rarely fills 32 lanes.
//! It takes one group of sixteen, or four SHA-NI streams when the CPU has SHA-NI.
//!
//! Every build of x86-64 compiles this module.
//! Each function that runs AVX-512 enables it, and the entry points run once the CPU is known to have it.
//!
//! A closure inside such a function carries its features, so `array::map` or `from_fn` cannot inline it.
//! Plain loops fill the arrays instead, and inline in every build.

mod rounds;

use core::arch::x86_64::*;

use p3_symmetric::{CryptographicHasher, PseudoCompressionFunction};

use self::rounds::{K, compress_blocks, compress_shared};
use crate::{H256_256, Sha256, Sha256Compress, x86_64_sha_ni};

/// Lanes in one register.
const WIDTH: usize = 16;

/// Register groups the widest pass advances together.
const GROUPS: usize = 2;

/// Messages one compression advances at once.
pub(crate) const LANES: usize = WIDTH * GROUPS;

/// Bytes in one compression block.
const BLOCK_BYTES: usize = 64;

/// Words in one compression block.
const BLOCK_WORDS: usize = 16;

/// Words in a chaining value.
const STATE_WORDS: usize = 8;

/// Rounds in one compression.
const ROUNDS: usize = 64;

/// Bytes the message length takes at the end of the last block.
const LENGTH_BYTES: usize = 8;

/// Byte permutation reversing each 32-bit word of a 128-bit block.
///
/// SHA-256 reads words big-endian, while a vector load is little-endian.
///
/// The permutation is its own inverse, so it serves loads and stores.
const REVERSE_WORD_BYTES: [u8; 16] = [3, 2, 1, 0, 7, 6, 5, 4, 11, 10, 9, 8, 15, 14, 13, 12];

/// Chaining values of every lane of `G` groups.
type State<const G: usize> = [[__m512i; STATE_WORDS]; G];

/// Message words of every lane of `G` groups.
type Block<const G: usize> = [[__m512i; BLOCK_WORDS]; G];

/// One digest per lane of `G` groups.
type Digests<const G: usize> = [[[u8; 32]; WIDTH]; G];

/// One block per lane of `G` groups.
type Rows<'a, const G: usize> = [[&'a [u8; BLOCK_BYTES]; WIDTH]; G];

/// How a CPU with AVX-512 shares a batch with SHA-NI, by the crossovers measured on it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Streams {
    /// The CPU has no SHA-NI, so the register passes and the scalar hasher take every message.
    Off,
    /// Four SHA-NI streams take three to eight leftover messages, as on Zen 5.
    FromThree,
    /// Four SHA-NI streams take four to eight leftover messages, as on Intel.
    ///
    /// Three messages leave one stream empty, and three scalar calls beat that there.
    FromFour,
    /// The crossovers of Zen 4, which runs each 512-bit operation as two 256-bit halves.
    ///
    /// - Messages of at least [`ZEN4_STREAMS_FROM_LEN`] bytes all go to the streams.
    /// - Shorter ones keep the register passes for whole groups, and the streams take most leftovers.
    Zen4,
}

/// Message length from which four SHA-NI streams beat whole register groups on Zen 4.
///
/// The two tie at 2 KiB, and the streams pull ahead on longer messages.
const ZEN4_STREAMS_FROM_LEN: usize = 2048;

impl Streams {
    /// Whether the CPU runs SHA-NI.
    const fn sha_ni(self) -> bool {
        !matches!(self, Self::Off)
    }
}

/// A way to hash the next few messages of a batch.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Pass {
    /// Both register groups, up to 32 messages.
    Wide,
    /// One register group, up to 16 messages.
    Narrow,
    /// Four interleaved SHA-NI streams.
    FourStreams,
    /// The scalar hasher, one message at a time.
    Single,
}

impl Pass {
    /// The pass for the next `left` messages, and how many of them it takes.
    ///
    /// A register pass costs the same however many of its lanes hold a message.
    /// So a pass is picked by the messages it can fill:
    ///
    /// | left   | `Off`        | `FromThree`  | `FromFour`   | `Zen4`       |
    /// |--------|--------------|--------------|--------------|--------------|
    /// | 25..   | Wide         | Wide         | Wide         | Wide         |
    /// | 21..=24| Wide         | Wide         | Wide         | FourStreams  |
    /// | 18..=20| Wide         | Narrow (16)  | Narrow (16)  | Narrow (16)  |
    /// | 17     | Narrow (16)  | Narrow (16)  | Narrow (16)  | Narrow (16)  |
    /// | 14..=16| Narrow       | Narrow       | Narrow       | Narrow       |
    /// | 9..=13 | Narrow       | Narrow       | Narrow       | FourStreams  |
    /// | 4..=8  | Narrow       | FourStreams  | FourStreams  | FourStreams  |
    /// | 3      | Narrow       | FourStreams  | Single       | FourStreams  |
    /// | 2      | Narrow       | Single       | Single       | Single       |
    /// | 1      | Single       | Single       | Single       | Single       |
    ///
    /// - A lone register group costs well under two, and about half on long messages.
    /// - Four SHA-NI streams beat a register group while they need at most two calls.
    /// - Two passes over a few leftover messages beat one mostly empty wide pass.
    /// - Zen 4 splits each 512-bit operation in two, so a register pass gains less there.
    const fn next(left: usize, streams: Streams) -> (Self, usize) {
        // The widest pass fills most of its lanes.
        let wide_from = match streams {
            Streams::Off => 18,
            Streams::FromThree | Streams::FromFour => 21,
            Streams::Zen4 => 25,
        };
        if left >= wide_from {
            return (Self::Wide, if left < LANES { left } else { LANES });
        }

        // On Zen 4, the streams beat a wide pass with up to a quarter of its lanes empty.
        if matches!(streams, Streams::Zen4) && left > 20 {
            return (Self::FourStreams, left);
        }

        // One group takes sixteen, and the few left over take the cheaper passes below.
        if left > WIDTH {
            return (Self::Narrow, WIDTH);
        }

        // One group holds all the rest.
        let narrow_from = match streams {
            Streams::Off => 2,
            Streams::FromThree | Streams::FromFour => 9,
            Streams::Zen4 => 14,
        };
        if left >= narrow_from {
            return (Self::Narrow, left);
        }

        // Too few messages to fill a register group.
        //
        // One or two messages run no faster as streams of one call than as calls of their own.
        let streams_from = match streams {
            Streams::FromFour => 4,
            _ => 3,
        };
        if streams.sha_ni() && left >= streams_from {
            (Self::FourStreams, left)
        } else {
            (Self::Single, left)
        }
    }

    /// The pass for a whole batch of `count` messages of `len` bytes, if a single one takes it all.
    ///
    /// Such a batch never reads the padding, so it skips computing it.
    const fn whole(count: usize, len: usize, streams: Streams) -> Option<Self> {
        if matches!(streams, Streams::Zen4) && len >= ZEN4_STREAMS_FROM_LEN {
            return Some(Self::FourStreams);
        }
        match Self::next(count, streams) {
            (pass @ (Self::FourStreams | Self::Single), _) => Some(pass),
            _ => None,
        }
    }
}

/// Hash `out.len()` equal-length messages laid end to end in `input`.
///
/// # Safety
///
/// The running CPU has AVX-512F and AVX-512BW, and SHA-NI with SSE4.1 unless `streams` is [`Streams::Off`].
///
/// # Panics
///
/// Panics if the input length is not a whole multiple of the digest count.
#[target_feature(enable = "avx512f,avx512bw")]
pub(crate) unsafe fn hash_many(input: &[u8], out: &mut [[u8; 32]], streams: Streams) {
    // No digests requested means there is nothing to read from the input.
    if out.is_empty() {
        return;
    }

    // Every message has the same length, so the split is exact by contract.
    assert!(
        input.len().is_multiple_of(out.len()),
        "input length ({}) must be a whole multiple of the digest count ({})",
        input.len(),
        out.len()
    );
    let len = input.len() / out.len();

    // A batch that no register pass touches never reads the padding.
    //
    // Its shared schedule costs about as much as one short message, so such a batch skips it.
    if let Some(pass) = Pass::whole(out.len(), len, streams) {
        // SAFETY: the passes pick four streams only when `streams` has SHA-NI, so the CPU has it.
        unsafe { hash_serial(input, out, pass == Pass::FourStreams) };
        return;
    }
    let padding = Padding::new(len);

    // Whole groups of 32 messages.
    let (groups, _) = out.as_chunks_mut::<LANES>();
    for (index, digests) in groups.iter_mut().enumerate() {
        let lanes = Lanes {
            input,
            len,
            first: index * LANES,
            count: LANES,
        };
        hash_group::<GROUPS>(
            &lanes,
            &padding,
            digests.as_chunks_mut().0.try_into().unwrap(),
        );
    }

    // The messages left over, pass by pass.
    //
    //     40 messages:  one group takes 32, then FourStreams takes the last 8
    let mut first = groups.len() * LANES;
    while first < out.len() {
        let (pass, count) = Pass::next(out.len() - first, streams);
        let digests = &mut out[first..first + count];
        let lanes = Lanes {
            input,
            len,
            first,
            count,
        };
        match pass {
            Pass::Wide => hash_pass::<GROUPS>(&lanes, &padding, digests),
            Pass::Narrow => hash_pass::<1>(&lanes, &padding, digests),
            // SAFETY: the passes pick four streams only when `streams` has SHA-NI, so the CPU has it.
            Pass::FourStreams | Pass::Single => unsafe {
                hash_serial(
                    &input[first * len..][..count * len],
                    digests,
                    pass == Pass::FourStreams,
                );
            },
        }
        first += count;
    }
}

/// Hash `out.len()` equal-length messages laid end to end in `messages`, without the registers.
///
/// Four SHA-NI streams take them when `four_streams` holds, and the scalar hasher otherwise.
///
/// # Safety
///
/// The running CPU has SHA-NI and SSE4.1 when `four_streams` holds.
unsafe fn hash_serial(messages: &[u8], out: &mut [[u8; 32]], four_streams: bool) {
    // An empty batch has nothing to hash, and would divide by zero below.
    if out.is_empty() {
        return;
    }

    if four_streams {
        // SAFETY: the caller vouches for SHA-NI and SSE4.1.
        unsafe { x86_64_sha_ni::hash_many(messages, out) };
        return;
    }

    // Indexing rather than chunking also covers empty messages.
    let len = messages.len() / out.len();
    for (index, digest) in out.iter_mut().enumerate() {
        *digest = Sha256.hash_slice(&messages[index * len..][..len]);
    }
}

/// Hash the messages of one register pass of `G` groups into `out`.
///
/// The spare lanes repeat the last message, and their digests are never written out.
///
/// Kept out of line so its stack buffers only cost a call that has messages left over.
#[inline(never)]
#[target_feature(enable = "avx512f,avx512bw")]
fn hash_pass<const G: usize>(lanes: &Lanes<'_>, padding: &Padding, out: &mut [[u8; 32]]) {
    let mut digests = [[[0u8; 32]; WIDTH]; G];
    hash_group::<G>(lanes, padding, &mut digests);
    out.copy_from_slice(&digests.as_flattened()[..out.len()]);
}

/// Compress each 64-byte pair from the initial hash value, without padding.
///
/// # Safety
///
/// The running CPU has AVX-512F and AVX-512BW, and SHA-NI with SSE4.1 unless `streams` is [`Streams::Off`].
///
/// # Panics
///
/// Panics if the input and output counts differ.
#[target_feature(enable = "avx512f,avx512bw")]
pub(crate) unsafe fn compress_many(
    inputs: &[[[u8; 32]; 2]],
    out: &mut [[u8; 32]],
    streams: Streams,
) {
    assert_eq!(
        inputs.len(),
        out.len(),
        "group count ({}) must equal the output count ({})",
        inputs.len(),
        out.len()
    );

    // A pair of digests is exactly one block.
    let (blocks, _) = inputs
        .as_flattened()
        .as_flattened()
        .as_chunks::<BLOCK_BYTES>();

    // Whole groups of 32 blocks.
    let (groups, _) = out.as_chunks_mut::<LANES>();
    for (index, digests) in groups.iter_mut().enumerate() {
        let group = &blocks[index * LANES..][..LANES];
        let mut rows: Rows<'_, GROUPS> = [[&group[0]; WIDTH]; GROUPS];
        for (row, block) in rows.as_flattened_mut().iter_mut().zip(group) {
            *row = block;
        }
        compress_group::<GROUPS>(&rows, digests.as_chunks_mut().0.try_into().unwrap());
    }

    // The blocks left over, in the same passes as the hash.
    let mut first = groups.len() * LANES;
    while first < out.len() {
        let (pass, count) = Pass::next(out.len() - first, streams);
        let digests = &mut out[first..first + count];
        let blocks = &blocks[first..first + count];
        match pass {
            Pass::Wide => compress_pass::<GROUPS>(blocks, digests),
            Pass::Narrow => compress_pass::<1>(blocks, digests),
            // SAFETY: as in the hash, four streams mean the CPU has SHA-NI.
            Pass::FourStreams => unsafe {
                x86_64_sha_ni::compress_many(&inputs[first..first + count], digests);
            },
            Pass::Single => {
                for (input, digest) in inputs[first..first + count].iter().zip(digests) {
                    *digest = Sha256Compress.compress(*input);
                }
            }
        }
        first += count;
    }
}

/// Compress the blocks of one register pass of `G` groups into `out`.
///
/// The spare lanes repeat the last block, and their digests are never written out.
///
/// Kept out of line so its stack buffers only cost a call that has blocks left over.
#[inline(never)]
#[target_feature(enable = "avx512f,avx512bw")]
fn compress_pass<const G: usize>(blocks: &[[u8; BLOCK_BYTES]], out: &mut [[u8; 32]]) {
    let last = blocks.len() - 1;
    let mut rows: Rows<'_, G> = [[&blocks[last]; WIDTH]; G];
    for (row, block) in rows.as_flattened_mut().iter_mut().zip(blocks) {
        *row = block;
    }
    let mut digests = [[[0u8; 32]; WIDTH]; G];
    compress_group::<G>(&rows, &mut digests);
    out.copy_from_slice(&digests.as_flattened()[..out.len()]);
}

/// One block per lane of `G` groups, from the initial hash value.
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
fn compress_group<const G: usize>(rows: &Rows<'_, G>, out: &mut Digests<G>) {
    // No padding: the digest is the chaining value after this single block.
    let mut state = initial_state::<G>();
    compress_blocks(&mut state, &load_block(rows));
    store_digests(&state, out);
}

/// The padding of FIPS 180-4 section 5.1.1, fixed once for a message length.
///
/// The message is followed by a 0x80 byte, then zeros, then its bit length as a 64-bit big-endian integer.
/// The zeros make the total a whole number of blocks.
struct Padding {
    /// Message bytes in the last block that holds any, or zero.
    tail: usize,
    /// The message length in bits, modulo 2^64.
    bit_len: u64,
    /// `K_t + W_t` of a last block that holds no message byte, if the message ends with one.
    ///
    /// That block is the same in every lane, so its schedule is computed once, in scalar code.
    shared: Option<[u32; ROUNDS]>,
}

impl Padding {
    fn new(len: usize) -> Self {
        let tail = len % BLOCK_BYTES;
        let bit_len = (len as u64).wrapping_mul(8);

        // Three shapes, by the message bytes in the last partial block.
        //
        //     tail = 0:        [ 0x80, zeros, length ]                         shared
        //     tail in 1..56:   [ tail bytes, 0x80, zeros, length ]
        //     tail in 56..64:  [ tail bytes, 0x80, zeros ] [ zeros, length ]    second one shared
        let shared = (tail == 0 || tail >= BLOCK_BYTES - LENGTH_BYTES).then(|| {
            let mut block = [0u32; BLOCK_WORDS];
            if tail == 0 {
                block[0] = 0x8000_0000;
            }
            block[14] = (bit_len >> 32) as u32;
            block[15] = bit_len as u32;
            shared_schedule(&block)
        });
        Self {
            tail,
            bit_len,
            shared,
        }
    }
}

/// `K_t + W_t` for every round of one block, from the recurrence of FIPS 180-4 section 6.2.2.
fn shared_schedule(block: &[u32; BLOCK_WORDS]) -> [u32; ROUNDS] {
    // The block opens the schedule.
    let mut w = [0u32; ROUNDS];
    w[..BLOCK_WORDS].copy_from_slice(block);

    // W_t = s1(W_{t-2}) + W_{t-7} + s0(W_{t-15}) + W_{t-16}.
    for t in BLOCK_WORDS..ROUNDS {
        let s0 = w[t - 15].rotate_right(7) ^ w[t - 15].rotate_right(18) ^ (w[t - 15] >> 3);
        let s1 = w[t - 2].rotate_right(17) ^ w[t - 2].rotate_right(19) ^ (w[t - 2] >> 10);
        w[t] = w[t - 16]
            .wrapping_add(s0)
            .wrapping_add(w[t - 7])
            .wrapping_add(s1);
    }

    // Each round adds its constant and its word together, so they are summed here once.
    core::array::from_fn(|t| K[t].wrapping_add(w[t]))
}

/// The messages of one group: `count` consecutive messages of `len` bytes, from message `first`.
///
/// Lanes past `count` repeat the last message.
struct Lanes<'a> {
    /// Every message of the batch, back to back.
    input: &'a [u8],
    /// Bytes in one message.
    len: usize,
    /// Index of the message in lane 0.
    first: usize,
    /// Distinct messages in the group, at least one and at most its lanes.
    count: usize,
}

impl Lanes<'_> {
    /// Byte offset of the message of lane `lane` in the batch.
    #[inline(always)]
    fn start(&self, lane: usize) -> usize {
        // A spare lane points at the last message again.
        (self.first + lane.min(self.count - 1)) * self.len
    }

    /// The whole block at byte `offset` of the message of every lane of `G` groups.
    ///
    /// # Panics
    ///
    /// Panics if a lane has fewer than `offset + 64` bytes left in the batch.
    #[inline(always)]
    fn rows<const G: usize>(&self, offset: usize) -> Rows<'_, G> {
        // One flat pass over the lanes, which the compiler turns into a few vector steps.
        let mut rows: Rows<'_, G> = [[&[0; BLOCK_BYTES]; WIDTH]; G];
        for (lane, row) in rows.as_flattened_mut().iter_mut().enumerate() {
            *row = self.input[self.start(lane) + offset..][..BLOCK_BYTES]
                .try_into()
                .unwrap();
        }
        rows
    }
}

/// Hash the message of every lane of `G` groups.
#[target_feature(enable = "avx512f,avx512bw")]
fn hash_group<const G: usize>(lanes: &Lanes<'_>, padding: &Padding, out: &mut Digests<G>) {
    let mut state = initial_state::<G>();

    // Every whole block of message bytes, then the last partial one if any.
    let whole_blocks = lanes.len / BLOCK_BYTES;
    let blocks = whole_blocks + usize::from(padding.tail > 0);

    // A message of several whole blocks takes them in a loop of its own.
    //
    // A short one keeps a single copy of the kernel in the loop below.
    let first = if whole_blocks > 1 {
        compress_whole_blocks(lanes, &mut state, whole_blocks);
        whole_blocks
    } else {
        0
    };
    for index in first..blocks {
        let offset = index * BLOCK_BYTES;
        let block = if index < whole_blocks {
            load_block(&lanes.rows::<G>(offset))
        } else {
            last_block(lanes, offset, padding)
        };
        compress_blocks(&mut state, &block);
    }

    // A last block of padding only is the same in every lane.
    if let Some(kw) = &padding.shared {
        compress_shared(&mut state, kw);
    }

    store_digests(&state, out);
}

/// Absorb the first `blocks` whole blocks of every lane of `G` groups.
///
/// The lanes' offsets are found once, before the loop.
/// So every load address is ready early, and more loads are in flight from memory.
#[inline(never)]
#[target_feature(enable = "avx512f,avx512bw")]
fn compress_whole_blocks<const G: usize>(lanes: &Lanes<'_>, state: &mut State<G>, blocks: usize) {
    let mut starts = [[0; WIDTH]; G];
    for (lane, start) in starts.as_flattened_mut().iter_mut().enumerate() {
        *start = lanes.start(lane);
    }
    for index in 0..blocks {
        let offset = index * BLOCK_BYTES;
        let mut rows: Rows<'_, G> = [[&[0; BLOCK_BYTES]; WIDTH]; G];
        for (row, start) in rows
            .as_flattened_mut()
            .iter_mut()
            .zip(starts.as_flattened())
        {
            *row = lanes.input[start + offset..][..BLOCK_BYTES]
                .try_into()
                .unwrap();
        }
        compress_blocks(state, &load_block(&rows));
    }
}

/// The padded last block of every lane of `G` groups, holding its final `padding.tail` message bytes.
#[inline(never)]
#[target_feature(enable = "avx512f,avx512bw")]
fn last_block<const G: usize>(lanes: &Lanes<'_>, offset: usize, padding: &Padding) -> Block<G> {
    let tail = padding.tail;

    // Every lane's message ends inside the batch, the last lane's furthest in.
    assert!(lanes.start(WIDTH * G - 1) + offset + tail <= lanes.input.len());

    // A masked load reads the tail bytes and zeroes the rest of the row.
    //
    // Masked-off bytes are never accessed, so no read goes past a message end.
    let mask: __mmask64 = (1 << tail) - 1;
    let mut block: Block<G> = [[zero(); BLOCK_WORDS]; G];
    for (g, words) in block.iter_mut().enumerate() {
        let mut rows = [zero(); WIDTH];
        for (l, row) in rows.iter_mut().enumerate() {
            // SAFETY: the `tail` enabled bytes lie inside the batch, by the assertion above.
            *row = unsafe {
                let start = lanes
                    .input
                    .as_ptr()
                    .add(lanes.start(WIDTH * g + l) + offset);
                _mm512_maskz_loadu_epi8(mask, start.cast())
            };
        }
        *words = transpose_rows(rows);
    }

    // Words are big-endian, so the first message byte is the top byte of its word.
    //
    // The marker takes the byte right after the message.
    //
    //     tail = 6:  word 0 holds 4 message bytes, word 1 holds 2 and then 0x80
    let marker = splat(0x8000_0000 >> (8 * (tail % 4)));
    for group in &mut block {
        group[tail / 4] = _mm512_or_si512(group[tail / 4], marker);

        // The length joins this block only when eight bytes are left after the marker.
        if padding.shared.is_none() {
            group[14] = splat((padding.bit_len >> 32) as u32);
            group[15] = splat(padding.bit_len as u32);
        }
    }
    block
}

/// The initial hash value of FIPS 180-4 section 5.3.3, in every lane of `G` groups.
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
fn initial_state<const G: usize>() -> State<G> {
    // Every lane of every group starts from the same eight words.
    let mut state = [[zero(); STATE_WORDS]; G];
    for group in &mut state {
        for (word, &h) in group.iter_mut().zip(&H256_256) {
            *word = splat(h);
        }
    }
    state
}

/// The zero word in every lane.
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
fn zero() -> __m512i {
    _mm512_setzero_si512()
}

/// The same word in every lane.
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
fn splat(word: u32) -> __m512i {
    _mm512_set1_epi32(word as i32)
}

/// Reverse the bytes of every 32-bit word.
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
fn byte_swap(x: __m512i) -> __m512i {
    // SAFETY: `[u8; 16]` and `__m128i` are both 16 bytes and every bit pattern is valid.
    let mask = unsafe { core::mem::transmute::<[u8; 16], __m128i>(REVERSE_WORD_BYTES) };

    // `vpshufb` on 512 bits is what needs AVX-512BW.
    _mm512_shuffle_epi8(x, _mm512_broadcast_i32x4(mask))
}

/// A 4 x 4 transpose inside every 128-bit block of four rows.
///
/// Block `k` of output `j` holds word `4k + j` of rows `a`, `b`, `c` and `d`, in that order.
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
fn transpose_blocks(a: __m512i, b: __m512i, c: __m512i, d: __m512i) -> [__m512i; 4] {
    // Interleave 32-bit words of row pairs, then 64-bit pairs of those.
    let ab_lo = _mm512_unpacklo_epi32(a, b);
    let ab_hi = _mm512_unpackhi_epi32(a, b);
    let cd_lo = _mm512_unpacklo_epi32(c, d);
    let cd_hi = _mm512_unpackhi_epi32(c, d);
    [
        _mm512_unpacklo_epi64(ab_lo, cd_lo),
        _mm512_unpackhi_epi64(ab_lo, cd_lo),
        _mm512_unpacklo_epi64(ab_hi, cd_hi),
        _mm512_unpackhi_epi64(ab_hi, cd_hi),
    ]
}

/// Load one block from each lane as the big-endian message words of `G` groups.
///
/// - Row `l` of group `g` is one block of the message in lane `16g + l`.
/// - Word `w` of group `g` holds word `w` of the messages in lanes `16g` to `16g + 15`.
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
fn load_block<const G: usize>(rows: &Rows<'_, G>) -> Block<G> {
    let mut block = [[zero(); BLOCK_WORDS]; G];
    for (words, rows) in block.iter_mut().zip(rows) {
        let mut loaded = [zero(); WIDTH];
        for (vector, row) in loaded.iter_mut().zip(rows) {
            // SAFETY: each row is 64 readable bytes, and the load has no alignment requirement.
            *vector = unsafe { _mm512_loadu_si512(row.as_ptr().cast()) };
        }
        *words = transpose_rows(loaded);
    }
    block
}

/// Turn sixteen rows of one block each into sixteen big-endian message words.
///
/// Word `w` of the result holds word `w` of every row, row `l` in lane `l`.
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
fn transpose_rows(rows: [__m512i; WIDTH]) -> [__m512i; BLOCK_WORDS] {
    let mut r = rows;
    for row in &mut r {
        *row = byte_swap(*row);
    }

    // Phase 1: block k of u[q][j] is word 4k + j of rows 4q to 4q + 3.
    let mut u = [[zero(); 4]; 4];
    for (q, u) in u.iter_mut().enumerate() {
        *u = transpose_blocks(r[4 * q], r[4 * q + 1], r[4 * q + 2], r[4 * q + 3]);
    }

    // Phase 2: word 4k + j gathers block k of u[0][j] to u[3][j], in row order.
    let mut words = [r[0]; BLOCK_WORDS];
    for j in 0..4 {
        // Blocks 0 and 1, then blocks 2 and 3, of each pair of quads.
        let q01_lo = _mm512_shuffle_i32x4::<0x44>(u[0][j], u[1][j]);
        let q01_hi = _mm512_shuffle_i32x4::<0xEE>(u[0][j], u[1][j]);
        let q23_lo = _mm512_shuffle_i32x4::<0x44>(u[2][j], u[3][j]);
        let q23_hi = _mm512_shuffle_i32x4::<0xEE>(u[2][j], u[3][j]);

        // Even blocks, then odd blocks, of each half.
        words[j] = _mm512_shuffle_i32x4::<0x88>(q01_lo, q23_lo);
        words[4 + j] = _mm512_shuffle_i32x4::<0xDD>(q01_lo, q23_lo);
        words[8 + j] = _mm512_shuffle_i32x4::<0x88>(q01_hi, q23_hi);
        words[12 + j] = _mm512_shuffle_i32x4::<0xDD>(q01_hi, q23_hi);
    }
    words
}

/// Write the big-endian digest of every lane of `G` groups.
///
/// Two adjacent digests fill one 64-byte store, so eight stores cover a group.
#[inline]
#[target_feature(enable = "avx512f,avx512bw")]
fn store_digests<const G: usize>(state: &State<G>, out: &mut Digests<G>) {
    for (state, out) in state.iter().zip(out) {
        let mut s = *state;
        for word in &mut s {
            *word = byte_swap(*word);
        }

        // Block k of lo[j] is words 0 to 3 of lane 4k + j, and hi[j] holds words 4 to 7.
        let lo = transpose_blocks(s[0], s[1], s[2], s[3]);
        let hi = transpose_blocks(s[4], s[5], s[6], s[7]);

        for j in [0, 2] {
            // SAFETY: each store writes the 64 bytes of two adjacent digests of `out`.
            unsafe {
                // Blocks 0 and 1 of each half, then blocks 2 and 3.
                let pairs = [
                    _mm512_shuffle_i32x4::<0x44>(lo[j], hi[j]),
                    _mm512_shuffle_i32x4::<0x44>(lo[j + 1], hi[j + 1]),
                    _mm512_shuffle_i32x4::<0xEE>(lo[j], hi[j]),
                    _mm512_shuffle_i32x4::<0xEE>(lo[j + 1], hi[j + 1]),
                ];

                // The digests of lanes 4k + j and 4k + j + 1, back to back.
                let digests = [
                    _mm512_shuffle_i32x4::<0x88>(pairs[0], pairs[1]),
                    _mm512_shuffle_i32x4::<0xDD>(pairs[0], pairs[1]),
                    _mm512_shuffle_i32x4::<0x88>(pairs[2], pairs[3]),
                    _mm512_shuffle_i32x4::<0xDD>(pairs[2], pairs[3]),
                ];
                for (k, digests) in digests.into_iter().enumerate() {
                    // Lanes 4k + j and 4k + j + 1 exist, and their digests are adjacent.
                    let pair = out[4 * k + j..][..2].as_flattened_mut();
                    _mm512_storeu_si512(pair.as_mut_ptr().cast(), digests);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec::Vec;

    use proptest::prelude::*;

    use super::*;
    use crate::tests::spec_compress;

    // The kernel tests run AVX-512, so they skip a CPU without it.
    cpufeatures::new!(cpu_avx512, "avx512f", "avx512bw");

    /// Every policy, in the column order of the table in `Pass::next`.
    const ALL_STREAMS: [Streams; 4] = [
        Streams::Off,
        Streams::FromThree,
        Streams::FromFour,
        Streams::Zen4,
    ];

    /// A deterministic stream of words, so a failing case reproduces exactly.
    fn words(mut seed: u64) -> impl Iterator<Item = u32> {
        core::iter::repeat_with(move || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed as u32
        })
    }

    /// One chaining value per lane, packed into `G` groups: lane `16g + l` is lane `l` of group `g`.
    fn pack<const G: usize>(states: &[[u32; STATE_WORDS]]) -> State<G> {
        core::array::from_fn(|g| {
            core::array::from_fn(|i| {
                let lanes: [u32; WIDTH] = core::array::from_fn(|l| states[WIDTH * g + l][i]);
                // SAFETY: a vector is exactly its sixteen lanes, lane 0 at the lowest address.
                unsafe { core::mem::transmute::<[u32; WIDTH], __m512i>(lanes) }
            })
        })
    }

    /// Serialize a chaining value as the big-endian digest the kernels store.
    fn digest(state: [u32; STATE_WORDS]) -> [u8; 32] {
        let mut out = [0u8; 32];
        for (bytes, word) in out.as_chunks_mut::<4>().0.iter_mut().zip(state) {
            *bytes = word.to_be_bytes();
        }
        out
    }

    /// Random chaining values, one per lane of `G` groups.
    fn random_states<const G: usize>(
        stream: &mut impl Iterator<Item = u32>,
    ) -> Vec<[u32; STATE_WORDS]> {
        (0..WIDTH * G)
            .map(|_| core::array::from_fn(|_| stream.next().unwrap()))
            .collect()
    }

    /// The block kernel of `G` groups against the specification, from random chaining values.
    #[target_feature(enable = "avx512f,avx512bw")]
    fn check_block_kernel<const G: usize>(seed: u64) -> Result<(), TestCaseError> {
        // Random chaining values and blocks, different in every lane.
        let mut stream = words(seed | 1);
        let states = random_states::<G>(&mut stream);
        let blocks: Vec<[u8; BLOCK_BYTES]> = (0..WIDTH * G)
            .map(|_| {
                let words: [u32; BLOCK_WORDS] = core::array::from_fn(|_| stream.next().unwrap());
                *words
                    .map(u32::to_le_bytes)
                    .as_flattened()
                    .as_array()
                    .unwrap()
            })
            .collect();

        // Load, compress and store through the vector path.
        let mut state = pack::<G>(&states);
        let rows = core::array::from_fn(|g| core::array::from_fn(|l| &blocks[WIDTH * g + l]));
        compress_blocks(&mut state, &load_block::<G>(&rows));
        let mut out = [[[0u8; 32]; WIDTH]; G];
        store_digests(&state, &mut out);

        // Every lane must match one compression of FIPS 180-4 section 6.2.2.
        for (lane, out) in out.as_flattened().iter().enumerate() {
            let mut expected = states[lane];
            spec_compress(&mut expected, &blocks[lane]);
            prop_assert_eq!(*out, digest(expected), "lane {}", lane);
        }
        Ok(())
    }

    /// The shared-block kernel of `G` groups against the specification, from random chaining values.
    #[target_feature(enable = "avx512f,avx512bw")]
    fn check_shared_kernel<const G: usize>(seed: u64) -> Result<(), TestCaseError> {
        // Random chaining values in every lane, and one random block for all of them.
        let mut stream = words(seed | 1);
        let states = random_states::<G>(&mut stream);
        let block: [u32; BLOCK_WORDS] = core::array::from_fn(|_| stream.next().unwrap());

        // The schedule is expanded once, in scalar code, and broadcast.
        let mut state = pack::<G>(&states);
        compress_shared(&mut state, &shared_schedule(&block));
        let mut out = [[[0u8; 32]; WIDTH]; G];
        store_digests(&state, &mut out);

        // The specification reads the block as big-endian bytes.
        let bytes = *block
            .map(u32::to_be_bytes)
            .as_flattened()
            .as_array()
            .unwrap();
        for (lane, out) in out.as_flattened().iter().enumerate() {
            let mut expected = states[lane];
            spec_compress(&mut expected, &bytes);
            prop_assert_eq!(*out, digest(expected), "lane {}", lane);
        }
        Ok(())
    }

    #[test]
    fn every_pass_fits_its_lanes_and_the_passes_cover_the_batch() {
        for streams in ALL_STREAMS {
            for count in 1..=4 * LANES {
                // Walk a batch of `count` messages the way the drivers do.
                let mut left = count;
                let mut passes = Vec::new();
                while left > 0 {
                    let (pass, take) = Pass::next(left, streams);

                    // A pass takes at least one message, and no more than its lanes hold.
                    let capacity = match pass {
                        Pass::Wide => LANES,
                        Pass::Narrow => WIDTH,
                        Pass::FourStreams | Pass::Single => left,
                    };
                    assert!((1..=capacity.min(left)).contains(&take), "{count} messages");
                    passes.push(pass);
                    left -= take;
                }

                // Whole groups of 32 always take the widest pass.
                let wide = passes.iter().filter(|&&pass| pass == Pass::Wide).count();
                assert!(wide >= count / LANES, "{count} messages");

                // Only a CPU with SHA-NI uses four streams.
                assert!(streams.sha_ni() || !passes.contains(&Pass::FourStreams));
            }
        }
    }

    #[test]
    fn each_cpu_takes_its_measured_crossovers() {
        use Pass::{FourStreams as S, Narrow as N, Single as One, Wide as W};

        // The first pass and its take for each leftover count, one column per policy.
        //
        // Rows sit on each side of every crossover in the table of `Pass::next`.
        let table: [(usize, [(Pass, usize); 4]); 16] = [
            (1, [(One, 1), (One, 1), (One, 1), (One, 1)]),
            (2, [(N, 2), (One, 2), (One, 2), (One, 2)]),
            (3, [(N, 3), (S, 3), (One, 3), (S, 3)]),
            (4, [(N, 4), (S, 4), (S, 4), (S, 4)]),
            (8, [(N, 8), (S, 8), (S, 8), (S, 8)]),
            (9, [(N, 9), (N, 9), (N, 9), (S, 9)]),
            (13, [(N, 13), (N, 13), (N, 13), (S, 13)]),
            (14, [(N, 14), (N, 14), (N, 14), (N, 14)]),
            (17, [(N, 16), (N, 16), (N, 16), (N, 16)]),
            (18, [(W, 18), (N, 16), (N, 16), (N, 16)]),
            (20, [(W, 20), (N, 16), (N, 16), (N, 16)]),
            (21, [(W, 21), (W, 21), (W, 21), (S, 21)]),
            (24, [(W, 24), (W, 24), (W, 24), (S, 24)]),
            (25, [(W, 25), (W, 25), (W, 25), (W, 25)]),
            (32, [(W, 32), (W, 32), (W, 32), (W, 32)]),
            (33, [(W, 32), (W, 32), (W, 32), (W, 32)]),
        ];
        for (left, expected) in table {
            for (streams, expected) in ALL_STREAMS.into_iter().zip(expected) {
                assert_eq!(
                    Pass::next(left, streams),
                    expected,
                    "{streams:?}, {left} left"
                );
            }
        }
    }

    #[test]
    fn only_zen4_sends_long_messages_to_the_streams() {
        // A full batch of long messages is all streams on Zen 4, and register passes elsewhere.
        let full = |len, streams| Pass::whole(4 * LANES, len, streams);
        assert_eq!(
            full(ZEN4_STREAMS_FROM_LEN, Streams::Zen4),
            Some(Pass::FourStreams)
        );
        assert_eq!(full(ZEN4_STREAMS_FROM_LEN - 1, Streams::Zen4), None);
        assert_eq!(full(ZEN4_STREAMS_FROM_LEN, Streams::FromThree), None);
        assert_eq!(full(ZEN4_STREAMS_FROM_LEN, Streams::FromFour), None);
        assert_eq!(full(ZEN4_STREAMS_FROM_LEN, Streams::Off), None);

        // A batch too small for a register pass skips the padding, and a CPU without SHA-NI has none.
        assert_eq!(Pass::whole(1, 64, Streams::Zen4), Some(Pass::Single));
        assert_eq!(
            Pass::whole(5, 64, Streams::FromThree),
            Some(Pass::FourStreams)
        );
        assert_eq!(Pass::whole(5, 64, Streams::Off), None);
    }

    proptest! {
        #[test]
        fn the_block_kernel_matches_the_specification_from_any_chaining_value(seed in any::<u64>()) {
            if cpu_avx512::get() {
                // SAFETY: the CPU has AVX-512F and AVX-512BW.
                unsafe {
                    check_block_kernel::<1>(seed)?;
                    check_block_kernel::<GROUPS>(seed)?;
                }
            }
        }

        #[test]
        fn the_shared_kernel_matches_the_specification_from_any_chaining_value(seed in any::<u64>()) {
            if cpu_avx512::get() {
                // SAFETY: the CPU has AVX-512F and AVX-512BW.
                unsafe {
                    check_shared_kernel::<1>(seed)?;
                    check_shared_kernel::<GROUPS>(seed)?;
                }
            }
        }
    }
}
