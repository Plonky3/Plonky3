//! 32-way SHA-256 for x86-64 AVX-512.
//!
//! A register holds the same word of sixteen messages, one per 32-bit lane.
//! Two groups of registers run side by side, so one compression advances 32 messages.
//!
//! Equal-length messages share their block count and their padding.
//! So every lane runs the same rounds, and a block of padding only is the same in every lane.

mod rounds;

use core::arch::x86_64::*;

use p3_symmetric::{CryptographicHasher, PseudoCompressionFunction};

use self::rounds::{K, compress_blocks, compress_shared};
use crate::{H256_256, Sha256, Sha256Compress};

/// Lanes in one register.
const WIDTH: usize = 16;

/// Register groups advanced together.
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

/// A short final group of fewer messages than this hashes them one at a time instead.
///
/// With SHA-NI in the build, one message costs a small fraction of a full group.
/// Without it, one message costs about as much as a whole group.
const ONE_AT_A_TIME_BELOW: usize = if cfg!(target_feature = "sha") {
    LANES / 2
} else {
    2
};

/// Byte permutation reversing each 32-bit word of a 128-bit block.
///
/// SHA-256 reads words big-endian, while a vector load is little-endian.
///
/// The permutation is its own inverse, so it serves loads and stores.
const REVERSE_WORD_BYTES: [u8; 16] = [3, 2, 1, 0, 7, 6, 5, 4, 11, 10, 9, 8, 15, 14, 13, 12];

/// Chaining values of every lane of both groups.
type State = [[__m512i; STATE_WORDS]; GROUPS];

/// Message words of every lane of both groups.
type Block = [[__m512i; BLOCK_WORDS]; GROUPS];

/// Hash `out.len()` equal-length messages laid end to end in `input`.
///
/// # Panics
///
/// Panics if the input length is not a whole multiple of the digest count.
pub(crate) fn hash_many(input: &[u8], out: &mut [[u8; 32]]) {
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
    let padding = Padding::new(len);

    // Whole groups of 32 messages.
    let (groups, rest) = out.as_chunks_mut::<LANES>();
    for (index, digests) in groups.iter_mut().enumerate() {
        let lanes = Lanes {
            input,
            len,
            first: index * LANES,
            count: LANES,
        };
        hash_group(&lanes, &padding, digests);
    }

    // A short final group costs a full compression per block, however few messages it holds.
    let first = groups.len() * LANES;
    if rest.len() < ONE_AT_A_TIME_BELOW {
        for (message, digest) in (first..).zip(&mut *rest) {
            *digest = Sha256.hash_slice(&input[message * len..][..len]);
        }
    } else {
        // The spare lanes repeat the last message, and their digests are never written out.
        let lanes = Lanes {
            input,
            len,
            first,
            count: rest.len(),
        };
        let mut digests = [[0u8; 32]; LANES];
        hash_group(&lanes, &padding, &mut digests);
        rest.copy_from_slice(&digests[..rest.len()]);
    }
}

/// Compress each 64-byte pair from the initial hash value, without padding.
///
/// # Panics
///
/// Panics if the input and output counts differ.
pub(crate) fn compress_many(inputs: &[[[u8; 32]; 2]], out: &mut [[u8; 32]]) {
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
    let (groups, rest) = out.as_chunks_mut::<LANES>();
    for (index, digests) in groups.iter_mut().enumerate() {
        let rows = core::array::from_fn(|lane| &blocks[index * LANES + lane]);
        compress_group(&rows, digests);
    }

    // The short final group, as in the hash.
    let first = groups.len() * LANES;
    if rest.len() < ONE_AT_A_TIME_BELOW {
        for (input, digest) in inputs[first..].iter().zip(&mut *rest) {
            *digest = Sha256Compress.compress(*input);
        }
    } else {
        // The spare lanes repeat the last block, and their digests are never written out.
        let rows = core::array::from_fn(|lane| &blocks[first + lane.min(rest.len() - 1)]);
        let mut digests = [[0u8; 32]; LANES];
        compress_group(&rows, &mut digests);
        rest.copy_from_slice(&digests[..rest.len()]);
    }
}

/// One block per lane, from the initial hash value.
#[inline]
fn compress_group(rows: &[&[u8; BLOCK_BYTES]; LANES], out: &mut [[u8; 32]; LANES]) {
    // No padding: the digest is the chaining value after this single block.
    let mut state = initial_state();
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
    /// Distinct messages in the group, between 1 and 32.
    count: usize,
}

impl Lanes<'_> {
    /// Byte offset of the message of lane `lane` in the batch.
    #[inline(always)]
    fn start(&self, lane: usize) -> usize {
        // A spare lane points at the last message again.
        (self.first + lane.min(self.count - 1)) * self.len
    }

    /// The whole block at byte `offset` of the message of every lane.
    ///
    /// # Panics
    ///
    /// Panics if a lane has fewer than `offset + 64` bytes left in the batch.
    #[inline(always)]
    fn rows(&self, offset: usize) -> [&[u8; BLOCK_BYTES]; LANES] {
        core::array::from_fn(|lane| {
            self.input[self.start(lane) + offset..][..BLOCK_BYTES]
                .try_into()
                .unwrap()
        })
    }
}

/// Hash the message of every lane.
fn hash_group(lanes: &Lanes<'_>, padding: &Padding, out: &mut [[u8; 32]; LANES]) {
    let mut state = initial_state();

    // Every whole block of message bytes, then the last partial one if any.
    //
    // One call site keeps a single copy of the kernel in the loop.
    let whole_blocks = lanes.len / BLOCK_BYTES;
    let blocks = whole_blocks + usize::from(padding.tail > 0);
    for index in 0..blocks {
        let offset = index * BLOCK_BYTES;
        let block = if index < whole_blocks {
            load_block(&lanes.rows(offset))
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

/// The padded last block of every lane, holding its final `padding.tail` message bytes.
#[inline(never)]
fn last_block(lanes: &Lanes<'_>, offset: usize, padding: &Padding) -> Block {
    let tail = padding.tail;

    // Every lane's message ends inside the batch, the last lane's furthest in.
    assert!(lanes.start(LANES - 1) + offset + tail <= lanes.input.len());

    // A masked load reads the tail bytes and zeroes the rest of the row.
    //
    // Masked-off bytes are never accessed, so no read goes past a message end.
    let mask: __mmask64 = (1 << tail) - 1;
    let mut block: Block = core::array::from_fn(|g| {
        transpose_rows(core::array::from_fn(|l| {
            // SAFETY:
            // - the module's gate enables AVX-512BW, which byte-masked loads need;
            // - the `tail` enabled bytes lie inside the batch, by the assertion above.
            unsafe {
                let row = lanes
                    .input
                    .as_ptr()
                    .add(lanes.start(WIDTH * g + l) + offset);
                _mm512_maskz_loadu_epi8(mask, row.cast())
            }
        }))
    });

    // Words are big-endian, so the first message byte is the top byte of its word.
    //
    // The marker takes the byte right after the message.
    //
    //     tail = 6:  word 0 holds 4 message bytes, word 1 holds 2 and then 0x80
    let marker = splat(0x8000_0000 >> (8 * (tail % 4)));
    for group in &mut block {
        // SAFETY: this module only compiles when the target enables AVX-512F.
        group[tail / 4] = unsafe { _mm512_or_si512(group[tail / 4], marker) };

        // The length joins this block only when eight bytes are left after the marker.
        if padding.shared.is_none() {
            group[14] = splat((padding.bit_len >> 32) as u32);
            group[15] = splat(padding.bit_len as u32);
        }
    }
    block
}

/// The initial hash value of FIPS 180-4 section 5.3.3, in every lane.
#[inline(always)]
fn initial_state() -> State {
    // Every lane of every group starts from the same eight words.
    [H256_256.map(splat); GROUPS]
}

/// The same word in every lane.
#[inline(always)]
fn splat(word: u32) -> __m512i {
    // SAFETY: this module only compiles when the target enables AVX-512F.
    unsafe { _mm512_set1_epi32(word as i32) }
}

/// Reverse the bytes of every 32-bit word.
#[inline(always)]
fn byte_swap(x: __m512i) -> __m512i {
    // SAFETY: `[u8; 16]` and `__m128i` are both 16 bytes and every bit pattern is valid.
    //
    // `vpshufb` on 512 bits needs AVX-512BW, which the module's gate requires.
    unsafe {
        let mask = core::mem::transmute::<[u8; 16], __m128i>(REVERSE_WORD_BYTES);
        _mm512_shuffle_epi8(x, _mm512_broadcast_i32x4(mask))
    }
}

/// A 4 x 4 transpose inside every 128-bit block of four rows.
///
/// Block `k` of output `j` holds word `4k + j` of rows `a`, `b`, `c` and `d`, in that order.
#[inline(always)]
fn transpose_blocks(a: __m512i, b: __m512i, c: __m512i, d: __m512i) -> [__m512i; 4] {
    // SAFETY: this module only compiles when the target enables AVX-512F.
    unsafe {
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
}

/// Load one block from each lane as the big-endian message words of both groups.
///
/// - Row `l` is one block of the message in lane `l`.
/// - Word `w` of group `g` holds word `w` of the messages in lanes `16g` to `16g + 15`.
#[inline(always)]
fn load_block(rows: &[&[u8; BLOCK_BYTES]; LANES]) -> Block {
    let (groups, _) = rows.as_chunks::<WIDTH>();
    core::array::from_fn(|g| {
        // SAFETY: each row is 64 readable bytes, and the load has no alignment requirement.
        transpose_rows(groups[g].map(|row| unsafe { _mm512_loadu_si512(row.as_ptr().cast()) }))
    })
}

/// Turn sixteen rows of one block each into sixteen big-endian message words.
///
/// Word `w` of the result holds word `w` of every row, row `l` in lane `l`.
#[inline(always)]
fn transpose_rows(rows: [__m512i; WIDTH]) -> [__m512i; BLOCK_WORDS] {
    let r = rows.map(byte_swap);

    // Phase 1: block k of u[q][j] is word 4k + j of rows 4q to 4q + 3.
    let u: [[__m512i; 4]; 4] = core::array::from_fn(|q| {
        transpose_blocks(r[4 * q], r[4 * q + 1], r[4 * q + 2], r[4 * q + 3])
    });

    // Phase 2: word 4k + j gathers block k of u[0][j] to u[3][j], in row order.
    let mut words = [r[0]; BLOCK_WORDS];
    for j in 0..4 {
        // SAFETY: this module only compiles when the target enables AVX-512F.
        unsafe {
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
    }
    words
}

/// Write the big-endian digest of every lane of both groups.
///
/// Two adjacent digests fill one 64-byte store, so eight stores cover a group.
#[inline(always)]
fn store_digests(state: &State, out: &mut [[u8; 32]; LANES]) {
    let (groups, _) = out.as_chunks_mut::<WIDTH>();
    for (state, out) in state.iter().zip(groups) {
        let s = state.map(byte_swap);

        // Block k of lo[j] is words 0 to 3 of lane 4k + j, and hi[j] holds words 4 to 7.
        let lo = transpose_blocks(s[0], s[1], s[2], s[3]);
        let hi = transpose_blocks(s[4], s[5], s[6], s[7]);

        for j in [0, 2] {
            // SAFETY: this module only compiles when the target enables AVX-512F.
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
    use proptest::prelude::*;

    use super::*;
    use crate::tests::spec_compress;

    /// A deterministic stream of words, so a failing case reproduces exactly.
    fn words(mut seed: u64) -> impl Iterator<Item = u32> {
        core::iter::repeat_with(move || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed as u32
        })
    }

    /// One chaining value per lane, packed into both groups: lane `16g + l` is lane `l` of group `g`.
    fn pack(states: &[[u32; STATE_WORDS]; LANES]) -> State {
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

    proptest! {
        #[test]
        fn the_block_kernel_matches_the_specification_from_any_chaining_value(seed in any::<u64>()) {
            // Random chaining values and blocks, different in every lane.
            let mut stream = words(seed | 1);
            let states: [[u32; STATE_WORDS]; LANES] =
                core::array::from_fn(|_| core::array::from_fn(|_| stream.next().unwrap()));
            let blocks: [[u8; BLOCK_BYTES]; LANES] = core::array::from_fn(|_| {
                let words: [u32; BLOCK_WORDS] = core::array::from_fn(|_| stream.next().unwrap());
                *words.map(u32::to_le_bytes).as_flattened().as_array().unwrap()
            });

            // Load, compress and store through the vector path.
            let mut state = pack(&states);
            compress_blocks(&mut state, &load_block(&blocks.each_ref()));
            let mut out = [[0u8; 32]; LANES];
            store_digests(&state, &mut out);

            // Every lane must match one compression of FIPS 180-4 section 6.2.2.
            for lane in 0..LANES {
                let mut expected = states[lane];
                spec_compress(&mut expected, &blocks[lane]);
                prop_assert_eq!(out[lane], digest(expected), "lane {}", lane);
            }
        }

        #[test]
        fn the_shared_kernel_matches_the_specification_from_any_chaining_value(seed in any::<u64>()) {
            // Random chaining values in every lane, and one random block for all of them.
            let mut stream = words(seed | 1);
            let states: [[u32; STATE_WORDS]; LANES] =
                core::array::from_fn(|_| core::array::from_fn(|_| stream.next().unwrap()));
            let block: [u32; BLOCK_WORDS] = core::array::from_fn(|_| stream.next().unwrap());

            // The schedule is expanded once, in scalar code, and broadcast.
            let mut state = pack(&states);
            compress_shared(&mut state, &shared_schedule(&block));
            let mut out = [[0u8; 32]; LANES];
            store_digests(&state, &mut out);

            // The specification reads the block as big-endian bytes.
            let bytes = *block.map(u32::to_be_bytes).as_flattened().as_array().unwrap();
            for lane in 0..LANES {
                let mut expected = states[lane];
                spec_compress(&mut expected, &bytes);
                prop_assert_eq!(out[lane], digest(expected), "lane {}", lane);
            }
        }
    }
}
