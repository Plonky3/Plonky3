//! Four-way SHA-256 for wasm32 SIMD128.
//!
//! Each `v128` holds the same SHA-256 word from four independent messages. This keeps the
//! compression rounds lane-independent while making every 32-bit operation four-way parallel.

use core::arch::wasm32::{
    i32x4_add, i32x4_shl, u32x4_shr, u32x4_splat, v128, v128_and, v128_andnot, v128_or, v128_xor,
};
use core::mem::transmute;

use p3_symmetric::{CryptographicHasher, PseudoCompressionFunction};

use crate::{H256_256, Sha256, Sha256Compress};

pub(crate) const LANES: usize = 4;
const BLOCK_BYTES: usize = 64;

const K: [u32; 64] = [
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
];

#[inline(always)]
fn add2(a: v128, b: v128) -> v128 {
    i32x4_add(a, b)
}

#[inline(always)]
fn add4(a: v128, b: v128, c: v128, d: v128) -> v128 {
    add2(add2(a, b), add2(c, d))
}

#[inline(always)]
fn add5(a: v128, b: v128, c: v128, d: v128, e: v128) -> v128 {
    add2(add4(a, b, c, d), e)
}

#[inline(always)]
fn rotr<const N: u32>(x: v128) -> v128 {
    v128_or(u32x4_shr(x, N), i32x4_shl(x, 32 - N))
}

#[inline(always)]
fn small_sigma0(x: v128) -> v128 {
    v128_xor(v128_xor(rotr::<7>(x), rotr::<18>(x)), u32x4_shr(x, 3))
}

#[inline(always)]
fn small_sigma1(x: v128) -> v128 {
    v128_xor(v128_xor(rotr::<17>(x), rotr::<19>(x)), u32x4_shr(x, 10))
}

#[inline(always)]
fn big_sigma0(x: v128) -> v128 {
    v128_xor(v128_xor(rotr::<2>(x), rotr::<13>(x)), rotr::<22>(x))
}

#[inline(always)]
fn big_sigma1(x: v128) -> v128 {
    v128_xor(v128_xor(rotr::<6>(x), rotr::<11>(x)), rotr::<25>(x))
}

#[inline(always)]
fn choose(x: v128, y: v128, z: v128) -> v128 {
    v128_xor(v128_and(x, y), v128_andnot(z, x))
}

#[inline(always)]
fn majority(x: v128, y: v128, z: v128) -> v128 {
    v128_xor(v128_xor(v128_and(x, y), v128_and(x, z)), v128_and(y, z))
}

#[inline]
fn initial_state() -> [v128; 8] {
    H256_256.map(|word| u32x4_splat(word))
}

#[inline]
fn load_word(blocks: [&[u8; BLOCK_BYTES]; LANES], word: usize) -> v128 {
    let offset = word * 4;
    let lanes: [u32; LANES] = core::array::from_fn(|lane| {
        u32::from_be_bytes(blocks[lane][offset..offset + 4].try_into().unwrap())
    });
    // SAFETY: `v128` and `[u32; 4]` are both 16 bytes and every bit pattern is valid.
    unsafe { transmute(lanes) }
}

/// Compress one 64-byte block for each of four independent SHA-256 states.
#[inline]
fn compress_blocks(state: &mut [v128; 8], blocks: [&[u8; BLOCK_BYTES]; LANES]) {
    let mut schedule: [v128; 16] = core::array::from_fn(|word| load_word(blocks, word));
    let [mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut h] = *state;

    for round in 0..64 {
        let schedule_word = if round < 16 {
            schedule[round]
        } else {
            let next = add4(
                small_sigma1(schedule[(round - 2) & 15]),
                schedule[(round - 7) & 15],
                small_sigma0(schedule[(round - 15) & 15]),
                schedule[round & 15],
            );
            schedule[round & 15] = next;
            next
        };

        let t1 = add5(
            h,
            big_sigma1(e),
            choose(e, f, g),
            u32x4_splat(K[round]),
            schedule_word,
        );
        let t2 = add2(big_sigma0(a), majority(a, b, c));

        h = g;
        g = f;
        f = e;
        e = add2(d, t1);
        d = c;
        c = b;
        b = a;
        a = add2(t1, t2);
    }

    state[0] = add2(state[0], a);
    state[1] = add2(state[1], b);
    state[2] = add2(state[2], c);
    state[3] = add2(state[3], d);
    state[4] = add2(state[4], e);
    state[5] = add2(state[5], f);
    state[6] = add2(state[6], g);
    state[7] = add2(state[7], h);
}

#[inline]
fn write_digests(state: &[v128; 8], out: &mut [[u8; 32]; LANES]) {
    for (word_index, &word) in state.iter().enumerate() {
        // SAFETY: `v128` and `[u32; 4]` are both 16 bytes and every bit pattern is valid.
        let lanes: [u32; LANES] = unsafe { transmute(word) };
        for lane in 0..LANES {
            out[lane][word_index * 4..word_index * 4 + 4]
                .copy_from_slice(&lanes[lane].to_be_bytes());
        }
    }
}

fn hash_four(messages: &[u8], len: usize) -> [[u8; 32]; LANES] {
    debug_assert_eq!(messages.len(), len * LANES);
    let lane_messages: [&[u8]; LANES] =
        core::array::from_fn(|lane| &messages[lane * len..(lane + 1) * len]);
    let mut state = initial_state();

    for block in 0..len / BLOCK_BYTES {
        let blocks = core::array::from_fn(|lane| {
            lane_messages[lane][block * BLOCK_BYTES..][..BLOCK_BYTES]
                .try_into()
                .unwrap()
        });
        compress_blocks(&mut state, blocks);
    }

    let remainder = len % BLOCK_BYTES;
    let mut final_blocks = [[0u8; BLOCK_BYTES]; LANES];
    for lane in 0..LANES {
        final_blocks[lane][..remainder].copy_from_slice(&lane_messages[lane][len - remainder..]);
        final_blocks[lane][remainder] = 0x80;
    }

    let bit_len = (len as u64).wrapping_mul(8).to_be_bytes();
    if remainder >= 56 {
        compress_blocks(&mut state, final_blocks.each_ref());
        final_blocks = [[0u8; BLOCK_BYTES]; LANES];
    }
    for block in &mut final_blocks {
        block[56..].copy_from_slice(&bit_len);
    }
    compress_blocks(&mut state, final_blocks.each_ref());

    let mut out = [[0u8; 32]; LANES];
    write_digests(&state, &mut out);
    out
}

pub(crate) fn hash_many(input: &[u8], out: &mut [[u8; 32]]) {
    if out.is_empty() {
        return;
    }
    assert!(
        input.len().is_multiple_of(out.len()),
        "input length ({}) must be a whole multiple of the digest count ({})",
        input.len(),
        out.len()
    );

    let len = input.len() / out.len();
    let full_groups = out.len() / LANES;
    for group in 0..full_groups {
        let message_start = group * LANES * len;
        let digests = hash_four(&input[message_start..message_start + LANES * len], len);
        out[group * LANES..(group + 1) * LANES].copy_from_slice(&digests);
    }

    for message in full_groups * LANES..out.len() {
        out[message] = Sha256.hash_slice(&input[message * len..(message + 1) * len]);
    }
}

pub(crate) fn compress_many(inputs: &[[[u8; 32]; 2]], out: &mut [[u8; 32]]) {
    assert_eq!(
        inputs.len(),
        out.len(),
        "group count ({}) must equal the output count ({})",
        inputs.len(),
        out.len()
    );

    let full_groups = out.len() / LANES;
    for group in 0..full_groups {
        let mut state = initial_state();
        let blocks: [&[u8; BLOCK_BYTES]; LANES] = core::array::from_fn(|lane| {
            // SAFETY: `[[u8; 32]; 2]` and `[u8; 64]` have identical contiguous byte layouts.
            unsafe { transmute(&inputs[group * LANES + lane]) }
        });
        compress_blocks(&mut state, blocks);

        let mut digests = [[0u8; 32]; LANES];
        write_digests(&state, &mut digests);
        out[group * LANES..(group + 1) * LANES].copy_from_slice(&digests);
    }

    for group in full_groups * LANES..out.len() {
        out[group] = Sha256Compress.compress(inputs[group]);
    }
}
