//! Four-way SHA-256 for wasm32 SIMD128.
//!
//! Each `v128` holds the same SHA-256 word from four independent messages. This keeps the
//! compression rounds lane-independent while making every 32-bit operation four-way parallel.
//!
//! Padding and batching live in [`crate::four_lane`]; this module supplies the vector core.

use core::arch::wasm32::{
    i32x4_add, i32x4_shl, u32x4_shr, u32x4_splat, v128, v128_and, v128_andnot, v128_or, v128_xor,
};
use core::mem::transmute;

use crate::H256_256;
use crate::four_lane::{BLOCK_BYTES, FourLane, LANES, ROUND_CONSTANTS};

/// The four-lane SIMD128 backend.
pub(crate) struct Simd128;

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
fn load_word(blocks: [&[u8; BLOCK_BYTES]; LANES], word: usize) -> v128 {
    let offset = word * 4;
    let lanes: [u32; LANES] = core::array::from_fn(|lane| {
        u32::from_be_bytes(blocks[lane][offset..offset + 4].try_into().unwrap())
    });
    // SAFETY: `v128` and `[u32; 4]` are both 16 bytes and every bit pattern is valid.
    unsafe { transmute(lanes) }
}

impl FourLane for Simd128 {
    /// One vector per state word, each holding that word for all four messages.
    type State = [v128; 8];

    #[inline]
    fn initial_state() -> Self::State {
        H256_256.map(|word| u32x4_splat(word))
    }

    #[inline]
    fn compress(state: &mut Self::State, blocks: [&[u8; BLOCK_BYTES]; LANES]) {
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
                u32x4_splat(ROUND_CONSTANTS[round]),
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
    fn write_digests(state: &Self::State, out: &mut [[u8; 32]; LANES]) {
        for (word_index, &word) in state.iter().enumerate() {
            // SAFETY: `v128` and `[u32; 4]` are both 16 bytes and every bit pattern is valid.
            let lanes: [u32; LANES] = unsafe { transmute(word) };
            for lane in 0..LANES {
                out[lane][word_index * 4..word_index * 4 + 4]
                    .copy_from_slice(&lanes[lane].to_be_bytes());
            }
        }
    }
}
