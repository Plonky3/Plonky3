//! One-byte subfield butterfly on NEON.
//!
//! Multiplying by an element of GF(2^8) is GF(2)-linear on each byte, so it splits over the
//! two nibbles: `t.x = LOW[x & 0xf] + HIGH[x >> 4]`. Each half is one `TBL` lookup.

use core::arch::aarch64::{
    uint8x16_t, vandq_u8, vdupq_n_u8, veorq_u8, vld1q_u8, vqtbl1q_u8, vshrq_n_u8, vst1q_u8,
};

/// Bytes in one register.
pub(crate) const REGISTER_BYTES: usize = 16;

/// Shortest run, in bytes, worth building the tables for. To be tuned with the benchmarks.
pub(crate) const MIN_BYTES: usize = 256;

/// The nibble tables of a multiplier, from its products with the basis elements `1 << b`.
#[inline]
fn nibble_tables(columns: &[u8; 8]) -> ([u8; 16], [u8; 16]) {
    let mut low = [0u8; 16];
    let mut high = [0u8; 16];
    for b in 0..4 {
        for n in 0..(1usize << b) {
            low[n | (1 << b)] = low[n] ^ columns[b];
            high[n | (1 << b)] = high[n] ^ columns[4 + b];
        }
    }
    (low, high)
}

/// Butterfly over the whole registers of two byte runs, returning the bytes covered.
///
/// # Panics
/// Panics if the runs differ in length.
#[inline]
pub(crate) fn byte_butterfly<const INVERSE: bool>(
    lo: &mut [u8],
    hi: &mut [u8],
    columns: &[u8; 8],
) -> usize {
    assert_eq!(lo.len(), hi.len(), "butterfly lengths differ");

    let (low, high) = nibble_tables(columns);
    let (lo, _) = lo.as_chunks_mut::<REGISTER_BYTES>();
    let (hi, _) = hi.as_chunks_mut::<REGISTER_BYTES>();
    let covered = lo.len() * REGISTER_BYTES;

    // SAFETY: `neon` is enabled for this module, the tables are 16 bytes and each chunk is
    // exactly one register. The loads and stores need no alignment.
    unsafe {
        let low = vld1q_u8(low.as_ptr());
        let high = vld1q_u8(high.as_ptr());
        let nibble = vdupq_n_u8(0x0f);
        let map = |x: uint8x16_t| {
            veorq_u8(
                vqtbl1q_u8(low, vandq_u8(x, nibble)),
                vqtbl1q_u8(high, vshrq_n_u8::<4>(x)),
            )
        };

        for (lo, hi) in lo.iter_mut().zip(hi.iter_mut()) {
            let a = vld1q_u8(lo.as_ptr());
            let b = vld1q_u8(hi.as_ptr());

            if INVERSE {
                let b = veorq_u8(b, a);
                vst1q_u8(hi.as_mut_ptr(), b);
                vst1q_u8(lo.as_mut_ptr(), veorq_u8(a, map(b)));
            } else {
                let a = veorq_u8(a, map(b));
                vst1q_u8(lo.as_mut_ptr(), a);
                vst1q_u8(hi.as_mut_ptr(), veorq_u8(b, a));
            }
        }
    }
    covered
}
