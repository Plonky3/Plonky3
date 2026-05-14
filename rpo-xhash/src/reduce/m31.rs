//! Mersenne31 reduction: u64 → u32 mod P, where P = 2^31 - 1.
//!
//! Uses the identity 2^31 ≡ 1 (mod P): split into low 31 bits and high bits, add.

pub const P: u64 = (1u64 << 31) - 1;

/// Reduce a u64 to [0, P]. Two rounds of shift-and-add.
#[inline(always)]
pub fn reduce(x: u64) -> u32 {
    let mut r = (x & P) + (x >> 31);
    r = (r & P) + (r >> 31);
    r as u32
}
