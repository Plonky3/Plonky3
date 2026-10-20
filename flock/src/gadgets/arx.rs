//! The add-rotate-XOR core BLAKE2s and BLAKE3 share: the IV, the G function and its lanes.

use crate::circuit::{Builder, Wire, Word};

/// The initial value of BLAKE2s (RFC 7693, Section 2.6) and BLAKE3, the SHA-256 IV.
pub(crate) const IV: [u32; 8] = [
    0x6A09_E667,
    0xBB67_AE85,
    0x3C6E_F372,
    0xA54F_F53A,
    0x510E_527F,
    0x9B05_688C,
    0x1F83_D9AB,
    0x5BE0_CD19,
];

/// The four columns, then the four diagonals, as `[a, b, c, d]` lanes.
pub(crate) const G_LANES: [[usize; 4]; 8] = [
    [0, 4, 8, 12],
    [1, 5, 9, 13],
    [2, 6, 10, 14],
    [3, 7, 11, 15],
    [0, 5, 10, 15],
    [1, 6, 11, 12],
    [2, 7, 8, 13],
    [3, 4, 9, 14],
];

/// One G on the working state (RFC 7693, Section 3.1): two halves, each a three-operand and a two-operand sum.
pub(crate) fn g(
    c: &mut Builder,
    v: &mut [Word; 16],
    [a, b, cc, d]: [usize; 4],
    mx: &[Wire],
    my: &[Wire],
) {
    for (m, r1, r2) in [(mx, 16, 12), (my, 8, 7)] {
        v[a] = c.wrapping_sum(&[&v[a], &v[b], m]);
        v[d] = c.xor_word(&v[d], &v[a]).rotate_right(r1);
        v[cc] = c.wrapping_sum(&[&v[cc], &v[d]]);
        v[b] = c.xor_word(&v[b], &v[cc]).rotate_right(r2);
    }
}
