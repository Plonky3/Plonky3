//! The add-rotate-XOR core BLAKE2s and BLAKE3 share: the IV, the G function and its lanes.

use super::rows::Rows;
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

/// One G on words, appending its rows; bit `i` of `literal` marks lane `i` still a constant.
#[inline(always)]
pub(crate) fn g_rows(
    rows: &mut Rows<'_>,
    v: &mut [u32; 16],
    literal: &mut u16,
    [a, b, c, d]: [usize; 4],
    mx: u32,
    my: u32,
) {
    for (m, r1, r2) in [(mx, 16, 12), (my, 8, 7)] {
        debug_assert_eq!(
            *literal & ((1 << a) | (1 << b)),
            0,
            "a summed lane is variable"
        );
        v[a] = rows.add3(v[a], v[b], m);
        v[d] = (v[d] ^ v[a]).rotate_right(r1);
        // An IV lane is a constant operand of its first sum.
        v[c] = if *literal & (1 << c) != 0 {
            rows.add_const(v[c], v[d])
        } else {
            rows.add2(v[c], v[d])
        };
        *literal &= !((1 << c) | (1 << d));
        v[b] = (v[b] ^ v[c]).rotate_right(r2);
    }
}
