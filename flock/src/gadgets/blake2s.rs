//! The BLAKE2s compression function `F` (RFC 7693, Section 3.2): ten rounds of eight G's, 184 products per G.
//!
//! Its ports: the counter, the finalization word, the chaining value, the message.

use alloc::vec::Vec;

use super::arx::{G_LANES, IV, g};
use super::{Gadget, pack_u32s, unpack_u32s};
use crate::circuit::{Builder, Circuit, Word};

/// The message word order of each round (RFC 7693, Section 2.7).
pub(crate) const SIGMA: [[usize; 16]; 10] = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
    [14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3],
    [11, 8, 12, 0, 5, 2, 15, 13, 10, 14, 3, 6, 7, 1, 9, 4],
    [7, 9, 3, 1, 13, 12, 11, 14, 2, 6, 5, 10, 4, 0, 15, 8],
    [9, 0, 5, 7, 2, 4, 10, 15, 14, 1, 11, 12, 6, 8, 3, 13],
    [2, 12, 6, 10, 0, 11, 8, 3, 4, 13, 7, 5, 15, 14, 1, 9],
    [12, 5, 1, 15, 14, 13, 4, 10, 0, 7, 6, 3, 9, 2, 8, 11],
    [13, 11, 7, 14, 12, 1, 3, 9, 5, 0, 15, 4, 8, 6, 2, 10],
    [6, 15, 14, 9, 11, 3, 0, 8, 12, 2, 13, 7, 1, 4, 10, 5],
    [10, 2, 8, 4, 7, 6, 1, 5, 15, 11, 9, 14, 3, 12, 13, 0],
];

/// Input port widths in bits: the counter, the finalization word, four words of `h`, eight of `m`.
const INPUT_BITS: [usize; 14] = [64, 32, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64];

/// One compression `compress(h, m, t, f0)`, with the last-node flag `f1` zero.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Blake2sCompress {
    /// The input chaining value.
    pub h: [u32; 8],
    /// The message block, as little-endian words.
    pub m: [u32; 16],
    /// The byte counter.
    pub t: u64,
    /// The finalization flag: all ones on the last block.
    pub f0: u32,
}

impl Gadget for Blake2sCompress {
    /// The counter, the finalization word, `h`, then `m`.
    const INPUT_WORDS: usize = 14;

    /// The new chaining value.
    type Output = [u32; 8];

    fn circuit() -> Circuit {
        let mut c = Builder::new(&INPUT_BITS, &[64; 4]);
        let (t, f0) = (Word::split_u32(c.input(0)), Word::from(c.input(1)));
        let h: Vec<Word> = (2..6).flat_map(|p| Word::split_u32(c.input(p))).collect();
        let m: Vec<Word> = (6..14).flat_map(|p| Word::split_u32(c.input(p))).collect();

        // The working vector: h, then the IV with the counter and the finalization word folded in.
        let mut v: [Word; 16] = core::array::from_fn(|i| match i {
            0..8 => h[i].clone(),
            _ => c.literal(u64::from(IV[i - 8]), 32),
        });
        for (i, x) in [&t[0], &t[1], &f0].into_iter().enumerate() {
            v[12 + i] = c.xor_word(&v[12 + i], x);
        }

        // Ten rounds of eight G's.
        for sigma in &SIGMA {
            for (i, &lanes) in G_LANES.iter().enumerate() {
                g(
                    &mut c,
                    &mut v,
                    lanes,
                    &m[sigma[2 * i]],
                    &m[sigma[2 * i + 1]],
                );
            }
        }

        // The new chaining value: `h ^ v[i] ^ v[i + 8]`.
        for i in 0..8 {
            let hv = c.xor_word(&h[i], &v[i]);
            let out = c.xor_word(&hv, &v[i + 8]);
            for (bit, &wire) in out.iter().enumerate() {
                c.output(i / 2, 32 * (i % 2) + bit, wire);
            }
        }
        c.finish()
    }

    fn write_input(&self, words: &mut [u64]) {
        words[0] = self.t;
        words[1] = u64::from(self.f0);
        pack_u32s(&self.h, &mut words[2..6]);
        pack_u32s(&self.m, &mut words[6..14]);
    }

    fn decode_output(words: &[u64]) -> [u32; 8] {
        unpack_u32s(words)
    }
}
