//! The BLAKE3 compression function (BLAKE3 specification, Section 2.2): seven rounds of eight G's, 184 products per G.

use super::arx::{G_LANES, IV, g};
use super::{Gadget, pack_u32s, unpack_u32s};
use crate::circuit::{Builder, Circuit, Word};

/// The message word permutation between rounds (BLAKE3 specification, Table 2).
pub(crate) const MSG_PERMUTATION: [usize; 16] =
    [2, 6, 3, 10, 7, 0, 4, 13, 1, 11, 12, 5, 9, 14, 15, 8];

/// Rounds.
const ROUNDS: usize = 7;

/// One compression: `compress(cv, block, counter, block_len, flags)`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Blake3Compress {
    /// The input chaining value.
    pub cv: [u32; 8],
    /// The message block, as little-endian words.
    pub block: [u32; 16],
    /// The chunk counter.
    pub counter: u64,
    /// Bytes used in the block.
    pub block_len: u32,
    /// Domain flags.
    pub flags: u32,
}

impl Gadget for Blake3Compress {
    /// The chaining value, the block, the counter, then `block_len | flags << 32`.
    const INPUT_WORDS: usize = 14;

    /// The sixteen output words.
    type Output = [u32; 16];

    fn circuit() -> Circuit {
        let mut c = Builder::new(&[256, 512, 64, 64], &[512]);
        let cv = Word::split_u32(c.input(0));
        let mut m = Word::split_u32(c.input(1));
        let (counter, len_flags) = (Word::split_u32(c.input(2)), Word::split_u32(c.input(3)));

        // The working state: the chaining value, the IV's first half, the counter, length and flags.
        let mut v: [Word; 16] = core::array::from_fn(|i| match i {
            0..8 => cv[i].clone(),
            8..12 => c.literal(u64::from(IV[i - 8]), 32),
            12 | 13 => counter[i - 12].clone(),
            _ => len_flags[i - 14].clone(),
        });

        // Seven rounds, the message permuted after each.
        for round in 0..ROUNDS {
            for (i, &lanes) in G_LANES.iter().enumerate() {
                g(&mut c, &mut v, lanes, &m[2 * i], &m[2 * i + 1]);
            }
            if round + 1 < ROUNDS {
                m = MSG_PERMUTATION.iter().map(|&j| m[j].clone()).collect();
            }
        }

        // The output: `v[i] ^ v[i + 8]`, then `v[i + 8] ^ cv[i]`.
        for i in 0..8 {
            let lo = c.xor_word(&v[i], &v[i + 8]);
            let hi = c.xor_word(&v[i + 8], &cv[i]);
            for (bit, (&lo, &hi)) in lo.iter().zip(&hi).enumerate() {
                c.output(0, 32 * i + bit, lo);
                c.output(0, 256 + 32 * i + bit, hi);
            }
        }
        c.finish()
    }

    fn write_input(&self, words: &mut [u64]) {
        pack_u32s(&self.cv, &mut words[..4]);
        pack_u32s(&self.block, &mut words[4..12]);
        words[12] = self.counter;
        pack_u32s(&[self.block_len, self.flags], &mut words[13..]);
    }

    fn decode_output(words: &[u64]) -> [u32; 16] {
        unpack_u32s(words)
    }
}
