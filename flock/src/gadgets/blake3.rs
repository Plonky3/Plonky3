//! The BLAKE3 compression function (BLAKE3 specification, Section 2.2): seven rounds of eight G's, 184 products per G.

use hex_literal::hex;

use super::arx::{G_LANES, IV, g, g_rows};
use super::rows::Rows;
use super::{Gadget, pack_u32s, unpack_u32s};
use crate::Witness;
use crate::circuit::{Builder, Circuit, Word};

/// The message word permutation between rounds (BLAKE3 specification, Table 2).
pub(crate) const MSG_PERMUTATION: [usize; 16] =
    [2, 6, 3, 10, 7, 0, 4, 13, 1, 11, 12, 5, 9, 14, 15, 8];

/// Rounds.
const ROUNDS: usize = 7;

/// Slot of the constant: after 14 input and 8 output words.
const CONST_POS: usize = 64 * 22;

/// Digest of the circuit, so the word-level generator refuses any other.
const CIRCUIT_DIGEST: [u8; 32] =
    hex!("0aaec33420c391e22858a6d2151938788105388524dc602b260d0ad03d856ffe");

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

    fn write_output(output: &[u32; 16], words: &mut [u64]) {
        pack_u32s(output, words);
    }

    fn witness(circuit: &Circuit, log_instances: usize, rows: &[Self], padding: &Self) -> Witness {
        assert_eq!(
            circuit.digest(),
            &CIRCUIT_DIGEST,
            "the circuit is not this gadget's"
        );
        Witness::by_instance(14, log_instances, rows, padding, Self::rows)
    }
}

impl Blake3Compress {
    /// One instance's `A z` and `B z`, computed on words.
    fn rows(&self, az: &mut [u64], bz: &mut [u64]) {
        let mut rows = Rows::new(az, bz, CONST_POS / 64);
        rows.push(1, 1, 1);
        let mut v: [u32; 16] = core::array::from_fn(|i| match i {
            0..8 => self.cv[i],
            8..12 => IV[i - 8],
            12 => self.counter as u32,
            13 => (self.counter >> 32) as u32,
            14 => self.block_len,
            _ => self.flags,
        });
        let mut literal = 0x0F00;
        let mut m = self.block;
        for _ in 0..ROUNDS {
            for (i, &lanes) in G_LANES.iter().enumerate() {
                g_rows(
                    &mut rows,
                    &mut v,
                    &mut literal,
                    lanes,
                    m[2 * i],
                    m[2 * i + 1],
                );
            }
            m = core::array::from_fn(|i| m[MSG_PERMUTATION[i]]);
        }

        // Ports: the input words, then the output words.
        let mut words = [0; 14];
        self.write_input(&mut words);
        for (w, &word) in words.iter().enumerate() {
            rows.port(w, word, u64::MAX);
        }
        let out: [u32; 16] = core::array::from_fn(|i| {
            if i < 8 {
                v[i] ^ v[i + 8]
            } else {
                v[i] ^ self.cv[i - 8]
            }
        });
        for (w, pair) in out.as_chunks::<2>().0.iter().enumerate() {
            rows.port(
                14 + w,
                u64::from(pair[0]) | (u64::from(pair[1]) << 32),
                u64::MAX,
            );
        }
        rows.finish();
    }
}
