//! The Keccak-f\[1600\] permutation (FIPS 202, Sections 3.3 and 3.4): 24 rounds, chi's 1600 products each the only nonlinear step.

use hex_literal::hex;

use super::Gadget;
use super::rows::Rows;
use crate::Witness;
use crate::circuit::{Builder, Circuit, Word};

/// The iota constants, one per round (FIPS 202, Section 3.2.5).
pub(crate) const RC: [u64; 24] = [
    0x0000_0000_0000_0001,
    0x0000_0000_0000_8082,
    0x8000_0000_0000_808A,
    0x8000_0000_8000_8000,
    0x0000_0000_0000_808B,
    0x0000_0000_8000_0001,
    0x8000_0000_8000_8081,
    0x8000_0000_0000_8009,
    0x0000_0000_0000_008A,
    0x0000_0000_0000_0088,
    0x0000_0000_8000_8009,
    0x0000_0000_8000_000A,
    0x0000_0000_8000_808B,
    0x8000_0000_0000_008B,
    0x8000_0000_0000_8089,
    0x8000_0000_0000_8003,
    0x8000_0000_0000_8002,
    0x8000_0000_0000_0080,
    0x0000_0000_0000_800A,
    0x8000_0000_8000_000A,
    0x8000_0000_8000_8081,
    0x8000_0000_0000_8080,
    0x0000_0000_8000_0001,
    0x8000_0000_8000_8008,
];

/// The rho rotation of lane `x + 5 y` (FIPS 202, Section 3.2.2).
pub(crate) const RHO: [usize; 25] = [
    0, 1, 62, 28, 27, 36, 44, 6, 55, 20, 3, 10, 43, 25, 39, 41, 45, 15, 21, 8, 18, 2, 61, 56, 14,
];

/// Slot of the constant: after 25 input and 25 output words.
const CONST_POS: usize = 64 * 50;

/// Digest of the circuit, so the word-level generator refuses any other.
const CIRCUIT_DIGEST: [u8; 32] =
    hex!("73303598c49f7c4cec25734ba48ecab457848a7f1b376eed15981f8acbd592c9");

/// One permutation of a state of 25 lanes, lane `x + 5 y` at index `x + 5 y`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct KeccakF1600(pub [u64; 25]);

impl Gadget for KeccakF1600 {
    /// The lanes in order.
    const INPUT_WORDS: usize = 25;

    /// The permuted lanes.
    type Output = [u64; 25];

    fn circuit() -> Circuit {
        let mut c = Builder::new(&[1600], &[1600]);
        let input = c.input(0);
        let mut a: [Word; 25] = core::array::from_fn(|i| Word::from(&input[64 * i..64 * (i + 1)]));

        for &rc in &RC {
            // Theta: each lane takes two neighbouring column parities.
            let col: [Word; 5] = core::array::from_fn(|x| {
                (1..5).fold(a[x].clone(), |p, y| c.xor_word(&p, &a[x + 5 * y]))
            });
            for x in 0..5 {
                let d = c.xor_word(&col[(x + 4) % 5], &col[(x + 1) % 5].rotate_left(1));
                for y in 0..5 {
                    a[x + 5 * y] = c.xor_word(&a[x + 5 * y], &d);
                }
            }

            // Rho and pi: lane `(x, y)` rotates into `(y, 2x + 3y)`.
            let mut b: [Word; 25] = core::array::from_fn(|_| Word::default());
            for x in 0..5 {
                for y in 0..5 {
                    b[y + 5 * ((2 * x + 3 * y) % 5)] = a[x + 5 * y].rotate_left(RHO[x + 5 * y]);
                }
            }

            // Chi: one product per bit.
            for y in 0..5 {
                for x in 0..5 {
                    let row = |dx: usize| &b[(x + dx) % 5 + 5 * y];
                    let t = c.andn_word(row(1), row(2));
                    a[x + 5 * y] = c.xor_word(row(0), &t);
                }
            }

            // Iota: a constant, free.
            let k = c.literal(rc, 64);
            a[0] = c.xor_word(&a[0], &k);
        }

        for (i, lane) in a.iter().enumerate() {
            for (bit, &wire) in lane.iter().enumerate() {
                c.output(0, 64 * i + bit, wire);
            }
        }
        c.finish()
    }

    fn write_input(&self, words: &mut [u64]) {
        words.copy_from_slice(&self.0);
    }

    fn decode_output(words: &[u64]) -> [u64; 25] {
        core::array::from_fn(|i| words[i])
    }

    fn write_output(output: &[u64; 25], words: &mut [u64]) {
        words.copy_from_slice(output);
    }

    fn witness(circuit: &Circuit, log_instances: usize, rows: &[Self], padding: &Self) -> Witness {
        assert_eq!(
            circuit.digest(),
            &CIRCUIT_DIGEST,
            "the circuit is not this gadget's"
        );
        Witness::by_instance(16, log_instances, rows, padding, Self::rows)
    }
}

impl KeccakF1600 {
    /// One instance's `A z` and `B z`, computed on lanes.
    fn rows(&self, az: &mut [u64], bz: &mut [u64]) {
        let mut rows = Rows::new(az, bz, CONST_POS / 64);
        rows.push(1, 1, 1);
        let mut a = self.0;
        for &rc in &RC {
            // Theta.
            let col: [u64; 5] =
                core::array::from_fn(|x| a[x] ^ a[x + 5] ^ a[x + 10] ^ a[x + 15] ^ a[x + 20]);
            for x in 0..5 {
                let d = col[(x + 4) % 5] ^ col[(x + 1) % 5].rotate_left(1);
                for y in 0..5 {
                    a[x + 5 * y] ^= d;
                }
            }

            // Rho and pi.
            let mut b = [0; 25];
            for x in 0..5 {
                for y in 0..5 {
                    b[y + 5 * ((2 * x + 3 * y) % 5)] =
                        a[x + 5 * y].rotate_left(RHO[x + 5 * y] as u32);
                }
            }

            // Chi: one row `!b1 * b2` per bit.
            for y in 0..5 {
                for x in 0..5 {
                    let (b1, b2) = (!b[(x + 1) % 5 + 5 * y], b[(x + 2) % 5 + 5 * y]);
                    rows.push(b1, b2, 64);
                    a[x + 5 * y] = b[x + 5 * y] ^ (b1 & b2);
                }
            }

            // Iota.
            a[0] ^= rc;
        }
        for (w, (&lane, &output)) in self.0.iter().zip(&a).enumerate() {
            rows.port(w, lane, u64::MAX);
            rows.port(25 + w, output, u64::MAX);
        }
        rows.finish();
    }
}
