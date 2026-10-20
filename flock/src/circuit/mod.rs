//! Boolean circuits as gate lists: an R1CS over GF(2) whose matrices are never built.
//!
//! One instance's witness has `2^k_log` bits: input ports, output ports, the constant one, the products, zero padding.

mod builder;
mod wire;

use alloc::vec::Vec;

pub use builder::Builder;
use p3_keccak::Keccak256Hash;
use p3_maybe_rayon::prelude::*;
use p3_symmetric::CryptographicHasher;
pub use wire::{Wire, Word};

/// Gates a digest chunk encodes and hashes on its own.
const DIGEST_GATES: usize = 1 << 12;

/// One gate; a gate's index is its wire.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Gate {
    /// Committed free wire at a slot, with row `z[s] * 1 = z[s]`.
    Free(u32),
    /// Uncommitted sum of two wires: no row.
    Xor(u32, u32),
    /// Product committed at a slot, with row `x * y = z[s]`.
    And(u32, u32, u32),
    /// Affine wire committed at a slot, with row `x * 1 = z[s]`.
    Copy(u32, u32),
}

/// A slot or gate index as stored in a gate.
///
/// # Panics
///
/// - The index does not fit in 32 bits.
fn slot(index: usize) -> u32 {
    u32::try_from(index).expect("circuit index fits in 32 bits")
}

/// A finished gate list.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Circuit {
    /// Gates in order.
    gates: Vec<Gate>,
    /// Slot of the constant one.
    const_pos: usize,
    /// Bits per instance, as a power of two.
    k_log: usize,
    /// Slots before the zero padding.
    useful_bits: usize,
    /// Words the input ports span.
    n_input_words: usize,
    /// Keccak-256 of the encoding, the one binding of the circuit every transcript absorbs.
    digest: [u8; 32],
}

impl Circuit {
    /// Seal a gate list, hashing its encoding once.
    fn new(
        gates: Vec<Gate>,
        const_pos: usize,
        k_log: usize,
        useful_bits: usize,
        n_input_words: usize,
    ) -> Self {
        let mut circuit = Self {
            gates,
            const_pos,
            k_log,
            useful_bits,
            n_input_words,
            digest: [0; 32],
        };
        circuit.digest = circuit.compute_digest();
        circuit
    }

    /// Bits per instance, as a power of two.
    #[must_use]
    pub const fn k_log(&self) -> usize {
        self.k_log
    }

    /// Slots that carry data; the rest are zero.
    #[must_use]
    pub const fn useful_bits(&self) -> usize {
        self.useful_bits
    }

    /// Slot of the constant one.
    #[must_use]
    pub const fn const_pos(&self) -> usize {
        self.const_pos
    }

    /// Words the input ports span.
    #[must_use]
    pub const fn n_input_words(&self) -> usize {
        self.n_input_words
    }

    /// Keccak-256 of the circuit's injective encoding.
    #[must_use]
    pub const fn digest(&self) -> &[u8; 32] {
        &self.digest
    }

    /// One instance's output port words, read from its packed `z`.
    ///
    /// # Panics
    ///
    /// - The instance is narrower than its ports.
    #[must_use]
    pub fn output_words<'a>(&self, z: &'a [u64]) -> &'a [u64] {
        // Output ports sit between the inputs and the constant.
        &z[self.n_input_words..self.const_pos / 64]
    }

    /// Committed products, the R1CS rows that are not ports or the constant.
    #[must_use]
    pub fn num_ands(&self) -> usize {
        self.gates
            .iter()
            .filter(|g| matches!(g, Gate::And(..)))
            .count()
    }

    /// The gates in order.
    pub(crate) fn gates(&self) -> &[Gate] {
        &self.gates
    }

    /// Keccak-256 of the shape, then of the digests of fixed chunks of gates.
    ///
    /// The shape counts the gates and every gate has a fixed width, so the encoding is injective.
    ///
    /// The chunks hash in parallel, so binding a large circuit costs little.
    fn compute_digest(&self) -> [u8; 32] {
        let chunks: Vec<[u8; 32]> = self
            .gates
            .par_chunks(DIGEST_GATES)
            .map(|gates| {
                let mut bytes = Vec::with_capacity(13 * gates.len());
                encode_gates(gates, &mut bytes);
                Keccak256Hash.hash_slice(&bytes)
            })
            .collect();
        let mut shape = Vec::with_capacity(40);
        for v in [
            self.k_log,
            self.const_pos,
            self.useful_bits,
            self.n_input_words,
            self.gates.len(),
        ] {
            shape.extend_from_slice(&(v as u64).to_be_bytes());
        }
        Keccak256Hash.hash_iter_slices(
            core::iter::once(shape.as_slice()).chain(chunks.iter().map(|c| c.as_slice())),
        )
    }

    /// One instance's packed `z`, `A z` and `B z`, by one walk on bits.
    ///
    /// This is the reference every faster generator is tested against.
    ///
    /// # Panics
    ///
    /// - The input or a table has the wrong length.
    #[cfg(test)]
    pub(crate) fn witness_instance(
        &self,
        inputs: &[u64],
        z: &mut [u64],
        az: &mut [u64],
        bz: &mut [u64],
    ) {
        assert_eq!(inputs.len(), self.n_input_words, "input words");
        let words = (1 << self.k_log) / 64;
        for table in [&mut *z, &mut *az, &mut *bz] {
            assert_eq!(table.len(), words, "table words");
            table.fill(0);
        }
        let set =
            |buf: &mut [u64], s: u32, v: bool| buf[s as usize / 64] |= u64::from(v) << (s % 64);

        // Wire i is gate i's value.
        let mut wires: Vec<bool> = Vec::with_capacity(self.gates.len());
        for &gate in &self.gates {
            let v = match gate {
                Gate::Free(s) => {
                    let v = s as usize == self.const_pos
                        || (inputs[s as usize / 64] >> (s % 64)) & 1 == 1;
                    set(z, s, v);
                    set(az, s, v);
                    set(bz, s, true);
                    v
                }
                Gate::Xor(x, y) => wires[x as usize] ^ wires[y as usize],
                Gate::And(x, y, s) => {
                    let (x, y) = (wires[x as usize], wires[y as usize]);
                    set(z, s, x & y);
                    set(az, s, x);
                    set(bz, s, y);
                    x & y
                }
                Gate::Copy(x, s) => {
                    let v = wires[x as usize];
                    set(z, s, v);
                    set(az, s, v);
                    set(bz, s, true);
                    v
                }
            };
            wires.push(v);
        }
    }
}

/// Append gates at a fixed width: a kind byte, then three big-endian words.
fn encode_gates(gates: &[Gate], out: &mut Vec<u8>) {
    for &gate in gates {
        let (kind, words) = match gate {
            Gate::Free(s) => (0, [s, 0, 0]),
            Gate::Xor(x, y) => (1, [x, y, 0]),
            Gate::And(x, y, s) => (2, [x, y, s]),
            Gate::Copy(x, s) => (3, [x, s, 0]),
        };
        out.push(kind);
        for w in words {
            out.extend_from_slice(&w.to_be_bytes());
        }
    }
}
