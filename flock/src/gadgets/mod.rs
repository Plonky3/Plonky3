//! Hash compression functions as Boolean circuits, with the native packing of their ports.
//!
//! Only ports, the constant and products are committed; every XOR, rotation and shift is free.

mod arx;
pub mod blake2s;
pub mod blake3;
pub mod keccak;
mod rows;
pub mod sha256;

#[cfg(test)]
mod tests;

use alloc::vec;
use alloc::vec::Vec;

use crate::Witness;
use crate::circuit::Circuit;

/// A function as a circuit, with the native packing of its ports.
pub trait Gadget {
    /// Input words the circuit reads.
    const INPUT_WORDS: usize;

    /// The decoded output.
    type Output;

    /// The circuit.
    fn circuit() -> Circuit;

    /// Pack the input into the circuit's input words.
    fn write_input(&self, words: &mut [u64]);

    /// The input as the circuit's input words.
    fn input_words(&self) -> Vec<u64> {
        let mut words = vec![0; Self::INPUT_WORDS];
        self.write_input(&mut words);
        words
    }

    /// Decode the output ports' words.
    fn decode_output(words: &[u64]) -> Self::Output;

    /// Pack an output into the output ports' words, the inverse of decoding.
    fn write_output(output: &Self::Output, words: &mut [u64]);

    /// The witness of `2^log_instances` instances of `circuit`, which must be this gadget's.
    ///
    /// The default walks the gates, and a gadget with a word-level generator overrides it.
    ///
    /// # Panics
    ///
    /// - There are more rows than instances.
    /// - A word-level generator gets another circuit.
    fn witness(circuit: &Circuit, log_instances: usize, rows: &[Self], padding: &Self) -> Witness
    where
        Self: Sized + Sync,
    {
        Witness::generate(circuit, log_instances, rows, padding, Self::write_input)
    }
}

/// Check that `circuit` has the shape a word-level generator writes.
///
/// # Panics
///
/// - The circuit is not the generator's.
fn assert_shape(circuit: &Circuit, k_log: usize, const_pos: usize, useful_bits: usize) {
    assert_eq!(
        (circuit.k_log(), circuit.const_pos(), circuit.useful_bits()),
        (k_log, const_pos, useful_bits),
        "the circuit is not this gadget's"
    );
}

/// Pack 32-bit words two per 64-bit word, low half first.
fn pack_u32s(words: &[u32], out: &mut [u64]) {
    for (o, pair) in out.iter_mut().zip(words.chunks(2)) {
        *o = pair
            .iter()
            .rev()
            .fold(0, |acc, &w| (acc << 32) | u64::from(w));
    }
}

/// Unpack 64-bit words into their 32-bit halves, low half first.
fn unpack_u32s<const N: usize>(words: &[u64]) -> [u32; N] {
    core::array::from_fn(|i| (words[i / 2] >> (32 * (i % 2))) as u32)
}
