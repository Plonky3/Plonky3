//! Hash compression functions as Boolean circuits, with the native packing of their ports.
//!
//! Only ports, the constant and products are committed; every XOR, rotation and shift is free.

mod arx;
pub mod blake2s;
pub mod blake3;
pub mod keccak;
pub mod sha256;

#[cfg(test)]
mod tests;

use alloc::vec;
use alloc::vec::Vec;

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
