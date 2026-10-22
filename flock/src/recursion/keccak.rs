//! Keccak-256 of a byte string of fixed length, on wires: the sponge of FIPS 202, Section 4, with the original `0x01` padding.

use alloc::vec::Vec;

use p3_keccak::Keccak256Hash;
use p3_symmetric::CryptographicHasher;

use super::gates::Byte;
use crate::circuit::{Builder, Wire, Word};
use crate::gadgets::keccak::permute;

/// Bytes absorbed per permutation.
const RATE: usize = 136;

/// The Keccak-256 digest of `input`, its wires and native value.
pub(crate) fn keccak256(c: &mut Builder, input: &[Byte]) -> [Byte; 32] {
    let native: Vec<u8> = input.iter().map(|b| b.value).collect();
    let digest = Keccak256Hash.hash_slice(&native);
    let one = c.one();
    // Pad: a one bit after the message, a one bit at the end of the last block.
    let mut padded: Vec<[Wire; 8]> = input.iter().map(|b| b.wires).collect();
    let pad_at = padded.len();
    padded.resize((pad_at + 1).div_ceil(RATE) * RATE, [Wire::ZERO; 8]);
    padded[pad_at][0] = one;
    let last = padded.len() - 1;
    padded[last][7] = one;
    let mut state: [Word; 25] = core::array::from_fn(|_| (0..64).map(|_| Wire::ZERO).collect());
    for block in padded.as_chunks::<RATE>().0 {
        for (lane, bytes) in state.iter_mut().zip(block.as_chunks::<8>().0) {
            let wires: Vec<Wire> = bytes.iter().flatten().copied().collect();
            *lane = c.xor_word(lane, &wires);
        }
        state = permute(c, state);
    }
    core::array::from_fn(|i| Byte {
        wires: core::array::from_fn(|b| state[i / 8][8 * (i % 8) + b]),
        value: digest[i],
    })
}
