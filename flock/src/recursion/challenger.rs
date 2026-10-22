//! [`Blake3Mirror`]: the Flock transcript over BLAKE3, mirrored byte for byte onto wires.

use alloc::sync::Arc;
use alloc::vec::Vec;

use p3_blake3::Blake3;
use p3_challenger::{
    CanObserve, CanSample, CanSampleBits, CanSampleUniformBits, FieldChallenger,
    GrindingChallenger, ResamplingError,
};
use p3_symmetric::MerkleCap;

use super::WireArith;
use super::blake3::hash;
use super::gates::{Byte, Gates, Sample};
use super::value::Shared;
use crate::circuit::Wire;
use crate::{FlockChallenger, Val};

/// A [`FlockChallenger`] over BLAKE3 whose every absorb and squeeze is mirrored onto circuit wires.
///
/// The native challenger still runs, so the native transcript sees exactly the bytes it expects.
///
/// The mirror keeps the hash challenger's input and output buffers as bytes of wires, and hashes the input on wires at every flush.
pub struct Blake3Mirror {
    /// The native challenger.
    native: FlockChallenger<Blake3>,
    /// The circuit under construction.
    gates: Shared,
    /// Absorbed bytes since the last flush, the last digest first.
    input: Vec<Byte>,
    /// Digest bytes not yet sampled, popped from the back.
    output: Vec<Byte>,
}

impl Blake3Mirror {
    /// A fresh transcript seeded with the constant `initial`, mirrored into `arith`.
    #[must_use]
    pub fn new(arith: &WireArith, initial: Vec<u8>) -> Self {
        let gates = Arc::clone(&arith.gates);
        let input = {
            let g = gates.lock();
            initial.iter().map(|&b| constant_byte(&g, b)).collect()
        };
        Self {
            native: FlockChallenger::from_hasher(initial, Blake3),
            gates,
            input,
            output: Vec::new(),
        }
    }

    /// The native challenger, in the state the mirror is in.
    #[must_use]
    pub const fn native(&self) -> &FlockChallenger<Blake3> {
        &self.native
    }

    /// Absorb native bytes: the announced ones, or constants while a transcript seeds itself.
    fn absorb(&mut self, bytes: &[u8]) {
        let mut gates = self.gates.lock();
        self.output.clear();
        for &value in bytes {
            if gates.seeding {
                self.input.push(constant_byte(&gates, value));
                // A seed that names a digest held on wires takes the wires.
                if let Some(start) = self.input.len().checked_sub(32) {
                    let tail: Vec<u8> = self.input[start..].iter().map(|b| b.value).collect();
                    if let Some((_, wires)) =
                        gates.substitutions.iter().find(|(d, _)| d[..] == tail[..])
                    {
                        self.input[start..].copy_from_slice(wires);
                    }
                }
                continue;
            }
            let byte = {
                let byte = gates.pending.pop_front().expect(
                    "every absorbed proof value is announced before the transcript absorbs it",
                );
                assert_eq!(
                    byte.value, value,
                    "the announced value is the one the transcript absorbs"
                );
                byte
            };
            self.input.push(byte);
        }
    }

    /// Squeeze `n` bytes in the order the native challenger reads them.
    fn squeeze(&mut self, n: usize) -> Vec<Byte> {
        let mut gates = self.gates.lock();
        (0..n)
            .map(|_| {
                if self.output.is_empty() {
                    let previous = gates.component("transcript");
                    let digest = hash(&mut gates.builder, &self.input);
                    gates.component(previous);
                    self.input = digest.to_vec();
                    self.output = digest.to_vec();
                }
                self.output.pop().expect("a flush fills the output")
            })
            .collect()
    }

    /// Eight squeezed bytes as 64 wires, low bit first, and their little-endian value.
    fn squeeze_word(&mut self) -> (Vec<Wire>, u64) {
        let bytes = self.squeeze(8);
        let wires = bytes.iter().flat_map(|b| b.wires).collect();
        let value = bytes
            .iter()
            .rev()
            .fold(0u64, |acc, b| (acc << 8) | u64::from(b.value));
        (wires, value)
    }
}

/// A constant byte's wires.
fn constant_byte(gates: &Gates, value: u8) -> Byte {
    Byte {
        wires: core::array::from_fn(|i| gates.constant((value >> i) & 1 == 1)),
        value,
    }
}

impl CanObserve<Val> for Blake3Mirror {
    fn observe(&mut self, value: Val) {
        self.native.observe(value);
        self.absorb(&value.to_bits().to_le_bytes());
    }
}

impl CanObserve<MerkleCap<Val, [u8; 32]>> for Blake3Mirror {
    fn observe(&mut self, cap: MerkleCap<Val, [u8; 32]>) {
        let bytes: Vec<u8> = cap.roots().iter().flatten().copied().collect();
        self.native.observe(cap);
        self.absorb(&bytes);
    }
}

impl CanSample<Val> for Blake3Mirror {
    fn sample(&mut self) -> Val {
        let value: Val = self.native.sample();
        let (wires, bits) = self.squeeze_word();
        assert_eq!(
            bits,
            value.to_bits(),
            "the mirror squeezes what the transcript squeezes"
        );
        self.gates
            .lock()
            .sampled
            .push_back(Sample::Value(wires, value));
        value
    }
}

impl CanSampleBits<usize> for Blake3Mirror {
    fn sample_bits(&mut self, bits: usize) -> usize {
        let value = self.native.sample_bits(bits);
        let (wires, word) = self.squeeze_word();
        let index: Vec<(Wire, bool)> = wires
            .into_iter()
            .take(bits)
            .enumerate()
            .map(|(i, w)| (w, (word >> i) & 1 == 1))
            .collect();
        self.gates.lock().sampled.push_back(Sample::Index(index));
        value
    }
}

impl CanSampleUniformBits<Val> for Blake3Mirror {
    /// Byte sampling is uniform without rejection, as natively.
    fn sample_uniform_bits<const RESAMPLE: bool>(
        &mut self,
        bits: usize,
    ) -> Result<usize, ResamplingError> {
        Ok(self.sample_bits(bits))
    }
}

impl GrindingChallenger for Blake3Mirror {
    type Witness = Val;

    /// # Panics
    ///
    /// Always: a verifier checks witnesses and never grinds for one.
    fn grind(&mut self, _bits: usize) -> Val {
        unreachable!("a verifier checks witnesses and never grinds for one")
    }

    fn check_witness(&mut self, bits: usize, witness: Val) -> bool {
        if bits == 0 {
            return true;
        }
        let passed = self.native.check_witness(bits, witness);
        self.absorb(&witness.to_bits().to_le_bytes());
        // The witness passes when the low bits of the next eight bytes are zero.
        let (wires, word) = self.squeeze_word();
        let mut gates = self.gates.lock();
        for &wire in wires.iter().take(bits) {
            gates.assert_zero(wire);
        }
        debug_assert_eq!(
            passed,
            word & ((1 << bits) - 1) == 0,
            "the mirror grinds as natively"
        );
        passed
    }
}

impl FieldChallenger<Val> for Blake3Mirror {}
