//! The Bars S-box layer trait for the Monolith permutation.
//!
//! The Bars layer is the only component of Monolith that differs between
//! prime field instantiations. It applies a non-linear S-box to the first
//! few state elements by decomposing each field element into small limbs,
//! applying a chi-like S-box per limb, and recomposing.
//!
//! The decomposition strategy (called Kintsugi in the paper) depends on the
//! binary structure of the prime:
//! - Mersenne31 (p = 2^31 - 1): 4 limbs of (8, 8, 8, 7) bits, lookup-table S-box
//! - Goldilocks (p = 2^64 - 2^32 + 1): 8 limbs of 8 bits, bitwise SWAR S-box

pub mod goldilocks;
pub mod mersenne31;

use sha3::Shake128Reader;

/// Trait capturing the field-specific Bars layer of a Monolith instance.
///
/// Implementors define how many state elements pass through the S-box,
/// how the S-box is computed, and the SHAKE128 domain-separation bytes
/// that encode the bucket decomposition.
pub trait MonolithBars<F, const WIDTH: usize>: Clone + Sync {
    /// Number of state elements that pass through the S-box per round.
    ///
    /// Chosen so that NUM_BARS * log2(p) ~ 256 bits of non-linearity.
    /// - Mersenne31: 8 (since 8 * 31 = 248 ~ 256)
    /// - Goldilocks: 4 (since 4 * 64 = 256)
    const NUM_BARS: usize;

    /// The field prime encoded as little-endian bytes for SHAKE128 domain separation.
    ///
    /// - Mersenne31: `&[0xFF, 0xFF, 0xFF, 0x7F]` (2^31 - 1 as 4 LE bytes)
    /// - Goldilocks: `&[0x01, 0x00, 0x00, 0x00, 0xFF, 0xFF, 0xFF, 0xFF]`
    const PRIME_BYTES: &[u8];

    /// Bucket bit-widths used in the SHAKE128 seed for domain separation.
    ///
    /// These bytes are appended to the SHAKE128 seed when generating round
    /// constants. They encode the limb decomposition of this instantiation.
    /// - Mersenne31: `&[8, 8, 8, 7]`
    /// - Goldilocks: `&[8, 8, 8, 8, 8, 8, 8, 8]`
    const LIMB_BITS: &[u8];

    /// Apply the Bars S-box layer to the first NUM_BARS elements of the state.
    ///
    /// Elements at indices NUM_BARS..WIDTH pass through unchanged.
    fn bars(&self, state: &mut [F; WIDTH]);

    /// Sample a uniformly random field element from a SHAKE128 stream.
    ///
    /// Uses rejection sampling to ensure uniform distribution over [0, p).
    fn random_field_element(shake: &mut Shake128Reader) -> F;
}
