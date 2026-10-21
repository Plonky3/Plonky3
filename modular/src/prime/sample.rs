//! Sampling a uniform prime of a given bit length from a caller-provided byte stream.

use alloc::vec;

use crate::error::PrimeError;
use crate::mont::MontModulus;
use crate::uint::Uint;

/// How a prime is sampled.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct PrimePolicy {
    /// The bit length of the prime.
    pub bits: u32,
    /// Miller-Rabin rounds with drawn bases after BPSW.
    pub mr_rounds: u32,
    /// The number of candidates tried before giving up.
    pub max_candidates: u32,
}

impl PrimePolicy {
    /// A policy for `bits`-bit primes, with 64 drawn rounds and up to `2^16` candidates.
    #[must_use]
    pub const fn new(bits: u32) -> Self {
        Self {
            bits,
            mr_rounds: 64,
            max_candidates: 1 << 16,
        }
    }
}

/// A sampled prime with its Montgomery constants, and the number of candidates it took.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct PrimeSample<const L: usize> {
    /// The ring modulo the prime.
    ring: MontModulus<L>,
    /// Candidates drawn, the accepted one included.
    candidates: u32,
}

impl<const L: usize> PrimeSample<L> {
    /// The prime.
    #[must_use]
    pub const fn prime(&self) -> Uint<L> {
        *self.ring.modulus_uint()
    }

    /// Candidates drawn, the accepted one included.
    #[must_use]
    pub const fn candidates(&self) -> u32 {
        self.candidates
    }

    /// The ring modulo the prime.
    pub(crate) const fn into_ring(self) -> MontModulus<L> {
        self.ring
    }
}

/// Draws candidates `2^(bits-1) | (x mod 2^(bits-1)) | 1` until one is prime, each `x` and each Miller-Rabin base read from `fill`.
///
/// Each candidate goes through trial division, Miller-Rabin to base 2, the strong Lucas test, then `mr_rounds` Miller-Rabin rounds on bases drawn in `[2, n - 2]`.
///
/// A composite is accepted with probability at most `4^-mr_rounds`.
pub fn sample_prime<const L: usize>(
    mut fill: impl FnMut(&mut [u8]),
    policy: &PrimePolicy,
) -> Result<PrimeSample<L>, PrimeError> {
    let bits = policy.bits;
    if bits < 2 || bits >= Uint::<L>::BITS || policy.max_candidates == 0 {
        return Err(PrimeError::InvalidPolicy);
    }
    let low_bits = bits - 1;
    let mask = Uint::<L>::MAX.shr(Uint::<L>::BITS - low_bits);
    let top = Uint::<L>::ONE.shl(low_bits);
    let mut buf = vec![0u8; low_bits.div_ceil(8) as usize];
    for candidates in 1..=policy.max_candidates {
        fill(&mut buf);
        let x = Uint::<L>::from_le_bytes(&buf).ok_or(PrimeError::InvalidPolicy)?;
        let mut n = Uint(core::array::from_fn(|i| (x.0[i] & mask.0[i]) | top.0[i]));
        n.0[0] |= 1;
        // The candidate is odd, at least 3 and below 2^(64 L - 1), so it always has a Montgomery form.
        let Ok(ring) = MontModulus::new(n) else {
            continue;
        };
        if super::is_prime_with(&ring, policy.mr_rounds, &mut fill)? {
            return Ok(PrimeSample { ring, candidates });
        }
    }
    Err(PrimeError::Exhausted)
}
