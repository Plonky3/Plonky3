//! Why a modulus or a prime sample is refused.

use thiserror::Error;

/// Why a modulus is refused.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Error)]
#[non_exhaustive]
pub enum ModulusError {
    /// The modulus is even, so it has no Montgomery form.
    #[error("the modulus is even")]
    Even,
    /// The modulus is too small for the ring.
    #[error("the modulus is too small")]
    TooSmall,
    /// The modulus does not leave the spare top bit free, or exceeds the ring's width.
    #[error("the modulus is too wide")]
    TooWide,
    /// The modulus of a field is composite.
    #[error("the modulus is composite")]
    Composite,
}

/// Why no prime was sampled.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Error)]
#[non_exhaustive]
pub enum PrimeError {
    /// The policy asks for a bit length outside `[2, 64 L - 1]` or for no candidates.
    #[error("the prime policy is invalid")]
    InvalidPolicy,
    /// Every allowed candidate or base draw was used up.
    #[error("the prime search ran out of candidates")]
    Exhausted,
}
