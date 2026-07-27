//! Grinding (proof-of-work) bits — additive contribution to security.

/// Bits added to the soundness budget by a `pow_bits`-bit grinding round.
/// Equal to `pow_bits` when grinding is honest; provided as a function
/// so future tweaks (multi-round PoW, variable difficulty) stay local.
pub const fn grinding_bits(pow_bits: usize) -> f64 {
    pow_bits as f64
}
