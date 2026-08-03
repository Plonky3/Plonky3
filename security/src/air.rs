//! AIR composition error: ε_ALI = L⁺ · num_constraints / |F|.
//!
//! Regime-independent — the list size L⁺ is passed in by the caller,
//! computed from the chosen proximity regime via [`crate::proximity`].

use libm::log2;

use crate::error::ErrorBits;

/// `-log2(ε_ALI)` in bits. Returns 0 bits if inputs are degenerate.
pub fn composition_error(num_constraints: usize, list_size: f64, modulus_bits: usize) -> ErrorBits {
    if num_constraints == 0 || modulus_bits == 0 || !list_size.is_finite() || list_size <= 0.0 {
        return ErrorBits::from_log2(0.0);
    }
    let bits = modulus_bits as f64 - log2(list_size) - log2(num_constraints as f64);
    ErrorBits::from_log2(bits.max(0.0))
}
