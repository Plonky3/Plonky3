//! DEEP-ALI out-of-domain sampling error.
//!
//! ε_DEEP = L⁺ · factor / |F|, with k = trace domain size and
//!
//! factor = max(max_deg · (k + max_combo − 1) + (k − 1), (c + 1) · k + max_combo − 1).
//!
//! The first term is ethSTARK's `X^i · h_i(X^d)` quotient split, matching
//! `soundcalc/circuits/deep_ali.py`. The second is Plonky3's own quotient split into
//! `c = 2^⌈log2(max(max_deg, 2) − 1)⌉` power-of-two chunks
//! (`uni_stark::symbolic::get_log_num_quotient_chunks`), which the first term only
//! dominates when `c ≤ max_deg`. `soundcalc` divides by `|F| − k − D`; the difference is
//! negligible.
//!
//! `c` assumes a non-`zk` quotient split; a `zk` instance further randomizes each chunk's
//! domain and is not modeled here.
//!
//! The list size `L⁺` enters linearly here, following `soundcalc`.
//! [2024/1553] Theorem 2 (`eps_2`) instead uses `L² ≈ (m/ρ)²`, which is
//! a few bits more conservative in the list-decoding regime; ALI/DEEP
//! don't bind at the LDR optimum in practice (see `fri::best_ldr_m`), so
//! the difference does not currently affect reported bounds.

use libm::log2;

use crate::error::ErrorBits;
use crate::shape::{InstanceShape, StarkAirParams};

/// `-log2(ε_DEEP)` in bits. Returns 0 bits if inputs are degenerate.
pub fn deep_ali_error(air: &StarkAirParams, shape: &InstanceShape, list_size: f64) -> ErrorBits {
    if shape.modulus_bits == 0 || !list_size.is_finite() || list_size <= 0.0 {
        return ErrorBits::from_log2(0.0);
    }
    let k = (1u64 << shape.log_trace_length) as f64;
    let max_deg = air.max_constraint_degree.max(1) as f64;
    let combo = air.max_combo as f64;
    let ethstark = max_deg * (k + combo - 1.0) + (k - 1.0);
    let chunks = (air.max_constraint_degree.max(2) - 1).next_power_of_two() as f64;
    let chunked = (chunks + 1.0) * k + combo - 1.0;
    let factor = ethstark.max(chunked).max(1.0);
    let bits = shape.modulus_bits as f64 - log2(list_size) - log2(factor);
    ErrorBits::from_log2(bits.max(0.0))
}
