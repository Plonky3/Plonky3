//! DEEP-ALI out-of-domain sampling error.
//!
//! ε_DEEP = L⁺ · factor / |F|, with k = trace domain size and
//!
//! factor = max(max_deg · (k + max_combo − 1) + (k − 1), (c + 1) · k + max_combo − 1).
//!
//! The first term is ethSTARK's `X^i · h_i(X^d)` quotient split, matching
//! `soundcalc/circuits/deep_ali.py`. The second is Plonky3's own quotient split into
//! `c` committed power-of-two chunks, which the first term only
//! dominates when `c ≤ max_deg`. `soundcalc` divides by `|F| − k − D`; the difference is
//! negligible.
//!
//! The caller supplies the exact chunk count. For degree `d` and `z = is_zk`, this is
//! `c = 2^z · next_power_of_two(max(d + z, 2) − 1)`, including both ZK adjustments.
//!
//! The list size enters linearly.
//!
//! The trace and quotient words are decoded jointly, as one interleaved word.
//!
//! So the list size enters once.
//!
//! Haböck's FRI summary (eprint 2022/1216) proves this in Theorem 8 and Remark 9.
//!
//! soundcalc uses the same charge.
//!
//! Eprint 2024/1553, Theorem 2, charges its square, a looser bound.
//!
//! The difference matters, because this term can set the reported level.
//!
//! Example: a degree-3 AIR over a 124-bit field, in the list-decoding regime.
//!
//! ```text
//!     trace 2^24, blowup 8, 70 queries, arity 2, m = 3
//!     grinding: 20 bits commit phase, 16 bits query phase, 8 bits batch combination
//!     2 batched functions
//!
//!     air composition      117.37 bits
//!     batch combination     95.16 bits
//!     low-degree test      105.43 bits
//!     DEEP, linear          94.69 bits   <- weakest, sets the level
//!     DEEP, squared         91.39 bits
//! ```
//!
//! Without batch grinding the batch term drops to 87.16 bits and sets the level instead.

use libm::log2;

use crate::error::ErrorBits;
use crate::shape::{InstanceShape, StarkAirParams};

/// `-log2(ε_DEEP)` in bits. Returns 0 bits if inputs are degenerate or the trace size
/// cannot be represented by `u64`.
pub fn deep_ali_error(air: &StarkAirParams, shape: &InstanceShape, list_size: f64) -> ErrorBits {
    if shape.modulus_bits == 0
        || !air.num_quotient_chunks.is_power_of_two()
        || shape.log_trace_length >= u64::BITS as usize
        || !list_size.is_finite()
        || list_size <= 0.0
    {
        return ErrorBits::from_log2(0.0);
    }
    let k = (1u64 << shape.log_trace_length) as f64;
    let max_deg = air.max_constraint_degree.max(1) as f64;
    let combo = air.max_combo as f64;
    let ethstark = max_deg * (k + combo - 1.0) + (k - 1.0);
    let chunks = air.num_quotient_chunks as f64;
    let chunked = (chunks + 1.0) * k + combo - 1.0;
    let factor = ethstark.max(chunked).max(1.0);
    let bits = shape.modulus_bits as f64 - log2(list_size) - log2(factor);
    ErrorBits::from_log2(bits.max(0.0))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn trace_length_shift_boundary_is_conservative() {
        let air = StarkAirParams {
            num_constraints: 1,
            max_constraint_degree: 2,
            num_quotient_chunks: 1,
            max_combo: 2,
        };
        let mut shape = InstanceShape {
            log_trace_length: 63,
            modulus_bits: 128,
            collision_resistance: 128,
            num_batched_functions: 1,
        };
        // 2^63 fits in u64; the DEEP factor is 3 * 2^63 + 1.
        let expected = 128.0 - 63.0 - log2(3.0);
        assert!((deep_ali_error(&air, &shape, 1.0).bits() - expected).abs() < 1e-12);

        // These exponents must not panic or wrap back to a small trace.
        for log_trace_length in [64, 65, usize::MAX] {
            shape.log_trace_length = log_trace_length;
            assert_eq!(deep_ali_error(&air, &shape, 1.0).bits(), 0.0);
        }
    }
}
