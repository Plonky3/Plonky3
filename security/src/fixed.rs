//! Deterministic integer analog of this crate's floating-point error-bit arithmetic.
//!
//! Every quantity elsewhere in this crate is `-log2(error)` as an `f64` via [`crate::error::ErrorBits`].
//! A protocol that grades proof soundness *inside a recursive verifier* — a circuit or a VM with no
//! floating-point unit — cannot reproduce that arithmetic bit-for-bit, and needs every rounding
//! step biased toward understating security rather than overstating it. This module provides that:
//! `-log2(error)` as a 16-fractional-bit fixed-point integer, computed with `const fn` integer
//! operations only, rounding down on added logarithms and up on subtracted ones so the result is
//! always a conservative lower bound on the true value.
//!
//! Not wired into the rest of the crate's `f64` composition machinery — a consumer needing
//! bit-exact recursive verification uses this module's primitives directly rather than
//! [`crate::error::ErrorBits`].

/// Number of fractional bits in the fixed-point representation.
pub const FRACTIONAL_BITS: u32 = 16;

/// The value `1.0` in fixed point.
pub const ONE: u64 = 1 << FRACTIONAL_BITS;

/// `log2(e)` in fixed point, rounded up.
///
/// Appears in the random-words cutoff of [2025/2010](https://eprint.iacr.org/2025/2010) section
/// 1.5, in [`bits_per_query`]'s `numerator`, whose `ceil_log2` is then subtracted from the
/// uncorrected rate. Rounding this constant down would shrink that correction and overstate the
/// reported per-query rate, contradicting this module's conservative-lower-bound invariant.
pub const LOG2_E: u64 = 94_549;

/// Converts a whole number of bits into fixed point.
pub const fn from_bits(bits: u32) -> u64 {
    (bits as u64) << FRACTIONAL_BITS
}

/// Truncates a fixed-point quantity to whole bits, rounding down.
pub const fn to_bits(value: u64) -> u32 {
    (value >> FRACTIONAL_BITS) as u32
}

/// Returns `log2(x)` in fixed point, rounded down.
///
/// # Panics
/// Panics if `x` is zero, which has no logarithm.
pub const fn floor_log2(x: u64) -> u64 {
    assert!(x != 0, "log2 is undefined at zero");

    let integer_part = 63 - x.leading_zeros() as u64;
    let mut result = integer_part << FRACTIONAL_BITS;

    // Normalize the mantissa to Q32, so it lies in `[1, 2)` and its square stays inside a `u128`.
    let mut mantissa: u128 = if integer_part >= 32 {
        (x as u128) >> (integer_part - 32)
    } else {
        (x as u128) << (32 - integer_part)
    };

    // Standard bit-by-bit binary logarithm: squaring the mantissa shifts the next fractional bit
    // of the result into the integer position.
    let mut bit = 0;
    while bit < FRACTIONAL_BITS {
        mantissa = (mantissa * mantissa) >> 32;
        if mantissa >= 1 << 33 {
            mantissa >>= 1;
            result |= 1 << (FRACTIONAL_BITS - 1 - bit);
        }
        bit += 1;
    }

    result
}

/// Returns `log2(x)` in fixed point, rounded up.
///
/// Exact for powers of two; otherwise one unit in the last place above [`floor_log2`], which is at
/// most `2^-16` of a bit coarser than the true value.
///
/// # Panics
/// Panics if `x` is zero, which has no logarithm.
pub const fn ceil_log2(x: u64) -> u64 {
    if x.is_power_of_two() {
        from_bits(x.trailing_zeros())
    } else {
        floor_log2(x) + 1
    }
}

/// Returns the conjectured security contributed by a single FRI query, in fixed point.
///
/// This is the "random words" rate of [2025/2010](https://eprint.iacr.org/2025/2010) section 1.5:
///
/// ```text
/// bits_per_query = -log2(ρ + η),   ρ = 2^-log_blowup,   η = log2(e/ρ) · ρ / log2(q)
/// ```
///
/// Rewriting to keep every step in fixed point and avoid materializing `ρ`:
///
/// ```text
/// -log2(ρ + η) = log_blowup - log2(1 + (log2(e) + log_blowup) / log2(q))
/// ```
///
/// so the correction is a ratio of two fixed-point integers whose logarithm [`ceil_log2`] can take
/// directly. The correction is rounded up and then subtracted, keeping the rate conservative.
///
/// `field_bits` is `log2(q)` for the field FRI operates over — the challenge field, in fixed point.
///
/// Returns zero when the correction would exceed `log_blowup`, i.e. when the rate is not positive
/// and queries contribute nothing.
pub const fn bits_per_query(log_blowup: u32, field_bits: u64) -> u64 {
    if log_blowup == 0 || field_bits == 0 {
        return 0;
    }

    let uncorrected = from_bits(log_blowup);
    let numerator = field_bits + LOG2_E + uncorrected;
    let correction = ceil_log2(numerator) - floor_log2(field_bits);

    uncorrected.saturating_sub(correction)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fri::{FriRegime, conjectured_error};
    use crate::shape::InstanceShape;

    /// The true value of `log2` must never be understated by [`ceil_log2`] nor overstated by
    /// [`floor_log2`], and both must track it to within one unit in the last place. These are the
    /// directions a round budget built on this module relies on to stay conservative.
    #[test]
    fn log2_brackets_the_true_logarithm() {
        let mut x = 1u64;
        while x < u64::MAX / 3 {
            for candidate in [x, x + 1, x * 3 / 2] {
                let truth = (candidate as f64).log2() * ONE as f64;
                let floor = floor_log2(candidate) as f64;
                let ceil = ceil_log2(candidate) as f64;

                assert!(
                    floor <= truth + 1e-3,
                    "floor_log2({candidate}) overstates log2"
                );
                assert!(
                    truth - floor < 2.0,
                    "floor_log2({candidate}) is more than 1 ulp low"
                );
                assert!(
                    ceil + 1e-3 >= truth,
                    "ceil_log2({candidate}) understates log2"
                );
                assert!(
                    ceil - truth < 2.0,
                    "ceil_log2({candidate}) is more than 1 ulp high"
                );
            }
            x *= 2;
        }
    }

    /// `LOG2_E` enters `bits_per_query`'s subtracted correction, so rounding it down would
    /// shrink the correction and overstate the reported per-query rate.
    #[test]
    fn log2_e_rounds_up_not_down() {
        let truth = core::f64::consts::LOG2_E * ONE as f64;
        assert!(
            LOG2_E as f64 >= truth,
            "LOG2_E must round up, not down: {LOG2_E} < {truth}"
        );
    }

    #[test]
    fn log2_is_exact_on_powers_of_two() {
        for exponent in 0..64u32 {
            assert_eq!(floor_log2(1 << exponent), from_bits(exponent));
            assert_eq!(ceil_log2(1 << exponent), from_bits(exponent));
        }
    }

    /// The fixed-point per-query rate must never exceed the real-valued random-words formula, and
    /// must track it within one ulp, across every blowup the PCS parameters admit.
    #[test]
    fn bits_per_query_never_overstates_the_random_words_rate() {
        for log_blowup in 1..=8u32 {
            for field_bits in [from_bits(64), from_bits(96), from_bits(128), from_bits(192)] {
                let rho = 2f64.powi(-(log_blowup as i32));
                let eta = (core::f64::consts::LOG2_E + f64::from(log_blowup)) * rho
                    / (field_bits as f64 / ONE as f64);
                let truth = -(rho + eta).log2() * ONE as f64;

                let actual = bits_per_query(log_blowup, field_bits) as f64;
                assert!(
                    actual <= truth,
                    "bits_per_query({log_blowup}, {field_bits}) overstates the rate"
                );
                assert!(
                    truth - actual < 2.0,
                    "bits_per_query({log_blowup}, {field_bits}) is more than 1 ulp low"
                );
            }
        }
    }

    #[test]
    fn bits_per_query_is_zero_without_a_blowup() {
        assert_eq!(bits_per_query(0, from_bits(128)), 0);
    }

    /// [`bits_per_query`] and this crate's own [`crate::fri::conjectured_error`] compute the same
    /// per-query rate through two independent code paths (fixed-point vs. `f64` + `libm`); at a
    /// single query with no grinding they must agree to within the fixed-point representation's own
    /// resolution.
    #[test]
    fn bits_per_query_matches_fri_conjectured_error_per_query_rate() {
        for log_blowup in 1..=8usize {
            let regime = FriRegime {
                log_blowup,
                num_queries: 1,
                log_final_poly_len: 0,
                max_log_arity: 0,
                commit_pow_bits: 0,
                query_pow_bits: 0,
            };
            let shape = InstanceShape {
                log_trace_length: 20,
                modulus_bits: 128,
                collision_resistance: 128,
                num_batched_functions: 1,
            };

            let f64_bits = conjectured_error(&regime, &shape).bits();
            let fixed_bits = bits_per_query(log_blowup as u32, from_bits(128)) as f64 / ONE as f64;

            assert!(
                fixed_bits <= f64_bits + 1e-9,
                "log_blowup={log_blowup}: fixed overstates f64"
            );
            assert!(
                f64_bits - fixed_bits < 2.0 / ONE as f64 + 1e-9,
                "log_blowup={log_blowup}: fixed is more than 1 ulp below f64"
            );
        }
    }
}
