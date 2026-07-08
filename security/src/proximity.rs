//! Proximity-gap and list-size helpers shared across low-degree-test
//! modules.
//!
//! - UDR (unique-decoding regime): agreement parameter α = (1 + ρ⁺)/2,
//!   list size L⁺ = 1.
//! - LDR (list-decoding regime, BCHKS25 explicit-m): α = (1 + 1/(2m))·√ρ,
//!   proximity parameter γ = 1 − α, list size L⁺ = (m + 1/2)/√ρ.
//!
//! References:
//! - [2020/654] Proximity Gaps for Reed–Solomon Codes
//! - [2024/1553] On the Security of STARKs with FRI
//! - [2025/2055] BCHKS25 Theorem 4.2

use libm::{ceil, pow, sqrt};

/// Performance cap on the proximity parameter `m` searched in LDR
/// analyses. Matches Ethereum's `soundcalc`.
pub const LDR_M_CAP: usize = 1000;

/// UDR agreement parameter α = (1 + ρ⁺)/2, where ρ⁺ accounts for the
/// trace-side expansion from out-of-domain openings.
pub fn alpha_udr(log_trace_length: usize, log_blowup: usize, max_combo: usize) -> f64 {
    let k = (1u64 << log_trace_length) as f64;
    let n = (1u64 << (log_trace_length + log_blowup)) as f64;
    let rho_plus = (k + max_combo as f64) / n;
    (1.0 + rho_plus) * 0.5
}

/// LDR agreement parameter α = (1 + 1/(2m))·√ρ. BCHKS25 §4.2.
pub fn alpha_ldr_m(log_blowup: usize, m: usize) -> f64 {
    let rho = pow(2.0, -(log_blowup as f64));
    (1.0 + 0.5 / m as f64) * sqrt(rho)
}

/// UDR proximity parameter γ = 1 − α used in the multi-point quotient
/// soundness precondition from [2020/654] §4.1.3.
pub fn gamma_udr(log_trace_length: usize, log_blowup: usize, max_combo: usize) -> f64 {
    1.0 - alpha_udr(log_trace_length, log_blowup, max_combo)
}

/// LDR proximity parameter γ = 1 − √ρ·(1 + 1/(2m)). BCHKS25 §4.2.
pub fn gamma_ldr_m(log_blowup: usize, m: usize) -> f64 {
    let rho = pow(2.0, -(log_blowup as f64));
    1.0 - sqrt(rho) * (1.0 + 0.5 / m as f64)
}

/// UDR list size: L⁺ = 1.
pub const fn list_size_udr() -> f64 {
    1.0
}

/// LDR list size: L⁺ = (m + 1/2)/√ρ. Matches `soundcalc`
/// `johnson_bound::get_max_list_size` (explicit-m branch).
pub fn list_size_ldr_m(log_blowup: usize, m: usize) -> f64 {
    let rho = pow(2.0, -(log_blowup as f64));
    (m as f64 + 0.5) / sqrt(rho)
}

/// Largest proximity parameter `m` such that the η > 0 precondition of
/// Theorem 1 in [2021/582] holds. Caller applies [`LDR_M_CAP`].
pub fn compute_upper_m(trace_length: usize) -> usize {
    if trace_length == 0 {
        return 0;
    }
    let h = trace_length as f64;
    let ratio = (h + 2.0) / h;
    ceil(1.0 / (2.0 * (sqrt(ratio) - 1.0))) as usize
}
