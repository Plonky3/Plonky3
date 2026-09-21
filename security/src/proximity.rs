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
//! - \[DKT26\] Reed–Solomon Codes Beyond Johnson, Theorem 5.12 and
//!   Appendix B.1–B.2 (<https://eprint.iacr.org/2026/2056>)

use libm::{floor, log2, pow, sqrt};

/// Performance cap on the proximity parameter `m` searched in LDR
/// analyses. Matches Ethereum's `soundcalc`.
pub const LDR_M_CAP: usize = 1000;

/// `log₂(C)` for the Johnson-regime exceptional line count from DKT26
/// Theorem 5.12 and Appendix B.1–B.2:
///
/// `C = 8 · n · (m + 1/2)^3 / (3 · ρ⁻)`, where
/// `n = 2^(log_degree + log_inv_rate)` and
/// `ρ⁻ = (2^log_degree - 1) / n`.
///
/// The bound is for one received line. A polynomial curve of degree `ell`
/// contributes `ell · C`; callers apply that curve-degree factor. The
/// positive-degree case uses the actual finite Johnson rate `ρ⁻`, rather than
/// the nominal rate `2^log_degree / n`. The logarithmic form avoids integer
/// shifts and sums overflowing for large domains.
///
/// The chosen agreement parameter
/// `alpha = (1 + 1/(2m))·sqrt(k/n)` gives `eta0 >= sqrt(rho_minus)/(2m)`;
/// therefore an explicit `m >= 3` can be larger than the theorem's minimum
/// and still satisfies the DKT26 Appendix B.1 interpolant. For a degree-`ell`
/// curve, write `D = k - 1` and `t = m + 1/2`. Its weighted monomials use
/// `i + D·j < t·sqrt(Dn)` and
/// `ell·j + h < ell·t²/(3·rho_minus)`, hence `B_Z < ell·t²/(3·rho_minus)`.
/// Section 7.2 Equation (88), with Appendix B.2's closed estimate, is linear
/// in `ell` and yields the displayed `ell·C` factor without multiplying a
/// rounded finite line count.
///
/// Theorem 5.12 requires positive degree, inverse rate, and `m >= 3`. The
/// constant-code case uses the conservative `binom(n, 2)` line count from
/// Lemma 3.2. Inputs outside those supported cases return `+∞` so callers
/// fail closed.
pub(crate) fn johnson_exceptional_line_count_log2(
    log_degree: usize,
    log_inv_rate: usize,
    m: usize,
) -> f64 {
    if m < 3 || log_inv_rate == 0 {
        return f64::INFINITY;
    }

    if log_degree == 0 {
        // log₂(n(n − 1)/2), written without constructing n.
        let log_n_minus_one = log_inv_rate as f64 + log2(1.0 - pow(2.0, -(log_inv_rate as f64)));
        return log_inv_rate as f64 + log_n_minus_one - 1.0;
    }

    let log_n = log_degree as f64 + log_inv_rate as f64;
    let log_rho_minus = log2(1.0 - pow(2.0, -(log_degree as f64))) - log_inv_rate as f64;
    3.0 + log_n + 3.0 * log2(m as f64 + 0.5) - log2(3.0) - log_rho_minus
}

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

/// Conjectured-regime list size: L⁺ = 1.
///
/// The random-words heuristic of [2025/2010] §1.5 treats a malicious prover's
/// committed word as distributed like a uniformly random word, whose distance
/// to the code concentrates at capacity `1 − ρ`. Conjecturing correlated
/// agreement up to that radius, the relevant decoding list is a single
/// codeword — numerically the same L⁺ = 1 as [`list_size_udr`], but justified
/// by the conjecture rather than by the unique-decoding radius. The ALI and
/// DEEP-ALI bounds therefore carry no list-size multiplier in this regime.
pub const fn list_size_conjectured() -> f64 {
    1.0
}

/// LDR list size: L⁺ = (m + 1/2)/√ρ. Matches `soundcalc`
/// `johnson_bound::get_max_list_size` (explicit-m branch).
///
/// Uses the tighter [2025/2055] BCHKS25 Theorem 4.2 list size at the caller's
/// explicit `m`, distinct from [`crate::assumption::SecurityAssumption::JohnsonBound`]'s
/// classical Johnson bound at a fixed safety margin — the two are not
/// interchangeable, see that variant's `list_size_bits_at_log_eta` doc.
pub fn list_size_ldr_m(log_blowup: usize, m: usize) -> f64 {
    let rho = pow(2.0, -(log_blowup as f64));
    (m as f64 + 0.5) / sqrt(rho)
}

/// Largest proximity parameter `m` such that the η > 0 precondition of
/// Theorem 1 in [2021/582] holds. Caller applies [`LDR_M_CAP`].
///
/// `max_combo` accounts for the trace-side expansion from out-of-domain
/// openings, matching [`alpha_udr`]'s `rho_plus`. The precondition is the
/// strict inequality `m < X` where `X = 1 / (2 * (sqrt((h + max_combo) / h) - 1))`,
/// so the admissible maximum is `floor(X)`, or `X - 1` on the non-generic
/// chance that `X` is itself an integer.
pub fn compute_upper_m(trace_length: usize, max_combo: usize) -> usize {
    if trace_length == 0 {
        return 0;
    }
    let h = trace_length as f64;
    let ratio = (h + max_combo as f64) / h;
    let x = 1.0 / (2.0 * (sqrt(ratio) - 1.0));
    let m = floor(x);
    if m == x {
        (m - 1.0) as usize
    } else {
        m as usize
    }
}

#[cfg(test)]
mod tests {
    use super::johnson_exceptional_line_count_log2;

    #[test]
    fn johnson_line_count_matches_dkt26_vectors() {
        for (log_degree, log_inv_rate, m, expected) in [
            (1, 1, 3, 10.837102265451657_f64),
            (2, 1, 3, 11.2521397647305),
            (20, 2, 3, 30.83710364131352),
            (20, 2, 10, 35.59199114347699),
        ] {
            let actual = johnson_exceptional_line_count_log2(log_degree, log_inv_rate, m);
            assert!(
                (actual - expected).abs() < 1e-12,
                "got {actual}, expected {expected}"
            );
        }
    }

    #[test]
    fn johnson_line_count_handles_constant_and_invalid_inputs() {
        let constant = johnson_exceptional_line_count_log2(0, 3, 3);
        let expected = libm::log2((8.0 * 7.0) / 2.0);
        assert!((constant - expected).abs() < 1e-12);
        assert!(johnson_exceptional_line_count_log2(1, 0, 3).is_infinite());
        assert!(johnson_exceptional_line_count_log2(1, 1, 2).is_infinite());
        assert!(johnson_exceptional_line_count_log2(usize::MAX, usize::MAX, 3).is_finite());
    }
}
