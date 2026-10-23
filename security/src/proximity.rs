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
//! - [2024/1553] STARK-based Signatures from the RPO Permutation
//! - [2025/2055] BCHKS25
//! - \[DKT26\] Reed–Solomon Codes Beyond Johnson, Theorem 5.12 and
//!   Appendix B.1–B.2 (<https://eprint.iacr.org/2026/2056>)

use libm::{log2, pow, sqrt};

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
/// and still satisfies the DKT26 Appendix B.1 interpolant.
///
/// # Polynomial-curve extension derived here
///
/// The paper states Proposition B.1 for a line. To extend its dimension
/// count, put `D = k - 1`, `t = m + 1/2`, `s = t·sqrt(n/D)` and choose
/// `M = ceil(Ds)`, `B = ceil(s) - 1`, `H = ceil(ell·s²/3) - 1`.
/// Use monomials `X^i Y^j Z^h` with `i + D·j < Ds` and
/// `ell·j + h < ell·s²/3`. Since `s²/3 > s > B >= m`, no slice is truncated.
/// Translation by a degree-`ell` curve bounds the challenge degree of each
/// multiplicity constraint indexed by `(a,b)` by `H - ell·b`. Thus
///
/// ```text
/// N_var = sum_{j=0}^B (M - D·j)(H + 1 - ell·j),
/// N_eq  = n·sum_{b=0}^{m-1} (m - b)(H + 1 - ell·b).
/// K = sum_{j=0}^B (M - D·j),  W = sum_{j=0}^B j(M - D·j),
/// R = n·m(m+1)/2,            V = n·m(m-1)(m+1)/6,
/// N_var - N_eq = (H + 1)(K - R) - ell·(W - V).
/// ```
///
/// The ceilings cannot be pulled outside `ell`. To account for their
/// residual, let `u = ceil(s) - s` and first use `M = Ds`, giving `K0,W0`.
/// Direct expansion yields
///
/// ```text
/// K0 - R = D·(s + u(1-u))/2 + n/8 > 0,
/// ((s²/3)(K0-R) - W0 + V)/D
///   = s/6 + s⁴/(24t²) + u(1-u)(s²-3s-2u+1)/6 + V/D > 0.
/// ```
///
/// Every summand is nonnegative: `s >= 7/2` and `0 <= u < 1` give
/// `s²-3s-2u+1 >= 3/4`. Rounding `M` upward increases the second expression
/// because `s²/3 > j`; rounding `H+1` upward increases the surplus because
/// `K-R > 0`. Hence `N_var > N_eq`, even with both rounding residuals.
/// The specialization has degree `< Ds <= m·A`, so multiplicity `m` at
/// `A = ceil(alpha·n)` agreements forces the required identity.
///
/// Apply Lemma 5.3, as stated in §7.2 Equation (88)'s order-zero case.
/// This is characteristic-free, unlike that section's positive-order
/// corollary. Since `H/ell < s²/3`, Appendix B.2's termwise estimate has
/// the same slack after dividing by `ell`, giving `E < ell·C`. This does
/// not multiply an already-rounded finite line count.
///
/// Theorem 5.12 requires positive degree, inverse rate, and `m >= 3`. The
/// constant-code case instead uses the conservative `binom(n, 2)` count
/// from Lemma 3.2, independent of `m`. This can be substantially tighter;
/// it is a separate bound for future callers admitting dimension one,
/// not a continuation of the positive-degree formula. Inputs outside
/// those supported cases return `+∞` so callers fail closed.
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

/// List-decoding agreement parameter, alpha = (1 + 1/(2m)) sqrt(rho).
///
/// This is the agreement of eprint 2020/654, Theorem 8.3, and of ethSTARK, Theorem 5.
pub fn alpha_ldr_m(log_blowup: usize, m: usize) -> f64 {
    let rho = pow(2.0, -(log_blowup as f64));
    (1.0 + 0.5 / m as f64) * sqrt(rho)
}

/// UDR proximity parameter γ = 1 − α used in the multi-point quotient
/// soundness precondition from [2020/654] §4.1.3.
pub fn gamma_udr(log_trace_length: usize, log_blowup: usize, max_combo: usize) -> f64 {
    1.0 - alpha_udr(log_trace_length, log_blowup, max_combo)
}

/// List-decoding proximity parameter, 1 - alpha with alpha = (1 + 1/(2m)) sqrt(rho).
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

/// List size in the list-decoding regime, L^+ = (m + 1/2) / sqrt(rho).
///
/// This bounds the list.
///
/// The ALI and DEEP rounds decode the trace and quotient words jointly.
///
/// Each word is a quotient of degree below k, times a linear factor, plus a constant.
///
/// Correlated agreement makes the candidates of all opening points coincide.
///
/// So the candidates are interleaved polynomials of degree at most k, at agreement alpha.
///
/// DKT26 (eprint 2026/2056), Theorem 5.12, Equation (67), bounds that list by B_jet.
///
/// Corollary 7.5 there keeps the bound at every interleaving width.
///
/// With degree bound k the rate is k / n = rho, which gives:
///
/// ```text
///     B_jet = ceil((m' + 1/2) / sqrt(rho)) - 1
///     m'    = max(ceil(sqrt(rho) / (2 eta_0)), 3)
///     m'   <= m,  because eta_0 >= sqrt(rho) / (2m)
///     → B_jet <= (m + 1/2) / sqrt(rho)
/// ```
///
/// This matches soundcalc's list size at an explicit m.
///
/// The classical Johnson bound at a fixed margin is a different bound, used for WHIR and STIR.
pub fn list_size_ldr_m(log_blowup: usize, m: usize) -> f64 {
    let rho = pow(2.0, -(log_blowup as f64));
    (m as f64 + 0.5) / sqrt(rho)
}

/// Largest proximity parameter m that ethSTARK (eprint 2021/582), Theorem 1, admits.
///
/// The caller applies its own cap on top.
///
/// # Overview
///
/// Theorem 1 needs eta > 0, which is the strict inequality m < X with
///
/// ```text
///     X = 1 / (2 (sqrt((h + c) / h) - 1))
/// ```
///
/// - h is the trace length.
/// - c is the number of extra points that out-of-domain openings add to the trace.
///
/// Squaring both sides of m < X gives an equivalent integer test:
///
/// ```text
///     4 c m^2 < h (4m + 1)
/// ```
///
/// The search decides that test exactly, so no floating-point rounding enters the result.
pub fn compute_upper_m(trace_length: usize, max_combo: usize) -> usize {
    // An empty trace admits no parameter.
    if trace_length == 0 {
        return 0;
    }

    // With no extra points, X is infinite and every m is admissible.
    if max_combo == 0 {
        return usize::MAX;
    }

    // Widen to 128 bits for the products below.
    let (h, c) = (trace_length as u128, max_combo as u128);

    // Decide m < X through 4 c m^2 < h (4m + 1).
    //
    // A side that overflows 128 bits counts as inadmissible.
    // Overflow only grows with m, so the admissible values stay a prefix of the integers.
    // It can only lower the result, which keeps the precondition true.
    let admissible = |m: u128| {
        // Left side: 4 c m^2.
        let lhs = m.checked_mul(m).and_then(|v| v.checked_mul(4 * c));

        // Right side: h (4m + 1).
        let rhs = m
            .checked_mul(4)
            .and_then(|v| v.checked_add(1))
            .and_then(|v| v.checked_mul(h));

        // Admissible only when both sides fit and the inequality is strict.
        matches!((lhs, rhs), (Some(l), Some(r)) if l < r)
    };

    // Binary search between two bounds.
    //
    // Invariant: lo is admissible and hi is not.
    //
    //     lo = 0:               0 < h
    //     hi = 5h / (4c) + 2:   an admissible m >= 1 has 4 c m^2 < h (4m + 1) <= 5 h m,
    //                           so m < 5h / (4c)
    //
    // The interval halves each step, so the loop runs at most about 66 times.
    let mut lo = 0u128;
    let mut hi = 5 * h / (4 * c) + 2;
    while hi - lo > 1 {
        // Midpoint, without overflow.
        let mid = lo + (hi - lo) / 2;

        // Keep the invariant on whichever side the midpoint falls.
        if admissible(mid) {
            lo = mid;
        } else {
            hi = mid;
        }
    }

    // Saturate on targets where usize is narrower than the result.
    usize::try_from(lo).unwrap_or(usize::MAX)
}

#[cfg(test)]
mod tests {
    use proptest::prelude::*;

    use super::johnson_exceptional_line_count_log2;
    use crate::assumption::SecurityAssumption;

    struct CurveCertificate {
        challenge_degree: u128,
        variables: u128,
        equations: u128,
        exceptional_count: f64,
    }

    /// Reconstruct Proposition B.1's scaled support and Equation (88), using
    /// exact integer cutoffs rather than the implementation's closed constant.
    fn finite_curve_certificate(d: usize, r: usize, m: usize, ell: usize) -> CurveCertificate {
        let ceil_sqrt_ratio = |num: u128, den: u128| {
            let root = (num / den).isqrt();
            root + u128::from(root * root * den < num)
        };
        let k = 1u128 << d;
        let n = k << r;
        let degree = k - 1;
        let m = m as u128;
        let ell = ell as u128;
        let four_t_squared = (2 * m + 1).pow(2);
        let spectral_cutoff = ceil_sqrt_ratio(four_t_squared * degree * n, 4);
        let jet_degree = ceil_sqrt_ratio(four_t_squared * n, 4 * degree) - 1;
        // ceil(x/y) - 1 = (x - 1)/y for positive integers x,y.
        let challenge_degree = (ell * four_t_squared * n - 1) / (12 * degree);
        let agreements = ceil_sqrt_ratio(four_t_squared * k * n, 4 * m * m);
        assert!(agreements > k && agreements <= n);
        assert!(jet_degree >= m && challenge_degree >= ell * jet_degree);

        let variables = (0..=jet_degree)
            .map(|j| (spectral_cutoff - degree * j) * (challenge_degree + 1 - ell * j))
            .sum();
        let equations = n
            * (0..m)
                .map(|b| (m - b) * (challenge_degree + 1 - ell * b))
                .sum::<u128>();
        let psi = 1
            + (2 * degree - 1) * (2 * jet_degree - 1)
            + 2 * jet_degree.saturating_sub(2 * degree + 1);
        let denominator = agreements - degree;
        let numerator = ((2 * jet_degree - 1) * challenge_degree
            + ell * (n - degree - 1) * jet_degree)
            * denominator
            + (n - degree) * (ell * jet_degree + challenge_degree * psi);
        CurveCertificate {
            challenge_degree,
            variables,
            equations,
            exceptional_count: numerator as f64 / denominator as f64,
        }
    }

    #[test]
    fn curve_interpolation_keeps_the_rounding_residuals() {
        // The ell=3 support has H=97, exceeding 3 times the line's H=32.
        // Exact-cutoff cases also catch replacing ceil(x)-1 by floor(x).
        for (d, r, m, ell, h, variables, equations) in [
            (1, 1, 3, 3, 48, 1204, 1128),
            (1, 2, 3, 1, 32, 1650, 1552),
            (1, 2, 3, 3, 97, 4895, 4608),
            (2, 2, 4, 1, 35, 6127, 5600),
        ] {
            let certificate = finite_curve_certificate(d, r, m, ell);
            assert_eq!(certificate.challenge_degree, h);
            assert_eq!(certificate.variables, variables);
            assert_eq!(certificate.equations, equations);
        }
        let line = finite_curve_certificate(1, 2, 3, 1);
        let curve = finite_curve_certificate(1, 2, 3, 3);
        assert_eq!(line.exceptional_count, 2293.75);
        assert_eq!(curve.exceptional_count, 6950.75);
        assert!(curve.exceptional_count > 3.0 * line.exceptional_count);
    }

    #[test]
    fn johnson_curve_bound_covers_finite_interpolation_certificates() {
        for d in [1, 2, 3, 5, 8, 12, 20] {
            for r in [1, 2, 3, 4, 8] {
                for m in [3, 4, 10, 31, 100, 1000] {
                    for ell in [1, 2, 3, 7, 31, 257, 1_572_866] {
                        let certificate = finite_curve_certificate(d, r, m, ell);
                        assert!(certificate.variables > certificate.equations);
                        let closed_log2 = 256.0
                            - SecurityAssumption::prox_gaps_error_jb_at_m(d, r, 256, ell + 1, m);
                        assert!(
                            libm::log2(certificate.exceptional_count) < closed_log2,
                            "d={d}, r={r}, m={m}, ell={ell}"
                        );
                    }
                }
            }
        }
    }

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
        for (r, pairs) in [(1, 1.0), (2, 6.0), (3, 28.0)] {
            for m in [3, 10, 1000] {
                let constant = johnson_exceptional_line_count_log2(0, r, m);
                assert!((constant - libm::log2(pairs)).abs() < 1e-12);
            }
        }
        assert!(johnson_exceptional_line_count_log2(1, 0, 3).is_infinite());
        assert!(johnson_exceptional_line_count_log2(1, 1, 2).is_infinite());
        assert!(johnson_exceptional_line_count_log2(usize::MAX, usize::MAX, 3).is_finite());
    }

    #[test]
    fn list_size_covers_the_degree_k_list_bound() {
        // Invariant: the list size is at least B_jet of DKT26, Equation (67), for degree <= k.
        //
        //     A     = ceil(alpha n)            agreements, alpha = (1 + 1/(2m)) sqrt(rho)
        //     eta_0 = A / n - sqrt(rho)
        //     m'    = max(ceil(sqrt(rho) / (2 eta_0)), 3)
        //     B_jet = ceil((m' + 1/2) / sqrt(rho)) - 1
        //
        // Fixture state: every trace 2^1 .. 2^24, blowup 2 .. 2^8, and m = 3 .. 1000.
        //
        // Why: a list size below B_jet would undercount the ALI and DEEP candidates.
        for log_trace_length in 1..=24u32 {
            for log_blowup in 1..=8u32 {
                // Domain size n and sqrt(rho) for this trace and blowup.
                let n = (1u64 << (log_trace_length + log_blowup)) as f64;
                let sqrt_rho = libm::sqrt(libm::pow(2.0, -f64::from(log_blowup)));
                for m in (3..=1000usize).step_by(7) {
                    // Recompute B_jet from the agreement count, as in the formula above.
                    let alpha = super::alpha_ldr_m(log_blowup as usize, m);
                    let agreements = libm::ceil(alpha * n);
                    let eta0 = agreements / n - sqrt_rho;
                    let m_prime = libm::ceil(sqrt_rho / (2.0 * eta0)).max(3.0);
                    let b_jet = libm::ceil((m_prime + 0.5) / sqrt_rho) - 1.0;

                    // The crate's list size must cover it.
                    let list = super::list_size_ldr_m(log_blowup as usize, m);
                    assert!(
                        b_jet <= list + 1e-9,
                        "trace 2^{log_trace_length}, blowup 2^{log_blowup}, m = {m}: B_jet = {b_jet}, list = {list}"
                    );
                }
            }
        }
    }

    #[test]
    fn upper_m_is_strict_on_integer_boundaries() {
        // Invariant: the result m satisfies m < X strictly.
        //
        // Fixture state: h = 144, c = 25.
        //
        //     sqrt((144 + 25) / 144) = 13 / 12
        //     X = 1 / (2 (13 / 12 - 1)) = 6
        //     → m = 6 breaks m < X, so the answer is 5
        assert_eq!(super::compute_upper_m(144, 25), 5);

        // Family: h = 4 x^2 and c = 4x + 1 give X = x exactly.
        //
        //     4 c x^2 = 4 (4x + 1) x^2 = h (4x + 1)
        //     → equality at m = x, so the answer is x - 1
        for x in 1usize..2000 {
            assert_eq!(
                super::compute_upper_m(4 * x * x, 4 * x + 1),
                x - 1,
                "x = {x}"
            );
        }
    }

    #[test]
    fn upper_m_search_is_bounded_on_extreme_inputs() {
        // Fixture state: c = 1, where the bound has a closed form.
        //
        //     4 m^2 < h (4m + 1)  holds at m = h:      4 h^2 < 4 h^2 + h
        //                         fails at m = h + 1:  4 h^2 + 8h + 4 >= 4 h^2 + 5h
        //     → the answer is h itself
        //
        // At h = 2^52 the floating-point X rounds to infinity.
        for log_h in [1, 20, 40, 52, 60] {
            let h = 1usize << log_h;
            assert_eq!(super::compute_upper_m(h, 1), h, "h = 2^{log_h}");
        }

        // Invariant: the largest inputs still terminate.
        //
        // Their products overflow 128 bits, so the search returns a smaller, still admissible m.
        let m = super::compute_upper_m(usize::MAX, 1) as u128;
        assert!(m > 0);

        // The returned m is admissible: a right side past 128 bits exceeds any left side that fits.
        let rhs = (usize::MAX as u128).checked_mul(4 * m + 1);
        assert!(rhs.is_none_or(|r| 4 * m * m < r));
    }

    proptest! {
        #[test]
        fn upper_m_is_the_largest_admissible(h in 1usize..1 << 40, c in 1usize..1 << 20) {
            // Invariant: the result m satisfies 4 c m^2 < h (4m + 1), and m + 1 does not.
            //
            // At these sizes every product fits in 128 bits.
            let m = super::compute_upper_m(h, c) as u128;
            let (h, c) = (h as u128, c as u128);
            let admissible = |m: u128| 4 * c * m * m < h * (4 * m + 1);
            prop_assert!(admissible(m));
            prop_assert!(!admissible(m + 1));
        }
    }
}
