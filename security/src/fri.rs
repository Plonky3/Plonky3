//! FRI low-degree-test soundness.
//!
//! Conjectured regime: random-words bound, [2025/2010] §1.5.
//! Proven regime: round-by-round, [2024/1553] Theorems 2 & 3, with the
//! BCHKS25 LDR commit bound ([2025/2055] Theorem 4.2). Cross-checked
//! against Ethereum's `soundcalc`.
//!
//! Correspondence with [`crate::assumption::SecurityAssumption`]:
//! - [`proven_error_udr`] is the FRI counterpart of `UniqueDecoding`
//!   (with FRI-specific α = (1 + ρ⁺)/2 incorporating OOD expansion).
//! - [`proven_error_ldr_m`] / [`best_ldr_m`] is the FRI counterpart of
//!   `JohnsonBound`, but searches `m ∈ [3, LDR_M_CAP]` rather than
//!   fixing `m = 10` as WHIR does.
//! - `CapacityBound` is not currently supported by FRI's commit-phase
//!   analysis.

use libm::{log2, pow};
use p3_util::log2_floor_usize;

use crate::error::ErrorBits;
use crate::proximity::{LDR_M_CAP, alpha_ldr_m, alpha_udr, compute_upper_m, gamma_ldr_m};
use crate::shape::{InstanceShape, StarkAirParams};

/// Security-relevant mirror of `p3_fri::FriParameters`.
///
/// Keep in sync with `FriParameters` whenever a security-affecting field
/// is added there. There is intentionally no `From<FriParameters>` impl:
/// the protocol crate assembles this from its own params + instance
/// shape, baking in protocol-specific assumptions at the call site.
#[derive(Copy, Clone, Debug)]
pub struct FriRegime {
    pub log_blowup: usize,
    pub num_queries: usize,
    pub log_final_poly_len: usize,
    pub max_log_arity: usize,
    pub commit_pow_bits: usize,
    pub query_pow_bits: usize,
}

impl FriRegime {
    const fn folding_factor(self) -> f64 {
        (1usize << self.max_log_arity) as f64
    }
}

/// Conjectured low-degree-test soundness (random-words, [2025/2010] §1.5).
///
/// `b = num_queries · (−log2(ρ + η)) + query_pow`,
/// with `η ≈ (log2(e/ρ) · ρ) / log2(q)`.
pub fn conjectured_error(regime: &FriRegime, shape: &InstanceShape) -> ErrorBits {
    if regime.log_blowup == 0 || shape.modulus_bits == 0 {
        return ErrorBits::from_log2(regime.query_pow_bits as f64);
    }
    let log_blowup_f = regime.log_blowup as f64;
    let rho = pow(2.0, -log_blowup_f);
    let log2_e_over_rho = core::f64::consts::LOG2_E + log_blowup_f;
    let eta = (log2_e_over_rho * rho) / shape.modulus_bits as f64;
    let effective = rho + eta;
    if effective <= 0.0 || effective >= 1.0 {
        return ErrorBits::from_log2(regime.query_pow_bits as f64);
    }
    let bits_per_query = -log2(effective);
    let bits = regime.num_queries as f64 * bits_per_query + regime.query_pow_bits as f64;
    ErrorBits::from_log2(bits)
}

/// FRI commit-phase per-round error in UDR.
///
/// ε ≤ (folding − 1)·(n + 1) / |F|, applied when at least one fold occurs.
/// Slightly conservative versus `soundcalc`'s tighter `(γn + 1)` factor.
pub fn commit_phase_error_udr(regime: &FriRegime, shape: &InstanceShape) -> Option<ErrorBits> {
    let lde_log = shape.log_trace_length + regime.log_blowup;
    let num_layers = log2_floor_usize(1usize << lde_log).saturating_sub(regime.log_final_poly_len);
    if num_layers == 0 {
        return None;
    }
    let folding_minus_one = (regime.folding_factor() - 1.0).max(1.0);
    let n = (1u64 << lde_log) as f64;
    let bits = shape.modulus_bits as f64 - log2(folding_minus_one * (n + 1.0))
        + regime.commit_pow_bits as f64;
    Some(ErrorBits::from_log2(bits.max(0.0)))
}

/// FRI commit-phase per-round error in LDR with explicit proximity
/// parameter `m`. BCHKS25 Theorem 4.2 (`error_powers`):
///
/// ε_lin   = ((2·m'⁵ + 3·m'·γρ)·n / (3·ρ^{3/2}) + m'/√ρ) / |F|,
/// ε_round = ε_lin · (folding − 1).
///
/// We also evaluate the n/q-style bound from [2024/1553] and report the
/// tighter of the two. Round-by-round soundness is dominated by round 0
/// (largest `n`), so we use `n = lde_domain_size` for every round.
pub fn commit_phase_error_ldr_m(regime: &FriRegime, shape: &InstanceShape, m: usize) -> ErrorBits {
    let rho = pow(2.0, -(regime.log_blowup as f64));
    let sqrt_rho = libm::sqrt(rho);
    let m_shifted = m as f64 + 0.5;
    let pp = gamma_ldr_m(regime.log_blowup, m);
    if pp <= 0.0 {
        return ErrorBits::from_log2(0.0);
    }
    let lde_log = shape.log_trace_length + regime.log_blowup;
    let n = (1u64 << lde_log) as f64;
    let folding_minus_one = (regime.folding_factor() - 1.0).max(1.0);

    let num = (2.0 * pow(m_shifted, 5.0) + 3.0 * m_shifted * pp * rho) * n;
    let den = 3.0 * rho * sqrt_rho;
    let eps_linear = num / den + m_shifted / sqrt_rho;
    let eps_powers = eps_linear * folding_minus_one;
    let bits_linear =
        shape.modulus_bits as f64 - log2(eps_powers.max(1.0)) + regime.commit_pow_bits as f64;

    let bits_n_over_q = shape.modulus_bits as f64
        - log2(regime.folding_factor())
        - log2(n + 1.0)
        - log2(2.0 * m as f64 + 1.0)
        + 0.5 * log2(rho)
        + regime.commit_pow_bits as f64;

    ErrorBits::from_log2(bits_linear.min(bits_n_over_q).max(0.0))
}

/// FRI query-phase error: ε ≤ αᵏ, contributing `query_pow − k·log2(α)` bits.
pub fn query_phase_error(alpha: f64, num_queries: usize, query_pow_bits: usize) -> ErrorBits {
    if !alpha.is_finite() || alpha <= 0.0 || alpha >= 1.0 {
        return ErrorBits::from_log2(0.0);
    }
    let bits = query_pow_bits as f64 - log2(pow(alpha, num_queries as f64));
    ErrorBits::from_log2(bits)
}

/// Proven LDT-only error in the UDR regime. Combines commit-phase and
/// query-phase contributions; AIR/DEEP terms compose at the protocol
/// call site (e.g. `crate::stark::proven_security`).
pub fn proven_error_udr(
    regime: &FriRegime,
    air: &StarkAirParams,
    shape: &InstanceShape,
) -> ErrorBits {
    if regime.log_blowup == 0 || shape.log_trace_length == 0 || shape.modulus_bits == 0 {
        return ErrorBits::from_log2(0.0);
    }
    let alpha = alpha_udr(shape.log_trace_length, regime.log_blowup, air.max_combo);
    let lde = (1u64 << (shape.log_trace_length + regime.log_blowup)) as f64;
    let k = (1u64 << shape.log_trace_length) as f64;
    if k + air.max_combo as f64 >= alpha * lde {
        return ErrorBits::from_log2(0.0);
    }
    let query = query_phase_error(alpha, regime.num_queries, regime.query_pow_bits);
    commit_phase_error_udr(regime, shape).map_or(query, |commit| ErrorBits::min(&[commit, query]))
}

/// Proven LDT-only error in the LDR regime with explicit `m`.
pub fn proven_error_ldr_m(
    regime: &FriRegime,
    air: &StarkAirParams,
    shape: &InstanceShape,
    m: usize,
) -> ErrorBits {
    if regime.log_blowup == 0 || shape.log_trace_length == 0 || shape.modulus_bits == 0 {
        return ErrorBits::from_log2(0.0);
    }
    let alpha = alpha_ldr_m(regime.log_blowup, m);
    if alpha >= 1.0 {
        return ErrorBits::from_log2(0.0);
    }
    let pp = gamma_ldr_m(regime.log_blowup, m);
    if pp <= 0.0 {
        return ErrorBits::from_log2(0.0);
    }
    let lde = (1u64 << (shape.log_trace_length + regime.log_blowup)) as f64;
    let k = (1u64 << shape.log_trace_length) as f64;
    if k + air.max_combo as f64 >= (1.0 - pp) * lde {
        return ErrorBits::from_log2(0.0);
    }
    let commit = commit_phase_error_ldr_m(regime, shape, m);
    let query = query_phase_error(alpha, regime.num_queries, regime.query_pow_bits);
    ErrorBits::min(&[commit, query])
}

/// Search `m ∈ [3, min(compute_upper_m, LDR_M_CAP)]` for the value
/// maximising LDR security; returns `(best_m, ldt_error_at_best_m)`.
///
/// Optimizes `min(commit, query)` only — the LDT-only error — not the full
/// `min(ALI, DEEP, commit, query)` composite ultimately reported by
/// [`crate::stark::proven_security`]. In practice ALI/DEEP don't bind at
/// the optimum, so the chosen `m` matches what optimizing the full
/// composite would pick, but this function does not verify that.
pub fn best_ldr_m(
    regime: &FriRegime,
    air: &StarkAirParams,
    shape: &InstanceShape,
) -> Option<(usize, ErrorBits)> {
    let trace_length = 1usize << shape.log_trace_length;
    let m_max = core::cmp::min(compute_upper_m(trace_length), LDR_M_CAP);
    let m_min = 3usize;
    if m_max < m_min {
        return None;
    }
    (m_min..=m_max)
        .map(|m| (m, proven_error_ldr_m(regime, air, shape, m)))
        .max_by(|a, b| {
            a.1.bits()
                .partial_cmp(&b.1.bits())
                .unwrap_or(core::cmp::Ordering::Equal)
        })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::stark::proven_security;

    fn benchmark_regime() -> FriRegime {
        FriRegime {
            log_blowup: 1,
            num_queries: 100,
            log_final_poly_len: 0,
            max_log_arity: 3,
            commit_pow_bits: 0,
            query_pow_bits: 16,
        }
    }

    fn benchmark_shape() -> InstanceShape {
        InstanceShape {
            log_trace_length: 20,
            modulus_bits: 252,
            collision_resistance: 128,
        }
    }

    fn benchmark_air() -> StarkAirParams {
        StarkAirParams {
            num_constraints: 1,
            max_constraint_degree: 2,
            max_combo: 2,
        }
    }

    /// Regression vector for the benchmark configuration: log_blowup=1,
    /// num_queries=100, query_pow=16, commit_pow=0, max_log_arity=3,
    /// |F|=252 bits, trace 2^20, num_constraints=1, max_deg=2,
    /// max_combo=2 → UDR=57 bits, LDR=65 bits.
    #[test]
    fn proven_security_regression_benchmark_high_arity() {
        let regime = benchmark_regime();
        let air = benchmark_air();
        let shape = benchmark_shape();

        let udr_ldt = proven_error_udr(&regime, &air, &shape);
        let (best_m, ldr_ldt) = best_ldr_m(&regime, &air, &shape).unwrap();

        let udr_bits = crate::stark::proven_security_udr(&air, &shape, udr_ldt, &[])
            .bits()
            .floor() as usize;
        let ldr_bits = crate::stark::proven_security_ldr_m(
            &air,
            &shape,
            regime.log_blowup,
            best_m,
            ldr_ldt,
            &[],
        )
        .bits()
        .floor() as usize;

        assert_eq!(udr_bits, 57);
        assert_eq!(ldr_bits, 65);

        let combined = proven_security(
            &air,
            &shape,
            regime.log_blowup,
            udr_ldt,
            best_m,
            ldr_ldt,
            &[],
        )
        .bits()
        .floor() as usize;
        assert_eq!(combined, 65);
    }

    #[test]
    fn conjectured_bounded_by_collision_resistance() {
        let regime = FriRegime {
            log_blowup: 8,
            num_queries: 32,
            log_final_poly_len: 0,
            max_log_arity: 1,
            commit_pow_bits: 0,
            query_pow_bits: 0,
        };
        let shape = InstanceShape {
            log_trace_length: 16,
            modulus_bits: 128,
            collision_resistance: 128,
        };
        let bits = conjectured_error(&regime, &shape)
            .bits()
            .min(shape.collision_resistance as f64)
            .min(shape.modulus_bits as f64)
            .floor() as usize;
        assert_eq!(bits, 128);
    }
}
