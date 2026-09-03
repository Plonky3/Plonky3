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

use alloc::vec;
use alloc::vec::Vec;

use libm::{log2, pow};

use crate::error::ErrorBits;
use crate::ldt::LowDegreeTest;
use crate::proximity::{LDR_M_CAP, alpha_ldr_m, alpha_udr, compute_upper_m, gamma_ldr_m};
use crate::report::{LDT_COMMIT_LABEL, LDT_QUERY_LABEL, SecurityTerm};
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

impl LowDegreeTest for FriRegime {
    fn log_blowup(&self) -> usize {
        self.log_blowup
    }

    fn proven_error_udr(&self, air: &StarkAirParams, shape: &InstanceShape) -> ErrorBits {
        proven_error_udr(self, air, shape)
    }

    fn best_ldr(&self, air: &StarkAirParams, shape: &InstanceShape) -> Option<(usize, ErrorBits)> {
        best_ldr_m(self, air, shape)
    }

    fn conjectured_error(&self, shape: &InstanceShape) -> ErrorBits {
        conjectured_error(self, shape)
    }

    fn conjectured_terms(&self, shape: &InstanceShape) -> Vec<SecurityTerm> {
        let mut terms = vec![SecurityTerm::new(
            LDT_QUERY_LABEL,
            conjectured_error(self, shape),
        )];
        terms.extend(
            conjectured_commit_phase_error(self, shape)
                .map(|bits| SecurityTerm::new(LDT_COMMIT_LABEL, bits)),
        );
        terms
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

/// Legacy conjectured low-degree-test soundness (ethSTARK
/// [2021/582](https://eprint.iacr.org/2021/582), pre-random-words).
///
/// `b = num_queries · log_blowup + query_pow`. Predates the random-words
/// correction in [`conjectured_error`] ([2025/2010] §1.5) and does not
/// account for the commit-phase folding round covered by
/// [`conjectured_commit_phase_error`]; kept for callers that specifically
/// want the older, simpler heuristic bound.
pub const fn legacy_conjectured_error(regime: &FriRegime) -> ErrorBits {
    ErrorBits::from_log2((regime.log_blowup * regime.num_queries + regime.query_pow_bits) as f64)
}

/// FRI commit-phase per-round error in the conjectured regime.
///
/// Identical to [`commit_phase_error_udr`], and deliberately so: the bound
/// counts the folding challenges that are *exceptional* for a given committed
/// word, and that count does not depend on the decoding regime. What the
/// random-words conjecture buys is the removal of the list-size multiplier
/// `L⁺` that the Johnson-bound analysis pays (see
/// [`crate::stark::conjectured_security_report`]) — it does not assert that a
/// bad folding challenge cannot exist. Dropping the round entirely, as an
/// LDT-only conjectured number does, therefore overstates security for a large
/// LDE domain over a small field.
///
/// Returns `None` when no fold occurs, in which case there is no such round.
pub fn conjectured_commit_phase_error(
    regime: &FriRegime,
    shape: &InstanceShape,
) -> Option<ErrorBits> {
    commit_phase_error_udr(regime, shape)
}

/// FRI commit-phase per-round error in UDR.
///
/// ε ≤ (folding − 1)·(n + 1) / |F|, applied when at least one fold occurs.
/// Slightly conservative versus `soundcalc`'s tighter `(γn + 1)` factor.
///
/// Returns `None` when `regime.max_log_arity` is `0` (folding factor `1`,
/// i.e. no fold at all) — such a regime has no commit-phase round, rather
/// than being indistinguishable from arity 2.
pub fn commit_phase_error_udr(regime: &FriRegime, shape: &InstanceShape) -> Option<ErrorBits> {
    let folding_minus_one = regime.folding_factor() - 1.0;
    if folding_minus_one <= 0.0 {
        return None;
    }
    let lde_log = shape.log_trace_length + regime.log_blowup;
    let num_layers = lde_log.saturating_sub(regime.log_final_poly_len) / regime.max_log_arity;
    if num_layers == 0 {
        return None;
    }
    let n = (1u64 << lde_log) as f64;
    let bits = shape.modulus_bits as f64 - log2(folding_minus_one * (n + 1.0))
        + regime.commit_pow_bits as f64;
    Some(ErrorBits::from_log2(bits.max(0.0)))
}

/// FRI commit-phase per-round error in LDR with explicit proximity
/// parameter `m`. BCHKS25 Theorem 1.5 (Equation (1)):
///
/// ε_lin   = ((2·m'⁵ + 3·m'·γρ)·n / (3·ρ^{3/2}) + m'/√ρ) / |F|,
/// ε_round = ε_lin · (folding − 1).
///
/// We also evaluate the n/q-style bound from [2024/1553] and report the
/// tighter of the two. Round-by-round soundness is dominated by round 0
/// (largest `n`), so we use `n = lde_domain_size` for every round.
///
/// Returns `None` when `regime.max_log_arity` is `0` (folding factor `1`,
/// i.e. no fold at all) — such a regime has no commit-phase round, rather
/// than being indistinguishable from arity 2.
pub fn commit_phase_error_ldr_m(
    regime: &FriRegime,
    shape: &InstanceShape,
    m: usize,
) -> Option<ErrorBits> {
    let rho = pow(2.0, -(regime.log_blowup as f64));
    let sqrt_rho = libm::sqrt(rho);
    let m_shifted = m as f64 + 0.5;
    let pp = gamma_ldr_m(regime.log_blowup, m);
    if pp <= 0.0 {
        return Some(ErrorBits::from_log2(0.0));
    }
    let folding_minus_one = regime.folding_factor() - 1.0;
    if folding_minus_one <= 0.0 {
        return None;
    }
    let lde_log = shape.log_trace_length + regime.log_blowup;
    let n = (1u64 << lde_log) as f64;

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

    Some(ErrorBits::from_log2(
        bits_linear.min(bits_n_over_q).max(0.0),
    ))
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
    let query = query_phase_error(alpha, regime.num_queries, regime.query_pow_bits);
    commit_phase_error_ldr_m(regime, shape, m)
        .map_or(query, |commit| ErrorBits::min(&[commit, query]))
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
    let m_max = core::cmp::min(compute_upper_m(trace_length, air.max_combo), LDR_M_CAP);
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
            num_batched_functions: 1,
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

    /// The conjectured commit-phase round is a real constraint, not a
    /// formality: over a 128-bit field with a 2^26 LDE domain it lands near 100
    /// bits, below a 128-bit target, so an LDT-only conjectured number that
    /// omits it overstates security. It also matches the UDR bound exactly —
    /// the conjecture removes the list-size multiplier, not the count of
    /// exceptional folding challenges.
    #[test]
    fn conjectured_commit_phase_binds_below_a_128_bit_target() {
        let regime = FriRegime {
            log_blowup: 3,
            num_queries: 27,
            log_final_poly_len: 0,
            max_log_arity: 2,
            commit_pow_bits: 0,
            query_pow_bits: 16,
        };
        let shape = InstanceShape {
            log_trace_length: 23,
            modulus_bits: 128,
            collision_resistance: 128,
            num_batched_functions: 1,
        };

        let commit = conjectured_commit_phase_error(&regime, &shape).expect("folds occur");
        // |F| - log2(3 * (2^26 + 1)) = 128 - 27.585
        assert!(
            (commit.bits() - (128.0 - libm::log2(3.0 * (65_536.0 * 1024.0 + 1.0)))).abs() < 1e-9
        );
        assert!(commit.bits() < 128.0, "got {}", commit.bits());
        assert_eq!(
            commit.bits(),
            commit_phase_error_udr(&regime, &shape).unwrap().bits()
        );
    }

    /// Commit-phase grinding is credited to the commit-phase term and to
    /// nothing else, and a fold-free configuration has no such round at all.
    #[test]
    fn conjectured_commit_phase_credits_only_its_own_grinding() {
        let base = benchmark_regime();
        let shape = benchmark_shape();

        let ground = FriRegime {
            commit_pow_bits: 12,
            ..base
        };
        let b0 = conjectured_commit_phase_error(&base, &shape).expect("folds occur");
        let b12 = conjectured_commit_phase_error(&ground, &shape).expect("folds occur");
        assert!((b12.bits() - b0.bits() - 12.0).abs() < 1e-12);
        // The query phase is untouched by it.
        assert_eq!(
            conjectured_error(&ground, &shape).bits(),
            conjectured_error(&base, &shape).bits()
        );

        // Folding down to the full domain leaves no commit round.
        let no_folds = FriRegime {
            log_final_poly_len: shape.log_trace_length + base.log_blowup,
            ..base
        };
        assert!(conjectured_commit_phase_error(&no_folds, &shape).is_none());
        assert_eq!(
            LowDegreeTest::conjectured_terms(&no_folds, &shape).len(),
            1,
            "a fold-free regime reports the query phase only"
        );
    }

    /// `max_log_arity: 0` (folding factor 1) is a genuinely fold-free regime,
    /// not a stand-in for arity 2: both the UDR and LDR commit-phase bounds
    /// must report no round rather than silently clamping to arity 2's
    /// `folding_minus_one = 1`.
    #[test]
    fn arity_one_reports_no_commit_round_rather_than_impersonating_arity_two() {
        let base = benchmark_regime();
        let shape = benchmark_shape();
        let air = benchmark_air();
        let no_arity = FriRegime {
            max_log_arity: 0,
            ..base
        };

        assert!(commit_phase_error_udr(&no_arity, &shape).is_none());
        assert!(commit_phase_error_ldr_m(&no_arity, &shape, 10).is_none());
        assert!(conjectured_commit_phase_error(&no_arity, &shape).is_none());

        // The UDR/LDR composites fall back to the query-phase error alone.
        assert_eq!(
            proven_error_udr(&no_arity, &air, &shape).bits(),
            query_phase_error(
                alpha_udr(shape.log_trace_length, no_arity.log_blowup, air.max_combo),
                no_arity.num_queries,
                no_arity.query_pow_bits,
            )
            .bits()
        );
    }

    /// `legacy_conjectured_error` reproduces the pre-random-words ethSTARK
    /// formula exactly: `num_queries * log_blowup + query_pow`.
    #[test]
    fn legacy_conjectured_error_matches_ethstark_formula() {
        let regime = benchmark_regime();
        let bits = legacy_conjectured_error(&regime).bits();
        assert_eq!(
            bits,
            (regime.num_queries * regime.log_blowup + regime.query_pow_bits) as f64
        );
        assert_eq!(bits, 116.0);
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
            num_batched_functions: 1,
        };
        let bits = conjectured_error(&regime, &shape)
            .bits()
            .min(shape.collision_resistance as f64)
            .min(shape.modulus_bits as f64)
            .floor() as usize;
        assert_eq!(bits, 128);
    }
}
