//! STARK proof security level computation.
//!
//! Provides conjectured and proven security level estimates in bits.
//!
//! # References
//! - ethSTARK ([2021/582](https://eprint.iacr.org/2021/582))
//! - Proximity Gaps for Reed-Solomon Codes ([2020/654](https://eprint.iacr.org/2020/654))
//! - On the Security of STARKs with FRI ([2024/1553](https://eprint.iacr.org/2024/1553))
//! - On the Distribution of the Distances of Random Words ([2025/2010](https://eprint.iacr.org/2025/2010))
//! - BCHKS25 — Improved LDR proximity gaps ([2025/2055](https://eprint.iacr.org/2025/2055))
//!
//! [2025/2010] recommends proven bounds in deployment. If a deployer prefers
//! conjectured bounds, it advises staying above the random-words cutoff
//! ([2025/2010] §1.5, formula (5)).
//!
//! The proven analysis is patterned on Theorems 2 & 3 of [2024/1553] (round-by-round
//! soundness; UDR and LDR), with the improved LDR FRI commit-phase bound from
//! [2025/2055] Theorem 4.2. Concrete formulas are cross-checked against Ethereum's
//! reference [`soundcalc`](https://github.com/ethereum/soundcalc) calculator.

#![allow(clippy::too_many_arguments)]

use alloc::vec::Vec;
use core::cmp::{max, min};

use libm::{ceil, log2, pow, sqrt};
use p3_air::Air;
use p3_air::symbolic::{AirLayout, SymbolicAirBuilder, get_all_symbolic_constraints};
use p3_field::{ExtensionField, Field};
use p3_fri::FriParameters;
use p3_util::log2_floor_usize;

/// Parameters required to compute STARK proof security level.
///
/// FRI-related fields are read from [`FriParameters`]; the AIR-shape fields
/// (`num_constraints`, `air_max_constraint_degree`, `max_combo`) describe the
/// AIR being proved and are used in the DEEP-ALI bounds. Use
/// [`StarkSecurityParams::from_air`] to derive them automatically when an AIR
/// is available.
#[derive(Debug, Clone)]
pub struct StarkSecurityParams {
    /// log2(blowup factor); the FRI rate is ρ = 2^{-log_blowup}.
    pub fri_log_blowup: usize,
    /// log2(final FRI polynomial length) — controls when FRI stops folding.
    pub fri_log_final_poly_len: usize,
    /// log2(maximum FRI folding arity).
    pub fri_max_log_arity: usize,
    /// Number of FRI queries.
    pub fri_num_queries: usize,
    /// Bits of grinding ground at every FRI commit-phase round.
    pub fri_commit_proof_of_work_bits: usize,
    /// Bits of grinding ground once before sampling FRI queries.
    pub fri_query_proof_of_work_bits: usize,
    /// Bit-length of the field where FRI operates (typically the extension field).
    pub num_modulus_bits: usize,
    /// Collision resistance of the commitment hash, in bits.
    pub collision_resistance: usize,
    /// Total number of AIR constraints batched in ALI (base + extension).
    pub num_constraints: usize,
    /// Maximum AIR constraint degree. The Plonky3 prover requires this to be at most
    /// `blowup + 1` for the quotient to fit in the LDE.
    pub air_max_constraint_degree: usize,
    /// Maximum number of out-of-domain points referenced per AIR column
    /// (DEEP-ALI's `max_combo`). For a uni-STARK using `local`/`next` rotations this
    /// is `2`; `1` if no transition constraint is present.
    pub max_combo: usize,
}

impl StarkSecurityParams {
    /// Build security parameters explicitly from FRI parameters and the AIR shape.
    ///
    /// Use [`from_air`](Self::from_air) when an AIR is available — it derives
    /// `num_constraints` and `air_max_constraint_degree` from symbolic evaluation.
    pub const fn new<M>(
        fri_params: &FriParameters<M>,
        num_modulus_bits: usize,
        collision_resistance: usize,
        num_constraints: usize,
        air_max_constraint_degree: usize,
        max_combo: usize,
    ) -> Self {
        Self {
            fri_log_blowup: fri_params.log_blowup,
            fri_log_final_poly_len: fri_params.log_final_poly_len,
            fri_max_log_arity: fri_params.max_log_arity,
            fri_num_queries: fri_params.num_queries,
            fri_commit_proof_of_work_bits: fri_params.commit_proof_of_work_bits,
            fri_query_proof_of_work_bits: fri_params.query_proof_of_work_bits,
            num_modulus_bits,
            collision_resistance,
            num_constraints,
            air_max_constraint_degree,
            max_combo,
        }
    }

    /// Build security parameters by inspecting the AIR's symbolic constraints to derive
    /// `num_constraints` and `air_max_constraint_degree`. The caller supplies `max_combo`
    /// (typically `2` for a uni-STARK that uses `local`/`next`, `1` if no transition).
    ///
    /// `layout` must reflect any permutation/lookup columns: a base-only layout (e.g.
    /// `AirLayout::from_air`, which fills only the `BaseAir` widths) leaves the
    /// permutation fields at `0`, so permutation-argument constraints are not counted
    /// and security is overstated.
    pub fn from_air<F, EF, A, M>(
        fri_params: &FriParameters<M>,
        air: &A,
        layout: AirLayout,
        num_modulus_bits: usize,
        collision_resistance: usize,
        max_combo: usize,
    ) -> Self
    where
        F: Field,
        EF: ExtensionField<F>,
        A: Air<SymbolicAirBuilder<F, EF>>,
    {
        let (base, ext) = get_all_symbolic_constraints::<F, EF, A>(air, layout);
        let num_constraints = base.len() + ext.len();
        // Clamp to 1 so log2(·) stays finite when the AIR has no constraints.
        let base_deg = base.iter().map(|c| c.degree_multiple()).max().unwrap_or(0);
        let ext_deg = ext.iter().map(|c| c.degree_multiple()).max().unwrap_or(0);
        let air_max_constraint_degree = base_deg.max(ext_deg).max(1);
        Self::new(
            fri_params,
            num_modulus_bits,
            collision_resistance,
            num_constraints,
            air_max_constraint_degree,
            max_combo,
        )
    }
}

/// Conjectured security level (in bits) using the "random words" regime
/// of [2025/2010](https://eprint.iacr.org/2025/2010) §1.5.
///
/// The cited paper recommends proven bounds for deployment; users staying with
/// conjectured bounds should remain above the cutoff.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ConjecturedSecurity {
    pub security_bits: usize,
}

impl ConjecturedSecurity {
    /// Conjectured security from FRI parameters using the random-words formula
    /// ([2025/2010] §1.5). Requires `num_modulus_bits` (log2 of field size) for the η cutoff.
    pub fn compute(
        log_blowup: usize,
        num_queries: usize,
        query_proof_of_work_bits: usize,
        collision_resistance: usize,
        num_modulus_bits: usize,
    ) -> Self {
        let fri_bits = conjectured_fri_bits_random_words(log_blowup, num_queries, num_modulus_bits)
            + query_proof_of_work_bits;
        let mut bits = min(fri_bits, collision_resistance);
        bits = min(bits, num_modulus_bits);

        Self {
            security_bits: bits,
        }
    }

    /// Compute conjectured security from a parameter bundle.
    pub fn compute_from_params(params: &StarkSecurityParams) -> Self {
        Self::compute(
            params.fri_log_blowup,
            params.fri_num_queries,
            params.fri_query_proof_of_work_bits,
            params.collision_resistance,
            params.num_modulus_bits,
        )
    }
}

/// Computes conjectured FRI security bits from the random-words formula in
/// [2025/2010](https://eprint.iacr.org/2025/2010) §1.5:
///
/// `b = num_queries . (−log2(ρ + η))`, with `η ≈ (log2(e/ρ) . ρ) / log2(q)`,
///
/// where `b` is the achieved security in bits, `ρ` is the FRI rate, and `q` is the field size.
fn conjectured_fri_bits_random_words(
    log_blowup: usize,
    num_queries: usize,
    num_modulus_bits: usize,
) -> usize {
    if log_blowup == 0 || num_modulus_bits == 0 {
        return 0;
    }
    let log_blowup_f = log_blowup as f64;
    let rho = pow(2.0, -log_blowup_f);
    let log2_e_over_rho = core::f64::consts::LOG2_E + log_blowup_f;
    let eta = (log2_e_over_rho * rho) / num_modulus_bits as f64;
    let effective = rho + eta;
    if effective <= 0.0 || effective >= 1.0 {
        return 0;
    }
    let bits_per_query = -log2(effective);
    (num_queries as f64 * bits_per_query) as usize
}

/// Proven security level (in bits) of a STARK configuration.
///
/// Follows Theorems 2 and 3 of [2024/1553](https://eprint.iacr.org/2024/1553)
/// (round-by-round soundness; unique-decoding and list-decoding regimes), with the
/// improved LDR FRI commit-phase bound from [2025/2055](https://eprint.iacr.org/2025/2055)
/// Theorem 4.2. Cross-checked against [`soundcalc`](https://github.com/ethereum/soundcalc).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ProvenSecurity {
    pub unique_decoding_bits: usize,
    pub list_decoding_bits: usize,
}

impl ProvenSecurity {
    /// Best of the two regimes (unique-decoding and list-decoding).
    ///
    /// Each regime is an independent valid lower bound on round-by-round soundness, so
    /// their maximum is itself a valid (and tighter) lower bound on the proven security.
    #[inline]
    pub fn security_bits(&self) -> usize {
        max(self.unique_decoding_bits, self.list_decoding_bits)
    }

    /// Compute proven security from protocol parameters and the trace length.
    pub fn compute(params: &StarkSecurityParams, trace_length: usize) -> Self {
        // ρ = 1 (no blowup) makes the proven bounds vacuous.
        if params.fri_log_blowup == 0 || trace_length == 0 || params.num_modulus_bits == 0 {
            return Self {
                unique_decoding_bits: 0,
                list_decoding_bits: 0,
            };
        }

        let extension_field_bits = params.num_modulus_bits as f64;
        let blowup_factor = 1usize << params.fri_log_blowup;
        let lde_domain_size = trace_length * blowup_factor;
        let trace_domain_size = trace_length as f64;
        let lde_domain_size_f = lde_domain_size as f64;
        let num_fri_queries = params.fri_num_queries as f64;
        let query_grinding = params.fri_query_proof_of_work_bits as f64;
        let commit_grinding = params.fri_commit_proof_of_work_bits as f64;
        let air_max_deg = params.air_max_constraint_degree.max(1) as f64;
        debug_assert!(
            params.air_max_constraint_degree <= blowup_factor + 1,
            "AIR max constraint degree {} exceeds blowup+1 ({}); the prover cannot commit a quotient",
            params.air_max_constraint_degree,
            blowup_factor + 1
        );
        let max_combo = params.max_combo as f64;
        let num_constraints = params.num_constraints.max(1) as f64;
        let folding_factor = (1usize << params.fri_max_log_arity) as f64;

        let unique_decoding = min(
            proven_security_unique_decoding(
                extension_field_bits,
                num_fri_queries,
                query_grinding,
                commit_grinding,
                trace_domain_size,
                lde_domain_size,
                lde_domain_size_f,
                air_max_deg,
                max_combo,
                num_constraints,
                params.fri_log_final_poly_len,
                folding_factor,
            ),
            params.collision_resistance as u64,
        ) as usize;

        // Theorem 4.2 of [2025/2055] requires η > 0; bracket m and search for the optimum.
        let m_min: usize = 3;
        let m_max = min(compute_upper_m(trace_length), LDR_M_CAP);
        let list_decoding = if m_max < m_min {
            // No valid m in range (e.g. trivially small traces); LDR is vacuous.
            0
        } else {
            let m_optimal = (m_min..=m_max)
                .max_by_key(|&m| {
                    proven_security_list_decoding_m(
                        extension_field_bits,
                        blowup_factor,
                        num_fri_queries,
                        query_grinding,
                        commit_grinding,
                        trace_domain_size,
                        lde_domain_size_f,
                        air_max_deg,
                        max_combo,
                        num_constraints,
                        folding_factor,
                        m,
                    )
                })
                .expect("non-empty range");

            min(
                proven_security_list_decoding_m(
                    extension_field_bits,
                    blowup_factor,
                    num_fri_queries,
                    query_grinding,
                    commit_grinding,
                    trace_domain_size,
                    lde_domain_size_f,
                    air_max_deg,
                    max_combo,
                    num_constraints,
                    folding_factor,
                    m_optimal,
                ),
                params.collision_resistance as u64,
            ) as usize
        };

        Self {
            unique_decoding_bits: unique_decoding,
            list_decoding_bits: list_decoding,
        }
    }

    /// Compute proven security using a parameter bundle and the proof's degree bits.
    ///
    /// `degree_bits` already reflects the committed-polynomial size (post-zk padding,
    /// when applicable), so the trace-domain size used for security analysis is `2^degree_bits`.
    pub fn compute_from_proof(degree_bits: usize, params: &StarkSecurityParams) -> Self {
        let trace_length = 1usize << degree_bits;
        Self::compute(params, trace_length)
    }
}

/// Performance bound on the searched proximity parameter `m`.
const LDR_M_CAP: usize = 1000;

/// Computes the largest proximity parameter `m` such that the η > 0 precondition of
/// the proof of Theorem 1 in [2021/582](https://eprint.iacr.org/2021/582) holds.
/// See also Theorem 2 and its proof in [2024/1553](https://eprint.iacr.org/2024/1553).
///
/// Returns the raw theorem-derived bound; the caller applies [`LDR_M_CAP`].
fn compute_upper_m(trace_domain_size: usize) -> usize {
    if trace_domain_size == 0 {
        return 0;
    }
    let h = trace_domain_size as f64;
    let ratio = (h + 2.0) / h;
    let m_max = ceil(1.0 / (2.0 * (sqrt(ratio) - 1.0))) as usize;

    assert!(
        (m_max as f64) >= h / 2.0,
        "compute_upper_m: m_max = {} < h/2 = {} (closed-form drifted from theorem)",
        m_max,
        h / 2.0,
    );
    m_max
}

/// Round-by-round soundness in the unique-decoding regime (Theorem 3 of [2024/1553]).
///
/// RbR soundness is bounded by `max_i ε_i`, so we report `min_i (-log2 ε_i)`. A strict
/// union bound would be tighter by ~`log2(num_components)` (≈ 2–3 bits); we keep `min`
/// to match `soundcalc/circuits/deep_ali.py:89`.
fn proven_security_unique_decoding(
    extension_field_bits: f64,
    num_fri_queries: f64,
    query_grinding: f64,
    commit_grinding: f64,
    trace_domain_size: f64,
    lde_domain_size: usize,
    lde_domain_size_f: f64,
    air_max_deg: f64,
    max_combo: f64,
    num_constraints: f64,
    log_final_poly_len: usize,
    folding_factor: f64,
) -> u64 {
    // UDR agreement parameter α = (1 + ρ⁺)/2; ρ⁺ accounts for trace-side expansion
    // from out-of-domain openings.
    let rho_plus = (trace_domain_size + max_combo) / lde_domain_size_f;
    let alpha = (1.0 + rho_plus) * 0.5;

    // Multi-point quotient soundness precondition from [2020/654] §4.1.3, written
    // against the UDR agreement parameter α used below (i.e. θ = 1 − α) rather than
    // the raw rate `soundcalc` uses; both hold for any blowup ≥ 2.
    if trace_domain_size + max_combo >= alpha * lde_domain_size_f {
        return 0;
    }

    let mut epsilons_bits_neg = Vec::new();

    // ALI: ε_ALI = L⁺ · num_constraints / |F|; in UDR, L⁺ = 1.
    epsilons_bits_neg.push(extension_field_bits - log2(num_constraints));

    // DEEP: ε_DEEP = (max_deg · (k + max_combo - 1) + (k - 1)) / |F|.
    // (`soundcalc` divides by `|F| - k - D` — negligibly different in practice.)
    let deep_factor =
        air_max_deg * (trace_domain_size + max_combo - 1.0) + (trace_domain_size - 1.0);
    epsilons_bits_neg.push(extension_field_bits - log2(deep_factor.max(1.0)));

    // FRI commit phase: per round, ε ≤ (folding_factor − 1)·(n + 1)/|F|. Each layer
    // yields the same bound, so push once when at least one fold occurs. We use the
    // plain `(n + 1)`; `soundcalc` uses the tighter `(γn + 1)` with γ = (1 − ρ)/2 ≤ 1,
    // so this is a conservative simplification (~1–2 bits).
    let num_fri_layers = log2_floor_usize(lde_domain_size).saturating_sub(log_final_poly_len);
    if num_fri_layers > 0 {
        let folding_minus_one = (folding_factor - 1.0).max(1.0);
        epsilons_bits_neg.push(
            extension_field_bits - log2(folding_minus_one * (lde_domain_size_f + 1.0))
                + commit_grinding,
        );
    }

    // FRI query phase: ε ≤ αᵏ.
    let epsilon_k = query_grinding - log2(pow(alpha, num_fri_queries));
    epsilons_bits_neg.push(epsilon_k);

    let min_bits = epsilons_bits_neg
        .into_iter()
        .fold(f64::INFINITY, |a, b| if b < a { b } else { a });
    min_bits.max(0.0) as u64
}

/// Round-by-round soundness in the list-decoding regime (Theorem 2 of [2024/1553]
/// with BCHKS25 [2025/2055] Theorem 4.2 commit-phase bound).
///
/// `m` is the explicit Johnson-bound multiplicity controlling the proximity gap η.
fn proven_security_list_decoding_m(
    extension_field_bits: f64,
    blowup_factor: usize,
    num_fri_queries: f64,
    query_grinding: f64,
    commit_grinding: f64,
    trace_domain_size: f64,
    lde_domain_size: f64,
    air_max_deg: f64,
    max_combo: f64,
    num_constraints: f64,
    folding_factor: f64,
    m: usize,
) -> u64 {
    let rho = 1.0 / blowup_factor as f64;
    let sqrt_rho = sqrt(rho);
    let m_f = m as f64;
    let m_shifted = m_f + 0.5;
    // BCHKS25 explicit-m proximity parameter γ = 1 − √ρ − η, with η = √ρ/(2m).
    let pp = 1.0 - sqrt_rho * (1.0 + 0.5 / m_f);
    if pp <= 0.0 {
        return 0;
    }
    let alpha = (1.0 + 0.5 / m_f) * sqrt_rho;

    // BCHKS25 list size L⁺ = (m + 0.5) / √ρ (matches `soundcalc/proxgaps/johnson_bound.py`
    // `get_max_list_size` explicit_m branch).
    let l = m_shifted / sqrt_rho;
    if !l.is_finite() || l <= 0.0 {
        return 0;
    }

    // Multi-point quotient soundness precondition from [2020/654] §4.1.3 with θ = γ.
    if trace_domain_size + max_combo >= (1.0 - pp) * lde_domain_size {
        return 0;
    }

    let mut epsilons_bits_neg = Vec::new();

    // ALI: ε_ALI = L⁺ · num_constraints / |F|.
    epsilons_bits_neg.push(-log2(l) - log2(num_constraints) + extension_field_bits);

    // DEEP: ε_DEEP = L⁺ · (max_deg · (k + max_combo - 1) + (k - 1)) / |F|.
    // Linear in L⁺, matching `soundcalc/circuits/deep_ali.py:170`.
    let deep_factor =
        air_max_deg * (trace_domain_size + max_combo - 1.0) + (trace_domain_size - 1.0);
    epsilons_bits_neg.push(-log2(l) - log2(deep_factor.max(1.0)) + extension_field_bits);

    // FRI commit phase, per round (BCHKS25 Theorem 4.2 / `error_powers`):
    //   ε_lin   = ((2·m'⁵ + 3·m'·γρ)·n / (3·ρ^{3/2}) + m'/√ρ) / |F|
    //   ε_round = ε_lin · (folding_factor − 1)
    // We use the round-0 LDE size `n = lde_domain_size` for every round. `soundcalc`
    // shrinks `n` per round, but RbR soundness is the max over rounds and the worst
    // round is round 0 (largest `n`), so this is exact for RbR — not a loosening.
    let n = lde_domain_size;
    let num = (2.0 * pow(m_shifted, 5.0) + 3.0 * m_shifted * pp * rho) * n;
    let den = 3.0 * rho * sqrt_rho;
    let epsilon_linear = num / den + m_shifted / sqrt_rho;
    let folding_minus_one = (folding_factor - 1.0).max(1.0);
    let epsilon_powers = epsilon_linear * folding_minus_one;
    let epsilon_3_bits_neg = extension_field_bits - log2(epsilon_powers.max(1.0)) + commit_grinding;

    // [2024/1553] also gives an n/q-style bound per round; report the smaller-bits of
    // the two. Factors `folding_factor` (vs `folding_factor − 1`) and `(2m + 1)` (vs
    // `m + 0.5`) here are slightly conservative.
    let term_from_n_over_q = extension_field_bits
        - log2(folding_factor)
        - log2(lde_domain_size + 1.0)
        - log2(2.0 * m_f + 1.0)
        + 0.5 * log2(rho)
        + commit_grinding;
    epsilons_bits_neg.push(epsilon_3_bits_neg.min(term_from_n_over_q));

    // FRI query phase: ε ≤ αᵏ.
    if alpha >= 1.0 {
        return 0;
    }
    let epsilon_k = query_grinding - log2(pow(alpha, num_fri_queries));
    epsilons_bits_neg.push(epsilon_k);

    let min_bits = epsilons_bits_neg
        .into_iter()
        .fold(f64::INFINITY, |a, b| if b < a { b } else { a });
    min_bits.max(0.0) as u64
}

#[cfg(test)]
mod tests {
    use super::*;

    const TEST_NUM_CONSTRAINTS: usize = 1;
    const TEST_AIR_MAX_DEG: usize = 2;
    const TEST_MAX_COMBO: usize = 2;

    #[test]
    fn conjectured_security_bounded_by_collision_resistance() {
        let s = ConjecturedSecurity::compute(8, 32, 0, 128, 128);
        assert_eq!(s.security_bits, 128);
    }

    #[test]
    fn conjectured_security_random_words_formula() {
        let s = ConjecturedSecurity::compute(4, 20, 8, 256, 128);
        assert!(s.security_bits > 0 && s.security_bits <= 256);
    }

    #[test]
    fn conjectured_security_log_blowup_zero_returns_zero_fri_bits() {
        let s = ConjecturedSecurity::compute(0, 100, 16, 128, 256);
        assert_eq!(s.security_bits, 16);
    }

    #[test]
    fn conjectured_fri_100_queries_benchmark_below_100_bits() {
        let fri_bits = conjectured_fri_bits_random_words(1, 100, 256);
        assert!(
            fri_bits < 100,
            "100 queries at ρ=1/2 (log_blowup=1) should give <100 bits per random-words formula, got {}",
            fri_bits
        );
    }

    fn benchmark_high_arity_params(num_modulus_bits: usize) -> StarkSecurityParams {
        // Mirrors `FriParameters::new_benchmark_high_arity`.
        StarkSecurityParams {
            fri_log_blowup: 1,
            fri_log_final_poly_len: 0,
            fri_max_log_arity: 3,
            fri_num_queries: 100,
            fri_commit_proof_of_work_bits: 0,
            fri_query_proof_of_work_bits: 16,
            num_modulus_bits,
            collision_resistance: 128,
            num_constraints: TEST_NUM_CONSTRAINTS,
            air_max_constraint_degree: TEST_AIR_MAX_DEG,
            max_combo: TEST_MAX_COMBO,
        }
    }

    #[test]
    fn proven_security_lower_than_conjectured_for_same_params() {
        let c = ConjecturedSecurity::compute(8, 32, 8, 256, 252);
        let mut params = benchmark_high_arity_params(252);
        params.fri_log_blowup = 8;
        params.fri_num_queries = 32;
        params.fri_query_proof_of_work_bits = 8;
        let p = ProvenSecurity::compute(&params, 1 << 16);
        assert!(p.security_bits() <= c.security_bits);
    }

    #[test]
    fn proven_security_log_blowup_zero_returns_zero() {
        let mut params = benchmark_high_arity_params(252);
        params.fri_log_blowup = 0;
        let p = ProvenSecurity::compute(&params, 1 << 16);
        assert_eq!(p.unique_decoding_bits, 0);
        assert_eq!(p.list_decoding_bits, 0);
    }

    #[test]
    fn proven_security_tiny_trace_returns_zero_ldr() {
        let params = benchmark_high_arity_params(252);
        let p = ProvenSecurity::compute(&params, 1);
        assert_eq!(p.list_decoding_bits, 0);
    }

    #[test]
    fn commit_pow_increases_or_holds_security() {
        let mut params = benchmark_high_arity_params(252);
        params.fri_commit_proof_of_work_bits = 0;
        let p0 = ProvenSecurity::compute(&params, 1 << 20);
        params.fri_commit_proof_of_work_bits = 16;
        let p16 = ProvenSecurity::compute(&params, 1 << 20);
        assert!(p16.unique_decoding_bits >= p0.unique_decoding_bits);
        assert!(p16.list_decoding_bits >= p0.list_decoding_bits);
    }

    #[test]
    fn more_constraints_decreases_or_holds_security() {
        let mut params = benchmark_high_arity_params(252);
        params.num_constraints = 1;
        let p1 = ProvenSecurity::compute(&params, 1 << 20);
        params.num_constraints = 1024;
        let p1024 = ProvenSecurity::compute(&params, 1 << 20);
        assert!(p1024.unique_decoding_bits <= p1.unique_decoding_bits);
        assert!(p1024.list_decoding_bits <= p1.list_decoding_bits);
    }

    #[test]
    fn more_max_combo_decreases_or_holds_security() {
        let mut params = benchmark_high_arity_params(252);
        params.max_combo = 1;
        let p1 = ProvenSecurity::compute(&params, 1 << 20);
        params.max_combo = 8;
        let p8 = ProvenSecurity::compute(&params, 1 << 20);
        assert!(p8.unique_decoding_bits <= p1.unique_decoding_bits);
        assert!(p8.list_decoding_bits <= p1.list_decoding_bits);
    }

    #[test]
    fn higher_arity_decreases_or_holds_security() {
        let mut params = benchmark_high_arity_params(252);
        params.fri_max_log_arity = 1;
        let p_a2 = ProvenSecurity::compute(&params, 1 << 20);
        params.fri_max_log_arity = 3;
        let p_a8 = ProvenSecurity::compute(&params, 1 << 20);
        assert!(p_a8.list_decoding_bits <= p_a2.list_decoding_bits);
        assert!(p_a8.unique_decoding_bits <= p_a2.unique_decoding_bits);
    }

    // Regression vector pinning the proven-security output for a fixed configuration:
    // log_blowup=1, num_queries=100, query_pow=16, commit_pow=0, max_log_arity=3,
    // |F|=252 bits, trace 2^20, num_constraints=1, max_deg=2, max_combo=2.
    #[test]
    fn proven_security_regression_benchmark_high_arity() {
        let params = benchmark_high_arity_params(252);
        let p = ProvenSecurity::compute(&params, 1 << 20);
        assert_eq!(p.unique_decoding_bits, 57);
        assert_eq!(p.list_decoding_bits, 65);
    }
}
