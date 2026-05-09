//! STARK proof security level computation.
//!
//! Provides conjectured and proven security level estimates in bits.
//!
//! References:
//! - ethSTARK eprint.iacr.org/2021/582,
//! - Proximity Gaps for Reed-Solomon Codes eprint.iacr.org/2020/654,
//!
//! - On the Distribution of the Distances of Random Words eprint.iacr.org/2025/2010,
//! - On the Security of STARKs with FRI eprint.iacr.org/2025/2046,
//!
//! The last two papers recommend using proven bounds in deployment. If desiring to keep using conjectural bounds,
//! they recommend staying above the "random words" cutoff (see [2025/2010](https://eprint.iacr.org/2025/2010) §1.5, formula (5)).

#![allow(clippy::too_many_arguments)]

use alloc::vec::Vec;
use core::cmp::{max, min};

use libm::{ceil, log2, pow, sqrt};
use p3_fri::FriParameters;
use p3_util::log2_floor_usize;

/// Parameters required to compute STARK proof security level.
///
/// All FRI-related fields come from the PCS FRI parameters; the remaining fields
/// must be supplied by the caller.
#[derive(Debug, Clone)]
pub struct StarkSecurityParams {
    pub fri_log_blowup: usize,
    pub fri_log_final_poly_len: usize,
    pub fri_max_log_arity: usize,
    pub fri_num_queries: usize,
    pub fri_query_proof_of_work_bits: usize,
    pub num_modulus_bits: usize,
    pub collision_resistance: usize,
}

impl StarkSecurityParams {
    /// Build security parameters from FRI parameters.
    ///
    /// `num_modulus_bits` is the bit-length of the field over which FRI operates
    /// (typically the extension field). `collision_resistance` is the bit-strength
    /// of the hash used to commit (e.g. 128 for typical 256-bit collision-resistant hashes).
    pub const fn new<M>(
        fri_params: &FriParameters<M>,
        num_modulus_bits: usize,
        collision_resistance: usize,
    ) -> Self {
        Self {
            fri_log_blowup: fri_params.log_blowup,
            fri_log_final_poly_len: fri_params.log_final_poly_len,
            fri_max_log_arity: fri_params.max_log_arity,
            fri_num_queries: fri_params.num_queries,
            fri_query_proof_of_work_bits: fri_params.query_proof_of_work_bits,
            num_modulus_bits,
            collision_resistance,
        }
    }
}

/// Conjectured security level (in bits) using the "random words" regime.
///
/// Uses the formula from [2025/2010](https://eprint.iacr.org/2025/2010) §1.5.
///
/// Note that [2025/2010](https://eprint.iacr.org/2025/2010) recommends proven bounds for deployment,
/// and advises those using the conjectured security to stay above the cutoff.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ConjecturedSecurity {
    pub security_bits: usize,
}

impl ConjecturedSecurity {
    /// Conjectured security from FRI parameters using the random-words formula
    /// (eprint.iacr.org/2025/2010 §1.5). Requires `num_modulus_bits` (log₂ of
    /// field size) for the η cutoff.
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
///     b = num_queries · (−log₂(ρ + η)),    η ≈ (log₂(e/ρ) · ρ) / log₂(q),
///
/// where `b` is the achieved security in bits, `ρ` is the FRI rate, and `q` is the field size.
///
/// Note that [2025/2010](https://eprint.iacr.org/2025/2010) recommends proven bounds for deployment,
/// and advises those using the conjectured security to stay above the cutoff.
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
/// Follows Theorem 2 and Theorem 3 in [2024/1553](https://eprint.iacr.org/2024/1553)
/// (round-by-round soundness; unique-decoding and list-decoding regimes), with the
/// improved LDR FRI commit-phase bounds from [2025/2055](https://eprint.iacr.org/2025/2055).
// The proven security calculation is inspired from Winterfell's implementation:
// https://github.com/facebook/winterfell/blob/main/winterfell/src/security.rs
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

    /// Compute proven security from protocol parameters (Theorem 2 & 3 in [2024/1553](https://eprint.iacr.org/2024/1553)).
    pub fn compute(
        log_blowup: usize,
        log_final_poly_len: usize,
        max_log_arity: usize,
        num_queries: usize,
        query_proof_of_work_bits: usize,
        num_modulus_bits: usize,
        trace_length: usize,
        collision_resistance: usize,
    ) -> Self {
        let extension_field_bits = num_modulus_bits as f64;
        let blowup_factor = 1usize << log_blowup;
        let lde_domain_size = trace_length * blowup_factor;
        let trace_domain_size = trace_length as f64;
        let lde_domain_size_f = lde_domain_size as f64;
        let num_fri_queries = num_queries as f64;
        let grinding_factor = query_proof_of_work_bits as f64;
        let num_openings = 2.0f64;
        let max_deg = blowup_factor as f64 + 1.0;
        let folding_factor = (1usize << max_log_arity) as f64;

        let unique_decoding = min(
            proven_security_unique_decoding(
                extension_field_bits,
                num_fri_queries,
                grinding_factor,
                trace_domain_size,
                lde_domain_size,
                lde_domain_size_f,
                max_deg,
                num_openings,
                log_final_poly_len,
                folding_factor,
            ),
            collision_resistance as u64,
        ) as usize;

        let m_min: usize = 3;
        let m_max = compute_upper_m(trace_length);
        let m_optimal = (m_min..=m_max)
            .max_by_key(|&m| {
                proven_security_list_decoding_m(
                    extension_field_bits,
                    blowup_factor,
                    num_fri_queries,
                    grinding_factor,
                    trace_domain_size,
                    lde_domain_size_f,
                    max_deg,
                    num_openings,
                    folding_factor,
                    m,
                )
            })
            .unwrap_or(m_min);

        let list_decoding = min(
            proven_security_list_decoding_m(
                extension_field_bits,
                blowup_factor,
                num_fri_queries,
                grinding_factor,
                trace_domain_size,
                lde_domain_size_f,
                max_deg,
                num_openings,
                folding_factor,
                m_optimal,
            ),
            collision_resistance as u64,
        ) as usize;

        Self {
            unique_decoding_bits: unique_decoding,
            list_decoding_bits: list_decoding,
        }
    }

    /// Compute proven security using a parameter bundle and the proof's degree bits.
    pub fn compute_from_proof(degree_bits: usize, params: &StarkSecurityParams) -> Self {
        // `degree_bits` already reflects the committed-polynomial size (post-zk padding,
        // when applicable), so the trace-domain size used for security analysis is `2^degree_bits`.
        let trace_length = 1usize << degree_bits;

        Self::compute(
            params.fri_log_blowup,
            params.fri_log_final_poly_len,
            params.fri_max_log_arity,
            params.fri_num_queries,
            params.fri_query_proof_of_work_bits,
            params.num_modulus_bits,
            trace_length,
            params.collision_resistance,
        )
    }
}

/// Computes the largest proximity parameter m such that eta is greater than 0 in the proof of
/// Theorem 1 in https://eprint.iacr.org/2021/582. See Theorem 2 in https://eprint.iacr.org/2024/1553
/// and its proof for more on this point.
///
/// The bound on m in Theorem 2 in https://eprint.iacr.org/2024/1553 is sufficient but we can use
/// the following to compute a better bound.
fn compute_upper_m(trace_domain_size: usize) -> usize {
    if trace_domain_size == 0 {
        return 3;
    }
    let h = trace_domain_size as f64;
    let ratio = (h + 2.0) / h;
    let m_max = ceil(1.0 / (2.0 * (sqrt(ratio) - 1.0))) as usize;
    debug_assert!(
        (m_max as f64) >= h / 2.0,
        "the bound in the theorem should be tighter"
    );
    // We cap the range to 1000 as the optimal m value will be in the lower range of [m_min, m_max]
    // since increasing m too much will lead to a deterioration in the FRI commit soundness making
    // any benefit gained in the FRI query soundess mute.
    min(m_max, 1000)
}

fn proven_security_unique_decoding(
    extension_field_bits: f64,
    num_fri_queries: f64,
    grinding_factor: f64,
    trace_domain_size: f64,
    lde_domain_size: usize,
    lde_domain_size_f: f64,
    max_deg: f64,
    num_openings: f64,
    log_final_poly_len: usize,
    folding_factor: f64,
) -> u64 {
    let rho_plus = (trace_domain_size + num_openings) / lde_domain_size_f;
    let alpha = (1.0 + rho_plus) * 0.5;
    let constraint_batching = 1.0f64;
    let deep_batching = 1.0f64;

    // Theorem 3 in https://eprint.iacr.org/2024/1553
    let mut epsilons_bits_neg = Vec::new();

    // ALI related soundness error.
    epsilons_bits_neg.push(-log2(constraint_batching) + extension_field_bits);

    // DEEP related soundness error.
    epsilons_bits_neg.push(
        -log2(max_deg * (trace_domain_size + num_openings - 1.0) + (trace_domain_size - 1.0))
            + extension_field_bits,
    );

    // FRI commit-phase (i.e., pre-query) soundness error.
    epsilons_bits_neg.push(extension_field_bits - log2(lde_domain_size_f * deep_batching));

    // Per-layer ε_i for i in [3..(k-1)]; each layer yields the same bound, so push it once
    // when at least one fold occurs.
    let num_fri_layers = log2_floor_usize(lde_domain_size).saturating_sub(log_final_poly_len);
    if num_fri_layers > 0 {
        epsilons_bits_neg
            .push(extension_field_bits - log2((folding_factor - 1.0) * (lde_domain_size_f + 1.0)));
    }

    // FRI query-phase soundness error.
    let epsilon_k = grinding_factor - log2(pow(alpha, num_fri_queries));
    epsilons_bits_neg.push(epsilon_k);

    // round-by-round (RbR) soundness error
    let min_bits = epsilons_bits_neg
        .into_iter()
        .fold(f64::INFINITY, |a, b| if b < a { b } else { a });
    min_bits.max(0.0) as u64
}

fn proven_security_list_decoding_m(
    extension_field_bits: f64,
    blowup_factor: usize,
    num_fri_queries: f64,
    grinding_factor: f64,
    trace_domain_size: f64,
    lde_domain_size: f64,
    max_deg: f64,
    num_openings: f64,
    folding_factor: f64,
    m: usize,
) -> u64 {
    let rho = 1.0 / blowup_factor as f64;
    let m_f = m as f64;
    let alpha = (1.0 + 0.5 / m_f) * sqrt(rho);
    let constraint_batching = 1.0f64;
    let deep_batching = 1.0f64;

    // list size
    let l = m_f / (rho - (2.0 * m_f / lde_domain_size));
    if l <= 0.0 || !l.is_finite() {
        return 0;
    }

    // We apply Theorem 2 in https://eprint.iacr.org/2024/1553, with the improved
    // LDR FRI commit phase bounds from https://eprint.iacr.org/2025/2055.
    let mut epsilons_bits_neg = Vec::new();

    // ALI related soundness error.
    epsilons_bits_neg.push(-log2(l) - log2(constraint_batching) + extension_field_bits);

    // DEEP related soundness error. The list-decoding analysis carries an `l²` factor
    // (see Theorem 2 in https://eprint.iacr.org/2024/1553).
    epsilons_bits_neg.push(
        -log2(
            l * l
                * (max_deg * (trace_domain_size + num_openings - 1.0) + (trace_domain_size - 1.0)),
        ) + extension_field_bits,
    );

    // FRI commit-phase (i.e., pre-query) soundness error, using the improved
    // proximity-gap bounds from 2025/2055.
    //
    // Base (without batching constant):
    //   -log2(ε₃') = extension_field_bits
    //                - log2( 2 * (m + 0.5)^5 / (3 * ρ^{3/2}) * n ),
    // where n is the LDE domain size and ρ is the FRI rate.
    let epsilon_3_no_batching = extension_field_bits
        - log2((2.0 * pow(m_f + 0.5, 5.0) / (3.0 * pow(rho, 1.5))) * lde_domain_size);

    // Include batching constant only in ε₃.
    let epsilon_3_bits_neg = epsilon_3_no_batching - log2(deep_batching);
    epsilons_bits_neg.push(epsilon_3_bits_neg);

    // Intermediate FRI layers ε_i, i ∈ [4..k−1].
    // We bound them by the minimum of:
    // - the commit-phase base term ε₃', and
    // - an n/q-style term with folding factor and Johnson-gap dependent constant.
    let term_from_e3 = epsilon_3_no_batching;
    let term_from_n_over_q = extension_field_bits
        - log2(folding_factor)
        - log2(lde_domain_size + 1.0)
        - log2(2.0 * m_f + 1.0)
        + 0.5 * log2(rho);
    let epsilon_i_min_bits_neg = term_from_e3.min(term_from_n_over_q);
    epsilons_bits_neg.push(epsilon_i_min_bits_neg);

    // FRI query-phase soundness error.
    let epsilon_k = grinding_factor - log2(pow(alpha, num_fri_queries));
    epsilons_bits_neg.push(epsilon_k);

    // round-by-round (RbR) soundness error
    let min_bits = epsilons_bits_neg
        .into_iter()
        .fold(f64::INFINITY, |a, b| if b < a { b } else { a });
    min_bits.max(0.0) as u64
}

#[cfg(test)]
mod tests {
    use super::*;

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
    fn proven_security_lower_than_conjectured_for_same_params() {
        let c = ConjecturedSecurity::compute(8, 32, 8, 256, 252);
        let p = ProvenSecurity::compute(8, 0, 1, 32, 8, 252, 1 << 16, 256);
        assert!(p.security_bits() <= c.security_bits);
    }

    // Prior [2025/2010](https://eprint.iacr.org/2025/2010) and [2025/2046](https://eprint.iacr.org/2025/2046),
    // conjectured security was simply computed as `num_queries * bits_per_query`.
    // This test highlights that the new conjectured security estimate based on
    // the random-words formula requires more queries.
    #[test]
    fn conjectured_fri_100_queries_benchmark_below_100_bits() {
        let fri_bits = conjectured_fri_bits_random_words(1, 100, 256);
        assert!(
            fri_bits < 100,
            "100 queries at rho=1/2 (log_blowup=1) should give <100 bits per random-words formula, got {}",
            fri_bits
        );
    }
}
