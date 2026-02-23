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
use core::cmp::min;

use libm::{ceil, log2, pow, sqrt};
use p3_air::{Air, BaseAir};
use p3_commit::Mmcs;
use p3_field::{ExtensionField, Field};
use p3_fri::FriParameters;

use crate::{SymbolicAirBuilder, get_log_num_quotient_chunks, get_symbolic_constraints};

/// Parameters required to compute STARK proof security level.
///
/// All FRI-related fields come from the PCS FRI parameters; the rest from the
/// AIR and proof (trace dimensions, constraint count, quotient chunks).
#[derive(Debug, Clone)]
pub struct StarkSecurityParams {
    pub fri_log_blowup: usize,
    pub fri_log_final_poly_len: usize,
    pub fri_max_log_arity: usize,
    pub fri_num_queries: usize,
    pub fri_query_proof_of_work_bits: usize,
    pub num_modulus_bits: usize,
    pub collision_resistance: usize,
    pub is_zk: usize,
    pub trace_width: usize,
    pub num_constraints: usize,
    pub num_quotient_chunks: usize,
}

impl StarkSecurityParams {
    /// Build security parameters from FRI parameters and the AIR.
    ///
    /// If the AIR has preprocessed columns or public values, use [`from_air`](Self::from_air) instead.
    pub fn new<F, EF, A, M>(is_zk: bool, fri_params: &FriParameters<M>, air: &A) -> Self
    where
        F: Field,
        EF: ExtensionField<F>,
        A: Air<SymbolicAirBuilder<F>> + BaseAir<F>,
        M: Mmcs<EF>,
    {
        let is_zk_usize = is_zk as usize;
        let trace_width = air.width();
        let num_constraints = get_symbolic_constraints(air, 0, 0).len();
        let log_num_quotient_chunks = get_log_num_quotient_chunks(air, 0, 0, is_zk_usize);
        let num_quotient_chunks = 1 << (log_num_quotient_chunks + is_zk_usize);

        Self {
            fri_log_blowup: fri_params.log_blowup,
            fri_log_final_poly_len: fri_params.log_final_poly_len,
            fri_max_log_arity: fri_params.max_log_arity,
            fri_num_queries: fri_params.num_queries,
            fri_query_proof_of_work_bits: fri_params.query_proof_of_work_bits,
            num_modulus_bits: EF::bits(),
            collision_resistance: 128,
            is_zk: is_zk_usize,
            trace_width,
            num_constraints,
            num_quotient_chunks,
        }
    }

    /// Build security parameters.
    ///
    /// This is the preferred way to build [`StarkSecurityParams`] if
    /// the AIR has preprocessed columns or public values.
    pub fn from_air<F, A>(
        fri_log_blowup: usize,
        fri_log_final_poly_len: usize,
        fri_max_log_arity: usize,
        fri_num_queries: usize,
        fri_query_proof_of_work_bits: usize,
        num_modulus_bits: usize,
        collision_resistance: usize,
        is_zk: usize,
        air: &A,
        preprocessed_width: usize,
        num_public_values: usize,
    ) -> Self
    where
        F: Field,
        A: Air<SymbolicAirBuilder<F>> + BaseAir<F>,
    {
        let trace_width = air.width();
        let num_constraints =
            get_symbolic_constraints(air, preprocessed_width, num_public_values).len();
        let log_num_quotient_chunks =
            get_log_num_quotient_chunks(air, preprocessed_width, num_public_values, is_zk);
        let num_quotient_chunks = 1 << (log_num_quotient_chunks + is_zk);

        Self {
            fri_log_blowup,
            fri_log_final_poly_len,
            fri_max_log_arity,
            fri_num_queries,
            fri_query_proof_of_work_bits,
            num_modulus_bits,
            collision_resistance,
            is_zk,
            trace_width,
            num_constraints,
            num_quotient_chunks,
        }
    }
}

fn log2_usize(n: usize) -> usize {
    assert!(n > 0, "log2(0) undefined");
    (usize::BITS - 1 - n.leading_zeros()) as usize
}

/// Computes conjectured FRI security bits from the random-words formula in
/// [2025/2010](https://eprint.iacr.org/2025/2010) §1.5.
///
/// It is given by `num_queries = b · (−log₂(ρ + η))` with `η ≈ (log₂(e/ρ)·ρ) / log₂(q)`
/// where `b` is the targeted security bits and `ρ` is the FRI rate.
///
/// Note that [2025/2010](https://eprint.iacr.org/2025/2010) recommends proven bounds for deployment,
/// and advises those using the conjectured security to stay above the cutoff.
const fn conjectured_fri_bits_random_words(
    log_blowup: usize,
    num_queries: usize,
    num_modulus_bits: usize,
) -> usize {
    /// Scaling factor for fixed-point arithmetic.
    const SCALING_FACTOR: u64 = 20;

    /// log₂(e) ≈ 1.4427, scaled by 1<<SCALING_FACTOR.
    const LOG2_E_SCALED: u64 = 1_513_561;

    /// ln(2) ≈ 0.693, scaled by 1<<SCALING_FACTOR.
    const LN2_SCALED: u64 = 726_817;

    if log_blowup >= 64 || num_modulus_bits == 0 {
        return 0;
    }
    let one: u64 = 1 << 63;
    let rho = one >> log_blowup;
    let log2_e_over_rho = LOG2_E_SCALED + ((log_blowup as u64) << SCALING_FACTOR);
    let log2_q = num_modulus_bits as u64;
    let divisor = (log2_q << SCALING_FACTOR) as u128;
    let eta = ((log2_e_over_rho as u128 * rho as u128) / divisor) as u64;
    let effective = rho + eta;
    if effective == 0 || effective >= one {
        return 0;
    }
    let int_bits = effective.leading_zeros() as usize;
    let top = one >> int_bits;
    let frac = effective - top;
    let bits_per_query_fp: u64 = if frac == 0 {
        (int_bits as u64) << SCALING_FACTOR
    } else {
        let denom = (top as u128) * (LN2_SCALED as u128);
        let correction = ((frac as u128) << 40) / denom;
        ((int_bits as u64) << SCALING_FACTOR).saturating_sub(correction as u64)
    };
    let total_fp = (num_queries as u128) * (bits_per_query_fp as u128);
    (total_fp >> SCALING_FACTOR) as usize
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

/// Proven security level (in bits) of a STARK configuration.
///
/// Follows Theorem 2 and Theorem 3 in [2024/1553](https://eprint.iacr.org/2024/1553)
/// (round-by-round soundness; unique-decoding and list-decoding regimes).
// The proven security calculation is inspired from Winterfell's implementation:
// https://github.com/facebook/winterfell/blob/main/winterfell/src/security.rs
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ProvenSecurity {
    pub unique_decoding_bits: usize,
    pub list_decoding_bits: usize,
}

impl ProvenSecurity {
    /// Minimum security level of the two regimes (unique-decoding and list-decoding).
    #[inline]
    pub fn security_bits(&self) -> usize {
        min(self.unique_decoding_bits, self.list_decoding_bits)
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
        let blowup_factor = 1 << log_blowup;
        let lde_domain_size = trace_length * blowup_factor;
        let trace_domain_size = trace_length as f64;
        let lde_domain_size_f = lde_domain_size as f64;
        let num_fri_queries = num_queries as f64;
        let grinding_factor = query_proof_of_work_bits as f64;
        let num_openings = 2.0f64;
        let max_deg = blowup_factor as f64 + 1.0;
        let folding_factor = (1 << max_log_arity) as f64;

        let unique_decoding = min(
            proven_security_unique_decoding(
                extension_field_bits,
                num_fri_queries,
                grinding_factor,
                trace_domain_size,
                lde_domain_size_f,
                max_deg,
                num_openings,
                trace_length,
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
        let trace_length = 1 << degree_bits.saturating_sub(params.is_zk);

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
    let h = trace_domain_size as f64;
    if h <= 0.0 {
        return 3;
    }
    let ratio = (h + 2.0) / h;
    let m_max = ceil(1.0 / (2.0 * (sqrt(ratio) - 1.0)));
    let m_max = if m_max >= h / 2.0 {
        m_max as usize
    } else {
        (h / 2.0) as usize
    };

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
    lde_domain_size: f64,
    max_deg: f64,
    num_openings: f64,
    trace_length: usize,
    log_final_poly_len: usize,
    folding_factor: f64,
) -> u64 {
    let rho_plus = (trace_domain_size + num_openings) / lde_domain_size;
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
    epsilons_bits_neg.push(extension_field_bits - log2(lde_domain_size * deep_batching));

    let num_fri_layers = log2_usize(trace_length).saturating_sub(log_final_poly_len);
    // epsilon_i for i in [3..(k-1)], where k is number of rounds
    let epsilon_i_min = (0..num_fri_layers)
        .map(|_| extension_field_bits - log2((folding_factor - 1.0) * (lde_domain_size + 1.0)))
        .fold(f64::INFINITY, |a, b| if b < a { b } else { a });
    epsilons_bits_neg.push(epsilon_i_min);

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

    // we apply Theorem 2 in https://eprint.iacr.org/2024/1553, which is based on Theorem 8 in
    // https://eprint.iacr.org/2022/1216.pdf and Theorem 5 in https://eprint.iacr.org/2021/582
    // Note that the range of m needs to be restricted in order to ensure that eta, the slackness
    // factor to the distance bound, is greater than 0.
    // Determining the range of m is the responsibility of the calling function.
    let mut epsilons_bits_neg = Vec::new();
    // ALI related soundness error.
    epsilons_bits_neg.push(-log2(l) - log2(constraint_batching) + extension_field_bits);

    // DEEP related soundness error.
    epsilons_bits_neg.push(
        -log2(
            l * l
                * (max_deg * (trace_domain_size + num_openings - 1.0) + (trace_domain_size - 1.0)),
        ) + extension_field_bits,
    );

    // FRI commit-phase (i.e., pre-query) soundness error.
    // This considers only the first term given in eq. 7 in https://eprint.iacr.org/2022/1216.pdf,
    // i.e. (m + 0.5)^7 * n^2 * (N - 1) / (3 * q * rho^1.5) as all other terms are negligible in
    // comparison. N is the number of batched polynomials.
    epsilons_bits_neg.push(
        extension_field_bits
            - log2(
                (pow(m_f + 0.5, 7.0) / (3.0 * pow(rho, 1.5)))
                    * lde_domain_size
                    * lde_domain_size
                    * deep_batching,
            ),
    );

    // epsilon_i for i in [3..(k-1)], where k is number of rounds, are also negligible

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
