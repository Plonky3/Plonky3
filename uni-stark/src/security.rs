//! STARK proof security level computation.
//!
//! Thin adapter over [`p3_security`]: bundles FRI parameters, AIR shape,
//! and crypto parameters into [`StarkSecurityParams`], builds the
//! corresponding regime / instance shape, and delegates the math.

use core::cmp::{max, min};

use p3_air::Air;
use p3_air::symbolic::{AirLayout, SymbolicAirBuilder};
use p3_field::{ExtensionField, Field};
use p3_security::fri::{FriRegime, conjectured_error};
use p3_security::shape::{InstanceShape, StarkAirParams as P3AirShape};
use p3_security::stark::proven_security_report;
use p3_util::log2_floor_usize;

/// Parameters required to compute STARK proof security level.
///
/// The FRI-related fields mirror [`FriRegime`]'s public fields (this crate is PCS-generic,
/// so it takes them as a standalone regime rather than depending on `p3-fri`); the
/// AIR-shape fields (`num_constraints`, `air_max_constraint_degree`, `max_combo`) describe the
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
    /// Number of committed codewords random-linear-combined into the single
    /// FRI instance (trace columns, quotient chunks, …). Defaults to `1` (no
    /// batching term); set it to the actual count to account for the
    /// batched-openings proximity error. Leaving it at `1` is optimistic.
    pub num_batched_functions: usize,
}

impl StarkSecurityParams {
    /// Build security parameters explicitly from the FRI shape and the AIR shape.
    ///
    /// Use [`from_air`](Self::from_air) when an AIR is available — it derives
    /// `num_constraints` and `air_max_constraint_degree` from symbolic evaluation.
    pub const fn new(
        fri: FriRegime,
        num_modulus_bits: usize,
        collision_resistance: usize,
        num_constraints: usize,
        air_max_constraint_degree: usize,
        max_combo: usize,
    ) -> Self {
        Self {
            fri_log_blowup: fri.log_blowup,
            fri_log_final_poly_len: fri.log_final_poly_len,
            fri_max_log_arity: fri.max_log_arity,
            fri_num_queries: fri.num_queries,
            fri_commit_proof_of_work_bits: fri.commit_pow_bits,
            fri_query_proof_of_work_bits: fri.query_pow_bits,
            num_modulus_bits,
            collision_resistance,
            num_constraints,
            air_max_constraint_degree,
            max_combo,
            num_batched_functions: 1,
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
    pub fn from_air<F, EF, A>(
        fri: FriRegime,
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
        let shape = P3AirShape::from_air::<F, EF, A>(air, layout, max_combo);
        Self::new(
            fri,
            num_modulus_bits,
            collision_resistance,
            shape.num_constraints,
            shape.max_constraint_degree,
            max_combo,
        )
    }

    const fn fri_regime(&self) -> FriRegime {
        FriRegime {
            log_blowup: self.fri_log_blowup,
            num_queries: self.fri_num_queries,
            log_final_poly_len: self.fri_log_final_poly_len,
            max_log_arity: self.fri_max_log_arity,
            commit_pow_bits: self.fri_commit_proof_of_work_bits,
            query_pow_bits: self.fri_query_proof_of_work_bits,
        }
    }

    const fn air_shape(&self) -> P3AirShape {
        P3AirShape {
            num_constraints: self.num_constraints,
            max_constraint_degree: self.air_max_constraint_degree,
            max_combo: self.max_combo,
        }
    }

    const fn instance_shape(&self, log_trace_length: usize) -> InstanceShape {
        InstanceShape {
            log_trace_length,
            modulus_bits: self.num_modulus_bits,
            collision_resistance: self.collision_resistance,
            num_batched_functions: self.num_batched_functions,
        }
    }
}

/// Conjectured security level (in bits) using the "random words" regime
/// of [2025/2010](https://eprint.iacr.org/2025/2010) §1.5.
///
/// The cited paper recommends proven bounds for deployment; users staying with
/// conjectured bounds should remain above the cutoff.
///
/// Unlike [`ProvenSecurity`], this does not model the batched-openings term
/// (`num_batched_functions`): the conjectured path is optimistic relative to
/// the proven one for instances that random-linear-combine more than one
/// committed codeword.
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
        let regime = FriRegime {
            log_blowup,
            num_queries,
            log_final_poly_len: 0,
            max_log_arity: 0,
            commit_pow_bits: 0,
            query_pow_bits: query_proof_of_work_bits,
        };
        let shape = InstanceShape {
            log_trace_length: 0,
            modulus_bits: num_modulus_bits,
            collision_resistance,
            num_batched_functions: 1,
        };
        let fri_bits = conjectured_error(&regime, &shape).bits() as usize;
        let bits = min(min(fri_bits, collision_resistance), num_modulus_bits);
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
    ///
    /// `trace_length` is floored to a power of two via [`log2_floor_usize`]. Plonky3
    /// commits only power-of-two-sized traces, so this should always be exact; a
    /// non-power-of-two input would silently analyze a smaller domain and report an
    /// optimistic bound. Use [`Self::compute_from_proof`] to pass `degree_bits` directly
    /// when it is available.
    pub fn compute(params: &StarkSecurityParams, trace_length: usize) -> Self {
        if trace_length == 0 {
            return Self {
                unique_decoding_bits: 0,
                list_decoding_bits: 0,
            };
        }
        debug_assert!(
            trace_length.is_power_of_two(),
            "trace_length {trace_length} is not a power of two; committed traces always are"
        );
        Self::compute_from_proof(log2_floor_usize(trace_length), params)
    }

    /// Compute proven security using a parameter bundle and the proof's degree bits.
    ///
    /// `degree_bits` already reflects the committed-polynomial size (post-zk padding,
    /// when applicable), so the trace-domain size used for security analysis is `2^degree_bits`.
    pub fn compute_from_proof(degree_bits: usize, params: &StarkSecurityParams) -> Self {
        if params.fri_log_blowup == 0 || params.num_modulus_bits == 0 {
            return Self {
                unique_decoding_bits: 0,
                list_decoding_bits: 0,
            };
        }
        debug_assert!(
            params.air_max_constraint_degree <= (1usize << params.fri_log_blowup) + 1,
            "AIR max constraint degree {} exceeds blowup+1 ({}); the prover cannot commit a quotient",
            params.air_max_constraint_degree,
            (1usize << params.fri_log_blowup) + 1
        );

        let regime = params.fri_regime();
        let air = params.air_shape();
        let shape = params.instance_shape(degree_bits);

        let report = proven_security_report(&regime, &air, &shape, &[]);

        Self {
            unique_decoding_bits: report.udr.security_bits() as usize,
            list_decoding_bits: report
                .ldr
                .as_ref()
                .map_or(0, |r| r.security_bits() as usize),
        }
    }
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
            num_batched_functions: 1,
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

    #[test]
    fn more_batched_functions_decreases_or_holds_security() {
        // Over a small field the batched-openings term is active.
        let mut params = benchmark_high_arity_params(64);
        params.num_batched_functions = 1;
        let p1 = ProvenSecurity::compute(&params, 1 << 20);
        params.num_batched_functions = 1 << 20;
        let p_batched = ProvenSecurity::compute(&params, 1 << 20);
        assert!(p_batched.security_bits() <= p1.security_bits());
    }

    // Regression vector pinning the proven-security output for a fixed configuration:
    // log_blowup=1, num_queries=100, query_pow=16, commit_pow=0, max_log_arity=3,
    // |F|=252 bits, trace 2^20, num_constraints=1, max_deg=2, max_combo=2.
    // num_batched_functions defaults to 1, so no batching term applies.
    #[test]
    fn proven_security_regression_benchmark_high_arity() {
        let params = benchmark_high_arity_params(252);
        let p = ProvenSecurity::compute(&params, 1 << 20);
        assert_eq!(p.unique_decoding_bits, 57);
        assert_eq!(p.list_decoding_bits, 65);
    }
}
