//! STARK proof security level computation.
//!
//! Thin adapter over [`p3_security`]: bundles FRI parameters, AIR shape,
//! and crypto parameters into [`StarkSecurityParams`], builds the
//! corresponding regime / instance shape, and delegates the math.

use core::cmp::max;

use p3_air::Air;
use p3_air::symbolic::{AirLayout, SymbolicAirBuilder};
use p3_field::{ExtensionField, Field};
use p3_security::fri::FriRegime;
use p3_security::grinding::GrindingSites;
use p3_security::shape::{InstanceShape, StarkAirParams as P3AirShape};
use p3_security::stark::{conjectured_security_report, proven_security_report};
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

    /// Where this configuration grinds, by error source. A uni-STARK grinds
    /// only inside FRI, so the DEEP and lookup-challenge sites stay at the
    /// neutral `0`.
    const fn grinding(&self) -> GrindingSites {
        GrindingSites {
            query_phase: self.fri_query_proof_of_work_bits,
            ldt_commit_phase: self.fri_commit_proof_of_work_bits,
            out_of_domain: 0,
            lookup_challenge: 0,
        }
    }

    /// The low-degree test owns the query- and commit-phase grinding sites,
    /// so they are carried here and never re-applied by the composite.
    const fn fri_regime(&self) -> FriRegime {
        let grinding = self.grinding();
        FriRegime {
            log_blowup: self.fri_log_blowup,
            num_queries: self.fri_num_queries,
            log_final_poly_len: self.fri_log_final_poly_len,
            max_log_arity: self.fri_max_log_arity,
            commit_pow_bits: grinding.ldt_commit_phase,
            query_pow_bits: grinding.query_phase,
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
    /// Conjectured security from FRI parameters alone, using the random-words
    /// formula ([2025/2010] §1.5). Requires `num_modulus_bits` (log2 of field
    /// size) for the η cutoff.
    ///
    /// This entry point takes no AIR shape and no trace length, so the
    /// AIR-composition and DEEP-ALI terms are evaluated on the smallest
    /// instance that carries them — one degree-1 constraint over a
    /// single-row trace — where both reduce to the field-size cap. A uni-STARK
    /// has no lookups, so no `extras` apply either, leaving the low-degree
    /// test and the commitment-collision cap as the only binding terms.
    ///
    /// The result is therefore a property of the parameter space, not of any
    /// particular instance: it answers "what can these FRI parameters attain
    /// at best", and an instance with a real AIR over a real trace attains at
    /// most this. Use [`Self::compute_from_params`] once the trace length is
    /// known — the DEEP-ALI term grows with it, so this bound is optimistic
    /// for every instance larger than a single row.
    pub fn compute_ldt_only(
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
        let air = P3AirShape {
            num_constraints: 1,
            max_constraint_degree: 1,
            max_combo: 1,
        };
        let shape = InstanceShape {
            log_trace_length: 0,
            modulus_bits: num_modulus_bits,
            collision_resistance,
            num_batched_functions: 1,
        };
        let report = conjectured_security_report(&regime, &air, &shape, &[], &GrindingSites::NONE);
        Self {
            security_bits: report.security_bits() as usize,
        }
    }

    /// Compute conjectured security from a parameter bundle and the proof's
    /// degree bits, composing the AIR-composition and DEEP-ALI terms at the
    /// instance's real shape rather than the degenerate one
    /// [`Self::compute_ldt_only`] uses.
    ///
    /// `degree_bits` already reflects the committed-polynomial size (post-zk
    /// padding, when applicable), so the trace domain is `2^degree_bits` —
    /// the same convention as [`ProvenSecurity::compute_from_proof`].
    ///
    /// # Two fields this does not consume
    ///
    /// `params` is threaded whole, but the conjectured composite reads less of
    /// it than the proven one does:
    ///
    /// - `num_batched_functions` is ignored. The batched-openings term has no
    ///   accepted conjectured analogue — the random-words heuristic bounds the
    ///   distance distribution, not the proximity gap of the batching RLC — so
    ///   [`conjectured_security_report`] omits it rather than guessing. An
    ///   instance that random-linear-combines several committed codewords is
    ///   graded optimistically here relative to [`ProvenSecurity`].
    /// - `fri_commit_proof_of_work_bits` is credited, but only to the
    ///   commit-phase term ([`p3_security::fri::conjectured_commit_phase_error`]),
    ///   which is one of several the composite takes a minimum over. Grinding
    ///   there raises the reported level only while that term is the binding
    ///   one.
    pub fn compute_from_params(params: &StarkSecurityParams, degree_bits: usize) -> Self {
        debug_assert!(
            params.air_max_constraint_degree <= (1usize << params.fri_log_blowup) + 1,
            "AIR max constraint degree {} exceeds blowup+1 ({}); the prover cannot commit a quotient",
            params.air_max_constraint_degree,
            (1usize << params.fri_log_blowup) + 1
        );

        let report = conjectured_security_report(
            &params.fri_regime(),
            &params.air_shape(),
            &params.instance_shape(degree_bits),
            &[],
            &params.grinding(),
        );
        Self {
            security_bits: report.security_bits() as usize,
        }
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

        let report = proven_security_report(&regime, &air, &shape, &[], &params.grinding());

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

    /// A 256-bit field puts every algebraic term — including the commit-phase
    /// folding round, which over a 128-bit field would bind here at 119 — well
    /// above the hash's 128-bit collision resistance, leaving the cap as the
    /// only thing that binds.
    #[test]
    fn conjectured_security_bounded_by_collision_resistance() {
        let s = ConjecturedSecurity::compute_ldt_only(8, 32, 0, 128, 256);
        assert_eq!(s.security_bits, 128);
    }

    #[test]
    fn conjectured_security_random_words_formula() {
        let s = ConjecturedSecurity::compute_ldt_only(4, 20, 8, 256, 128);
        assert!(s.security_bits > 0 && s.security_bits <= 256);
    }

    #[test]
    fn conjectured_security_log_blowup_zero_returns_zero_fri_bits() {
        let s = ConjecturedSecurity::compute_ldt_only(0, 100, 16, 128, 256);
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
        let c = ConjecturedSecurity::compute_ldt_only(8, 32, 8, 256, 252);
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

    /// Regression vector pinning [`ConjecturedSecurity::compute_ldt_only`]
    /// across the parameter space it is routed through. Entries are
    /// `(log_blowup, num_queries, query_pow, collision_resistance,
    /// num_modulus_bits, expected_bits)`.
    ///
    /// A uni-STARK has no lookups and batches nothing in the conjectured
    /// path, so the composite reduces to `min` over the query phase, the
    /// commit-phase folding round, `collision_resistance`, and
    /// `num_modulus_bits`.
    ///
    /// This entry point pins `max_log_arity = 0` and a single-row trace, so
    /// the folding round is evaluated at `n = 2^log_blowup` — it binds only in
    /// the two rows where the query phase would otherwise clear the collision
    /// cap (`log_blowup` 2 and 8), which is exactly the overstatement that
    /// omitting the round produced.
    #[test]
    fn conjectured_security_ldt_only_regression_vector() {
        const VECTOR: [(usize, usize, usize, usize, usize, usize); 8] = [
            (1, 100, 16, 128, 252, 114),
            (1, 100, 0, 128, 128, 97),
            (2, 64, 20, 128, 128, 125),
            (4, 20, 8, 256, 128, 86),
            (8, 32, 0, 128, 128, 119),
            (0, 100, 16, 128, 256, 16),
            (1, 84, 16, 100, 128, 97),
            (3, 27, 21, 128, 128, 100),
        ];

        for (log_blowup, num_queries, query_pow, collision, modulus, expected) in VECTOR {
            let s = ConjecturedSecurity::compute_ldt_only(
                log_blowup,
                num_queries,
                query_pow,
                collision,
                modulus,
            );
            assert_eq!(
                s.security_bits, expected,
                "conjectured security drifted at \
                 (log_blowup={log_blowup}, num_queries={num_queries}, query_pow={query_pow}, \
                 collision={collision}, modulus={modulus})"
            );
        }
    }

    /// More query grinding can only raise the conjectured bound, up to the
    /// collision-resistance cap.
    #[test]
    fn conjectured_more_query_grinding_is_not_less_security() {
        let s0 = ConjecturedSecurity::compute_ldt_only(1, 64, 0, 256, 256);
        let s16 = ConjecturedSecurity::compute_ldt_only(1, 64, 16, 256, 256);
        assert!(s16.security_bits >= s0.security_bits);
    }

    /// Regression vector pinning [`ConjecturedSecurity::compute_from_params`]
    /// across trace heights, over a field small enough that the shape-bearing
    /// DEEP-ALI term is in range of the query-phase term rather than masked by
    /// the collision cap.
    ///
    /// The FRI commit phase binds throughout, at
    /// `|F| − log2((folding − 1)·(n + 1))` with `n = 2^(degree_bits + blowup)`
    /// and `folding − 1 = 7` — so the level is `92.19 − degree_bits`, falling
    /// exactly one bit per degree bit. DEEP-ALI tracks about 2.2 bits above it
    /// (`|F| − log2(3·k + 1)`) and would bind at a lower folding arity.
    ///
    /// That slope is the whole point of this entry point:
    /// [`ConjecturedSecurity::compute_ldt_only`] reports a single constant
    /// across all five heights, because neither term is visible to it.
    #[test]
    fn conjectured_security_from_params_regression_vector() {
        const DEGREE_BITS: [usize; 5] = [10, 16, 20, 24, 28];
        const EXPECTED: [usize; 5] = [82, 76, 72, 68, 64];

        let params = benchmark_high_arity_params(96);
        let actual = DEGREE_BITS
            .map(|degree_bits| ConjecturedSecurity::compute_from_params(&params, degree_bits))
            .map(|s| s.security_bits);
        assert_eq!(actual, EXPECTED, "conjectured security drifted");
    }

    /// The shape-aware path can only report at or below the shape-free one:
    /// [`ConjecturedSecurity::compute_ldt_only`] evaluates the AIR-composition
    /// and DEEP-ALI terms on a single-row, one-constraint instance, which is
    /// the best case every real instance is measured against.
    #[test]
    fn conjectured_from_params_never_exceeds_ldt_only() {
        let params = benchmark_high_arity_params(96);
        let ldt_only = ConjecturedSecurity::compute_ldt_only(
            params.fri_log_blowup,
            params.fri_num_queries,
            params.fri_query_proof_of_work_bits,
            params.collision_resistance,
            params.num_modulus_bits,
        );

        for degree_bits in 1..=28 {
            let shaped = ConjecturedSecurity::compute_from_params(&params, degree_bits);
            assert!(
                shaped.security_bits <= ldt_only.security_bits,
                "shape-aware bound exceeded the shape-free one at degree_bits={degree_bits}"
            );
        }
    }

    /// The DEEP-ALI term grows with the trace domain, so a taller instance can
    /// never grade above a shorter one under the same parameters. This is the
    /// dependence the shape-free entry point cannot express.
    #[test]
    fn conjectured_from_params_is_monotone_in_trace_height() {
        let params = benchmark_high_arity_params(96);
        let mut previous = usize::MAX;
        for degree_bits in 1..=28 {
            let bits = ConjecturedSecurity::compute_from_params(&params, degree_bits).security_bits;
            assert!(bits <= previous, "level rose at degree_bits={degree_bits}");
            previous = bits;
        }
    }

    /// Proven security is never above conjectured at the same shape — the
    /// conjectured regime drops the list-size multiplier the proven one pays.
    #[test]
    fn proven_never_exceeds_conjectured_at_the_same_shape() {
        let params = benchmark_high_arity_params(252);
        for degree_bits in [8, 16, 20, 24] {
            let c = ConjecturedSecurity::compute_from_params(&params, degree_bits);
            let p = ProvenSecurity::compute_from_proof(degree_bits, &params);
            assert!(
                p.security_bits() <= c.security_bits,
                "proven exceeded conjectured at degree_bits={degree_bits}"
            );
        }
    }
}
