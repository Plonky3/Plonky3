//! STARK proof security level computation.
//!
//! Thin adapter over [`p3_security`]: bundles FRI parameters, AIR shape,
//! and crypto parameters into [`StarkSecurityParams`], builds the
//! corresponding regime / instance shape, and delegates the math.

use core::cmp::max;

use p3_air::Air;
use p3_air::symbolic::{AirLayout, SymbolicAirBuilder};
use p3_field::{BasedVectorSpace, ExtensionField, Field};
use p3_security::fri::FriRegime;
// Re-exported (rather than merely imported) so that declaring
// [`StarkSecurityParams::grinding`] does not oblige a caller to add a direct
// dependency on `p3-security`.
pub use p3_security::grinding::GrindingSites;
use p3_security::shape::{InstanceShape, StarkAirParams as P3AirShape};
use p3_security::stark::{conjectured_security_report, proven_security_report};
use p3_util::{log2_ceil_usize, log2_floor_usize};

/// What the polynomial commitment scheme commits beyond what the AIR itself
/// determines.
///
/// This is a property of the proof configuration rather than of the AIR, so
/// [`StarkSecurityParams::from_air`] cannot read it off the AIR and takes it
/// here instead. The degree of the challenge field over the base field — the
/// width in base-field columns of one committed quotient chunk, and of the zk
/// randomizing codeword — is not part of this shape: `from_air` already has
/// `EF` in scope and derives it from there, so there is no second value that
/// could disagree with the type parameter.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct OpeningShape {
    /// Whether zero-knowledge is enabled, which adds one randomizing codeword
    /// to the batch and one to the quotient's degree.
    pub is_zk: bool,
}

impl OpeningShape {
    /// Non-zk configuration.
    pub const fn new() -> Self {
        Self { is_zk: false }
    }

    /// The same, with zero-knowledge enabled.
    #[must_use]
    pub const fn with_zk(mut self) -> Self {
        self.is_zk = true;
        self
    }
}

/// Number of `(column, opening point)` pairs the polynomial commitment scheme
/// random-linear-combines into its single low-degree-test instance.
///
/// `p3_fri::TwoAdicFriPcs::open` charges one power of the batching challenge
/// per column per opening point, so the count is a sum over committed
/// matrices of `width × (points opened at)`:
///
/// - the main trace, at `zeta`, and at `zeta·g` as well unless the AIR
///   declares no next-row access;
/// - the preprocessed trace, on the same rule;
/// - every quotient chunk, at `zeta`, each `challenge_dimension` columns wide;
/// - under zk, the randomizing codeword, likewise `challenge_dimension` wide.
///
/// This is the `k` of the batched-openings proximity term, whose error grows
/// with `log2(k − 1)`. Under [`p3_fri::HidingFriPcs`] each committed matrix
/// carries additional random columns, so this is then a lower bound.
pub const fn num_batched_openings(
    main_width: usize,
    main_next: bool,
    preprocessed_width: usize,
    preprocessed_next: bool,
    num_quotient_chunks: usize,
    challenge_dimension: usize,
    is_zk: bool,
) -> usize {
    let main = if main_next { 2 } else { 1 } * main_width;
    let preprocessed = if preprocessed_next { 2 } else { 1 } * preprocessed_width;
    let quotient = num_quotient_chunks * challenge_dimension;
    let random = if is_zk { challenge_dimension } else { 0 };
    main + preprocessed + quotient + random
}

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
    /// Number of `(column, opening point)` pairs the polynomial commitment
    /// scheme random-linear-combines into the single low-degree-test instance:
    /// trace columns, preprocessed columns, quotient chunks, and the zk
    /// randomizing codeword.
    ///
    /// `1` means "nothing is batched", which switches the batched-openings
    /// proximity term off entirely — a real uni-STARK batches on the order of
    /// twice its trace width, so passing `1` for one overstates security by
    /// several bits. [`StarkSecurityParams::from_air`] derives the true count;
    /// [`num_batched_openings`] is the formula it uses.
    pub num_batched_functions: usize,
    /// Grinding sited outside the low-degree test, in bits per site.
    ///
    /// The FRI query- and commit-phase grinds live in the fields above,
    /// because the low-degree test folds them into its own terms; everything
    /// else the protocol grinds before goes here. Defaults to
    /// [`GrindingSites::NONE`].
    ///
    /// Only declare a site the prover actually grinds and the verifier
    /// actually checks — these bits are added to the reported level verbatim,
    /// so a site claimed here but not enforced in the protocol overstates
    /// security. For a uni-STARK over `p3-fri` the two enforced sites are
    /// `p3_fri::FriParameters::grinding_sites` (the batch-combination
    /// challenge) and [`crate::StarkGenericConfig::ood_proof_of_work_bits`]
    /// (the out-of-domain point), so a faithful value is:
    ///
    /// ```ignore
    /// GrindingSites {
    ///     out_of_domain: config.ood_proof_of_work_bits(),
    ///     ..fri_params.grinding_sites()
    /// }
    /// ```
    ///
    /// A `p3-batch-stark` proof enforces a third,
    /// [`crate::StarkGenericConfig::lookup_proof_of_work_bits`], and so adds
    /// `lookup_challenge: config.lookup_proof_of_work_bits()`. A uni-STARK has
    /// no lookups and must leave that site at `0`.
    pub grinding: GrindingSites,
}

impl StarkSecurityParams {
    /// Build security parameters explicitly from the FRI shape and the AIR shape.
    ///
    /// Use [`from_air`](Self::from_air) when an AIR is available — it derives
    /// `num_constraints`, `air_max_constraint_degree` and
    /// `num_batched_functions` from the AIR and its commitment layout.
    ///
    /// `num_batched_functions` is required rather than defaulted because there
    /// is no safe default: `1` silently removes the batched-openings term from
    /// the report, and every real instance batches more than that. Callers
    /// without an AIR to inspect compute it with [`num_batched_openings`].
    ///
    /// # Panics (debug only)
    ///
    /// If `air_max_constraint_degree` exceeds `blowup + 1`, the prover cannot
    /// fit the quotient in the LDE and the resulting parameters describe an
    /// unbuildable configuration. This does not account for zk: under zk the
    /// prover's real bound is one tighter (`blowup`, not `blowup + 1`), but
    /// `StarkSecurityParams` carries no `is_zk` flag to check that here.
    pub const fn new(
        fri: FriRegime,
        num_modulus_bits: usize,
        collision_resistance: usize,
        num_constraints: usize,
        air_max_constraint_degree: usize,
        max_combo: usize,
        num_batched_functions: usize,
    ) -> Self {
        debug_assert!(
            air_max_constraint_degree <= (1usize << fri.log_blowup) + 1,
            "AIR max constraint degree exceeds blowup+1; the prover cannot commit a quotient"
        );
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
            num_batched_functions,
            grinding: GrindingSites::NONE,
        }
    }

    /// Declare the grinding sited outside the low-degree test.
    ///
    /// See [`Self::grinding`] for how to derive a faithful value from the
    /// runtime config, and why an unfaithful one overstates security.
    #[must_use]
    pub const fn with_grinding(mut self, grinding: GrindingSites) -> Self {
        self.grinding = grinding;
        self
    }

    /// Build security parameters by inspecting the AIR's symbolic constraints to derive
    /// `num_constraints` and `air_max_constraint_degree`. The caller supplies `max_combo`
    /// (typically `2` for a uni-STARK that uses `local`/`next`, `1` if no transition).
    ///
    /// `layout` must reflect any permutation/lookup columns: a base-only layout (e.g.
    /// `AirLayout::from_air`, which fills only the `BaseAir` widths) leaves the
    /// permutation fields at `0`, so permutation-argument constraints are not counted
    /// and security is overstated.
    ///
    /// `openings` supplies the one fact neither the AIR nor `EF` carries: whether the prover
    /// commits zk's randomizing codeword. The other input `num_batched_functions` needs — how
    /// wide a committed quotient chunk is — is `EF`'s own degree over `F`, already in scope here;
    /// see [`num_batched_openings`] for the formula the two combine into.
    ///
    /// `grinding` is required rather than defaulted for the same reason: a
    /// silent [`GrindingSites::NONE`] is safe in direction, since it only
    /// understates, but it makes every grinding knob inert in the report. A
    /// caller who raises `p3_fri::FriParameters::batch_proof_of_work_bits` and
    /// sees the reported level not move has hit exactly that. Pass
    /// `fri_params.grinding_sites()`, extended with the sites the surrounding
    /// protocol enforces — see [`Self::grinding`] for the full recipe.
    // The list is the inputs a security level is a function of: the low-degree
    // test, the AIR and its layout, the two crypto parameters, the opening
    // shape, and where the protocol grinds. Grouping any of them behind another
    // name would only move the same values one level down.
    #[allow(clippy::too_many_arguments)]
    pub fn from_air<F, EF, A>(
        fri: FriRegime,
        air: &A,
        layout: AirLayout,
        num_modulus_bits: usize,
        collision_resistance: usize,
        max_combo: usize,
        openings: OpeningShape,
        grinding: GrindingSites,
    ) -> Self
    where
        F: Field,
        EF: ExtensionField<F>,
        A: Air<SymbolicAirBuilder<F, EF>>,
    {
        let main_next = !air.main_next_row_columns().is_empty();
        let preprocessed_next = !air.preprocessed_next_row_columns().is_empty();
        debug_assert!(
            max_combo >= 1 + (main_next || preprocessed_next) as usize,
            "max_combo ({max_combo}) must cover every rotation the AIR reads: \
             main_next={main_next}, preprocessed_next={preprocessed_next}"
        );

        let shape = P3AirShape::from_air::<F, EF, A>(air, layout, max_combo);
        let challenge_dimension = <EF as BasedVectorSpace<F>>::DIMENSION;

        // `get_log_num_quotient_chunks`'s formula for the chunk count, then the same `<< is_zk`
        // doubling `prove_with_preprocessed` applies on top of it (`num_quotient_chunks = 1 <<
        // (log_num_quotient_chunks + is_zk)`, `uni-stark/src/prover.rs`) to get what is actually
        // committed and therefore what the PCS batches.
        let constraint_degree = (shape.max_constraint_degree + openings.is_zk as usize).max(2);
        let log_num_quotient_chunks = log2_ceil_usize(constraint_degree - 1);
        let num_quotient_chunks = 1usize << (log_num_quotient_chunks + openings.is_zk as usize);

        let num_batched_functions = num_batched_openings(
            layout.main_width,
            main_next,
            layout.preprocessed_width,
            preprocessed_next,
            num_quotient_chunks,
            challenge_dimension,
            openings.is_zk,
        );

        Self::new(
            fri,
            num_modulus_bits,
            collision_resistance,
            shape.num_constraints,
            shape.max_constraint_degree,
            max_combo,
            num_batched_functions,
        )
        .with_grinding(grinding)
    }

    /// The low-degree test owns the query- and commit-phase grinding sites,
    /// so they are read straight from `self` and never re-applied by the
    /// composite.
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
/// Like [`ProvenSecurity`], this models the batched-openings term
/// (`num_batched_functions`), charged at the conjectured regime's own list
/// size — see [`conjectured_security_report`]'s "The batched-openings round".
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
        // `max_log_arity: 1` is the minimum arity `p3-fri` accepts (folding
        // factor 2). The commit-phase error is non-increasing in arity (see
        // `commit_phase_error_udr`), so evaluating it at the smallest legal
        // arity keeps this a true upper bound over every real FRI config,
        // whatever arity it actually folds at.
        let regime = FriRegime {
            log_blowup,
            num_queries,
            log_final_poly_len: 0,
            max_log_arity: 1,
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
    /// # How this differs from the proven composite
    ///
    /// `params` is threaded whole and every field is consumed, but two are
    /// worth spelling out:
    ///
    /// - `num_batched_functions` is charged, at the conjectured regime's list
    ///   size rather than the proven regime's Johnson-bound one. The result is
    ///   therefore looser than [`ProvenSecurity`]'s batching term, not absent:
    ///   an instance that random-linear-combines several committed codewords
    ///   pays for that round in both regimes.
    /// - `fri_commit_proof_of_work_bits` is credited, but only to the
    ///   commit-phase term ([`p3_security::fri::conjectured_commit_phase_error`]),
    ///   which is one of several the composite takes a minimum over. Grinding
    ///   there raises the reported level only while that term is the binding
    ///   one.
    pub fn compute_from_params(params: &StarkSecurityParams, degree_bits: usize) -> Self {
        let report = conjectured_security_report(
            &params.fri_regime(),
            &params.air_shape(),
            &params.instance_shape(degree_bits),
            &[],
            &params.grinding,
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
        let regime = params.fri_regime();
        let air = params.air_shape();
        let shape = params.instance_shape(degree_bits);

        let report = proven_security_report(&regime, &air, &shape, &[], &params.grinding);

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
    use alloc::vec::Vec;

    use p3_air::symbolic::SymbolicVariable;
    use p3_air::{AirBuilder, BaseAir, WindowAccess};
    use p3_baby_bear::BabyBear;
    use p3_field::extension::BinomialExtensionField;

    use super::*;

    /// The degree-4 extension the `from_air`-derivation tests use as `EF`, so
    /// `challenge_dimension` is a real property of the type parameter rather than a
    /// number picked to match by coincidence.
    type Ext = BinomialExtensionField<BabyBear, 4>;

    /// A trace-only AIR of a chosen width, with `num_constraints` degree-1
    /// constraints. `reads_next_row` drives [`BaseAir::main_next_row_columns`],
    /// which is what decides whether the trace is opened at one point or two.
    struct MockAir {
        width: usize,
        num_constraints: usize,
        reads_next_row: bool,
    }

    impl BaseAir<BabyBear> for MockAir {
        fn width(&self) -> usize {
            self.width
        }

        fn main_next_row_columns(&self) -> Vec<usize> {
            if self.reads_next_row {
                (0..self.width).collect()
            } else {
                Vec::new()
            }
        }
    }

    impl<EF: ExtensionField<BabyBear>> Air<SymbolicAirBuilder<BabyBear, EF>> for MockAir {
        fn eval(&self, builder: &mut SymbolicAirBuilder<BabyBear, EF>) {
            for i in 0..self.num_constraints {
                let var: SymbolicVariable<BabyBear> =
                    builder.main().current(i % self.width).unwrap();
                builder.assert_zero(var);
            }
        }
    }

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
            grinding: GrindingSites::NONE,
        }
    }

    /// The counting formula sums `width × (points opened at)` over every
    /// committed matrix, which is what `TwoAdicFriPcs::open` charges powers of
    /// the batching challenge for.
    #[test]
    fn num_batched_openings_counts_every_committed_column_and_point() {
        // Width 100 read at both rows, two quotient chunks over a degree-4
        // extension, no preprocessed columns: 2·100 + 2·4.
        assert_eq!(num_batched_openings(100, true, 0, false, 2, 4, false), 208);
        // Without next-row access the trace is opened at `zeta` alone.
        assert_eq!(num_batched_openings(100, false, 0, false, 2, 4, false), 108);
        // Preprocessed columns follow the same rule.
        assert_eq!(num_batched_openings(100, true, 6, true, 2, 4, false), 220);
        // Zero-knowledge adds one randomizing codeword.
        assert_eq!(num_batched_openings(100, true, 0, false, 2, 4, true), 212);
    }

    /// `from_air` derives the count instead of leaving it at the `1` that
    /// switches the batched-openings term off.
    #[test]
    fn from_air_derives_the_batched_openings_count() {
        let regime = benchmark_high_arity_params(124).fri_regime();
        let air = MockAir {
            width: 8,
            num_constraints: 3,
            reads_next_row: true,
        };
        let layout = AirLayout::from_air(&air);

        let params = StarkSecurityParams::from_air::<BabyBear, Ext, _>(
            regime,
            &air,
            layout,
            124,
            128,
            2,
            OpeningShape::new(),
            GrindingSites::NONE,
        );

        // Degree-1 constraints give a single quotient chunk of 4 columns, and
        // the trace is opened at two points: 2·8 + 1·4.
        assert_eq!(params.num_batched_functions, 20);
        assert!(
            params.num_batched_functions > 1,
            "a derived count must switch the batching term on"
        );
    }

    /// Under zero-knowledge, `prove_with_preprocessed` doubles the derived chunk count
    /// (`num_quotient_chunks = 1 << (log_num_quotient_chunks + is_zk)`, `uni-stark/src/prover.rs`)
    /// on top of the `+1` already folded into `constraint_degree`. `from_air` must apply that same
    /// doubling, not stop at the `+1`.
    #[test]
    fn from_air_derives_the_doubled_zk_quotient_chunk_count() {
        let regime = benchmark_high_arity_params(124).fri_regime();
        let air = MockAir {
            width: 8,
            num_constraints: 3,
            reads_next_row: true,
        };
        let layout = AirLayout::from_air(&air);

        let params = StarkSecurityParams::from_air::<BabyBear, Ext, _>(
            regime,
            &air,
            layout,
            124,
            128,
            2,
            OpeningShape::new().with_zk(),
            GrindingSites::NONE,
        );

        // Two quotient chunks (the zk doubling), each 4 columns, plus one zk randomizing
        // codeword of 4 columns, plus the trace opened at both points: 2·8 + 2·4 + 4.
        assert_eq!(params.num_batched_functions, 28);
    }

    /// An AIR that never reads the next row is opened at one point, so it
    /// batches half as many trace functions.
    #[test]
    fn from_air_counts_one_opening_point_without_next_row_access() {
        let regime = benchmark_high_arity_params(124).fri_regime();
        let layout_of = |air: &MockAir| AirLayout::from_air(air);

        let with_next = MockAir {
            width: 8,
            num_constraints: 3,
            reads_next_row: true,
        };
        let without_next = MockAir {
            width: 8,
            num_constraints: 3,
            reads_next_row: false,
        };

        let a = StarkSecurityParams::from_air::<BabyBear, Ext, _>(
            regime,
            &with_next,
            layout_of(&with_next),
            124,
            128,
            2,
            OpeningShape::new(),
            GrindingSites::NONE,
        );
        let b = StarkSecurityParams::from_air::<BabyBear, Ext, _>(
            regime,
            &without_next,
            layout_of(&without_next),
            124,
            128,
            1,
            OpeningShape::new(),
            GrindingSites::NONE,
        );

        assert_eq!(a.num_batched_functions, 20);
        assert_eq!(b.num_batched_functions, 12);
    }

    /// A grinding site the protocol enforces must move the reported level.
    ///
    /// This is the regression guard for the shape of bug where the knob exists,
    /// the prover pays it and the verifier checks it, but the report never sees
    /// it: raising `p3_fri::FriParameters::batch_proof_of_work_bits` then
    /// changes nothing at all, at any value.
    #[test]
    fn from_air_threads_the_grinding_sites_into_the_report() {
        let regime = benchmark_high_arity_params(124).fri_regime();
        // Wide enough that the batched-openings round is the binding one, which
        // is the round the batch site protects.
        let air = MockAir {
            width: 1000,
            num_constraints: 3,
            reads_next_row: true,
        };
        let layout = AirLayout::from_air(&air);
        let degree_bits = 16;

        let build = |grinding| {
            StarkSecurityParams::from_air::<BabyBear, Ext, _>(
                regime,
                &air,
                layout,
                124,
                128,
                2,
                OpeningShape::new(),
                grinding,
            )
        };

        let ungrounded = build(GrindingSites::NONE);
        let ground = build(GrindingSites {
            batch_combination: 16,
            ..GrindingSites::NONE
        });

        assert_eq!(ungrounded.grinding, GrindingSites::NONE);
        assert_eq!(ground.grinding.batch_combination, 16);

        let before = ConjecturedSecurity::compute_from_params(&ungrounded, degree_bits);
        let after = ConjecturedSecurity::compute_from_params(&ground, degree_bits);
        assert!(
            after.security_bits > before.security_bits,
            "batch grinding bought nothing in the report: {} -> {}",
            before.security_bits,
            after.security_bits
        );

        let before = ProvenSecurity::compute_from_proof(degree_bits, &ungrounded);
        let after = ProvenSecurity::compute_from_proof(degree_bits, &ground);
        assert!(
            after.security_bits() > before.security_bits(),
            "batch grinding bought nothing in the proven report: {} -> {}",
            before.security_bits(),
            after.security_bits()
        );
    }

    /// The derived count lowers the reported proven level relative to the `1`
    /// that used to be assumed: the batching round is real and now charged.
    #[test]
    fn deriving_the_count_lowers_the_reported_level() {
        let optimistic = StarkSecurityParams {
            num_batched_functions: 1,
            ..benchmark_high_arity_params(124)
        };
        let honest = StarkSecurityParams {
            num_batched_functions: 208,
            ..benchmark_high_arity_params(124)
        };

        let a = ProvenSecurity::compute_from_proof(20, &optimistic).security_bits();
        let b = ProvenSecurity::compute_from_proof(20, &honest).security_bits();

        assert!(
            b < a,
            "charging the batching round must not raise the level: {a} -> {b}"
        );
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
    /// This entry point pins `max_log_arity = 1` (the minimum arity `p3-fri`
    /// accepts) and a single-row trace, so the folding round is evaluated at
    /// `n = 2^log_blowup` — it binds only in the two rows where the query
    /// phase would otherwise clear the collision cap (`log_blowup` 2 and 8),
    /// which is exactly the overstatement that omitting the round produced.
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
