//! Composite STARK soundness: AIR composition + DEEP-ALI + LDT, evaluated
//! once per proximity regime (UDR and best-`m` LDR). Generic over the LDT
//! via plain function arguments — `fri.rs`, `whir.rs`, and downstream
//! drop-in LDTs all compose with the same orchestrator.
//!
//! The conjectured counterpart ([`conjectured_security_report`]) composes the
//! same sources in the single random-words regime, where the list size is 1.
//!
//! Extra protocol-specific error terms (lookup arguments, custom DEEP
//! variants, batched openings, …) are passed through `extras: &[ErrorBits]`
//! at every entry point and folded into the same round-by-round min.
//! Pass `&[]` when only the baseline AIR + DEEP + LDT terms apply.

use alloc::vec::Vec;

use crate::assumption::SecurityAssumption;
use crate::error::ErrorBits;
use crate::grinding::{GrindingSites, boost};
use crate::ldt::LowDegreeTest;
use crate::proximity::{list_size_conjectured, list_size_ldr_m, list_size_udr};
use crate::report::{
    ALI_LABEL, BATCH_LABEL, COLLISION_LABEL, DEEP_LABEL, LDT_LABEL, Regime, RegimeReport,
    SecurityReport, SecurityTerm,
};
use crate::shape::{InstanceShape, StarkAirParams};
use crate::{air, deep};

/// Bits attained in a single proximity regime, given the LDT-only error,
/// the regime's list size, and any extra protocol-specific error terms.
///
/// `ldt_error` is the round-by-round min over the LDT's commit and query
/// phases (see e.g. [`crate::fri::proven_error_udr`]). `list_size` is the
/// regime's L⁺. `extras` lets the caller fold in additional independent
/// error sources (lookup, custom DEEP, …) without dropping the orchestrator.
///
/// The result is capped at `shape.collision_resistance`: a collision in the
/// commitment hash forges the proof regardless of the algebraic bound, so
/// real security is `min(algebraic soundness, hash collision resistance)`.
pub fn proven_security_regime(
    air: &StarkAirParams,
    shape: &InstanceShape,
    list_size: f64,
    ldt_error: ErrorBits,
    extras: &[ErrorBits],
) -> ErrorBits {
    let ali = air::composition_error(air.num_constraints, list_size, shape.modulus_bits);
    let deep = deep::deep_ali_error(air, shape, list_size);
    let mut all: Vec<ErrorBits> = Vec::with_capacity(3 + extras.len());
    all.push(ali);
    all.push(deep);
    all.push(ldt_error);
    all.extend_from_slice(extras);
    let algebraic = ErrorBits::min(&all);
    ErrorBits::from_log2(algebraic.bits().min(shape.collision_resistance as f64))
}

/// Composite STARK bits in the UDR regime, with optional `extras`.
pub fn proven_security_udr(
    air: &StarkAirParams,
    shape: &InstanceShape,
    ldt_error: ErrorBits,
    extras: &[ErrorBits],
) -> ErrorBits {
    proven_security_regime(air, shape, list_size_udr(), ldt_error, extras)
}

/// Composite STARK bits in the LDR regime with explicit `m`, with
/// optional `extras`.
pub fn proven_security_ldr_m(
    air: &StarkAirParams,
    shape: &InstanceShape,
    log_blowup: usize,
    m: usize,
    ldt_error: ErrorBits,
    extras: &[ErrorBits],
) -> ErrorBits {
    proven_security_regime(
        air,
        shape,
        list_size_ldr_m(log_blowup, m),
        ldt_error,
        extras,
    )
}

/// Best of UDR and a precomputed best-`m` LDR, with optional `extras`
/// applied to both regimes. Each regime is an independent valid lower
/// bound, so the max is itself a valid (and tighter) bound on
/// round-by-round soundness.
pub fn proven_security(
    air: &StarkAirParams,
    shape: &InstanceShape,
    log_blowup: usize,
    udr_ldt_error: ErrorBits,
    ldr_best_m: usize,
    ldr_ldt_error: ErrorBits,
    extras: &[ErrorBits],
) -> ErrorBits {
    let udr = proven_security_udr(air, shape, udr_ldt_error, extras);
    let ldr = proven_security_ldr_m(air, shape, log_blowup, ldr_best_m, ldr_ldt_error, extras);
    ErrorBits::from_log2(udr.bits().max(ldr.bits()))
}

/// Proximity-gap error of the initial random linear combination that batches
/// `shape.num_batched_functions` committed codewords into a single LDT
/// instance, evaluated in `assumption`'s regime (UD in the unique-decoding
/// regime, JB in the list-decoding regime). Returns `None` when nothing is
/// batched (fewer than two functions).
///
/// `ldr_m` is the proximity parameter the surrounding [`Regime::ListDecoding`]
/// actually decodes at (e.g. FRI's `best_m`); `None` for the unique-decoding
/// regime, where it does not apply. The batch RLC must be δ-close at the
/// same radius the rest of the regime's terms (ALI/DEEP/LDT) are evaluated
/// at, so the Johnson-bound branch is computed at `ldr_m` rather than the
/// fixed `m = 10` WHIR safety choice.
///
/// `pow_bits` is [`GrindingSites::batch_combination`]: the grinding sited
/// immediately before the batching challenge, which boosts this round and no
/// other.
fn batching_term(
    assumption: SecurityAssumption,
    shape: &InstanceShape,
    log_blowup: usize,
    ldr_m: Option<usize>,
    pow_bits: usize,
) -> Option<SecurityTerm> {
    let num_functions = shape.num_batched_functions;
    if num_functions < 2 {
        return None;
    }
    let bits = match (assumption, ldr_m) {
        (SecurityAssumption::JohnsonBound, Some(m)) => SecurityAssumption::prox_gaps_error_jb_at_m(
            shape.log_trace_length,
            log_blowup,
            shape.modulus_bits,
            num_functions,
            m,
        ),
        _ => assumption.prox_gaps_error(
            shape.log_trace_length,
            log_blowup,
            shape.modulus_bits,
            num_functions,
        ),
    };
    Some(SecurityTerm::new(
        BATCH_LABEL,
        boost(ErrorBits::from_log2(bits.max(0.0)), pow_bits),
    ))
}

/// Labeled term list for one proximity regime: ALI, DEEP, LDT, the optional
/// batch-combination term, `extras`, and the commitment-collision cap.
/// Attained security is the min over these (see
/// [`RegimeReport::security_bits`]), matching [`proven_security_regime`] plus
/// the batching term.
///
/// `grinding.out_of_domain` boosts the DEEP term and
/// `grinding.batch_combination` the batch term (applied by the caller, which
/// builds `batch`); the low-degree test's own sites are already folded into
/// `ldt_error` by the [`LowDegreeTest`] impl.
fn regime_report(
    regime: Regime,
    air: &StarkAirParams,
    shape: &InstanceShape,
    list_size: f64,
    ldt_error: ErrorBits,
    batch: Option<SecurityTerm>,
    extras: &[SecurityTerm],
    grinding: &GrindingSites,
) -> RegimeReport {
    let ali = air::composition_error(air.num_constraints, list_size, shape.modulus_bits);
    let deep = boost(
        deep::deep_ali_error(air, shape, list_size),
        grinding.out_of_domain,
    );
    let mut terms = Vec::with_capacity(5 + extras.len());
    terms.push(SecurityTerm::new(ALI_LABEL, ali));
    terms.push(SecurityTerm::new(DEEP_LABEL, deep));
    terms.push(SecurityTerm::new(LDT_LABEL, ldt_error));
    terms.extend(batch);
    terms.extend_from_slice(extras);
    terms.push(SecurityTerm::new(
        COLLISION_LABEL,
        ErrorBits::from_log2(shape.collision_resistance as f64),
    ));
    RegimeReport::new(regime, terms)
}

/// Composite proven-security report, generic over the low-degree test.
///
/// Evaluates the UDR and best-`m` LDR regimes via `ldt`, composes each with
/// the ALI, DEEP, `extras`, and commitment-collision terms, and returns the
/// full labeled breakdown. `extras` fold protocol-specific error sources
/// (lookup arguments, custom DEEP variants, batched openings, …) into every
/// regime; pass `&[]` for the baseline AIR + DEEP + LDT composite.
///
/// [`SecurityReport::security_bits`] reproduces [`proven_security`]; the report
/// additionally exposes which term binds in each regime.
///
/// `grinding` sites the protocol's proof-of-work per error source; pass
/// [`GrindingSites::NONE`] when the low-degree test carries all of it.
pub fn proven_security_report<L: LowDegreeTest>(
    ldt: &L,
    air: &StarkAirParams,
    shape: &InstanceShape,
    extras: &[SecurityTerm],
    grinding: &GrindingSites,
) -> SecurityReport {
    let log_blowup = ldt.log_blowup();

    let udr_ldt = ldt.proven_error_udr(air, shape);
    let udr = regime_report(
        Regime::UniqueDecoding,
        air,
        shape,
        list_size_udr(),
        udr_ldt,
        batching_term(
            SecurityAssumption::UniqueDecoding,
            shape,
            log_blowup,
            None,
            grinding.batch_combination,
        ),
        extras,
        grinding,
    );

    let ldr = ldt.best_ldr(air, shape).map(|(m, ldr_ldt)| {
        let list_size = list_size_ldr_m(log_blowup, m);
        regime_report(
            Regime::ListDecoding { m },
            air,
            shape,
            list_size,
            ldr_ldt,
            batching_term(
                SecurityAssumption::JohnsonBound,
                shape,
                log_blowup,
                Some(m),
                grinding.batch_combination,
            ),
            extras,
            grinding,
        )
    });

    SecurityReport { udr, ldr }
}

/// Composite conjectured bits: the min over the AIR-composition, DEEP-ALI,
/// low-degree-test, and `extras` terms, capped at the commitment-collision
/// resistance. Scalar mirror of [`conjectured_security_report`], standing to
/// it exactly as [`proven_security_regime`] stands to the report path.
///
/// The low-degree test's terms are taken from [`LowDegreeTest::conjectured_terms`]
/// rather than passed in. Unlike the proven scalar path — where the caller has
/// already resolved a per-regime error via [`crate::fri::proven_error_udr`] or
/// [`crate::fri::best_ldr_m`] — the conjectured regime has no such search, so an
/// `ErrorBits` parameter here would silently accept
/// [`crate::fri::conjectured_error`] alone and drop the commit-phase round,
/// overstating security for a large LDE domain over a small field. Taking the
/// test itself makes that unrepresentable.
///
/// The ALI and DEEP terms are evaluated at [`list_size_conjectured`] — see
/// [`conjectured_security_report`] for why the proven path's L⁺ multiplier is
/// absent here.
///
/// `grinding.out_of_domain` boosts the DEEP term, exactly as in
/// [`conjectured_security_report`]; the low-degree test's own sites are
/// already folded into `ldt.conjectured_terms`'s output by the
/// [`LowDegreeTest`] impl.
pub fn conjectured_security<L: LowDegreeTest>(
    ldt: &L,
    air: &StarkAirParams,
    shape: &InstanceShape,
    extras: &[ErrorBits],
    grinding: &GrindingSites,
) -> ErrorBits {
    let list_size = list_size_conjectured();
    let ali = air::composition_error(air.num_constraints, list_size, shape.modulus_bits);
    let deep = boost(
        deep::deep_ali_error(air, shape, list_size),
        grinding.out_of_domain,
    );
    let ldt_terms = ldt.conjectured_terms(shape);
    let mut all: Vec<ErrorBits> = Vec::with_capacity(2 + ldt_terms.len() + extras.len());
    all.push(ali);
    all.push(deep);
    all.extend(ldt_terms.iter().map(|t| t.bits));
    all.extend_from_slice(extras);
    let algebraic = ErrorBits::min(&all);
    ErrorBits::from_log2(algebraic.bits().min(shape.collision_resistance as f64))
}

/// Composite conjectured-security report, generic over the low-degree test.
///
/// Composes the LDT's conjectured terms ([`LowDegreeTest::conjectured_terms`],
/// which for FRI splits the query phase from the commit-phase folding rounds)
/// with the ALI, DEEP-ALI, `extras`, and commitment-collision terms and
/// returns the labeled breakdown. Attained security is the min over the terms,
/// exactly as in [`proven_security_report`], so the binding term stays
/// inspectable — which is the point: for an AIR with lookups the LogUp
/// fingerprint error grows linearly in the trace length and can bind well
/// below the query-phase term, an overstatement an LDT-only conjectured number
/// cannot express.
///
/// # Why one regime, and why no L⁺
///
/// The conjectured regime is a single proximity regime, so the result is one
/// [`RegimeReport`] rather than a [`SecurityReport`]: the latter's job is to
/// maximize over the two independent proven regimes (UDR and best-`m` LDR),
/// and there is nothing to maximize over here.
///
/// Within it, the random-words heuristic of
/// [2025/2010](https://eprint.iacr.org/2025/2010) §1.5 conjectures correlated
/// agreement up to list-decoding capacity at list size 1
/// ([`list_size_conjectured`]). ALI is then `ε ≤ num_constraints / |F|` and
/// DEEP-ALI is `ε ≤ (max_deg·(k + max_combo − 1) + (k − 1)) / |F|`, neither
/// carrying the `L⁺` factor the proven path's Johnson-bound list size forces
/// ([2024/1553](https://eprint.iacr.org/2024/1553) Theorem 2). Both drop out
/// of [`air::composition_error`] and [`deep::deep_ali_error`] at `list_size =
/// 1`, since `log2(1) = 0`.
///
/// # Not modeled
///
/// The batched-openings random-linear-combination term
/// ([`proven_security_report`]'s `batch-combination`) has no accepted
/// conjectured analogue — the random-words heuristic bounds the distance
/// distribution, not the proximity gap of the batching RLC — so it is omitted
/// rather than guessed. A caller with a bound for it passes it via `extras`.
pub fn conjectured_security_report<L: LowDegreeTest>(
    ldt: &L,
    air: &StarkAirParams,
    shape: &InstanceShape,
    extras: &[SecurityTerm],
    grinding: &GrindingSites,
) -> RegimeReport {
    let list_size = list_size_conjectured();
    let ali = air::composition_error(air.num_constraints, list_size, shape.modulus_bits);
    let deep = boost(
        deep::deep_ali_error(air, shape, list_size),
        grinding.out_of_domain,
    );

    let ldt_terms = ldt.conjectured_terms(shape);
    let mut terms = Vec::with_capacity(3 + ldt_terms.len() + extras.len());
    terms.push(SecurityTerm::new(ALI_LABEL, ali));
    terms.push(SecurityTerm::new(DEEP_LABEL, deep));
    terms.extend(ldt_terms);
    terms.extend_from_slice(extras);
    terms.push(SecurityTerm::new(
        COLLISION_LABEL,
        ErrorBits::from_log2(shape.collision_resistance as f64),
    ));
    RegimeReport::new(Regime::Conjectured, terms)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::report::LDT_QUERY_LABEL;

    fn shape() -> InstanceShape {
        InstanceShape {
            log_trace_length: 20,
            modulus_bits: 252,
            collision_resistance: 128,
            num_batched_functions: 1,
        }
    }

    fn air() -> StarkAirParams {
        StarkAirParams {
            num_constraints: 1,
            max_constraint_degree: 2,
            max_combo: 2,
        }
    }

    /// Extras tighten (never loosen) the regime's bound, and a tight enough
    /// extra dominates ALI/DEEP/LDT.
    #[test]
    fn extras_tighten_proven_security_regime() {
        let air = air();
        let shape = shape();
        let ldt = ErrorBits::from_log2(80.0);

        let baseline = proven_security_regime(&air, &shape, 1.0, ldt, &[]);
        let with_loose =
            proven_security_regime(&air, &shape, 1.0, ldt, &[ErrorBits::from_log2(200.0)]);
        let with_tight =
            proven_security_regime(&air, &shape, 1.0, ldt, &[ErrorBits::from_log2(40.0)]);

        // A loose extra (200 bits) sits above every other term — bound unchanged.
        assert!((baseline.bits() - with_loose.bits()).abs() < 1e-12);
        // A tight extra (40 bits) becomes the binding term.
        assert!((with_tight.bits() - 40.0).abs() < 1e-12);
        // Monotone: extras can only tighten, never loosen.
        assert!(with_tight.bits() <= baseline.bits());
    }

    /// A regime report's attained bits equal the scalar `proven_security_regime`
    /// for the same list size, LDT error, and extras.
    #[test]
    fn regime_report_matches_proven_security_regime() {
        let air = air();
        let shape = shape();
        let ldt = ErrorBits::from_log2(80.0);
        let list_size = 1.0;

        let extra = ErrorBits::from_log2(40.0);
        let expected = proven_security_regime(&air, &shape, list_size, ldt, &[extra]);
        let report = regime_report(
            Regime::UniqueDecoding,
            &air,
            &shape,
            list_size,
            ldt,
            None,
            &[SecurityTerm::new("extra", extra)],
            &GrindingSites::NONE,
        );

        assert!((report.security_bits() - expected.bits()).abs() < 1e-12);
        // The tight extra (40 bits) is the binding term.
        assert_eq!(report.binding().label, "extra");
    }

    /// The report path reproduces the scalar composite for the FRI benchmark
    /// vector and reports the low-degree test as the binding term.
    #[test]
    fn proven_security_report_matches_scalar_composite() {
        use crate::fri::{FriRegime, best_ldr_m, proven_error_udr};

        let regime = FriRegime {
            log_blowup: 1,
            num_queries: 100,
            log_final_poly_len: 0,
            max_log_arity: 3,
            commit_pow_bits: 0,
            query_pow_bits: 16,
        };
        let air = air();
        let shape = shape();

        let report = proven_security_report(&regime, &air, &shape, &[], &GrindingSites::NONE);

        // Same per-regime and combined numbers as ProvenSecurity / proven_security.
        assert_eq!(report.udr.security_bits().floor() as usize, 57);
        let ldr = report
            .ldr
            .as_ref()
            .expect("benchmark has a valid LDR regime");
        assert_eq!(ldr.security_bits().floor() as usize, 65);
        assert_eq!(report.security_bits().floor() as usize, 65);

        // The LDR regime wins and the low-degree test binds it.
        let (regime_kind, binding) = report.binding();
        assert!(matches!(regime_kind, Regime::ListDecoding { .. }));
        assert_eq!(binding.label, LDT_LABEL);

        // Cross-check against the untyped composite.
        let udr_ldt = proven_error_udr(&regime, &air, &shape);
        let (best_m, ldr_ldt) = best_ldr_m(&regime, &air, &shape).unwrap();
        let scalar = proven_security(
            &air,
            &shape,
            regime.log_blowup,
            udr_ldt,
            best_m,
            ldr_ldt,
            &[],
        );
        assert_eq!(
            report.security_bits().floor() as usize,
            scalar.bits().floor() as usize
        );
    }

    /// `FriRegime`'s `LowDegreeTest` methods delegate to the free functions.
    #[test]
    fn fri_regime_ldt_impl_delegates() {
        use crate::fri::{FriRegime, conjectured_error, proven_error_udr};
        use crate::ldt::LowDegreeTest;

        let regime = FriRegime {
            log_blowup: 1,
            num_queries: 100,
            log_final_poly_len: 0,
            max_log_arity: 3,
            commit_pow_bits: 0,
            query_pow_bits: 16,
        };
        let air = air();
        let shape = shape();

        assert_eq!(LowDegreeTest::log_blowup(&regime), regime.log_blowup);
        assert_eq!(
            LowDegreeTest::proven_error_udr(&regime, &air, &shape).bits(),
            proven_error_udr(&regime, &air, &shape).bits()
        );
        assert_eq!(
            LowDegreeTest::conjectured_error(&regime, &shape).bits(),
            conjectured_error(&regime, &shape).bits()
        );
    }

    fn benchmark_regime() -> crate::fri::FriRegime {
        crate::fri::FriRegime {
            log_blowup: 1,
            num_queries: 100,
            log_final_poly_len: 0,
            max_log_arity: 3,
            commit_pow_bits: 0,
            query_pow_bits: 16,
        }
    }

    /// A single committed function is not batched, so no batch-combination
    /// term is emitted in either regime.
    #[test]
    fn no_batch_term_for_single_function() {
        let report = proven_security_report(
            &benchmark_regime(),
            &air(),
            &shape(),
            &[],
            &GrindingSites::NONE,
        );
        assert!(report.udr.terms().iter().all(|t| t.label != BATCH_LABEL));
        if let Some(ldr) = &report.ldr {
            assert!(ldr.terms().iter().all(|t| t.label != BATCH_LABEL));
        }
    }

    /// Batching many functions over a small field only tightens the bound and,
    /// once large enough, becomes the binding term.
    #[test]
    fn batching_lowers_security_when_binding() {
        let regime = benchmark_regime();
        let air = air();
        let base = InstanceShape {
            log_trace_length: 20,
            modulus_bits: 64,
            collision_resistance: 128,
            num_batched_functions: 1,
        };
        let batched = InstanceShape {
            num_batched_functions: 1 << 20,
            ..base
        };

        let no_batch = proven_security_report(&regime, &air, &base, &[], &GrindingSites::NONE);
        let with_batch = proven_security_report(&regime, &air, &batched, &[], &GrindingSites::NONE);

        // Batching is an extra independent error source: it can only tighten.
        assert!(with_batch.security_bits() <= no_batch.security_bits());
        // With 2^20 batched functions over a 64-bit field, the batch term binds.
        let (_, binding) = with_batch.binding();
        assert_eq!(binding.label, BATCH_LABEL);
    }

    /// The LDR batch term is evaluated at the same `m` the surrounding
    /// `ListDecoding` regime reports (`best_m`), not the fixed `m = 10`
    /// WHIR safety choice — at the benchmark shape `best_m` is far from 10,
    /// so this pins the two diverging.
    #[test]
    fn ldr_batch_term_uses_regime_m_not_fixed_ten() {
        let regime = benchmark_regime();
        let air = air();
        let shape = InstanceShape {
            num_batched_functions: 2,
            ..shape()
        };

        let report = proven_security_report(&regime, &air, &shape, &[], &GrindingSites::NONE);
        let ldr = report
            .ldr
            .as_ref()
            .expect("benchmark has a valid LDR regime");
        let Regime::ListDecoding { m } = ldr.regime else {
            panic!("expected a list-decoding regime");
        };
        assert_ne!(m, 10, "test only pins the m != 10 path if best_m != 10");

        let batch_term = ldr
            .terms()
            .iter()
            .find(|t| t.label == BATCH_LABEL)
            .expect("batching two functions emits a batch-combination term");

        let expected_bits = SecurityAssumption::prox_gaps_error_jb_at_m(
            shape.log_trace_length,
            regime.log_blowup,
            shape.modulus_bits,
            shape.num_batched_functions,
            m,
        )
        .max(0.0);
        assert!((batch_term.bits.bits() - expected_bits).abs() < 1e-9);

        // The fixed m = 10 WHIR default would report a tighter (larger)
        // batch error here, since (m + 1/2)^5 grows with m.
        let fixed_m_bits = SecurityAssumption::JohnsonBound
            .prox_gaps_error(
                shape.log_trace_length,
                regime.log_blowup,
                shape.modulus_bits,
                shape.num_batched_functions,
            )
            .max(0.0);
        assert!(batch_term.bits.bits() < fixed_m_bits);
    }

    /// Grinding before the batching challenge is credited to the batch term,
    /// bit for bit, in both proven regimes — and to no other term. The
    /// "no other term" half is the point: the boost is applied where the
    /// batching term is built, so wiring it to the wrong round would show up
    /// as a neighbouring term moving instead.
    #[test]
    fn batch_grinding_lifts_only_the_batch_term() {
        let regime = benchmark_regime();
        let air = air();
        // A small field keeps every term in range of the others, so a
        // misdirected boost would be visible rather than clamped away.
        let shape = InstanceShape {
            modulus_bits: 100,
            num_batched_functions: 1 << 10,
            ..shape()
        };
        let ground = GrindingSites {
            batch_combination: 12,
            ..GrindingSites::NONE
        };

        let b0 = proven_security_report(&regime, &air, &shape, &[], &GrindingSites::NONE);
        let b12 = proven_security_report(&regime, &air, &shape, &[], &ground);

        for (before, after) in
            core::iter::once((&b0.udr, &b12.udr)).chain(b0.ldr.iter().zip(b12.ldr.iter()))
        {
            for (t0, t12) in before.terms().iter().zip(after.terms()) {
                assert_eq!(t0.label, t12.label, "term order changed");
                let expected = if t0.label == BATCH_LABEL {
                    t0.bits.bits() + 12.0
                } else {
                    t0.bits.bits()
                };
                assert!(
                    (t12.bits.bits() - expected).abs() < 1e-12,
                    "{} moved to {} (expected {expected})",
                    t0.label,
                    t12.bits.bits()
                );
            }
        }
    }

    /// With fewer than two functions there is no batching round to protect, so
    /// grinding before a challenge the protocol never samples must buy nothing
    /// rather than being credited to whatever term happens to bind.
    #[test]
    fn batch_grinding_buys_nothing_when_nothing_is_batched() {
        let regime = benchmark_regime();
        let air = air();
        let shape = InstanceShape {
            modulus_bits: 100,
            num_batched_functions: 1,
            ..shape()
        };

        let b0 = proven_security_report(&regime, &air, &shape, &[], &GrindingSites::NONE);
        let b32 = proven_security_report(
            &regime,
            &air,
            &shape,
            &[],
            &GrindingSites {
                batch_combination: 32,
                ..GrindingSites::NONE
            },
        );

        assert!((b32.security_bits() - b0.security_bits()).abs() < 1e-12);
        assert!(b32.udr.terms().iter().all(|t| t.label != BATCH_LABEL));
    }

    /// The conjectured report's attained bits equal the scalar
    /// `conjectured_security` for the same LDT, shape, and extras, and it is
    /// tagged as the conjectured regime.
    ///
    /// Checked with and without `extras`: a binding extra would mask a
    /// divergence between the two paths' LDT term sets, which is exactly what
    /// went unnoticed while the scalar path took a caller-supplied
    /// `ErrorBits` and the report path composed `conjectured_terms`.
    #[test]
    fn conjectured_report_matches_scalar_composite() {
        let regime = benchmark_regime();
        let air = air();
        let shape = shape();
        let extra = ErrorBits::from_log2(40.0);

        // Without extras, the LDT terms are what the two paths must agree on.
        let bare_report =
            conjectured_security_report(&regime, &air, &shape, &[], &GrindingSites::NONE);
        let bare_scalar = conjectured_security(&regime, &air, &shape, &[], &GrindingSites::NONE);
        assert!((bare_report.security_bits() - bare_scalar.bits()).abs() < 1e-12);

        let report = conjectured_security_report(
            &regime,
            &air,
            &shape,
            &[SecurityTerm::new("extra", extra)],
            &GrindingSites::NONE,
        );
        let scalar = conjectured_security(&regime, &air, &shape, &[extra], &GrindingSites::NONE);

        assert_eq!(report.regime, Regime::Conjectured);
        assert!((report.security_bits() - scalar.bits()).abs() < 1e-12);
        assert_eq!(report.binding().label, "extra");

        // And at a nonzero grinding site — the axis the two paths could only
        // diverge on before `conjectured_security` modeled `GrindingSites`
        // itself, since it had no way to express the DEEP-term boost that
        // `conjectured_security_report` applies.
        let ground = GrindingSites {
            out_of_domain: 24,
            ..GrindingSites::NONE
        };
        let ground_report = conjectured_security_report(&regime, &air, &shape, &[], &ground);
        let ground_scalar = conjectured_security(&regime, &air, &shape, &[], &ground);
        assert!((ground_report.security_bits() - ground_scalar.bits()).abs() < 1e-12);
    }

    /// Conjectured mode decodes at list size 1, so ALI and DEEP carry no
    /// `L⁺` multiplier: both match the proven UDR terms (also `L⁺ = 1`) and
    /// are strictly looser than the proven LDR terms, whose Johnson-bound
    /// list size costs `log2(L⁺)` bits.
    #[test]
    fn conjectured_ali_and_deep_carry_no_list_size() {
        let regime = benchmark_regime();
        let air = air();
        let shape = shape();

        let conjectured =
            conjectured_security_report(&regime, &air, &shape, &[], &GrindingSites::NONE);
        let proven = proven_security_report(&regime, &air, &shape, &[], &GrindingSites::NONE);
        let ldr = proven
            .ldr
            .as_ref()
            .expect("benchmark has a valid LDR regime");

        for label in [ALI_LABEL, DEEP_LABEL] {
            let find = |r: &RegimeReport| {
                r.terms()
                    .iter()
                    .find(|t| t.label == label)
                    .expect("every regime carries the ALI and DEEP terms")
                    .bits
                    .bits()
            };
            assert!((find(&conjectured) - find(&proven.udr)).abs() < 1e-12);
            assert!(find(&conjectured) > find(ldr));
        }
    }

    /// The gap this composite closes: at a large trace the LogUp fingerprint
    /// error binds well below the LDT's conjectured query-phase term, so an
    /// LDT-only conjectured number overstates security.
    #[test]
    fn conjectured_logup_extra_binds_below_the_ldt_term() {
        use crate::logup::{LOGUP_LABEL, LogUpAir, security_term};

        // Queries chosen so the LDT term sits between the DEEP term above it
        // and the lookup term below it, isolating what binds.
        let regime = crate::fri::FriRegime {
            num_queries: 80,
            ..benchmark_regime()
        };
        let air = air();
        let shape = InstanceShape {
            log_trace_length: 28,
            modulus_bits: 128,
            collision_resistance: 128,
            num_batched_functions: 1,
        };
        let logup = LogUpAir {
            num_interactions: 64,
            max_message_width: 8,
        };

        let term = security_term(&logup, &shape, &GrindingSites::NONE).expect("has interactions");
        let ldt_only =
            conjectured_security_report(&regime, &air, &shape, &[], &GrindingSites::NONE);
        let with_logup =
            conjectured_security_report(&regime, &air, &shape, &[term], &GrindingSites::NONE);

        // Without the lookup term the low-degree test's query phase is what binds.
        assert_eq!(ldt_only.binding().label, LDT_QUERY_LABEL);
        // With it, the lookup term binds strictly lower — the overstatement.
        assert_eq!(with_logup.binding().label, LOGUP_LABEL);
        assert!(with_logup.security_bits() < ldt_only.security_bits());
    }

    /// More constraints can only tighten (never loosen) the conjectured
    /// bound, via the AIR-composition term.
    #[test]
    fn conjectured_more_constraints_is_not_more_security() {
        let regime = benchmark_regime();
        let shape = shape();
        let few = StarkAirParams {
            num_constraints: 1,
            ..air()
        };
        let many = StarkAirParams {
            num_constraints: 1 << 20,
            ..air()
        };

        let b_few = conjectured_security_report(&regime, &few, &shape, &[], &GrindingSites::NONE);
        let b_many = conjectured_security_report(&regime, &many, &shape, &[], &GrindingSites::NONE);
        assert!(b_many.security_bits() <= b_few.security_bits());
    }

    /// Grinding before the out-of-domain challenge can only loosen (never
    /// tighten) the DEEP term, so security is non-decreasing in it — and the
    /// neutral default reproduces the ungrounded report exactly.
    #[test]
    fn conjectured_more_grinding_is_not_less_security() {
        let regime = benchmark_regime();
        let air = air();
        // A small field puts DEEP in range of the other terms, so the grind
        // is observable rather than masked by the collision cap.
        let shape = InstanceShape {
            modulus_bits: 100,
            ..shape()
        };
        let ground = GrindingSites {
            out_of_domain: 24,
            ..GrindingSites::NONE
        };

        let b0 = conjectured_security_report(&regime, &air, &shape, &[], &GrindingSites::NONE);
        let b24 = conjectured_security_report(&regime, &air, &shape, &[], &ground);
        assert!(b24.security_bits() >= b0.security_bits());

        let default_sites =
            conjectured_security_report(&regime, &air, &shape, &[], &GrindingSites::default());
        assert!((default_sites.security_bits() - b0.security_bits()).abs() < 1e-12);
    }

    /// The same monotonicity holds on the proven path, in both regimes.
    #[test]
    fn proven_more_grinding_is_not_less_security() {
        let regime = benchmark_regime();
        let air = air();
        let shape = InstanceShape {
            modulus_bits: 100,
            ..shape()
        };
        let ground = GrindingSites {
            out_of_domain: 24,
            ..GrindingSites::NONE
        };

        let b0 = proven_security_report(&regime, &air, &shape, &[], &GrindingSites::NONE);
        let b24 = proven_security_report(&regime, &air, &shape, &[], &ground);
        assert!(b24.udr.security_bits() >= b0.udr.security_bits());
        assert!(b24.security_bits() >= b0.security_bits());
    }
}
