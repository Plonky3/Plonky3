//! Composite STARK soundness: AIR composition + DEEP-ALI + LDT, evaluated
//! once per proximity regime (UDR and best-`m` LDR). Generic over the LDT
//! via plain function arguments — `fri.rs`, `whir.rs`, and downstream
//! drop-in LDTs all compose with the same orchestrator.
//!
//! Extra protocol-specific error terms (lookup arguments, custom DEEP
//! variants, batched openings, …) are passed through `extras: &[ErrorBits]`
//! at every entry point and folded into the same round-by-round min.
//! Pass `&[]` when only the baseline AIR + DEEP + LDT terms apply.

use alloc::vec::Vec;

use crate::assumption::SecurityAssumption;
use crate::error::ErrorBits;
use crate::ldt::LowDegreeTest;
use crate::proximity::{list_size_ldr_m, list_size_udr};
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
fn batching_term(
    assumption: SecurityAssumption,
    shape: &InstanceShape,
    log_blowup: usize,
    ldr_m: Option<usize>,
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
        ErrorBits::from_log2(bits.max(0.0)),
    ))
}

/// Labeled term list for one proximity regime: ALI, DEEP, LDT, the optional
/// batch-combination term, `extras`, and the commitment-collision cap.
/// Attained security is the min over these (see
/// [`RegimeReport::security_bits`]), matching [`proven_security_regime`] plus
/// the batching term.
fn regime_report(
    regime: Regime,
    air: &StarkAirParams,
    shape: &InstanceShape,
    list_size: f64,
    ldt_error: ErrorBits,
    batch: Option<SecurityTerm>,
    extras: &[SecurityTerm],
) -> RegimeReport {
    let ali = air::composition_error(air.num_constraints, list_size, shape.modulus_bits);
    let deep = deep::deep_ali_error(air, shape, list_size);
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
pub fn proven_security_report<L: LowDegreeTest>(
    ldt: &L,
    air: &StarkAirParams,
    shape: &InstanceShape,
    extras: &[SecurityTerm],
) -> SecurityReport {
    let log_blowup = ldt.log_blowup();

    let udr_ldt = ldt.proven_error_udr(air, shape);
    let udr = regime_report(
        Regime::UniqueDecoding,
        air,
        shape,
        list_size_udr(),
        udr_ldt,
        batching_term(SecurityAssumption::UniqueDecoding, shape, log_blowup, None),
        extras,
    );

    let ldr = ldt.best_ldr(air, shape).map(|(m, ldr_ldt)| {
        let list_size = list_size_ldr_m(log_blowup, m);
        regime_report(
            Regime::ListDecoding { m },
            air,
            shape,
            list_size,
            ldr_ldt,
            batching_term(SecurityAssumption::JohnsonBound, shape, log_blowup, Some(m)),
            extras,
        )
    });

    SecurityReport { udr, ldr }
}

#[cfg(test)]
mod tests {
    use super::*;

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

        let report = proven_security_report(&regime, &air, &shape, &[]);

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
        let report = proven_security_report(&benchmark_regime(), &air(), &shape(), &[]);
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

        let no_batch = proven_security_report(&regime, &air, &base, &[]);
        let with_batch = proven_security_report(&regime, &air, &batched, &[]);

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

        let report = proven_security_report(&regime, &air, &shape, &[]);
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
}
