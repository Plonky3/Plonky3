//! Cross-checks [`p3_security::budget`]'s fixed-point rounds against this crate's own `f64`
//! composite (`fri`, `logup`, `deep`, `air`) on a synthetic grid, so the two representations of
//! the same conjectured-security math can never drift apart silently.
//!
//! See `budget`'s module doc for the folding-accounting and DEEP-quotient-batching design notes
//! this test exercises.

use p3_security::budget::report::{
    COLLISION_LABEL, COMPOSITION_LABEL, DEEP_COMPOSITION_LABEL, FOLDING_LABEL, LOOKUP_LABEL,
    OUT_OF_DOMAIN_LABEL, QUERY_LABEL,
};
use p3_security::budget::{
    AirShape, InstanceShape, LookupShape, ProtocolParams, SecurityReport, security_report,
};
use p3_security::fri::{FriRegime, commit_phase_error_udr, conjectured_error};
use p3_security::grinding::{GrindingSites, boost};
use p3_security::logup::{LogUpAir, security_term as logup_security_term};
use p3_security::shape::{InstanceShape as F64InstanceShape, StarkAirParams};
use p3_security::{air, deep, fixed};

const TIGHT_TOL: f64 = 1e-4;
const LOG_BLOWUP: u32 = 3;
const LOG_FOLDING_ARITY: u32 = 2;
const CAP_BITS: usize = 128;
const FIELD_BITS: u64 = fixed::from_bits(128) - 2; // shy of 128.0, as a real field's log2 would be

struct AirVector {
    num_composed_constraints: u32,
    max_constraint_degree: u32,
    num_deep_terms: u32,
    fractions_per_row: u32,
    max_message_width: u32,
}

const SMALL: AirVector = AirVector {
    num_composed_constraints: 40,
    max_constraint_degree: 3,
    num_deep_terms: 20,
    fractions_per_row: 4,
    max_message_width: 4,
};

const LARGE: AirVector = AirVector {
    num_composed_constraints: 587,
    max_constraint_degree: 9,
    num_deep_terms: 770,
    fractions_per_row: 244,
    max_message_width: 18,
};

const PARAM_ROWS: &[(u32, u32, u32, u32)] = &[(27, 17, 12, 4), (100, 0, 0, 0), (150, 31, 31, 31)];

/// Out-of-domain points referenced per column: `local` and `next` rotations.
const OOD_MAX_COMBO: u32 = 2;

fn to_f64(fixed_bits: u64) -> f64 {
    fixed_bits as f64 / fixed::ONE as f64
}

const fn f64_instance(log_max_height: u32) -> F64InstanceShape {
    F64InstanceShape {
        log_trace_length: log_max_height as usize,
        modulus_bits: CAP_BITS,
        collision_resistance: CAP_BITS,
        num_batched_functions: 1,
    }
}

fn term_bits(report: &SecurityReport, label: &str) -> f64 {
    to_f64(
        report
            .terms()
            .iter()
            .find(|t| t.label == label)
            .expect("label present")
            .bits,
    )
}

/// Caps an uncapped `f64` reference value the same way [`security_report`] caps its fixed-point
/// terms, so the two sides are compared on equal footing near the ceiling.
const fn cap(bits: f64) -> f64 {
    if bits < CAP_BITS as f64 {
        bits
    } else {
        CAP_BITS as f64
    }
}

#[test]
fn every_round_direction_and_tightness() {
    for air_vector in [&SMALL, &LARGE] {
        for &(num_queries, query_pow_bits, deep_pow_bits, folding_pow_bits) in PARAM_ROWS {
            let params = ProtocolParams {
                log_blowup: LOG_BLOWUP,
                log_folding_arity: LOG_FOLDING_ARITY,
                num_queries,
                query_pow_bits,
                deep_pow_bits,
                folding_pow_bits,
                lookup_pow_bits: 0,
            };
            let air_shape = AirShape {
                num_composed_constraints: air_vector.num_composed_constraints,
                max_constraint_degree: air_vector.max_constraint_degree,
                max_combo: OOD_MAX_COMBO,
                num_deep_terms: Some(air_vector.num_deep_terms),
                lookup: LookupShape {
                    fractions_per_row: air_vector.fractions_per_row,
                    max_message_width: air_vector.max_message_width,
                },
            };

            for log_max_height in 6..=29u32 {
                let instance = InstanceShape {
                    log_max_height,
                    field_bits: FIELD_BITS,
                    collision_resistance: CAP_BITS as u32,
                };
                let report = security_report(&params, &instance, &air_shape);

                check_query(&report, num_queries, query_pow_bits);
                check_lookup(&report, air_vector, log_max_height);
                check_composition(&report, air_vector);
                check_ood(&report, air_vector, log_max_height);
                check_folding(&report, log_max_height, folding_pow_bits);
                check_deep_composition(&report, air_vector, deep_pow_bits);
            }
        }
    }
}

fn check_query(report: &SecurityReport, num_queries: u32, query_pow_bits: u32) {
    let fixed_bits = term_bits(report, QUERY_LABEL);
    let regime = FriRegime {
        log_blowup: LOG_BLOWUP as usize,
        num_queries: num_queries as usize,
        log_final_poly_len: 0,
        max_log_arity: LOG_FOLDING_ARITY as usize,
        commit_pow_bits: 0,
        query_pow_bits: query_pow_bits as usize,
    };
    let p3_bits = cap(conjectured_error(&regime, &f64_instance(20)).bits());

    // `bits_per_query` rounds its per-query rate down by less than 2 ulps (see its doc test's own
    // margin), and the fixed-point total multiplies that single per-query rounding by
    // `num_queries` before capping — the same tolerance bounds both directions.
    let tolerance = (num_queries as f64).mul_add(2.0 / fixed::ONE as f64, TIGHT_TOL);
    assert!(
        fixed_bits <= p3_bits + tolerance,
        "query: {fixed_bits} > {p3_bits} + {tolerance}"
    );
    assert!(
        p3_bits - fixed_bits < tolerance,
        "query: gap {} >= tolerance {tolerance}, fixed drifted conservative",
        p3_bits - fixed_bits
    );
}

fn check_lookup(report: &SecurityReport, air_vector: &AirVector, log_max_height: u32) {
    let fixed_bits = term_bits(report, LOOKUP_LABEL);
    let logup_air = LogUpAir {
        num_interactions: air_vector.fractions_per_row as usize,
        max_message_width: air_vector.max_message_width as usize,
    };
    let p3_bits = cap(logup_security_term(
        &logup_air,
        &f64_instance(log_max_height),
        &GrindingSites::NONE,
    )
    .expect("has interactions")
    .bits
    .bits());

    assert!(
        fixed_bits <= p3_bits + TIGHT_TOL,
        "lookup: {fixed_bits} > {p3_bits}"
    );
    assert!(
        p3_bits - fixed_bits < TIGHT_TOL,
        "lookup: gap {} >= tolerance, fixed drifted conservative",
        p3_bits - fixed_bits
    );
}

fn check_composition(report: &SecurityReport, air_vector: &AirVector) {
    let fixed_bits = term_bits(report, COMPOSITION_LABEL);
    let p3_bits =
        cap(
            air::composition_error(air_vector.num_composed_constraints as usize, 1.0, CAP_BITS)
                .bits(),
        );

    assert!(
        fixed_bits <= p3_bits + TIGHT_TOL,
        "composition: {fixed_bits} > {p3_bits}"
    );
    assert!(
        p3_bits - fixed_bits < TIGHT_TOL,
        "composition: gap {} >= tolerance, fixed drifted conservative",
        p3_bits - fixed_bits
    );
}

fn check_ood(report: &SecurityReport, air_vector: &AirVector, log_max_height: u32) {
    let fixed_bits = term_bits(report, OUT_OF_DOMAIN_LABEL);
    let stark_air = StarkAirParams {
        num_constraints: air_vector.num_composed_constraints as usize,
        max_constraint_degree: air_vector.max_constraint_degree as usize,
        max_combo: OOD_MAX_COMBO as usize,
    };
    let p3_bits = cap(deep::deep_ali_error(&stark_air, &f64_instance(log_max_height), 1.0).bits());

    assert!(
        fixed_bits <= p3_bits + TIGHT_TOL,
        "ood: {fixed_bits} > {p3_bits} at h={log_max_height}"
    );
    assert!(
        p3_bits - fixed_bits < TIGHT_TOL,
        "ood: gap {} >= tolerance at h={log_max_height}, fixed drifted conservative",
        p3_bits - fixed_bits
    );
}

fn check_folding(report: &SecurityReport, log_max_height: u32, folding_pow_bits: u32) {
    let fixed_bits = term_bits(report, FOLDING_LABEL);
    let regime = FriRegime {
        log_blowup: LOG_BLOWUP as usize,
        num_queries: 1,
        log_final_poly_len: 0,
        max_log_arity: LOG_FOLDING_ARITY as usize,
        commit_pow_bits: folding_pow_bits as usize,
        query_pow_bits: 0,
    };
    let p3_bits = cap(
        commit_phase_error_udr(&regime, &f64_instance(log_max_height))
            .expect("folds at every swept height")
            .bits(),
    );

    assert!(
        fixed_bits <= p3_bits + TIGHT_TOL,
        "folding: {fixed_bits} > {p3_bits}"
    );
    assert!(
        p3_bits - fixed_bits < 1.0 + TIGHT_TOL,
        "folding: gap {} >= 1 bit",
        p3_bits - fixed_bits
    );
}

fn check_deep_composition(report: &SecurityReport, air_vector: &AirVector, deep_pow_bits: u32) {
    let fixed_bits = term_bits(report, DEEP_COMPOSITION_LABEL);
    let p3_bits = cap(boost(
        air::composition_error(air_vector.num_deep_terms as usize, 1.0, CAP_BITS),
        deep_pow_bits as usize,
    )
    .bits());

    assert!(
        fixed_bits <= p3_bits + TIGHT_TOL,
        "deep-composition: {fixed_bits} > {p3_bits}"
    );
    assert!(
        p3_bits - fixed_bits < TIGHT_TOL,
        "deep-composition: gap {} >= tolerance, fixed drifted conservative",
        p3_bits - fixed_bits
    );
}

/// The collision-resistance term is reported at the cap verbatim on both sides — a degenerate but
/// worthwhile check, since it is what every other round's capping is measured against above.
#[test]
fn collision_term_is_the_cap() {
    let params = ProtocolParams {
        log_blowup: LOG_BLOWUP,
        log_folding_arity: LOG_FOLDING_ARITY,
        num_queries: 27,
        query_pow_bits: 17,
        deep_pow_bits: 12,
        folding_pow_bits: 4,
        lookup_pow_bits: 0,
    };
    let instance = InstanceShape {
        log_max_height: 20,
        field_bits: FIELD_BITS,
        collision_resistance: 96,
    };
    let air_shape = AirShape {
        num_composed_constraints: LARGE.num_composed_constraints,
        max_constraint_degree: LARGE.max_constraint_degree,
        max_combo: OOD_MAX_COMBO,
        num_deep_terms: Some(LARGE.num_deep_terms),
        lookup: LookupShape {
            fractions_per_row: LARGE.fractions_per_row,
            max_message_width: LARGE.max_message_width,
        },
    };
    let report = security_report(&params, &instance, &air_shape);
    assert_eq!(term_bits(&report, COLLISION_LABEL), 96.0);
}
