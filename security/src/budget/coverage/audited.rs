//! A budget that has been checked against the rounds its protocol runs.

use super::error::Unaccounted;
use super::schedule::ChallengeSchedule;
use crate::budget::report::{Accounting, NUM_TERMS, SecurityReport, SecurityTerm};

/// A budget every declared round of which was actually bounded.
///
/// A waived round reports the transcript ceiling, and so does a genuinely strong one.
///
/// So each round records whether its bits bound anything, and the caller names the rounds it runs.
///
/// Only this form reports a security level.
///
/// A level therefore cannot be read off a budget that skipped one of the protocol's rounds.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
#[must_use]
pub struct AuditedReport {
    report: SecurityReport,
}

impl AuditedReport {
    /// Grades the budget against the rounds the protocol runs, refusing the first ungraded one.
    pub const fn new(
        report: SecurityReport,
        schedule: &ChallengeSchedule<'_>,
    ) -> Result<Self, Unaccounted> {
        let labels = schedule.labels();
        let mut i = 0;
        while i < labels.len() {
            match find(report.terms(), labels[i]) {
                None => return Err(Unaccounted::Unmodeled(labels[i])),
                Some(term) => {
                    if matches!(term.accounting, Accounting::Waived) {
                        return Err(Unaccounted::Waived(labels[i]));
                    }
                }
            }
            i += 1;
        }
        Ok(Self { report })
    }

    /// The underlying breakdown.
    pub const fn report(&self) -> &SecurityReport {
        &self.report
    }

    /// Every round.
    pub const fn terms(&self) -> &[SecurityTerm; NUM_TERMS] {
        self.report.terms()
    }

    /// The round that binds.
    pub const fn binding_term(&self) -> SecurityTerm {
        self.report.binding_term()
    }

    /// Attained conjectured security, in fixed point.
    pub const fn attained(&self) -> u64 {
        self.report.attained()
    }

    /// Attained conjectured security in whole bits, rounded down.
    pub const fn security_level(&self) -> u32 {
        self.report.security_level()
    }
}

/// The round carrying that name, if the budget has one.
const fn find(terms: &[SecurityTerm; NUM_TERMS], label: &str) -> Option<SecurityTerm> {
    let mut i = 0;
    while i < NUM_TERMS {
        if same(terms[i].label, label) {
            return Some(terms[i]);
        }
        i += 1;
    }
    None
}

/// Whether two names are the same text, comparable in a constant.
const fn same(a: &str, b: &str) -> bool {
    let (a, b) = (a.as_bytes(), b.as_bytes());
    if a.len() != b.len() {
        return false;
    }
    let mut i = 0;
    while i < a.len() {
        if a[i] != b[i] {
            return false;
        }
        i += 1;
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::budget::report::{
        COLLISION_LABEL, COMPOSITION_LABEL, DEEP_COMPOSITION_LABEL, FOLDING_LABEL, LOOKUP_LABEL,
        OUT_OF_DOMAIN_LABEL, QUERY_LABEL,
    };
    use crate::budget::security_report;
    use crate::budget::shape::{AirShape, InstanceShape, LookupShape, ProtocolParams};

    // Every round of a lifted multi-AIR statement with a bus and a batched DEEP quotient.
    const FULL: &[&str] = &[
        LOOKUP_LABEL,
        COMPOSITION_LABEL,
        OUT_OF_DOMAIN_LABEL,
        DEEP_COMPOSITION_LABEL,
        FOLDING_LABEL,
        QUERY_LABEL,
        COLLISION_LABEL,
    ];

    const PARAMS: ProtocolParams = ProtocolParams {
        log_blowup: 3,
        log_folding_arity: 2,
        num_queries: 27,
        query_pow_bits: 17,
        ood_pow_bits: 0,
        deep_pow_bits: 12,
        folding_pow_bits: 4,
        lookup_pow_bits: 0,
    };

    const INSTANCE: InstanceShape = InstanceShape {
        log_max_height: 20,
        field_bits: crate::fixed::from_bits(128) - 2,
        collision_resistance: 128,
    };

    const AIR: AirShape = AirShape {
        num_composed_constraints: 531,
        max_constraint_degree: 9,
        num_quotient_chunks: 8,
        max_combo: 2,
        num_deep_terms: Some(130),
        lookup: Some(LookupShape {
            fractions_per_row: 27,
            max_message_width: 16,
        }),
    };

    #[test]
    fn a_fully_declared_protocol_passes_and_reports_its_level() {
        let report = security_report(&PARAMS, &INSTANCE, &AIR);
        let audited = AuditedReport::new(report, &ChallengeSchedule::new(FULL)).unwrap();
        assert_eq!(audited.security_level(), 96);
    }

    #[test]
    fn a_bus_left_out_of_the_shape_is_refused_rather_than_graded() {
        // The caller forgot the bus, so the lookup round is waived and reported at the ceiling.
        let air = AirShape {
            lookup: None,
            ..AIR
        };
        let report = security_report(&PARAMS, &INSTANCE, &air);

        // Without the gate this configuration looks as strong as the fully declared one.
        assert_eq!(report.terms()[0].bits, INSTANCE.cap());
        assert_eq!(report.security_level(), 96);

        let refusal = AuditedReport::new(report, &ChallengeSchedule::new(FULL)).unwrap_err();
        assert_eq!(refusal, Unaccounted::Waived(LOOKUP_LABEL));
    }

    #[test]
    fn a_batched_deep_quotient_left_out_of_the_shape_is_refused() {
        let air = AirShape {
            num_deep_terms: None,
            ..AIR
        };
        let report = security_report(&PARAMS, &INSTANCE, &air);
        assert_eq!(report.security_level(), 96);

        let refusal = AuditedReport::new(report, &ChallengeSchedule::new(FULL)).unwrap_err();
        assert_eq!(refusal, Unaccounted::Waived(DEEP_COMPOSITION_LABEL));
    }

    #[test]
    fn a_round_the_budget_does_not_model_is_refused() {
        // A protocol that samples a challenge this budget has no bound for at all.
        let schedule = ChallengeSchedule::new(&["univariate-skip"]);
        let report = security_report(&PARAMS, &INSTANCE, &AIR);
        let refusal = AuditedReport::new(report, &schedule).unwrap_err();
        assert_eq!(refusal, Unaccounted::Unmodeled("univariate-skip"));
        assert_eq!(refusal.label(), "univariate-skip");
    }

    #[test]
    fn a_protocol_without_a_bus_may_declare_so_and_still_be_graded() {
        // Waiving is only an error when the protocol actually runs the round.
        let air = AirShape {
            lookup: None,
            ..AIR
        };
        let without = [
            COMPOSITION_LABEL,
            OUT_OF_DOMAIN_LABEL,
            DEEP_COMPOSITION_LABEL,
            FOLDING_LABEL,
            QUERY_LABEL,
            COLLISION_LABEL,
        ];
        let report = security_report(&PARAMS, &INSTANCE, &air);
        let audited = AuditedReport::new(report, &ChallengeSchedule::new(&without)).unwrap();
        assert_eq!(audited.security_level(), 96);
    }

    // The check is available before runtime, so a configuration can be refused at compile time.
    const UNDECLARED: AirShape = AirShape {
        num_deep_terms: None,
        lookup: None,
        ..AIR
    };
    const _: () = assert!(
        AuditedReport::new(
            security_report(&PARAMS, &INSTANCE, &UNDECLARED),
            &ChallengeSchedule::new(&[LOOKUP_LABEL]),
        )
        .is_err()
    );

    #[test]
    fn every_round_the_budget_bounds_is_reachable_by_name() {
        // A label typo would otherwise turn a real round into an unmodeled one forever.
        let report = security_report(&PARAMS, &INSTANCE, &AIR);
        for term in report.terms() {
            assert!(
                FULL.contains(&term.label),
                "{} is not in the declared schedule",
                term.label
            );
        }
        assert_eq!(FULL.len(), NUM_TERMS);
    }
}
