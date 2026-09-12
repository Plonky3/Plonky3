//! Security accounting derived from the verifier's statement.
//!
//! The report union-composes the AIR and lookup reductions, both prescribed-point
//! commitment openings when present, and the configured collision cap. It inherits
//! each PCS's proximity and Fiat-Shamir assumptions; a conjectural PCS assessment
//! does not become a proven bound here. Uniform transcript challenges are assumed.
//! The outer sumcheck's grinding is conservatively given no credit.
//!
//! A commitment can leave multiple candidate trace polynomials while the outer
//! AIR and lookup challenges are drawn. Every outer reduction term therefore
//! includes a union over the product of the main and preprocessed candidate
//! list sizes, fixed at commitment time. PCS errors and collision security are
//! then composed separately.

use alloc::vec::Vec;

use p3_air::symbolic::AirLayout;
use p3_challenger::{CanObserve, CanSampleUniformBits, FieldChallenger, GrindingChallenger};
use p3_field::Field;
use p3_lookup::InteractionSymbolicBuilder;
use p3_security::multilinear::{MultilinearAirParams, MultilinearLookupParams, reduction_terms};
use p3_security::{ErrorBits, SecurityTerm};
use p3_sumcheck::{PrescribedOpeningSecurity, PrescribedPointPcs};
use thiserror::Error;

use crate::VerifierInstances;
use crate::config::{Commitment, MultiStarkConfig};
use crate::folder::VerifierAir;
use crate::instance::Instances;
use crate::lookup::{LookupError, LookupPlan};
use crate::selectors::{PeriodicError, periodic_num_variables};
use crate::zerocheck::get_air_degrees;

/// Reasons a security target cannot be certified for this statement.
#[derive(Debug, Error)]
pub enum SecurityError {
    /// Statement dimensions or AIR metadata do not describe a supported proof.
    #[error("invalid multi-STARK security shape: {0}")]
    InvalidShape(&'static str),
    /// The lookup counting argument or its field is unsupported.
    #[error("lookup security: {0}")]
    Lookup(#[from] LookupError),
    /// Periodic columns do not fit the declared trace dimensions.
    #[error("periodic security: {0}")]
    Periodic(#[from] PeriodicError),
    /// A component has supplied no usable soundness evidence.
    #[error("unassessed multi-STARK security component: {0}")]
    UnassessedComponent(&'static str),
    /// The complete union bound is below the requested target.
    #[error("multi-STARK security is {available:.2} bits, below the {requested}-bit target")]
    InsufficientSecurity {
        /// Requested minimum.
        requested: usize,
        /// Composed bound, under the report's stated assumptions.
        available: f64,
    },
}

/// Labeled, union-composed accounting for a concrete multilinear AIR statement.
///
/// Missing PCS or collision evidence leaves [`Self::security_bits`] as `None`.
/// The numeric result is conditional on the PCS implementations' security
/// assessments and the configured hash cap, not a claim about arbitrary primitives.
#[derive(Clone, Debug)]
pub struct MultiStarkSecurityReport {
    terms: Vec<SecurityTerm>,
    unassessed: Vec<&'static str>,
}

impl MultiStarkSecurityReport {
    /// Every assessed contribution, including the commitment and transcript cap.
    pub fn terms(&self) -> &[SecurityTerm] {
        &self.terms
    }

    /// Components that prevent a complete bound from being returned.
    pub fn unassessed_components(&self) -> &[&'static str] {
        &self.unassessed
    }

    /// Negative log of the union bound, or `None` if any component is unassessed.
    pub fn security_bits(&self) -> Option<f64> {
        self.unassessed.is_empty().then(|| {
            ErrorBits::sum(&self.terms.iter().map(|term| term.bits).collect::<Vec<_>>())
                .bits()
                .max(0.0)
        })
    }

    /// Fail closed on missing evidence or an unattainable minimum.
    pub fn require_security(&self, target_bits: usize) -> Result<(), SecurityError> {
        if let Some(&component) = self.unassessed.first() {
            return Err(SecurityError::UnassessedComponent(component));
        }
        let available = self.security_bits().expect("all components were assessed");
        if available < target_bits as f64 {
            return Err(SecurityError::InsufficientSecurity {
                requested: target_bits,
                available,
            });
        }
        Ok(())
    }

    fn add_evidence(&mut self, label: &'static str, bits: Option<ErrorBits>) {
        match bits {
            Some(bits) if bits.bits().is_finite() && bits.bits() >= 0.0 => {
                self.terms.push(SecurityTerm::new(label, bits));
            }
            _ => self.unassessed.push(label),
        }
    }

    /// Add one PCS's error and return its contribution to the joint candidate count.
    fn add_opening_evidence(
        &mut self,
        label: &'static str,
        evidence: Option<PrescribedOpeningSecurity>,
    ) -> f64 {
        match evidence {
            Some(evidence)
                if evidence.error.bits().is_finite()
                    && evidence.error.bits() >= 0.0
                    && evidence.log2_max_candidates.is_finite()
                    && evidence.log2_max_candidates >= 0.0 =>
            {
                self.terms.push(SecurityTerm::new(label, evidence.error));
                evidence.log2_max_candidates
            }
            _ => {
                self.unassessed.push(label);
                0.0
            }
        }
    }
}

/// Assess this verifier statement without reading any proof-supplied counts.
///
/// The PCS receives the same opening protocol used by verification, including
/// main and preprocessed successor columns. It must assess that exact shape.
/// AIR definitions and configurations remain trusted inputs, as in verification.
pub fn security_report<C, A>(
    config: &C,
    instances: &VerifierInstances<'_, C, A>,
) -> Result<MultiStarkSecurityReport, SecurityError>
where
    C: MultiStarkConfig,
    C::Pcs: PrescribedPointPcs<C::Challenge, C::Challenger>,
    C::Challenger: FieldChallenger<C::Val>
        + GrindingChallenger<Witness = C::Val>
        + CanSampleUniformBits<C::Val>
        + CanObserve<Commitment<C>>,
    A: VerifierAir<C::Val, C::Challenge>,
{
    assess_statement(config, &instances.statement())
}

pub(super) fn assess_statement<C, A>(
    config: &C,
    instances: &Instances<'_, C, A>,
) -> Result<MultiStarkSecurityReport, SecurityError>
where
    C: MultiStarkConfig,
    C::Pcs: PrescribedPointPcs<C::Challenge, C::Challenger>,
    C::Challenger: FieldChallenger<C::Val>
        + GrindingChallenger<Witness = C::Val>
        + CanSampleUniformBits<C::Val>
        + CanObserve<Commitment<C>>,
    A: VerifierAir<C::Val, C::Challenge>,
{
    let invalid = SecurityError::InvalidShape;
    if instances.is_empty() {
        return Err(invalid("empty AIR batch"));
    }
    let heights = instances.num_variables();
    let mut main_cells = 0usize;
    let mut preprocessed_cells = 0usize;
    let mut num_fractions = 0usize;
    let mut max_num_constraints = 0;
    let mut max_degree = 0;

    for (instance, &height) in instances.iter().zip(&heights) {
        if height == 0 || height >= usize::BITS as usize || height < config.min_num_variables() {
            return Err(invalid("trace arity is outside the supported range"));
        }
        let air = instance.air;
        let rows = 1usize << height;
        if air.width() == 0 || instance.public_values.len() != air.num_public_values() {
            return Err(invalid(
                "trace width or public-value count disagrees with the AIR",
            ));
        }
        periodic_num_variables(air.num_periodic_columns(), &air.periodic_columns(), height)?;
        for (width, next, total) in [
            (air.width(), air.main_next_row_columns(), &mut main_cells),
            (
                air.preprocessed_width(),
                air.preprocessed_next_row_columns(),
                &mut preprocessed_cells,
            ),
        ] {
            if next.iter().any(|&column| column >= width) {
                return Err(invalid("successor column is outside the trace width"));
            }
            *total = rows
                .checked_mul(width)
                .and_then(|cells| total.checked_add(cells))
                .ok_or_else(|| invalid("stacked trace dimensions overflow"))?;
        }

        let builder = InteractionSymbolicBuilder::<C::Val, C::Challenge>::from_air(
            air,
            AirLayout::from_air::<C::Val>(air),
        );
        if !builder.exclusive_interactions().is_empty() {
            return Err(invalid("exclusive lookups are unsupported"));
        }
        let constraints = builder
            .base_constraints()
            .len()
            .checked_add(builder.extension_constraints().len())
            .ok_or_else(|| invalid("constraint count overflow"))?;
        max_num_constraints = max_num_constraints.max(constraints);
        let symbolic_degree = builder
            .base_constraints()
            .iter()
            .map(|expression| expression.poly_degree(2, &[]))
            .chain(
                builder
                    .extension_constraints()
                    .iter()
                    .map(|expression| expression.poly_degree(2, &[])),
            )
            .max()
            .unwrap_or(0);
        if air
            .max_constraint_degree()
            .is_some_and(|hint| hint < symbolic_degree)
        {
            return Err(invalid(
                "constraint degree hint understates the symbolic degree",
            ));
        }
        if constraints > 0 && symbolic_degree == 0 {
            return Err(invalid("constant constraint families are unsupported"));
        }
        let tuples = builder
            .local_interactions()
            .iter()
            .try_fold(builder.global_interactions().len(), |total, local| {
                total.checked_add(local.tuples.len())
            })
            .ok_or_else(|| invalid("lookup tuple count overflow"))?;
        num_fractions = rows
            .checked_mul(tuples)
            .and_then(|count| num_fractions.checked_add(count))
            .ok_or_else(|| invalid("lookup dimensions overflow"))?;
        if constraints == 0 && tuples == 0 {
            return Err(invalid("AIR declares no constraints or lookup tuples"));
        }
        max_degree = max_degree.max(get_air_degrees::<C::Val, C::Challenge, A>(air).max());
    }
    // Both layout engines round the stacked size up to a power of two.
    if main_cells.checked_next_power_of_two().is_none()
        || preprocessed_cells.checked_next_power_of_two().is_none()
        || num_fractions.checked_next_power_of_two().is_none()
        || max_degree.checked_add(1).is_none()
    {
        return Err(invalid("padded dimensions or round degree overflow"));
    }
    let plan = LookupPlan::build::<C::Challenge, A>(&instances.airs(), &heights)?;
    let lookup = plan.map(|plan| MultilinearLookupParams {
        num_variables: plan.num_variables,
        num_fractions,
        max_message_width: plan.max_width,
    });
    let num_variables = heights
        .iter()
        .copied()
        .max()
        .unwrap()
        .max(lookup.map_or(0, |plan| plan.num_variables));
    let order = C::Challenge::order();
    let field_bits = order.bits().saturating_sub(1) as usize;
    let nonzero_field_bits = (order - 1u32).bits().saturating_sub(1) as usize;
    let mut report = MultiStarkSecurityReport {
        terms: reduction_terms(
            &MultilinearAirParams {
                num_instances: instances.len(),
                max_num_constraints,
                num_variables,
                constraint_degree: max_degree,
                lookup,
            },
            field_bits,
            nonzero_field_bits,
        ),
        unassessed: Vec::new(),
    };
    let num_reduction_terms = report.terms.len();
    let mut log2_candidates = report.add_opening_evidence(
        "main-pcs",
        config
            .pcs()
            .prescribed_security(&instances.opening_protocol()),
    );
    if preprocessed_cells > 0 {
        log2_candidates += report.add_opening_evidence(
            "preprocessed-pcs",
            config
                .preprocessed_pcs()
                .prescribed_security(&instances.preprocessed_opening_protocol()),
        );
    }
    for term in &mut report.terms[..num_reduction_terms] {
        term.bits = ErrorBits::from_log2((term.bits.bits() - log2_candidates).max(0.0));
    }
    report.add_evidence(
        "commitment-and-transcript-collision",
        config
            .collision_resistance_bits()
            .map(|bits| ErrorBits::from_log2(bits as f64)),
    );
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn missing_or_malformed_opening_evidence_cannot_certify_a_target() {
        for evidence in [
            None,
            Some(PrescribedOpeningSecurity {
                error: ErrorBits::from_log2(100.0),
                log2_max_candidates: f64::NAN,
            }),
            Some(PrescribedOpeningSecurity {
                error: ErrorBits::from_log2(100.0),
                log2_max_candidates: f64::INFINITY,
            }),
            Some(PrescribedOpeningSecurity {
                error: ErrorBits::from_log2(100.0),
                log2_max_candidates: -1.0,
            }),
            Some(PrescribedOpeningSecurity {
                error: ErrorBits::from_log2(f64::NAN),
                log2_max_candidates: 0.0,
            }),
            Some(PrescribedOpeningSecurity {
                error: ErrorBits::from_log2(f64::INFINITY),
                log2_max_candidates: 0.0,
            }),
            Some(PrescribedOpeningSecurity {
                error: ErrorBits::from_log2(-1.0),
                log2_max_candidates: 0.0,
            }),
        ] {
            let mut report = MultiStarkSecurityReport {
                terms: Vec::new(),
                unassessed: Vec::new(),
            };
            report.add_opening_evidence("main-pcs", evidence);
            assert!(matches!(
                report.require_security(1),
                Err(SecurityError::UnassessedComponent("main-pcs"))
            ));
            assert_eq!(report.security_bits(), None);
        }
    }
}
