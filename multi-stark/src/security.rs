//! Security accounting derived from the verifier's statement.
//!
//! The report union-composes the AIR, lookup, and binary-bus reductions, both
//! prescribed-point commitment openings when present, and the configured collision cap.
//! It inherits each PCS's proximity and Fiat-Shamir assumptions; a conjectural PCS
//! assessment does not become a proven bound here. Uniform transcript challenges are assumed.
//! The outer sumcheck's grinding is conservatively given no credit.
//!
//! A commitment can leave multiple candidate trace polynomials while the outer
//! AIR and lookup challenges are drawn. Every outer reduction term therefore
//! includes a union over the product of the main and preprocessed candidate
//! list sizes, fixed at commitment time. PCS errors and collision security are
//! then composed separately.

use alloc::vec::Vec;
use core::num::NonZeroUsize;

use p3_air::boundary;
use p3_air::symbolic::AirLayout;
use p3_challenger::{CanObserve, CanSampleUniformBits, FieldChallenger, GrindingChallenger};
use p3_field::Field;
use p3_lookup::{IndexedLookupError, InteractionSymbolicBuilder};
use p3_security::multilinear::{
    MultilinearAirParams, MultilinearLogupStarParams, MultilinearLookupParams, reduction_terms,
};
use p3_security::{ErrorBits, SecurityTerm};
use p3_sumcheck::{PrescribedOpeningSecurity, PrescribedPointPcs};
use p3_util::log2_ceil_usize;
use thiserror::Error;

use crate::VerifierInstances;
use crate::bus::BusContext;
use crate::config::{Commitment, MultiStarkConfig};
use crate::folder::{VerifierAir, boundary_io_pins};
use crate::indexed::IndexedPlan;
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
    /// The indexed lookups the AIRs declare do not describe a reduction.
    #[error("indexed lookup security: {0}")]
    IndexedLookup(#[from] IndexedLookupError),
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

    /// Add one PCS's labelled errors and return its contribution to the candidate count.
    ///
    /// A scheme stacking a reduction on a commitment charges one term per source.
    ///
    /// Each keeps the label its own crate gave it, so the report says which one is short.
    ///
    /// Every term is also attributed to the commitment it was charged for.
    ///
    /// ```text
    ///     one scheme, both commitments  ->  the same label twice
    ///     component                     ->  which of the two the term belongs to
    /// ```
    ///
    /// A component with nothing to charge is recorded as unassessed under the same name.
    fn add_opening_evidence(
        &mut self,
        component: &'static str,
        evidence: Option<PrescribedOpeningSecurity>,
    ) -> f64 {
        // A term is usable when it names a probability in `[0, 1]`, so its bits are `>= 0`.
        //
        // Infinite bits are a zero error, which a reduction that never runs reports.
        //
        // Rejecting that would leave a component unassessed for having nothing to charge.
        //
        // The union of the terms is what has to be a real bound, and it is checked below.
        let charged = |term: &SecurityTerm| !term.bits.bits().is_nan() && term.bits.bits() >= 0.0;

        // One unusable term makes the whole component unassessed, rather than shrinking it.
        let usable = |evidence: &PrescribedOpeningSecurity| {
            !evidence.terms.is_empty()
                && evidence.terms.iter().all(charged)
                && evidence.error().bits().is_finite()
                && evidence.log2_max_candidates.is_finite()
                && evidence.log2_max_candidates >= 0.0
        };
        match evidence {
            Some(evidence) if usable(&evidence) => {
                self.terms.extend(
                    evidence
                        .terms
                        .iter()
                        .map(|term| term.in_component(component)),
                );
                evidence.log2_max_candidates
            }
            _ => {
                self.unassessed.push(component);
                0.0
            }
        }
    }
}

fn bus_composition_terms(
    field_bits: NonZeroUsize,
    num_variables: usize,
    degree: usize,
) -> [SecurityTerm; 2] {
    // One fresh scalar batches the push and pull identities.
    let direction = SecurityTerm::new(
        "binary-bus-direction-batching",
        ErrorBits::from_log2(field_bits.get() as f64),
    );

    // Schwartz--Zippel charges at most one degree-D identity test per round.
    let events = num_variables.saturating_mul(degree);
    let composition = SecurityTerm::new(
        "binary-bus-composition-sumcheck",
        ErrorBits::from_log2((field_bits.get() as f64 - libm::log2(events as f64)).max(0.0)),
    );
    [direction, composition]
}

impl IndexedPlan {
    /// Read the shape the soundness bound is charged against.
    ///
    /// Every number here comes from the AIRs and the trace heights.
    ///
    /// None of them is read off a proof, which is what lets a verifier trust the bound.
    ///
    /// # Panics
    ///
    /// Never in practice.
    ///
    /// A plan exists only when some AIR declares a table, and a table always has a reader.
    pub(crate) fn security_params(&self) -> MultilinearLogupStarParams {
        // Blocks are laid down tallest first, each at a multiple of its own height.
        //
        // The run of used leaves therefore has no gaps.
        //
        //     leaves = sum over tables of (entries + rows of every reader)
        //
        // The padded table is the next power of two.
        //
        // Its extra leaves carry a zero over a one, so they hold no pole.
        let num_leaves = self
            .tables()
            .iter()
            .map(|plan| {
                let readers = plan
                    .readers
                    .iter()
                    .map(|reader| 1usize << reader.num_variables)
                    .sum::<usize>();
                (1usize << plan.table.num_variables) + readers
            })
            .sum::<usize>();

        MultilinearLogupStarParams {
            num_variables: log2_ceil_usize(num_leaves),
            num_leaves,
            max_readers_per_table: self
                .tables()
                .iter()
                .map(|plan| plan.readers.len())
                .max()
                .expect("a plan exists only when some table is declared"),
            max_reader_variables: self
                .tables()
                .iter()
                .flat_map(|plan| plan.readers.iter())
                .map(|reader| reader.num_variables)
                .max()
                .expect("a table with no reader is rejected when the plan is built"),
            max_table_variables: self
                .tables()
                .iter()
                .map(|plan| plan.table.num_variables)
                .max()
                .expect("a plan exists only when some table is declared"),
            num_column_claims: self
                .tables()
                .iter()
                .map(|plan| plan.table.columns.len())
                .sum(),
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

        // A malformed declaration is rejected by prove and by verify.
        // Reporting a security level for a statement neither accepts would mislead.
        if boundary::validate(
            air.public_boundary_io(),
            air.width(),
            air.num_public_values(),
        )
        .is_err()
        {
            return Err(invalid("public boundary declaration is malformed"));
        }

        let builder = InteractionSymbolicBuilder::<C::Val, C::Challenge>::from_air(
            air,
            AirLayout::from_air::<C::Val>(air),
        );
        if !builder.exclusive_interactions().is_empty() {
            return Err(invalid("exclusive lookups are unsupported"));
        }
        let own_constraints = builder
            .base_constraints()
            .len()
            .checked_add(builder.extension_constraints().len())
            .ok_or_else(|| invalid("constraint count overflow"))?;
        // The folder batches one pin per listed cell with the AIR's own constraints,
        // and no symbolic pass sees them.
        let pins = boundary_io_pins(air.public_boundary_io());
        let constraints = own_constraints
            .checked_add(pins.count)
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
        // A constant family has no round polynomial of its own.
        // A listed cell lifts it to the pin's degree, which is what `get_air_degrees` scores,
        // so both entry points accept and reject the same statements.
        if own_constraints > 0 && symbolic_degree.max(pins.degree) == 0 {
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
    // The same plan feeds the soundness terms and the opening shapes below.
    //
    // An assessment covering fewer batches than the proof opens overstates the bound.
    let indexed = IndexedPlan::build::<C::Val, C::Challenge, A>(&instances.airs(), &heights)?;
    let bus = BusContext::<C::Val, C::Challenge>::build(&instances.airs(), &heights)
        .map_err(|_| invalid("binary-bus declarations do not define a supported plan"))?;
    let logup_star = indexed.as_ref().map(IndexedPlan::security_params);
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
                // This prover spends one round per variable.
                // A skip round would change the accounting, so it is declared
                // absent rather than defaulted.
                skip: None,
                logup_star,
            },
            field_bits,
            nonzero_field_bits,
        ),
        unassessed: Vec::new(),
    };
    if let Some(context) = &bus {
        let field_bits = NonZeroUsize::new(field_bits)
            .ok_or_else(|| invalid("binary-bus challenge field is trivial"))?;
        report.terms.push(context.plan().security_term(field_bits));
        report.terms.extend(bus_composition_terms(
            field_bits,
            context.max_num_variables(),
            context.composition_degree(),
        ));
    }
    let num_reduction_terms = report.terms.len();
    // The scheme is assessed against the opening protocol verification actually runs.
    //
    // The indexed-lookup reduction closes on two further points per table it touches.
    //
    // Those batches have to reach the protocol built here, not only the one opened with.
    //
    // A protocol missing them assesses a smaller opening than the proof performs.
    //
    // It also understates the candidate count subtracted from every reduction term below.
    let mut log2_candidates = report.add_opening_evidence(
        "main-pcs",
        config.pcs().prescribed_security(
            &instances
                .main_schedule(indexed.as_ref(), bus.as_ref(), |_, _| ())
                .into_protocol(),
        ),
    );
    if preprocessed_cells > 0 {
        log2_candidates += report.add_opening_evidence(
            "preprocessed-pcs",
            config.preprocessed_pcs().prescribed_security(
                &instances
                    .preprocessed_schedule(indexed.as_ref(), bus.as_ref(), |_, _| ())
                    .into_protocol(),
            ),
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
    fn bus_composition_terms_charge_only_their_actual_randomness() {
        let [direction, composition] =
            bus_composition_terms(NonZeroUsize::new(128).unwrap(), 12, 4);

        assert_eq!(direction.label, "binary-bus-direction-batching");
        assert_eq!(direction.bits.bits(), 128.0);
        assert_eq!(composition.label, "binary-bus-composition-sumcheck");
        assert_eq!(composition.bits.bits(), 128.0 - 48.0_f64.log2());
    }

    #[test]
    fn a_reduction_with_nothing_to_charge_is_still_assessed() {
        // A reduction that never runs reports a zero error, which is infinitely many bits.
        //
        //     zero reductions  ->  no challenge at all  ->  error 0
        //
        // Fixture state: one finite term at 100 bits, one term carrying no error.
        //
        //     union  ->  2^-100 + 0  ->  100 bits
        //
        // Rejecting the zero term would leave the component unassessed on a sound
        // statement, and the proving path fails closed on an unassessed component.
        let mut report = MultiStarkSecurityReport {
            terms: Vec::new(),
            unassessed: Vec::new(),
        };
        let evidence = PrescribedOpeningSecurity {
            terms: alloc::vec![
                SecurityTerm::new("commitment", ErrorBits::from_log2(100.0)),
                SecurityTerm::new("reduction", ErrorBits::from_log2(f64::INFINITY)),
            ],
            log2_max_candidates: 0.0,
        };

        assert_eq!(report.add_opening_evidence("main-pcs", Some(evidence)), 0.0);
        assert!(report.unassessed_components().is_empty());

        // Both terms are kept, each attributed to the commitment it was charged for.
        assert_eq!(report.terms().len(), 2);
        assert!(
            report
                .terms()
                .iter()
                .all(|term| term.component == Some("main-pcs"))
        );

        // The zero-error term contributes nothing to the union, so the bound is the other.
        assert_eq!(report.security_bits(), Some(100.0));

        // A component whose every term is a zero error has no bound at all, so it is
        // unassessed rather than certified at infinite security.
        let mut empty = MultiStarkSecurityReport {
            terms: Vec::new(),
            unassessed: Vec::new(),
        };
        let nothing = PrescribedOpeningSecurity::single(
            "reduction",
            ErrorBits::from_log2(f64::INFINITY),
            0.0,
        );
        empty.add_opening_evidence("main-pcs", Some(nothing));
        assert_eq!(empty.unassessed_components(), ["main-pcs"]);
    }

    #[test]
    fn missing_or_malformed_opening_evidence_cannot_certify_a_target() {
        // A usable error paired with a candidate count that is not a bound.
        let bad_candidates = [f64::NAN, f64::INFINITY, -1.0].map(|candidates| {
            PrescribedOpeningSecurity::single("t", ErrorBits::from_log2(100.0), candidates)
        });
        // A usable candidate count paired with an error that is not a bound.
        let bad_error = [f64::NAN, f64::INFINITY, -1.0]
            .map(|bits| PrescribedOpeningSecurity::single("t", ErrorBits::from_log2(bits), 0.0));
        // No evidence at all, and evidence carrying no term to charge.
        let empty = PrescribedOpeningSecurity {
            terms: Vec::new(),
            log2_max_candidates: 0.0,
        };
        for evidence in core::iter::once(None)
            .chain(core::iter::once(Some(empty)))
            .chain(bad_candidates.into_iter().map(Some))
            .chain(bad_error.into_iter().map(Some))
        {
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
