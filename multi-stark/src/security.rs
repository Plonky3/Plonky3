//! Security accounting derived from the verifier's statement.
//!
//! The report union-composes the AIR, lookup, and binary-bus reductions.
//!
//! Both prescribed-point openings join them, when present, along with the collision cap.
//!
//! It inherits each scheme's proximity and Fiat-Shamir assumptions.
//!
//! A conjectural assessment does not become a proven bound here.
//!
//! Uniform transcript challenges are assumed throughout.
//!
//! The outer sumcheck's grinding is conservatively given no credit.
//!
//! # The candidate-set charge
//!
//! A commitment can leave several candidate traces open while the outer challenges land.
//!
//! Every outer term therefore unions over the product of both commitments' lists.
//!
//! That product is fixed at commitment time, and the charge for it lives in one place.
//!
//! This crate is the top of the stack it prices.
//!
//! Each opening already charged the reductions it drew, and forwarded its count intact.
//!
//! What is left for this layer is its own AIR, lookup, and binary-bus draws.
//!
//! The builder below keeps those apart from the settled terms.
//!
//! It applies the charge exactly once, when the report is closed.
//!
//! Its lists, and the finished report's terms, are private to the module they live in.
//!
//! So a term reaches the report only through a builder method that decides its charge.
//!
//! Nothing outside that module can append to a report after it is closed.

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
use p3_sumcheck::PrescribedPointPcs;
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

/// Label the report gives the commitment and transcript collision cap.
const COLLISION_CAP_LABEL: &str = "commitment-and-transcript-collision";

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

pub use builder::MultiStarkSecurityReport;
use builder::{CommittedTrace, ReportBuilder};

/// Assembly of the report, behind a boundary the rest of this module cannot reach past.
///
/// The builder's fields are private to this child module.
///
/// The finished report is defined here too, so its terms are just as private.
///
/// So a term reaches the report only through one of the builder methods below.
///
/// And nothing outside this module can append to a report once it is closed.
///
/// Each decides the charge itself, rather than trusting the caller to have done it.
mod builder {
    use alloc::vec::Vec;

    use p3_security::{CandidateSet, ErrorBits, SecurityTerm};
    use p3_sumcheck::PrescribedOpeningSecurity;

    use super::{COLLISION_CAP_LABEL, SecurityError};

    /// Labeled, union-composed accounting for a concrete multilinear AIR statement.
    ///
    /// Missing scheme or collision evidence leaves the report with no number to give.
    ///
    /// The number is conditional on the schemes' own assessments and the configured cap.
    ///
    /// It is not a claim about arbitrary primitives.
    ///
    /// Every term here is final.
    ///
    /// The draws made before an opening named a candidate paid for the set once, on assembly.
    ///
    /// The finished report keeps no set, so a reader composes these terms as they stand.
    #[derive(Clone, Debug)]
    pub struct MultiStarkSecurityReport {
        terms: Vec<SecurityTerm>,
        unassessed: Vec<&'static str>,
    }

    impl MultiStarkSecurityReport {
        /// Every assessed contribution, including the commitment and transcript cap.
        ///
        /// Each is already charged for whatever candidate set it had to pay for.
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
    }

    /// Which commitment an opening's errors belong to.
    ///
    /// The set is closed because the report has exactly two commitments.
    ///
    /// Naming anything else is then a compile error rather than a label nobody checks.
    ///
    /// A reduction this composition draws itself is the case that matters.
    #[derive(Copy, Clone, Debug, PartialEq, Eq)]
    pub(super) enum CommittedTrace {
        /// The main trace commitment.
        Main,
        /// The preprocessed trace commitment.
        Preprocessed,
    }

    impl CommittedTrace {
        /// The name this commitment's terms are attributed to in the report.
        const fn label(self) -> &'static str {
            match self {
                Self::Main => "main-pcs",
                Self::Preprocessed => "preprocessed-pcs",
            }
        }
    }

    /// Assembles the report, charging the candidate set exactly once.
    ///
    /// The two kinds of term are kept in separate lists until the report is closed.
    ///
    /// ```text
    ///     outer     draws this composition makes after the commitments and before an
    ///               opening names a candidate: AIR, lookup, and binary-bus reductions
    ///     settled   terms that are already final: each opening's own errors, charged by
    ///               the layer that drew them, and the collision cap
    /// ```
    ///
    /// Closing the builder consumes it, and is the single place the charge is applied.
    ///
    /// # What is enforced, and what is not
    ///
    /// Both lists are private to this module.
    ///
    /// Nothing outside can push a term onto either one.
    ///
    /// Three methods add terms, and each fixes the charge for what it adds.
    ///
    /// ```text
    ///     outer draws       charged over the set when the builder is closed
    ///     opening evidence  charged already, by the layer that drew it
    ///     collision cap     owes nothing, and the method takes no term of its own
    /// ```
    ///
    /// A draw this composition makes cannot be attributed to a commitment either.
    ///
    /// The commitment name is a closed set of two, so there is no third name to give.
    ///
    /// What is left open is handing this a fabricated opening.
    ///
    /// That means a term of one's own, wrapped as evidence for one of the two commitments.
    ///
    /// It is a false claim about that commitment rather than a term taking a wrong turn.
    ///
    /// Sealing it belongs where the evidence type lives, not here.
    ///
    /// Nothing here sees a draw that was never added at all.
    ///
    /// A component with no usable evidence leaves the report with no number to give.
    ///
    /// That catches a missing component, not a missing term inside one.
    ///
    /// The tests naming each expected label are what cover the rest.
    pub(super) struct ReportBuilder {
        /// Reductions this composition draws, still uncharged.
        outer: Vec<SecurityTerm>,
        /// Terms that are final as they stand.
        settled: Vec<SecurityTerm>,
        /// Every candidate both commitments together still leave open.
        candidates: CandidateSet,
        /// Components that prevent a complete bound from being returned.
        unassessed: Vec<&'static str>,
    }

    impl ReportBuilder {
        /// A builder whose outer draws are the ones the AIR and lookup reductions make.
        ///
        /// Nothing is committed yet, so the candidate set starts at unique decoding.
        pub(super) const fn new(outer: Vec<SecurityTerm>) -> Self {
            Self {
                outer,
                settled: Vec::new(),
                candidates: CandidateSet::UNIQUE,
                unassessed: Vec::new(),
            }
        }

        /// Add more draws this composition makes before an opening names a candidate.
        ///
        /// They join the list that is charged when the builder closes.
        pub(super) fn add_outer_draws(&mut self, terms: impl IntoIterator<Item = SecurityTerm>) {
            self.outer.extend(terms);
        }

        /// Record the commitment and transcript collision cap.
        ///
        /// The cap is the one term that settles without a charge and is not an opening's.
        ///
        /// A collision is available to a prover before it commits to anything.
        ///
        /// So no candidate set applies to it, and it is not an outer draw.
        ///
        /// The label is fixed here, so nothing else may enter the report this way.
        pub(super) fn add_collision_cap(&mut self, bits: Option<ErrorBits>) {
            match bits {
                Some(bits) if bits.bits().is_finite() && bits.bits() >= 0.0 => {
                    self.settled
                        .push(SecurityTerm::new(COLLISION_CAP_LABEL, bits));
                }
                _ => self.unassessed.push(COLLISION_CAP_LABEL),
            }
        }

        /// Add one commitment's labelled errors and widen the set every outer draw pays for.
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
        /// The terms arrive already charged, so they settle here untouched.
        ///
        /// What the opening adds to this layer is its candidate count.
        ///
        /// The set is still open while every outer draw is made.
        ///
        /// Two open commitments multiply the tries a single outer draw gets.
        ///
        /// A commitment with nothing to charge is recorded as unassessed under its name.
        pub(super) fn add_opening_evidence(
            &mut self,
            commitment: CommittedTrace,
            evidence: Option<PrescribedOpeningSecurity>,
        ) {
            // A term is usable when it names a probability in `[0, 1]`, so bits are `>= 0`.
            //
            // Infinite bits are a zero error, which a reduction that never runs reports.
            //
            // Rejecting that would leave a component unassessed for nothing to charge.
            //
            // The union of the terms is what has to be a real bound, checked below.
            let charged =
                |term: &SecurityTerm| !term.bits.bits().is_nan() && term.bits.bits() >= 0.0;

            // One unusable term makes the whole component unassessed, not just smaller.
            let usable = |evidence: &PrescribedOpeningSecurity| {
                !evidence.terms.is_empty()
                    && evidence.terms.iter().all(charged)
                    && evidence.error().bits().is_finite()
                    && evidence.candidates().is_some()
            };
            match evidence {
                Some(evidence) if usable(&evidence) => {
                    let candidates = evidence
                        .candidates()
                        .expect("a usable count names a candidate set");
                    self.settled.extend(
                        evidence
                            .terms
                            .iter()
                            .map(|term| term.in_component(commitment.label())),
                    );
                    self.candidates = self.candidates.product(candidates);
                }
                _ => self.unassessed.push(commitment.label()),
            }
        }

        /// Charge every outer draw over the candidate set, once, and close the report.
        pub(super) fn finish(self) -> MultiStarkSecurityReport {
            let Self {
                outer,
                settled,
                candidates,
                unassessed,
            } = self;
            MultiStarkSecurityReport {
                // A prover may pick which trace it meant after seeing every outer challenge.
                //
                // So each outer draw is union-bounded over the whole set.
                terms: outer
                    .into_iter()
                    .map(|term| term.over_candidates(candidates))
                    .chain(settled)
                    .collect(),
                unassessed,
            }
        }
    }
}

/// Charge the one challenge that folds the bus family into the shared sumcheck.
///
/// Lambda weighs pull against push, then the bus against the AIR family.
/// A false claim leaves this residual in the shared sumcheck:
///
/// ```text
///     residual(tau, lambda) = P(tau) + lambda * d_push + lambda^2 * d_pull
/// ```
///
/// - `P` is the AIR family's residual as a polynomial in tau, fixed before lambda.
/// - `d_push` and `d_pull` are the bus share errors, fixed by ProductGKR before lambda.
/// - tau's free coordinates are drawn after lambda.
/// - So `P(tau)` itself is not fixed when lambda is drawn.
///
/// Two cases cover every forgery:
/// - `P` nonconstant: tau catches it, and the `zerocheck` term already charges that event.
/// - `P` constant: the residual is a nonzero polynomial of degree at most two in lambda.
///
/// Schwartz--Zippel bounds the second case by `2 / |EF|`.
/// So the `zerocheck` term does not grow, and this term charges only the second case.
///
/// The sumcheck rounds themselves are charged by the `constraint-sumcheck` term.
/// Its degree already covers the bus composition.
fn bus_batching_term(field_bits: NonZeroUsize) -> SecurityTerm {
    SecurityTerm::new(
        "binary-bus-batching",
        ErrorBits::from_log2(field_bits.get() as f64 - 1.0),
    )
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
/// The scheme receives the same opening protocol verification uses.
///
/// That includes the main and preprocessed successor columns, and it must assess them.
///
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
        //
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
        // The folder batches one pin per listed cell with the AIR's own constraints.
        //
        // No symbolic pass sees them.
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
        //
        // A listed cell lifts it to the pin's degree, which is what the scorer reads.
        //
        // So both entry points accept and reject the same statements.
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
        // Binary-bus declarations raise the shared round degree below, not the constraint count.
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
    // The bus family shares the constraint sumcheck.
    // Its composition degree already counts its own equality weight.
    // The AIR degree is charged one more for the zerocheck weight.
    //
    //     round degree = max(air + 1, bus)
    let max_degree = bus.as_ref().map_or(max_degree, |context| {
        max_degree.max(context.composition_degree().saturating_sub(1))
    });
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
    // Every term below is a draw made while both commitments still leave a list open.
    //
    // Each is charged over that list once, when the builder is closed.
    //
    // Nothing here subtracts anything itself.
    let mut builder = ReportBuilder::new(reduction_terms(
        &MultilinearAirParams {
            num_instances: instances.len(),
            max_num_constraints,
            num_variables,
            constraint_degree: max_degree,
            lookup,
            // This prover spends one round per variable.
            //
            // A skip round would change the accounting.
            //
            // So it is declared absent rather than defaulted.
            skip: None,
            logup_star,
        },
        field_bits,
        nonzero_field_bits,
    ));
    if let Some(context) = &bus {
        let field_bits = NonZeroUsize::new(field_bits)
            .ok_or_else(|| invalid("binary-bus challenge field is trivial"))?;
        builder.add_outer_draws([
            context.plan().security_term(field_bits),
            bus_batching_term(field_bits),
        ]);
    }
    // The scheme is assessed against the opening protocol verification actually runs.
    //
    // The indexed-lookup reduction closes on two further points per table it touches.
    //
    // Those batches have to reach the protocol built here, not only the one opened with.
    //
    // A protocol missing them assesses a smaller opening than the proof performs.
    //
    // It also understates the candidate count charged to every outer draw above.
    builder.add_opening_evidence(
        CommittedTrace::Main,
        config.pcs().prescribed_security(
            &instances
                .main_schedule(indexed.as_ref(), |_, _| ())
                .into_protocol(),
        ),
    );
    if preprocessed_cells > 0 {
        builder.add_opening_evidence(
            CommittedTrace::Preprocessed,
            config.preprocessed_pcs().prescribed_security(
                &instances
                    .preprocessed_schedule(indexed.as_ref(), |_, _| ())
                    .into_protocol(),
            ),
        );
    }
    builder.add_collision_cap(
        config
            .collision_resistance_bits()
            .map(|bits| ErrorBits::from_log2(bits as f64)),
    );
    Ok(builder.finish())
}

#[cfg(test)]
mod tests {
    use alloc::vec::Vec;

    use p3_sumcheck::PrescribedOpeningSecurity;

    use super::*;

    #[test]
    fn bus_batching_charges_a_degree_two_polynomial_in_lambda() {
        // Two roots out of 2^128 candidates leave 127 bits.
        let term = bus_batching_term(NonZeroUsize::new(128).unwrap());

        assert_eq!(term.label, "binary-bus-batching");
        assert_eq!(term.bits.bits(), 127.0);
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
        // Rejecting the zero term would leave the component unassessed on a sound claim.
        //
        // The proving path then fails closed on an unassessed component.
        let mut builder = ReportBuilder::new(Vec::new());
        let evidence = PrescribedOpeningSecurity {
            terms: alloc::vec![
                SecurityTerm::new("commitment", ErrorBits::from_log2(100.0)),
                SecurityTerm::new("reduction", ErrorBits::from_log2(f64::INFINITY)),
            ],
            log2_max_candidates: 0.0,
        };

        builder.add_opening_evidence(CommittedTrace::Main, Some(evidence));
        let report = builder.finish();
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

        // A component whose every term is a zero error has no bound at all.
        //
        // It is recorded unassessed rather than certified at infinite security.
        let mut empty = ReportBuilder::new(Vec::new());
        let nothing = PrescribedOpeningSecurity::single(
            "reduction",
            ErrorBits::from_log2(f64::INFINITY),
            0.0,
        );
        empty.add_opening_evidence(CommittedTrace::Main, Some(nothing));
        assert_eq!(empty.finish().unassessed_components(), ["main-pcs"]);
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
            let mut builder = ReportBuilder::new(Vec::new());
            builder.add_opening_evidence(CommittedTrace::Main, evidence);
            let report = builder.finish();
            assert!(matches!(
                report.require_security(1),
                Err(SecurityError::UnassessedComponent("main-pcs"))
            ));
            assert_eq!(report.security_bits(), None);
        }
    }

    #[test]
    fn every_outer_draw_is_charged_once_over_both_open_commitments() {
        // Fixture state: two outer draws at 100 bits, over two open commitments.
        //
        // They leave 2^4 and 2^3 candidates, and each charges one 90-bit term of its own.
        //
        // ```text
        //     outer draws      100 - (4 + 3)  ->  93 bits each
        //     opening terms    already charged by the layer that drew them  ->  90 bits
        // ```
        //
        // A prover picks one trace from each list after seeing both outer challenges.
        //
        // So an outer draw faces the product of the two.
        let outer = alloc::vec![
            SecurityTerm::new("air", ErrorBits::from_log2(100.0)),
            SecurityTerm::new("lookup", ErrorBits::from_log2(100.0)),
        ];
        let mut builder = ReportBuilder::new(outer);
        builder.add_opening_evidence(
            CommittedTrace::Main,
            Some(PrescribedOpeningSecurity::single(
                "opening",
                ErrorBits::from_log2(90.0),
                4.0,
            )),
        );
        builder.add_opening_evidence(
            CommittedTrace::Preprocessed,
            Some(PrescribedOpeningSecurity::single(
                "opening",
                ErrorBits::from_log2(90.0),
                3.0,
            )),
        );
        let report = builder.finish();
        let bits = |label, component| {
            report
                .terms()
                .iter()
                .find(|term| term.label == label && term.component == component)
                .unwrap_or_else(|| panic!("missing {label}"))
                .bits
                .bits()
        };
        assert_eq!(bits("air", None), 93.0);
        assert_eq!(bits("lookup", None), 93.0);

        // The openings charged their own reductions, so this layer leaves them alone.
        //
        // Charging them again would report 83 bits for errors that are worth 90.
        //
        // Nothing downstream could tell the difference.
        assert_eq!(bits("opening", Some("main-pcs")), 90.0);
        assert_eq!(bits("opening", Some("preprocessed-pcs")), 90.0);
    }

    #[test]
    fn an_outer_draw_drowned_by_the_candidate_set_reports_no_security_at_all() {
        // A draw three bits weaker than the list it is charged over is not a bound.
        //
        // The floor stops it at zero bits rather than letting it go negative.
        //
        // A union holding a zero-bit term composes to zero, leaving no margin to read.
        //
        // Grading the report against any positive target therefore fails.
        let outer = alloc::vec![SecurityTerm::new("air", ErrorBits::from_log2(5.0))];
        let mut builder = ReportBuilder::new(outer);
        builder.add_opening_evidence(
            CommittedTrace::Main,
            Some(PrescribedOpeningSecurity::single(
                "opening",
                ErrorBits::from_log2(128.0),
                8.0,
            )),
        );
        builder.add_collision_cap(Some(ErrorBits::from_log2(128.0)));

        let report = builder.finish();
        assert_eq!(report.terms()[0].bits.bits(), 0.0);
        assert_eq!(report.security_bits(), Some(0.0));
        assert!(matches!(
            report.require_security(1),
            Err(SecurityError::InsufficientSecurity { .. })
        ));
    }

    #[test]
    fn the_only_term_that_settles_without_a_charge_is_the_collision_cap() {
        // A term reaches the report by three routes, and each fixes its own charge.
        //
        // An outer draw goes on the list that is charged when the builder closes.
        //
        // An opening's terms arrive charged already, by the layer that drew them.
        //
        // The cap has a method of its own that fixes the label, so nothing else fits.
        //
        // A new outer draw therefore has nowhere to land at full strength by accident.
        //
        // Pushing one straight onto the settled list does not compile from out here.
        //
        // Naming it as a commitment does not either, the names being a closed set.
        let mut builder = ReportBuilder::new(Vec::new());
        builder.add_outer_draws([SecurityTerm::new("air", ErrorBits::from_log2(100.0))]);
        builder.add_opening_evidence(
            CommittedTrace::Main,
            Some(PrescribedOpeningSecurity::single(
                "opening",
                ErrorBits::from_log2(128.0),
                8.0,
            )),
        );
        builder.add_collision_cap(Some(ErrorBits::from_log2(128.0)));

        let report = builder.finish();
        let bits = |label| {
            report
                .terms()
                .iter()
                .find(|term| term.label == label)
                .unwrap_or_else(|| panic!("missing {label}"))
                .bits
                .bits()
        };

        // The outer draw paid for every candidate the opening left open.
        assert_eq!(bits("air"), 92.0);

        // The opening's own term was charged by its own layer, so it stands as given.
        assert_eq!(bits("opening"), 128.0);

        // A collision is available before the prover commits, so no set applies to it.
        assert_eq!(bits(COLLISION_CAP_LABEL), 128.0);
    }

    #[test]
    fn a_commitment_that_names_one_trace_costs_the_outer_draws_nothing() {
        // Unique decoding leaves one candidate, so an outer draw keeps its own strength.
        let outer = alloc::vec![SecurityTerm::new("air", ErrorBits::from_log2(100.0))];
        let mut builder = ReportBuilder::new(outer);
        builder.add_opening_evidence(
            CommittedTrace::Main,
            Some(PrescribedOpeningSecurity::single(
                "opening",
                ErrorBits::from_log2(90.0),
                0.0,
            )),
        );
        assert_eq!(builder.finish().terms()[0].bits.bits(), 100.0);
    }
}
