//! Initial statement handling for the WHIR protocol.
//!
//! This module manages the polynomial and constraint system during the initial
//! phase of the protocol. It supports two execution strategies:
//!
//! - **Classic**: Standard sumcheck with explicit constraint batching.
//!
//! - **SVO**: Split-Value Optimization for faster proving on large polynomials.
//!
//! # Overview
//!
//! The prover commits to a multilinear polynomial `p: {0,1}^n → F` and must later
//! prove evaluations at verifier-chosen points. The initial statement captures:
//!
//! 1. The polynomial `p` in evaluation form over the Boolean hypercube.
//!
//! 2. A set of evaluation constraints `p(z_i) = s_i` accumulated during the protocol.

use alloc::vec;
use alloc::vec::Vec;

use p3_field::{ExtensionField, Field, PackedValue};
use p3_multilinear_util::point::Point;
use p3_multilinear_util::poly::Poly;
use p3_util::log2_strict_usize;

use crate::constraints::statement::EqStatement;
use crate::parameters::SumcheckStrategy;
use crate::sumcheck::svo::SvoClaim;

/// Manages the polynomial and constraints during the initial phase of WHIR.
#[derive(Clone, Debug)]
pub struct InitialStatement<F: Field, EF: ExtensionField<F>> {
    /// The multilinear polynomial in evaluation form over `{0,1}^n`.
    ///
    /// Stores `2^n` evaluations on the Boolean hypercube.
    pub poly: Poly<F>,

    /// Inner constraint state, either classic or SVO-based.
    pub inner: InitialStatementInner<F, EF>,
}

/// Inner state for constraint tracking.
///
/// Allows switching between two constraint representations:
///
/// - **Classic**: Batches equality constraints explicitly. Each constraint `p(z_i) = s_i`
///   is stored as a (point, value) pair and combined using a random challenge.
///
/// - **SVO**: Exploits the structure of the equality polynomial for faster proving.
///   Pre-computes accumulators to avoid materializing large intermediate tables.
///   Only beneficial when the polynomial is large enough.
#[derive(Clone, Debug)]
pub enum InitialStatementInner<F: Field, EF: ExtensionField<F>> {
    /// Classic sumcheck representation.
    ///
    /// Stores evaluation constraints as explicit (point, value) pairs.
    Classic(EqStatement<EF>),

    /// Split-Value Optimization representation.
    ///
    /// Uses pre-computed accumulators for faster sumcheck proving.
    Svo {
        /// Folding factor for the first SVO round.
        l0: usize,

        /// Split equality polynomials for each constraint.
        statement: Vec<SvoClaim<F, EF>>,
    },
}

impl<F: Field, EF: ExtensionField<F>> InitialStatement<F, EF> {
    /// Creates a statement using the classic sumcheck strategy.
    ///
    /// Initializes an empty constraint set that accumulates as the protocol progresses.
    const fn new_classic(poly: Poly<F>) -> Self {
        let num_variables = poly.num_variables();
        Self {
            poly,
            inner: InitialStatementInner::new_classic(num_variables),
        }
    }

    /// Creates a statement using the SVO strategy.
    ///
    /// Initializes an empty SVO state that accumulates split equality constraints.
    const fn new_svo(poly: Poly<F>, l0: usize) -> Self {
        Self {
            poly,
            inner: InitialStatementInner::new_svo(l0),
        }
    }

    /// Creates a new initial statement with the specified mode.
    ///
    /// Automatically selects the appropriate internal representation based on the
    /// mode and polynomial size. For SVO, falls back to classic if the polynomial
    /// is too small to benefit from the optimization.
    ///
    /// # SVO Requirements
    ///
    /// The SVO mode requires:
    ///
    /// ```text
    /// k > 2 * log2(SIMD_WIDTH) + l0
    /// ```
    ///
    /// where `k` is the number of variables. This ensures enough parallelism
    /// for packed field operations.
    #[must_use]
    pub const fn new(poly: Poly<F>, l0: usize, mode: SumcheckStrategy) -> Self {
        match mode {
            // Classic path: always available.
            SumcheckStrategy::Classic => Self::new_classic(poly),
            SumcheckStrategy::Svo => {
                // SVO is only worthwhile above the packing threshold.
                let k = poly.num_variables();
                if k > 2 * log2_strict_usize(F::Packing::WIDTH) + l0 {
                    Self::new_svo(poly, l0)
                } else {
                    // Too small for SVO: silently use the scalar path.
                    Self::new_classic(poly)
                }
            }
        }
    }

    /// Evaluates the polynomial at the given point and records the constraint.
    ///
    /// Steps:
    ///
    /// 1. Computes `p(point)` using multilinear interpolation.
    ///
    /// 2. Records the constraint `p(point) = eval` in the inner state.
    ///
    /// 3. Returns the computed evaluation.
    ///
    /// - For classic strategy, adds an explicit constraint.
    /// - For SVO, creates a split equality polynomial with pre-computed accumulators.
    ///
    /// # Panics
    ///
    /// Panics if the point's dimension doesn't match the polynomial's.
    #[must_use]
    pub fn evaluate(&mut self, point: &Point<EF>) -> EF {
        assert_eq!(
            point.num_variables(),
            self.num_variables(),
            "Point has {} variables but statement expects {}",
            point.num_variables(),
            self.num_variables()
        );
        self.inner.evaluate(point, &self.poly)
    }

    /// Returns the number of variables in the polynomial.
    ///
    /// This is `n` where the polynomial is defined over `{0,1}^n`.
    pub const fn num_variables(&self) -> usize {
        self.poly.num_variables()
    }

    /// Returns true if no constraints have been added yet.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        match &self.inner {
            InitialStatementInner::Classic(statement) => statement.is_empty(),
            InitialStatementInner::Svo { statement, .. } => statement.is_empty(),
        }
    }

    /// Returns the number of constraints currently recorded.
    #[must_use]
    pub const fn len(&self) -> usize {
        match &self.inner {
            InitialStatementInner::Classic(statement) => statement.len(),
            InitialStatementInner::Svo { statement, .. } => statement.len(),
        }
    }

    /// Converts the statement to a normalized equality statement representation.
    ///
    /// - For classic statements, returns a clone of the inner state.
    /// - For SVO statements, extracts the (point, evaluation) pairs from each
    ///   split equality polynomial.
    ///
    /// Useful for verification or when the explicit constraint form is needed.
    #[must_use]
    pub fn normalize(&self) -> EqStatement<EF> {
        match &self.inner {
            InitialStatementInner::Classic(statement) => statement.clone(),
            InitialStatementInner::Svo { statement, .. } => {
                let points: Vec<_> = statement.iter().map(SvoClaim::original).cloned().collect();
                let evals: Vec<_> = statement.iter().map(SvoClaim::eval).collect();

                let mut statement = EqStatement::initialize(self.num_variables());
                points
                    .iter()
                    .cloned()
                    .zip(evals.iter())
                    .for_each(|(point, &ev)| statement.add_evaluated_constraint(point, ev));
                statement
            }
        }
    }
}

impl<F: Field, EF: ExtensionField<F>> InitialStatementInner<F, EF> {
    /// Creates a new classic inner state with no constraints.
    const fn new_classic(num_variables: usize) -> Self {
        Self::Classic(EqStatement::initialize(num_variables))
    }

    /// Creates a new SVO inner state with no constraints.
    const fn new_svo(l0: usize) -> Self {
        Self::Svo {
            statement: vec![],
            l0,
        }
    }

    /// Evaluates the polynomial and records the constraint.
    ///
    /// - For classic: computes the evaluation and adds it to the constraint set.
    /// - For SVO: creates a split equality polynomial with pre-computed accumulators.
    pub(crate) fn evaluate(&mut self, point: &Point<EF>, poly: &Poly<F>) -> EF {
        match self {
            Self::Classic(statement) => {
                let eval = poly.eval_base(point);
                statement.add_evaluated_constraint(point.clone(), eval);
                eval
            }
            Self::Svo { statement, l0 } => {
                let claim = SvoClaim::new(point, *l0, poly);
                let eval = claim.eval();
                statement.push(claim);
                eval
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use p3_baby_bear::BabyBear;
    use p3_field::PrimeCharacteristicRing;
    use p3_field::extension::BinomialExtensionField;

    use super::*;

    type F = BabyBear;
    type EF = BinomialExtensionField<F, 4>;

    /// Creates a simple test polynomial with known evaluations.
    ///
    /// Returns p(x_0, x_1) with evaluations:
    /// p(0,0)=1, p(0,1)=2, p(1,0)=3, p(1,1)=4
    fn make_test_poly() -> Poly<F> {
        Poly::new(vec![
            F::from_u64(1),
            F::from_u64(2),
            F::from_u64(3),
            F::from_u64(4),
        ])
    }

    /// Creates a larger test polynomial for SVO testing.
    ///
    /// Returns a polynomial with 2^16 evaluations (16 variables).
    fn make_large_poly() -> Poly<F> {
        let evals: Vec<F> = (0..(1 << 16)).map(|i| F::from_u64(i as u64)).collect();
        Poly::new(evals)
    }

    #[test]
    fn test_new_classic_strategy() {
        let poly = make_test_poly();
        let statement = InitialStatement::<F, EF>::new(poly, 2, SumcheckStrategy::Classic);

        assert_eq!(statement.num_variables(), 2);
        assert_eq!(statement.poly.num_evals(), 4);
        assert!(matches!(statement.inner, InitialStatementInner::Classic(_)));
    }

    #[test]
    fn test_new_svo_strategy_fallback_to_classic() {
        // Small polynomial should fallback to classic even with SVO strategy.
        let poly = make_test_poly();
        let statement = InitialStatement::<F, EF>::new(poly, 2, SumcheckStrategy::Svo);

        assert!(matches!(statement.inner, InitialStatementInner::Classic(_)));

        let normalized = statement.normalize();
        assert!(normalized.is_empty());
    }

    #[test]
    fn test_new_svo_strategy_with_large_poly() {
        let poly = make_large_poly();
        let statement = InitialStatement::<F, EF>::new(poly, 4, SumcheckStrategy::Svo);

        assert!(matches!(statement.inner, InitialStatementInner::Svo { .. }));
        assert_eq!(statement.num_variables(), 16);
    }

    #[test]
    fn test_evaluate_classic() {
        let poly = make_test_poly();
        let mut statement = InitialStatement::<F, EF>::new(poly, 2, SumcheckStrategy::Classic);

        // Evaluate at Boolean point (0, 1).
        let point = Point::new(vec![EF::ZERO, EF::ONE]);
        let eval = statement.evaluate(&point);

        // p(0, 1) = 2
        assert_eq!(eval, EF::from_u64(2));
        assert_eq!(statement.normalize().len(), 1);
    }

    #[test]
    fn test_evaluate_extension_point() {
        let poly = make_test_poly();
        let mut statement =
            InitialStatement::<F, EF>::new(poly.clone(), 2, SumcheckStrategy::Classic);

        // Evaluate at non-Boolean extension field point.
        let point = Point::new(vec![EF::from_u64(3), EF::from_u64(7)]);
        let eval = statement.evaluate(&point);

        let expected = poly.eval_base(&point);
        assert_eq!(eval, expected);
    }

    #[test]
    fn test_multiple_evaluations() {
        let poly = make_test_poly();
        let mut statement =
            InitialStatement::<F, EF>::new(poly.clone(), 2, SumcheckStrategy::Classic);

        let point1 = Point::new(vec![EF::ZERO, EF::ZERO]);
        let point2 = Point::new(vec![EF::ONE, EF::ONE]);
        let point3 = Point::new(vec![EF::from_u64(5), EF::from_u64(7)]);

        let eval1 = statement.evaluate(&point1);
        let eval2 = statement.evaluate(&point2);
        let eval3 = statement.evaluate(&point3);

        assert_eq!(eval1, EF::from_u64(1)); // p(0, 0) = 1
        assert_eq!(eval2, EF::from_u64(4)); // p(1, 1) = 4
        assert_eq!(eval3, poly.eval_base(&point3));
        assert_eq!(statement.normalize().len(), 3);
    }

    #[test]
    #[should_panic(expected = "Point has 3 variables but statement expects 2")]
    fn test_evaluate_wrong_variable_count() {
        let poly = make_test_poly();
        let mut statement = InitialStatement::<F, EF>::new(poly, 2, SumcheckStrategy::Classic);

        let wrong_point = Point::new(vec![EF::ONE, EF::ZERO, EF::ONE]);
        let _ = statement.evaluate(&wrong_point);
    }

    #[test]
    fn test_normalize_classic() {
        let poly = make_test_poly();
        let mut statement = InitialStatement::<F, EF>::new(poly, 2, SumcheckStrategy::Classic);

        let point1 = Point::new(vec![EF::ZERO, EF::ONE]);
        let point2 = Point::new(vec![EF::ONE, EF::ZERO]);
        let eval1 = statement.evaluate(&point1);
        let eval2 = statement.evaluate(&point2);

        let normalized = statement.normalize();
        assert_eq!(normalized.len(), 2);
        assert_eq!(normalized.num_variables(), 2);

        let (stored_point1, &stored_eval1) = normalized.iter().next().unwrap();
        assert_eq!(*stored_point1, point1);
        assert_eq!(stored_eval1, eval1);

        let (stored_point2, &stored_eval2) = normalized.iter().nth(1).unwrap();
        assert_eq!(*stored_point2, point2);
        assert_eq!(stored_eval2, eval2);
    }

    #[test]
    fn test_normalize_svo() {
        let poly = make_large_poly();
        let mut statement = InitialStatement::<F, EF>::new(poly, 4, SumcheckStrategy::Svo);

        let point = Point::new((0..16).map(|i| EF::from_u64(i as u64)).collect());
        let eval = statement.evaluate(&point);

        let normalized = statement.normalize();
        assert_eq!(normalized.len(), 1);

        let (stored_point, &stored_eval) = normalized.iter().next().unwrap();
        assert_eq!(*stored_point, point);
        assert_eq!(stored_eval, eval);
    }

    #[test]
    fn test_num_variables() {
        let poly = make_test_poly();
        let statement = InitialStatement::<F, EF>::new(poly, 2, SumcheckStrategy::Classic);
        assert_eq!(statement.num_variables(), 2);

        let large_poly = make_large_poly();
        let large_statement = InitialStatement::<F, EF>::new(large_poly, 4, SumcheckStrategy::Svo);
        assert_eq!(large_statement.num_variables(), 16);
    }

    #[test]
    fn test_inner_classic_new() {
        let inner = InitialStatementInner::<F, EF>::new_classic(3);

        if let InitialStatementInner::Classic(statement) = inner {
            assert_eq!(statement.num_variables(), 3);
            assert_eq!(statement.len(), 0);
        } else {
            panic!("Expected Classic variant");
        }
    }

    #[test]
    fn test_inner_svo_new() {
        let inner = InitialStatementInner::<F, EF>::new_svo(5);

        if let InitialStatementInner::Svo { l0, statement } = inner {
            assert_eq!(l0, 5);
            assert!(statement.is_empty());
        } else {
            panic!("Expected Svo variant");
        }
    }

    #[test]
    fn test_inner_evaluate_classic() {
        let poly = make_test_poly();
        let mut inner = InitialStatementInner::<F, EF>::new_classic(2);

        let point = Point::new(vec![EF::ONE, EF::ZERO]);
        let eval = inner.evaluate(&point, &poly);

        // p(1, 0) = 3
        assert_eq!(eval, EF::from_u64(3));

        if let InitialStatementInner::Classic(statement) = &inner {
            assert_eq!(statement.len(), 1);
        }
    }

    #[test]
    fn test_svo_creates_split_eq() {
        let poly = make_large_poly();
        let mut inner = InitialStatementInner::<F, EF>::new_svo(4);

        let point = Point::new((0..16).map(|_| EF::from_u64(1)).collect());
        let eval = inner.evaluate(&point, &poly);

        if let InitialStatementInner::Svo { statement, .. } = &inner {
            assert_eq!(statement.len(), 1);
            assert_eq!(statement[0].eval(), eval);
        }
    }
}
