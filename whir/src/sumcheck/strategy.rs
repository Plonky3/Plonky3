//! Sumcheck prover: constructs and executes the sumcheck protocol for multilinear polynomials.

use p3_challenger::{FieldChallenger, GrindingChallenger};
use p3_field::{Algebra, ExtensionField, Field, PrimeCharacteristicRing};
use p3_maybe_rayon::prelude::*;
use p3_multilinear_util::point::Point;
use p3_multilinear_util::poly::Poly;

use crate::constraints::Constraint;
use crate::sumcheck::SumcheckData;
use crate::sumcheck::product_polynomial::ProductPolynomial;

/// Computes the sumcheck round polynomial coefficients `(h(0), h(inf))` for the product
/// of two multilinear polynomials of the same type.
///
/// Given two multilinear polynomials `evals` and `weights` of the same size,
/// computes the univariate polynomial h(X) = sum_{b in {0,1}^{n-1}} evals(X, b) * weights(X, b).
///
/// Returns `(h(0), h(inf))`:
/// - h(0) = sum_{b} evals(0, b) * weights(0, b)
/// - h(inf) = sum_{b} (evals(1,b) - evals(0,b)) * (weights(1,b) - weights(0,b))
///
/// Mixed-type variant of sumcheck coefficients where the evaluation polynomial
/// is over a base type `B` and the weight polynomial is over an algebra type `A`
/// that contains `B` (e.g., base field evals with extension field weights).
pub fn sumcheck_coefficients_prefix<B, A>(evals: &[B], weights: &[A]) -> (A, A)
where
    B: PrimeCharacteristicRing + Copy + Send + Sync,
    A: p3_field::Algebra<B> + Copy + Send + Sync,
{
    assert_eq!(evals.len(), weights.len());
    let half = evals.len() / 2;
    let (e_lo, e_hi) = evals.split_at(half);
    let (w_lo, w_hi) = weights.split_at(half);

    if evals.len() > (1 << 14) {
        e_lo.par_iter()
            .zip(e_hi.par_iter())
            .zip(w_lo.par_iter().zip(w_hi.par_iter()))
            .par_fold_reduce(
                || (A::ZERO, A::ZERO),
                |(acc0, acc_inf), ((&e0, &e1), (&w0, &w1))| {
                    (acc0 + w0 * e0, acc_inf + (w1 - w0) * (e1 - e0))
                },
                |(acc0, acc_inf), (val0, val_inf)| (acc0 + val0, acc_inf + val_inf),
            )
    } else {
        e_lo.iter()
            .zip(e_hi.iter())
            .zip(w_lo.iter().zip(w_hi.iter()))
            .fold(
                (A::ZERO, A::ZERO),
                |(acc0, acc_inf), ((&e0, &e1), (&w0, &w1))| {
                    (acc0 + w0 * e0, acc_inf + (w1 - w0) * (e1 - e0))
                },
            )
    }
}

/// Computes the sumcheck round polynomial coefficients `(h(0), h(inf))` for the product
/// of two multilinear polynomials where the suffix variable is the round variable.
///
/// Given two multilinear polynomials `evals` and `weights` of the same size,
/// computes the univariate polynomial
/// h(X) = sum_{b in {0,1}^{n-1}} evals(b, X) * weights(b, X).
///
/// Returns `(h(0), h(inf))`:
/// - h(0) = sum_{b} evals(b, 0) * weights(b, 0)
/// - h(inf) = sum_{b} (evals(b,1) - evals(b,0)) * (weights(b,1) - weights(b,0))
///
/// Mixed-type variant of sumcheck coefficients where the evaluation polynomial
/// is over a base type `B` and the weight polynomial is over an algebra type `A`
/// that contains `B` (e.g., base field evals with extension field weights).
pub fn sumcheck_coefficients_suffix<B, A>(evals: &[B], weights: &[A]) -> (A, A)
where
    B: PrimeCharacteristicRing + Copy + Send + Sync,
    A: p3_field::Algebra<B> + Copy + Send + Sync,
{
    assert_eq!(evals.len(), weights.len());

    if evals.len() > (1 << 14) {
        evals
            .par_chunks(2)
            .zip(weights.par_chunks(2))
            .par_fold_reduce(
                || (A::ZERO, A::ZERO),
                |(acc0, acc_inf), (e, w)| {
                    (acc0 + w[0] * e[0], acc_inf + (w[1] - w[0]) * (e[1] - e[0]))
                },
                |(acc0, acc_inf), (val0, val_inf)| (acc0 + val0, acc_inf + val_inf),
            )
    } else {
        evals
            .chunks(2)
            .zip(weights.chunks(2))
            .fold((A::ZERO, A::ZERO), |(acc0, acc_inf), (e, w)| {
                (acc0 + w[0] * e[0], acc_inf + (w[1] - w[0]) * (e[1] - e[0]))
            })
    }
}

/// Strategy hooks for choosing variable order and residual constraint evaluation.
pub trait SumcheckStrategy {
    /// Computes `(h(0), h(inf))` for one quadratic sumcheck round.
    fn sumcheck_coefficients<B, A>(evals: &[B], weights: &[A]) -> (A, A)
    where
        B: PrimeCharacteristicRing + Copy + Send + Sync,
        A: Algebra<B> + Copy + Send + Sync;

    /// Binds the active round variable of `poly` to challenge `r`.
    fn fix_var<A: Algebra<Challenge> + Copy + Send + Sync, Challenge: Copy + Send + Sync>(
        poly: &mut Poly<A>,
        r: Challenge,
    );

    /// Evaluates the batched verifier constraints at the final challenge point.
    fn eval_constraints_poly<F: Field, EF: ExtensionField<F>>(
        constraints: &[Constraint<F, EF>],
        point: &Point<EF>,
    ) -> EF;

    /// Returns the variable order used by this sumcheck strategy.
    fn var_order() -> VariableOrder;
}

/// Which side of the variable order is consumed first by a strategy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VariableOrder {
    /// Prefix variables are bound first.
    Prefix,
    /// Suffix variables are bound first.
    Suffix,
}

/// Sumcheck strategy that binds suffix variables first.
#[derive(Debug, Clone, Default)]
pub struct SuffixSumcheck;

impl SumcheckStrategy for SuffixSumcheck {
    fn sumcheck_coefficients<B, A>(evals: &[B], weights: &[A]) -> (A, A)
    where
        B: PrimeCharacteristicRing + Copy + Send + Sync,
        A: p3_field::Algebra<B> + Copy + Send + Sync,
    {
        sumcheck_coefficients_suffix(evals, weights)
    }

    fn fix_var<A: Algebra<Challenge> + Copy + Send + Sync, Challenge: Copy + Send + Sync>(
        poly: &mut Poly<A>,
        r: Challenge,
    ) {
        poly.fix_suffix_var_mut(r);
    }

    fn eval_constraints_poly<F: Field, EF: ExtensionField<F>>(
        constraints: &[Constraint<F, EF>],
        challenge: &Point<EF>,
    ) -> EF {
        constraints
            .iter()
            .map(|constraint| {
                let challenge = challenge
                    .reversed()
                    .get_subpoint_over_range(..constraint.num_variables());

                let eq_contrib = constraint
                    .iter_eqs()
                    .map(|(point, coeff)| coeff * point.eq_poly(&challenge))
                    .sum::<EF>();
                let sel_contrib = constraint
                    .iter_sels()
                    .map(|(&var, coeff)| coeff * challenge.select_poly(var))
                    .sum::<EF>();
                eq_contrib + sel_contrib
            })
            .sum()
    }

    fn var_order() -> VariableOrder {
        VariableOrder::Suffix
    }
}

/// Sumcheck strategy that binds prefix variables first.
#[derive(Debug, Clone, Default)]
pub struct PrefixSumcheck;

impl SumcheckStrategy for PrefixSumcheck {
    fn sumcheck_coefficients<B, A>(evals: &[B], weights: &[A]) -> (A, A)
    where
        B: PrimeCharacteristicRing + Copy + Send + Sync,
        A: p3_field::Algebra<B> + Copy + Send + Sync,
    {
        sumcheck_coefficients_prefix(evals, weights)
    }

    fn fix_var<A: Algebra<Challenge> + Copy + Send + Sync, Challenge: Copy + Send + Sync>(
        poly: &mut Poly<A>,
        r: Challenge,
    ) {
        poly.fix_prefix_var_mut(r);
    }

    fn eval_constraints_poly<F: Field, EF: ExtensionField<F>>(
        constraints: &[Constraint<F, EF>],
        challenge: &Point<EF>,
    ) -> EF {
        constraints
            .iter()
            .map(|constraint| {
                let challenge = challenge
                    .reversed()
                    .get_subpoint_over_range(0..constraint.num_variables())
                    .reversed();

                let eq_contrib = constraint
                    .iter_eqs()
                    .map(|(point, coeff)| coeff * point.eq_poly(&challenge))
                    .sum::<EF>();
                let sel_contrib = constraint
                    .iter_sels()
                    .map(|(&var, coeff)| coeff * challenge.select_poly(var))
                    .sum::<EF>();
                eq_contrib + sel_contrib
            })
            .sum()
    }

    fn var_order() -> VariableOrder {
        VariableOrder::Prefix
    }
}

/// Prover state for the sumcheck protocol over a multilinear polynomial.
///
/// Holds the partially-folded polynomial pair `(f, w)` and the current claimed sum.
///
/// # Invariant
///
/// At every point during the protocol:
///
/// ```text
/// sum == sum_{x in {0,1}^n} f(x) * w(x)
/// ```
///
/// where `n` is the number of remaining unbound variables.
/// It decreases by one per round as variables are bound to verifier challenges.
#[derive(Debug, Clone)]
pub struct SumcheckProver<F: Field, EF: ExtensionField<F>, St: SumcheckStrategy> {
    /// Paired evaluation and weight polynomials for the quadratic sumcheck.
    ///
    /// Stores both `f(x)` and `w(x)` in either SIMD-packed or scalar format.
    /// The format is chosen automatically based on polynomial size.
    poly: ProductPolynomial<F, EF, St>,

    /// Current claimed sum for the remaining variables.
    ///
    /// Tracks `sum_{x in {0,1}^n} f(x) * w(x)`.
    ///
    /// After each round binding variable `X_i` to challenge `r_i`, updated via:
    ///
    /// ```text
    /// sum := h(r_i)  where  h(X) = c_0 + c_1 * X + c_2 * X^2
    /// ```
    sum: EF,
}

impl<F, EF, St> SumcheckProver<F, EF, St>
where
    F: Field,
    EF: ExtensionField<F>,
    St: SumcheckStrategy,
{
    /// Creates a prover state from a product polynomial and its claimed sum.
    pub fn new(poly: ProductPolynomial<F, EF, St>, sum: EF) -> Self {
        debug_assert_eq!(poly.dot_product(), sum);
        Self { poly, sum }
    }

    /// Returns the current evaluation polynomial as scalar extension-field values.
    pub fn poly(&self) -> Poly<EF> {
        self.poly.evals()
    }

    /// Returns the number of remaining (unbound) variables.
    ///
    /// Decreases by one per round.
    /// Reaches zero when the polynomial is fully evaluated at a single point.
    pub fn num_vars(&self) -> usize {
        self.poly.num_vars()
    }

    /// Extracts the current evaluation polynomial as scalar extension field elements.
    ///
    /// If the internal representation is SIMD-packed, unpacks all lanes first.
    #[tracing::instrument(skip_all)]
    pub fn evals(&self) -> Poly<EF> {
        self.poly.evals()
    }

    /// Evaluates `f` at a given multilinear point via interpolation.
    ///
    /// The weight polynomial is not involved in this evaluation.
    pub fn eval(&self, point: &Point<EF>) -> EF {
        self.poly.eval(point)
    }

    /// Runs additional sumcheck rounds, optionally incorporating new constraints.
    ///
    /// Two phases:
    ///
    /// 1. **Constraint folding** (optional):
    ///    If a constraint is provided, fold it into the weight polynomial
    ///    and update the claimed sum before any rounds.
    ///    Used to incorporate STIR challenges between batches.
    ///
    /// 2. **Round execution**:
    ///    Performs `folding_factor` rounds of one-variable-per-round sumcheck.
    ///    Each round computes `(c_0, c_2)`, commits, receives a challenge, and folds.
    ///
    /// # Returns
    ///
    /// The verifier challenges `(r_1, ..., r_{folding_factor})` from this batch.
    ///
    /// # Panics
    ///
    /// If `folding_factor` exceeds the current number of remaining variables.
    #[tracing::instrument(skip_all)]
    pub fn compute_sumcheck_polynomials<Challenger>(
        &mut self,
        sumcheck_data: &mut SumcheckData<F, EF>,
        challenger: &mut Challenger,
        folding_factor: usize,
        pow_bits: usize,
        constraint: Option<Constraint<F, EF>>,
    ) -> Point<EF>
    where
        Challenger: FieldChallenger<F> + GrindingChallenger<Witness = F>,
    {
        // If a new constraint is provided, fold it into the weight polynomial
        // and update the claimed sum before starting rounds.
        if let Some(constraint) = constraint {
            self.poly.combine(&mut self.sum, &constraint);
        }

        // Execute rounds of the standard sumcheck protocol.
        // Each call computes coefficients, commits, receives a challenge, and folds.
        let res = (0..folding_factor)
            .map(|_| {
                self.poly
                    .round(sumcheck_data, challenger, &mut self.sum, pow_bits)
            })
            .collect();

        // Return the collected verifier challenges.
        Point::new(res)
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec::Vec;

    use p3_baby_bear::BabyBear;
    use p3_field::PrimeCharacteristicRing;
    use p3_field::extension::BinomialExtensionField;
    use p3_multilinear_util::point::Point;
    use p3_multilinear_util::poly::Poly;
    use proptest::prelude::*;
    use rand::rngs::SmallRng;
    use rand::{RngExt, SeedableRng};

    use super::{PrefixSumcheck, SuffixSumcheck, SumcheckStrategy};
    use crate::constraints::Constraint;
    use crate::constraints::statement::{EqStatement, SelectStatement};

    type F = BabyBear;
    type EF = BinomialExtensionField<BabyBear, 4>;

    fn eval_constraints_poly_reference<St: SumcheckStrategy>(
        constraints: &[Constraint<F, EF>],
        challenge: &Point<EF>,
    ) -> EF {
        constraints
            .iter()
            .map(|constraint| {
                let mut combined = Poly::zero(constraint.num_variables());
                let mut eval = EF::ZERO;
                constraint.combine(&mut combined, &mut eval);

                let point = match St::var_order() {
                    super::VariableOrder::Prefix => challenge
                        .reversed()
                        .get_subpoint_over_range(..constraint.num_variables())
                        .reversed(),
                    super::VariableOrder::Suffix => challenge
                        .reversed()
                        .get_subpoint_over_range(..constraint.num_variables()),
                };

                combined.eval_ext::<F>(&point)
            })
            .sum()
    }

    fn random_constraints(
        rng: &mut SmallRng,
        num_vars: usize,
        rounds: usize,
    ) -> Vec<Constraint<F, EF>> {
        (0..rounds)
            .map(|_| {
                let num_vars = rng.random_range(1..=num_vars);
                let gamma = rng.random();

                let mut eq_statement = EqStatement::initialize(num_vars);
                (0..rng.random_range(0..=3)).for_each(|_| {
                    eq_statement.add_evaluated_constraint(Point::rand(rng, num_vars), rng.random());
                });

                let mut sel_statement = SelectStatement::<F, EF>::initialize(num_vars);
                (0..rng.random_range(0..=3))
                    .for_each(|_| sel_statement.add_constraint(rng.random(), rng.random()));

                Constraint::new(gamma, eq_statement, sel_statement)
            })
            .collect()
    }

    #[test]
    fn test_eval_constraints_poly_prefix() {
        let mut rng = SmallRng::seed_from_u64(0);
        let constraints = random_constraints(&mut rng, 20, 6);
        let challenge = Point::rand(&mut rng, 20);

        let got = PrefixSumcheck::eval_constraints_poly(&constraints, &challenge);
        let expected = eval_constraints_poly_reference::<PrefixSumcheck>(&constraints, &challenge);

        assert_eq!(got, expected);
    }

    #[test]
    fn test_eval_constraints_poly_suffix() {
        let mut rng = SmallRng::seed_from_u64(1);
        let constraints = random_constraints(&mut rng, 20, 6);
        let challenge = Point::rand(&mut rng, 20);

        let got = SuffixSumcheck::eval_constraints_poly(&constraints, &challenge);
        let expected = eval_constraints_poly_reference::<SuffixSumcheck>(&constraints, &challenge);

        assert_eq!(got, expected);
    }

    proptest! {
        #[test]
        fn prop_eval_constraints_poly_matches_reference(
            total_num_vars in 2usize..=20,
            rounds in 1usize..=8,
            seed in any::<u64>(),
        ) {
            let mut rng = SmallRng::seed_from_u64(seed);
            let constraints = random_constraints(&mut rng, total_num_vars, rounds);
            let challenge = Point::rand(&mut rng, total_num_vars);

            prop_assert_eq!(
                PrefixSumcheck::eval_constraints_poly(&constraints, &challenge),
                eval_constraints_poly_reference::<PrefixSumcheck>(&constraints, &challenge),
            );

            prop_assert_eq!(
                SuffixSumcheck::eval_constraints_poly(&constraints, &challenge),
                eval_constraints_poly_reference::<SuffixSumcheck>(&constraints, &challenge),
            );
        }
    }
}
