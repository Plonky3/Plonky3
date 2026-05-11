//! Sumcheck helpers: variable ordering, round coefficients, and the prover state.
//!
//! # Layout
//!
//! - `sumcheck_coefficients_{prefix,suffix}`: the two round-coefficient routines.
//! - `VariableOrder`: tag enum carrying inherent methods that dispatch to either routine.
//! - `SumcheckProver`: drives rounds over a paired product polynomial.

use p3_challenger::{FieldChallenger, GrindingChallenger};
use p3_field::{Algebra, ExtensionField, Field, PrimeCharacteristicRing};
use p3_maybe_rayon::prelude::*;
use p3_multilinear_util::point::Point;
use p3_multilinear_util::poly::Poly;

use crate::constraints::Constraint;
use crate::sumcheck::SumcheckData;
use crate::sumcheck::product_polynomial::ProductPolynomial;

/// Input size at which the round-coefficient routines switch from serial to parallel execution.
///
/// # Why this value
///
/// - Below `2^14` paired elements, the rayon splitting and join overhead outweighs the parallel work.
/// - Above it, the fold-reduce amortises the splitting cost.
const PAR_THRESHOLD: usize = 1 << 14;

/// Tile size for the chunked round-coefficient kernel.
///
/// On Monty-31 packings, hand-written delayed-reduction primitives exist for tile sizes `2, 4, 5, 8`;
///
/// `8` is the deepest available on every supported target.
///
/// - Larger overruns the integer-multiply pipeline depth;
/// - Smaller dilutes the delayed-reduction win.
const K: usize = 8;

/// Per-tile MAC: extends a `(constant, leading)` accumulator pair.
///
/// # Algorithm
///
/// Folding the active variable in `h(X) = sum_b f(X, b) * w(X, b)` gives:
///
/// ```text
///     constant += sum_i  w_lo[i] * e_lo[i]
///     leading  += sum_i  (w_hi[i] - w_lo[i]) * (e_hi[i] - e_lo[i])
/// ```
///
/// where `lo`, `hi` are the two faces of the active variable. Each sum is
/// one delayed-reduction dot product over `K` pairs, collapsing `K`
/// widening multiplies into one Montgomery reduce per output coordinate.
#[inline(always)]
fn chunk_round_step<B, A>(e_lo: &[B; K], e_hi: &[B; K], w_lo: &[A; K], w_hi: &[A; K]) -> (A, A)
where
    B: PrimeCharacteristicRing + Copy,
    A: Algebra<B> + Copy,
{
    // Constant term: one delayed-reduction dot product over the b_0 = 0 face.
    let acc0 = A::mixed_dot_product::<K>(w_lo, e_lo);

    // Materialise the differences (b_0 = 1 minus b_0 = 0) tile-locally so
    // they can feed the same primitive. `K` base subs, no reductions.
    let diffs_e: [B; K] = core::array::from_fn(|i| e_hi[i] - e_lo[i]);
    let diffs_w: [A; K] = core::array::from_fn(|i| w_hi[i] - w_lo[i]);

    // Leading coefficient: dot product of the differences.
    let acc_inf = A::mixed_dot_product::<K>(&diffs_w, &diffs_e);

    (acc0, acc_inf)
}

/// Per-pair MAC for the streaming tail (at most `K - 1` leftover pairs).
#[inline(always)]
fn round_step<B, A>((acc0, acc_inf): (A, A), e0: B, e1: B, w0: A, w1: A) -> (A, A)
where
    B: PrimeCharacteristicRing + Copy,
    A: Algebra<B> + Copy,
{
    (acc0 + w0 * e0, acc_inf + (w1 - w0) * (e1 - e0))
}

/// Component-wise sum of two `(constant, leading)` accumulator pairs.
#[inline(always)]
fn round_reduce<A: Copy + PrimeCharacteristicRing>(a: (A, A), b: (A, A)) -> (A, A) {
    (a.0 + b.0, a.1 + b.1)
}

/// Computes `(h(0), h(inf))` for a prefix-binding sumcheck round.
///
/// # Inputs
///
/// - `evals`   — multilinear evaluations of `f(X)` over the hypercube.
/// - `weights` — multilinear evaluations of `w(X)` over the hypercube.
///
/// # Returns
///
/// - `h(0)`   = sum_{b in {0,1}^{n-1}} f(0, b) * w(0, b)
/// - `h(inf)` = sum_{b} (f(1, b) - f(0, b)) * (w(1, b) - w(0, b))
///
/// # Complexity
///
/// O(2^n). Parallelised above a 2^14 threshold. The main loop is tiled by
/// `K` over a delayed-reduction dot product; the `half mod K` tail uses a
/// streaming fold.
pub fn sumcheck_coefficients_prefix<B, A>(evals: &[B], weights: &[A]) -> (A, A)
where
    B: PrimeCharacteristicRing + Copy + Send + Sync,
    A: Algebra<B> + Copy + Send + Sync,
{
    // Precondition: paired slices must be aligned; half-and-half split addresses the prefix bit.
    assert_eq!(evals.len(), weights.len());
    assert!(evals.len().is_multiple_of(2));
    let half = evals.len() / 2;
    let (e_lo, e_hi) = evals.split_at(half);
    let (w_lo, w_hi) = weights.split_at(half);

    let body = (half / K) * K;
    let (e_lo_main, e_lo_tail) = e_lo.split_at(body);
    let (e_hi_main, e_hi_tail) = e_hi.split_at(body);
    let (w_lo_main, w_lo_tail) = w_lo.split_at(body);
    let (w_hi_main, w_hi_tail) = w_hi.split_at(body);

    // Main chunked loop: K pairs per iteration via delayed-reduction dot products.
    let main: (A, A) = if half > PAR_THRESHOLD {
        e_lo_main
            .par_chunks_exact(K)
            .zip(e_hi_main.par_chunks_exact(K))
            .zip(
                w_lo_main
                    .par_chunks_exact(K)
                    .zip(w_hi_main.par_chunks_exact(K)),
            )
            .par_fold_reduce(
                || (A::ZERO, A::ZERO),
                |acc, ((e_lo_c, e_hi_c), (w_lo_c, w_hi_c))| {
                    let chunk = chunk_round_step::<B, A>(
                        e_lo_c.try_into().unwrap(),
                        e_hi_c.try_into().unwrap(),
                        w_lo_c.try_into().unwrap(),
                        w_hi_c.try_into().unwrap(),
                    );
                    round_reduce(acc, chunk)
                },
                round_reduce,
            )
    } else {
        e_lo_main
            .chunks_exact(K)
            .zip(e_hi_main.chunks_exact(K))
            .zip(w_lo_main.chunks_exact(K).zip(w_hi_main.chunks_exact(K)))
            .fold(
                (A::ZERO, A::ZERO),
                |acc, ((e_lo_c, e_hi_c), (w_lo_c, w_hi_c))| {
                    let chunk = chunk_round_step::<B, A>(
                        e_lo_c.try_into().unwrap(),
                        e_hi_c.try_into().unwrap(),
                        w_lo_c.try_into().unwrap(),
                        w_hi_c.try_into().unwrap(),
                    );
                    round_reduce(acc, chunk)
                },
            )
    };

    // Tail: at most K-1 pairs; streaming fold with eager reduction is fine.
    let tail = e_lo_tail
        .iter()
        .zip(e_hi_tail.iter())
        .zip(w_lo_tail.iter().zip(w_hi_tail.iter()))
        .fold((A::ZERO, A::ZERO), |acc, ((&e0, &e1), (&w0, &w1))| {
            round_step(acc, e0, e1, w0, w1)
        });

    round_reduce(main, tail)
}

/// Computes `(h(0), h(inf))` for a suffix-binding sumcheck round.
///
/// # Inputs
///
/// - `evals`   — multilinear evaluations of `f(X)` over the hypercube.
/// - `weights` — multilinear evaluations of `w(X)` over the hypercube.
///
/// # Returns
///
/// - `h(0)`   = sum_{b in {0,1}^{n-1}} f(b, 0) * w(b, 0)
/// - `h(inf)` = sum_{b} (f(b, 1) - f(b, 0)) * (w(b, 1) - w(b, 0))
///
/// # Complexity
///
/// O(2^n). Parallelised above a 2^14 threshold. The main loop walks the
/// buffer in `2K`-wide chunks: each chunk gathers `K` adjacent
/// `(b_n=0, b_n=1)` pairs and dispatches to a delayed-reduction dot
/// product.
pub fn sumcheck_coefficients_suffix<B, A>(evals: &[B], weights: &[A]) -> (A, A)
where
    B: PrimeCharacteristicRing + Copy + Send + Sync,
    A: Algebra<B> + Copy + Send + Sync,
{
    // Precondition: paired slices must be aligned; adjacent pairs address the suffix bit.
    assert_eq!(evals.len(), weights.len());
    assert!(evals.len().is_multiple_of(2));

    let half = evals.len() / 2;
    // Each chunk consumes 2K consecutive elements (K pairs).
    let body_pairs = (half / K) * K;
    let body_elems = body_pairs * 2;
    let (evals_main, evals_tail) = evals.split_at(body_elems);
    let (weights_main, weights_tail) = weights.split_at(body_elems);

    #[inline(always)]
    fn gather_pairs<T: Copy>(chunk: &[T]) -> ([T; K], [T; K]) {
        // Layout: [t0, t1, t2, t3, ...]; even indices = "0", odd indices = "1".
        let lo: [T; K] = core::array::from_fn(|i| chunk[2 * i]);
        let hi: [T; K] = core::array::from_fn(|i| chunk[2 * i + 1]);
        (lo, hi)
    }

    let main: (A, A) = if evals.len() > PAR_THRESHOLD {
        evals_main
            .par_chunks_exact(2 * K)
            .zip(weights_main.par_chunks_exact(2 * K))
            .par_fold_reduce(
                || (A::ZERO, A::ZERO),
                |acc, (e_chunk, w_chunk)| {
                    let (e_lo, e_hi) = gather_pairs::<B>(e_chunk);
                    let (w_lo, w_hi) = gather_pairs::<A>(w_chunk);
                    let chunk = chunk_round_step::<B, A>(&e_lo, &e_hi, &w_lo, &w_hi);
                    round_reduce(acc, chunk)
                },
                round_reduce,
            )
    } else {
        evals_main
            .chunks_exact(2 * K)
            .zip(weights_main.chunks_exact(2 * K))
            .fold((A::ZERO, A::ZERO), |acc, (e_chunk, w_chunk)| {
                let (e_lo, e_hi) = gather_pairs::<B>(e_chunk);
                let (w_lo, w_hi) = gather_pairs::<A>(w_chunk);
                let chunk = chunk_round_step::<B, A>(&e_lo, &e_hi, &w_lo, &w_hi);
                round_reduce(acc, chunk)
            })
    };

    // Tail: at most K-1 pairs; streaming fold over adjacent (0,1) chunks.
    let tail = evals_tail
        .chunks(2)
        .zip(weights_tail.chunks(2))
        .fold((A::ZERO, A::ZERO), |acc, (e, w)| {
            round_step(acc, e[0], e[1], w[0], w[1])
        });

    round_reduce(main, tail)
}

/// Which side of the variable order is bound first by the sumcheck rounds.
///
/// # Role
///
/// - Round-coefficient math differs in which axis is summed over.
/// - Variable binding differs in which coordinate is fixed to the challenge.
/// - Verifier constraint evaluation differs in how the final challenge is spliced.
///
/// All three dispatches go through inherent methods below, so the runtime
/// branch sits in the outer frame and never inside the O(2^n) inner loops.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VariableOrder {
    /// Prefix variables are bound first (round `i` binds `X_i`).
    Prefix,
    /// Suffix variables are bound first (round `i` binds `X_{n-i}`).
    Suffix,
}

impl VariableOrder {
    /// Computes `(h(0), h(inf))` for one quadratic sumcheck round.
    pub fn sumcheck_coefficients<B, A>(self, evals: &[B], weights: &[A]) -> (A, A)
    where
        B: PrimeCharacteristicRing + Copy + Send + Sync,
        A: Algebra<B> + Copy + Send + Sync,
    {
        match self {
            Self::Prefix => sumcheck_coefficients_prefix(evals, weights),
            Self::Suffix => sumcheck_coefficients_suffix(evals, weights),
        }
    }

    /// Binds the active round variable of `poly` to challenge `r`.
    pub fn fix_var<A, Ch>(self, poly: &mut Poly<A>, r: Ch)
    where
        A: Algebra<Ch> + Copy + Send + Sync,
        Ch: Copy + Send + Sync,
    {
        match self {
            Self::Prefix => poly.fix_prefix_var_mut(r),
            Self::Suffix => poly.fix_suffix_var_mut(r),
        }
    }

    /// Evaluates the batched verifier constraints at the final challenge point.
    ///
    /// # Slicing rule
    ///
    /// - Prefix binding folds variables low-to-high, so each constraint sees
    ///   the last `k` original variables of the challenge.
    /// - Suffix binding folds variables high-to-low, so each constraint sees
    ///   the last `k` original variables of the challenge, reversed.
    pub fn eval_constraints_poly<F, EF>(
        self,
        constraints: &[Constraint<F, EF>],
        challenge: &Point<EF>,
    ) -> EF
    where
        F: Field,
        EF: ExtensionField<F>,
    {
        // Reverse once outside the per-constraint loop; both branches reuse it.
        let reversed = challenge.reversed();

        constraints
            .iter()
            .map(|constraint| {
                // Slice the reversed challenge to the constraint arity; flip back for prefix binding.
                let local_challenge = match self {
                    Self::Prefix => reversed
                        .get_subpoint_over_range(..constraint.num_variables())
                        .reversed(),
                    Self::Suffix => reversed.get_subpoint_over_range(..constraint.num_variables()),
                };

                // Equality term: sum_{(z, c)} c * eq(z, local_challenge).
                let eq_contrib = constraint
                    .iter_eqs()
                    .map(|(point, coeff)| coeff * point.eq_poly(&local_challenge))
                    .sum::<EF>();
                // Selector term: sum_{(var, c)} c * select(local_challenge, var).
                let sel_contrib = constraint
                    .iter_sels()
                    .map(|(&var, coeff)| coeff * local_challenge.select_poly(var))
                    .sum::<EF>();
                eq_contrib + sel_contrib
            })
            .sum()
    }
}

/// Sumcheck prover: drives rounds of the quadratic sumcheck protocol.
///
/// # Invariant
///
/// At every point during the protocol:
///
/// ```text
///     sum == sum_{x in {0,1}^n} f(x) * w(x)
/// ```
///
/// where `n` is the number of remaining unbound variables. It decreases by
/// one per round as variables are bound to verifier challenges.
#[derive(Debug, Clone)]
pub struct SumcheckProver<F: Field, EF: ExtensionField<F>> {
    /// Paired evaluation and weight polynomials for the quadratic sumcheck.
    poly: ProductPolynomial<F, EF>,
    /// Current claimed sum over the remaining unbound variables.
    sum: EF,
}

impl<F: Field, EF: ExtensionField<F>> SumcheckProver<F, EF> {
    /// Creates a prover state from a product polynomial and its claimed sum.
    pub fn new(poly: ProductPolynomial<F, EF>, sum: EF) -> Self {
        // Sanity: the claimed sum must match the polynomial pair's dot product.
        debug_assert_eq!(poly.dot_product(), sum);
        Self { poly, sum }
    }

    /// Returns the number of remaining (unbound) variables.
    pub fn num_variables(&self) -> usize {
        self.poly.num_variables()
    }

    /// Extracts the current evaluation polynomial as scalar extension-field elements.
    #[tracing::instrument(skip_all)]
    pub fn evals(&self) -> Poly<EF> {
        self.poly.evals()
    }

    /// Evaluates `f` at a given multilinear point via interpolation.
    pub fn eval(&self, point: &Point<EF>) -> EF {
        self.poly.eval(point)
    }

    /// Runs additional sumcheck rounds, optionally incorporating a new constraint.
    ///
    /// # Phases
    ///
    /// - Constraint folding (optional): fold an extra constraint into the weight
    ///   polynomial and update the claimed sum before any rounds.
    /// - Round execution: perform `folding_factor` rounds of one-variable-per-round
    ///   sumcheck; each round emits coefficients, absorbs a challenge, and folds.
    ///
    /// # Returns
    ///
    /// The verifier challenges sampled during this batch.
    ///
    /// # Panics
    ///
    /// - Folding factor must not exceed the current number of remaining variables.
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
        // Optional constraint absorption: fold into the weight polynomial and update the sum.
        if let Some(constraint) = constraint {
            self.poly.combine(&mut self.sum, &constraint);
        }

        // Drive `folding_factor` standard rounds, collecting each round's challenge.
        let res = (0..folding_factor)
            .map(|_| {
                self.poly
                    .round(sumcheck_data, challenger, &mut self.sum, pow_bits)
            })
            .collect();

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

    use super::VariableOrder;
    use crate::constraints::Constraint;
    use crate::constraints::statement::{EqStatement, SelectStatement};

    type F = BabyBear;
    type EF = BinomialExtensionField<BabyBear, 4>;

    // Reference implementation: evaluate each constraint's combined polynomial at
    // the appropriately sliced challenge and sum. Used to cross-check the fast path.
    fn eval_constraints_poly_reference(
        order: VariableOrder,
        constraints: &[Constraint<F, EF>],
        challenge: &Point<EF>,
    ) -> EF {
        constraints
            .iter()
            .map(|constraint| {
                // Combine eq + sel contributions into one weight polynomial.
                let mut combined = Poly::zero(constraint.num_variables());
                let mut eval = EF::ZERO;
                constraint.combine(&mut combined, &mut eval);

                // Slice the challenge per binding direction; evaluate at that local point.
                let point = match order {
                    VariableOrder::Prefix => challenge
                        .reversed()
                        .get_subpoint_over_range(..constraint.num_variables())
                        .reversed(),
                    VariableOrder::Suffix => challenge
                        .reversed()
                        .get_subpoint_over_range(..constraint.num_variables()),
                };

                combined.eval_ext::<F>(&point)
            })
            .sum()
    }

    // Generates a random list of constraints for fuzzing the evaluator.
    fn random_constraints(
        rng: &mut SmallRng,
        num_variables: usize,
        rounds: usize,
    ) -> Vec<Constraint<F, EF>> {
        (0..rounds)
            .map(|_| {
                let num_variables = rng.random_range(1..=num_variables);
                let gamma = rng.random();

                // Up to 3 equality constraints at random points.
                let mut eq_statement = EqStatement::initialize(num_variables);
                (0..rng.random_range(0..=3)).for_each(|_| {
                    eq_statement
                        .add_evaluated_constraint(Point::rand(rng, num_variables), rng.random());
                });

                // Up to 3 selector constraints at random variables.
                let mut sel_statement = SelectStatement::<F, EF>::initialize(num_variables);
                (0..rng.random_range(0..=3))
                    .for_each(|_| sel_statement.add_constraint(rng.random(), rng.random()));

                Constraint::new(gamma, eq_statement, sel_statement)
            })
            .collect()
    }

    #[test]
    fn test_eval_constraints_poly_prefix() {
        // Fixture: 6 random constraints over 20 variables.
        let mut rng = SmallRng::seed_from_u64(0);
        let constraints = random_constraints(&mut rng, 20, 6);
        let challenge = Point::rand(&mut rng, 20);

        // Fast path vs reference implementation must agree.
        let got = VariableOrder::Prefix.eval_constraints_poly(&constraints, &challenge);
        let expected =
            eval_constraints_poly_reference(VariableOrder::Prefix, &constraints, &challenge);
        assert_eq!(got, expected);
    }

    #[test]
    fn test_eval_constraints_poly_suffix() {
        // Fixture: 6 random constraints over 20 variables.
        let mut rng = SmallRng::seed_from_u64(1);
        let constraints = random_constraints(&mut rng, 20, 6);
        let challenge = Point::rand(&mut rng, 20);

        // Fast path vs reference implementation must agree.
        let got = VariableOrder::Suffix.eval_constraints_poly(&constraints, &challenge);
        let expected =
            eval_constraints_poly_reference(VariableOrder::Suffix, &constraints, &challenge);
        assert_eq!(got, expected);
    }

    proptest! {
        // Invariant:
        //     VariableOrder::eval_constraints_poly must agree with the reference
        //     implementation across random constraint sets and challenge points.
        #[test]
        fn prop_eval_constraints_poly_matches_reference(
            total_num_variables in 2usize..=20,
            rounds in 1usize..=8,
            seed in any::<u64>(),
        ) {
            let mut rng = SmallRng::seed_from_u64(seed);
            let constraints = random_constraints(&mut rng, total_num_variables, rounds);
            let challenge = Point::rand(&mut rng, total_num_variables);

            prop_assert_eq!(
                VariableOrder::Prefix.eval_constraints_poly(&constraints, &challenge),
                eval_constraints_poly_reference(VariableOrder::Prefix, &constraints, &challenge),
            );
            prop_assert_eq!(
                VariableOrder::Suffix.eval_constraints_poly(&constraints, &challenge),
                eval_constraints_poly_reference(VariableOrder::Suffix, &constraints, &challenge),
            );
        }
    }
}
