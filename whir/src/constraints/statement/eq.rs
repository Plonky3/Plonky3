use alloc::vec::Vec;

use itertools::Itertools;
use p3_field::{
    ExtensionField, Field, PackedFieldExtension, PackedValue, PrimeCharacteristicRing, dot_product,
};
use p3_matrix::Matrix;
use p3_matrix::dense::{RowMajorMatrix, RowMajorMatrixView};
use p3_maybe_rayon::prelude::*;
use p3_multilinear_util::eq_batch::eval_eq_batch;
use p3_multilinear_util::evals::Poly;
use p3_multilinear_util::multilinear::Point;
use p3_util::log2_strict_usize;
use tracing::instrument;

/// Batched equality polynomial evaluation via butterfly expansion.
///
/// # Overview
///
/// Given `n` points as columns of a `k × n` matrix, computes
/// `eq(P_i, X)` for every `X in {0,1}^k` and every point `P_i`,
/// scaled by powers of `alpha`:
///
/// ```text
/// mat[X, i] = alpha^i * eq(P_i, X)
/// ```
///
/// # Algorithm
///
/// Uses the standard eq-polynomial butterfly. Each input row `i`
/// (variable) doubles the filled region:
///
/// ```text
/// hi[j] = lo[j] * var[j]        (contribution when X_i = 1)
/// lo[j] -= hi[j]                 (contribution when X_i = 0)
/// ```
///
/// The seed row is `[alpha^0, alpha^1, ..., alpha^{n-1}]`, which
/// bakes the batching weights directly into the expansion.
fn batch_eqs<F: Field, EF: ExtensionField<F>>(
    points: RowMajorMatrixView<'_, EF>,
    alpha: EF,
) -> RowMajorMatrix<EF> {
    let k = points.height();
    let n = points.width();
    assert_ne!(n, 0);

    let mut mat = RowMajorMatrix::new(EF::zero_vec(n * (1 << k)), n);

    // Seed with alpha powers: column j starts with alpha^j.
    mat.row_mut(0).copy_from_slice(&alpha.powers().collect_n(n));

    // Butterfly: process one variable per step.
    points.row_slices().enumerate().for_each(|(i, vars)| {
        let (mut lo, mut hi) = mat.split_rows_mut(1 << i);
        lo.rows_mut().zip(hi.rows_mut()).for_each(|(lo, hi)| {
            vars.iter()
                .zip(lo.iter_mut().zip(hi.iter_mut()))
                .for_each(|(&var, (lo, hi))| {
                    // hi = lo * var (X_i = 1 branch)
                    *hi = *lo * var;
                    // lo = lo * (1 - var) = lo - hi (X_i = 0 branch)
                    *lo -= *hi;
                });
        });
    });
    mat
}

/// SIMD-packed variant of the batched equality polynomial evaluation.
///
/// # Overview
///
/// Produces the same result as the scalar version, but stores each
/// column's eq coefficients in packed form. One SIMD element covers
/// `Packing::WIDTH` consecutive hypercube entries.
///
/// # Algorithm
///
/// 1. **Packing phase** (first `k_pack` variables): For each column,
///    build the small scalar eq table over `{0,1}^{k_pack}`, then
///    pack into a single SIMD element. This fills one row of the
///    packed output.
///
/// 2. **Butterfly phase** (remaining `k - k_pack` variables): Same
///    eq butterfly as the scalar version, but each multiply/subtract
///    operates on packed elements — all SIMD lanes in parallel.
///
/// No alpha scaling is applied (the caller handles batching weights
/// separately via the dot product in `combine_hypercube_packed`).
fn packed_batch_eqs<F: Field, EF: ExtensionField<F>>(
    points: RowMajorMatrixView<'_, EF>,
) -> RowMajorMatrix<EF::ExtensionPacking> {
    let k = points.height();
    let n = points.width();
    assert_ne!(n, 0);
    let k_pack = log2_strict_usize(F::Packing::WIDTH);
    assert!(k >= k_pack);

    let (init_vars, rest_vars) = points.split_rows(k_pack);
    let mut mat = RowMajorMatrix::new(EF::ExtensionPacking::zero_vec(n * (1 << (k - k_pack))), n);

    if k_pack > 0 {
        // Packing phase: build a scalar eq table per column over
        // the first k_pack variables, then pack into one SIMD element.
        //
        // The transpose gives us one row per column of the input,
        // containing the k_pack variable values for that point.
        init_vars
            .transpose()
            .row_slices()
            .zip(mat.values.iter_mut())
            .for_each(|(vars, packed)| {
                // Reverse order: the butterfly processes variables from
                // MSB to LSB, but new_from_point expects LSB-first.
                let point = vars.iter().rev().copied().collect::<Vec<_>>();
                *packed = EF::ExtensionPacking::from_ext_slice(
                    Poly::new_from_point(&point, EF::ONE).as_slice(),
                );
            });
    } else {
        // No packing needed: WIDTH = 1, seed row is all ones.
        mat.row_mut(0).fill(EF::ExtensionPacking::ONE);
    }

    // Butterfly phase: same eq expansion as the scalar version,
    // but operating on packed elements for SIMD parallelism.
    rest_vars.row_slices().enumerate().for_each(|(i, vars)| {
        let (mut lo, mut hi) = mat.split_rows_mut(1 << i);
        lo.rows_mut().zip(hi.rows_mut()).for_each(|(lo, hi)| {
            vars.iter()
                .zip(lo.iter_mut().zip(hi.iter_mut()))
                .for_each(|(&var, (lo, hi))| {
                    // hi = lo * var (X_i = 1 branch)
                    *hi = *lo * var;
                    // lo = lo * (1 - var) = lo - hi (X_i = 0 branch)
                    *lo -= *hi;
                });
        });
    });
    mat
}

/// A batched system of evaluation constraints $p(z_i) = s_i$ on $\{0,1\}^m$.
///
/// Each entry ties a Boolean point `z_i` to an expected polynomial evaluation `s_i`.
///
/// Batching with a random challenge $\gamma$ produces a single combined weight
/// polynomial $W$ and a single scalar $S$ that summarize all constraints.
///
/// Invariants
/// ----------
/// - `points.len() == evaluations.len()`.
/// - Every `Point` in `points` has exactly `num_variables` coordinates.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EqStatement<F> {
    /// Number of variables in the multilinear polynomial space.
    num_variables: usize,
    /// List of evaluation points.
    pub points: Vec<Point<F>>,
    /// List of target evaluations.
    pub evaluations: Vec<F>,
}

impl<F: Field> EqStatement<F> {
    /// Creates an empty `EqStatement<F>` for polynomials with `num_variables` variables.
    #[must_use]
    pub const fn initialize(num_variables: usize) -> Self {
        Self {
            num_variables,
            points: Vec::new(),
            evaluations: Vec::new(),
        }
    }

    /// Returns the number of variables defining the polynomial space.
    #[must_use]
    pub const fn num_variables(&self) -> usize {
        self.num_variables
    }

    /// Returns true if the statement contains no constraints.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        debug_assert!(self.points.is_empty() == self.evaluations.is_empty());
        self.points.is_empty()
    }

    /// Returns an iterator over the evaluation constraints in the statement.
    pub fn iter(&self) -> impl Iterator<Item = (&Point<F>, &F)> {
        self.points.iter().zip(self.evaluations.iter())
    }

    /// Returns the number of constraints in the statement.
    #[must_use]
    pub const fn len(&self) -> usize {
        debug_assert!(self.points.len() == self.evaluations.len());
        self.points.len()
    }

    /// Verifies that a given polynomial satisfies all constraints in the statement.
    #[must_use]
    pub fn verify(&self, poly: &Poly<F>) -> bool {
        self.iter()
            .all(|(point, &expected_eval)| poly.eval_base(point) == expected_eval)
    }

    /// Concatenates another statement's constraints into this one.
    pub fn concatenate(&mut self, other: &Self) {
        assert_eq!(self.num_variables, other.num_variables);
        self.points.extend_from_slice(&other.points);
        self.evaluations.extend_from_slice(&other.evaluations);
    }

    /// Adds an evaluation constraint `p(z) = s` to the system.
    ///
    /// Assumes the evaluation `s` is already known.
    ///
    /// # Panics
    /// Panics if the number of variables in the `point` does not match the statement.
    pub fn add_evaluated_constraint(&mut self, point: Point<F>, eval: F) {
        assert_eq!(point.num_vars(), self.num_variables());
        self.points.push(point);
        self.evaluations.push(eval);
    }

    /// Combine all constraints into a single batched weight polynomial and expected sum.
    ///
    /// Computes `W(x) = sum_i gamma^i * eq(x, z_i)` for all `x in {0,1}^k`
    /// using the Plonky3 `eval_eq_batch` kernel, and accumulates the
    /// scalar sum `S = sum_i gamma^i * s_i`.
    ///
    /// The `INITIALIZED` const generic controls whether the accumulator
    /// is added to (true) or overwritten (false).
    #[instrument(skip_all, fields(num_constraints = self.len(), num_variables = self.num_variables()))]
    pub fn combine_hypercube<Base, const INITIALIZED: bool>(
        &self,
        acc_weights: &mut Poly<F>,
        acc_sum: &mut F,
        challenge: F,
    ) where
        Base: Field,
        F: ExtensionField<Base>,
    {
        if self.points.is_empty() {
            return;
        }

        let num_constraints = self.len();

        // Precompute challenge powers gamma^0, gamma^1, ..., gamma^{n-1}.
        let challenges = challenge.powers().collect_n(num_constraints);

        // Transpose the points into a k × n matrix (rows = variables,
        // columns = constraint points). Uses Plonky3's transpose which
        // writes directly into a flat buffer.
        let points_matrix = Point::transpose(&self.points, false);

        // Delegate to Plonky3's batched eq-polynomial kernel.
        // This is the hot path — computes all 2^k evaluations in one pass.
        eval_eq_batch::<Base, F, INITIALIZED>(
            points_matrix.as_view(),
            acc_weights.as_mut_slice(),
            &challenges,
        );

        // Accumulate the scalar target sum: S += sum_i gamma^i * s_i.
        *acc_sum +=
            dot_product::<F, _, _>(self.evaluations.iter().copied(), challenges.into_iter());
    }

    /// SIMD-packed variant of constraint batching on the hypercube.
    ///
    /// Produces the same result as the scalar version but stores the
    /// weight polynomial in packed form (one element per
    /// `Packing::WIDTH` consecutive hypercube entries).
    ///
    /// # Algorithm
    ///
    /// For small `k` (where `2 * k_pack > k`), falls back to a
    /// per-constraint naive loop that builds each eq polynomial
    /// separately and packs it chunk-by-chunk.
    ///
    /// For larger `k`, uses the split-and-dot strategy:
    ///
    /// 1. Transpose the points and split at `k / 2`.
    /// 2. Left half  → `packed_batch_eqs` (SIMD lanes).
    /// 3. Right half → `batch_eqs` (scalar, with alpha scaling).
    /// 4. Parallel dot product: for each right-half row, dot all
    ///    left-half rows weighted by the scalar eq values.
    #[instrument(skip_all, fields(num_constraints = self.len(), num_variables = self.num_variables()))]
    pub fn combine_hypercube_packed<Base, const INITIALIZED: bool>(
        &self,
        weights: &mut Poly<F::ExtensionPacking>,
        sum: &mut F,
        challenge: F,
    ) where
        Base: Field,
        F: ExtensionField<Base>,
    {
        if self.points.is_empty() {
            return;
        }

        let k = self.num_variables();
        let k_pack = log2_strict_usize(Base::Packing::WIDTH);
        assert!(k >= k_pack);
        assert_eq!(weights.num_vars() + k_pack, k);

        // Combine expected evaluations: S = ∑_i γ^i * s_i
        self.combine_evals(sum, challenge);

        // Apply naive method if number of variables is too small for packed split method
        if k_pack * 2 > k {
            self.points
                .iter()
                .zip(challenge.powers())
                .enumerate()
                .for_each(|(i, (point, challenge))| {
                    let eq = Poly::new_from_point(point.as_slice(), challenge);
                    weights
                        .as_mut_slice()
                        .iter_mut()
                        .zip_eq(eq.as_slice().chunks(Base::Packing::WIDTH))
                        .for_each(|(out, chunk)| {
                            let packed = F::ExtensionPacking::from_ext_slice(chunk);
                            if INITIALIZED || i > 0 {
                                *out += packed;
                            } else {
                                *out = packed;
                            }
                        });
                });
            return;
        }

        let points = Point::transpose(&self.points, true);
        let (left, right) = points.split_rows(k / 2);
        let left = packed_batch_eqs::<Base, F>(left);
        let right = batch_eqs::<Base, F>(right, challenge);

        weights
            .as_mut_slice()
            .par_chunks_mut(left.height())
            .zip_eq(right.par_row_slices())
            .for_each(|(out, right)| {
                out.iter_mut().zip(left.rows()).for_each(|(out, left)| {
                    if INITIALIZED {
                        *out +=
                            dot_product::<F::ExtensionPacking, _, _>(left, right.iter().copied());
                    } else {
                        *out = dot_product(left, right.iter().copied());
                    }
                });
            });
    }

    /// Combines a list of evals into a single linear combination using powers of `gamma`,
    /// and updates the running claimed_eval in place.
    ///
    /// # Arguments
    /// - `claimed_eval`: Mutable reference to the total accumulated claimed eval so far. Updated in place.
    /// - `gamma`: A random extension field element used to weight the evals.
    pub fn combine_evals(&self, claimed_eval: &mut F, gamma: F) {
        *claimed_eval += dot_product(self.evaluations.iter().copied(), gamma.powers());
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec;

    use p3_baby_bear::BabyBear;
    use p3_field::PrimeCharacteristicRing;
    use p3_field::extension::BinomialExtensionField;
    use proptest::prelude::*;
    use rand::rngs::SmallRng;
    use rand::{RngExt, SeedableRng};

    use super::*;

    impl<F: Field> EqStatement<F> {
        /// Creates a filled `EqStatement<F>` for polynomials with `num_variables` variables.
        ///
        /// # Standard Hypercube Representation
        ///
        /// This constructor is for the standard case where the polynomial is represented as
        /// evaluations over the Boolean hypercube `{0,1}^num_variables`, and will be evaluated
        /// at arbitrary constraint points using standard multilinear interpolation. Each point
        /// has exactly `num_variables` coordinates.
        #[must_use]
        pub fn new_hypercube(points: Vec<Point<F>>, evaluations: Vec<F>) -> Self {
            // Validate that we have one evaluation per point.
            assert_eq!(
                points.len(),
                evaluations.len(),
                "Number of points ({}) must match number of evaluations ({})",
                points.len(),
                evaluations.len()
            );

            // Validate that each point has the correct number of variables.
            let num_variables = points
                .iter()
                .map(Point::num_vars)
                .all_equal_value()
                .unwrap();
            Self {
                num_variables,
                points,
                evaluations,
            }
        }
    }

    type F = BabyBear;
    type EF = BinomialExtensionField<F, 4>;

    #[test]
    fn test_statement_combine_single_constraint() {
        let mut statement = EqStatement::initialize(1);
        let point = Point::new(vec![F::ONE]);
        let expected_eval = F::from_u64(7);
        statement.add_evaluated_constraint(point.clone(), expected_eval);

        let challenge = F::from_u64(2); // This is unused with one constraint.
        let mut combined_evals = Poly::zero(statement.num_variables());
        let mut combined_sum = F::ZERO;
        statement.combine_hypercube::<_, false>(&mut combined_evals, &mut combined_sum, challenge);

        // Expected evals for eq_z(X) where z = (1).
        // For x=0, eq=0. For x=1, eq=1.
        let expected_combined_evals = Poly::new_from_point(point.as_slice(), F::ONE);

        assert_eq!(combined_evals, expected_combined_evals);
        assert_eq!(combined_sum, expected_eval);
    }

    #[test]
    fn test_statement_with_multiple_constraints() {
        let mut statement = EqStatement::initialize(2);

        // Constraint 1: evaluate at z1 = (1,0), expected value 5
        let point1 = Point::new(vec![F::ONE, F::ZERO]);
        let eval1 = F::from_u64(5);
        statement.add_evaluated_constraint(point1.clone(), eval1);

        // Constraint 2: evaluate at z2 = (0,1), expected value 7
        let point2 = Point::new(vec![F::ZERO, F::ONE]);
        let eval2 = F::from_u64(7);
        statement.add_evaluated_constraint(point2.clone(), eval2);

        let challenge = F::from_u64(2);
        let mut combined_evals = Poly::zero(statement.num_variables());
        let mut combined_sum = F::ZERO;
        statement.combine_hypercube::<_, false>(&mut combined_evals, &mut combined_sum, challenge);

        // Expected evals: W(X) = eq_z1(X) + challenge * eq_z2(X)
        let expected_eq1 = Poly::new_from_point(point1.as_slice(), F::ONE);
        let expected_eq2 = Poly::new_from_point(point2.as_slice(), challenge);
        let expected_combined_evals = Poly::new(
            expected_eq1
                .iter()
                .zip(expected_eq2.iter())
                .map(|(&a, &b)| a + b)
                .collect(),
        );

        // Expected sum: S = s1 + challenge * s2
        let expected_combined_sum = eval1 + challenge * eval2;

        assert_eq!(combined_evals, expected_combined_evals);
        assert_eq!(combined_sum, expected_combined_sum);
    }

    #[test]
    fn test_compute_evaluation_weight() {
        // Define an evaluation weight at a specific point
        let point = Point::new(vec![F::from_u64(3)]);

        // Define a randomness point for folding
        let folding_randomness = Point::new(vec![F::from_u64(2)]);

        // Expected result is the evaluation of eq_poly at the given randomness
        let expected = point.eq_poly(&folding_randomness);

        assert_eq!(point.eq_poly(&folding_randomness), expected);
    }

    #[test]
    fn test_constructors_and_basic_properties() {
        // Test new_hypercube constructor
        let point = Point::new(vec![F::ONE]);
        let eval = F::from_u64(42);
        let statement = EqStatement::new_hypercube(vec![point], vec![eval]);

        assert_eq!(statement.num_variables(), 1);
        assert_eq!(statement.len(), 1);
        assert!(!statement.is_empty());

        // Test initialize constructor
        let empty_statement = EqStatement::<F>::initialize(2);
        assert_eq!(empty_statement.num_variables(), 2);
        assert_eq!(empty_statement.len(), 0);
        assert!(empty_statement.is_empty());
    }

    #[test]
    fn test_verify_constraints() {
        // Create polynomial with evaluations [1, 2]
        let poly = Poly::new(vec![F::from_u64(1), F::from_u64(2)]);
        let mut statement = EqStatement::<F>::initialize(1);

        // Test matching constraint: f(0) = 1
        statement.add_evaluated_constraint(Point::new(vec![F::ZERO]), F::from_u64(1));
        assert!(statement.verify(&poly));

        // Test mismatched constraint: f(1) = 5 (but poly has f(1) = 2)
        statement.add_evaluated_constraint(Point::new(vec![F::ONE]), F::from_u64(5));
        assert!(!statement.verify(&poly));
    }

    #[test]
    fn test_concatenate() {
        // Test successful concatenation
        let mut statement1 = EqStatement::<F>::initialize(1);
        let mut statement2 = EqStatement::<F>::initialize(1);
        statement1.add_evaluated_constraint(Point::new(vec![F::ZERO]), F::from_u64(10));
        statement2.add_evaluated_constraint(Point::new(vec![F::ONE]), F::from_u64(20));

        statement1.concatenate(&statement2);
        assert_eq!(statement1.len(), 2);
    }

    #[test]
    #[should_panic(expected = "assertion `left == right` failed")]
    fn test_concatenate_mismatched_variables() {
        let mut statement1 = EqStatement::<F>::initialize(2);
        let statement2 = EqStatement::<F>::initialize(3);
        statement1.concatenate(&statement2); // Should panic
    }

    #[test]
    fn test_add_evaluated_constraint() {
        let poly = Poly::new(vec![F::from_u64(1), F::from_u64(2)]);
        let point = Point::new(vec![F::ZERO]);

        let mut statement = EqStatement::<F>::initialize(1);

        // Add constraint with pre-computed evaluation
        let eval = poly.eval_base(&point);
        statement.add_evaluated_constraint(point, eval);

        // Statement should have one constraint
        assert_eq!(statement.len(), 1);

        // Should verify against the polynomial
        assert!(statement.verify(&poly));

        // Points should be stored
        assert_eq!(statement.points.len(), 1);
    }

    #[test]
    #[should_panic(expected = "assertion `left == right` failed")]
    fn test_wrong_variable_count() {
        let mut statement = EqStatement::<F>::initialize(1);
        let wrong_point = Point::new(vec![F::ONE, F::ZERO]); // 2 vars for 1-var statement
        statement.add_evaluated_constraint(wrong_point, F::from_u64(5));
    }

    #[test]
    fn test_combine_operations() {
        // Test empty statement combine
        let empty_statement = EqStatement::<F>::initialize(1);

        let mut combined_evals = Poly::zero(empty_statement.num_variables());
        let mut combined_sum = F::ZERO;
        empty_statement.combine_hypercube::<_, false>(
            &mut combined_evals,
            &mut combined_sum,
            F::from_u64(42),
        );
        assert_eq!(combined_sum, F::ZERO);

        // Test combine_evals with constraints
        let mut statement = EqStatement::<F>::initialize(1);
        statement.add_evaluated_constraint(Point::new(vec![F::ZERO]), F::from_u64(3));
        statement.add_evaluated_constraint(Point::new(vec![F::ONE]), F::from_u64(7));

        let mut claimed_eval = F::ZERO;
        statement.combine_evals(&mut claimed_eval, F::from_u64(2));

        // Verify: 3*1 + 7*2 = 17
        assert_eq!(claimed_eval, F::from_u64(17));
    }

    proptest! {
        #[test]
        fn prop_statement_workflow(
            // Random 4-var polynomial: 16 evaluations (2^4)
            poly_evals in prop::collection::vec(0u32..100, 16),
            // Random challenge value
            challenge in 1u32..50,
            // Random constraint points (4 coords × 2 points)
            point_coords in prop::collection::vec(0u32..10, 8),
        ) {
            // Create a 4-variable polynomial from random evaluations
            let poly = Poly::new(poly_evals.into_iter().map(F::from_u32).collect());

            // Create statement with random constraints that match the polynomial
            let mut statement = EqStatement::<F>::initialize(4);
            let point1 = Point::new(vec![
                F::from_u32(point_coords[0]), F::from_u32(point_coords[1]),
                F::from_u32(point_coords[2]), F::from_u32(point_coords[3])
            ]);
            let point2 = Point::new(vec![
                F::from_u32(point_coords[4]), F::from_u32(point_coords[5]),
                F::from_u32(point_coords[6]), F::from_u32(point_coords[7])
            ]);

            // Add constraints: poly(point1) = actual_eval1, poly(point2) = actual_eval2
            let eval1 = poly.eval_base(&point1);
            let eval2 = poly.eval_base(&point2);
            statement.add_evaluated_constraint(point1, eval1);
            statement.add_evaluated_constraint(point2, eval2);

            // Statement should verify against polynomial (consistent constraints)
            prop_assert!(statement.verify(&poly));

            // Combine constraints with challenge
            let gamma = F::from_u32(challenge);
            let mut combined_poly = Poly::zero(statement.num_variables());
            let mut combined_sum = F::ZERO;
            statement.combine_hypercube::<_, false>(&mut combined_poly, &mut combined_sum, gamma);

            // Combined polynomial should have same number of variables
            prop_assert_eq!(combined_poly.num_vars(), 4);

            // Combined evaluations should match combine result
            let mut claimed_eval = F::ZERO;
            statement.combine_evals(&mut claimed_eval, gamma);
            // Both methods should give same sum
            prop_assert_eq!(combined_sum, claimed_eval);

            // Adding wrong constraint should break verification
            let wrong_point = Point::new(vec![F::ZERO, F::ZERO, F::ZERO, F::ZERO]);
            // Obviously wrong evaluation
            let wrong_eval = F::from_u32(999);
            let actual_eval = poly.eval_base(&wrong_point);
            // Only test if actually different
            if wrong_eval != actual_eval {
                statement.add_evaluated_constraint(wrong_point, wrong_eval);
                // Should fail verification
                prop_assert!(!statement.verify(&poly));
            }
        }
    }

    #[test]
    #[should_panic(expected = "Number of points (2) must match number of evaluations (1)")]
    fn test_new_mismatched_lengths() {
        // Should panic when points.len() != evaluations.len()
        let points = vec![Point::new(vec![F::ONE]), Point::new(vec![F::ZERO])];
        let evaluations = vec![F::from_u64(100)];

        let _ = EqStatement::new_hypercube(points, evaluations);
    }

    proptest! {
        #[test]
        fn prop_packed_combine_roundtrip(
            // Number of variables (covers both naive and split paths).
            k in 4usize..10,
            // Number of constraints per batch.
            n in 1usize..12,
            // RNG seed for reproducible randomness.
            seed in 0u64..100,
        ) {
            let k_pack = log2_strict_usize(<F as Field>::Packing::WIDTH);
            if k < k_pack {
                // Skip configurations where k is too small for packing.
                return Ok(());
            }

            let mut rng = SmallRng::seed_from_u64(seed);
            let challenge: EF = rng.random();

            // Generate n random constraint points in {F}^k.
            let points = (0..n)
                .map(|_| Point::rand(&mut rng, k))
                .collect::<Vec<_>>();
            // Generate n random expected evaluations.
            let evals = (0..n).map(|_| rng.random()).collect::<Vec<EF>>();

            let statement = EqStatement::<EF>::new_hypercube(points, evals);

            // Scalar path: combine into a 2^k evaluation list.
            let mut scalar_weights = Poly::<EF>::zero(k);
            let mut scalar_sum = EF::ZERO;
            statement.combine_hypercube::<F, false>(
                &mut scalar_weights, &mut scalar_sum, challenge,
            );

            // Packed path: combine into a 2^{k - k_pack} packed list.
            let mut packed_weights =
                Poly::<<EF as ExtensionField<F>>::ExtensionPacking>::zero(k - k_pack);
            let mut packed_sum = EF::ZERO;
            statement.combine_hypercube_packed::<F, false>(
                &mut packed_weights, &mut packed_sum, challenge,
            );

            // Unpack the packed result and compare element-by-element.
            let unpacked =
                <<EF as ExtensionField<F>>::ExtensionPacking as PackedFieldExtension<F, EF>>::to_ext_iter(
                    packed_weights.as_slice().iter().copied(),
                )
                .collect::<Vec<_>>();
            prop_assert_eq!(scalar_weights.as_slice(), &unpacked[..]);

            // The scalar sums must match exactly.
            prop_assert_eq!(scalar_sum, packed_sum);
        }

        #[test]
        fn prop_packed_combine_accumulation(
            k in 4usize..10,
            seed in 0u64..50,
        ) {
            let k_pack = log2_strict_usize(<F as Field>::Packing::WIDTH);
            if k < k_pack {
                return Ok(());
            }

            let mut rng = SmallRng::seed_from_u64(seed);
            let challenge: EF = rng.random();

            // First batch: INITIALIZED=false (overwrite).
            let points1 = (0..3)
                .map(|_| Point::rand(&mut rng, k))
                .collect::<Vec<_>>();
            let evals1 = (0..3).map(|_| rng.random()).collect::<Vec<EF>>();
            let stmt1 = EqStatement::<EF>::new_hypercube(points1, evals1);

            let mut s_wt = Poly::<EF>::zero(k);
            let mut s_sum = EF::ZERO;
            stmt1.combine_hypercube::<F, false>(&mut s_wt, &mut s_sum, challenge);

            let mut p_wt =
                Poly::<<EF as ExtensionField<F>>::ExtensionPacking>::zero(k - k_pack);
            let mut p_sum = EF::ZERO;
            stmt1.combine_hypercube_packed::<F, false>(&mut p_wt, &mut p_sum, challenge);

            // Second batch: INITIALIZED=true (accumulate on top).
            let points2 = (0..5)
                .map(|_| Point::rand(&mut rng, k))
                .collect::<Vec<_>>();
            let evals2 = (0..5).map(|_| rng.random()).collect::<Vec<EF>>();
            let stmt2 = EqStatement::<EF>::new_hypercube(points2, evals2);

            stmt2.combine_hypercube::<F, true>(&mut s_wt, &mut s_sum, challenge);
            stmt2.combine_hypercube_packed::<F, true>(&mut p_wt, &mut p_sum, challenge);

            // Verify accumulated results match.
            let unpacked =
                <<EF as ExtensionField<F>>::ExtensionPacking as PackedFieldExtension<F, EF>>::to_ext_iter(
                    p_wt.as_slice().iter().copied(),
                )
                .collect::<Vec<_>>();
            prop_assert_eq!(s_wt.as_slice(), &unpacked[..]);
            prop_assert_eq!(s_sum, p_sum);
        }
    }
}
