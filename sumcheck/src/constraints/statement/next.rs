use alloc::vec::Vec;

use itertools::Itertools;
use p3_field::{ExtensionField, Field, PackedFieldExtension, PackedValue, dot_product};
use p3_multilinear_util::point::Point;
use p3_multilinear_util::poly::Poly;
use p3_util::log2_strict_usize;
use tracing::instrument;

/// A batched system of shifted-evaluation constraints on `{0,1}^m`.
///
/// Each entry ties a point `z_i` to a claimed weighted sum over the hypercube:
///
/// ```text
/// sum_{b in {0,1}^m} next(z_i, b) * P(b) = s_i
/// ```
///
/// `next` is the multilinear extension of the row-successor indicator:
/// rows are indexed big-endian and the last row maps to itself.
///
/// # Overview
///
/// - Let `g` be the row-shifted column: `g[r] = P[r + 1]`, `g[last] = P[last]`.
/// - Each constraint asserts that the multilinear extension of `g` evaluates to `s_i` at `z_i`.
/// - This is the "next row" opening required by AIR transition constraints
///   over a multilinear commitment.
///
/// # Soundness
///
/// - The weight is linear in the committed value and multilinear in the point.
/// - It therefore fits the constrained Reed-Solomon framework
///   (WHIR, ePrint 2024/1586) with the same analysis as an equality weight.
/// - The verifier evaluates the weight at the folding point in `O(m)` operations.
///
/// # Scope
///
/// - The kernel spans the statement's full variable space: one committed column.
/// - Stacked layouts need the factored form `eq(selector, .) (x) next(z, .)`.
/// - The SVO fast path needs the kernel split into `l_0 + 2` eq-shaped groups.
/// - Both belong to the layout layer and are out of scope here.
///
/// # Invariants
///
/// - `points.len() == evaluations.len()`.
/// - Every `Point` in `points` has exactly `num_variables` coordinates.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NextStatement<F> {
    /// Number of variables in the multilinear polynomial space.
    num_variables: usize,
    /// List of shifted-evaluation points.
    pub points: Vec<Point<F>>,
    /// List of target evaluations.
    pub evaluations: Vec<F>,
}

impl<F: Field> NextStatement<F> {
    /// Creates an empty `NextStatement<F>` for polynomials with `num_variables` variables.
    #[must_use]
    pub const fn initialize(num_variables: usize) -> Self {
        Self {
            num_variables,
            points: Vec::new(),
            evaluations: Vec::new(),
        }
    }

    /// Creates a filled `NextStatement<F>` from aligned points and evaluations.
    ///
    /// # Panics
    /// Panics if the lists have different lengths or inconsistent arities.
    #[must_use]
    pub fn new(points: Vec<Point<F>>, evaluations: Vec<F>) -> Self {
        assert_eq!(
            points.len(),
            evaluations.len(),
            "Number of points ({}) must match number of evaluations ({})",
            points.len(),
            evaluations.len()
        );
        let num_variables = points
            .iter()
            .map(Point::num_variables)
            .all_equal_value()
            .unwrap();
        Self {
            num_variables,
            points,
            evaluations,
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

    /// Returns the number of constraints in the statement.
    #[must_use]
    pub const fn len(&self) -> usize {
        debug_assert!(self.points.len() == self.evaluations.len());
        self.points.len()
    }

    /// Returns an iterator over the shifted-evaluation constraints in the statement.
    pub fn iter(&self) -> impl Iterator<Item = (&Point<F>, &F)> {
        self.points.iter().zip(self.evaluations.iter())
    }

    /// Concatenates another statement's constraints into this one.
    pub fn concatenate(&mut self, other: &Self) {
        assert_eq!(self.num_variables, other.num_variables);
        self.points.extend_from_slice(&other.points);
        self.evaluations.extend_from_slice(&other.evaluations);
    }

    /// Adds a shifted-evaluation constraint `sum_b next(z, b) * P(b) = s` to the system.
    ///
    /// Assumes the evaluation `s` is already known.
    ///
    /// # Panics
    /// Panics if the number of variables in the `point` does not match the statement.
    pub fn add_evaluated_constraint(&mut self, point: Point<F>, eval: F) {
        assert_eq!(point.num_variables(), self.num_variables());
        self.points.push(point);
        self.evaluations.push(eval);
    }

    /// Verifies that a given polynomial satisfies all constraints in the statement.
    #[must_use]
    pub fn verify(&self, poly: &Poly<F>) -> bool {
        self.iter().all(|(point, &expected_eval)| {
            let weights = Poly::new_next_from_point(point.as_slice(), F::ONE);
            dot_product::<F, _, _>(weights.iter().copied(), poly.iter().copied()) == expected_eval
        })
    }

    /// Combine all constraints into a single batched weight polynomial and expected sum.
    ///
    /// Adds, for all `b in {0,1}^k`:
    ///
    /// ```text
    /// W(b) += sum_i gamma^{i+shift} * next(z_i, b)
    /// S    += sum_i gamma^{i+shift} * s_i
    /// ```
    ///
    /// # Arguments
    ///
    /// - `acc_weights`: weight accumulator with `2^k` entries; always added to.
    /// - `acc_sum`: target-sum accumulator; always added to.
    /// - `challenge`: random batching challenge `gamma`.
    /// - `shift`: power offset so statement types share one challenge
    ///   with non-overlapping powers.
    ///
    /// # Performance
    ///
    /// One `O(2^k)` butterfly per constraint, matching an equality statement
    /// of the same size.
    #[instrument(skip_all, fields(num_constraints = self.len(), num_variables = self.num_variables()))]
    pub fn combine(&self, acc_weights: &mut Poly<F>, acc_sum: &mut F, challenge: F, shift: usize) {
        if self.points.is_empty() {
            return;
        }

        // W(b) += sum_i gamma^{i+shift} * next(z_i, b)
        self.points
            .iter()
            .zip(challenge.shifted_powers(challenge.exp_u64(shift as u64)))
            .for_each(|(point, gamma_i)| {
                let table = Poly::new_next_from_point(point.as_slice(), gamma_i);
                acc_weights
                    .as_mut_slice()
                    .iter_mut()
                    .zip_eq(table.iter())
                    .for_each(|(out, &w)| *out += w);
            });

        // S += sum_i gamma^{i+shift} * s_i
        self.combine_evals(acc_sum, challenge, shift);
    }

    /// SIMD-packed variant of constraint batching.
    ///
    /// Produces the same result as the scalar version.
    /// Accumulates into a packed weight polynomial, one element per
    /// `Packing::WIDTH` consecutive hypercube entries.
    #[instrument(skip_all, fields(num_constraints = self.len(), num_variables = self.num_variables()))]
    pub fn combine_packed<Base>(
        &self,
        weights: &mut Poly<F::ExtensionPacking>,
        sum: &mut F,
        challenge: F,
        shift: usize,
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
        assert_eq!(weights.num_variables() + k_pack, k);

        // S += sum_i gamma^{i+shift} * s_i
        self.combine_evals(sum, challenge, shift);

        // W(b) += sum_i gamma^{i+shift} * next(z_i, b), packed chunk by chunk.
        self.points
            .iter()
            .zip(challenge.shifted_powers(challenge.exp_u64(shift as u64)))
            .for_each(|(point, gamma_i)| {
                let table = Poly::new_next_from_point(point.as_slice(), gamma_i);
                weights
                    .as_mut_slice()
                    .iter_mut()
                    .zip_eq(table.as_slice().chunks(Base::Packing::WIDTH))
                    .for_each(|(out, chunk)| {
                        *out += F::ExtensionPacking::from_ext_slice(chunk);
                    });
            });
    }

    /// Batches expected evaluation values into a single target sum using challenge powers.
    ///
    /// Adds `S = sum_i gamma^{i+shift} * s_i` to the accumulator.
    pub fn combine_evals(&self, claimed_eval: &mut F, gamma: F, shift: usize) {
        *claimed_eval += dot_product(
            self.evaluations.iter().copied(),
            gamma.shifted_powers(gamma.exp_u64(shift as u64)),
        );
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec;
    use alloc::vec::Vec;

    use p3_baby_bear::BabyBear;
    use p3_field::PrimeCharacteristicRing;
    use p3_field::extension::BinomialExtensionField;
    use proptest::prelude::*;
    use rand::rngs::SmallRng;
    use rand::{RngExt, SeedableRng};

    use super::*;

    type F = BabyBear;
    type EF = BinomialExtensionField<F, 4>;

    /// Reference shifted column: g[r] = f[r + 1], with a self-loop on the last row.
    fn shifted_column<T: Copy>(f: &[T]) -> Vec<T> {
        let mut g = f.to_vec();
        g.rotate_left(1);
        *g.last_mut().unwrap() = f[f.len() - 1];
        g
    }

    #[test]
    fn test_constructors_and_basic_properties() {
        let point = Point::new(vec![F::ONE]);
        let eval = F::from_u64(42);
        let statement = NextStatement::new(vec![point], vec![eval]);

        assert_eq!(statement.num_variables(), 1);
        assert_eq!(statement.len(), 1);
        assert!(!statement.is_empty());

        let empty_statement = NextStatement::<F>::initialize(2);
        assert_eq!(empty_statement.num_variables(), 2);
        assert_eq!(empty_statement.len(), 0);
        assert!(empty_statement.is_empty());
    }

    #[test]
    #[should_panic(expected = "Number of points (2) must match number of evaluations (1)")]
    fn test_new_mismatched_lengths() {
        let points = vec![Point::new(vec![F::ONE]), Point::new(vec![F::ZERO])];
        let evaluations = vec![F::from_u64(100)];
        let _ = NextStatement::new(points, evaluations);
    }

    #[test]
    #[should_panic(expected = "assertion `left == right` failed")]
    fn test_wrong_variable_count() {
        let mut statement = NextStatement::<F>::initialize(1);
        let wrong_point = Point::new(vec![F::ONE, F::ZERO]);
        statement.add_evaluated_constraint(wrong_point, F::from_u64(5));
    }

    #[test]
    fn test_concatenate() {
        let mut statement1 = NextStatement::<F>::initialize(1);
        let mut statement2 = NextStatement::<F>::initialize(1);
        statement1.add_evaluated_constraint(Point::new(vec![F::ZERO]), F::from_u64(10));
        statement2.add_evaluated_constraint(Point::new(vec![F::ONE]), F::from_u64(20));

        statement1.concatenate(&statement2);
        assert_eq!(statement1.len(), 2);
    }

    #[test]
    fn test_verify_against_shifted_column() {
        // A next constraint on f is an evaluation constraint on the
        // shifted column g.
        let mut rng = SmallRng::seed_from_u64(0);
        let num_variables = 4;
        let f: Vec<F> = (0..1 << num_variables).map(|_| rng.random()).collect();
        let g = Poly::new(shifted_column(&f));
        let poly = Poly::new(f);

        let point = Point::rand(&mut rng, num_variables);
        let eval = g.eval_ext::<F>(&point);

        let mut statement = NextStatement::<F>::initialize(num_variables);
        statement.add_evaluated_constraint(point, eval);
        assert!(statement.verify(&poly));

        // A mismatched evaluation must fail.
        let mut bad = NextStatement::<F>::initialize(num_variables);
        bad.add_evaluated_constraint(Point::rand(&mut rng, num_variables), F::from_u64(999));
        assert!(!bad.verify(&poly));
    }

    #[test]
    fn test_combine_single_constraint() {
        let mut statement = NextStatement::<F>::initialize(2);
        let point = Point::new(vec![F::from_u64(3), F::from_u64(5)]);
        let expected_eval = F::from_u64(7);
        statement.add_evaluated_constraint(point.clone(), expected_eval);

        let challenge = F::from_u64(2);
        let mut combined_weights = Poly::zero(statement.num_variables());
        let mut combined_sum = F::ZERO;
        statement.combine(&mut combined_weights, &mut combined_sum, challenge, 0);

        // With one constraint and shift 0 the weight is gamma^0 = 1.
        let expected_weights = Poly::new_next_from_point(point.as_slice(), F::ONE);
        assert_eq!(combined_weights, expected_weights);
        assert_eq!(combined_sum, expected_eval);
    }

    #[test]
    fn test_combine_respects_shift() {
        let mut statement = NextStatement::<F>::initialize(2);
        let point = Point::new(vec![F::from_u64(3), F::from_u64(5)]);
        let eval = F::from_u64(7);
        statement.add_evaluated_constraint(point.clone(), eval);

        let challenge = F::from_u64(2);
        let shift = 3;
        let mut weights = Poly::zero(statement.num_variables());
        let mut sum = F::ZERO;
        statement.combine(&mut weights, &mut sum, challenge, shift);

        // With shift 3 the single constraint is weighted by gamma^3 = 8.
        let gamma_shift = challenge.exp_u64(shift as u64);
        let expected_weights = Poly::new_next_from_point(point.as_slice(), gamma_shift);
        assert_eq!(weights, expected_weights);
        assert_eq!(sum, gamma_shift * eval);
    }

    #[test]
    fn test_combine_accumulates() {
        // `combine` must add on top of existing accumulator contents.
        let mut statement = NextStatement::<F>::initialize(1);
        statement.add_evaluated_constraint(Point::new(vec![F::from_u64(4)]), F::from_u64(6));

        let mut weights = Poly::new(vec![F::from_u64(100), F::from_u64(200)]);
        let baseline = weights.clone();
        let mut sum = F::from_u64(50);
        statement.combine(&mut weights, &mut sum, F::from_u64(3), 0);

        let table = Poly::new_next_from_point(&[F::from_u64(4)], F::ONE);
        for ((&got, &base), &w) in weights.iter().zip(baseline.iter()).zip(table.iter()) {
            assert_eq!(got, base + w);
        }
        assert_eq!(sum, F::from_u64(56));
    }

    proptest! {
        #[test]
        fn prop_combine_matches_naive(
            num_variables in 1usize..=8,
            num_constraints in 1usize..=5,
            shift in 0usize..=4,
            seed in any::<u64>(),
        ) {
            // Invariant: the batched combine must equal the naive per-constraint
            // sum  W(b) = sum_i gamma^{i+shift} * next(z_i, b)  on every row.
            let mut rng = SmallRng::seed_from_u64(seed);
            let challenge: EF = rng.random();

            let points = (0..num_constraints)
                .map(|_| Point::rand(&mut rng, num_variables))
                .collect::<Vec<_>>();
            let evals = (0..num_constraints).map(|_| rng.random()).collect::<Vec<EF>>();
            let statement = NextStatement::new(points.clone(), evals.clone());

            let mut weights = Poly::<EF>::zero(num_variables);
            let mut sum = EF::ZERO;
            statement.combine(&mut weights, &mut sum, challenge, shift);

            let mut expected_weights = Poly::<EF>::zero(num_variables);
            let mut expected_sum = EF::ZERO;
            let mut gamma_i = challenge.exp_u64(shift as u64);
            for (point, eval) in points.iter().zip(&evals) {
                let table = Poly::new_next_from_point(point.as_slice(), gamma_i);
                for (out, &w) in expected_weights.as_mut_slice().iter_mut().zip(table.iter()) {
                    *out += w;
                }
                expected_sum += gamma_i * *eval;
                gamma_i *= challenge;
            }

            prop_assert_eq!(weights, expected_weights);
            prop_assert_eq!(sum, expected_sum);
        }

        #[test]
        fn prop_packed_combine_roundtrip(
            k in 4usize..=10,
            n in 1usize..=8,
            shift in 0usize..=4,
            seed in 0u64..100,
        ) {
            use p3_field::{Field, PackedFieldExtension};

            let k_pack = log2_strict_usize(<F as Field>::Packing::WIDTH);
            if k < k_pack {
                return Ok(());
            }

            let mut rng = SmallRng::seed_from_u64(seed);
            let challenge: EF = rng.random();

            let points = (0..n)
                .map(|_| Point::rand(&mut rng, k))
                .collect::<Vec<_>>();
            let evals = (0..n).map(|_| rng.random()).collect::<Vec<EF>>();
            let statement = NextStatement::new(points, evals);

            // Scalar path.
            let mut scalar_weights = Poly::<EF>::zero(k);
            let mut scalar_sum = EF::ZERO;
            statement.combine(&mut scalar_weights, &mut scalar_sum, challenge, shift);

            // Packed path.
            let mut packed_weights =
                Poly::<<EF as ExtensionField<F>>::ExtensionPacking>::zero(k - k_pack);
            let mut packed_sum = EF::ZERO;
            statement.combine_packed::<F>(&mut packed_weights, &mut packed_sum, challenge, shift);

            let unpacked =
                <<EF as ExtensionField<F>>::ExtensionPacking as PackedFieldExtension<F, EF>>::to_ext_iter(
                    packed_weights.as_slice().iter().copied(),
                )
                .collect::<Vec<_>>();
            prop_assert_eq!(scalar_weights.as_slice(), &unpacked[..]);
            prop_assert_eq!(scalar_sum, packed_sum);
        }

        #[test]
        fn prop_next_claim_workflow(
            num_variables in 1usize..=8,
            seed in any::<u64>(),
        ) {
            // End-to-end semantics: build a next claim from the shifted column,
            // verify it, and check the batched sumcheck identity
            //     sum_b W(b) * f(b) == S.
            let mut rng = SmallRng::seed_from_u64(seed);
            let f: Vec<F> = (0..1usize << num_variables).map(|_| rng.random()).collect();
            let g = Poly::new(shifted_column(&f));
            let poly = Poly::new(f);

            let mut statement = NextStatement::<F>::initialize(num_variables);
            for _ in 0..3 {
                let point = Point::rand(&mut rng, num_variables);
                let eval = g.eval_ext::<F>(&point);
                statement.add_evaluated_constraint(point, eval);
            }
            prop_assert!(statement.verify(&poly));

            let challenge: F = rng.random();
            let mut weights = Poly::<F>::zero(num_variables);
            let mut sum = F::ZERO;
            statement.combine(&mut weights, &mut sum, challenge, 0);

            let lhs = dot_product::<F, _, _>(weights.iter().copied(), poly.iter().copied());
            prop_assert_eq!(lhs, sum);
        }
    }
}
