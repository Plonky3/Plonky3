use itertools::Itertools;
use p3_field::{ExtensionField, Field};
use p3_matrix::dense::RowMajorMatrix;
use p3_maybe_rayon::prelude::*;

use crate::eq::{eval_eq, eval_eq_base};
use crate::point::MultilinearPoint;

/// The number of variables at which the multilinear evaluation algorithm switches
/// from a recursive to a non-recursive, chunk-based strategy.
///
/// The value `20` is an empirically chosen threshold representing the crossover point where the
/// benefits of the non-recursive strategy begin to outweigh its setup overhead.
pub const MLE_RECURSION_THRESHOLD: usize = 20;

const PARALLEL_THRESHOLD: usize = 4096;

/// A multilinear polynomial `f` stored by its evaluations over the boolean hypercube.
///
/// The evaluations are ordered using the lexicographic order on hypercube points.
/// The number of variables `n` is inferred from the length of this vector, via `vec.len() = 2^n`.
#[derive(Debug, Clone, Eq, PartialEq)]
#[must_use]
pub struct EvaluationsList<F>(Vec<F>);

impl<F> EvaluationsList<F>
where
    F: Field,
{
    /// Constructs an `EvaluationsList` from a vector of evaluations.
    ///
    /// The `evals` vector must adhere to the following constraints:
    /// - Its length must be a power of two, as it represents evaluations over a
    ///   binary hypercube of some dimension `n`.
    /// - The evaluations must be ordered lexicographically corresponding to the points
    ///   on the hypercube.
    ///
    /// # Panics
    /// Panics if `evals.len()` is not a power of two.
    #[inline]
    pub const fn new(evals: Vec<F>) -> Self {
        assert!(
            evals.len().is_power_of_two(),
            "Evaluation list length must be a power of two."
        );

        Self(evals)
    }

    /// Given a multilinear point `P`, compute the evaluation vector of the equality function `eq(P, X)`
    /// for all points `X` in the boolean hypercube.
    #[inline]
    pub fn new_from_point(point: &MultilinearPoint<F>, value: F) -> Self {
        let n = point.num_variables();
        let mut evals = F::zero_vec(1 << n);
        eval_eq::<_, _, false>(point.as_slice(), &mut evals, value);
        Self(evals)
    }

    /// Evaluates the polynomial as a constant.
    /// This is only valid for constant polynomials (i.e., when `num_variables` is 0).
    ///
    /// Returns None in other cases.
    ///
    /// # Panics
    /// Panics if `num_variables` is not 0.
    #[must_use]
    #[inline]
    pub fn as_constant(&self) -> Option<F> {
        (self.num_evals() == 1).then_some(self.0[0])
    }

    /// Given a multilinear point `P`, compute the evaluation vector of the equality function `eq(P, X)`
    /// for all points `X` in the boolean hypercube and add it to the current evaluation vector.
    #[inline]
    pub fn accumulate(&mut self, point: &MultilinearPoint<F>, value: F) {
        assert_eq!(self.num_variables(), point.num_variables());
        eval_eq::<_, _, true>(point.as_slice(), &mut self.0, value);
    }

    /// Given a multilinear point `P`, compute the evaluation vector of the equality function `eq(P, X)`
    /// for all points `X` in the boolean hypercube and add it to the current evaluation vector.
    ///
    /// This is a variant of `accumulate` where the new point lies in a sub-field.
    #[inline]
    pub fn accumulate_base<BF: Field>(&mut self, point: &MultilinearPoint<BF>, value: F)
    where
        F: ExtensionField<BF>,
    {
        assert_eq!(self.num_variables(), point.num_variables());
        eval_eq_base::<_, _, true>(point.as_slice(), &mut self.0, value);
    }

    /// Returns the total number of stored evaluations.
    #[must_use]
    #[inline]
    pub const fn num_evals(&self) -> usize {
        self.0.len()
    }

    /// Returns the number of variables in the multilinear polynomial.
    #[must_use]
    #[inline]
    pub const fn num_variables(&self) -> usize {
        // Safety: The length is guaranteed to be a power of two.
        self.0.len().ilog2() as usize
    }

    /// Evaluates the multilinear polynomial at `point ∈ EF^n`.
    ///
    /// Computes
    /// ```text
    ///     f(point) = \sum_{x ∈ {0,1}^n} eq(x, point) * f(x),
    /// ```
    /// where
    /// ```text
    ///     eq(x, point) = \prod_{i=1}^{n} (1 - p_i + 2 p_i x_i).
    /// ```
    #[must_use]
    #[inline]
    pub fn evaluate<EF: ExtensionField<F>>(&self, point: &MultilinearPoint<EF>) -> EF {
        eval_multilinear(&self.0, point)
    }

    /// Folds a multilinear polynomial stored in evaluation form along the last `k` variables.
    ///
    /// Given evaluations `f: {0,1}^n → F`, this method returns a new evaluation list `g` such that:
    ///
    /// ```text
    ///     g(x_0, ..., x_{n-k-1}) = f(x_0, ..., x_{n-k-1}, r_0, ..., r_{k-1})
    /// ```
    ///
    /// # Arguments
    /// - `folding_randomness`: The extension-field values to substitute for the last `k` variables.
    ///
    /// # Returns
    /// - A new `EvaluationsList<EF>` representing the folded function over the remaining `n - k` variables.
    ///
    /// # Panics
    /// - If the evaluation list is not sized `2^n` for some `n`.
    #[inline]
    pub fn fold<EF>(&self, folding_randomness: &MultilinearPoint<EF>) -> EvaluationsList<EF>
    where
        EF: ExtensionField<F>,
    {
        let folding_factor = folding_randomness.num_variables();
        let evals = self
            .0
            .par_chunks_exact(1 << folding_factor)
            .map(|ev| eval_multilinear(ev, folding_randomness))
            .collect();

        EvaluationsList(evals)
    }

    /// Create a matrix representation of the evaluation list.
    #[inline]
    #[must_use]
    pub fn into_mat(self, width: usize) -> RowMajorMatrix<F> {
        RowMajorMatrix::new(self.0, width)
    }

    /// Returns a reference to the underlying slice of evaluations.
    #[inline]
    #[must_use]
    pub fn as_slice(&self) -> &[F] {
        &self.0
    }

    /// Returns an iterator over the evaluations.
    #[inline]
    pub fn iter(&self) -> std::slice::Iter<'_, F> {
        self.0.iter()
    }

    /// Compresses a list of evaluations in-place using a random challenge.
    ///
    /// ## Arguments
    /// * `evals`: A mutable reference to an `EvaluationsList<F>`, which will be modified in-place.
    /// * `r`: A value from the field `F`, used as the random folding challenge.
    ///
    /// ## Mathematical Formula
    /// The compression is achieved by applying the following formula to pairs of evaluations:
    /// ```text
    ///     p'(X_2, ..., X_n) = (p(1, X_2, ..., X_n) - p(0, X_2, ..., X_n)) \cdot r + p(0, X_2, ..., X_n)
    /// ```
    #[inline]
    pub fn compress(&mut self, r: F) {
        // Ensure the polynomial is not a constant (i.e., has variables to fold).
        assert_ne!(self.num_variables(), 0);

        // For large inputs, we use a parallel, out-of-place strategy.
        if self.num_evals() >= PARALLEL_THRESHOLD {
            // Define the folding operation for a pair of elements.
            let fold = |slice: &[F]| -> F { r * (slice[1] - slice[0]) + slice[0] };
            // Execute the fold in parallel and collect into a new vector.
            let folded = self.0.par_chunks_exact(2).map(fold).collect();
            // Replace the old evaluations with the new, folded evaluations.
            self.0 = folded;
        } else {
            // For smaller inputs, we use a sequential, in-place strategy.
            let mid = self.num_evals() / 2;
            for i in 0..mid {
                let p0 = self.0[2 * i];
                let p1 = self.0[2 * i + 1];
                self.0[i] = r * (p1 - p0) + p0;
            }
            self.0.truncate(mid);
        }
    }

    /// Folds a list of evaluations from a base field `F` into an extension field `EF`.
    ///
    /// ## Arguments
    /// * `r`: A value `r` from the extension field `EF`, used as the random challenge for folding.
    ///
    /// ## Returns
    /// A new `EvaluationsList<EF>` containing the compressed evaluations in the extension field.
    ///
    /// The compression is achieved by applying the following formula to pairs of evaluations:
    /// ```text
    ///     p'(X_2, ..., X_n) = (p(1, X_2, ..., X_n) - p(0, X_2, ..., X_n)) \cdot r + p(0, X_2, ..., X_n)
    /// ```
    #[inline]
    pub fn compress_ext<EF: ExtensionField<F>>(&self, r: EF) -> EvaluationsList<EF> {
        assert_ne!(self.num_variables(), 0);

        // Fold between base and extension field elements
        let fold = |slice: &[F]| -> EF { r * (slice[1] - slice[0]) + slice[0] };

        // Threshold below which sequential computation is faster
        //
        // This was chosen based on experiments with the `compress` function.
        // It is possible that the threshold can be tuned further.
        let folded = if self.num_evals() >= PARALLEL_THRESHOLD {
            self.0.par_chunks_exact(2).map(fold).collect()
        } else {
            self.0.chunks_exact(2).map(fold).collect()
        };

        EvaluationsList::new(folded)
    }
}

impl<'a, F> IntoIterator for &'a EvaluationsList<F> {
    type Item = &'a F;
    type IntoIter = std::slice::Iter<'a, F>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.0.iter()
    }
}

impl<F> IntoIterator for EvaluationsList<F> {
    type Item = F;
    type IntoIter = std::vec::IntoIter<F>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.0.into_iter()
    }
}

/// Evaluates a multilinear polynomial at an arbitrary point using fast interpolation.
///
/// It's given the polynomial's evaluations over all corners of the boolean hypercube `{0,1}^n` and
/// can find the value at any other point `p` (even with coordinates outside of `{0,1}`).
///
/// ## Algorithm
///
/// The core idea is to recursively reduce the number of variables by one at each step.
/// Imagine a 3D cube where we know the value at its 8 corners. To find the value at some
/// point `p` inside the cube, we can:
/// 1.  Find the values at the midpoints of the 4 edges along the x-axis.
/// 2.  Use those 4 points to find the values at the midpoints of the 2 "ribs" along the y-axis.
/// 3.  Finally, use those 2 points to find the single value at `p` along the z-axis.
///
/// This function implements this idea using the recurrence relation:
/// ```text
///     f(x_0, ..., x_{n-1}) = f_0(x_1, ..., x_{n-1}) * (1 - x_0) + f_1(x_1, ..., x_{n-1}) * x_0,
/// ```
/// where `f_0` is the polynomial with `x_0` fixed to `0` and `f_1` is with `x_0` fixed to `1`.
///
/// ## Implementation Strategies
///
/// To maximize performance, this function uses several strategies:
/// - **Hardcoded Paths:** For polynomials with 0 to 4 variables, the recursion is fully unrolled
///   into highly efficient, direct calculations.
/// - **Recursive Method:** For 5 to 19 variables, a standard recursive approach with a `rayon::join`
///   is used for parallelism on sufficiently large subproblems.
/// - **Non-Recursive Method:** For 20 or more variables, the algorithm switches to a non-recursive,
///   chunk-based method. This avoids deep recursion stacks and uses a memory access pattern
///   that is more friendly to parallelization.
///
/// ## Arguments
///
/// - `evals`: A slice containing the `2^n` evaluations of the polynomial over the boolean
///   hypercube, ordered lexicographically. For `n=2`, the order is `f(0,0), f(0,1), f(1,0), f(1,1)`.
/// - `point`: A slice containing the `n` coordinates of the point `p` at which to evaluate.
fn eval_multilinear<F, EF>(evals: &[F], point: &MultilinearPoint<EF>) -> EF
where
    F: Field,
    EF: ExtensionField<F>,
{
    // Ensure that the number of evaluations matches the number of variables in the point.
    //
    // This is a critical invariant: `evals.len()` must be exactly `2^point.len()`.
    debug_assert_eq!(evals.len(), 1 << point.num_variables());

    // Select the optimal evaluation strategy based on the number of variables.
    match point.as_slice() {
        // Case: 0 Variables (Constant Polynomial)
        //
        // A polynomial with zero variables is just a constant.
        [] => evals[0].into(),

        // Case: 1 Variable (Linear Interpolation)
        //
        // This is the base case for the recursion: f(x) = f(0) * (1-x) + f(1) * x.
        // The expression is an optimized form: f(0) + x * (f(1) - f(0)).
        [x] => *x * (evals[1] - evals[0]) + evals[0],

        // Case: 2 Variables (Bilinear Interpolation)
        //
        // This is a fully unrolled version for 2 variables, avoiding recursive calls.
        [x0, x1] => {
            // Interpolate along the x1-axis for x0=0 to get `a0`.
            let a0 = *x1 * (evals[1] - evals[0]) + evals[0];
            // Interpolate along the x1-axis for x0=1 to get `a1`.
            let a1 = *x1 * (evals[3] - evals[2]) + evals[2];
            // Finally, interpolate between `a0` and `a1` along the x0-axis.
            a0 + (a1 - a0) * *x0
        }

        // Cases: 3 and 4 Variables
        //
        // These are further unrolled versions for 3 and 4 variables for maximum speed.
        // The logic is the same as the 2-variable case, just with more steps.
        [x0, x1, x2] => {
            let a00 = *x2 * (evals[1] - evals[0]) + evals[0];
            let a01 = *x2 * (evals[3] - evals[2]) + evals[2];
            let a10 = *x2 * (evals[5] - evals[4]) + evals[4];
            let a11 = *x2 * (evals[7] - evals[6]) + evals[6];
            let a0 = a00 + *x1 * (a01 - a00);
            let a1 = a10 + *x1 * (a11 - a10);
            a0 + (a1 - a0) * *x0
        }
        [x0, x1, x2, x3] => {
            let a000 = *x3 * (evals[1] - evals[0]) + evals[0];
            let a001 = *x3 * (evals[3] - evals[2]) + evals[2];
            let a010 = *x3 * (evals[5] - evals[4]) + evals[4];
            let a011 = *x3 * (evals[7] - evals[6]) + evals[6];
            let a100 = *x3 * (evals[9] - evals[8]) + evals[8];
            let a101 = *x3 * (evals[11] - evals[10]) + evals[10];
            let a110 = *x3 * (evals[13] - evals[12]) + evals[12];
            let a111 = *x3 * (evals[15] - evals[14]) + evals[14];
            let a00 = a000 + *x2 * (a001 - a000);
            let a01 = a010 + *x2 * (a011 - a010);
            let a10 = a100 + *x2 * (a101 - a100);
            let a11 = a110 + *x2 * (a111 - a110);
            let a0 = a00 + *x1 * (a01 - a00);
            let a1 = a10 + *x1 * (a11 - a10);
            a0 + (a1 - a0) * *x0
        }

        // General Case (5+ Variables)
        //
        // This handles all other cases, using one of two different strategies.
        [x, tail @ ..] => {
            // For a very large number of variables we use a non-recursive algorithm better suited for wide parallelization.
            if point.num_variables() >= MLE_RECURSION_THRESHOLD {
                let mid = point.num_variables() / 2;
                let (hi, lo) = point.as_slice().split_at(mid);

                // Precomputation of Basis Polynomials
                //
                // The basis polynomial eq(v, p) can be split: eq(v, p) = eq(v_low, p_low) * eq(v_high, p_high).
                //
                // We precompute all `2^|lo|` values of eq(v_low, p_low) and store them in `left`.
                // We precompute all `2^|hi|` values of eq(v_high, p_high) and store them in `right`.

                // Allocate uninitialized memory for the polynomial evaluations.
                #[allow(clippy::uninit_vec)]
                let mut left = unsafe {
                    let mut vec = Vec::with_capacity(1 << lo.len());
                    vec.set_len(1 << lo.len());
                    vec
                };
                #[allow(clippy::uninit_vec)]
                let mut right = unsafe {
                    let mut vec = Vec::with_capacity(1 << hi.len());
                    vec.set_len(1 << hi.len());
                    vec
                };

                // Compute all eq(v_low, p_low) values and fill the `left` and `right` vectors.
                eval_eq::<_, _, false>(lo, &mut left, EF::ONE);
                eval_eq::<_, _, false>(hi, &mut right, EF::ONE);

                // Parallelized Final Summation
                //
                // This chain of operations computes the regrouped sum:
                // Σ_{v_high} eq(v_high, p_high) * (Σ_{v_low} f(v_high, v_low) * eq(v_low, p_low))
                evals
                    .par_chunks(left.len())
                    .zip_eq(right.par_iter())
                    .map(|(part, &c)| {
                        // This is the inner sum: a dot product between the evaluation chunk and the `left` basis values.
                        part.iter()
                            .zip_eq(left.iter())
                            .map(|(&a, &b)| b * a)
                            .sum::<EF>()
                            * c
                    })
                    .sum()
            } else {
                // Create a new point with the remaining coordinates.
                let sub_point = MultilinearPoint::new(tail.to_vec());

                // For moderately sized inputs (5 to 19 variables), use the recursive strategy.
                //
                // Split the evaluations into two halves, corresponding to the first variable being 0 or 1.
                let (f0, f1) = evals.split_at(evals.len() / 2);

                // Recursively evaluate on the two smaller hypercubes.
                let (f0_eval, f1_eval) = {
                    // Only spawn parallel tasks if the subproblem is large enough to overcome
                    // the overhead of threading.
                    let work_size: usize = (1 << 15) / std::mem::size_of::<F>();
                    if evals.len() > work_size {
                        join(
                            || eval_multilinear(f0, &sub_point),
                            || eval_multilinear(f1, &sub_point),
                        )
                    } else {
                        // For smaller subproblems, execute sequentially.
                        (
                            eval_multilinear(f0, &sub_point),
                            eval_multilinear(f1, &sub_point),
                        )
                    }
                };
                // Perform the final linear interpolation for the first variable `x`.
                f0_eval + (f1_eval - f0_eval) * *x
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use p3_baby_bear::BabyBear;
    use p3_field::extension::BinomialExtensionField;
    use p3_field::{PrimeCharacteristicRing, PrimeField64};
    use proptest::prelude::*;
    use rand::rngs::SmallRng;
    use rand::{Rng, SeedableRng};

    use super::*;

    type F = BabyBear;
    type EF4 = BinomialExtensionField<F, 4>;

    #[test]
    fn test_new_evaluations_list() {
        let evals = vec![F::ZERO, F::ONE, F::ZERO, F::ONE];
        let evaluations_list = EvaluationsList::new(evals.clone());

        assert_eq!(evaluations_list.num_evals(), evals.len());
        assert_eq!(evaluations_list.num_variables(), 2);
        assert_eq!(evaluations_list.as_slice(), &evals);
    }

    #[test]
    #[should_panic]
    fn test_new_evaluations_list_invalid_length() {
        // Length is not a power of two, should panic
        let _ = EvaluationsList::new(vec![F::ONE, F::ZERO, F::ONE]);
    }

    #[test]
    fn test_indexing() {
        let evals = vec![
            F::from_u64(1),
            F::from_u64(2),
            F::from_u64(3),
            F::from_u64(4),
        ];
        let evaluations_list = EvaluationsList::new(evals.clone());

        assert_eq!(evaluations_list.0[0], evals[0]);
        assert_eq!(evaluations_list.0[1], evals[1]);
        assert_eq!(evaluations_list.0[2], evals[2]);
        assert_eq!(evaluations_list.0[3], evals[3]);
    }

    #[test]
    #[should_panic]
    fn test_index_out_of_bounds() {
        let evals = vec![F::ZERO, F::ONE, F::ZERO, F::ONE];
        let evaluations_list = EvaluationsList::new(evals);

        let _ = evaluations_list.0[4]; // Index out of range, should panic
    }

    #[test]
    fn test_mutability_of_evals() {
        let mut evals = EvaluationsList::new(vec![F::ZERO, F::ONE, F::ZERO, F::ONE]);

        assert_eq!(evals.0[1], F::ONE);

        evals.0[1] = F::from_u64(5);

        assert_eq!(evals.0[1], F::from_u64(5));
    }

    #[test]
    fn test_evaluate_edge_cases() {
        let e1 = F::from_u64(7);
        let e2 = F::from_u64(8);
        let e3 = F::from_u64(9);
        let e4 = F::from_u64(10);

        let evals = EvaluationsList::new(vec![e1, e2, e3, e4]);

        // Evaluating at a binary hypercube point should return the direct value
        assert_eq!(
            evals.evaluate(&MultilinearPoint::new(vec![F::ZERO, F::ZERO])),
            e1
        );
        assert_eq!(
            evals.evaluate(&MultilinearPoint::new(vec![F::ZERO, F::ONE])),
            e2
        );
        assert_eq!(
            evals.evaluate(&MultilinearPoint::new(vec![F::ONE, F::ZERO])),
            e3
        );
        assert_eq!(
            evals.evaluate(&MultilinearPoint::new(vec![F::ONE, F::ONE])),
            e4
        );
    }

    #[test]
    fn test_num_evals() {
        let evals = EvaluationsList::new(vec![F::ONE, F::ZERO, F::ONE, F::ZERO]);
        assert_eq!(evals.num_evals(), 4);
    }

    #[test]
    fn test_num_variables() {
        let evals = EvaluationsList::new(vec![F::ONE, F::ZERO, F::ONE, F::ZERO]);
        assert_eq!(evals.num_variables(), 2);
    }

    #[test]
    fn test_eval_extension_on_non_hypercube_points() {
        let evals = EvaluationsList::new(vec![
            F::from_u64(1),
            F::from_u64(2),
            F::from_u64(3),
            F::from_u64(4),
        ]);

        let point = MultilinearPoint::new(vec![F::from_u64(2), F::from_u64(3)]);

        let result = evals.evaluate(&point);

        // Expected result using `eval_multilinear`
        let expected = eval_multilinear(evals.as_slice(), &point);

        assert_eq!(result, expected);
    }

    #[test]
    fn test_eval_multilinear_1d() {
        let a = F::from_u64(5);
        let b = F::from_u64(10);
        let evals = vec![a, b];

        // Evaluate at midpoint `x = 1/2`
        let x = F::from_u64(1) / F::from_u64(2);
        let expected = a + (b - a) * x;

        assert_eq!(
            eval_multilinear(&evals, &MultilinearPoint::new(vec![x])),
            expected
        );
    }

    #[test]
    fn test_eval_multilinear_2d() {
        let a = F::from_u64(1);
        let b = F::from_u64(2);
        let c = F::from_u64(3);
        let d = F::from_u64(4);

        // The evaluations are stored in lexicographic order for (x, y)
        // f(0,0) = a, f(0,1) = c, f(1,0) = b, f(1,1) = d
        let evals = vec![a, b, c, d];

        // Evaluate at `(x, y) = (1/2, 1/2)`
        let x = F::from_u64(1) / F::from_u64(2);
        let y = F::from_u64(1) / F::from_u64(2);

        // Interpolation formula:
        // f(x, y) = (1-x)(1-y) * f(0,0) + (1-x)y * f(0,1) + x(1-y) * f(1,0) + xy * f(1,1)
        let expected = (F::ONE - x) * (F::ONE - y) * a
            + (F::ONE - x) * y * c
            + x * (F::ONE - y) * b
            + x * y * d;

        assert_eq!(
            eval_multilinear(&evals, &MultilinearPoint::new(vec![x, y])),
            expected
        );
    }

    #[test]
    fn test_eval_multilinear_3d() {
        let a = F::from_u64(1);
        let b = F::from_u64(2);
        let c = F::from_u64(3);
        let d = F::from_u64(4);
        let e = F::from_u64(5);
        let f = F::from_u64(6);
        let g = F::from_u64(7);
        let h = F::from_u64(8);

        // The evaluations are stored in lexicographic order for (x, y, z)
        // f(0,0,0) = a, f(0,0,1) = c, f(0,1,0) = b, f(0,1,1) = e
        // f(1,0,0) = d, f(1,0,1) = f, f(1,1,0) = g, f(1,1,1) = h
        let evals = vec![a, b, c, e, d, f, g, h];

        let x = F::from_u64(1) / F::from_u64(3);
        let y = F::from_u64(1) / F::from_u64(3);
        let z = F::from_u64(1) / F::from_u64(3);

        // Using trilinear interpolation formula:
        let expected = (F::ONE - x) * (F::ONE - y) * (F::ONE - z) * a
            + (F::ONE - x) * (F::ONE - y) * z * c
            + (F::ONE - x) * y * (F::ONE - z) * b
            + (F::ONE - x) * y * z * e
            + x * (F::ONE - y) * (F::ONE - z) * d
            + x * (F::ONE - y) * z * f
            + x * y * (F::ONE - z) * g
            + x * y * z * h;

        assert_eq!(
            eval_multilinear(&evals, &MultilinearPoint::new(vec![x, y, z])),
            expected
        );
    }

    #[test]
    fn test_eval_multilinear_4d() {
        let a = F::from_u64(1);
        let b = F::from_u64(2);
        let c = F::from_u64(3);
        let d = F::from_u64(4);
        let e = F::from_u64(5);
        let f = F::from_u64(6);
        let g = F::from_u64(7);
        let h = F::from_u64(8);
        let i = F::from_u64(9);
        let j = F::from_u64(10);
        let k = F::from_u64(11);
        let l = F::from_u64(12);
        let m = F::from_u64(13);
        let n = F::from_u64(14);
        let o = F::from_u64(15);
        let p = F::from_u64(16);

        // Evaluations stored in lexicographic order for (x, y, z, w)
        let evals = vec![a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p];

        let x = F::from_u64(1) / F::from_u64(2);
        let y = F::from_u64(2) / F::from_u64(3);
        let z = F::from_u64(1) / F::from_u64(4);
        let w = F::from_u64(3) / F::from_u64(5);

        // Quadlinear interpolation formula
        let expected = (F::ONE - x) * (F::ONE - y) * (F::ONE - z) * (F::ONE - w) * a
            + (F::ONE - x) * (F::ONE - y) * (F::ONE - z) * w * b
            + (F::ONE - x) * (F::ONE - y) * z * (F::ONE - w) * c
            + (F::ONE - x) * (F::ONE - y) * z * w * d
            + (F::ONE - x) * y * (F::ONE - z) * (F::ONE - w) * e
            + (F::ONE - x) * y * (F::ONE - z) * w * f
            + (F::ONE - x) * y * z * (F::ONE - w) * g
            + (F::ONE - x) * y * z * w * h
            + x * (F::ONE - y) * (F::ONE - z) * (F::ONE - w) * i
            + x * (F::ONE - y) * (F::ONE - z) * w * j
            + x * (F::ONE - y) * z * (F::ONE - w) * k
            + x * (F::ONE - y) * z * w * l
            + x * y * (F::ONE - z) * (F::ONE - w) * m
            + x * y * (F::ONE - z) * w * n
            + x * y * z * (F::ONE - w) * o
            + x * y * z * w * p;

        // Validate against the function output
        assert_eq!(
            eval_multilinear(&evals, &MultilinearPoint::new(vec![x, y, z, w])),
            expected
        );
    }

    proptest! {
        #[test]
        fn prop_eval_multilinear_equiv_between_f_and_ef4(
            values in prop::collection::vec(0u64..100, 8),
            x0 in 0u64..100,
            x1 in 0u64..100,
            x2 in 0u64..100,
        ) {
            // Base field evaluations
            let coeffs_f: Vec<F> = values.iter().copied().map(F::from_u64).collect();
            let poly_f = EvaluationsList::new(coeffs_f);

            // Lift to extension field EF4
            let coeffs_ef: Vec<EF4> = values.iter().copied().map(EF4::from_u64).collect();
            let poly_ef = EvaluationsList::new(coeffs_ef);

            // Evaluation point in EF4
            let point_ef = MultilinearPoint::new(vec![
                EF4::from_u64(x0),
                EF4::from_u64(x1),
                EF4::from_u64(x2),
            ]);

            // Evaluate using both base and extension representations
            let eval_f = poly_f.evaluate(&point_ef);
            let eval_ef = poly_ef.evaluate(&point_ef);

            prop_assert_eq!(eval_f, eval_ef);
        }
    }

    #[test]
    fn test_multilinear_eval_two_vars() {
        // Define a simple 2-variable multilinear polynomial:
        //
        // Variables: X₁, X₂
        // Coefficients ordered in lexicographic order: (X₁, X₂)
        //
        // - coeffs[0] → constant term
        // - coeffs[1] → X₂ term
        // - coeffs[2] → X₁ term
        // - coeffs[3] → X₁·X₂ term
        //
        // Thus, the polynomial is:
        //
        //   f(X₁, X₂) = c0 + c1·X₂ + c2·X₁ + c3·X₁·X₂
        //
        // where:
        let c0 = F::from_u64(5); // constant
        let c1 = F::from_u64(6); // X₂ coefficient
        let c2 = F::from_u64(7); // X₁ coefficient
        let c3 = F::from_u64(8); // X₁·X₂ coefficient

        let f = |x0, x1| c0 + c1 * x1 + c2 * x0 + c3 * x0 * x1;

        // Convert coefficients to evaluations
        let evals = EvaluationsList::new(vec![
            f(F::ZERO, F::ZERO),
            f(F::ZERO, F::ONE),
            f(F::ONE, F::ZERO),
            f(F::ONE, F::ONE),
        ]);

        // Choose evaluation point:
        //
        // Let's pick (x₁, x₂) = (2, 1)
        let x1 = F::from_u64(2);
        let x2 = F::from_u64(1);
        let coords = MultilinearPoint::new(vec![x1, x2]);

        // Manually compute the expected value step-by-step:
        //
        // Reminder:
        //   f(X₁, X₂) = 5 + 6·X₂ + 7·X₁ + 8·X₁·X₂
        //
        // Substituting (X₁, X₂):
        let expected = c0 + c1 * x2 + c2 * x1 + c3 * x1 * x2;

        // Now evaluate using the function under test
        let result = evals.evaluate(&coords);

        // Check that it matches the manual computation
        assert_eq!(result, expected);
    }

    #[test]
    fn test_evaluate_3_variables() {
        // Define a multilinear polynomial in 3 variables: X₀, X₁, X₂
        //
        // Coefficients ordered lex in index order:
        //
        // - coeffs[0] → constant term
        // - coeffs[1] → X₂
        // - coeffs[2] → X₁
        // - coeffs[3] → X₁·X₂
        // - coeffs[4] → X₀
        // - coeffs[5] → X₀·X₂
        // - coeffs[6] → X₀·X₁
        // - coeffs[7] → X₀·X₁·X₂
        //
        // Thus:
        //    f(X₀,X₁,X₂) = c0 + c1·X₂ + c2·X₁ + c3·X₁·X₂
        //                + c4·X₀ + c5·X₀·X₂ + c6·X₀·X₁ + c7·X₀·X₁·X₂
        let c0 = F::from_u64(1);
        let c1 = F::from_u64(2);
        let c2 = F::from_u64(3);
        let c3 = F::from_u64(4);
        let c4 = F::from_u64(5);
        let c5 = F::from_u64(6);
        let c6 = F::from_u64(7);
        let c7 = F::from_u64(8);

        // Define the polynomial as a closure for clarity
        let f = |x0, x1, x2| {
            c0 + c1 * x2
                + c2 * x1
                + c3 * x1 * x2
                + c4 * x0
                + c5 * x0 * x2
                + c6 * x0 * x1
                + c7 * x0 * x1 * x2
        };

        let evals = EvaluationsList::new(vec![
            f(F::ZERO, F::ZERO, F::ZERO), // f(0,0,0)
            f(F::ZERO, F::ZERO, F::ONE),  // f(0,0,1)
            f(F::ZERO, F::ONE, F::ZERO),  // f(0,1,0)
            f(F::ZERO, F::ONE, F::ONE),   // f(0,1,1)
            f(F::ONE, F::ZERO, F::ZERO),  // f(1,0,0)
            f(F::ONE, F::ZERO, F::ONE),   // f(1,0,1)
            f(F::ONE, F::ONE, F::ZERO),   // f(1,1,0)
            f(F::ONE, F::ONE, F::ONE),    // f(1,1,1)
        ]);

        // Pick point: (x₀,x₁,x₂) = (2, 3, 4)
        let x0 = F::from_u64(2);
        let x1 = F::from_u64(3);
        let x2 = F::from_u64(4);

        let point = MultilinearPoint::new(vec![x0, x1, x2]);

        // Manually compute:
        //
        // expected = 1
        //          + 2·4
        //          + 3·3
        //          + 4·3·4
        //          + 5·2
        //          + 6·2·4
        //          + 7·2·3
        //          + 8·2·3·4
        let expected = c0
            + c1 * x2
            + c2 * x1
            + c3 * x1 * x2
            + c4 * x0
            + c5 * x0 * x2
            + c6 * x0 * x1
            + c7 * x0 * x1 * x2;

        let result = evals.evaluate(&point);
        assert_eq!(result, expected);
    }

    #[test]
    fn test_eval_extension_3_variables() {
        // Define a multilinear polynomial in 3 variables: X₀, X₁, X₂
        //
        // Coefficients ordered lex in index order:
        //
        // - coeffs[0] → constant term
        // - coeffs[1] → X₂ term
        // - coeffs[2] → X₁ term
        // - coeffs[3] → X₁·X₂ term
        // - coeffs[4] → X₀ term
        // - coeffs[5] → X₀·X₂ term
        // - coeffs[6] → X₀·X₁ term
        // - coeffs[7] → X₀·X₁·X₂ term
        //
        // Thus:
        //    f(X₀,X₁,X₂) = c0 + c1·X₂ + c2·X₁ + c3·X₁·X₂
        //                + c4·X₀ + c5·X₀·X₂ + c6·X₀·X₁ + c7·X₀·X₁·X₂
        let c0 = F::from_u64(1);
        let c1 = F::from_u64(2);
        let c2 = F::from_u64(3);
        let c3 = F::from_u64(4);
        let c4 = F::from_u64(5);
        let c5 = F::from_u64(6);
        let c6 = F::from_u64(7);
        let c7 = F::from_u64(8);

        let f = |x0, x1, x2| {
            c0 + c1 * x2
                + c2 * x1
                + c3 * x1 * x2
                + c4 * x0
                + c5 * x0 * x2
                + c6 * x0 * x1
                + c7 * x0 * x1 * x2
        };

        let evals = EvaluationsList::new(vec![
            f(F::ZERO, F::ZERO, F::ZERO), // f(0,0,0)
            f(F::ZERO, F::ZERO, F::ONE),  // f(0,0,1)
            f(F::ZERO, F::ONE, F::ZERO),  // f(0,1,0)
            f(F::ZERO, F::ONE, F::ONE),   // f(0,1,1)
            f(F::ONE, F::ZERO, F::ZERO),  // f(1,0,0)
            f(F::ONE, F::ZERO, F::ONE),   // f(1,0,1)
            f(F::ONE, F::ONE, F::ZERO),   // f(1,1,0)
            f(F::ONE, F::ONE, F::ONE),    // f(1,1,1)
        ]);

        // Choose evaluation point: (x₀, x₁, x₂) = (2, 3, 4)
        //
        // Here we lift into the extension field EF4
        let x0 = EF4::from_u64(2);
        let x1 = EF4::from_u64(3);
        let x2 = EF4::from_u64(4);

        let point = MultilinearPoint::new(vec![x0, x1, x2]);

        // Manually compute expected value
        //
        // Substituting (X₀,X₁,X₂) = (2,3,4) into:
        //
        //   f(X₀,X₁,X₂) = 1
        //               + 2·4
        //               + 3·3
        //               + 4·3·4
        //               + 5·2
        //               + 6·2·4
        //               + 7·2·3
        //               + 8·2·3·4
        //
        // and lifting each constant into EF4 for correct typing
        let expected = EF4::from(c0)
            + EF4::from(c1) * x2
            + EF4::from(c2) * x1
            + EF4::from(c3) * x1 * x2
            + EF4::from(c4) * x0
            + EF4::from(c5) * x0 * x2
            + EF4::from(c6) * x0 * x1
            + EF4::from(c7) * x0 * x1 * x2;

        // Evaluate via `evaluate` method
        let result = evals.evaluate(&point);

        // Verify that result matches manual computation
        assert_eq!(result, expected);
    }

    #[test]
    fn test_folding_and_evaluation() {
        // Set number of Boolean input variables n = 10.
        let num_variables = 10;

        // Create a multilinear polynomial
        let evals = (0..(1 << num_variables)).map(F::from_u64).collect();
        let evals_list = EvaluationsList::new(evals);

        // Define a fixed evaluation point in F^n: [0, 35, 70, ..., 35*(n-1)]
        let randomness: Vec<_> = (0..num_variables)
            .map(|i| F::from_u64(35 * i as u64))
            .collect();

        // Try folding at every possible prefix of the randomness vector: k = 0 to n-1
        for k in 0..num_variables {
            // Use the first k values as the fold coordinates.
            // NOTE: The logic in the original test had a small bug. It should fold over the *last*
            // k variables, so we take the folding randomness from the end of the point.
            let fold_part = randomness[num_variables - k..].to_vec();

            // The remaining coordinates are used as the evaluation input into the folded poly.
            let eval_part = randomness[..num_variables - k].to_vec();

            // Convert to a MultilinearPoint for folding
            let fold_random = MultilinearPoint::new(fold_part);

            // The full, original point
            let eval_point = MultilinearPoint::new(randomness.clone());

            // Fold the evaluation list over the last `k` variables
            let folded_evals = evals_list.fold(&fold_random);

            // Verify that the number of variables has been folded correctly
            assert_eq!(folded_evals.num_variables(), num_variables - k);

            // Verify correctness: folding and then evaluating the partial point `e`
            // should be the same as evaluating the original list at the full point `[e, r]`.
            assert_eq!(
                folded_evals.evaluate(&MultilinearPoint::new(eval_part)),
                evals_list.evaluate(&eval_point)
            );
        }
    }

    #[test]
    fn test_fold_with_extension_one_var() {
        // Define a 2-variable polynomial:
        // f(X₀, X₁) = 1 + 2·X₁ + 3·X₀ + 4·X₀·X₁
        let c0 = F::from_u64(1); // constant
        let c1 = F::from_u64(2); // X₁
        let c2 = F::from_u64(3); // X₀
        let c3 = F::from_u64(4); // X₀·X₁

        let f = |x0, x1| c0 + c1 * x1 + c2 * x0 + c3 * x0 * x1;

        let evals_list = EvaluationsList::new(vec![
            f(F::ZERO, F::ZERO), // f(0,0)
            f(F::ZERO, F::ONE),  // f(0,1)
            f(F::ONE, F::ZERO),  // f(1,0)
            f(F::ONE, F::ONE),   // f(1,1)
        ]);

        // We fold over the last variable (X₁) by setting X₁ = 5 in EF4
        let r1 = EF4::from_u64(5);

        // Perform the fold: f(X₀, 5) becomes a new function g(X₀)
        let folded = evals_list.fold(&MultilinearPoint::new(vec![r1]));

        // For 10 test points x₀ = 0, 1, ..., 9
        for x0_f in 0..10 {
            // Lift to EF4 for extension-field evaluation
            let x0 = EF4::from_u64(x0_f);

            // Construct the full point (x₀, X₁ = 5)
            let full_point = MultilinearPoint::new(vec![x0, r1]);

            // Construct folded point (x₀)
            let folded_point = MultilinearPoint::new(vec![x0]);

            // Evaluate original evals_list at (x₀, 5) to get the ground truth.
            let expected = evals_list.evaluate(&full_point);

            // Evaluate folded poly at x₀
            let actual = folded.evaluate(&folded_point);

            // Ensure the results agree
            assert_eq!(expected, actual);
        }
    }

    proptest! {
        #[test]
        fn prop_eval_eq_matches_naive_for_eval_list(
            // number of variables (keep small to avoid blowup)
            n in 1usize..5,
             // always at least 5 elements
            evals_raw in prop::collection::vec(0u64..F::ORDER_U64, 5),
        ) {
            // Slice out exactly n elements, guaranteed present
            let evals: Vec<F> = evals_raw[..n].iter().map(|&x| F::from_u64(x)).collect();

            // Allocate output buffer of size 2^n
            let mut out = vec![F::ZERO; 1 << n];

            // Run eval_eq with scalar = 1
            eval_eq::<F, F, false>(&evals, &mut out, F::ONE);

            // Naively compute expected values for each binary assignment
            let mut expected = vec![F::ZERO; 1 << n];
            for (i, e) in expected.iter_mut().enumerate().take(1 << n) {
                let mut weight = F::ONE;
                for (j, &val) in evals.iter().enumerate() {
                    let bit = (i >> (n - 1 - j)) & 1;
                    if bit == 1 {
                        weight *= val;
                    } else {
                        weight *= F::ONE - val;
                    }
                }
                *e = weight;
            }

            prop_assert_eq!(out, expected);
        }
    }

    #[test]
    fn test_eval_multilinear_large_input_brute_force() {
        // Define the number of variables.
        //
        // We use 20 to trigger the case where the recursive algorithm is not optimal.
        const NUM_VARS: usize = 20;

        // Use a seeded random number generator for a reproducible test case.
        let mut rng = SmallRng::seed_from_u64(42);

        // The number of evaluations on the boolean hypercube is 2^n.
        let num_evals = 1 << NUM_VARS;

        // Create a vector of random evaluations for our polynomial `f`.
        let evals_vec: Vec<F> = (0..num_evals).map(|_| rng.random()).collect();
        let evals_list = EvaluationsList::new(evals_vec);

        // Create a random point `p` where we will evaluate the polynomial.
        let point_vec: Vec<EF4> = (0..NUM_VARS).map(|_| rng.random()).collect();
        let point = MultilinearPoint::new(point_vec);

        // BRUTE-FORCE CALCULATION (GROUND TRUTH)
        //
        // We will now calculate the expected result using the fundamental formula:
        // f(p) = Σ_{v ∈ {0,1}^n} f(v) * eq(v, p)
        // where eq(v, p) = Π_{i=0..n-1} (v_i*p_i + (1-v_i)*(1-p_i))

        // This variable will accumulate the sum. It must be in the extension field.
        let mut expected_sum = EF4::ZERO;

        // Iterate through every point `v` on the boolean hypercube {0,1}^20.
        //
        // The loop counter `i` represents the integer value of the bit-string for `v`.
        for i in 0..num_evals {
            // This will hold the eq(v, p) value for the current hypercube point `v`.
            let mut eq_term = EF4::ONE;

            // To build eq(v, p), we iterate through each dimension of the hypercube.
            for j in 0..NUM_VARS {
                // Get the j-th bit of `i`. This corresponds to the coordinate v_j.
                // We read bits from most-significant to least-significant to match the
                // lexicographical ordering of the `evals_list`.
                let v_j = (i >> (NUM_VARS - 1 - j)) & 1;

                // Get the corresponding j-th coordinate of our evaluation point `p`.
                let p_j = point.as_slice()[j];

                if v_j == 1 {
                    // If the hypercube coordinate v_j is 1, the factor is p_j.
                    eq_term *= p_j;
                } else {
                    // If the hypercube coordinate v_j is 0, the factor is (1 - p_j).
                    eq_term *= EF4::ONE - p_j;
                }
            }

            // Get the pre-computed evaluation f(v) from our list. The index `i`
            // directly corresponds to the lexicographically ordered point `v`.
            let f_v = evals_list.0[i];

            // Add the term f(v) * eq(v, p) to the total sum. We must lift `f_v` from the
            // base field `F` to the extension field `EF4` for the multiplication.
            expected_sum += eq_term * f_v;
        }

        // Now, run the optimized function that we want to test.
        let actual_result = evals_list.evaluate(&point);

        // Finally, assert that the results are equal.
        assert_eq!(actual_result, expected_sum);
    }

    #[test]
    fn test_new_from_point_zero_vars() {
        let point = MultilinearPoint::<F>::new(vec![]);
        let value = F::from_u64(42);
        let evals_list = EvaluationsList::new_from_point(&point, value);

        // For n=0, the hypercube has one point, and the `eq` polynomial is the constant 1.
        // The result should be a list with a single element: `value`.
        assert_eq!(evals_list.num_variables(), 0);
        assert_eq!(evals_list.as_slice(), &[value]);
    }

    #[test]
    fn test_new_from_point_one_var() {
        let p0 = F::from_u64(7);
        let point = MultilinearPoint::new(vec![p0]);
        let value = F::from_u64(3);
        let evals_list = EvaluationsList::new_from_point(&point, value);

        // For a point `p = [p0]`, the `eq` evaluations over `X={0,1}` are:
        // - eq(p, 0) = 1 - p0
        // - eq(p, 1) = p0
        // These are then scaled by `value`.
        let expected = vec![value * (F::ONE - p0), value * p0];

        assert_eq!(evals_list.num_variables(), 1);
        assert_eq!(evals_list.as_slice(), &expected);
    }

    #[test]
    #[allow(clippy::identity_op)]
    fn test_new_from_point_three_vars() {
        let p = [F::from_u64(2), F::from_u64(3), F::from_u64(5)];
        let point = MultilinearPoint::new(p.to_vec());
        let value = F::from_u64(10);
        let evals_list = EvaluationsList::new_from_point(&point, value);

        // Manually compute the expected result for eq(p, b) * value for all 8 points `b`.
        // The implementation's lexicographical order means the index `i` is formed as
        // i = 4*b0 + 2*b1 + 1*b2, where `b` is the hypercube point (b0, b1, b2).
        let mut expected = Vec::with_capacity(8);
        for i in 0..8 {
            // We extract the bits of `i` to determine the coordinates of the hypercube point `b`.
            //
            // MSB of `i` corresponds to the first variable, p[0].
            let b0 = (i >> 2) & 1;
            // Middle bit of `i` corresponds to the second variable, p[1].
            let b1 = (i >> 1) & 1;
            // LSB of `i` corresponds to the last variable, p[2].
            let b2 = (i >> 0) & 1;

            // Calculate the eq(p, b) term for this specific point `b`.
            let term0 = if b0 == 1 { p[0] } else { F::ONE - p[0] };
            let term1 = if b1 == 1 { p[1] } else { F::ONE - p[1] };
            let term2 = if b2 == 1 { p[2] } else { F::ONE - p[2] };

            expected.push(value * term0 * term1 * term2);
        }

        assert_eq!(evals_list.num_variables(), 3);
        assert_eq!(evals_list.as_slice(), &expected);
    }

    #[test]
    fn test_as_constant_for_constant_poly() {
        // A polynomial with 0 variables is a constant. Its evaluation
        // list contains a single value.
        let constant_value = F::from_u64(42);
        let evals = EvaluationsList::new(vec![constant_value]);

        // `as_constant` should return the value wrapped in `Some`.
        assert_eq!(evals.num_variables(), 0);
        assert_eq!(evals.as_constant(), Some(constant_value));
    }

    #[test]
    fn test_as_constant_for_non_constant_poly() {
        // A polynomial with 2 variables is not a constant. Its evaluation
        // list has 2^2 = 4 entries.
        let evals = EvaluationsList::new(vec![F::ONE, F::ZERO, F::ONE, F::ZERO]);

        // For any non-constant polynomial, `as_constant` should return `None`.
        assert_ne!(evals.num_variables(), 0);
        assert_eq!(evals.as_constant(), None);
    }

    #[test]
    #[allow(clippy::identity_op)]
    fn test_accumulate() {
        // Set up an initial list of evaluations.
        let n = 2;
        let initial_values = vec![
            F::from_u64(10),
            F::from_u64(20),
            F::from_u64(30),
            F::from_u64(40),
        ];
        let mut evals_list = EvaluationsList::new(initial_values.clone());

        // Define the point and value to accumulate.
        let p = [F::from_u64(2), F::from_u64(3)];
        let point = MultilinearPoint::new(p.to_vec());
        let value = F::from_u64(5);

        // Manually compute the `eq` evaluations that should be added.
        let mut eq_evals_to_add = Vec::with_capacity(1 << n);
        for i in 0..(1 << n) {
            let b0 = (i >> 1) & 1; // MSB for p[0]
            let b1 = (i >> 0) & 1; // LSB for p[1]
            let term0 = if b0 == 1 { p[0] } else { F::ONE - p[0] };
            let term1 = if b1 == 1 { p[1] } else { F::ONE - p[1] };
            eq_evals_to_add.push(value * term0 * term1);
        }

        // Calculate the final expected evaluations after addition.
        let expected: Vec<F> = initial_values
            .iter()
            .zip(eq_evals_to_add.iter())
            .map(|(&initial, &to_add)| initial + to_add)
            .collect();

        // Call accumulate and assert that the result matches the expected sum.
        evals_list.accumulate(&point, value);
        assert_eq!(evals_list.as_slice(), &expected);
    }

    #[test]
    #[allow(clippy::identity_op)]
    fn test_accumulate_base() {
        // Set up an initial list of evaluations.
        let n = 2;
        // Initial evaluations in the extension field.
        let initial_values: Vec<EF4> = vec![
            EF4::from_u64(10),
            EF4::from_u64(20),
            EF4::from_u64(30),
            EF4::from_u64(40),
        ];
        let mut evals_list = EvaluationsList::new(initial_values.clone());

        // Point in the base field `F`, value in the extension field `EF4`.
        let p_base = [F::from_u64(2), F::from_u64(3)];
        let point_base = MultilinearPoint::new(p_base.to_vec());
        let value_ext = EF4::from_u64(5);

        // Manually compute `eq` evals,
        // lifting base field elements to extension field.
        let mut eq_evals_to_add = Vec::with_capacity(1 << n);
        for i in 0..(1 << n) {
            let b0 = (i >> 1) & 1; // MSB
            let b1 = (i >> 0) & 1; // LSB
            let term0 = if b0 == 1 {
                EF4::from(p_base[0])
            } else {
                EF4::ONE - EF4::from(p_base[0])
            };
            let term1 = if b1 == 1 {
                EF4::from(p_base[1])
            } else {
                EF4::ONE - EF4::from(p_base[1])
            };
            eq_evals_to_add.push(value_ext * term0 * term1);
        }

        // Calculate the final expected sum in the extension field.
        let expected: Vec<EF4> = initial_values
            .iter()
            .zip(eq_evals_to_add.iter())
            .map(|(&initial, &to_add)| initial + to_add)
            .collect();

        // Accumulate and assert the result.
        evals_list.accumulate_base(&point_base, value_ext);
        assert_eq!(evals_list.as_slice(), &expected);
    }

    #[test]
    fn test_compress() {
        let initial_evals: Vec<F> = (1..=8).map(F::from_u64).collect();
        let mut evals_list = EvaluationsList::new(initial_evals);
        let r = F::from_u64(10);

        // Manually compute the expected folded values using the formula:
        // p' = r * (p1 - p0) + p0
        let expected = vec![
            r * (F::from_u64(2) - F::from_u64(1)) + F::from_u64(1),
            r * (F::from_u64(4) - F::from_u64(3)) + F::from_u64(3),
            r * (F::from_u64(6) - F::from_u64(5)) + F::from_u64(5),
            r * (F::from_u64(8) - F::from_u64(7)) + F::from_u64(7),
        ];

        // The method modifies the list in-place.
        evals_list.compress(r);

        assert_eq!(evals_list.num_variables(), 2);
        assert_eq!(evals_list.num_evals(), 4);
        assert_eq!(evals_list.as_slice(), &expected);
    }

    #[test]
    #[should_panic]
    fn test_compress_panics_on_constant() {
        // A constant polynomial has 0 variables and cannot be compressed.
        let mut evals_list = EvaluationsList::new(vec![F::from_u64(42)]);
        evals_list.compress(F::ONE); // This should panic.
    }

    #[test]
    fn test_compress_ext() {
        // This test verifies the out-of-place compression into an extension field.
        let initial_evals: Vec<F> = (1..=8).map(F::from_u64).collect();
        let evals_list = EvaluationsList::new(initial_evals);
        let r_ext = EF4::from_u64(10);

        // The expected result is the same as `test_compress`, but with elements
        // lifted into the extension field EF4.
        let expected: Vec<EF4> = vec![
            r_ext * (EF4::from_u64(2) - EF4::from_u64(1)) + EF4::from_u64(1),
            r_ext * (EF4::from_u64(4) - EF4::from_u64(3)) + EF4::from_u64(3),
            r_ext * (EF4::from_u64(6) - EF4::from_u64(5)) + EF4::from_u64(5),
            r_ext * (EF4::from_u64(8) - EF4::from_u64(7)) + EF4::from_u64(7),
        ];

        // The method returns a new list and does not modify the original.
        let compressed_ext_list = evals_list.compress_ext(r_ext);

        assert_eq!(compressed_ext_list.num_variables(), 2);
        assert_eq!(compressed_ext_list.num_evals(), 4);
        assert_eq!(compressed_ext_list.as_slice(), &expected);
    }

    #[test]
    #[should_panic]
    fn test_compress_ext_panics_on_constant() {
        // A constant polynomial has 0 variables and cannot be compressed.
        let evals_list = EvaluationsList::new(vec![F::from_u64(42)]);
        let _ = evals_list.compress_ext(EF4::ONE); // This should panic.
    }

    proptest! {
        #[test]
        fn prop_compress_and_compress_ext_agree(
            n in 1..=6,
            seed in any::<u64>(),
        ) {
            let mut rng = SmallRng::seed_from_u64(seed);
            let num_evals = 1 << n;
            let evals: Vec<F> = (0..num_evals).map(|_| rng.random()).collect();
            let r_base: F = rng.random();

            // Path A: Use the in-place `compress` method.
            let mut list_a = EvaluationsList::new(evals.clone());
            list_a.compress(r_base);
            // Lift the result into the extension field for comparison.
            let result_a_lifted: Vec<EF4> = list_a.as_slice().iter().map(|&x| EF4::from(x)).collect();

            // Path B: Use the `compress_ext` method with the same challenge, lifted.
            let list_b = EvaluationsList::new(evals);
            let r_ext = EF4::from(r_base);
            let result_b_ext = list_b.compress_ext(r_ext);

            // The results should be identical.
            prop_assert_eq!(result_a_lifted, result_b_ext.as_slice());
        }
    }
}
