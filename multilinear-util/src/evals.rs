use alloc::vec;
use alloc::vec::Vec;

use itertools::Itertools;
use p3_field::{
    Algebra, ExtensionField, Field, PackedFieldExtension, PackedValue, PrimeCharacteristicRing,
};
use p3_matrix::dense::RowMajorMatrixView;
use p3_maybe_rayon::prelude::*;
use p3_util::log2_strict_usize;
use serde::{Deserialize, Serialize};
use tracing::instrument;

use crate::eq_batch::eval_eq_batch;
use crate::multilinear::MultilinearPoint;

const PARALLEL_THRESHOLD: usize = 4096;

/// Number of variables at which we switch from recursive to chunk-based MLE evaluation.
const MLE_RECURSION_THRESHOLD: usize = 20;

/// Returns a vector of uninitialized elements of type `A` with the specified length.
///
/// # Safety
///
/// Entries should be overwritten before use.
#[must_use]
unsafe fn uninitialized_vec<A>(len: usize) -> Vec<A> {
    #[allow(clippy::uninit_vec)]
    unsafe {
        let mut vec = Vec::with_capacity(len);
        vec.set_len(len);
        vec
    }
}

/// Represents a multilinear polynomial `f` in `n` variables, stored by its evaluations
/// over the boolean hypercube `{0,1}^n`.
///
/// The inner vector stores function evaluations at points of the hypercube in lexicographic
/// order. The number of variables `n` is inferred from the length of this vector, where
/// `self.len() = 2^n`.
#[allow(clippy::unsafe_derive_deserialize)]
#[derive(Debug, Clone, Eq, PartialEq, Serialize, Deserialize)]
#[must_use]
pub struct EvaluationsList<F>(pub(crate) Vec<F>);

impl<F: Copy + Clone + Send + Sync> EvaluationsList<F> {
    /// Given a number of points initializes a new zero polynomial
    #[inline]
    pub fn zero(num_variables: usize) -> Self
    where
        F: PrimeCharacteristicRing,
    {
        Self(F::zero_vec(1 << num_variables))
    }

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

    /// Returns a reference to the underlying slice of evaluations.
    #[inline]
    #[must_use]
    pub fn as_slice(&self) -> &[F] {
        &self.0
    }

    /// Returns an iterator over the evaluations.
    #[inline]
    pub fn iter(&self) -> core::slice::Iter<'_, F> {
        self.0.iter()
    }
}

impl<A: Clone + Copy + Default + Send + Sync> EvaluationsList<A> {
    /// Compresses the evaluation list by folding the **first** variable ($X_1$) with a challenge.
    ///
    /// This function is the core operation for the standard rounds of a sumcheck prover,
    /// where variables are folded one by one in lexicographical order.
    ///
    /// ## Mathematical Formula
    ///
    /// Given a polynomial $p(X_1, \ldots, X_n)$ represented by its evaluations, this
    /// function computes the evaluations of the folded
    /// polynomial $p'(X_2, \ldots, X_n) = p(r, X_2, \ldots, X_n)$.
    ///
    /// It uses the multilinear extension formula for the first variable:
    ///
    /// ```text
    /// p(r, x') = p(0, x') + r \cdot (p(1, x') - p(0, x'))
    /// ```
    ///
    /// where $x' = (x_2, \ldots, x_n)$ represents all other variables.
    ///
    /// ## Memory Access Pattern
    ///
    /// This function relies on the **lexicographical order** of the evaluation list.
    /// - The first half of the slice contains all evaluations where $X_1 = 0$,
    /// - The second half contains all evaluations where $X_1 = 1$.
    ///
    /// ```text
    /// Before:
    /// [ p(0, 0..0), p(0, 0..1), ..., p(0, 1..1) | p(1, 0..0), p(1, 0..1), ..., p(1, 1..1) ]
    ///  └────────── Left Half (p(0, x')) ──────┘   └────────── Right Half (p(1, x')) ────┘
    ///
    /// After: (Computed in-place into the left half)
    /// [ p(r, 0..0), p(r, 0..1), ..., p(r, 1..1) ]
    ///   └───────── Folded result ─────────────┘
    /// ```
    ///
    /// The function computes `result[i] = left[i] + r * (right[i] - left[i])` for
    /// all `i` in the first half, and then truncates the list.
    pub fn compress<F: Clone + Copy + Default + Send + Sync>(&mut self, r: F)
    where
        A: Algebra<F>,
    {
        assert_ne!(self.num_variables(), 0);
        let num_evals = self.num_evals();
        let mid = num_evals / 2;

        // Evaluations at `a_i` and `a_{i + n/2}` slots are folded with `r` into `a_i` slot
        let (p0, p1) = self.0.split_at_mut(mid);
        if num_evals >= PARALLEL_THRESHOLD {
            p0.par_iter_mut()
                .zip(p1.par_iter())
                .for_each(|(a0, &a1)| *a0 += (a1 - *a0) * r);
        } else {
            p0.iter_mut()
                .zip(p1.iter())
                .for_each(|(a0, &a1)| *a0 += (a1 - *a0) * r);
        }
        // Free higher part of the evaluations
        self.0.truncate(mid);
    }

    /// Folds a list of evaluations from a base field `F` into packed form of extension field `EF`.
    ///
    /// ## Arguments
    /// * `r`: A value `r` from the extension field `EF`, used as the random challenge for folding.
    ///
    /// ## Returns
    /// A new `EvaluationsList<EF::ExtensionPacking>` containing the compressed evaluations in the extension field.
    ///
    /// The compression is achieved by applying the following formula to pairs of evaluations:
    /// ```text
    ///     p'(X_2, ..., X_n) = (p(1, X_2, ..., X_n) - p(0, X_2, ..., X_n)) \cdot r + p(0, X_2, ..., X_n)
    /// ```
    pub fn compress_into_packed<EF>(&self, zi: EF) -> EvaluationsList<EF::ExtensionPacking>
    where
        A: Field,
        EF: ExtensionField<A>,
    {
        let zi = EF::ExtensionPacking::from(zi);
        let poly = A::Packing::pack_slice(self.as_slice());
        let mid = poly.len() / 2;
        let (p0, p1) = poly.split_at(mid);

        let mut out = EF::ExtensionPacking::zero_vec(mid);
        if self.num_evals() >= PARALLEL_THRESHOLD {
            out.par_iter_mut()
                .zip(p0.par_iter().zip(p1.par_iter()))
                .for_each(|(out, (&a0, &a1))| *out = zi * (a1 - a0) + a0);
        } else {
            out.iter_mut()
                .zip(p0.iter().zip(p1.iter()))
                .for_each(|(out, (&a0, &a1))| *out = zi * (a1 - a0) + a0);
        }
        EvaluationsList(out)
    }
}

impl<Packed: Copy + Send + Sync> EvaluationsList<Packed> {
    /// Given a point `P` (as a slice), compute the evaluation vector of the equality
    /// function `eq(P, X)` for all points `X` in the boolean hypercube, scaled by a value.
    ///
    /// ## Arguments
    /// * `point`: A multilinear point.
    /// * `value`: A scalar value to multiply all evaluations by.
    ///
    /// ## Returns
    /// An packed `EvaluationsList` containing `value * eq(point, X)` for all `X` in `{0,1}^n`.
    #[inline]
    pub fn new_packed_from_point<F, EF>(point: &[EF], scale: EF) -> Self
    where
        F: Field,
        EF: ExtensionField<F, ExtensionPacking = Packed>,
        Packed: PackedFieldExtension<F, EF>,
    {
        /// Computes eq(point, X) * scale for all X in {0,1}^n, writing results into `out`.
        ///
        /// # Safety invariant
        ///
        /// This function initializes **every** entry of `out`.
        /// Callers rely on this guarantee when passing uninitialized memory.
        fn eq_serial<F: Field, A: Algebra<F> + Copy>(out: &mut [A], point: &[F], scale: A) {
            assert_eq!(out.len(), 1 << point.len());
            out[0] = scale;
            for (i, &var) in point.iter().rev().enumerate() {
                let (lo, hi) = out.split_at_mut(1 << i);
                lo.iter_mut().zip(hi.iter_mut()).for_each(|(lo, hi)| {
                    *hi = *lo * var;
                    *lo -= *hi;
                });
            }
        }

        let n = point.len();
        assert_ne!(scale, EF::ZERO);
        let n_pack = log2_strict_usize(F::Packing::WIDTH);
        assert!(n >= n_pack);

        let (point_rest, point_init) = point.split_at(n - n_pack);

        // COMPUTE SUFFIX (Inside the SIMD lanes)
        //
        // We compute the equality polynomial for the last `n_pack` variables.
        // This forms a single `Packed` element which acts as the "seed" for the next stage.e
        let mut init: Vec<EF> = EF::zero_vec(1 << n_pack);
        eq_serial(&mut init, point_init, scale);

        // COMPUTE PREFIX (Vector Expansion)
        //
        // We expand the seed across the remaining variables using Packed arithmetic.
        let mut packed = unsafe { uninitialized_vec::<Packed>(1 << (n - n_pack)) };
        eq_serial(
            &mut packed,
            point_rest,
            // Initialize the first element with the seed computed above
            Packed::from_ext_slice(&init),
        );

        Self(packed)
    }

    /// Evaluates the multilinear polynomial at `point ∈ EF^n`.
    /// Polynomial evaluations are in packed form.
    ///
    /// Computes
    /// ```text
    ///     f(point) = \sum_{x ∈ {0,1}^n} eq(x, point) * f(x),
    /// ```
    /// where
    /// ```text
    ///     eq(x, point) = \prod_{i=1}^{n} (1 - p_i + 2 p_i x_i).
    /// ```
    pub fn evaluate_hypercube_packed<F, EF>(&self, point: &MultilinearPoint<EF>) -> EF
    where
        F: Field,
        EF: ExtensionField<F, ExtensionPacking = Packed>,
        Packed: PackedFieldExtension<F, EF>,
    {
        let n = point.num_variables();
        let n_pack = log2_strict_usize(F::Packing::WIDTH);
        assert_eq!(self.num_variables() + n_pack, n);
        assert!(n >= 2 * n_pack);

        let (right, left) = point.split_at(n / 2);
        let left = Self::new_packed_from_point(left.as_slice(), EF::ONE);
        let right = EvaluationsList::<EF>::new_from_point(right.as_slice(), EF::ONE);

        let sum = if self.num_evals() > PARALLEL_THRESHOLD {
            self.0
                .par_chunks(left.num_evals())
                .zip_eq(right.0.par_iter())
                .map(|(part, &c)| {
                    part.iter()
                        .zip_eq(left.iter())
                        .map(|(&a, &b)| b * a)
                        .sum::<Packed>()
                        * c
                })
                .sum()
        } else {
            self.0
                .chunks(left.num_evals())
                .zip_eq(right.0.iter())
                .map(|(part, &c)| {
                    part.iter()
                        .zip_eq(left.iter())
                        .map(|(&a, &b)| b * a)
                        .sum::<Packed>()
                        * c
                })
                .sum()
        };
        EF::ExtensionPacking::to_ext_iter([sum]).sum()
    }
}

impl<F> EvaluationsList<F>
where
    F: Field,
{
    /// Given a point `P` (as a slice), compute the evaluation vector of the equality
    /// function `eq(P, X)` for all points `X` in the boolean hypercube, scaled by a value.
    ///
    /// ## Arguments
    /// * `point`: A multilinear point.
    /// * `value`: A scalar value to multiply all evaluations by.
    ///
    /// ## Returns
    /// An `EvaluationsList` containing `value * eq(point, X)` for all `X` in `{0,1}^n`.
    #[inline]
    pub fn new_from_point(point: &[F], scale: F) -> Self {
        let n = point.len();
        if n == 0 {
            return Self(vec![scale]);
        }
        let len: usize = 1_usize
            .checked_shl(n as u32)
            .expect("Point length too large: 2^n overflows usize.");
        debug_assert!(
            len.is_power_of_two(),
            "Evaluation list length must be a power of two."
        );
        let mut evals = F::zero_vec(len);
        eval_eq_batch::<_, _, false>(RowMajorMatrixView::new_col(point), &mut evals, &[scale]);
        Self(evals)
    }

    /// Evaluates the multilinear polynomial at `point ∈ F^n`.
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
    pub fn evaluate_hypercube_base<EF: ExtensionField<F>>(
        &self,
        point: &MultilinearPoint<EF>,
    ) -> EF {
        if point.num_variables() < MLE_RECURSION_THRESHOLD {
            eval_multilinear_recursive(&self.0, point.as_slice())
        } else {
            eval_multilinear_base::<F, EF>(&self.0, point.as_slice())
        }
    }

    /// Evaluates the multilinear polynomial at `point ∈ F^n`.
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
    pub fn evaluate_hypercube_ext<BaseField: Field>(&self, point: &MultilinearPoint<F>) -> F
    where
        F: ExtensionField<BaseField>,
    {
        if point.num_variables() < MLE_RECURSION_THRESHOLD {
            eval_multilinear_recursive(&self.0, point.as_slice())
        } else {
            eval_multilinear_ext::<BaseField, F>(&self.0, point.as_slice())
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
    #[instrument(skip_all, level = "debug")]
    pub fn compress_ext<EF: ExtensionField<F>>(&self, r: EF) -> EvaluationsList<EF> {
        assert_ne!(self.num_variables(), 0);
        let num_evals = self.num_evals();
        let mid = num_evals / 2;
        // Evaluations at `a_i` and `a_{i + n/2}` slots are folded with `r`
        let (p0, p1) = self.0.split_at(mid);
        // Create new EvaluationsList in the extension field
        EvaluationsList(if num_evals >= PARALLEL_THRESHOLD {
            p0.par_iter()
                .zip(p1.par_iter())
                .map(|(&a0, &a1)| r * (a1 - a0) + a0)
                .collect()
        } else {
            p0.iter()
                .zip(p1.iter())
                .map(|(&a0, &a1)| r * (a1 - a0) + a0)
                .collect()
        })
    }

    /// Folds a multilinear polynomial stored in evaluation form along the last `k` variables.
    ///
    /// Given evaluations `f: {0,1}^n → F`, this method returns a new evaluation list `g` such that:
    ///
    /// ```text
    ///     g(x_0, ..., x_{n-k-1}) = f(x_0, ..., x_{n-k-1}, r_0, ..., r_{k-1})
    /// ```
    /// Folds a multilinear polynomial stored in evaluation form along the last `k` variables.
    ///
    /// Given evaluations `f: {0,1}^n → F`, this method returns a new evaluation list `g` such that:
    ///
    /// ```text
    ///     g(x_0, ..., x_{n-k-1}) = f(x_0, ..., x_{n-k-1}, r_0, ..., r_{k-1})
    /// ```
    ///
    /// # Arguments
    /// - `point`: The extension-field values to substitute for the last `k` variables.
    ///
    /// # Returns
    /// - A new `EvaluationsList<EF::ExtensionPacking>` representing the folded function over the remaining `n - k` variables.
    pub fn compress_multi_into_packed<EF: ExtensionField<F>>(
        &self,
        point: &[EF],
    ) -> EvaluationsList<EF::ExtensionPacking> {
        assert!(point.len() <= self.num_variables());
        let point = MultilinearPoint::new(point.to_vec());
        let eq = EvaluationsList::new_from_point(point.as_slice(), EF::ONE);

        let mut out = EF::ExtensionPacking::zero_vec(
            1 << (self.num_variables()
                - point.num_variables()
                - log2_strict_usize(F::Packing::WIDTH)),
        );

        self.0
            .chunks(self.num_evals() / eq.num_evals())
            .zip_eq(eq.iter())
            .for_each(|(chunk, &r)| {
                let r = EF::ExtensionPacking::from(r);
                let chunk = F::Packing::pack_slice(chunk);
                out.par_iter_mut()
                    .zip_eq(chunk.par_iter())
                    .for_each(|(acc, &poly)| *acc += r * poly);
            });
        EvaluationsList(out)
    }
}

impl<A: Copy + Send + Sync + PrimeCharacteristicRing> EvaluationsList<A> {
    /// Computes the constant and quadratic coefficients of the sumcheck polynomial.
    ///
    /// Given evaluations `self[i]` and weights `weights[i]`, this computes the coefficients
    /// of the univariate polynomial:
    ///
    /// ```text
    /// h(X) = \sum_{b \in \{0,1\}^{n-1}} self(X, b) * weights(X, b)
    /// ```
    ///
    /// which is a quadratic polynomial in `X`.
    ///
    /// # Coefficient Formulas
    ///
    /// The polynomial `h(X) = c_0 + c_1 * X + c_2 * X^2` has coefficients:
    ///
    /// ```text
    /// c_0 = h(0) = \sum_b self(0, b) * weights(0, b)
    ///
    /// c_2 = \sum_b (self(1,b) - self(0,b)) * (weights(1,b) - weights(0,b))
    /// ```
    ///
    /// The linear coefficient `c_1` is not computed here; it's derived by the verifier
    /// from the sum constraint `h(0) + h(1) = claimed_sum`.
    ///
    /// # Memory Layout
    ///
    /// The arrays are organized such that:
    /// - First half (`lo`): evaluations where `X = 0`
    /// - Second half (`hi`): evaluations where `X = 1`
    ///
    /// # Arguments
    ///
    /// * `weights` - Weight polynomial evaluations (same length as `self`).
    ///
    /// # Returns
    ///
    /// A tuple `(c_0, c_2)` of the constant and quadratic coefficients.
    ///
    /// # Panics
    ///
    /// Panics if `self.len() != weights.len()` or `self.len() < 2`.
    #[instrument(skip_all, level = "debug")]
    pub fn sumcheck_coefficients<B>(&self, weights: &EvaluationsList<B>) -> (B, B)
    where
        B: Copy + Send + Sync + Algebra<A>,
    {
        let evals = self.as_slice();
        let weights = weights.as_slice();

        // Validate inputs: need at least 2 elements (1 variable).
        assert!(log2_strict_usize(evals.len()) >= 1);
        assert_eq!(evals.len(), weights.len());

        // Split arrays into lo (X=0) and hi (X=1) halves.
        let mid = evals.len() / 2;
        let (evals_lo, evals_hi) = evals.split_at(mid);
        let (weights_lo, weights_hi) = weights.split_at(mid);

        // Parallel computation of c_0 and c_2.
        evals_lo
            .par_iter()
            .zip(evals_hi.par_iter())
            .zip(weights_lo.par_iter().zip(weights_hi.par_iter()))
            .map(|((&e_lo, &e_hi), (&w_lo, &w_hi))| {
                // c_0 term: product at X=0.
                let c0_term = w_lo * e_lo;
                // c_2 term: cross-product of differences.
                let c2_term = (w_hi.double() - w_lo) * (e_hi.double() - e_lo);
                (c0_term, c2_term)
            })
            .par_fold_reduce(
                || (B::ZERO, B::ZERO),
                |(a0, a2), (b0, b2)| (a0 + b0, a2 + b2),
                |(a0, a2), (b0, b2)| (a0 + b0, a2 + b2),
            )
    }
}

impl<'a, F> IntoIterator for &'a EvaluationsList<F> {
    type Item = &'a F;
    type IntoIter = core::slice::Iter<'a, F>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.0.iter()
    }
}

impl<F> IntoIterator for EvaluationsList<F> {
    type Item = F;
    type IntoIter = alloc::vec::IntoIter<F>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.0.into_iter()
    }
}

/// Evaluates a multilinear polynomial `evals` at `point` using a recursive strategy.
/// For small numbers of variables (<=4) it switches to the unrolled strategy.
fn eval_multilinear_recursive<F, EF>(evals: &[F], point: &[EF]) -> EF
where
    F: Field,
    EF: ExtensionField<F>,
{
    // Ensure that the number of evaluations matches the number of variables in the point.
    //
    // This is a critical invariant: `evals.len()` must be exactly `2^point.len()`.
    debug_assert_eq!(evals.len(), 1 << point.len());

    // Select the optimal evaluation strategy based on the number of variables.
    match point {
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
        [x, sub_point @ ..] => {
            // Split the evaluations into two halves, corresponding to the first variable being 0 or 1.
            let (f0, f1) = evals.split_at(evals.len() / 2);

            // Recursively evaluate on the two smaller hypercubes.
            let (f0_eval, f1_eval) = {
                // Only spawn parallel tasks if the subproblem is large enough to overcome
                // the overhead of threading.
                let work_size: usize = (1 << 15) / core::mem::size_of::<F>();
                if evals.len() > work_size {
                    join(
                        || eval_multilinear_recursive(f0, sub_point),
                        || eval_multilinear_recursive(f1, sub_point),
                    )
                } else {
                    // For smaller subproblems, execute sequentially.
                    (
                        eval_multilinear_recursive(f0, sub_point),
                        eval_multilinear_recursive(f1, sub_point),
                    )
                }
            };
            // Perform the final linear interpolation for the first variable `x`.
            f0_eval + (f1_eval - f0_eval) * *x
        }
    }
}

/// Evaluates a multilinear polynomial `evals` at `point` where `evals` are in the base field `F` and `point` is in the extension field `EF`.
fn eval_multilinear_base<F, EF>(evals: &[F], point: &[EF]) -> EF
where
    F: Field,
    EF: ExtensionField<F>,
{
    const PARALLEL_THRESHOLD: usize = 1 << 14;

    let num_vars = point.len();
    if num_vars < 2 * log2_strict_usize(F::Packing::WIDTH) {
        return eval_multilinear_recursive(evals, point);
    }

    let mid = num_vars / 2;
    let (right, left) = point.split_at(mid);
    let left = EvaluationsList::new_packed_from_point(left, EF::ONE);
    let right = EvaluationsList::new_from_point(right, EF::ONE);

    let evals = F::Packing::pack_slice(evals);
    let sum = if evals.len() > PARALLEL_THRESHOLD {
        evals
            .par_chunks(left.num_evals())
            .zip_eq(right.0.par_iter())
            .map(|(part, &c)| {
                part.iter()
                    .zip_eq(left.iter())
                    .map(|(&a, &b)| b * a)
                    .sum::<EF::ExtensionPacking>()
                    * c
            })
            .sum::<EF::ExtensionPacking>()
    } else {
        evals
            .chunks(left.num_evals())
            .zip_eq(right.0.iter())
            .map(|(part, &c)| {
                part.iter()
                    .zip_eq(left.iter())
                    .map(|(&a, &b)| b * a)
                    .sum::<EF::ExtensionPacking>()
                    * c
            })
            .sum::<EF::ExtensionPacking>()
    };
    EF::ExtensionPacking::to_ext_iter([sum]).sum()
}

/// Evaluates a multilinear polynomial `evals` at `point` where `evals` and `point` are in the extension field `EF`.
fn eval_multilinear_ext<F, EF>(evals: &[EF], point: &[EF]) -> EF
where
    F: Field,
    EF: ExtensionField<F>,
{
    const PARALLEL_THRESHOLD: usize = 1 << 14;

    let num_vars = point.len();
    if num_vars < 2 * log2_strict_usize(F::Packing::WIDTH) {
        return eval_multilinear_recursive(evals, point);
    }

    let mid = num_vars / 2;
    let (right, left) = point.split_at(mid);
    let left = EvaluationsList::new_packed_from_point(left, EF::ONE);
    let right = EvaluationsList::new_from_point(right, EF::ONE);

    let sum = if evals.len() > PARALLEL_THRESHOLD {
        evals
            .chunks(F::Packing::WIDTH * left.num_evals())
            .zip_eq(right.0.iter())
            .map(|(part, &c)| {
                part.chunks(F::Packing::WIDTH)
                    .zip_eq(left.iter())
                    .map(|(chunk, &b)| EF::ExtensionPacking::from_ext_slice(chunk) * b)
                    .sum::<EF::ExtensionPacking>()
                    * c
            })
            .sum::<EF::ExtensionPacking>()
    } else {
        evals
            .par_chunks(F::Packing::WIDTH * left.num_evals())
            .zip_eq(right.0.par_iter())
            .map(|(part, &c)| {
                part.chunks(F::Packing::WIDTH)
                    .zip_eq(left.iter())
                    .map(|(chunk, &b)| EF::ExtensionPacking::from_ext_slice(chunk) * b)
                    .sum::<EF::ExtensionPacking>()
                    * c
            })
            .sum::<EF::ExtensionPacking>()
    };
    EF::ExtensionPacking::to_ext_iter([sum]).sum()
}

#[cfg(test)]
mod tests {

    use alloc::vec;

    use p3_baby_bear::BabyBear;
    use p3_field::extension::BinomialExtensionField;
    use p3_field::{PrimeCharacteristicRing, PrimeField64, dot_product};
    use proptest::prelude::*;
    use rand::rngs::SmallRng;
    use rand::{RngExt, SeedableRng};

    use super::*;

    impl<F: Copy + Clone + Send + Sync> EvaluationsList<F>
    where
        F: Field,
    {
        /// Evaluates the polynomial as a constant.
        ///
        /// This is only valid for constant polynomials (i.e., when `num_variables` is 0).
        ///
        /// Returns None in other cases.
        #[must_use]
        #[inline]
        pub fn as_constant(&self) -> Option<F> {
            (self.num_evals() == 1).then_some(self.0[0])
        }

        /// Folds the polynomial by substituting the last `k` variables with the given point.
        pub(crate) fn compress_multi<EF: ExtensionField<F>>(
            &self,
            point: &[EF],
        ) -> EvaluationsList<EF> {
            assert!(point.len() <= self.num_variables());
            let eq = EvaluationsList::new_from_point(point, EF::ONE);
            let mut out = EF::zero_vec(1 << (self.num_variables() - point.len()));
            self.0
                .chunks(self.num_evals() / eq.num_evals())
                .zip_eq(eq.iter())
                .for_each(|(chunk, &r)| {
                    out.par_iter_mut()
                        .zip_eq(chunk.par_iter())
                        .for_each(|(acc, &poly)| *acc += r * poly);
                });
            EvaluationsList(out)
        }
    }

    type F = BabyBear;
    type EF4 = BinomialExtensionField<F, 4>;

    /// Naive method to evaluate a multilinear polynomial for testing.
    fn eval_multilinear<F: Field, EF: ExtensionField<F>>(evals: &[F], point: &[EF]) -> EF {
        let eq = EvaluationsList::new_from_point(point, EF::ONE);
        dot_product(eq.iter().copied(), evals.iter().copied())
    }

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

        let _ = evaluations_list.as_slice()[4]; // Index out of range, should panic
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
            evals.evaluate_hypercube_base(&MultilinearPoint::new(vec![F::ZERO, F::ZERO])),
            e1
        );
        assert_eq!(
            evals.evaluate_hypercube_base(&MultilinearPoint::new(vec![F::ZERO, F::ONE])),
            e2
        );
        assert_eq!(
            evals.evaluate_hypercube_base(&MultilinearPoint::new(vec![F::ONE, F::ZERO])),
            e3
        );
        assert_eq!(
            evals.evaluate_hypercube_base(&MultilinearPoint::new(vec![F::ONE, F::ONE])),
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

        let result = evals.evaluate_hypercube_base(&point);

        // Expected result using `eval_multilinear`
        let expected = eval_multilinear(evals.as_slice(), point.as_slice());

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

        assert_eq!(eval_multilinear(&evals, &[x]), expected);
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

        assert_eq!(eval_multilinear(&evals, &[x, y]), expected);
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

        assert_eq!(eval_multilinear(&evals, &[x, y, z]), expected);
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
        assert_eq!(eval_multilinear(&evals, &[x, y, z, w]), expected);
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
            let eval_f = poly_f.evaluate_hypercube_base(&point_ef);
            let eval_ef = poly_ef.evaluate_hypercube_ext::<F>(&point_ef);

            prop_assert_eq!(eval_f, eval_ef);
        }
    }

    #[test]
    fn test_multilinear_eval_two_vars() {
        // Define a simple 2-variable multilinear polynomial:
        //
        // Variables: x_1, x_2
        // Evaluations ordered in lexicographic order of input points: (x_1, x_2)
        //
        // - evals[0] → f(0, 0)
        // - evals[1] → f(0, 1)
        // - evals[2] → f(1, 0)
        // - evals[3] → f(1, 1)
        //
        // Thus, the polynomial is represented by its values
        // on the Boolean hypercube {0,1}².
        //
        // where:
        let e0 = F::from_u64(5); // f(0, 0)
        let e1 = F::from_u64(6); // increment when x_2 = 1
        let e2 = F::from_u64(7); // increment when x_1 = 1
        let e3 = F::from_u64(8); // increment when x_1 = x_2 = 1
        //
        // So concretely:
        //
        //   f(0, 0) = 5
        //   f(0, 1) = 5 + 6 = 11
        //   f(1, 0) = 5 + 7 = 12
        //   f(1, 1) = 5 + 6 + 7 + 8 = 26
        let evals = EvaluationsList::new(vec![e0, e0 + e1, e0 + e2, e0 + e1 + e2 + e3]);

        // Choose evaluation point:
        //
        // Let's pick (x_1, x_2) = (2, 1)
        let x1 = F::from_u64(2);
        let x2 = F::from_u64(1);
        let coords = MultilinearPoint::new(vec![x1, x2]);

        // Manually compute the expected value step-by-step:
        //
        // Reminder:
        //   f(x_1, x_2) = 5 + 6·x_2 + 7·x_1 + 8·x_1·x_2
        //
        // Substituting (x_1, x_2):
        let expected = e0 + e1 * x2 + e2 * x1 + e3 * x1 * x2;

        // Now evaluate using the function under test
        let result = evals.evaluate_hypercube_base(&coords);

        // Check that it matches the manual computation
        assert_eq!(result, expected);
    }

    #[test]
    fn test_evaluate_3_variables() {
        // Define a multilinear polynomial in 3 variables: x_0, x_1, x_2
        //
        // Coefficients ordered lex in index order:
        //
        // - coeffs[0] → constant term
        // - coeffs[1] → x_2
        // - coeffs[2] → x_1
        // - coeffs[3] → x_1·x_2
        // - coeffs[4] → x_0
        // - coeffs[5] → x_0·x_2
        // - coeffs[6] → x_0·x_1
        // - coeffs[7] → x_0·x_1·x_2
        //
        // Thus:
        //    f(x_0,x_1,x_2) = c0 + c1·x_2 + c2·x_1 + c3·x_1·x_2
        //                + c4·x_0 + c5·x_0·x_2 + c6·x_0·x_1 + c7·x_0·x_1·x_2
        let e0 = F::from_u64(1);
        let e1 = F::from_u64(2);
        let e2 = F::from_u64(3);
        let e3 = F::from_u64(4);
        let e4 = F::from_u64(5);
        let e5 = F::from_u64(6);
        let e6 = F::from_u64(7);
        let e7 = F::from_u64(8);

        let evals = EvaluationsList::new(vec![
            e0,
            e0 + e1,
            e0 + e2,
            e0 + e1 + e2 + e3,
            e0 + e4,
            e0 + e1 + e4 + e5,
            e0 + e2 + e4 + e6,
            e0 + e1 + e2 + e3 + e4 + e5 + e6 + e7,
        ]);

        // Pick point: (x_0,x_1,x_2) = (2, 3, 4)
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
        let expected = e0
            + e1 * x2
            + e2 * x1
            + e3 * x1 * x2
            + e4 * x0
            + e5 * x0 * x2
            + e6 * x0 * x1
            + e7 * x0 * x1 * x2;

        let result = evals.evaluate_hypercube_base(&point);
        assert_eq!(result, expected);
    }

    #[test]
    fn test_eval_extension_3_variables() {
        let e0 = F::from_u64(1);
        let e1 = F::from_u64(2);
        let e2 = F::from_u64(3);
        let e3 = F::from_u64(4);
        let e4 = F::from_u64(5);
        let e5 = F::from_u64(6);
        let e6 = F::from_u64(7);
        let e7 = F::from_u64(8);

        let evals = EvaluationsList::new(vec![
            e0,
            e0 + e1,
            e0 + e2,
            e0 + e1 + e2 + e3,
            e0 + e4,
            e0 + e1 + e4 + e5,
            e0 + e2 + e4 + e6,
            e0 + e1 + e2 + e3 + e4 + e5 + e6 + e7,
        ]);

        let x0 = EF4::from_u64(2);
        let x1 = EF4::from_u64(3);
        let x2 = EF4::from_u64(4);

        let point = MultilinearPoint::new(vec![x0, x1, x2]);

        let expected = EF4::from(e0)
            + EF4::from(e1) * x2
            + EF4::from(e2) * x1
            + EF4::from(e3) * x1 * x2
            + EF4::from(e4) * x0
            + EF4::from(e5) * x0 * x2
            + EF4::from(e6) * x0 * x1
            + EF4::from(e7) * x0 * x1 * x2;

        let result = evals.evaluate_hypercube_base(&point);
        assert_eq!(result, expected);
    }

    #[test]
    fn test_folding_and_evaluation() {
        let num_variables = 10;
        let evals = (0..(1 << num_variables)).map(F::from_u64).collect();
        let evals_list = EvaluationsList::new(evals);
        let randomness: Vec<_> = (0..num_variables)
            .map(|i| F::from_u64(35 * i as u64))
            .collect();

        for k in 1..num_variables {
            let fold_part = randomness[0..k].to_vec();
            let eval_part = MultilinearPoint::new(randomness[k..randomness.len()].to_vec());
            let fold_random = MultilinearPoint::new(fold_part.clone());
            let eval_point1 =
                MultilinearPoint::new([fold_part.clone(), eval_part.0.clone()].concat());
            let folded_evals = evals_list.compress_multi(fold_random.as_slice());
            assert_eq!(folded_evals.num_variables(), num_variables - k);
            let folded_coeffs = evals_list.compress_multi(fold_random.as_slice());
            assert_eq!(folded_coeffs.num_variables(), num_variables - k);
            assert_eq!(
                folded_evals.evaluate_hypercube_base(&eval_part),
                evals_list.evaluate_hypercube_base(&eval_point1)
            );
        }
    }

    #[test]
    fn test_fold_with_extension_one_var() {
        let evals = vec![
            F::from_u64(1),
            F::from_u64(2),
            F::from_u64(3),
            F::from_u64(4),
        ];
        let poly = EvaluationsList::new(evals);
        let evals_list: EvaluationsList<F> = poly;
        let r1 = EF4::from_u64(5);
        let folded = evals_list.compress_multi(&[r1]);

        for x0_f in 0..10 {
            let x0 = EF4::from_u64(x0_f);
            let full_point = MultilinearPoint::new(vec![r1, x0]);
            let folded_point = MultilinearPoint::new(vec![x0]);
            let expected = evals_list.evaluate_hypercube_base(&full_point);
            let actual = folded.evaluate_hypercube_base(&folded_point);
            assert_eq!(expected, actual);
        }
    }

    proptest! {
        #[test]
        fn prop_eval_eq_matches_naive_for_eval_list(
            n in 1usize..5,
            evals_raw in prop::collection::vec(0u64..F::ORDER_U64, 5),
        ) {
            let evals: Vec<F> = evals_raw[..n].iter().map(|&x| F::from_u64(x)).collect();
            let mut out = vec![F::ZERO; 1 << n];
            eval_eq_batch::<F, F, false>(
                RowMajorMatrixView::new_col(&evals),
                &mut out,
                &[F::ONE],
            );
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
        const NUM_VARS: usize = 20;
        let mut rng = SmallRng::seed_from_u64(42);
        let num_evals = 1 << NUM_VARS;
        let evals_vec: Vec<F> = (0..num_evals).map(|_| rng.random()).collect();
        let evals_list = EvaluationsList::new(evals_vec);
        let point_vec: Vec<EF4> = (0..NUM_VARS).map(|_| rng.random()).collect();
        let point = MultilinearPoint::new(point_vec);

        let mut expected_sum = EF4::ZERO;
        for i in 0..num_evals {
            let mut eq_term = EF4::ONE;
            for j in 0..NUM_VARS {
                let v_j = (i >> (NUM_VARS - 1 - j)) & 1;
                let p_j = point.as_slice()[j];
                if v_j == 1 {
                    eq_term *= p_j;
                } else {
                    eq_term *= EF4::ONE - p_j;
                }
            }
            let f_v = evals_list.as_slice()[i];
            expected_sum += eq_term * f_v;
        }

        let actual_result = evals_list.evaluate_hypercube_base(&point);
        assert_eq!(actual_result, expected_sum);
    }

    #[test]
    fn test_new_from_point_zero_vars() {
        let point = MultilinearPoint::<F>::new(vec![]);
        let value = F::from_u64(42);
        let evals_list = EvaluationsList::new_from_point(point.as_slice(), value);
        assert_eq!(evals_list.num_variables(), 0);
        assert_eq!(evals_list.as_slice(), &[value]);
    }

    #[test]
    fn test_new_from_point_one_var() {
        let p0 = F::from_u64(7);
        let point = MultilinearPoint::new(vec![p0]);
        let value = F::from_u64(3);
        let evals_list = EvaluationsList::new_from_point(point.as_slice(), value);
        let expected = vec![value * (F::ONE - p0), value * p0];
        assert_eq!(evals_list.num_variables(), 1);
        assert_eq!(evals_list.as_slice(), &expected);
    }

    #[test]
    #[allow(clippy::identity_op)]
    fn test_new_from_point_three_vars() {
        let p = [F::from_u64(2), F::from_u64(3), F::from_u64(5)];
        let value = F::from_u64(10);
        let evals_list = EvaluationsList::new_from_point(&p, value);
        let mut expected = Vec::with_capacity(8);
        for i in 0..8 {
            let b0 = (i >> 2) & 1;
            let b1 = (i >> 1) & 1;
            let b2 = (i >> 0) & 1;
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
        let constant_value = F::from_u64(42);
        let evals = EvaluationsList::new(vec![constant_value]);
        assert_eq!(evals.num_variables(), 0);
        assert_eq!(evals.as_constant(), Some(constant_value));
    }

    #[test]
    fn test_as_constant_for_non_constant_poly() {
        let evals = EvaluationsList::new(vec![F::ONE, F::ZERO, F::ONE, F::ZERO]);
        assert_ne!(evals.num_variables(), 0);
        assert_eq!(evals.as_constant(), None);
    }

    #[test]
    #[should_panic]
    fn test_compress_panics_on_constant() {
        let mut evals_list = EvaluationsList::new(vec![F::from_u64(42)]);
        evals_list.compress(F::ONE);
    }

    #[test]
    fn test_compress_basic() {
        let p_000 = F::from_u64(1);
        let p_001 = F::from_u64(2);
        let p_010 = F::from_u64(3);
        let p_011 = F::from_u64(4);
        let p_100 = F::from_u64(5);
        let p_101 = F::from_u64(6);
        let p_110 = F::from_u64(7);
        let p_111 = F::from_u64(8);
        let initial_evals = vec![p_000, p_001, p_010, p_011, p_100, p_101, p_110, p_111];
        let mut evals_list = EvaluationsList::new(initial_evals);
        let r = F::from_u64(10);
        let expected = vec![
            r * (p_100 - p_000) + p_000,
            r * (p_101 - p_001) + p_001,
            r * (p_110 - p_010) + p_010,
            r * (p_111 - p_011) + p_011,
        ];
        assert_eq!(evals_list.num_variables(), 3);
        assert_eq!(evals_list.num_evals(), 8);
        evals_list.compress(r);
        assert_eq!(evals_list.num_variables(), 2);
        assert_eq!(evals_list.num_evals(), 4);
        assert_eq!(evals_list.as_slice(), &expected);
    }

    #[test]
    fn test_compress_parallel_path() {
        let num_evals = PARALLEL_THRESHOLD;
        let mid = num_evals / 2;
        let p_left_0 = F::from_u64(1);
        let p_right_0 = F::from_usize(mid + 1);
        let p_left_1 = F::from_u64(2);
        let p_right_1 = F::from_usize(mid + 2);
        let initial_evals: Vec<F> = (0..num_evals).map(|i| F::from_usize(i + 1)).collect();
        let mut evals_list = EvaluationsList::new(initial_evals);
        let r = F::from_u64(3);
        let num_variables_before = evals_list.num_variables();
        evals_list.compress(r);
        assert_eq!(evals_list.num_variables(), num_variables_before - 1);
        assert_eq!(evals_list.num_evals(), mid);
        assert_eq!(
            evals_list.as_slice()[0],
            r * (p_right_0 - p_left_0) + p_left_0
        );
        assert_eq!(
            evals_list.as_slice()[1],
            r * (p_right_1 - p_left_1) + p_left_1
        );
    }

    #[test]
    fn test_compress_multiple_rounds() {
        let initial_evals: Vec<F> = (1..=16).map(F::from_u64).collect();
        let mut evals_list = EvaluationsList::new(initial_evals);
        let challenges = vec![F::from_u64(3), F::from_u64(7), F::from_u64(11)];
        for &r in &challenges {
            evals_list.compress(r);
        }
        assert_eq!(evals_list.num_variables(), 1);
        assert_eq!(evals_list.num_evals(), 2);
    }

    #[test]
    fn test_compress_single_variable() {
        let p_0 = F::from_u64(5);
        let p_1 = F::from_u64(9);
        let mut evals_list = EvaluationsList::new(vec![p_0, p_1]);
        let r = F::from_u64(7);
        evals_list.compress(r);
        assert_eq!(evals_list.num_variables(), 0);
        assert_eq!(evals_list.num_evals(), 1);
        let expected = r * (p_1 - p_0) + p_0;
        assert_eq!(evals_list.as_slice(), vec![expected]);
    }

    #[test]
    fn test_compress_with_zero_challenge() {
        let p_000 = F::from_u64(1);
        let p_001 = F::from_u64(2);
        let p_010 = F::from_u64(3);
        let p_011 = F::from_u64(4);
        let p_100 = F::from_u64(5);
        let p_101 = F::from_u64(6);
        let p_110 = F::from_u64(7);
        let p_111 = F::from_u64(8);
        let mut evals_list =
            EvaluationsList::new(vec![p_000, p_001, p_010, p_011, p_100, p_101, p_110, p_111]);
        let r = F::ZERO;
        evals_list.compress(r);
        let expected = vec![p_000, p_001, p_010, p_011];
        assert_eq!(evals_list.as_slice(), &expected);
    }

    #[test]
    fn test_compress_with_one_challenge() {
        let p_000 = F::from_u64(1);
        let p_001 = F::from_u64(2);
        let p_010 = F::from_u64(3);
        let p_011 = F::from_u64(4);
        let p_100 = F::from_u64(5);
        let p_101 = F::from_u64(6);
        let p_110 = F::from_u64(7);
        let p_111 = F::from_u64(8);
        let mut evals_list =
            EvaluationsList::new(vec![p_000, p_001, p_010, p_011, p_100, p_101, p_110, p_111]);
        let r = F::ONE;
        evals_list.compress(r);
        let expected = vec![p_100, p_101, p_110, p_111];
        assert_eq!(evals_list.as_slice(), &expected);
    }

    #[test]
    fn test_compress_ext() {
        let initial_evals: Vec<F> = [1u64, 3, 5, 7, 2, 4, 6, 8]
            .into_iter()
            .map(F::from_u64)
            .collect();
        let evals_list = EvaluationsList::new(initial_evals);
        let r_ext = EF4::from_u64(10);
        let expected: Vec<EF4> = vec![
            r_ext * (EF4::from_u64(2) - EF4::from_u64(1)) + EF4::from_u64(1),
            r_ext * (EF4::from_u64(4) - EF4::from_u64(3)) + EF4::from_u64(3),
            r_ext * (EF4::from_u64(6) - EF4::from_u64(5)) + EF4::from_u64(5),
            r_ext * (EF4::from_u64(8) - EF4::from_u64(7)) + EF4::from_u64(7),
        ];
        let compressed_ext_list = evals_list.compress_ext(r_ext);
        assert_eq!(compressed_ext_list.num_variables(), 2);
        assert_eq!(compressed_ext_list.num_evals(), 4);
        assert_eq!(compressed_ext_list.as_slice(), &expected);
    }

    #[test]
    #[should_panic]
    fn test_compress_ext_panics_on_constant() {
        let evals_list = EvaluationsList::new(vec![F::from_u64(42)]);
        let _ = evals_list.compress_ext(EF4::ONE);
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

            let mut list_a = EvaluationsList::new(evals.clone());
            list_a.compress(r_base);
            let result_a_lifted: Vec<EF4> = list_a.as_slice().iter().map(|&x| EF4::from(x)).collect();

            let list_b = EvaluationsList::new(evals);
            let r_ext = EF4::from(r_base);
            let result_b_ext = list_b.compress_ext(r_ext);

            prop_assert_eq!(result_a_lifted, result_b_ext.as_slice());
        }

        #[test]
        fn prop_compress_dimensions(
            n in 1usize..=10,
            seed in any::<u64>(),
        ) {
            let mut rng = SmallRng::seed_from_u64(seed);
            let num_evals = 1 << n;
            let evals: Vec<F> = (0..num_evals).map(|_| rng.random()).collect();
            let r: F = rng.random();

            let mut list = EvaluationsList::new(evals);
            list.compress(r);

            prop_assert_eq!(list.num_variables(), n - 1);
            prop_assert_eq!(list.num_evals(), num_evals / 2);
        }

        #[test]
        fn prop_compress_boundary_challenges(
            n in 2usize..=8,
            seed in any::<u64>(),
        ) {
            let mut rng = SmallRng::seed_from_u64(seed);
            let num_evals = 1 << n;
            let evals: Vec<F> = (0..num_evals).map(|_| rng.random()).collect();

            let mut list_zero = EvaluationsList::new(evals.clone());
            list_zero.compress(F::ZERO);
            prop_assert_eq!(list_zero.num_evals(), num_evals / 2);

            let mut list_one = EvaluationsList::new(evals);
            list_one.compress(F::ONE);
            prop_assert_eq!(list_one.num_evals(), num_evals / 2);

            if list_zero.as_slice() != list_one.as_slice() {
                prop_assert_ne!(list_zero.as_slice(), list_one.as_slice());
            }
        }

        #[test]
        fn prop_compress_multiple_rounds(
            n in 2usize..=8,
            num_rounds in 1usize..=5,
            seed in any::<u64>(),
        ) {
            let mut rng = SmallRng::seed_from_u64(seed);
            let num_evals = 1 << n;
            let evals: Vec<F> = (0..num_evals).map(|_| rng.random()).collect();

            let actual_rounds = num_rounds.min(n);
            let challenges: Vec<F> = (0..actual_rounds).map(|_| rng.random()).collect();

            let mut list = EvaluationsList::new(evals);
            for &r in &challenges {
                list.compress(r);
            }

            prop_assert_eq!(list.num_variables(), n - actual_rounds);
            prop_assert_eq!(list.num_evals(), 1 << (n - actual_rounds));
        }
    }

    #[test]
    fn test_fold_batch_no_challenges() {
        let poly = EvaluationsList::new(vec![
            F::from_u64(1),
            F::from_u64(2),
            F::from_u64(3),
            F::from_u64(4),
            F::from_u64(5),
            F::from_u64(6),
            F::from_u64(7),
            F::from_u64(8),
        ]);
        let challenges: &[EF4] = &[];
        let result = poly.compress_multi(challenges);
        assert_eq!(result.0.len(), 8, "Result should have 8 evaluations");
        let expected_poly = EvaluationsList::new(vec![
            EF4::from_u64(1),
            EF4::from_u64(2),
            EF4::from_u64(3),
            EF4::from_u64(4),
            EF4::from_u64(5),
            EF4::from_u64(6),
            EF4::from_u64(7),
            EF4::from_u64(8),
        ]);
        assert_eq!(
            result, expected_poly,
            "Result should be the original polynomial"
        );
    }

    #[test]
    fn test_fold_batch_single_variable() {
        let poly = EvaluationsList::new((1..=8).map(F::from_u64).collect());
        let r2 = EF4::from_u64(3);
        let challenges = vec![r2];
        let result = poly.compress_multi(&challenges);
        assert_eq!(
            result.0.len(),
            4,
            "Folded polynomial should have 4 evaluations"
        );
        let eq_r2_0 = EF4::ONE - r2;
        let eq_r2_1 = r2;
        let p_000 = F::from_u64(1);
        let p_001 = F::from_u64(2);
        let p_010 = F::from_u64(3);
        let p_011 = F::from_u64(4);
        let p_100 = F::from_u64(5);
        let p_101 = F::from_u64(6);
        let p_110 = F::from_u64(7);
        let p_111 = F::from_u64(8);
        let q_00 = eq_r2_0 * p_000 + eq_r2_1 * p_100;
        let q_01 = eq_r2_0 * p_001 + eq_r2_1 * p_101;
        let q_10 = eq_r2_0 * p_010 + eq_r2_1 * p_110;
        let q_11 = eq_r2_0 * p_011 + eq_r2_1 * p_111;
        assert_eq!(result.0[0], q_00, "q(0,0) mismatch");
        assert_eq!(result.0[1], q_01, "q(0,1) mismatch");
        assert_eq!(result.0[2], q_10, "q(1,0) mismatch");
        assert_eq!(result.0[3], q_11, "q(1,1) mismatch");
    }

    #[test]
    fn test_fold_batch_two_variables() {
        let poly = EvaluationsList::new(vec![
            F::from_u64(1),
            F::from_u64(2),
            F::from_u64(4),
            F::from_u64(8),
            F::from_u64(16),
            F::from_u64(32),
            F::from_u64(64),
            F::from_u64(128),
        ]);
        let r2 = EF4::from_u64(2);
        let r1 = EF4::from_u64(3);
        let challenges = vec![r2, r1];
        let result = poly.compress_multi(&challenges);
        assert_eq!(
            result.0.len(),
            2,
            "Folded polynomial should have 2 evaluations"
        );
        let eq_r2_0 = EF4::ONE - r2;
        let eq_r2_1 = r2;
        let eq_r1_0 = EF4::ONE - r1;
        let eq_r1_1 = r1;
        let eq_00 = eq_r2_0 * eq_r1_0;
        let eq_01 = eq_r2_0 * eq_r1_1;
        let eq_10 = eq_r2_1 * eq_r1_0;
        let eq_11 = eq_r2_1 * eq_r1_1;
        let p_000 = EF4::from_u64(1);
        let p_001 = EF4::from_u64(2);
        let p_010 = EF4::from_u64(4);
        let p_011 = EF4::from_u64(8);
        let p_100 = EF4::from_u64(16);
        let p_101 = EF4::from_u64(32);
        let p_110 = EF4::from_u64(64);
        let p_111 = EF4::from_u64(128);
        let q_0 = eq_00 * p_000 + eq_01 * p_010 + eq_10 * p_100 + eq_11 * p_110;
        let q_1 = eq_00 * p_001 + eq_01 * p_011 + eq_10 * p_101 + eq_11 * p_111;
        assert_eq!(result.0[0], q_0, "q(0) mismatch");
        assert_eq!(result.0[1], q_1, "q(1) mismatch");
    }

    #[test]
    fn test_fold_batch_all_variables() {
        let poly = EvaluationsList::new(vec![
            F::from_u64(1),
            F::from_u64(2),
            F::from_u64(3),
            F::from_u64(4),
        ]);
        let r1 = EF4::from_u64(5);
        let r0 = EF4::from_u64(7);
        let challenges = vec![r1, r0];
        let result = poly.compress_multi(&challenges);
        assert_eq!(
            result.0.len(),
            1,
            "Folding all variables should produce a single value"
        );
        let eq_r1_0 = EF4::ONE - r1;
        let eq_r1_1 = r1;
        let eq_r0_0 = EF4::ONE - r0;
        let eq_r0_1 = r0;
        let eq_00 = eq_r1_0 * eq_r0_0;
        let eq_01 = eq_r1_0 * eq_r0_1;
        let eq_10 = eq_r1_1 * eq_r0_0;
        let eq_11 = eq_r1_1 * eq_r0_1;
        let p_00 = EF4::from_u64(1);
        let p_01 = EF4::from_u64(2);
        let p_10 = EF4::from_u64(3);
        let p_11 = EF4::from_u64(4);
        let q = eq_00 * p_00 + eq_01 * p_01 + eq_10 * p_10 + eq_11 * p_11;
        assert_eq!(result.0[0], q, "Folded value mismatch");
    }

    #[test]
    #[should_panic(expected = "assertion failed: point.len() <= self.num_variables()")]
    fn test_fold_batch_too_many_challenges() {
        let poly = EvaluationsList::new(vec![
            F::from_u64(1),
            F::from_u64(2),
            F::from_u64(3),
            F::from_u64(4),
        ]);
        let challenges = vec![EF4::from_u64(2), EF4::from_u64(3), EF4::from_u64(5)];
        let _ = poly.compress_multi(&challenges);
    }

    #[test]
    fn test_base_eval_consistency() {
        let mut rng = SmallRng::seed_from_u64(1);
        for k in 0..=18 {
            let poly: Vec<F> = (0..1 << k).map(|_| rng.random()).collect();
            let point = MultilinearPoint::<EF4>::rand(&mut rng, k);
            let point = point.as_slice();
            let e0 = eval_multilinear_recursive(&poly, point);
            let e1 = eval_multilinear(&poly, point);
            assert_eq!(e0, e1);
            let e1 = eval_multilinear_base(&poly, point);
            assert_eq!(e0, e1);
        }
    }

    #[test]
    fn test_ext_eval_consistency() {
        let mut rng = SmallRng::seed_from_u64(1);
        for k in 0..=18 {
            let poly: Vec<EF4> = (0..1 << k).map(|_| rng.random()).collect();
            let point = MultilinearPoint::<EF4>::rand(&mut rng, k);
            let point = point.as_slice();
            let e0 = eval_multilinear_recursive(&poly, point);
            let e1 = eval_multilinear(&poly, point);
            assert_eq!(e0, e1);
            let e1 = eval_multilinear_base(&poly, point);
            assert_eq!(e0, e1);
            let e1 = eval_multilinear_ext::<F, _>(&poly, point);
            assert_eq!(e0, e1);
        }
    }

    #[test]
    fn test_ext_eval_packed_consistency() {
        let mut rng = SmallRng::seed_from_u64(1);
        let min_k = 2 * log2_strict_usize(<F as Field>::Packing::WIDTH);
        for k in min_k..=18 {
            let poly: Vec<EF4> = (0..1 << k).map(|_| rng.random()).collect();
            let point = MultilinearPoint::<EF4>::rand(&mut rng, k);
            let e0 = eval_multilinear_recursive(&poly, point.as_slice());
            let packed = poly
                .par_chunks(<F as Field>::Packing::WIDTH)
                .map(<EF4 as ExtensionField<F>>::ExtensionPacking::from_ext_slice)
                .collect();
            let e1 = EvaluationsList::new(packed).evaluate_hypercube_packed::<F, _>(&point);
            assert_eq!(e0, e1);
        }
    }

    #[test]
    fn test_sumcheck_coefficients_one_variable() {
        let e0 = EF4::from_u64(3);
        let e1 = EF4::from_u64(7);
        let w0 = EF4::from_u64(2);
        let w1 = EF4::from_u64(5);
        let evals = EvaluationsList::new(vec![e0, e1]);
        let weights = EvaluationsList::new(vec![w0, w1]);
        let (h0, h2) = evals.sumcheck_coefficients(&weights);
        let expected_h0 = e0 * w0;
        assert_eq!(h0, expected_h0);
        let expected_h2 = (e1.double() - e0) * (w1.double() - w0);
        assert_eq!(h2, expected_h2);
    }

    #[test]
    fn test_sumcheck_coefficients_two_variables() {
        let e0 = EF4::from_u64(1);
        let e1 = EF4::from_u64(2);
        let e2 = EF4::from_u64(5);
        let e3 = EF4::from_u64(8);
        let w0 = EF4::from_u64(3);
        let w1 = EF4::from_u64(4);
        let w2 = EF4::from_u64(6);
        let w3 = EF4::from_u64(7);
        let evals = EvaluationsList::new(vec![e0, e1, e2, e3]);
        let weights = EvaluationsList::new(vec![w0, w1, w2, w3]);
        let (h0, h2) = evals.sumcheck_coefficients(&weights);
        let expected_h0 = e0 * w0 + e1 * w1;
        assert_eq!(h0, expected_h0);
        let expected_h2 =
            (e2.double() - e0) * (w2.double() - w0) + (e3.double() - e1) * (w3.double() - w1);
        assert_eq!(h2, expected_h2);
    }

    #[test]
    fn test_sumcheck_coefficients_three_variables() {
        let e0 = EF4::from_u64(1);
        let e1 = EF4::from_u64(2);
        let e2 = EF4::from_u64(3);
        let e3 = EF4::from_u64(4);
        let e4 = EF4::from_u64(5);
        let e5 = EF4::from_u64(6);
        let e6 = EF4::from_u64(7);
        let e7 = EF4::from_u64(8);
        let w0 = EF4::from_u64(10);
        let w1 = EF4::from_u64(20);
        let w2 = EF4::from_u64(30);
        let w3 = EF4::from_u64(40);
        let w4 = EF4::from_u64(50);
        let w5 = EF4::from_u64(60);
        let w6 = EF4::from_u64(70);
        let w7 = EF4::from_u64(80);
        let evals = EvaluationsList::new(vec![e0, e1, e2, e3, e4, e5, e6, e7]);
        let weights = EvaluationsList::new(vec![w0, w1, w2, w3, w4, w5, w6, w7]);
        let (h0, h2) = evals.sumcheck_coefficients(&weights);
        let expected_h0 = e0 * w0 + e1 * w1 + e2 * w2 + e3 * w3;
        assert_eq!(h0, expected_h0);
        let expected_h2 = (e4.double() - e0) * (w4.double() - w0)
            + (e5.double() - e1) * (w5.double() - w1)
            + (e6.double() - e2) * (w6.double() - w2)
            + (e7.double() - e3) * (w7.double() - w3);
        assert_eq!(h2, expected_h2);
    }

    #[test]
    fn test_sumcheck_coefficients_sum_constraint() {
        let e0 = EF4::from_u64(3);
        let e1 = EF4::from_u64(7);
        let w0 = EF4::from_u64(2);
        let w1 = EF4::from_u64(5);
        let evals = EvaluationsList::new(vec![e0, e1]);
        let weights = EvaluationsList::new(vec![w0, w1]);
        let (c0, c2) = evals.sumcheck_coefficients(&weights);
        let h_0 = c0;
        let h_1 = e1 * w1;
        let claimed_sum = e0 * w0 + e1 * w1;
        assert_eq!(h_0 + h_1, claimed_sum);
        let c1 = claimed_sum - c0.double() - c2;
        let h_1_from_coeffs = c0 + c1 + c2;
        assert_eq!(h_1_from_coeffs, h_1);
    }

    #[test]
    fn test_sumcheck_coefficients_evaluate_at_challenge() {
        let e0 = EF4::from_u64(1);
        let e1 = EF4::from_u64(2);
        let e2 = EF4::from_u64(5);
        let e3 = EF4::from_u64(8);
        let w0 = EF4::from_u64(3);
        let w1 = EF4::from_u64(4);
        let w2 = EF4::from_u64(6);
        let w3 = EF4::from_u64(7);
        let evals = EvaluationsList::new(vec![e0, e1, e2, e3]);
        let weights = EvaluationsList::new(vec![w0, w1, w2, w3]);
        let (h0, h2) = evals.sumcheck_coefficients(&weights);
        let h1 = e2 * w2 + e3 * w3;
        let r = EF4::from_u64(7);
        let two = EF4::TWO;
        let l0 = (r - EF4::ONE) * (r - two) / two;
        let l1 = r * (two - r);
        let l2 = r * (r - EF4::ONE) / two;
        let h_r = h0 * l0 + h1 * l1 + h2 * l2;
        let folded_e0 = e0 + r * (e2 - e0);
        let folded_e1 = e1 + r * (e3 - e1);
        let folded_w0 = w0 + r * (w2 - w0);
        let folded_w1 = w1 + r * (w3 - w1);
        let h_r_from_folding = folded_e0 * folded_w0 + folded_e1 * folded_w1;
        assert_eq!(h_r, h_r_from_folding);
    }

    #[test]
    fn test_sumcheck_coefficients_mixed_field_types() {
        let e0 = F::from_u64(3);
        let e1 = F::from_u64(7);
        let w0 = EF4::from_u64(2);
        let w1 = EF4::from_u64(5);
        let evals = EvaluationsList::new(vec![e0, e1]);
        let weights = EvaluationsList::new(vec![w0, w1]);
        let (h0, h2): (EF4, EF4) = evals.sumcheck_coefficients(&weights);
        let expected_h0 = w0 * e0;
        assert_eq!(h0, expected_h0);
        let expected_h2 = (w1.double() - w0) * (e1.double() - e0);
        assert_eq!(h2, expected_h2);
    }

    #[test]
    fn test_sumcheck_coefficients_all_zeros() {
        let evals = EvaluationsList::new(vec![EF4::ZERO; 4]);
        let weights = EvaluationsList::new(vec![EF4::ZERO; 4]);
        let (c0, c2) = evals.sumcheck_coefficients(&weights);
        assert_eq!(c0, EF4::ZERO);
        assert_eq!(c2, EF4::ZERO);
    }

    #[test]
    fn test_sumcheck_coefficients_constant_polynomial() {
        let c = EF4::from_u64(5);
        let d = EF4::from_u64(3);
        let evals = EvaluationsList::new(vec![c, c, c, c]);
        let weights = EvaluationsList::new(vec![d, d, d, d]);
        let (c0, c2) = evals.sumcheck_coefficients(&weights);
        assert_eq!(c0, c * d + c * d);
        assert_eq!(c2, c * d + c * d);
    }

    #[test]
    fn test_sumcheck_coefficients_linear_in_first_variable() {
        let zero = EF4::ZERO;
        let one = EF4::ONE;
        let evals = EvaluationsList::new(vec![zero, zero, one, one]);
        let weights = EvaluationsList::new(vec![one, one, one, one]);
        let (h0, h2) = evals.sumcheck_coefficients(&weights);
        let expected_h0 = zero * one + zero * one;
        assert_eq!(h0, expected_h0);
        let two = EF4::TWO;
        let expected_h2 =
            (two * one - zero) * (two * one - one) + (two * one - zero) * (two * one - one);
        assert_eq!(h2, expected_h2);
    }

    #[test]
    #[should_panic]
    fn test_sumcheck_coefficients_mismatched_lengths() {
        let evals = EvaluationsList::new(vec![EF4::ONE; 4]);
        let weights = EvaluationsList::new(vec![EF4::ONE; 8]);
        let _ = evals.sumcheck_coefficients(&weights);
    }

    #[test]
    #[should_panic]
    fn test_sumcheck_coefficients_single_element() {
        let evals = EvaluationsList::new(vec![EF4::ONE]);
        let weights = EvaluationsList::new(vec![EF4::ONE]);
        let _ = evals.sumcheck_coefficients(&weights);
    }
}
