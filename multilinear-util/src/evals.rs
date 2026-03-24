use alloc::vec;
use alloc::vec::Vec;

use p3_field::{
    Algebra, ExtensionField, Field, PackedFieldExtension, PackedValue, PrimeCharacteristicRing,
};
use p3_matrix::dense::RowMajorMatrixView;
use p3_maybe_rayon::prelude::*;
use p3_util::log2_strict_usize;
use rand::RngExt;
use rand::distr::{Distribution, StandardUniform};
use serde::{Deserialize, Serialize};

use crate::eq_batch::eval_eq_batch;
use crate::multilinear::Point;
use crate::split_eq::SplitEq;

pub(crate) const PARALLEL_THRESHOLD: usize = 4096;

/// Number of variables at which we switch from recursive to chunk-based MLE evaluation.
const MLE_RECURSION_THRESHOLD: usize = 10;

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
pub struct Poly<F>(pub(crate) Vec<F>);

impl<F> Poly<F> {
    /// Initializes the zero polynomial in `num_variables` variables.
    #[inline]
    pub fn zero(num_variables: usize) -> Self
    where
        F: PrimeCharacteristicRing,
    {
        Self(F::zero_vec(1 << num_variables))
    }

    /// Constructs a polynomial from a vector of evaluations.
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
    pub const fn num_vars(&self) -> usize {
        // Safety: The length is guaranteed to be a power of two.
        self.0.len().ilog2() as usize
    }

    /// Returns a reference to the underlying slice of evaluations.
    #[inline]
    #[must_use]
    pub fn as_slice(&self) -> &[F] {
        &self.0
    }

    /// Returns a mutable reference to the underlying slice of evaluations.
    #[inline]
    #[must_use]
    pub fn as_mut_slice(&mut self) -> &mut [F] {
        &mut self.0
    }

    /// Returns an iterator over the evaluations.
    #[inline]
    pub fn iter(&self) -> core::slice::Iter<'_, F> {
        self.0.iter()
    }

    /// Evaluates the polynomial as a constant.
    ///
    /// This is only valid for constant polynomials (i.e., when `num_variables` is 0).
    ///
    /// Returns `None` in other cases.
    #[must_use]
    #[inline]
    pub fn as_constant(&self) -> Option<F>
    where
        F: Copy,
    {
        (self.num_evals() == 1).then_some(self.0[0])
    }

    /// Samples all `2^k` evaluations independently at random.
    pub fn rand(rng: &mut impl rand::Rng, k: usize) -> Self
    where
        StandardUniform: Distribution<F>,
    {
        Self(rng.random_iter().take(1 << k).collect::<Vec<F>>())
    }
}

impl<Packed> Poly<Packed> {
    /// Given a point `P` (as a slice), compute the evaluation vector of the equality
    /// function `eq(P, X)` for all points `X` in the boolean hypercube, scaled by a value.
    ///
    /// ## Arguments
    /// * `point`: A multilinear point.
    /// * `scale`: A scalar value to multiply all evaluations by.
    ///
    /// ## Returns
    /// A packed polynomial containing `scale * eq(point, X)` for all `X` in `{0,1}^n`.
    #[inline]
    pub fn new_packed_from_point<F, EF>(point: &[EF], scale: EF) -> Self
    where
        F: Field,
        EF: ExtensionField<F, ExtensionPacking = Packed>,
        Packed: PackedFieldExtension<F, EF> + Copy,
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
        // This forms a single `Packed` element which acts as the seed for the next stage.
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
    pub fn eval_packed<F, EF>(&self, point: &Point<EF>) -> EF
    where
        F: Field,
        EF: ExtensionField<F, ExtensionPacking = Packed>,
        Packed: PackedFieldExtension<F, EF>,
    {
        SplitEq::new_packed(point, EF::ONE).eval_packed(self)
    }

    pub fn unpack<F, EF>(&self) -> Poly<EF>
    where
        F: Field,
        EF: ExtensionField<F, ExtensionPacking = Packed>,
        Packed: PackedFieldExtension<F, EF> + Copy,
    {
        let mut out = Poly(unsafe {
            uninitialized_vec(1 << (self.num_vars() + log2_strict_usize(F::Packing::WIDTH)))
        });
        self.unpack_into(&mut out);
        out
    }

    pub fn unpack_into<F, EF>(&self, out: &mut Poly<EF>)
    where
        F: Field,
        EF: ExtensionField<F, ExtensionPacking = Packed>,
        Packed: PackedFieldExtension<F, EF> + Copy,
    {
        assert_eq!(
            out.num_vars(),
            self.num_vars() + log2_strict_usize(F::Packing::WIDTH)
        );
        // TODO: should we do in parallel?
        out.0
            .iter_mut()
            .zip(Packed::to_ext_iter(self.iter().copied()))
            .for_each(|(out, packed)| {
                *out = packed;
            });
    }
}

impl<F: Field> Poly<F> {
    /// Given a point `P` (as a slice), compute the evaluation vector of the equality
    /// function `eq(P, X)` for all points `X` in the boolean hypercube, scaled by a value.
    ///
    /// ## Arguments
    /// * `point`: A multilinear point.
    /// * `scale`: A scalar value to multiply all evaluations by.
    ///
    /// ## Returns
    /// A polynomial containing `scale * eq(point, X)` for all `X` in `{0,1}^n`.
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
    pub fn eval_base<EF: ExtensionField<F>>(&self, point: &Point<EF>) -> EF {
        if point.num_vars() < MLE_RECURSION_THRESHOLD {
            eval_multilinear_recursive(&self.0, point.as_slice())
        } else {
            SplitEq::new_packed(point, EF::ONE).eval_base(self)
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
    pub fn eval_ext<BaseField: Field>(&self, point: &Point<F>) -> F
    where
        F: ExtensionField<BaseField>,
    {
        if point.num_vars() < MLE_RECURSION_THRESHOLD {
            eval_multilinear_recursive(&self.0, point.as_slice())
        } else {
            SplitEq::new_packed(point, F::ONE).eval_ext(self)
        }
    }

    /// Fixes the low variables of a multilinear polynomial using the split eq
    /// tables, returning a reduced polynomial over the remaining high variables.
    ///
    /// Given `poly` with `n` variables and split eq with `m ≤ n` variables, computes:
    /// ```text
    ///   out(x_hi) = Σ_{y_lo ∈ {0,1}^m} eq(point, y_lo) · poly(y_lo, x_hi)
    /// ```
    pub fn compress_lo<EF>(&self, point: &Point<EF>, scale: EF) -> Poly<EF>
    where
        EF: ExtensionField<F>,
    {
        SplitEq::<F, EF>::new_packed(point, scale).compress_lo(self)
    }

    /// Like [`compress_lo`](Self::compress_lo), but returns the result in packed
    /// extension-field representation. Requires that `poly` has enough variables
    /// to fill at least one packed element after compression.
    ///
    /// ```text
    ///   out(x_hi) = Σ_{y_lo ∈ {0,1}^m} eq(point, y_lo) · poly(y_lo, x_hi)
    /// ```
    pub fn compress_lo_to_packed<EF>(
        &self,
        point: &Point<EF>,
        scale: EF,
    ) -> Poly<EF::ExtensionPacking>
    where
        EF: ExtensionField<F>,
    {
        SplitEq::<F, EF>::new_packed(point, scale).compress_lo_to_packed(self)
    }

    /// Fixes the high variables of a multilinear polynomial using the split eq
    /// tables, returning a reduced polynomial over the remaining low variables.
    ///
    /// Given `poly` with `n` variables and split eq with `m ≤ n` variables, computes:
    /// ```text
    ///   out(x_lo) = Σ_{y_hi ∈ {0,1}^m} eq(point, y_hi) · poly(x_lo, y_hi)
    /// ```
    pub fn compress_hi<EF>(&self, point: &Point<EF>, scale: EF) -> Poly<EF>
    where
        EF: ExtensionField<F>,
    {
        SplitEq::<F, EF>::new_packed(point, scale).compress_hi(self)
    }

    /// Like [`compress_hi`](Self::compress_hi), but writes into a pre-allocated buffer.
    pub fn compress_hi_into<EF>(&self, out: &mut [EF], point: &Point<EF>, scale: EF)
    where
        EF: ExtensionField<F>,
    {
        SplitEq::<F, EF>::new_packed(point, scale).compress_hi_into(out, self);
    }
}

impl<A: Copy + Send + Sync + PrimeCharacteristicRing> Poly<A> {
    /// Fixes the lowest variable at `zi`, returning a polynomial with one
    /// fewer variable: `p(zi, x') = (1 - zi) · p(0, x') + zi · p(1, x')`.
    ///
    /// Intended for base-field polynomials where the result lifts to `F`.
    /// For extension-field polynomials, consider [`fix_lo_var_mut`](Self::fix_lo_var_mut)
    /// to fix in-place and avoid extra allocation.
    pub fn fix_lo_var<F>(&self, zi: F) -> Poly<F>
    where
        F: Algebra<A> + Copy + Send + Sync,
    {
        assert!(self.as_constant().is_none(), "no free variables");
        let (p0, p1) = self.0.split_at(self.num_evals() / 2);
        if self.num_evals() >= PARALLEL_THRESHOLD {
            Poly::new(
                p0.par_iter()
                    .zip(p1.par_iter())
                    .map(|(&a0, &a1)| zi * (a1 - a0) + a0)
                    .collect::<Vec<_>>(),
            )
        } else {
            Poly::new(
                p0.iter()
                    .zip(p1.iter())
                    .map(|(&a0, &a1)| zi * (a1 - a0) + a0)
                    .collect(),
            )
        }
    }

    /// In-place version of [`fix_lo_var`](Self::fix_lo_var). Writes the folded values
    /// into the first half of the buffer and truncates.
    pub fn fix_lo_var_mut<F: Copy + Send + Sync>(&mut self, zi: F)
    where
        A: Algebra<F>,
    {
        assert!(self.as_constant().is_none(), "no free variables");
        let num_evals = self.num_evals();
        let mid = num_evals / 2;
        let (p0, p1) = self.0.split_at_mut(mid);

        if num_evals >= PARALLEL_THRESHOLD {
            p0.par_iter_mut()
                .zip(p1.par_iter())
                .for_each(|(a0, &a1)| *a0 += (a1 - *a0) * zi);
        } else {
            p0.iter_mut()
                .zip(p1.iter())
                .for_each(|(a0, &a1)| *a0 += (a1 - *a0) * zi);
        }

        self.0.truncate(mid);
    }

    /// Fixes the highest variable at `zi`, returning a polynomial with one
    /// fewer variable: `p(x', zi) = (1 - zi) · p(x', 0) + zi · p(x', 1)`.
    pub fn fix_hi_var<F>(&self, zi: F) -> Poly<F>
    where
        F: Algebra<A> + Copy + Send + Sync,
    {
        assert!(self.as_constant().is_none(), "no free variables");
        if self.num_evals() >= PARALLEL_THRESHOLD {
            Poly::new(
                self.0
                    .par_chunks(2)
                    .map(|a| zi * (a[1] - a[0]) + a[0])
                    .collect(),
            )
        } else {
            Poly::new(
                self.0
                    .chunks(2)
                    .map(|a| zi * (a[1] - a[0]) + a[0])
                    .collect(),
            )
        }
    }

    /// In-place version of [`fix_hi_var`](Self::fix_hi_var). Computes the folded values
    /// into a temporary buffer, then truncates and copies them back.
    pub fn fix_hi_var_mut<F: Copy + Send + Sync>(&mut self, zi: F)
    where
        A: Algebra<F>,
    {
        assert!(self.as_constant().is_none(), "no free variables");
        let src = if self.num_evals() < PARALLEL_THRESHOLD {
            self.0
                .chunks(2)
                .map(|a| (a[1] - a[0]) * zi + a[0])
                .collect::<Vec<_>>()
        } else {
            self.0
                .par_chunks(2)
                .map(|a| (a[1] - a[0]) * zi + a[0])
                .collect::<Vec<_>>()
        };
        let mid = self.num_evals() / 2;
        self.0.truncate(mid);
        self.0.copy_from_slice(&src);
    }

    pub fn pack<F, EF>(&self) -> Poly<A::ExtensionPacking>
    where
        F: Field,
        A: ExtensionField<F>,
    {
        assert!(self.num_vars() >= log2_strict_usize(F::Packing::WIDTH));
        Poly(
            self.0
                .par_chunks(F::Packing::WIDTH)
                .map(|ext| A::ExtensionPacking::from_ext_slice(ext))
                .collect(),
        )
    }
}

impl<'a, F> IntoIterator for &'a Poly<F> {
    type Item = &'a F;
    type IntoIter = core::slice::Iter<'a, F>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.0.iter()
    }
}

impl<F> IntoIterator for Poly<F> {
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

#[cfg(test)]
pub(crate) mod test {

    use alloc::vec;
    use alloc::vec::Vec;

    use p3_baby_bear::BabyBear;
    use p3_field::extension::BinomialExtensionField;
    use p3_field::{
        ExtensionField, Field, PackedValue, PrimeCharacteristicRing, PrimeField64, dot_product,
    };
    use p3_matrix::dense::RowMajorMatrixView;
    use p3_util::log2_strict_usize;
    use proptest::prelude::*;
    use rand::rngs::SmallRng;
    use rand::{RngExt, SeedableRng};

    use crate::eq_batch::eval_eq_batch;
    use crate::evals::{PARALLEL_THRESHOLD, Poly};
    use crate::multilinear::Point;

    type F = BabyBear;
    type PackedF = <F as p3_field::Field>::Packing;
    type EF = BinomialExtensionField<F, 4>;

    /// Naive method to evaluate a multilinear polynomial for testing.
    pub(crate) fn eval_reference<F: Field, EF: ExtensionField<F>>(evals: &[F], point: &[EF]) -> EF {
        let eq = Poly::new_from_point(point, EF::ONE);
        dot_product(eq.iter().copied(), evals.iter().copied())
    }

    #[test]
    fn test_new_evaluations_list() {
        let evals = vec![F::ZERO, F::ONE, F::ZERO, F::ONE];
        let evaluations_list = Poly::new(evals.clone());

        assert_eq!(evaluations_list.num_evals(), evals.len());
        assert_eq!(evaluations_list.num_vars(), 2);
        assert_eq!(evaluations_list.as_slice(), &evals);
    }

    #[test]
    #[should_panic]
    fn test_new_evaluations_list_invalid_length() {
        // Length is not a power of two, should panic
        let _ = Poly::new(vec![F::ONE, F::ZERO, F::ONE]);
    }

    #[test]
    fn test_indexing() {
        let evals = vec![
            F::from_u64(1),
            F::from_u64(2),
            F::from_u64(3),
            F::from_u64(4),
        ];
        let evaluations_list = Poly::new(evals.clone());

        assert_eq!(evaluations_list.0[0], evals[0]);
        assert_eq!(evaluations_list.0[1], evals[1]);
        assert_eq!(evaluations_list.0[2], evals[2]);
        assert_eq!(evaluations_list.0[3], evals[3]);
    }

    #[test]
    #[should_panic]
    fn test_index_out_of_bounds() {
        let evals = vec![F::ZERO, F::ONE, F::ZERO, F::ONE];
        let evaluations_list = Poly::new(evals);

        let _ = evaluations_list.as_slice()[4]; // Index out of range, should panic
    }

    #[test]
    fn test_mutability_of_evals() {
        let mut evals = Poly::new(vec![F::ZERO, F::ONE, F::ZERO, F::ONE]);

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

        let evals = Poly::new(vec![e1, e2, e3, e4]);

        // Evaluating at a binary hypercube point should return the direct value
        assert_eq!(evals.eval_base(&Point::new(vec![F::ZERO, F::ZERO])), e1);
        assert_eq!(evals.eval_base(&Point::new(vec![F::ZERO, F::ONE])), e2);
        assert_eq!(evals.eval_base(&Point::new(vec![F::ONE, F::ZERO])), e3);
        assert_eq!(evals.eval_base(&Point::new(vec![F::ONE, F::ONE])), e4);
    }

    #[test]
    fn test_dimensions() {
        for k in 0..5 {
            let poly = Poly::<F>::zero(k);
            assert_eq!(poly.num_evals(), 1 << k);
            assert_eq!(poly.num_vars(), k);
        }
    }

    #[test]
    fn test_eval_extension_on_non_hypercube_points() {
        let evals = Poly::new(vec![
            F::from_u64(1),
            F::from_u64(2),
            F::from_u64(3),
            F::from_u64(4),
        ]);

        let point = Point::new(vec![F::from_u64(2), F::from_u64(3)]);

        let result = evals.eval_base(&point);

        // Expected result using `eval_reference`
        let expected = eval_reference(evals.as_slice(), point.as_slice());
        assert_eq!(result, expected);
    }

    #[test]
    fn test_eval_multilinear_1d() {
        let a = F::from_u64(5);
        let b = F::from_u64(10);
        let poly = Poly::new(vec![a, b]);

        // Evaluate at midpoint `x = 1/2`
        let x = F::TWO.inverse();
        let expected = a + (b - a) * x;
        assert_eq!(poly.eval_base(&Point::new(vec![x])), expected);
    }

    #[test]
    fn test_eval_multilinear_2d() {
        let a = F::from_u64(1);
        let b = F::from_u64(2);
        let c = F::from_u64(3);
        let d = F::from_u64(4);

        // The evaluations are stored in lexicographic order for (x, y)
        // f(0,0) = a, f(0,1) = c, f(1,0) = b, f(1,1) = d
        let poly = Poly::new(vec![a, b, c, d]);

        // Evaluate at `(x, y) = (1/2, 1/2)`
        let x = F::from_u64(1) / F::from_u64(2);
        let y = F::from_u64(1) / F::from_u64(2);

        // Interpolation formula:
        // f(x, y) = (1-x)(1-y) * f(0,0) + (1-x)y * f(0,1) + x(1-y) * f(1,0) + xy * f(1,1)
        let expected = (F::ONE - x) * (F::ONE - y) * a
            + (F::ONE - x) * y * c
            + x * (F::ONE - y) * b
            + x * y * d;

        assert_eq!(poly.eval_base(&Point::new(vec![x, y])), expected);
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
        let poly = Poly::new(vec![a, b, c, e, d, f, g, h]);

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

        assert_eq!(poly.eval_base(&Point::new(vec![x, y, z])), expected);
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
        let poly = Poly::new(vec![a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p]);

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
        assert_eq!(poly.eval_base(&Point::new(vec![x, y, z, w])), expected);
    }

    proptest! {
        #[test]
        fn prop_eval_multilinear_equiv_between_f_and_ef(
            values in prop::collection::vec(0u64..100, 8),
            x0 in 0u64..100,
            x1 in 0u64..100,
            x2 in 0u64..100,
        ) {
            // Base field evaluations
            let coeffs_f: Vec<F> = values.iter().copied().map(F::from_u64).collect();
            let poly_f = Poly::new(coeffs_f);

            // Lift to extension field EF
            let coeffs_ef: Vec<EF> = values.iter().copied().map(EF::from_u64).collect();
            let poly_ef = Poly::new(coeffs_ef);

            // Evaluation point in EF
            let point_ef = Point::new(vec![
                EF::from_u64(x0),
                EF::from_u64(x1),
                EF::from_u64(x2),
            ]);

            // Evaluate using both base and extension representations
            let eval_f = poly_f.eval_base(&point_ef);
            let eval_ef = poly_ef.eval_ext::<F>(&point_ef);

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
        let evals = Poly::new(vec![e0, e0 + e1, e0 + e2, e0 + e1 + e2 + e3]);

        // Choose evaluation point:
        //
        // Let's pick (x_1, x_2) = (2, 1)
        let x1 = F::from_u64(2);
        let x2 = F::from_u64(1);
        let coords = Point::new(vec![x1, x2]);

        // Manually compute the expected value step-by-step:
        //
        // Reminder:
        //   f(x_1, x_2) = 5 + 6·x_2 + 7·x_1 + 8·x_1·x_2
        //
        // Substituting (x_1, x_2):
        let expected = e0 + e1 * x2 + e2 * x1 + e3 * x1 * x2;

        // Now evaluate using the function under test
        let result = evals.eval_base(&coords);

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

        let evals = Poly::new(vec![
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

        let point = Point::new(vec![x0, x1, x2]);

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

        let result = evals.eval_base(&point);
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

        let evals = Poly::new(vec![
            e0,
            e0 + e1,
            e0 + e2,
            e0 + e1 + e2 + e3,
            e0 + e4,
            e0 + e1 + e4 + e5,
            e0 + e2 + e4 + e6,
            e0 + e1 + e2 + e3 + e4 + e5 + e6 + e7,
        ]);

        let x0 = EF::from_u64(2);
        let x1 = EF::from_u64(3);
        let x2 = EF::from_u64(4);

        let point = Point::new(vec![x0, x1, x2]);

        let expected = EF::from(e0)
            + EF::from(e1) * x2
            + EF::from(e2) * x1
            + EF::from(e3) * x1 * x2
            + EF::from(e4) * x0
            + EF::from(e5) * x0 * x2
            + EF::from(e6) * x0 * x1
            + EF::from(e7) * x0 * x1 * x2;

        let result = evals.eval_base(&point);
        assert_eq!(result, expected);
    }

    #[test]
    fn test_folding_and_evaluation() {
        let num_variables = 10;
        let evals = (0..(1 << num_variables)).map(F::from_u64).collect();
        let evals_list = Poly::new(evals);
        let randomness: Vec<_> = (0..num_variables)
            .map(|i| F::from_u64(35 * i as u64))
            .collect();

        for k in 1..num_variables {
            let fold_part = randomness[0..k].to_vec();
            let eval_part = Point::new(randomness[k..randomness.len()].to_vec());
            let fold_random = Point::new(fold_part.clone());
            let eval_point1 = Point::new([fold_part.clone(), eval_part.0.clone()].concat());
            let folded_evals = evals_list.compress_lo(&fold_random, F::ONE);
            assert_eq!(folded_evals.num_vars(), num_variables - k);
            let folded_coeffs = evals_list.compress_lo(&fold_random, F::ONE);
            assert_eq!(folded_coeffs.num_vars(), num_variables - k);
            assert_eq!(
                folded_evals.eval_base(&eval_part),
                evals_list.eval_base(&eval_point1)
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
        let poly = Poly::new(evals);
        let evals_list: Poly<F> = poly;
        let r1 = EF::from_u64(5);
        let folded = evals_list.compress_lo(&Point::new(vec![r1]), EF::ONE);

        for x0_f in 0..10 {
            let x0 = EF::from_u64(x0_f);
            let full_point = Point::new(vec![r1, x0]);
            let folded_point = Point::new(vec![x0]);
            let expected = evals_list.eval_base(&full_point);
            let actual = folded.eval_base(&folded_point);
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
    fn test_new_from_point_zero_vars() {
        let point = Point::<F>::new(vec![]);
        let value = F::from_u64(42);
        let evals_list = Poly::new_from_point(point.as_slice(), value);
        assert_eq!(evals_list.num_vars(), 0);
        assert_eq!(evals_list.as_slice(), &[value]);
        assert_eq!(evals_list.as_constant(), Some(value));
    }

    #[test]
    fn test_new_from_point_one_var() {
        let p0 = F::from_u64(7);
        let point = Point::new(vec![p0]);
        let value = F::from_u64(3);
        let evals_list = Poly::new_from_point(point.as_slice(), value);
        let expected = vec![value * (F::ONE - p0), value * p0];
        assert_eq!(evals_list.num_vars(), 1);
        assert_eq!(evals_list.as_slice(), &expected);
    }

    #[test]
    #[allow(clippy::identity_op)]
    fn test_new_from_point_three_vars() {
        let p = [F::from_u64(2), F::from_u64(3), F::from_u64(5)];
        let value = F::from_u64(10);
        let evals_list = Poly::new_from_point(&p, value);
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
        assert_eq!(evals_list.num_vars(), 3);
        assert_eq!(evals_list.as_slice(), &expected);
    }

    #[test]
    fn test_as_constant_for_constant_poly() {
        let constant_value = F::from_u64(42);
        let evals = Poly::new(vec![constant_value]);
        assert_eq!(evals.num_vars(), 0);
        assert_eq!(evals.as_constant(), Some(constant_value));
    }

    #[test]
    fn test_as_constant_for_non_constant_poly() {
        let evals = Poly::new(vec![F::ONE, F::ZERO, F::ONE, F::ZERO]);
        assert_ne!(evals.num_vars(), 0);
        assert_eq!(evals.as_constant(), None);
    }

    #[test]
    #[should_panic]
    fn test_compress_panics_on_constant() {
        let mut evals_list = Poly::new(vec![F::from_u64(42)]);
        evals_list.fix_hi_var_mut(F::ONE);
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
        let mut evals_list = Poly::new(initial_evals);
        let r = F::from_u64(10);
        let expected = vec![
            r * (p_100 - p_000) + p_000,
            r * (p_101 - p_001) + p_001,
            r * (p_110 - p_010) + p_010,
            r * (p_111 - p_011) + p_011,
        ];
        assert_eq!(evals_list.num_vars(), 3);
        assert_eq!(evals_list.num_evals(), 8);
        evals_list.fix_lo_var_mut(r);
        assert_eq!(evals_list.num_vars(), 2);
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
        let mut evals_list = Poly::new(initial_evals);
        let r = F::from_u64(3);
        let num_variables_before = evals_list.num_vars();
        evals_list.fix_lo_var_mut(r);
        assert_eq!(evals_list.num_vars(), num_variables_before - 1);
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
        let mut evals_list = Poly::new(initial_evals);
        let challenges = vec![F::from_u64(3), F::from_u64(7), F::from_u64(11)];
        for &r in &challenges {
            evals_list.fix_lo_var_mut(r);
        }
        assert_eq!(evals_list.num_vars(), 1);
        assert_eq!(evals_list.num_evals(), 2);
    }

    #[test]
    fn test_compress_single_variable() {
        let p_0 = F::from_u64(5);
        let p_1 = F::from_u64(9);
        let mut evals_list = Poly::new(vec![p_0, p_1]);
        let r = F::from_u64(7);
        evals_list.fix_lo_var_mut(r);
        assert_eq!(evals_list.num_vars(), 0);
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
            Poly::new(vec![p_000, p_001, p_010, p_011, p_100, p_101, p_110, p_111]);
        let r = F::ZERO;
        evals_list.fix_lo_var_mut(r);
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
            Poly::new(vec![p_000, p_001, p_010, p_011, p_100, p_101, p_110, p_111]);
        let r = F::ONE;
        evals_list.fix_lo_var_mut(r);
        let expected = vec![p_100, p_101, p_110, p_111];
        assert_eq!(evals_list.as_slice(), &expected);
    }

    proptest! {
        #[test]
        fn prop_compress_dimensions(
            n in 1usize..=10,
            seed in any::<u64>(),
        ) {
            let mut rng = SmallRng::seed_from_u64(seed);
            let num_evals = 1 << n;
            let evals: Vec<F> = (0..num_evals).map(|_| rng.random()).collect();
            let r: F = rng.random();

            let mut list = Poly::new(evals);
            list.fix_lo_var_mut(r);

            prop_assert_eq!(list.num_vars(), n - 1);
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

            let mut list_zero = Poly::new(evals.clone());
            list_zero.fix_lo_var_mut(F::ZERO);
            prop_assert_eq!(list_zero.num_evals(), num_evals / 2);

            let mut list_one = Poly::new(evals);
            list_one.fix_lo_var_mut(F::ONE);
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

            let mut list = Poly::new(evals);
            for &r in &challenges {
                list.fix_lo_var_mut(r);
            }

            prop_assert_eq!(list.num_vars(), n - actual_rounds);
            prop_assert_eq!(list.num_evals(), 1 << (n - actual_rounds));
        }
    }

    #[test]
    fn test_fold_batch_no_challenges() {
        let poly = Poly::new(vec![
            F::from_u64(1),
            F::from_u64(2),
            F::from_u64(3),
            F::from_u64(4),
            F::from_u64(5),
            F::from_u64(6),
            F::from_u64(7),
            F::from_u64(8),
        ]);
        let result = poly.compress_lo(&Point::new(vec![]), EF::ONE);
        assert_eq!(result.0.len(), 8, "Result should have 8 evaluations");
        let expected_poly = Poly::new(vec![
            EF::from_u64(1),
            EF::from_u64(2),
            EF::from_u64(3),
            EF::from_u64(4),
            EF::from_u64(5),
            EF::from_u64(6),
            EF::from_u64(7),
            EF::from_u64(8),
        ]);
        assert_eq!(
            result, expected_poly,
            "Result should be the original polynomial"
        );
    }

    #[test]
    fn test_fold_batch_single_variable() {
        let poly = Poly::new((1..=8).map(F::from_u64).collect());
        let r2 = EF::from_u64(3);
        let challenges = vec![r2];
        let result = poly.compress_lo(&Point::new(challenges), EF::ONE);
        assert_eq!(
            result.0.len(),
            4,
            "Folded polynomial should have 4 evaluations"
        );
        let eq_r2_0 = EF::ONE - r2;
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
        let poly = Poly::new(vec![
            F::from_u64(1),
            F::from_u64(2),
            F::from_u64(4),
            F::from_u64(8),
            F::from_u64(16),
            F::from_u64(32),
            F::from_u64(64),
            F::from_u64(128),
        ]);
        let r2 = EF::from_u64(2);
        let r1 = EF::from_u64(3);
        let challenges = vec![r2, r1];
        let result = poly.compress_lo(&Point::new(challenges), EF::ONE);
        assert_eq!(
            result.0.len(),
            2,
            "Folded polynomial should have 2 evaluations"
        );
        let eq_r2_0 = EF::ONE - r2;
        let eq_r2_1 = r2;
        let eq_r1_0 = EF::ONE - r1;
        let eq_r1_1 = r1;
        let eq_00 = eq_r2_0 * eq_r1_0;
        let eq_01 = eq_r2_0 * eq_r1_1;
        let eq_10 = eq_r2_1 * eq_r1_0;
        let eq_11 = eq_r2_1 * eq_r1_1;
        let p_000 = EF::from_u64(1);
        let p_001 = EF::from_u64(2);
        let p_010 = EF::from_u64(4);
        let p_011 = EF::from_u64(8);
        let p_100 = EF::from_u64(16);
        let p_101 = EF::from_u64(32);
        let p_110 = EF::from_u64(64);
        let p_111 = EF::from_u64(128);
        let q_0 = eq_00 * p_000 + eq_01 * p_010 + eq_10 * p_100 + eq_11 * p_110;
        let q_1 = eq_00 * p_001 + eq_01 * p_011 + eq_10 * p_101 + eq_11 * p_111;
        assert_eq!(result.0[0], q_0, "q(0) mismatch");
        assert_eq!(result.0[1], q_1, "q(1) mismatch");
    }

    #[test]
    fn test_fold_batch_all_variables() {
        let poly = Poly::new(vec![
            F::from_u64(1),
            F::from_u64(2),
            F::from_u64(3),
            F::from_u64(4),
        ]);
        let r1 = EF::from_u64(5);
        let r0 = EF::from_u64(7);
        let challenges = vec![r1, r0];
        let result = poly.compress_lo(&Point::new(challenges), EF::ONE);
        assert_eq!(
            result.0.len(),
            1,
            "Folding all variables should produce a single value"
        );
        let eq_r1_0 = EF::ONE - r1;
        let eq_r1_1 = r1;
        let eq_r0_0 = EF::ONE - r0;
        let eq_r0_1 = r0;
        let eq_00 = eq_r1_0 * eq_r0_0;
        let eq_01 = eq_r1_0 * eq_r0_1;
        let eq_10 = eq_r1_1 * eq_r0_0;
        let eq_11 = eq_r1_1 * eq_r0_1;
        let p_00 = EF::from_u64(1);
        let p_01 = EF::from_u64(2);
        let p_10 = EF::from_u64(3);
        let p_11 = EF::from_u64(4);
        let q = eq_00 * p_00 + eq_01 * p_01 + eq_10 * p_10 + eq_11 * p_11;
        assert_eq!(result.0[0], q, "Folded value mismatch");
    }

    #[test]
    #[should_panic(expected = "assertion failed: self.num_vars() <= poly.num_vars()")]
    fn test_fold_batch_too_many_challenges() {
        let poly = Poly::new(vec![
            F::from_u64(1),
            F::from_u64(2),
            F::from_u64(3),
            F::from_u64(4),
        ]);
        let challenges = vec![EF::from_u64(2), EF::from_u64(3), EF::from_u64(5)];
        let _ = poly.compress_lo(&Point::new(challenges), EF::ONE);
    }

    #[test]
    fn test_eval_base_field() {
        let mut rng = SmallRng::seed_from_u64(0);
        for k in 1..=20 {
            let poly = Poly::<F>::rand(&mut rng, k);
            assert_eq!(poly.num_evals(), 1 << k);
            assert_eq!(poly.num_vars(), k);
            let point: Point<EF> = Point::rand(&mut rng, k);
            assert_eq!(
                eval_reference(poly.as_slice(), point.as_slice()),
                poly.eval_base(&point)
            );
        }
    }

    #[test]
    fn test_eval_ext_field() {
        let mut rng = SmallRng::seed_from_u64(0);
        for k in 1..=20 {
            let poly = Poly::<EF>::rand(&mut rng, k);
            assert_eq!(poly.num_evals(), 1 << k);
            assert_eq!(poly.num_vars(), k);
            let point: Point<EF> = Point::rand(&mut rng, k);

            assert_eq!(
                eval_reference(poly.as_slice(), point.as_slice()),
                poly.eval_ext::<F>(&point)
            );

            // base field eval should work
            assert_eq!(
                eval_reference(poly.as_slice(), point.as_slice()),
                poly.eval_base(&point)
            );
        }
    }

    #[test]
    fn test_eval_packed_field() {
        let mut rng = SmallRng::seed_from_u64(0);
        let k_pack = log2_strict_usize(PackedF::WIDTH);
        for k in k_pack..=20 {
            let poly = Poly::<EF>::rand(&mut rng, k);
            assert_eq!(poly.num_evals(), 1 << k);
            assert_eq!(poly.num_vars(), k);

            let point: Point<EF> = Point::rand(&mut rng, k);
            let packed_poly = poly.pack::<F, EF>();
            assert_eq!(packed_poly.num_evals(), 1 << (k - k_pack));
            assert_eq!(packed_poly.num_vars(), k - k_pack);

            assert_eq!(
                eval_reference(poly.as_slice(), point.as_slice()),
                poly.eval_packed(&point)
            );
        }
    }

    #[test]
    fn test_fix_lo_var() {
        let mut rng = SmallRng::seed_from_u64(0);
        for k in 1..=20 {
            let poly = Poly::<F>::rand(&mut rng, k);
            let point: Point<EF> = Point::rand(&mut rng, k);

            // returning variant
            let z0 = point.as_slice().first().copied().unwrap();
            let mut compressed = poly.fix_lo_var(z0);
            for &zi in point.as_slice().iter().skip(1) {
                compressed = compressed.fix_lo_var(zi);
            }
            assert_eq!(compressed.as_constant().unwrap(), poly.eval_base(&point));

            // mutable variant
            let z0 = point.as_slice().first().copied().unwrap();
            let mut compressed = poly.fix_lo_var(z0);
            for &zi in point.as_slice().iter().skip(1) {
                compressed.fix_lo_var_mut(zi);
            }
            assert_eq!(compressed.as_constant().unwrap(), poly.eval_base(&point));
        }
    }

    #[test]
    fn test_fix_hi_var() {
        let mut rng = SmallRng::seed_from_u64(0);
        for k in 1..=20 {
            let poly = Poly::<F>::rand(&mut rng, k);
            let point: Point<EF> = Point::rand(&mut rng, k);

            // returning variant
            let z0 = point.as_slice().last().copied().unwrap();
            let mut compressed = poly.fix_hi_var(z0);
            for &zi in point.as_slice().iter().rev().skip(1) {
                compressed = compressed.fix_hi_var(zi);
            }
            assert_eq!(compressed.as_constant().unwrap(), poly.eval_base(&point));

            // mutable variant
            let z0 = point.as_slice().last().copied().unwrap();
            let mut compressed = poly.fix_hi_var(z0);
            for &zi in point.as_slice().iter().rev().skip(1) {
                compressed.fix_hi_var_mut(zi);
            }
            assert_eq!(compressed.as_constant().unwrap(), poly.eval_base(&point));
        }
    }

    #[test]
    fn test_compress_hi() {
        let mut rng = SmallRng::seed_from_u64(0);
        for k in 1..=20 {
            let poly = Poly::<F>::rand(&mut rng, k);
            for point_k in 1..k {
                let point: Point<EF> = Point::rand(&mut rng, point_k);

                let z0 = point.as_slice().first().copied().unwrap();
                let mut compressed0 = poly.fix_lo_var(z0);
                for &zi in point.as_slice().iter().skip(1) {
                    compressed0.fix_lo_var_mut(zi);
                }
                let compressed1 = poly.compress_lo(&point, EF::ONE);
                assert_eq!(compressed0.num_vars(), compressed1.num_vars());
                assert_eq!(compressed0, compressed1);

                if k > point_k + log2_strict_usize(PackedF::WIDTH) {
                    let compressed1 = poly
                        .compress_lo_to_packed(&point, EF::ONE)
                        .unpack::<F, EF>();
                    assert_eq!(compressed0.num_vars(), compressed1.num_vars());
                    assert_eq!(compressed0, compressed1);
                }
            }
        }
    }

    #[test]
    fn test_compress_lo() {
        let mut rng = SmallRng::seed_from_u64(0);
        for k in 1..=20 {
            let poly = Poly::<F>::rand(&mut rng, k);
            for point_k in 1..k {
                let point: Point<EF> = Point::rand(&mut rng, point_k);

                let z0 = point.as_slice().last().copied().unwrap();
                let mut compressed0 = poly.fix_hi_var(z0);
                for &zi in point.as_slice().iter().rev().skip(1) {
                    compressed0.fix_hi_var_mut(zi);
                }
                let compressed1 = poly.compress_hi(&point, EF::ONE);
                assert_eq!(compressed0.num_vars(), compressed1.num_vars());
                assert_eq!(compressed0, compressed1);
            }
        }
    }
}
