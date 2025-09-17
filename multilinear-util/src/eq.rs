//! This module provides optimized routines for computing the **multilinear equality polynomial**
//! over the Boolean hypercube `{0,1}^n`.
//!
//! The equality polynomial `eq(x, z)` evaluates to 1 if `x == z`, and 0 otherwise.
//! It is defined as:
//!
//! ```text
//! eq(x, z) = \prod_{i=0}^{n-1} (x_i ⋅ z_i + (1 - x_i)(1 - z_i))
//! ```
//!
//! These values are computed over all `x ∈ {0,1}^n` efficiently using a recursive strategy.
//! The key relation used is:
//!
//! ```text
//! eq((0, x), z) = (1 - z_0) ⋅ eq(x, z[1:])
//! eq((1, x), z) = z_0 ⋅ eq(x, z[1:])
//! ```
//!
//! Which allows us to reuse the common factor `eq(x, z[1:])`.
//!
//! ## `INITIALIZED` flag
//!
//! Each function accepts a `const INITIALIZED: bool` flag to control how output is written:
//!
//! - If `INITIALIZED = false`: the result is **written** into the output buffer.
//! - If `INITIALIZED = true`: the result is **added** to the output buffer.
//!
//! The output buffer must always be of length `2^n` for `n` variables.

use p3_field::{
    Algebra, ExtensionField, Field, PackedFieldExtension, PackedValue, PrimeCharacteristicRing,
};
use p3_matrix::Matrix;
use p3_matrix::dense::{RowMajorMatrix, RowMajorMatrixView, RowMajorMatrixViewMut};
use p3_maybe_rayon::prelude::*;
use p3_util::{iter_array_chunks_padded, log2_strict_usize};

#[inline]
pub fn eval_eq<F, EF, const INITIALIZED: bool>(eval: &[EF], out: &mut [EF], scalar: EF)
where
    F: Field,
    EF: ExtensionField<F>,
{
    // Pass the combined method using the `ExtFieldEvaluator` strategy.
    eval_eq_batch::<F, EF, INITIALIZED>(RowMajorMatrixView::new(eval, 1), out, &[scalar]);
}

#[inline]
pub fn eval_eq_base<F, EF, const INITIALIZED: bool>(eval: &[F], out: &mut [EF], scalar: EF)
where
    F: Field,
    EF: ExtensionField<F>,
{
    // Pass the combined method using the `BaseFieldEvaluator` strategy.
    eval_eq_base_batch::<F, EF, INITIALIZED>(RowMajorMatrixView::new(eval, 1), out, &[scalar]);
}

/// Computes the multilinear equality polynomial `α ⋅ eq(x, z)` over all `x ∈ \{0,1\}^n` for a point `z ∈ EF^n` and a
/// scalar `α ∈ EF`.
///
/// The multilinear equality polynomial is defined as:
/// ```text
///     eq(x, z) = \prod_{i=0}^{n-1} (x_i z_i + (1 - x_i)(1 - z_i)).
/// ```
///
/// # Output Structure
/// The `out` buffer must have length exactly `2^n`, where `n = eval.len()`.
///
/// Each index `i` in `out` corresponds to the binary vector `x` given by the **big-endian** bit decomposition of `i`.
/// That is:
/// - `out[0]` corresponds to `x = (0, 0, ..., 0)`
/// - `out[1]` corresponds to `x = (0, 0, ..., 1)`
/// - ...
/// - `out[2^n - 1]` corresponds to `x = (1, 1, ..., 1)`
///
/// # Arguments
/// - `eval`: Evaluation point `z ∈ EF^n`
/// - `out`: Mutable slice of `EF` of size `2^n`
/// - `scalar`: Scalar multiplier `α ∈ EF`
#[inline]
pub fn eval_eq_batch<F, EF, const INITIALIZED: bool>(
    evals: RowMajorMatrixView<EF>,
    out: &mut [EF],
    scalars: &[EF],
) where
    F: Field,
    EF: ExtensionField<F>,
{
    // Pass the combined method using the `ExtFieldEvaluator` strategy.
    eval_eq_common::<F, EF, EF, ExtFieldEvaluator<F, EF>, INITIALIZED>(evals, out, scalars);
}

/// Computes the multilinear equality polynomial `α ⋅ eq(x, z)` over all `x ∈ \{0,1\}^n` for a point `z ∈ F^n` and a
/// scalar `α ∈ EF`.
///
/// The multilinear equality polynomial is defined as:
/// ```text
///     eq(x, z) = \prod_{i=0}^{n-1} (x_i z_i + (1 - x_i)(1 - z_i)).
/// ```
///
/// and stores the scaled results into the `out` buffer.
///
/// # Output Structure
/// The `out` buffer must have length exactly `2^n`, where `n = eval.len()`.
///
/// Each index `i` in `out` corresponds to the binary vector `x` given by the **big-endian** bit decomposition of `i`.
/// That is:
/// - `out[0]` corresponds to `x = (0, 0, ..., 0)`
/// - `out[1]` corresponds to `x = (0, 0, ..., 1)`
/// - ...
/// - `out[2^n - 1]` corresponds to `x = (1, 1, ..., 1)`
///
/// # Arguments
/// - `eval`: Evaluation point `z ∈ F^n`
/// - `out`: Mutable slice of `EF` of size `2^n`
/// - `scalar`: Scalar multiplier `α ∈ EF`
#[inline]
pub fn eval_eq_base_batch<F, EF, const INITIALIZED: bool>(
    evals: RowMajorMatrixView<F>,
    out: &mut [EF],
    scalars: &[EF],
) where
    F: Field,
    EF: ExtensionField<F>,
{
    // Pass the combined method using the `BaseFieldEvaluator` strategy.
    eval_eq_common::<F, F, EF, BaseFieldEvaluator<F, EF>, INITIALIZED>(evals, out, scalars)
}

/// Fills the `buffer` with evaluations of the equality polynomial
/// of degree `points.len()` multiplied by the value at `buffer[0]`.
///
/// Assume that `buffer[0]` contains `{eq(i, x)}` for `i \in \{0, 1\}^j` packed into a single
/// PackedExtensionField element. This function fills out the remainder of the buffer so that
/// `buffer[ind]` contains `{eq(ind, points) * eq(i, x)}` for `i \in \{0, 1\}^j`. Note that
/// `ind` is interpreted as an element of `\{0, 1\}^{points.len()}`.
#[inline(always)]
fn fill_buffer<F, A>(evals: RowMajorMatrixView<F>, mut buffer: RowMajorMatrixViewMut<A>)
where
    F: Field,
    A: Algebra<F> + Send + Sync,
{
    for (ind, eval_row) in evals.row_slices().rev().enumerate() {
        let stride = 1 << ind;

        let (mut current_buffer, mut new_buffer) = buffer.split_rows_mut(stride);

        current_buffer
            .rows_mut()
            .zip(new_buffer.rows_mut())
            .for_each(|(currents, news)| {
                currents
                    .iter_mut()
                    .zip(news.iter_mut().zip(eval_row))
                    .for_each(|(val, (new, &eval))| {
                        *new = val.clone() * eval;
                        *val -= new.clone();
                    });
            });
    }
}

/// Compute the scaled multilinear equality polynomial over `{0,1}`.
///
/// # Arguments
/// - `evals`: Matrix of height 1 containing a set of evaluation points `[z_0, z_1, ...]`.
/// - `scalars`: A slice of field elements `[α_0, α_1, ...]` used to scale the result.
///
/// # Returns
/// An array of summed scaled evaluations:
/// - The first element is `∑ α_i ⋅ eq(0, z_i) = ∑ α_i ⋅ (1 - z_i)`
/// - The second element is `∑ α_i ⋅ eq(1, z_i) = ∑ α_i ⋅ z_i`
#[inline(always)]
fn eval_eq_1<F, FP>(evals: RowMajorMatrixView<F>, scalars: &[FP]) -> [FP; 2]
where
    F: Field,
    FP: Algebra<F> + Copy,
{
    assert!(evals.height() == 1);

    // Compute ∑ α_i
    let sum: FP = scalars.iter().cloned().sum();

    // Compute ∑ α_i ⋅ z_i
    let eq_1_sum: FP = evals
        .values
        .iter()
        .zip(scalars.iter())
        .map(|(&z_0, &scalar)| scalar * z_0)
        .sum();

    [eq_1_sum - sum, eq_1_sum]
}

/// Compute the scaled multilinear equality polynomial over `{0,1}²`.
///
/// # Arguments
/// - `evals`: Matrix of height 2 whose columns correspond to evaluations points `[z_00, z_10, ..., z_01, z_11, ...]`
/// - `scalars`: A slice of field elements `[α_0, α_1, ...]` used to scale the result.
///
/// # Returns
/// An array of summed scaled evaluations:
/// - The first element is `∑ α_i ⋅ eq([0, 0], [z_i0, z_i1]) = ∑ α_i ⋅ (1 - z_i0) ⋅ (1 - z_i1)`
/// - The second element is `∑ α_i ⋅ eq([0, 1], [z_i0, z_i1]) = ∑ α_i ⋅ (1 - z_i0) ⋅ z_i1`
/// - The third element is `∑ α_i ⋅ eq([1, 0], [z_i0, z_i1]) = ∑ α_i ⋅ z_i0 ⋅ (1 - z_i1)`
/// - The fourth element is `∑ α_i ⋅ eq([1, 1], [z_i0, z_i1]) = ∑ α_i ⋅ z_i0 ⋅ z_i1`
#[inline(always)]
fn eval_eq_2<F, FP>(evals: RowMajorMatrixView<F>, scalars: &[FP]) -> [FP; 4]
where
    F: Field,
    FP: Algebra<F> + Copy,
{
    assert!(evals.height() == 2);
    let (first_row, second_row) = evals.split_rows(1);

    let (eq_0s, eq_1s): (Vec<_>, Vec<_>) = first_row
        .values
        .into_iter()
        .zip(scalars.iter())
        .map(|(&z_0, &scalar)| {
            let eq_1 = scalar * z_0;
            let eq_0 = scalar - eq_1;
            (eq_0, eq_1)
        })
        .unzip();

    // Recurse to calculate evaluations for the remaining variable
    let [eq_00, eq_01] = eval_eq_1(second_row, &eq_0s);
    let [eq_10, eq_11] = eval_eq_1(second_row, &eq_1s);

    // Return values in lexicographic order of x = (x_0, x_1)
    [eq_00, eq_01, eq_10, eq_11]
}

/// Compute the scaled multilinear equality polynomial over `{0,1}³`.
///
/// # Arguments
/// - `eval`: Matrix of height 3 whose columns correspond to evaluations points.
/// - `scalar`: A slice of field elements `[α_0, α_1, ...]` used to scale the result.
///
/// # Returns
/// An array of summed scaled evaluations containing `∑ α_i ⋅ eq(x, z_i)` for `x ∈ {0,1}³` arranged using lexicographic order of `x`.
#[inline(always)]
fn eval_eq_3<F, FP>(evals: RowMajorMatrixView<F>, scalars: &[FP]) -> [FP; 8]
where
    F: Field,
    FP: Algebra<F> + Copy,
{
    assert_eq!(evals.height(), 3);

    let (first_row, remainder) = evals.split_rows(1);

    let (eq_0s, eq_1s): (Vec<_>, Vec<_>) = first_row
        .values
        .into_iter()
        .zip(scalars.iter())
        .map(|(&z_0, &scalar)| {
            let eq_1 = scalar * z_0;
            let eq_0 = scalar - eq_1;
            (eq_0, eq_1)
        })
        .unzip();

    // Recurse to calculate evaluations for the remaining variables
    let [eq_000, eq_001, eq_010, eq_011] = eval_eq_2(remainder, &eq_0s);
    let [eq_100, eq_101, eq_110, eq_111] = eval_eq_2(remainder, &eq_1s);

    // Return all 8 evaluations in lexicographic order of x ∈ {0,1}³
    [
        eq_000, eq_001, eq_010, eq_011, eq_100, eq_101, eq_110, eq_111,
    ]
}

/// A trait which allows us to define similar but subtly different evaluation strategies depending
/// on the incoming field types.
trait EqualityEvaluator {
    type InputField;
    type OutputField;
    type PackedField: Algebra<Self::InputField> + Copy + Send + Sync;
    type PackedFieldExt: Algebra<Self::InputField> + Copy + Send + Sync;

    fn init_packed(
        evals: RowMajorMatrixView<Self::InputField>,
        init_values: &[Self::OutputField],
    ) -> Vec<Self::PackedField>;

    fn process_chunk<const INITIALIZED: bool>(
        eval: RowMajorMatrixView<Self::InputField>,
        out_chunk: &mut [Self::OutputField],
        buffer_vals: &[Self::PackedField],
        scalars: &[Self::OutputField],
    );

    fn convert(buffer_vals: &[Self::PackedField], scalars: &[Self::OutputField]) -> Vec<Self::PackedFieldExt>;

    fn accumulate_results<const INITIALIZED: bool, const N: usize>(
        out: &mut [Self::OutputField],
        eq_evals: [Self::PackedFieldExt; N],
    );
}

/// Evaluation Strategy for the base field case.
///
/// We stay in the base field for as long as possible to simplify instructions and
/// reduce the amount of data transferred between cores. In particular this means we
/// hold off on scaling by `scalar` until the very end.
struct BaseFieldEvaluator<F, EF>(std::marker::PhantomData<(F, EF)>);

/// Implementation for extension field case.
///
/// We initialise with `scalar` instead of `1` as this reduces the total
/// number of multiplications we need to do.
struct ExtFieldEvaluator<F, EF>(std::marker::PhantomData<(F, EF)>);

impl<F: Field, EF: ExtensionField<F>> EqualityEvaluator for ExtFieldEvaluator<F, EF> {
    type InputField = EF;
    type OutputField = EF;
    type PackedField = EF::ExtensionPacking;
    type PackedFieldExt = EF::ExtensionPacking;

    #[inline]
    fn init_packed(
        evals: RowMajorMatrixView<Self::InputField>,
        init_values: &[Self::OutputField],
    ) -> Vec<Self::PackedField> {
        packed_eq_poly(evals, init_values)
    }

    #[inline]
    fn process_chunk<const INITIALIZED: bool>(
        eval: RowMajorMatrixView<Self::InputField>,
        out_chunk: &mut [Self::OutputField],
        buffer_vals: &[Self::PackedField],
        scalars: &[Self::OutputField],
    ) {
        eval_eq_packed::<F, EF, EF, Self, INITIALIZED>(eval, out_chunk, buffer_vals, scalars);
    }

    #[inline]
    fn convert(buffer_vals: &[Self::PackedField], _scalars: &[Self::OutputField]) -> Vec<Self::PackedFieldExt> {
        buffer_vals.to_vec()
    }

    #[inline]
    fn accumulate_results<const INITIALIZED: bool, const N: usize>(
        out: &mut [Self::OutputField],
        eq_evals: [Self::PackedFieldExt; N],
    ) {
        // Unpack the evaluations back into EF elements and add to output.
        // We use `iter_array_chunks_padded` to allow us to use `add_slices` without
        // needing a vector allocation. Note that `eq_evaluations: [EF::ExtensionPacking: N]`
        // so we know that `out.len() = N * F::Packing::WIDTH` meaning we can use `chunks_exact_mut`
        // and `iter_array_chunks_padded` will never actually pad anything.
        // This avoids needing to allocation the extension iter to a vector.
        iter_array_chunks_padded::<_, N>(EF::ExtensionPacking::to_ext_iter(eq_evals), EF::ZERO)
            .zip(out.chunks_exact_mut(N))
            .for_each(|(res, out_chunk)| {
                add_or_set::<_, INITIALIZED>(out_chunk, &res);
            });
    }
}

impl<F: Field, EF: ExtensionField<F>> EqualityEvaluator for BaseFieldEvaluator<F, EF> {
    type InputField = F;
    type OutputField = EF;
    type PackedField = F::Packing;
    type PackedFieldExt = EF::ExtensionPacking;

    #[inline]
    fn init_packed(
        evals: RowMajorMatrixView<Self::InputField>,
        _init_values: &[Self::OutputField],
    ) -> Vec<Self::PackedField> {
        let const_scalars = vec![F::ONE; evals.width()];
        packed_eq_poly(evals, &const_scalars)
    }

    #[inline]
    fn process_chunk<const INITIALIZED: bool>(
        eval: RowMajorMatrixView<Self::InputField>,
        out_chunk: &mut [Self::OutputField],
        buffer_vals: &[Self::PackedField],
        scalars: &[Self::OutputField],
    ) {
        eval_eq_packed::<F, F, EF, Self, INITIALIZED>(eval, out_chunk, buffer_vals, scalars);
    }

    #[inline]
    fn convert(buffer_vals: &[Self::PackedField], scalars: &[Self::OutputField]) -> Vec<Self::PackedFieldExt> {
        scalars.iter().zip(buffer_vals).map(|(&s, &b)| Into::<Self::PackedFieldExt>::into(s) * b).collect()
    }

    #[inline]
    fn accumulate_results<const INITIALIZED: bool, const N: usize>(
        out: &mut [Self::OutputField],
        eq_evals: [Self::PackedFieldExt; N],
    ) {
        // Unpack the evaluations back into EF elements and add to output.
        // We use `iter_array_chunks_padded` to allow us to use `add_slices` without
        // needing a vector allocation. Note that `eq_evaluations: [EF::ExtensionPacking: N]`
        // so we know that `out.len() = N * F::Packing::WIDTH` meaning we can use `chunks_exact_mut`
        // and `iter_array_chunks_padded` will never actually pad anything.
        // This avoids needing to allocation the extension iter to a vector.
        iter_array_chunks_padded::<_, N>(EF::ExtensionPacking::to_ext_iter(eq_evals), EF::ZERO)
            .zip(out.chunks_exact_mut(N))
            .for_each(|(res, out_chunk)| {
                add_or_set::<_, INITIALIZED>(out_chunk, &res);
            });
    }
}

/// Computes the multilinear equality polynomial `α ⋅ eq(x, z)` over all `x ∈ \{0,1\}^n` for a point `z ∈ IF^n` and a
/// scalar `α ∈ EF`.
///
/// The multilinear equality polynomial is defined as:
/// ```text
///     eq(x, z) = \prod_{i=0}^{n-1} (x_i z_i + (1 - x_i)(1 - z_i)).
/// ```
///
/// The parameter: `E: EqualityEvaluator` lets this function adopt slightly different optimization strategies depending
/// on whether `F = IF` or `IF = EF`.
///
/// # Arguments:
/// - `eval_points`: The point the equality function is being evaluated at.
/// - `out`: The output buffer to store or accumulate the results.
/// - `eq_evals`: The packed evaluations of the equality polynomial.
/// - `scalar`: An optional value which may be used to scale the result depending on the strategy used
///   by the `EqualityEvaluator`.
#[inline]
fn eval_eq_common<F, IF, EF, E, const INITIALIZED: bool>(
    evals: RowMajorMatrixView<IF>,
    out: &mut [EF],
    scalars: &[EF],
) where
    F: Field,
    IF: Field,
    EF: ExtensionField<F> + ExtensionField<IF>,
    E: EqualityEvaluator<InputField = IF, OutputField = EF>,
{
    // Ensure that the scalar slice is of the correct length.
    debug_assert_eq!(evals.width(), scalars.len());

    let num_variables = evals.height();

    // we assume that packing_width is a power of 2.
    let packing_width = F::Packing::WIDTH;
    let num_threads = current_num_threads().next_power_of_two();
    let log_num_threads = log2_strict_usize(num_threads);

    // If the number of variables is small, there is no need to use
    // parallelization or packings.
    if num_variables <= packing_width + 1 + log_num_threads {
        // A basic recursive approach.
        eval_eq_basic::<F, IF, EF, INITIALIZED>(evals, out, scalars);
    } else {
        let log_packing_width = log2_strict_usize(packing_width);
        let eval_len_min_packing = num_variables - log_packing_width;

        // We split eval into three parts:
        // - eval[..log_num_threads] (the first log_num_threads elements)
        // - eval[log_num_threads..eval_len_min_packing] (the middle elements)
        // - eval[eval_len_min_packing..] (the last log_packing_width elements)

        // The middle elements are the ones which will be computed in parallel.
        // The last log_packing_width elements are the ones which will be packed.

        // We make a buffer of PackedField elements of size `NUM_THREADS`.
        // Note that this is a slightly different strategy to `eval_eq` which instead
        // uses PackedExtensionField elements. Whilst this involves slightly more mathematical
        // operations, it seems to be faster in practice due to less data moving around.
        let mut parallel_buffer = RowMajorMatrix::new(
            E::PackedField::zero_vec(evals.width() * num_threads),
            evals.width(),
        );

        // As num_threads is a power of two we can divide using a bit-shift.
        let out_chunk_size = out.len() >> log_num_threads;

        // Compute the equality polynomial corresponding to the last log_packing_width elements
        // and pack these.
        let (front_rows, packed_rows) = evals.split_rows(eval_len_min_packing);
        let init_packings = E::init_packed(packed_rows, scalars);
        parallel_buffer.row_mut(0).copy_from_slice(&init_packings);

        let (buffer_rows, middle_rows) = front_rows.split_rows(log_num_threads);

        // Update the buffer so it contains the evaluations of the equality polynomial
        // with respect to parts one and three.
        fill_buffer(buffer_rows, parallel_buffer.as_view_mut());

        // Finally do all computations involving the middle elements.
        out.par_chunks_exact_mut(out_chunk_size)
            .zip(parallel_buffer.par_row_slices())
            .for_each(|(out_chunk, buffer_row)| {
                E::process_chunk::<INITIALIZED>(middle_rows, out_chunk, buffer_row, scalars);
            });
    }
}

/// Computes the equality polynomial evaluation via a recursive algorithm.
///
/// Unlike [`eval_eq_basic`], this function makes heavy use of packed values and parallelism to speed up computations.
///
/// In particular, it computes
/// ```text
/// eq(X) = eq_evals[j] * ∏ (1 - X_i + 2X_i z_i)
/// ```
///
/// Here `eq_evals[j]` should be thought of as evaluations of an equality polynomial over different variables
/// so `eq(X)` ends up being the evaluation of the equality polynomial over the combined set of variables.
///
/// # Behavior of `INITIALIZED`
/// If `INITIALIZED = false`, each value in `out` is overwritten with the computed result.
/// If `INITIALIZED = true`, the computed result is added to the existing value in `out`.
///
/// # Arguments
/// - `eval`: Evaluation point `z ∈ EF^n`
/// - `out`: Mutable slice of `EF` of size `2^n`
/// - `eq_evals`: Stores the current state of the equality polynomial evaluation in the recursive call.
/// - `scalar`: Scalar multiplier `α ∈ EF`. Depending on the `EqualityEvaluator` strategy, this may
///   be used to scale the result or may have already been applied to `eq_evals` and thus be ignored.
#[inline]
fn eval_eq_packed<F, IF, EF, E, const INITIALIZED: bool>(
    eval_points: RowMajorMatrixView<IF>,
    out: &mut [EF],
    eq_evals: &[E::PackedField],
    scalars: &[EF],
) where
    F: Field,
    IF: Field,
    EF: ExtensionField<F>,
    E: EqualityEvaluator<InputField = IF, OutputField = EF>,
{
    // Ensure that the output buffer size is correct:
    // It should be of size `2^n`, where `n` is the number of variables.
    let width = F::Packing::WIDTH;
    debug_assert_eq!(out.len(), width << eval_points.height());

    match eval_points.height() {
        0 => {
            // TODO
            E::accumulate_results::<INITIALIZED, 1>(out, [converted[0]]);
        }
        1 => {
            let converted = E::convert(eq_evals, scalars);
            let eq_evaluations = eval_eq_1(eval_points, &converted);
            E::accumulate_results::<INITIALIZED, 2>(out, eq_evaluations);
        }
        2 => {
            let eq_evaluations = eval_eq_2(eval_points, eq_evals);
            E::accumulate_results::<INITIALIZED, 4>(out, eq_evaluations);
        }
        3 => {
            let eq_evaluations = eval_eq_3(eval_points, eq_evals);
            E::accumulate_results::<INITIALIZED, 8>(out, eq_evaluations);
        }
        _ => {
            let (&x, tail) = eval_points.split_first().unwrap();

            // Divide the output buffer into two halves: one for `X_i = 0` and one for `X_i = 1`
            let (low, high) = out.split_at_mut(out.len() / 2);

            // Compute weight updates for the two branches following the recurrence:
            // ```
            // eq_{X1, ..., Xn}(X) = (1 - X_1) * eq_{X2, ..., Xn}(X) + X_1 * eq_{X2, ..., Xn}(X)
            // ```
            let s1 = eq_evals * x; // Contribution when `X_i = 1`
            let s0 = eq_evals - s1; // Contribution when `X_i = 0`

            eval_eq_packed::<F, IF, EF, E, INITIALIZED>(tail, low, s0, scalars);
            eval_eq_packed::<F, IF, EF, E, INITIALIZED>(tail, high, s1, scalars);
        }
    }
}

/// Computes the equality polynomial evaluations via a recursive algorithm.
///
/// Designed for use in cases where `eval().len()` is small and so
/// there is little to no advantage to packing or parallelism.
///
/// # Behavior of `INITIALIZED`
/// If `INITIALIZED = false`, each value in `out` is overwritten with the computed result.
/// If `INITIALIZED = true`, the computed result is added to the existing value in `out`.
///
/// # Arguments:
/// - `eval`: The point the equality function is being evaluated at.
/// - `out`: The output buffer to store or accumulate the results.
/// - `scalar`: Stores the current state of the equality polynomial evaluation in the recursive call.
#[inline]
fn eval_eq_basic<F, IF, EF, const INITIALIZED: bool>(
    evals: RowMajorMatrixView<IF>,
    out: &mut [EF],
    scalars: &[EF],
) where
    F: Field,
    IF: Field,
    EF: ExtensionField<F> + Algebra<IF>,
{
    // All invaraiants should have been checked by the caller.

    match evals.height() {
        0 => {
            if INITIALIZED {
                EF::add_slices(out, scalars);
            } else {
                out.copy_from_slice(scalars);
            }
        }
        1 => {
            // Manually unroll for single variable case
            let eq_evaluations = eval_eq_1(evals, scalars);
            add_or_set::<_, INITIALIZED>(out, &eq_evaluations);
        }
        2 => {
            // Manually unroll for two variable case
            let eq_evaluations = eval_eq_2(evals, scalars);
            add_or_set::<_, INITIALIZED>(out, &eq_evaluations);
        }
        3 => {
            // Manually unroll for three variable case
            let eq_evaluations = eval_eq_3(evals, scalars);
            add_or_set::<_, INITIALIZED>(out, &eq_evaluations);
        }
        _ => {
            let (&x, tail) = evals.split_first().unwrap();

            // Divide the output buffer into two halves: one for `X_i = 0` and one for `X_i = 1`
            let (low, high) = out.split_at_mut(out.len() / 2);

            // Compute weight updates for the two branches:
            // - `s0` corresponds to the case when `X_i = 0`
            // - `s1` corresponds to the case when `X_i = 1`
            //
            // Mathematically, this follows the recurrence:
            // ```text
            // eq_{X1, ..., Xn}(X) = (1 - X_1) * eq_{X2, ..., Xn}(X) + X_1 * eq_{X2, ..., Xn}(X)
            // ```
            let s1 = scalar * x; // Contribution when `X_i = 1`
            let s0 = scalar - s1; // Contribution when `X_i = 0`

            // The recursive approach turns out to be faster than the iterative one here.
            // Probably related to nice cache locality.
            eval_eq_basic::<_, _, _, INITIALIZED>(tail, low, s0);
            eval_eq_basic::<_, _, _, INITIALIZED>(tail, high, s1);
        }
    }
}

/// Computes a small equality polynomial evaluation and packs the result into a packed vector.
///
/// While this will always output a `PackedFieldExtension` element, if `F = EF`, that
/// element is also a `PackedField` element.
///
/// The length of `eval` must be equal to the `log2` of `F::Packing::WIDTH`.
#[inline(always)]
fn packed_eq_poly<F, EF>(evals: RowMajorMatrixView<EF>, scalars: &[EF]) -> Vec<EF::ExtensionPacking>
where
    F: Field,
    EF: ExtensionField<F>,
{
    // As this function is only available in this file, debug_assert should be fine here.
    // If this function becomes public, this should be changed to an assert.
    debug_assert_eq!(F::Packing::WIDTH, 1 << evals.height());

    // We build up the evaluations of the equality polynomial in buffer.
    let mut buffer =
        RowMajorMatrix::new(EF::zero_vec(evals.width() << evals.height()), evals.width());
    buffer.row_mut(0).copy_from_slice(scalars);

    fill_buffer(evals, buffer.as_view_mut());
    // Need to transpose the buffer.

    // Finally we need to "transpose" to get `PackedFieldExtension` element.
    EF::ExtensionPacking::from_ext_slice(&buffer)
}

/// Adds or sets the equality polynomial evaluations in the output buffer.
///
/// If the output buffer is already initialized, it adds the evaluations otherwise
/// it copies the evaluations into the buffer directly.
#[inline]
fn add_or_set<F: Field, const INITIALIZED: bool>(out: &mut [F], evaluations: &[F]) {
    debug_assert_eq!(out.len(), evaluations.len());
    if INITIALIZED {
        F::add_slices(out, evaluations);
    } else {
        out.copy_from_slice(evaluations);
    }
}

/// Scales the evaluations by scalar and either adds the result to the output buffer or
/// sets the output buffer directly depending on the `INITIALIZED` flag.
///
/// If the output buffer is already initialized, it adds the evaluations otherwise
/// it copies the evaluations into the buffer directly.
#[inline]
fn scale_and_add<F: Field, EF: ExtensionField<F>, const INITIALIZED: bool>(
    out: &mut [EF],
    base_vals: &[F],
    scalar: EF,
) {
    if INITIALIZED {
        out.iter_mut().zip(base_vals).for_each(|(out, &eq_eval)| {
            *out += scalar * eq_eval;
        });
    } else {
        out.iter_mut().zip(base_vals).for_each(|(out, &eq_eval)| {
            *out = scalar * eq_eval;
        });
    }
}

#[cfg(test)]
mod tests {
    use p3_baby_bear::BabyBear;
    use p3_field::extension::BinomialExtensionField;
    use p3_field::{PrimeCharacteristicRing, PrimeField64};
    use proptest::prelude::*;

    use super::*;

    type F = BabyBear;
    type EF4 = BinomialExtensionField<F, 4>;

    #[test]
    fn test_eval_eq_functionality() {
        let mut output = vec![F::ZERO; 4]; // n=2 → 2^2 = 4 elements
        let eval = vec![F::from_u64(1), F::from_u64(0)]; // (X1, X2) = (1,0)
        let scalar = F::from_u64(2);

        eval_eq::<_, _, true>(&eval, &mut output, scalar);

        // Expected results for (X1, X2) = (1,0)
        let expected_output = vec![F::ZERO, F::ZERO, F::from_u64(2), F::ZERO];

        assert_eq!(output, expected_output);
    }

    /// Compute the multilinear equality polynomial over the boolean hypercube.
    ///
    /// Given an evaluation point `z ∈ 𝔽ⁿ` and a scalar `α ∈ 𝔽`, this function returns the vector of
    /// evaluations of the equality polynomial `eq(x, z)` over all boolean inputs `x ∈ {0,1}ⁿ`,
    /// scaled by the scalar.
    ///
    /// The equality polynomial is defined as:
    ///
    /// \begin{equation}
    /// \mathrm{eq}(x, z) = \prod_{i=0}^{n-1} \left( x_i z_i + (1 - x_i)(1 - z_i) \right)
    /// \end{equation}
    ///
    /// This function evaluates:
    ///
    /// \begin{equation}
    /// α \cdot \mathrm{eq}(x, z)
    /// \end{equation}
    ///
    /// for all `x ∈ {0,1}ⁿ`, and returns a vector of size `2ⁿ` containing these values in lexicographic order.
    ///
    /// # Arguments
    /// - `eval`: The vector `z ∈ 𝔽ⁿ`, representing the evaluation point.
    /// - `scalar`: The scalar `α ∈ 𝔽` to scale the result by.
    ///
    /// # Returns
    /// A vector `v` of length `2ⁿ`, where `v[i] = α ⋅ eq(xᵢ, z)`, and `xᵢ` is the binary vector corresponding
    /// to the `i`-th index in lex order (i.e., big-endian bit decomposition of `i`).
    fn naive_eq(eval: &[EF4], scalar: EF4) -> Vec<EF4> {
        // Number of boolean variables `n` = length of evaluation point
        let n = eval.len();

        // Allocate result vector of size 2^n, initialized to zero
        let mut result = vec![EF4::ZERO; 1 << n];

        // Iterate over each binary input `x ∈ {0,1}ⁿ`, indexed by `i`
        for (i, out) in result.iter_mut().enumerate() {
            // Convert index `i` to a binary vector `x ∈ {0,1}ⁿ` in big-endian order
            let x: Vec<EF4> = (0..n)
                .map(|j| {
                    let bit = (i >> (n - 1 - j)) & 1;
                    if bit == 1 { EF4::ONE } else { EF4::ZERO }
                })
                .collect();

            // Compute the equality polynomial:
            // eq(x, z) = ∏_{i=0}^{n-1} (xᵢ ⋅ zᵢ + (1 - xᵢ)(1 - zᵢ))
            let eq = x
                .iter()
                .zip(eval.iter())
                .map(|(xi, zi)| {
                    // Each term: xᵢ zᵢ + (1 - xᵢ)(1 - zᵢ)
                    *xi * *zi + (EF4::ONE - *xi) * (EF4::ONE - *zi)
                })
                .product::<EF4>(); // Take product over all coordinates

            // Store the scaled result: α ⋅ eq(x, z)
            *out = scalar * eq;
        }

        result
    }

    proptest! {
        #[test]
        fn prop_eval_eq_matches_eval_eq_base_when_lifted(
            // Number of boolean variables
            n in 1usize..5,

            // Random input values from the base field, capped at length 4
            eval_values in prop::collection::vec(0u64..F::ORDER_U64, 1..=4),

            // A random scalar value from the base field
            scalar_u64 in 0u64..F::ORDER_U64,
        ) {
            // Construct evaluation point in base field
            let mut eval_point_base: Vec<_> = eval_values.into_iter()
                .map(F::from_u64)
                .collect();

            // Resize to exactly `n` elements:
            // - if too short: pad with F::ZERO
            // - if too long: truncate
            //
            // This ensures that the input has exactly `n` variables
            eval_point_base.resize(n, F::ZERO);

            // Convert base field evaluation point to extension field
            let eval_point_ext: Vec<EF4> = eval_point_base.iter()
                .map(|&f| EF4::from(f))
                .collect();

            // Setup the scalar
            let scalar_ext = EF4::from_u64(scalar_u64);

            // Prepare output buffers
            //
            // For a multilinear polynomial over `n` variables, there are 2^n evaluations
            let num_outputs = 1 << n;

            // Output from eval_eq using extension field evaluation point
            let mut output_ext = vec![EF4::ZERO; num_outputs];

            // Output from eval_eq_base using base field evaluation point
            let mut output_base = vec![EF4::ZERO; num_outputs];

            // Evaluate the equality polynomial using both methods

            // Evaluate using `eval_eq`
            eval_eq::<F, EF4, false>(&eval_point_ext, &mut output_ext, scalar_ext);

            // Evaluate using `eval_eq_base`
            eval_eq_base::<F, EF4, false>(&eval_point_base, &mut output_base, scalar_ext);

            // Assert both outputs match
            prop_assert_eq!(output_ext, output_base);
        }
    }
}
