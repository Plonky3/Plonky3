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
//! In addition to single-point evaluation, this module includes **batched** variants that compute
//! a linear combination of equality tables in one pass:
//!
//! ```text
//! W(x) = \sum_i \gamma_i ⋅ eq(x, z_i)  ,  x ∈ {0,1}^n .
//! ```
//!
//! ## Batched Evaluation
//!
//! The batched methods (`eval_eq_batch`, `eval_eq_base_batch`) are designed to efficiently compute
//! linear combinations of multiple equality polynomial evaluations. Instead of computing each
//! equality polynomial individually and then summing the results, these functions leverage linearity
//! to perform the summation within the recursive evaluation process.
//!
//! ### Key Performance Benefits:
//! - **Reduced complexity**: From O(m⋅2^n) to O(2^n + m⋅n) for m evaluation points
//! - **SIMD optimization**: Uses vectorized operations via the new `sub_slices` method
//! - **Memory efficiency**: Single buffer allocation with batched processing
//! - **Parallel processing**: Full utilization of multi-core systems
//!
//! ### Mathematical Foundation:
//! The batched algorithm exploits the recursive structure by updating entire vectors of scalars:
//!
//! At each variable z_j, the scalar vector γ = (γ_0, γ_1, ..., γ_{m-1}) splits into:
//! - γ_0 = γ ⊙ (1 - z_j) for the x_j = 0 branch
//! - γ_1 = γ ⊙ z_j for the x_j = 1 branch
//!
//! Where ⊙ denotes element-wise (Hadamard) product.
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
    dot_product,
};
use p3_matrix::Matrix;
use p3_matrix::dense::{RowMajorMatrix, RowMajorMatrixView};
use p3_maybe_rayon::prelude::*;
use p3_util::{iter_array_chunks_padded, log2_strict_usize};

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
/// # Behavior of `INITIALIZED`
/// If `INITIALIZED = false`, each value in `out` is overwritten with the computed result.
/// If `INITIALIZED = true`, the computed result is added to the existing value in `out`.
///
/// # Arguments
/// - `eval`: Evaluation point `z ∈ EF^n`
/// - `out`: Mutable slice of `EF` of size `2^n`
/// - `scalar`: Scalar multiplier `α ∈ EF`
#[inline]
pub fn eval_eq<F, EF, const INITIALIZED: bool>(eval: &[EF], out: &mut [EF], scalar: EF)
where
    F: Field,
    EF: ExtensionField<F>,
{
    // Pass the combined method using the `ExtFieldEvaluator` strategy.
    eval_eq_common::<F, EF, EF, ExtFieldEvaluator<F, EF>, INITIALIZED>(eval, out, scalar);
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
/// # Behavior of `INITIALIZED`
/// If `INITIALIZED = false`, each value in `out` is overwritten with the computed result.
/// If `INITIALIZED = true`, the computed result is added to the existing value in `out`.
///
/// # Arguments
/// - `eval`: Evaluation point `z ∈ F^n`
/// - `out`: Mutable slice of `EF` of size `2^n`
/// - `scalar`: Scalar multiplier `α ∈ EF`
#[inline]
pub fn eval_eq_base<F, EF, const INITIALIZED: bool>(eval: &[F], out: &mut [EF], scalar: EF)
where
    F: Field,
    EF: ExtensionField<F>,
{
    // Pass the combined method using the `BaseFieldEvaluator` strategy.
    eval_eq_common::<F, F, EF, BaseFieldEvaluator<F, EF>, INITIALIZED>(eval, out, scalar);
}

/// Computes the batched multilinear equality polynomial `\sum_i \gamma_i ⋅ eq(x, z_i)` over all
/// `x ∈ \{0,1\}^n` for multiple points `z_i ∈ EF^n` with weights `\gamma_i ∈ EF`.
///
/// This evaluates multiple equality tables simultaneously by pushing the linear combination
/// through the recursion.
///
/// # Mathematical statement
/// Given:
/// - evaluation points `z_0, z_1, ..., z_{m-1} ∈ F^n`,
/// - weights `\gamma_0, \gamma_1, ..., \gamma_{m-1} ∈ EF`,
/// this computes, for all `x ∈ {0,1}^n`,
/// ```text
/// W(x) = \sum_i \gamma_i ⋅ eq(x, z_i).
/// ```
///
/// # Arguments
/// - `evals`: Matrix where each column is one point `z_i`.
///     - height = number of variables `n`,
///     - width = number of points `m`
/// - `scalars`: Weights `[ \gamma_0, \gamma_1, ..., \gamma_{m-1} ]`
/// - `out`: Output buffer of size `2^n` storing `W(x)` in big-endian `x` order
///
/// # Panics
/// Panics in debug builds if `evals.width() != scalars.len()` or if the output buffer size is incorrect.
#[inline]
pub fn eval_eq_batch<F, EF, const INITIALIZED: bool>(
    evals: RowMajorMatrixView<EF>,
    scalars: &[EF],
    out: &mut [EF],
) where
    F: Field,
    EF: ExtensionField<F>,
{
    eval_eq_batch_common::<F, EF, EF, ExtFieldEvaluator<F, EF>, INITIALIZED>(evals, scalars, out);
}

/// Computes the batched multilinear equality polynomial `\sum_i \gamma_i ⋅ eq(x, z_i)` over all
/// `x ∈ \{0,1\}^n` for multiple points `z_i ∈ F^n` with weights `\gamma_i ∈ EF`.
///
/// This evaluates multiple equality tables simultaneously by pushing the linear combination
/// through the recursion.
///
/// # Mathematical statement
/// Given:
/// - evaluation points `z_0, z_1, ..., z_{m-1} ∈ EF^n`,
/// - weights `\gamma_0, \gamma_1, ..., \gamma_{m-1} ∈ EF`,
/// this computes, for all `x ∈ {0,1}^n`,
/// ```text
/// W(x) = \sum_i \gamma_i ⋅ eq(x, z_i).
/// ```
///
/// # Arguments
/// - `evals`: Matrix where each column is one point `z_i`.
///     - height = number of variables `n`,
///     - width = number of points `m`
/// - `scalars`: Weights `[ \gamma_0, \gamma_1, ..., \gamma_{m-1} ]`
/// - `out`: Output buffer of size `2^n` storing `W(x)` in big-endian `x` order
///
/// # Panics
/// Panics in debug builds if `evals.width() != scalars.len()` or if the output buffer size is incorrect.
#[inline]
pub fn eval_eq_base_batch<F, EF, const INITIALIZED: bool>(
    evals: RowMajorMatrixView<F>,
    scalars: &[EF],
    out: &mut [EF],
) where
    F: Field,
    EF: ExtensionField<F>,
{
    eval_eq_batch_common::<F, F, EF, BaseFieldEvaluator<F, EF>, INITIALIZED>(evals, scalars, out);
}

/// Fills the `buffer` with evaluations of the equality polynomial
/// of degree `points.len()` multiplied by the value at `buffer[0]`.
///
/// Assume that `buffer[0]` contains `{eq(i, x)}` for `i \in \{0, 1\}^j` packed into a single
/// PackedExtensionField element. This function fills out the remainder of the buffer so that
/// `buffer[ind]` contains `{eq(ind, points) * eq(i, x)}` for `i \in \{0, 1\}^j`. Note that
/// `ind` is interpreted as an element of `\{0, 1\}^{points.len()}`.
#[inline(always)]
fn fill_buffer<'a, F, A>(points: impl ExactSizeIterator<Item = &'a F>, buffer: &mut [A])
where
    F: Field,
    A: Algebra<F>,
{
    for (ind, &entry) in points.enumerate() {
        let stride = 1 << ind;

        for index in 0..stride {
            let val = buffer[index].clone();
            let scaled_val = val.clone() * entry;
            let new_val = val - scaled_val.clone();

            buffer[index] = new_val;
            buffer[index + stride] = scaled_val;
        }
    }
}

/// Fills the `buffer` with evaluations of the equality polynomial for multiple points simultaneously.
///
/// This is the batched version of `fill_buffer` that operates on matrices where each column
/// represents a different evaluation point. The function expands a matrix of partial equality
/// polynomial evaluations across multiple variables.
///
/// Given a buffer with `2^k` rows (where each column holds partial products for a specific point
/// after `k` variables have been processed), this function processes the evaluation points
/// for the remaining variables to complete the equality polynomial computation.
///
/// # Arguments
/// - `evals`: Matrix where each column is an evaluation point z_i, each row is a variable
/// - `buffer`: Mutable matrix buffer to be filled with equality polynomial evaluations
///
/// # Panics
/// Panics in debug builds if `evals.width() != buffer.width()`.
#[inline(always)]
fn fill_buffer_batch<F, A>(evals: RowMajorMatrixView<F>, buffer: &mut RowMajorMatrix<A>)
where
    F: Field,
    A: Algebra<F> + Send + Sync + Clone,
{
    // Process variables in reverse order to maintain correct bit ordering in output buffer.
    // This follows the same recursive update rule as the single-point fill_buffer,
    // but applies it simultaneously across all columns (evaluation points).
    for (ind, eval_row) in evals.row_slices().rev().enumerate() {
        let stride = 1 << ind;
        let width = buffer.width();

        // Expand the buffer in-place by doubling its height at each step.
        // Each existing row generates two new rows: one for x_j = 0, one for x_j = 1.
        for idx in 0..stride {
            // Read current row values before modifying to avoid data races
            let current_row_values: Vec<A> = (0..width)
                .map(|col| buffer.values[idx * width + col].clone())
                .collect();

            // Apply the recursive equality polynomial update rule to each column:
            // new_row_for_x_j=0 = old_row * (1 - z_j)
            // new_row_for_x_j=1 = old_row * z_j
            for col in 0..width {
                let val = current_row_values[col].clone();
                let eval_point = eval_row[col];
                let scaled_val = val.clone() * eval_point;
                let new_val = val - scaled_val.clone();

                buffer.values[idx * width + col] = new_val;
                buffer.values[(idx + stride) * width + col] = scaled_val;
            }
        }
    }
}

/// Compute the scaled multilinear equality polynomial over `{0,1}`.
///
/// # Arguments
/// - `eval`: Slice containing the evaluation point `[z_0]` (must have length 1).
/// - `scalar`: A field element `α ∈ 𝔽` used to scale the result.
///
/// # Returns
/// An array of scaled evaluations `[α ⋅ eq(0, z), α ⋅ eq(1, z)] = [α ⋅ (1 - z_0), α ⋅ z_0]`.
#[inline(always)]
fn eval_eq_1<F, FP>(eval: &[F], scalar: FP) -> [FP; 2]
where
    F: Field,
    FP: Algebra<F>,
{
    assert_eq!(eval.len(), 1);

    // Extract the evaluation point z_0
    let z_0 = eval[0];

    // Compute α ⋅ z_0 = α ⋅ eq(1, z) and α ⋅ (1 - z_0) = α - α ⋅ z_0 = α ⋅ eq(0, z)
    let eq_1 = scalar.clone() * z_0;
    let eq_0 = scalar - eq_1.clone();

    [eq_0, eq_1]
}

/// Computes the batched scaled multilinear equality polynomial over `{0,1}` for multiple points.
///
/// This is the batched version of `eval_eq_1` that efficiently computes the final summation
/// for a single variable across multiple evaluation points simultaneously.
///
/// The equality polynomial for one variable is:
/// ```text
/// eq(x, z) = x * z + (1 - x) * (1 - z)
/// ```
///
/// For the batched case, we compute:
/// ```text
/// eq_sum(0) = ∑_i scalars[i] * (1 - evals[0][i])  // when x = 0
/// eq_sum(1) = ∑_i scalars[i] * evals[0][i]        // when x = 1
/// ```
///
/// # Arguments
/// - `evals`: Matrix where each column is an evaluation point z_i (must have height = 1)
/// - `scalars`: Vector of scalars [γ_0, γ_1, ..., γ_{m-1}] for weighting each evaluation
///
/// # Returns
/// An array `[eq_sum(0), eq_sum(1)]` containing the summed evaluations for x = 0 and x = 1.
///
/// # Panics
/// Panics in debug builds if `evals.height() != 1` or `evals.width() != scalars.len()`.
#[inline(always)]
fn eval_eq_1_batch<F, FP>(evals: RowMajorMatrixView<F>, scalars: &[FP]) -> [FP; 2]
where
    F: Field,
    FP: Algebra<F> + Copy,
{
    debug_assert_eq!(evals.height(), 1);
    debug_assert_eq!(evals.width(), scalars.len());

    // Use linearity to avoid redundant operations.
    //
    // Instead of computing each term individually and summing, we leverage
    // the mathematical relationship between eq(0,z) and eq(1,z).

    // Compute the total sum of all scalars: ∑_i γ_i
    let sum: FP = scalars.iter().copied().sum();

    // Compute ∑_i γ_i * z_{i,0} using a dot product
    // This gives us eq_sum(1) directly since eq(1, z) = z
    let eq_1_sum: FP = dot_product(scalars.iter().copied(), evals.values.iter().copied());

    // Use the identity: eq(0, z_i) = 1 - z_i
    // So ∑_i γ_i * (1 - z_i) = ∑_i γ_i - ∑_i γ_i * z_i
    // This saves approximately m operations compared to computing each term individually
    let eq_0_sum = sum - eq_1_sum;

    [eq_0_sum, eq_1_sum]
}

/// Computes the batched scaled multilinear equality polynomial over `{0,1}` using packed values.
///
/// This is the packed version of `eval_eq_1_batch`, designed for use within the parallel
/// evaluation framework where scalars are already packed into SIMD-friendly formats.
///
/// The function computes the same mathematical result as `eval_eq_1_batch` but operates
/// on packed scalar values for improved SIMD performance.
///
/// # Arguments
/// - `evals`: Matrix where each column is an evaluation point z_i (must have height = 1)
/// - `packed_scalars`: Vector of packed scalars for SIMD processing
///
/// # Returns
/// An array `[eq_sum(0), eq_sum(1)]` containing the summed evaluations for x = 0 and x = 1.
#[inline(always)]
fn eval_eq_1_batch_packed<F, FP>(evals: RowMajorMatrixView<F>, packed_scalars: &[FP]) -> [FP; 2]
where
    F: Field,
    FP: Algebra<F> + Copy,
{
    debug_assert_eq!(evals.height(), 1);
    debug_assert_eq!(evals.width(), packed_scalars.len());

    // Compute ∑ᵢ γᵢ
    let sum: FP = packed_scalars.iter().copied().sum();

    // Compute ∑ᵢ γᵢ ⋅ zᵢ using dot product
    let eq_1_sum: FP = dot_product(packed_scalars.iter().copied(), evals.values.iter().copied());

    // eq(0, zᵢ) = 1 - zᵢ, so ∑ᵢ γᵢ ⋅ (1 - zᵢ) = ∑ᵢ γᵢ - ∑ᵢ γᵢ ⋅ zᵢ
    let eq_0_sum = sum - eq_1_sum;

    [eq_0_sum, eq_1_sum]
}

/// Compute the scaled multilinear equality polynomial over `{0,1}²`.
///
/// # Arguments
/// - `eval`: Slice containing the evaluation point `[z_0, z_1]` (must have length 2).
/// - `scalar`: A field element `α ∈ 𝔽` used to scale the result.
///
/// # Returns
/// An array containing `α ⋅ eq(x, z)` for `x ∈ {0,1}²` arranged using lexicographic order of `x`.
#[inline(always)]
fn eval_eq_2<F, FP>(eval: &[F], scalar: FP) -> [FP; 4]
where
    F: Field,
    FP: Algebra<F>,
{
    assert_eq!(eval.len(), 2);

    // Extract z_0 from the evaluation point
    let z_0 = eval[0];

    // Compute eq_1 = α ⋅ z_0 = α ⋅ eq(1, -) and eq_0 = α - s1 = α ⋅ (1 - z_0) = α ⋅ eq(0, -)
    let eq_1 = scalar.clone() * z_0;
    let eq_0 = scalar - eq_1.clone();

    // Recurse to calculate evaluations for the remaining variable
    let [eq_00, eq_01] = eval_eq_1(&eval[1..], eq_0);
    let [eq_10, eq_11] = eval_eq_1(&eval[1..], eq_1);

    // Return values in lexicographic order of x = (x_0, x_1)
    [eq_00, eq_01, eq_10, eq_11]
}

/// Computes the batched scaled multilinear equality polynomial over `{0,1}^2` for multiple points.
///
/// This is the batched version of `eval_eq_2` that efficiently handles the two-variable case
/// across multiple evaluation points simultaneously. It serves as an unrolled base case
/// for the recursive batch algorithm.
///
/// The equality polynomial for two variables is:
/// ```text
/// eq(x, z) = (x_0 * z_0 + (1 - x_0) * (1 - z_0)) * (x_1 * z_1 + (1 - x_1) * (1 - z_1))
/// ```
///
/// For the batched case with evaluation points z_i = (z_{i,0}, z_{i,1}), we compute:
/// ```text
/// result[j] = ∑_i scalars[i] * eq(x_j, z_i)  for x_j ∈ {(0,0), (0,1), (1,0), (1,1)}
/// ```
///
/// # Arguments
/// - `evals`: Matrix where each column is an evaluation point z_i ∈ F^2 (must have height = 2)
/// - `scalars`: Vector of scalars [γ_0, γ_1, ..., γ_{m-1}] for weighting each evaluation
///
/// # Returns
/// An array of length 4 containing `∑_i scalars[i] * eq(x, z_i)` for all x ∈ {0,1}^2
/// in lexicographic order: [eq(00), eq(01), eq(10), eq(11)].
///
/// # Panics
/// Panics in debug builds if `evals.height() != 2` or `evals.width() != scalars.len()`.
#[inline(always)]
fn eval_eq_2_batch<F, FP>(evals: RowMajorMatrixView<F>, scalars: &[FP]) -> [FP; 4]
where
    F: Field,
    FP: Algebra<F> + Copy + Field,
{
    debug_assert_eq!(evals.height(), 2);
    debug_assert_eq!(evals.width(), scalars.len());

    let (first_row, second_row) = evals.split_rows(1);

    // Split on the first variable z_0 using vectorized operations for efficiency.
    // This leverages the recursive property: eq((x_0, x_1), (z_0, z_1)) splits into
    // two sub-problems based on x_0, each with updated scalar weights.

    // Compute eq_1s[i] = scalars[i] * z_{i,0} for all points i
    // These are the scalar weights for the x_0 = 1 branch
    let eq_1s: Vec<_> = first_row
        .values
        .iter()
        .zip(scalars)
        .map(|(&z_0, &scalar)| scalar * z_0)
        .collect();

    // Compute eq_0s[i] = scalars[i] - eq_1s[i] using vectorized SIMD subtraction
    // These are the scalar weights for the x_0 = 0 branch
    // This approach leverages the new sub_slices method for efficient vectorized operations
    let mut eq_0s: Vec<_> = scalars.to_vec();
    FP::sub_slices(&mut eq_0s, &eq_1s);

    // Recurse to calculate evaluations for the remaining variable
    let [eq_00, eq_01] = eval_eq_1_batch(second_row, &eq_0s);
    let [eq_10, eq_11] = eval_eq_1_batch(second_row, &eq_1s);

    // Return values in lexicographic order of x = (x_0, x_1)
    [eq_00, eq_01, eq_10, eq_11]
}

/// Computes the batched scaled multilinear equality polynomial over `{0,1}^2` using packed values.
///
/// Designed for use within the parallel evaluation framework where scalars are processed
/// in packed SIMD format for improved performance.
///
/// Unlike the regular batch version, this function works with packed scalars but uses
/// element-wise operations rather than vectorized `sub_slices` since packed types
/// don't necessarily implement the `Field` trait required for vectorized operations.
///
/// # Arguments
/// - `evals`: Matrix where each column is an evaluation point z_i ∈ F^2 (must have height = 2)
/// - `packed_scalars`: Vector of packed scalars for SIMD processing
///
/// # Returns
/// An array of length 4 containing the summed evaluations in lexicographic order.
#[inline(always)]
fn eval_eq_2_batch_packed<F, FP>(evals: RowMajorMatrixView<F>, packed_scalars: &[FP]) -> [FP; 4]
where
    F: Field,
    FP: Algebra<F> + Copy,
{
    debug_assert_eq!(evals.height(), 2);
    debug_assert_eq!(evals.width(), packed_scalars.len());

    let (first_row, second_row) = evals.split_rows(1);

    // Split on the first variable z₀
    let (eq_0s, eq_1s): (Vec<_>, Vec<_>) = first_row
        .values
        .iter()
        .zip(packed_scalars)
        .map(|(&z_0, &scalar)| {
            let eq_1 = scalar * z_0;
            let eq_0 = scalar - eq_1;
            (eq_0, eq_1)
        })
        .unzip();

    // Recurse to calculate evaluations for the remaining variable
    let [eq_00, eq_01] = eval_eq_1_batch_packed(second_row, &eq_0s);
    let [eq_10, eq_11] = eval_eq_1_batch_packed(second_row, &eq_1s);

    // Return values in lexicographic order of x = (x_0, x_1)
    [eq_00, eq_01, eq_10, eq_11]
}

/// Compute the scaled multilinear equality polynomial over `{0,1}³`.
///
/// # Arguments
/// - `eval`: Slice containing the evaluation point `[z_0, z_1, z_2]` (must have length 3).
/// - `scalar`: A field element `α ∈ 𝔽` used to scale the result.
///
/// # Returns
/// An array containing `α ⋅ eq(x, z)` for `x ∈ {0,1}³` arranged using lexicographic order of `x`.
#[inline(always)]
fn eval_eq_3<F, FP>(eval: &[F], scalar: FP) -> [FP; 8]
where
    F: Field,
    FP: Algebra<F>,
{
    assert_eq!(eval.len(), 3);

    // Extract z_0 from the evaluation point
    let z_0 = eval[0];

    // Compute eq_1 = α ⋅ z_0 = α ⋅ eq(1, -) and eq_0 = α - s1 = α ⋅ (1 - z_0) = α ⋅ eq(0, -)
    let eq_1 = scalar.clone() * z_0;
    let eq_0 = scalar - eq_1.clone();

    // Recurse to calculate evaluations for the remaining variables
    let [eq_000, eq_001, eq_010, eq_011] = eval_eq_2(&eval[1..], eq_0);
    let [eq_100, eq_101, eq_110, eq_111] = eval_eq_2(&eval[1..], eq_1);

    // Return all 8 evaluations in lexicographic order of x ∈ {0,1}³
    [
        eq_000, eq_001, eq_010, eq_011, eq_100, eq_101, eq_110, eq_111,
    ]
}

/// Computes the batched scaled multilinear equality polynomial over `{0,1}^3` for multiple points.
///
/// This is the batched version of `eval_eq_3` that efficiently handles the three-variable case
/// across multiple evaluation points simultaneously. It serves as an unrolled base case
/// for the recursive batch algorithm.
///
/// The equality polynomial for three variables is:
/// ```text
/// eq(x, z) = ∏_{i=0}^{2} (x_i * z_i + (1 - x_i) * (1 - z_i))
/// ```
///
/// For the batched case with evaluation points z_i = (z_{i,0}, z_{i,1}, z_{i,2}), we compute:
/// ```text
/// result[j] = ∑_i scalars[i] * eq(x_j, z_i)  for all x_j ∈ {0,1}^3
/// ```
///
/// # Arguments
/// - `evals`: Matrix where each column is an evaluation point z_i ∈ F^3 (must have height = 3)
/// - `scalars`: Vector of scalars [γ_0, γ_1, ..., γ_{m-1}] for weighting each evaluation
///
/// # Returns
/// An array of length 8 containing `∑_i scalars[i] * eq(x, z_i)` for all x ∈ {0,1}^3
/// in lexicographic order: [eq(000), eq(001), eq(010), eq(011), eq(100), eq(101), eq(110), eq(111)].
///
/// # Panics
/// Panics in debug builds if `evals.height() != 3` or `evals.width() != scalars.len()`.
#[inline(always)]
fn eval_eq_3_batch<F, FP>(evals: RowMajorMatrixView<F>, scalars: &[FP]) -> [FP; 8]
where
    F: Field,
    FP: Algebra<F> + Copy + Field,
{
    debug_assert_eq!(evals.height(), 3);
    debug_assert_eq!(evals.width(), scalars.len());

    let (first_row, remainder) = evals.split_rows(1);

    // Split on the first variable z_0, following the same vectorized strategy as eval_eq_2_batch.
    //
    // The three-variable case reduces to two two-variable sub-problems.

    // Compute eq_1s[i] = scalars[i] * z_{i,0} for all points i.
    //
    // These become the scalar weights for the x_0 = 1 branch.
    let eq_1s: Vec<_> = first_row
        .values
        .iter()
        .zip(scalars)
        .map(|(&z_0, &scalar)| scalar * z_0)
        .collect();

    // Compute eq_0s[i] = scalars[i] - eq_1s[i] using vectorized subtraction.
    //
    // These become the scalar weights for the x_0 = 0 branch.
    let mut eq_0s: Vec<_> = scalars.to_vec();
    FP::sub_slices(&mut eq_0s, &eq_1s);

    // Recurse to calculate evaluations for the remaining variables
    let [eq_000, eq_001, eq_010, eq_011] = eval_eq_2_batch(remainder, &eq_0s);
    let [eq_100, eq_101, eq_110, eq_111] = eval_eq_2_batch(remainder, &eq_1s);

    // Return all 8 evaluations in lexicographic order of x ∈ {0,1}³
    [
        eq_000, eq_001, eq_010, eq_011, eq_100, eq_101, eq_110, eq_111,
    ]
}

/// Computes the batched scaled multilinear equality polynomial over `{0,1}^3` using packed values.
///
/// This is the packed version of `eval_eq_3_batch`, designed for use within the parallel
/// evaluation framework where scalars are processed in packed SIMD format.
///
/// Like `eval_eq_2_batch_packed`, this function uses element-wise operations rather than
/// vectorized operations since packed types don't necessarily implement the Field trait.
///
/// # Arguments
/// - `evals`: Matrix where each column is an evaluation point z_i ∈ F^3 (must have height = 3)
/// - `packed_scalars`: Vector of packed scalars for SIMD processing
///
/// # Returns
/// An array of length 8 containing the summed evaluations in lexicographic order.
#[inline(always)]
fn eval_eq_3_batch_packed<F, FP>(evals: RowMajorMatrixView<F>, packed_scalars: &[FP]) -> [FP; 8]
where
    F: Field,
    FP: Algebra<F> + Copy,
{
    debug_assert_eq!(evals.height(), 3);
    debug_assert_eq!(evals.width(), packed_scalars.len());

    let (first_row, remainder) = evals.split_rows(1);

    // Split on the first variable z₀
    let (eq_0s, eq_1s): (Vec<_>, Vec<_>) = first_row
        .values
        .iter()
        .zip(packed_scalars)
        .map(|(&z_0, &scalar)| {
            let eq_1 = scalar * z_0;
            let eq_0 = scalar - eq_1;
            (eq_0, eq_1)
        })
        .unzip();

    // Recurse to calculate evaluations for the remaining variables
    let [eq_000, eq_001, eq_010, eq_011] = eval_eq_2_batch_packed(remainder, &eq_0s);
    let [eq_100, eq_101, eq_110, eq_111] = eval_eq_2_batch_packed(remainder, &eq_1s);

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

    fn init_packed(eval: &[Self::InputField], init_value: Self::OutputField) -> Self::PackedField;

    fn process_chunk<const INITIALIZED: bool>(
        eval: &[Self::InputField],
        out_chunk: &mut [Self::OutputField],
        buffer_val: Self::PackedField,
        scalar: Self::OutputField,
    );

    fn accumulate_results<const INITIALIZED: bool, const N: usize>(
        out: &mut [Self::OutputField],
        eq_evals: [Self::PackedField; N],
        scalar: Self::OutputField,
    );

    fn init_packed_batch(
        evals: RowMajorMatrixView<Self::InputField>,
        scalars: &[Self::OutputField],
    ) -> Vec<Self::PackedField>;

    fn process_chunk_batch<const INITIALIZED: bool>(
        evals: RowMajorMatrixView<Self::InputField>,
        out_chunk: &mut [Self::OutputField],
        buffer_vals: &[Self::PackedField],
        scalars: &[Self::OutputField],
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

    fn init_packed(eval: &[Self::InputField], init_value: Self::OutputField) -> Self::PackedField {
        packed_eq_poly(eval, init_value)
    }

    fn process_chunk<const INITIALIZED: bool>(
        eval: &[Self::InputField],
        out_chunk: &mut [Self::OutputField],
        buffer_val: Self::PackedField,
        scalar: Self::OutputField,
    ) {
        eval_eq_packed::<F, EF, EF, Self, INITIALIZED>(eval, out_chunk, buffer_val, scalar);
    }

    fn accumulate_results<const INITIALIZED: bool, const N: usize>(
        out: &mut [Self::OutputField],
        eq_evals: [Self::PackedField; N],
        _scalar: Self::OutputField,
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

    fn init_packed_batch(
        evals: RowMajorMatrixView<Self::InputField>,
        scalars: &[Self::OutputField],
    ) -> Vec<Self::PackedField> {
        packed_eq_poly_batch(evals, scalars)
    }

    fn process_chunk_batch<const INITIALIZED: bool>(
        evals: RowMajorMatrixView<Self::InputField>,
        out_chunk: &mut [Self::OutputField],
        buffer_vals: &[Self::PackedField],
        scalars: &[Self::OutputField],
    ) {
        eval_eq_packed_batch::<F, EF, EF, Self, INITIALIZED>(
            evals,
            out_chunk,
            buffer_vals,
            scalars,
        );
    }
}

impl<F: Field, EF: ExtensionField<F>> EqualityEvaluator for BaseFieldEvaluator<F, EF> {
    type InputField = F;
    type OutputField = EF;
    type PackedField = F::Packing;

    fn init_packed(eval: &[Self::InputField], _init_value: Self::OutputField) -> Self::PackedField {
        packed_eq_poly(eval, F::ONE)
    }

    fn process_chunk<const INITIALIZED: bool>(
        eval: &[Self::InputField],
        out_chunk: &mut [Self::OutputField],
        buffer_val: Self::PackedField,
        scalar: Self::OutputField,
    ) {
        eval_eq_packed::<F, F, EF, Self, INITIALIZED>(eval, out_chunk, buffer_val, scalar);
    }

    fn accumulate_results<const INITIALIZED: bool, const N: usize>(
        out: &mut [Self::OutputField],
        eq_evals: [Self::PackedField; N],
        scalar: Self::OutputField,
    ) {
        let eq_evals_unpacked = F::Packing::unpack_slice(&eq_evals);
        scale_and_add::<_, _, INITIALIZED>(out, eq_evals_unpacked, scalar);
    }

    fn init_packed_batch(
        evals: RowMajorMatrixView<Self::InputField>,
        _scalars: &[Self::OutputField],
    ) -> Vec<Self::PackedField> {
        let const_scalars = vec![F::ONE; evals.width()];
        packed_eq_poly_batch(evals, &const_scalars)
    }

    fn process_chunk_batch<const INITIALIZED: bool>(
        evals: RowMajorMatrixView<Self::InputField>,
        out_chunk: &mut [Self::OutputField],
        buffer_vals: &[Self::PackedField],
        scalars: &[Self::OutputField],
    ) {
        eval_eq_packed_batch::<F, F, EF, Self, INITIALIZED>(evals, out_chunk, buffer_vals, scalars);
    }
}

/// Computes the batched multilinear equality polynomial `∑ᵢ γᵢ ⋅ eq(x, zᵢ)` over all `x ∈ \{0,1\}^n`
/// for multiple points `zᵢ ∈ IF^n` and corresponding scalars `γᵢ ∈ EF`.
///
/// This is the core batched evaluation function that leverages the linearity of summation
/// to efficiently compute multiple equality polynomial evaluations simultaneously.
///
/// # Performance Benefits
/// Instead of computing each equality polynomial individually and summing the results, this approach performs
/// the summation *within* the recursive evaluation.
///
/// # Arguments
/// - `evals`: Matrix where each column represents one evaluation point zᵢ.
/// - `scalars`: Vector of scalars [γ₀, γ₁, ..., γ_{m-1}] corresponding to each evaluation point.
/// - `out`: Output buffer of size `2^n` to store the combined evaluations.
#[inline]
fn eval_eq_batch_common<F, IF, EF, E, const INITIALIZED: bool>(
    evals: RowMajorMatrixView<IF>,
    scalars: &[EF],
    out: &mut [EF],
) where
    F: Field,
    IF: Field,
    EF: ExtensionField<F> + ExtensionField<IF>,
    E: EqualityEvaluator<InputField = IF, OutputField = EF>,
{
    // Handle empty batch case
    if evals.width() == 0 {
        debug_assert!(scalars.is_empty());
        return;
    }

    // Validate input dimensions
    let num_vars = evals.height();
    debug_assert_eq!(evals.width(), scalars.len());
    debug_assert_eq!(out.len(), 1 << num_vars);

    // For small problems, use the basic recursive approach
    let packing_width = F::Packing::WIDTH;
    let num_threads = current_num_threads().next_power_of_two();
    let log_num_threads = log2_strict_usize(num_threads);

    if num_vars <= packing_width.ilog2() as usize + 1 + log_num_threads {
        eval_eq_batch_basic::<F, IF, EF, INITIALIZED>(evals, scalars, out);
    } else {
        let log_packing_width = log2_strict_usize(packing_width);
        let eval_len_min_packing = num_vars - log_packing_width;

        // Split the variables into three parts (same strategy as eval_eq_common):
        // - evals[..log_num_threads] (the first log_num_threads variables)
        // - evals[log_num_threads..eval_len_min_packing] (the middle variables)
        // - evals[eval_len_min_packing..] (the last log_packing_width variables)

        // The middle variables are the ones which will be computed in parallel.
        // The last log_packing_width variables are the ones which will be packed.

        // Create a buffer matrix of PackedField elements of size `num_threads × num_points`
        let mut parallel_buffer = RowMajorMatrix::new(
            E::PackedField::zero_vec(num_threads * evals.width()),
            evals.width(),
        );

        // As num_threads is a power of two we can divide using a bit-shift.
        let out_chunk_size = out.len() >> log_num_threads;

        // Compute the equality polynomial corresponding to the last log_packing_width variables
        // and pack these for all evaluation points.
        let (front_rows, packed_rows) = evals.split_rows(eval_len_min_packing);
        let init_packings = E::init_packed_batch(packed_rows, scalars);
        parallel_buffer.row_mut(0).copy_from_slice(&init_packings);

        let (buffer_rows, middle_rows) = front_rows.split_rows(log_num_threads);

        // Update the buffer so it contains the evaluations of the equality polynomial
        // with respect to parts one and three for all evaluation points.
        fill_buffer_batch(buffer_rows, &mut parallel_buffer);

        // Finally do all computations involving the middle variables in parallel.
        out.par_chunks_exact_mut(out_chunk_size)
            .zip(parallel_buffer.par_row_slices())
            .for_each(|(out_chunk, buffer_row)| {
                E::process_chunk_batch::<INITIALIZED>(middle_rows, out_chunk, buffer_row, scalars);
            });
    }
}

/// Computes the batched equality polynomial evaluations via a recursive algorithm.
///
/// This function directly implements the batched recursive strategy, updating the entire
/// vector of scalars at each recursive step. It serves as the basic implementation for
/// smaller problem sizes where parallelism and SIMD overhead is not warranted.
///
/// # Mathematical Foundation
/// For a batch of evaluation points z_0, z_1, ..., z_{m-1} ∈ IF^n and scalars
/// γ_0, γ_1, ..., γ_{m-1}, this computes:
/// ```text
/// W(x) = ∑_i γ_i * eq(x, z_i) for all x ∈ {0,1}^n
/// ```
///
/// # Arguments
/// - `evals`: Matrix where each column represents one evaluation point z_i
/// - `scalars`: Vector of scalars [γ_0, γ_1, ..., γ_{m-1}]
/// - `out`: Output buffer of size 2^n to store the combined evaluations
///
/// # Behavior of `INITIALIZED`
/// If `INITIALIZED = false`, each value in `out` is overwritten with the computed result.
/// If `INITIALIZED = true`, the computed result is added to the existing value in `out`.
#[inline]
fn eval_eq_batch_basic<F, IF, EF, const INITIALIZED: bool>(
    evals: RowMajorMatrixView<IF>,
    scalars: &[EF],
    out: &mut [EF],
) where
    F: Field,
    IF: Field,
    EF: ExtensionField<F> + Algebra<IF>,
{
    if evals.width() == 0 {
        return;
    }

    let num_vars = evals.height();
    debug_assert_eq!(out.len(), 1 << num_vars);

    match num_vars {
        0 => {
            // Base case: sum all scalars
            let sum: EF = scalars.iter().copied().sum();
            if INITIALIZED {
                out[0] += sum;
            } else {
                out[0] = sum;
            }
        }
        1 => {
            // Use optimized 1-variable batch evaluation
            let eq_evaluations = eval_eq_1_batch(evals, scalars);
            add_or_set::<_, INITIALIZED>(out, &eq_evaluations);
        }
        2 => {
            // Use optimized 2-variable batch evaluation
            let eq_evaluations = eval_eq_2_batch(evals, scalars);
            add_or_set::<_, INITIALIZED>(out, &eq_evaluations);
        }
        3 => {
            // Use optimized 3-variable batch evaluation
            let eq_evaluations = eval_eq_3_batch(evals, scalars);
            add_or_set::<_, INITIALIZED>(out, &eq_evaluations);
        }
        _ => {
            // General recursive case: split the problem in half based on the first variable.
            // This implements the core batched recursive strategy by updating the entire
            // vector of scalars according to the recursive equality polynomial property.
            let (low, high) = out.split_at_mut(out.len() / 2);
            let (first_row, remainder) = evals.split_rows(1);

            // At each variable z_j, split the scalar vector γ into two new vectors:
            // γ_1[i] = γ[i] * z_{i,j} for the x_j = 1 branch
            let scalars_1: Vec<_> = first_row
                .values
                .iter()
                .zip(scalars)
                .map(|(&z_0, &scalar)| scalar * z_0)
                .collect();

            // γ_0[i] = γ[i] * (1 - z_{i,j}) for the x_j = 0 branch
            // Use vectorized subtraction: γ_0[i] = γ[i] - γ_1[i]
            let mut scalars_0: Vec<_> = scalars.to_vec();
            EF::sub_slices(&mut scalars_0, &scalars_1);

            // Recurse on both branches with updated scalar vectors
            eval_eq_batch_basic::<F, IF, EF, INITIALIZED>(remainder, &scalars_0, low);
            eval_eq_batch_basic::<F, IF, EF, INITIALIZED>(remainder, &scalars_1, high);
        }
    }
}

/// Computes the batched equality polynomial evaluation using packed values and parallelism.
///
/// This is the batched version of `eval_eq_packed` that processes multiple evaluation points
/// simultaneously within each parallel thread. It operates on packed scalar values for
/// improved SIMD performance while maintaining the recursive batched structure.
///
/// # Arguments
/// - `eval_points`: Matrix where each column represents one evaluation point z_i
/// - `out`: Mutable slice of output buffer for this parallel chunk
/// - `eq_evals`: Vector of packed evaluations, one for each evaluation point
/// - `scalars`: Vector of scalars [γ_0, γ_1, ..., γ_{m-1}] for weighting
///
/// # Behavior of `INITIALIZED`
/// If `INITIALIZED = false`, each value in `out` is overwritten with the computed result.
/// If `INITIALIZED = true`, the computed result is added to the existing value in `out`.
#[inline]
fn eval_eq_packed_batch<F, IF, EF, E, const INITIALIZED: bool>(
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
    debug_assert_eq!(eval_points.width(), eq_evals.len());
    debug_assert_eq!(eval_points.width(), scalars.len());

    match eval_points.height() {
        0 => {
            // Base case: sum all packed evaluations and accumulate
            let sum_packed = eq_evals
                .iter()
                .fold(E::PackedField::ZERO, |acc, &x| acc + x);
            E::accumulate_results::<INITIALIZED, 1>(out, [sum_packed], scalars[0]);
        }
        1 => {
            let eq_evaluations = eval_eq_1_batch_packed(eval_points, eq_evals);
            E::accumulate_results::<INITIALIZED, 2>(out, eq_evaluations, scalars[0]);
        }
        2 => {
            let eq_evaluations = eval_eq_2_batch_packed(eval_points, eq_evals);
            E::accumulate_results::<INITIALIZED, 4>(out, eq_evaluations, scalars[0]);
        }
        3 => {
            let eq_evaluations = eval_eq_3_batch_packed(eval_points, eq_evals);
            E::accumulate_results::<INITIALIZED, 8>(out, eq_evaluations, scalars[0]);
        }
        _ => {
            // General recursive case
            let (low, high) = out.split_at_mut(out.len() / 2);
            let (first_row, remainder) = eval_points.split_rows(1);

            // Split on the first variable: compute new packed scalars for both branches.
            // This implements the batched packed version of the recursive equality polynomial update.
            // Given packed evaluations eq_evals[i] representing partial products for evaluation point i,
            // and the current variable z_j, compute:
            // eq_evals_0[i] = eq_evals[i] * (1 - z_{i,j})  // for x_j = 0 branch
            // eq_evals_1[i] = eq_evals[i] * z_{i,j}        // for x_j = 1 branch
            let (eq_evals_0, eq_evals_1): (Vec<_>, Vec<_>) = first_row
                .values
                .iter()
                .zip(eq_evals)
                .map(|(&z_0, &eq_eval)| {
                    let eq_1 = eq_eval * z_0; // Contribution when x_0 = 1
                    let eq_0 = eq_eval - eq_1; // Contribution when x_0 = 0
                    (eq_0, eq_1)
                })
                .unzip();

            // Recurse for both branches with the remaining rows
            eval_eq_packed_batch::<F, IF, EF, E, INITIALIZED>(remainder, low, &eq_evals_0, scalars);
            eval_eq_packed_batch::<F, IF, EF, E, INITIALIZED>(
                remainder,
                high,
                &eq_evals_1,
                scalars,
            );
        }
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
/// # Behavior of `INITIALIZED`
/// If `INITIALIZED = false`, each value in `out` is overwritten with the computed result.
/// If `INITIALIZED = true`, the computed result is added to the existing value in `out`.
///
/// # Arguments:
/// - `eval_points`: The point the equality function is being evaluated at.
/// - `out`: The output buffer to store or accumulate the results.
/// - `eq_evals`: The packed evaluations of the equality polynomial.
/// - `scalar`: An optional value which may be used to scale the result depending on the strategy used
///   by the `EqualityEvaluator`.
#[inline]
fn eval_eq_common<F, IF, EF, E, const INITIALIZED: bool>(eval: &[IF], out: &mut [EF], scalar: EF)
where
    F: Field,
    IF: Field,
    EF: ExtensionField<F> + ExtensionField<IF>,
    E: EqualityEvaluator<InputField = IF, OutputField = EF>,
{
    // we assume that packing_width is a power of 2.
    let packing_width = F::Packing::WIDTH;
    let num_threads = current_num_threads().next_power_of_two();
    let log_num_threads = log2_strict_usize(num_threads);

    // Ensure that the output buffer size is correct:
    // It should be of size `2^n`, where `n` is the number of variables.
    debug_assert_eq!(out.len(), 1 << eval.len());

    // If the number of variables is small, there is no need to use
    // parallelization or packings.
    if eval.len() <= packing_width + 1 + log_num_threads {
        // A basic recursive approach.
        eval_eq_basic::<F, IF, EF, INITIALIZED>(eval, out, scalar);
    } else {
        let log_packing_width = log2_strict_usize(packing_width);
        let eval_len_min_packing = eval.len() - log_packing_width;

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
        let mut parallel_buffer = E::PackedField::zero_vec(num_threads);

        // As num_threads is a power of two we can divide using a bit-shift.
        let out_chunk_size = out.len() >> log_num_threads;

        // Compute the equality polynomial corresponding to the last log_packing_width elements
        // and pack these.
        parallel_buffer[0] = E::init_packed(&eval[eval_len_min_packing..], scalar);

        // Update the buffer so it contains the evaluations of the equality polynomial
        // with respect to parts one and three.
        fill_buffer(eval[..log_num_threads].iter().rev(), &mut parallel_buffer);

        // Finally do all computations involving the middle elements.
        out.par_chunks_exact_mut(out_chunk_size)
            .zip(parallel_buffer.par_iter())
            .for_each(|(out_chunk, &buffer_val)| {
                E::process_chunk::<INITIALIZED>(
                    &eval[log_num_threads..eval_len_min_packing],
                    out_chunk,
                    buffer_val,
                    scalar,
                );
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
    eval_points: &[IF],
    out: &mut [EF],
    eq_evals: E::PackedField,
    scalar: EF,
) where
    F: Field,
    IF: Field,
    EF: ExtensionField<F>,
    E: EqualityEvaluator<InputField = IF, OutputField = EF>,
{
    // Ensure that the output buffer size is correct:
    // It should be of size `2^n`, where `n` is the number of variables.
    let width = F::Packing::WIDTH;
    debug_assert_eq!(out.len(), width << eval_points.len());

    match eval_points.len() {
        0 => {
            E::accumulate_results::<INITIALIZED, 1>(out, [eq_evals], scalar);
        }
        1 => {
            let eq_evaluations = eval_eq_1(eval_points, eq_evals);
            E::accumulate_results::<INITIALIZED, 2>(out, eq_evaluations, scalar);
        }
        2 => {
            let eq_evaluations = eval_eq_2(eval_points, eq_evals);
            E::accumulate_results::<INITIALIZED, 4>(out, eq_evaluations, scalar);
        }
        3 => {
            let eq_evaluations = eval_eq_3(eval_points, eq_evals);
            E::accumulate_results::<INITIALIZED, 8>(out, eq_evaluations, scalar);
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

            eval_eq_packed::<F, IF, EF, E, INITIALIZED>(tail, low, s0, scalar);
            eval_eq_packed::<F, IF, EF, E, INITIALIZED>(tail, high, s1, scalar);
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
fn eval_eq_basic<F, IF, EF, const INITIALIZED: bool>(eval: &[IF], out: &mut [EF], scalar: EF)
where
    F: Field,
    IF: Field,
    EF: ExtensionField<F> + Algebra<IF>,
{
    // Ensure that the output buffer size is correct:
    // It should be of size `2^n`, where `n` is the number of variables.
    debug_assert_eq!(out.len(), 1 << eval.len());

    match eval.len() {
        0 => {
            if INITIALIZED {
                out[0] += scalar;
            } else {
                out[0] = scalar;
            }
        }
        1 => {
            // Manually unroll for single variable case
            let eq_evaluations = eval_eq_1(eval, scalar);
            add_or_set::<_, INITIALIZED>(out, &eq_evaluations);
        }
        2 => {
            // Manually unroll for two variable case
            let eq_evaluations = eval_eq_2(eval, scalar);
            add_or_set::<_, INITIALIZED>(out, &eq_evaluations);
        }
        3 => {
            // Manually unroll for three variable case
            let eq_evaluations = eval_eq_3(eval, scalar);
            add_or_set::<_, INITIALIZED>(out, &eq_evaluations);
        }
        _ => {
            let (&x, tail) = eval.split_first().unwrap();

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
fn packed_eq_poly<F, EF>(eval: &[EF], scalar: EF) -> EF::ExtensionPacking
where
    F: Field,
    EF: ExtensionField<F>,
{
    // As this function is only available in this file, debug_assert should be fine here.
    // If this function becomes public, this should be changed to an assert.
    debug_assert_eq!(F::Packing::WIDTH, 1 << eval.len());

    // We build up the evaluations of the equality polynomial in buffer.
    let mut buffer = EF::zero_vec(1 << eval.len());
    buffer[0] = scalar;

    fill_buffer(eval.iter().rev(), &mut buffer);

    // Finally we need to do a "transpose" to get a `PackedFieldExtension` element.
    EF::ExtensionPacking::from_ext_slice(&buffer)
}

/// Computes batched small equality polynomial evaluations and packs the results into packed vectors.
///
/// Handles multiple evaluation points simultaneously during the packing phase of parallel
/// evaluation. Processes the bottom log_packing_width variables for all points in the batch
/// and returns packed results.
///
/// The function builds up the evaluations of the equality polynomial in a matrix buffer
/// where rows represent the 2^{height} possible input combinations and columns represent
/// different evaluation points. It then transposes and packs each column into packed vectors.
///
/// # Mathematical Foundation
/// For evaluation points z_i with height variables each, this computes:
/// ```text
/// eq(x, z_i) for all x ∈ {0,1}^{height} and all points z_i
/// ```
/// The results are packed into SIMD-friendly formats for efficient parallel processing.
///
/// # Arguments
/// - `evals`: Matrix where each column is an evaluation point z_i (height = log_packing_width)
/// - `scalars`: Vector of scalars [γ_0, γ_1, ..., γ_{m-1}] for weighting each point
///
/// # Returns
/// A vector of packed field elements, one for each evaluation point in the batch.
///
/// # Panics
/// Panics in debug builds if `F::Packing::WIDTH != 1 << evals.height()` or
/// `evals.width() != scalars.len()`.
#[inline(always)]
fn packed_eq_poly_batch<F, EF>(
    evals: RowMajorMatrixView<EF>,
    scalars: &[EF],
) -> Vec<EF::ExtensionPacking>
where
    F: Field,
    EF: ExtensionField<F>,
{
    debug_assert_eq!(F::Packing::WIDTH, 1 << evals.height());
    debug_assert_eq!(evals.width(), scalars.len());

    // We build up the evaluations of the equality polynomial in buffer.
    // Buffer is organized as: rows = 2^evals.height(), columns = num_points
    let mut buffer = RowMajorMatrix::new(
        EF::zero_vec((1 << evals.height()) * evals.width()),
        evals.width(),
    );

    // Initialize first row with scalars
    buffer.row_mut(0).copy_from_slice(scalars);

    fill_buffer_batch(evals, &mut buffer);

    // Transpose and pack each column
    (0..evals.width())
        .map(|col_idx| {
            let column: Vec<EF> = (0..(1 << evals.height()))
                .map(|row_idx| buffer.values[row_idx * buffer.width() + col_idx])
                .collect();
            EF::ExtensionPacking::from_ext_slice(&column)
        })
        .collect()
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
    use rand::distr::StandardUniform;
    use rand::rngs::SmallRng;
    use rand::{Rng, SeedableRng};

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
        fn prop_eval_eq_matches_naive(
            n in 1usize..6, // number of variables
            evals in prop::collection::vec(0u64..F::ORDER_U64, 1..6),
            scalar_val in 0u64..F::ORDER_U64,
        ) {
            // Take exactly n elements and map to EF4
            let evals: Vec<EF4> = evals.into_iter().take(n).map(EF4::from_u64).collect();
            let scalar = EF4::from_u64(scalar_val);

            // Make sure output has correct size: 2^n
            let out_len = 1 << evals.len();
            let mut output = vec![EF4::ZERO; out_len];

            eval_eq::<F, EF4, true>(&evals, &mut output, scalar);

            let expected = naive_eq(&evals, scalar);

            prop_assert_eq!(output, expected);
        }
    }

    #[test]
    fn test_eval_eq_1_against_naive() {
        let rng = &mut SmallRng::seed_from_u64(0);

        // Choose a few values of z_0 and α to test
        let test_cases = vec![
            (rng.sample(StandardUniform), rng.sample(StandardUniform)),
            (rng.sample(StandardUniform), rng.sample(StandardUniform)),
            (rng.sample(StandardUniform), rng.sample(StandardUniform)),
            (rng.sample(StandardUniform), rng.sample(StandardUniform)),
            (rng.sample(StandardUniform), rng.sample(StandardUniform)),
        ];

        for (z_0, alpha) in test_cases {
            // Compute using the optimized eval_eq_1 function
            let result = eval_eq_1::<F, F>(&[z_0], alpha);

            // Compute eq(0, z_0) and eq(1, z_0) naively using the full formula:
            //
            // eq(x, z_0) = x * z_0 + (1 - x) * (1 - z_0)
            //
            let x0 = F::ZERO;
            let x1 = F::ONE;

            let eq_0 = x0 * z_0 + (F::ONE - x0) * (F::ONE - z_0);
            let eq_1 = x1 * z_0 + (F::ONE - x1) * (F::ONE - z_0);

            // Scale by α
            let expected_0 = alpha * eq_0;
            let expected_1 = alpha * eq_1;

            assert_eq!(
                result[0], expected_0,
                "eq(0, z_0) mismatch for z_0 = {z_0:?}"
            );
            assert_eq!(
                result[1], expected_1,
                "eq(1, z_0) mismatch for z_0 = {z_0:?}"
            );
        }
    }

    #[test]
    fn test_eval_eq_2_against_naive() {
        let rng = &mut SmallRng::seed_from_u64(42);

        // Generate a few random test cases for (z_0, z_1) and α
        let test_cases = (0..5)
            .map(|_| {
                let z_0: F = rng.sample(StandardUniform);
                let z_1: F = rng.sample(StandardUniform);
                let alpha: F = rng.sample(StandardUniform);
                ([z_0, z_1], alpha)
            })
            .collect::<Vec<_>>();

        for ([z_0, z_1], alpha) in test_cases {
            // Optimized output
            let result = eval_eq_2::<F, F>(&[z_0, z_1], alpha);

            // Naive computation using the full formula:
            //
            // eq(x, z) = ∏ (x_i z_i + (1 - x_i)(1 - z_i))
            // for x ∈ { (0,0), (0,1), (1,0), (1,1) }

            let inputs = [
                (F::ZERO, F::ZERO), // x = (0,0)
                (F::ZERO, F::ONE),  // x = (0,1)
                (F::ONE, F::ZERO),  // x = (1,0)
                (F::ONE, F::ONE),   // x = (1,1)
            ];

            for (i, (x0, x1)) in inputs.iter().enumerate() {
                let eq_val = (*x0 * z_0 + (F::ONE - *x0) * (F::ONE - z_0))
                    * (*x1 * z_1 + (F::ONE - *x1) * (F::ONE - z_1));
                let expected = alpha * eq_val;

                assert_eq!(
                    result[i], expected,
                    "Mismatch at x = ({x0:?}, {x1:?}), z = ({z_0:?}, {z_1:?})"
                );
            }
        }
    }

    #[test]
    fn test_eval_eq_3_against_naive() {
        let rng = &mut SmallRng::seed_from_u64(123);

        // Generate random test cases for (z_0, z_1, z_2) and α
        let test_cases = (0..5)
            .map(|_| {
                let z_0: F = rng.sample(StandardUniform);
                let z_1: F = rng.sample(StandardUniform);
                let z_2: F = rng.sample(StandardUniform);
                let alpha: F = rng.sample(StandardUniform);
                ([z_0, z_1, z_2], alpha)
            })
            .collect::<Vec<_>>();

        for ([z_0, z_1, z_2], alpha) in test_cases {
            // Optimized computation
            let result = eval_eq_3::<F, F>(&[z_0, z_1, z_2], alpha);

            // Naive computation using:
            // eq(x, z) = ∏ (x_i z_i + (1 - x_i)(1 - z_i))
            let inputs = [
                (F::ZERO, F::ZERO, F::ZERO), // (0,0,0)
                (F::ZERO, F::ZERO, F::ONE),  // (0,0,1)
                (F::ZERO, F::ONE, F::ZERO),  // (0,1,0)
                (F::ZERO, F::ONE, F::ONE),   // (0,1,1)
                (F::ONE, F::ZERO, F::ZERO),  // (1,0,0)
                (F::ONE, F::ZERO, F::ONE),   // (1,0,1)
                (F::ONE, F::ONE, F::ZERO),   // (1,1,0)
                (F::ONE, F::ONE, F::ONE),    // (1,1,1)
            ];

            for (i, (x0, x1, x2)) in inputs.iter().enumerate() {
                let eq_val = (*x0 * z_0 + (F::ONE - *x0) * (F::ONE - z_0))
                    * (*x1 * z_1 + (F::ONE - *x1) * (F::ONE - z_1))
                    * (*x2 * z_2 + (F::ONE - *x2) * (F::ONE - z_2));
                let expected = alpha * eq_val;

                assert_eq!(
                    result[i], expected,
                    "Mismatch at x = ({x0:?}, {x1:?}, {x2:?}), z = ({z_0:?}, {z_1:?}, {z_2:?})"
                );
            }
        }
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

    #[test]
    fn test_eval_eq_batch_functionality() {
        // Test batched evaluation with 2 variables and 3 evaluation points
        // Matrix layout: rows are variables, columns are evaluation points
        // Point 1: (1, 0), Point 2: (0, 1), Point 3: (1, 1)
        let evals_data = vec![
            F::from_u64(1),
            F::from_u64(0),
            F::from_u64(1), // z_0 values for all points
            F::from_u64(0),
            F::from_u64(1),
            F::from_u64(1), // z_1 values for all points
        ];
        let evals = RowMajorMatrixView::new(&evals_data, 3); // 2 rows (variables) × 3 columns (points)
        let scalars = vec![F::from_u64(2), F::from_u64(3), F::from_u64(5)]; // γ₀=2, γ₁=3, γ₂=5

        let mut output_batch = vec![F::ZERO; 4]; // 2^2 = 4 elements
        eval_eq_batch::<_, _, false>(evals, &scalars, &mut output_batch);

        // Compute expected result by evaluating individual equality polynomials and summing
        let mut expected_output = vec![F::ZERO; 4];
        let points = [
            vec![F::from_u64(1), F::from_u64(0)], // Point 1: (1, 0)
            vec![F::from_u64(0), F::from_u64(1)], // Point 2: (0, 1)
            vec![F::from_u64(1), F::from_u64(1)], // Point 3: (1, 1)
        ];
        for (point, &scalar) in points.iter().zip(scalars.iter()) {
            let mut temp_output = vec![F::ZERO; 4];
            eval_eq::<_, _, false>(point, &mut temp_output, scalar);
            F::add_slices(&mut expected_output, &temp_output);
        }

        assert_eq!(output_batch, expected_output);
    }

    #[test]
    fn test_eval_eq_base_batch_functionality() {
        // Test base field batch evaluation
        // Point 1: (1, 0), Point 2: (0, 1)
        let evals_data = vec![
            F::from_u64(1),
            F::from_u64(0), // z_0 values for all points
            F::from_u64(0),
            F::from_u64(1), // z_1 values for all points
        ];
        let evals = RowMajorMatrixView::new(&evals_data, 2); // 2 rows (variables) × 2 columns (points)
        let scalars = vec![EF4::from_u64(2), EF4::from_u64(3)];

        let mut output_batch = vec![EF4::ZERO; 4];
        eval_eq_base_batch::<_, _, false>(evals, &scalars, &mut output_batch);

        // Compare with individual evaluations
        let mut expected_output = vec![EF4::ZERO; 4];
        let points = [
            vec![F::from_u64(1), F::from_u64(0)], // Point 1: (1, 0)
            vec![F::from_u64(0), F::from_u64(1)], // Point 2: (0, 1)
        ];
        for (point, &scalar) in points.iter().zip(scalars.iter()) {
            let mut temp_output = vec![EF4::ZERO; 4];
            eval_eq_base::<_, _, false>(point, &mut temp_output, scalar);
            EF4::add_slices(&mut expected_output, &temp_output);
        }

        assert_eq!(output_batch, expected_output);
    }

    proptest! {
        #[test]
        fn prop_eval_eq_batch_matches_individual_sum(
            n in 1usize..4, // number of variables (small to keep test fast)
            num_points in 1usize..6, // number of evaluation points
            point_values in prop::collection::vec(0u64..F::ORDER_U64, 1..24), // flatten point values
            scalar_values in prop::collection::vec(0u64..F::ORDER_U64, 1..6),
        ) {
            // Create evaluation points matrix: num_points × n
            let mut eval_points = Vec::new();
            let points_data: Vec<Vec<F>> = point_values
                .chunks(n)
                .take(num_points)
                .map(|chunk| {
                    let mut point = chunk.iter().map(|&x| F::from_u64(x)).collect::<Vec<_>>();
                    point.resize(n, F::ZERO);
                    point
                })
                .collect();

            for point in &points_data {
                eval_points.push(point.as_slice());
            }
            eval_points.truncate(num_points);

            // Create scalars
            let scalars: Vec<EF4> = scalar_values
                .iter()
                .take(num_points)
                .map(|&x| EF4::from_u64(x))
                .collect();

            // Pad scalars if needed
            let mut scalars = scalars;
            scalars.resize(eval_points.len(), EF4::ZERO);

            let out_len = 1 << n;

            // Create matrix for batch evaluation (convert to extension field)
            // Matrix layout: rows are variables, columns are evaluation points
            let mut evals_data = Vec::with_capacity(n * eval_points.len());
            for var_idx in 0..n {
                for point in &eval_points {
                    evals_data.push(EF4::from(point[var_idx]));
                }
            }
            let evals = RowMajorMatrixView::new(&evals_data, eval_points.len());

            let mut output_batch = vec![EF4::ZERO; out_len];
            eval_eq_batch::<F, EF4, false>(evals, &scalars, &mut output_batch);

            // Compute using individual evaluations and manual summation
            let mut expected_output = vec![EF4::ZERO; out_len];
            for (point, &scalar) in eval_points.iter().zip(scalars.iter()) {
                let point_ext: Vec<EF4> = point.iter().map(|&f| EF4::from(f)).collect();
                let mut temp_output = vec![EF4::ZERO; out_len];
                eval_eq::<F, EF4, false>(&point_ext, &mut temp_output, scalar);
                EF4::add_slices(&mut expected_output, &temp_output);
            }

            prop_assert_eq!(output_batch, expected_output);
        }
    }

    #[test]
    fn test_eval_eq_parallel_path() {
        // Calculate the threshold for parallel execution:
        // threshold = packing_width + 1 + log_num_threads
        let packing_width = <F as Field>::Packing::WIDTH;
        let num_threads = current_num_threads().next_power_of_two();
        let log_num_threads = log2_strict_usize(num_threads);
        let threshold = packing_width + 1 + log_num_threads;

        // Use variables > threshold to force parallel path
        let num_vars = threshold + 2;

        // Create random evaluation point
        let mut rng = SmallRng::seed_from_u64(12345);
        let eval_point: Vec<EF4> = (0..num_vars).map(|_| rng.random()).collect();

        let scalar = EF4::from_u64(7);

        // Test parallel path
        let mut output_parallel = EF4::zero_vec(1 << num_vars);
        eval_eq::<F, EF4, false>(&eval_point, &mut output_parallel, scalar);

        // Verify correctness by comparing against basic evaluation for a small subset
        let mut output_basic = EF4::zero_vec(1 << num_vars);
        eval_eq_basic::<F, EF4, EF4, false>(&eval_point, &mut output_basic, scalar);

        assert_eq!(
            output_parallel, output_basic,
            "Parallel path should match basic evaluation"
        );
    }

    #[test]
    fn test_eval_eq_batch_parallel_path() {
        // Calculate threshold for parallel execution in batched case:
        // threshold = packing_width.ilog2() + 1 + log_num_threads
        let packing_width = <F as Field>::Packing::WIDTH;
        let num_threads = current_num_threads().next_power_of_two();
        let log_num_threads = log2_strict_usize(num_threads);
        let threshold = packing_width.ilog2() as usize + 1 + log_num_threads;

        // Use variables > threshold to force parallel path
        let num_vars = threshold + 2;
        let num_points = 3;

        // Create random evaluation points and scalars
        let mut rng = SmallRng::seed_from_u64(54321);
        let eval_points: Vec<Vec<F>> = (0..num_points)
            .map(|_| (0..num_vars).map(|_| rng.random()).collect())
            .collect();

        let scalars: Vec<EF4> = (0..num_points)
            .map(|i| EF4::from_u64(i as u64 + 1))
            .collect();

        // Create matrix layout: rows are variables, columns are evaluation points
        let mut evals_data = Vec::with_capacity(num_vars * num_points);
        for var_idx in 0..num_vars {
            for point in &eval_points {
                evals_data.push(EF4::from(point[var_idx]));
            }
        }
        let evals = RowMajorMatrixView::new(&evals_data, num_points);

        // Test parallel batched path
        let mut output_batch_parallel = EF4::zero_vec(1 << num_vars);
        eval_eq_batch::<F, EF4, false>(evals, &scalars, &mut output_batch_parallel);

        // Verify correctness by comparing against basic batch evaluation
        let mut output_batch_basic = EF4::zero_vec(1 << num_vars);
        eval_eq_batch_basic::<F, EF4, EF4, false>(evals, &scalars, &mut output_batch_basic);

        assert_eq!(
            output_batch_parallel, output_batch_basic,
            "Parallel batched path should match basic batched evaluation"
        );
    }
}
