//! This module provides optimized routines for computing **batched multilinear equality polynomials**
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
//! ## Batched Evaluation
//!
//! The batched methods (`eval_eq_batch`, `eval_eq_base_batch`) are designed to efficiently compute
//! linear combinations of multiple equality polynomial evaluations. Instead of computing each
//! equality polynomial individually and then summing the results, these functions leverage linearity
//! to perform the summation within the recursive evaluation process.
//!
//! The batched variants compute a linear combination of equality tables in one pass:
//!
//! ```text
//! W(x) = \sum_i \γ_i ⋅ eq(x, z_i)  ,  x ∈ {0,1}^n .
//! ```
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

use alloc::vec;
use alloc::vec::Vec;

use p3_field::{
    Algebra, ExtensionField, Field, PackedFieldExtension, PackedValue, PrimeCharacteristicRing,
    dot_product,
};
use p3_matrix::Matrix;
use p3_matrix::dense::{RowMajorMatrix, RowMajorMatrixView};
use p3_maybe_rayon::prelude::*;
use p3_util::log2_strict_usize;

/// Computes the batched multilinear equality polynomial `\sum_i \γ_i ⋅ eq(x, z_i)` over all
/// `x ∈ \{0,1\}^n` for multiple points `z_i ∈ EF^n` with weights `\γ_i ∈ EF`.
///
/// This evaluates multiple equality tables simultaneously by pushing the linear combination
/// through the recursion.
///
/// # Mathematical statement
/// Given:
/// - evaluation points `z_0, z_1, ..., z_{m-1} ∈ F^n`,
/// - weights `\γ_0, \γ_1, ..., \γ_{m-1} ∈ EF`, this computes, for all `x ∈ {0,1}^n`:
/// ```text
/// W(x) = \sum_i \γ_i ⋅ eq(x, z_i).
/// ```
///
/// # Arguments
/// - `evals`: Matrix where each column is one point `z_i`.
///     - height = number of variables `n`,
///     - width = number of points `m`
/// - `out`: Output buffer of size `2^n` storing `W(x)` in big-endian `x` order
/// - `scalars`: Weights `[ \γ_0, \γ_1, ..., \γ_{m-1} ]`
///
/// # Panics
/// Panics in debug builds if `evals.width() != scalars.len()` or if the output buffer size is incorrect.
#[inline]
pub fn eval_eq_batch<F, EF, const INITIALIZED: bool>(
    evals: RowMajorMatrixView<'_, EF>,
    out: &mut [EF],
    scalars: &[EF],
) where
    F: Field,
    EF: ExtensionField<F>,
{
    eval_eq_batch_common::<F, EF, EF, ExtFieldEvaluator<F, EF>, INITIALIZED>(evals, out, scalars);
}

/// Computes the batched multilinear equality polynomial `\sum_i \γ_i ⋅ eq(x, z_i)` over all
/// `x ∈ \{0,1\}^n` for multiple points `z_i ∈ F^n` with weights `\γ_i ∈ EF`.
///
/// This evaluates multiple equality tables simultaneously by pushing the linear combination
/// through the recursion.
///
/// # Mathematical statement
/// Given:
/// - evaluation points `z_0, z_1, ..., z_{m-1} ∈ EF^n`,
/// - weights `\γ_0, \γ_1, ..., \γ_{m-1} ∈ EF`, this computes, for all `x ∈ {0,1}^n`:
/// ```text
/// W(x) = \sum_i \γ_i ⋅ eq(x, z_i).
/// ```
///
/// # Arguments
/// - `evals`: Matrix where each column is one point `z_i`.
///     - height = number of variables `n`,
///     - width = number of points `m`
/// - `out`: Output buffer of size `2^n` storing `W(x)` in big-endian `x` order
/// - `scalars`: Weights `[ \γ_0, \γ_1, ..., \γ_{m-1} ]`
///
/// # Panics
/// Panics in debug builds if `evals.width() != scalars.len()` or if the output buffer size is incorrect.
#[inline]
pub fn eval_eq_base_batch<F, EF, const INITIALIZED: bool>(
    evals: RowMajorMatrixView<'_, F>,
    out: &mut [EF],
    scalars: &[EF],
) where
    F: Field,
    EF: ExtensionField<F>,
{
    eval_eq_batch_common::<F, F, EF, BaseFieldEvaluator<F, EF>, INITIALIZED>(evals, out, scalars);
}

/// Fills the `buffer` with evaluations of the equality polynomial for multiple points simultaneously.
///
/// This is the batched operation that operates on matrices where each column
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
fn fill_buffer_batch<F, A>(evals: RowMajorMatrixView<'_, F>, buffer: &mut RowMajorMatrix<A>)
where
    F: Field,
    A: Algebra<F> + Send + Sync + Clone,
{
    // Process variables in reverse order to maintain correct bit ordering in output buffer.
    //
    // We apply it simultaneously across all columns (evaluation points).
    for (ind, eval_row) in evals.row_slices().rev().enumerate() {
        let stride = 1 << ind;
        let width = buffer.width();

        // Expand the buffer in-place by doubling its height at each step.
        //
        // Each existing row generates two new rows: one for x_j = 0, one for x_j = 1.
        for idx in 0..stride {
            // Process each column (each point in the batch) for the current row `idx`.
            for (col, &eval_point) in eval_row.iter().enumerate().take(width) {
                // Read the current value directly from the buffer.
                let val = buffer.values[idx * width + col].clone();

                // Compute the two new values for the next level of the hypercube.
                let scaled_val = val.clone() * eval_point;
                let new_val = val - scaled_val.clone();

                // Write the results back in-place.
                buffer.values[idx * width + col] = new_val;
                buffer.values[(idx + stride) * width + col] = scaled_val;
            }
        }
    }
}

/// Computes the batched scaled multilinear equality polynomial over `{0,1}` for multiple points.
///
/// We compute:
/// ```text
/// eq_sum(0) = ∑_i scalars[i] * (1 - evals[0][i])
/// eq_sum(1) = ∑_i scalars[i] * evals[0][i]
/// ```
///
/// # Arguments
/// - `evals`: Matrix where each column is an evaluation point z_i (must have height = 1)
/// - `scalars`: Vector of scalars [γ_0, γ_1, ..., γ_{m-1}] for weighting each evaluation
///
/// # Returns
/// An array `[eq_sum(0), eq_sum(1)]`.
#[inline(always)]
fn eval_eq_1_batch<F, FP>(evals: RowMajorMatrixView<'_, F>, scalars: &[FP]) -> [FP; 2]
where
    F: Field,
    FP: Algebra<F>,
{
    debug_assert_eq!(evals.height(), 1);
    debug_assert_eq!(evals.width(), scalars.len());

    // Compute the total sum of all scalars: ∑_i γ_i
    let sum: FP = scalars.iter().cloned().sum();

    // Compute ∑_i γ_i * z_{i,0}
    //
    // This gives us eq_sum(1) directly since eq(1, z) = z
    let eq_1_sum: FP = dot_product(scalars.iter().cloned(), evals.values.iter().copied());

    // Use the identity: eq(0, z_i) = 1 - z_i.
    //
    // So ∑_i γ_i * (1 - z_i) = ∑_i γ_i - ∑_i γ_i * z_i.
    //
    // This saves approximately m adds compared to computing each term individually
    let eq_0_sum = sum - eq_1_sum.clone();

    [eq_0_sum, eq_1_sum]
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
/// - `workspace`: Mutable workspace buffer (must have a size of exactly `2 * num_points`)
///
/// # Returns
/// An array of length 4 containing `∑_i scalars[i] * eq(x, z_i)` for all x ∈ {0,1}^2
/// in lexicographic order: [eq(00), eq(01), eq(10), eq(11)].
///
/// # Panics
/// Panics in debug builds if `evals.height() != 2` or `evals.width() != scalars.len()`.
#[inline(always)]
fn eval_eq_2_batch<F, FP>(
    evals: RowMajorMatrixView<'_, F>,
    scalars: &[FP],
    workspace: &mut [FP],
) -> [FP; 4]
where
    F: Field,
    FP: Algebra<F> + Field,
{
    debug_assert_eq!(evals.height(), 2);
    debug_assert_eq!(evals.width(), scalars.len());

    let (first_row, second_row) = evals.split_rows(1);
    let num_points = evals.width();

    // Split workspace into two buffers for the two scalar vectors
    let (eq_0s, remaining) = workspace.split_at_mut(num_points);
    let eq_1s = &mut remaining[..num_points];

    // Compute the two scalar vectors in-place using workspace buffers
    for i in 0..num_points {
        let s = scalars[i];
        let z = first_row.values[i];
        let s1 = s * z;
        eq_1s[i] = s1;
        eq_0s[i] = s - s1;
    }

    // Recurse to calculate evaluations for the remaining variable
    let [eq_00, eq_01] = eval_eq_1_batch(second_row, eq_0s);
    let [eq_10, eq_11] = eval_eq_1_batch(second_row, eq_1s);

    // Return values in lexicographic order of x = (x_0, x_1)
    [eq_00, eq_01, eq_10, eq_11]
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
/// - `workspace`: Mutable workspace buffer (must have size ≥ 6 * num_points)
///
/// # Returns
/// An array of length 8 containing `∑_i scalars[i] * eq(x, z_i)` for all x ∈ {0,1}^3
/// in lexicographic order: [eq(000), eq(001), eq(010), eq(011), eq(100), eq(101), eq(110), eq(111)].
///
/// # Panics
/// Panics in debug builds if `evals.height() != 3` or `evals.width() != scalars.len()`.
#[inline(always)]
fn eval_eq_3_batch<F, FP>(
    evals: RowMajorMatrixView<'_, F>,
    scalars: &[FP],
    workspace: &mut [FP],
) -> [FP; 8]
where
    F: Field,
    FP: Algebra<F> + Field,
{
    debug_assert_eq!(evals.height(), 3);
    debug_assert_eq!(evals.width(), scalars.len());

    let (first_row, remainder) = evals.split_rows(1);
    let num_points = evals.width();

    // Split workspace into buffers for the scalar vectors
    let (eq_0s, next_workspace) = workspace.split_at_mut(num_points);
    let (eq_1s, next_workspace) = next_workspace.split_at_mut(num_points);

    // Compute the two scalar vectors in-place using workspace buffers
    for i in 0..num_points {
        let s = scalars[i];
        let z = first_row.values[i];
        let s1 = s * z;
        eq_1s[i] = s1;
        eq_0s[i] = s - s1;
    }

    // Split the remaining workspace for the recursive calls
    let (ws0, remaining) = next_workspace.split_at_mut(2 * num_points);
    let ws1 = &mut remaining[..2 * num_points];

    // Recurse to calculate evaluations for the remaining variables
    let [eq_000, eq_001, eq_010, eq_011] = eval_eq_2_batch(remainder, eq_0s, ws0);
    let [eq_100, eq_101, eq_110, eq_111] = eval_eq_2_batch(remainder, eq_1s, ws1);

    // Return all 8 evaluations in lexicographic order of x ∈ {0,1}^3
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

    fn init_packed_batch(
        evals: RowMajorMatrixView<'_, Self::InputField>,
        scalars: &[Self::OutputField],
    ) -> Vec<Self::PackedField>;

    fn process_chunk_batch<const INITIALIZED: bool>(
        evals: RowMajorMatrixView<'_, Self::InputField>,
        out_chunk: &mut [Self::OutputField],
        buffer_vals: &[Self::PackedField],
        scalars: &[Self::OutputField],
    );

    fn accumulate_packed_batch<const INITIALIZED: bool>(
        out: &mut [Self::OutputField],
        final_packed_evals: &[Self::PackedField],
        scalars: &[Self::OutputField],
    );
}

/// Evaluation Strategy for the base field case.
///
/// We stay in the base field for as long as possible to simplify instructions and
/// reduce the amount of data transferred between cores. In particular this means we
/// hold off on scaling by `scalar` until the very end.
struct BaseFieldEvaluator<F, EF>(core::marker::PhantomData<(F, EF)>);

/// Implementation for extension field case.
///
/// We initialise with `scalar` instead of `1` as this reduces the total
/// number of multiplications we need to do.
struct ExtFieldEvaluator<F, EF>(core::marker::PhantomData<(F, EF)>);

impl<F: Field, EF: ExtensionField<F>> EqualityEvaluator for ExtFieldEvaluator<F, EF> {
    type InputField = EF;
    type OutputField = EF;
    type PackedField = EF::ExtensionPacking;

    fn init_packed_batch(
        evals: RowMajorMatrixView<'_, Self::InputField>,
        scalars: &[Self::OutputField],
    ) -> Vec<Self::PackedField> {
        packed_eq_poly_batch(evals, scalars)
    }

    fn process_chunk_batch<const INITIALIZED: bool>(
        evals: RowMajorMatrixView<'_, Self::InputField>,
        out_chunk: &mut [Self::OutputField],
        buffer_vals: &[Self::PackedField],
        scalars: &[Self::OutputField],
    ) {
        let num_vars = evals.height();
        let num_points = evals.width();
        // Allocate workspace for the recursive evaluation.
        //
        // The size is the maximum of:
        // - The space needed for a deep recursion,
        // - The largest unrolled base case (n=3).
        //
        // This ensures enough memory for all scenarios.
        let workspace_len = (2 * (num_vars + 1) * num_points).max(8 * num_points);
        let mut workspace = Self::PackedField::zero_vec(workspace_len);
        eval_eq_packed_batch::<F, EF, EF, Self, INITIALIZED>(
            evals,
            out_chunk,
            buffer_vals,
            scalars,
            &mut workspace,
        );
    }

    fn accumulate_packed_batch<const INITIALIZED: bool>(
        out: &mut [Self::OutputField],
        final_packed_evals: &[Self::PackedField],
        _scalars: &[Self::OutputField],
    ) {
        // Handle the empty batch case.
        if final_packed_evals.is_empty() {
            if !INITIALIZED {
                out.fill(Self::OutputField::ZERO);
            }
            return;
        }

        // Sum all packed field elements first.
        let packed_sum: Self::PackedField = final_packed_evals.iter().copied().sum();

        // Now unpack the single sum result.
        let unpacked_iter = Self::PackedField::to_ext_iter([packed_sum]);

        // Write or add the unpacked result to the output buffer.
        if INITIALIZED {
            out.iter_mut()
                .zip(unpacked_iter)
                .for_each(|(out_val, unpacked_val)| *out_val += unpacked_val);
        } else {
            out.iter_mut()
                .zip(unpacked_iter)
                .for_each(|(out_val, unpacked_val)| *out_val = unpacked_val);
        }
    }
}

impl<F: Field, EF: ExtensionField<F>> EqualityEvaluator for BaseFieldEvaluator<F, EF> {
    type InputField = F;
    type OutputField = EF;
    type PackedField = F::Packing;

    fn init_packed_batch(
        evals: RowMajorMatrixView<'_, Self::InputField>,
        _scalars: &[Self::OutputField],
    ) -> Vec<Self::PackedField> {
        packed_eq_poly_batch(evals, &vec![F::ONE; evals.width()])
    }

    fn process_chunk_batch<const INITIALIZED: bool>(
        evals: RowMajorMatrixView<'_, Self::InputField>,
        out_chunk: &mut [Self::OutputField],
        buffer_vals: &[Self::PackedField],
        scalars: &[Self::OutputField],
    ) {
        let num_vars = evals.height();
        let num_points = evals.width();

        // Allocate workspace for the recursive evaluation.
        //
        // The size is the maximum of:
        // - The space needed for a deep recursion,
        // - The largest unrolled base case (n=3).
        //
        // This ensures enough memory for all scenarios.
        let workspace_len = (2 * (num_vars + 1) * num_points).max(8 * num_points);
        let mut workspace = Self::PackedField::zero_vec(workspace_len);
        eval_eq_packed_batch::<F, F, EF, Self, INITIALIZED>(
            evals,
            out_chunk,
            buffer_vals,
            scalars,
            &mut workspace,
        );
    }

    fn accumulate_packed_batch<const INITIALIZED: bool>(
        out: &mut [Self::OutputField],
        final_packed_evals: &[Self::PackedField],
        scalars: &[Self::OutputField],
    ) {
        debug_assert_eq!(out.len(), F::Packing::WIDTH);
        debug_assert_eq!(final_packed_evals.len(), scalars.len());

        // Handle the empty batch case.
        if scalars.is_empty() {
            if !INITIALIZED {
                out.fill(Self::OutputField::ZERO);
            }
            return;
        }

        // Iterate through each lane of the packed values (from 0 to WIDTH-1).
        //
        // For each lane `k`, we compute the dot product across the entire batch.
        for (k, out_val) in out.iter_mut().enumerate() {
            // This computes: ∑_i scalars[i] * final_packed_evals[i][k]
            let dot_product = scalars
                .iter()
                .zip(final_packed_evals)
                .map(|(&scalar, packed_eval)| scalar * packed_eval.as_slice()[k])
                .sum::<Self::OutputField>();

            if INITIALIZED {
                *out_val += dot_product;
            } else {
                *out_val = dot_product;
            }
        }
    }
}

/// Computes the batched multilinear equality polynomial `∑_i γ_i ⋅ eq(x, z_i)` over all `x ∈ \{0,1\}^n`
/// for multiple points `z_i ∈ IF^n` and corresponding scalars `γ_i ∈ EF`.
///
/// This is the core batched evaluation function that leverages the linearity of summation
/// to efficiently compute multiple equality polynomial evaluations simultaneously.
///
/// # Performance Benefits
/// Instead of computing each equality polynomial individually and summing the results, this approach performs
/// the summation *within* the recursive evaluation.
///
/// # Arguments
/// - `evals`: Matrix where each column represents one evaluation point z_i.
/// - `scalars`: Vector of scalars [γ_0, γ_1, ..., γ_{m-1}] corresponding to each evaluation point.
/// - `out`: Output buffer of size `2^n` to store the combined evaluations.
#[inline]
fn eval_eq_batch_common<F, IF, EF, E, const INITIALIZED: bool>(
    evals: RowMajorMatrixView<'_, IF>,
    out: &mut [EF],
    scalars: &[EF],
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

    // If the number of variables is small, there is no need to use
    // parallelization or packings.
    if num_vars <= packing_width + 1 + log_num_threads {
        // Allocate workspace once for all recursive calls
        //
        // The max function ensures we allocate at least 1 element to avoid empty slice issues
        let mut workspace = EF::zero_vec((2 * evals.width() * num_vars).max(1));
        eval_eq_batch_basic::<F, IF, EF, INITIALIZED>(evals, scalars, out, &mut workspace);
    } else {
        let log_packing_width = log2_strict_usize(packing_width);
        let eval_len_min_packing = num_vars - log_packing_width;

        // Split the variables into three parts:
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
/// - `workspace`: Mutable workspace buffer for storing intermediate scalar values
///
/// # Behavior of `INITIALIZED`
/// If `INITIALIZED = false`, each value in `out` is overwritten with the computed result.
/// If `INITIALIZED = true`, the computed result is added to the existing value in `out`.
#[inline]
fn eval_eq_batch_basic<F, IF, EF, const INITIALIZED: bool>(
    evals: RowMajorMatrixView<'_, IF>,
    scalars: &[EF],
    out: &mut [EF],
    workspace: &mut [EF],
) where
    F: Field,
    IF: Field,
    EF: ExtensionField<F> + Algebra<IF>,
{
    let num_vars = evals.height();
    let num_points = evals.width();
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
            // Use optimized 2-variable batch evaluation with workspace
            let eq_evaluations = eval_eq_2_batch(evals, scalars, workspace);
            add_or_set::<_, INITIALIZED>(out, &eq_evaluations);
        }
        3 => {
            // Use optimized 3-variable batch evaluation with workspace
            let eq_evaluations = eval_eq_3_batch(evals, scalars, workspace);
            add_or_set::<_, INITIALIZED>(out, &eq_evaluations);
        }
        _ => {
            // General recursive case: split the problem in half based on the first variable.
            let (low, high) = out.split_at_mut(out.len() / 2);
            let (first_row, remainder) = evals.split_rows(1);

            // Split workspace into two buffers for the two scalar vectors
            let (s0_buffer, next_workspace) = workspace.split_at_mut(num_points);
            let (s1_buffer, next_workspace) = next_workspace.split_at_mut(num_points);

            // Compute the two scalar vectors in-place using workspace buffers
            for i in 0..num_points {
                let s = scalars[i];
                let z = first_row.values[i];
                let s1 = s * z;
                s1_buffer[i] = s1;
                s0_buffer[i] = s - s1;
            }

            // Recurse on both branches with updated scalar vectors
            eval_eq_batch_basic::<F, IF, EF, INITIALIZED>(
                remainder,
                s0_buffer,
                low,
                next_workspace,
            );
            eval_eq_batch_basic::<F, IF, EF, INITIALIZED>(
                remainder,
                s1_buffer,
                high,
                next_workspace,
            );
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
/// - `workspace`: Mutable workspace buffer for storing intermediate packed scalar values
///
/// # Behavior of `INITIALIZED`
/// If `INITIALIZED = false`, each value in `out` is overwritten with the computed result.
/// If `INITIALIZED = true`, the computed result is added to the existing value in `out`.
#[inline]
fn eval_eq_packed_batch<F, IF, EF, E, const INITIALIZED: bool>(
    eval_points: RowMajorMatrixView<'_, IF>,
    out: &mut [EF],
    eq_evals: &[E::PackedField],
    scalars: &[EF],
    workspace: &mut [E::PackedField],
) where
    F: Field,
    IF: Field,
    EF: ExtensionField<F>,
    E: EqualityEvaluator<InputField = IF, OutputField = EF>,
{
    let num_vars = eval_points.height();
    let num_points = eval_points.width();
    debug_assert_eq!(out.len(), F::Packing::WIDTH << num_vars);

    match num_vars {
        0 => {
            // Base case of the recursion.
            E::accumulate_packed_batch::<INITIALIZED>(out, eq_evals, scalars);
        }
        1 => {
            // Optimized base case for 1 variable
            let first_row = eval_points;

            // Compute the two packed scalar vectors for both branches
            let (s0_buffer, s1_buffer) = workspace.split_at_mut(num_points);
            for i in 0..num_points {
                let z_0 = first_row.values[i];
                let eq_eval = eq_evals[i];
                let s1 = eq_eval * z_0;
                let s0 = eq_eval - s1;
                s0_buffer[i] = s0;
                s1_buffer[i] = s1;
            }

            // Split output for the two branches
            let (low, high) = out.split_at_mut(out.len() / 2);
            E::accumulate_packed_batch::<INITIALIZED>(low, s0_buffer, scalars);
            E::accumulate_packed_batch::<INITIALIZED>(high, s1_buffer, scalars);
        }
        2 => {
            // Optimized base case for 2 variables
            debug_assert!(workspace.len() >= 4 * num_points);
            let (first_row, second_row) = eval_points.split_rows(1);

            // Split workspace for all 4 leaf nodes.
            let (s00_buffer, rest) = workspace.split_at_mut(num_points);
            let (s01_buffer, rest) = rest.split_at_mut(num_points);
            let (s10_buffer, s11_buffer) = rest.split_at_mut(num_points);

            // Single loop to compute all 4 leaf scalars.
            for i in 0..num_points {
                let eq_eval = eq_evals[i];
                let z_0 = first_row.values[i];
                let z_1 = second_row.values[i];

                // Level 1
                let s1 = eq_eval * z_0;
                let s0 = eq_eval - s1;

                // Level 2
                let s01 = s0 * z_1;
                let s11 = s1 * z_1;
                s00_buffer[i] = s0 - s01;
                s01_buffer[i] = s01;
                s10_buffer[i] = s1 - s11;
                s11_buffer[i] = s11;
            }

            let quarter = out.len() / 4;
            let (out_00, rest) = out.split_at_mut(quarter);
            let (out_01, rest) = rest.split_at_mut(quarter);
            let (out_10, out_11) = rest.split_at_mut(quarter);

            E::accumulate_packed_batch::<INITIALIZED>(out_00, s00_buffer, scalars);
            E::accumulate_packed_batch::<INITIALIZED>(out_01, s01_buffer, scalars);
            E::accumulate_packed_batch::<INITIALIZED>(out_10, s10_buffer, scalars);
            E::accumulate_packed_batch::<INITIALIZED>(out_11, s11_buffer, scalars);
        }
        3 => {
            // Optimized base case for 3 variables
            debug_assert!(
                workspace.len() >= 8 * num_points,
                "Workspace for n=3 unrolled case must be >= 8 * num_points, but was only {}",
                workspace.len()
            );

            let (first_row, remainder) = eval_points.split_rows(1);
            let (second_row, third_row) = remainder.split_rows(1);

            // Split workspace for all 8 leaf nodes.
            let (s000_buffer, rest) = workspace.split_at_mut(num_points);
            let (s001_buffer, rest) = rest.split_at_mut(num_points);
            let (s010_buffer, rest) = rest.split_at_mut(num_points);
            let (s011_buffer, rest) = rest.split_at_mut(num_points);
            let (s100_buffer, rest) = rest.split_at_mut(num_points);
            let (s101_buffer, rest) = rest.split_at_mut(num_points);
            let (s110_buffer, s111_buffer) = rest.split_at_mut(num_points);

            // Single loop to compute all 8 leaf scalars.
            for i in 0..num_points {
                let eq_eval = eq_evals[i];
                let z_0 = first_row.values[i];
                let z_1 = second_row.values[i];
                let z_2 = third_row.values[i];

                // Level 1
                let s1 = eq_eval * z_0;
                let s0 = eq_eval - s1;

                // Level 2
                let s01 = s0 * z_1;
                let s11 = s1 * z_1;
                let s00 = s0 - s01;
                let s10 = s1 - s11;

                // Level 3 (leaf nodes)
                let s001 = s00 * z_2;
                let s011 = s01 * z_2;
                let s101 = s10 * z_2;
                let s111 = s11 * z_2;
                s000_buffer[i] = s00 - s001;
                s001_buffer[i] = s001;
                s010_buffer[i] = s01 - s011;
                s011_buffer[i] = s011;
                s100_buffer[i] = s10 - s101;
                s101_buffer[i] = s101;
                s110_buffer[i] = s11 - s111;
                s111_buffer[i] = s111;
            }

            let eighth = out.len() / 8;
            let (out_000, rest) = out.split_at_mut(eighth);
            let (out_001, rest) = rest.split_at_mut(eighth);
            let (out_010, rest) = rest.split_at_mut(eighth);
            let (out_011, rest) = rest.split_at_mut(eighth);
            let (out_100, rest) = rest.split_at_mut(eighth);
            let (out_101, rest) = rest.split_at_mut(eighth);
            let (out_110, out_111) = rest.split_at_mut(eighth);

            E::accumulate_packed_batch::<INITIALIZED>(out_000, s000_buffer, scalars);
            E::accumulate_packed_batch::<INITIALIZED>(out_001, s001_buffer, scalars);
            E::accumulate_packed_batch::<INITIALIZED>(out_010, s010_buffer, scalars);
            E::accumulate_packed_batch::<INITIALIZED>(out_011, s011_buffer, scalars);
            E::accumulate_packed_batch::<INITIALIZED>(out_100, s100_buffer, scalars);
            E::accumulate_packed_batch::<INITIALIZED>(out_101, s101_buffer, scalars);
            E::accumulate_packed_batch::<INITIALIZED>(out_110, s110_buffer, scalars);
            E::accumulate_packed_batch::<INITIALIZED>(out_111, s111_buffer, scalars);
        }
        _ => {
            // General recursive case for any number of variables.
            let (low, high) = out.split_at_mut(out.len() / 2);
            let (first_row, remainder) = eval_points.split_rows(1);

            // Slice the pre-allocated workspace, do not allocate a new one.
            let (s0_buffer, rest_workspace) = workspace.split_at_mut(num_points);
            let (s1_buffer, next_workspace) = rest_workspace.split_at_mut(num_points);

            // Compute new packed scalars for both branches.
            for i in 0..num_points {
                let z_0 = first_row.values[i];
                let eq_eval = eq_evals[i];
                let s1 = eq_eval * z_0;
                let s0 = eq_eval - s1;
                s0_buffer[i] = s0;
                s1_buffer[i] = s1;
            }

            // Recurse, passing down the remainder of the workspace.
            eval_eq_packed_batch::<F, IF, EF, E, INITIALIZED>(
                remainder,
                low,
                s0_buffer,
                scalars,
                next_workspace,
            );
            eval_eq_packed_batch::<F, IF, EF, E, INITIALIZED>(
                remainder,
                high,
                s1_buffer,
                scalars,
                next_workspace,
            );
        }
    }
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
    evals: RowMajorMatrixView<'_, EF>,
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

    /// Naive implementation of equality polynomial evaluation for extension field points.
    ///
    /// This is the core naive implementation used for testing.
    fn eval_eq<F, EF, const INITIALIZED: bool>(eval_point: &[EF], out: &mut [EF], scalar: EF)
    where
        F: Field,
        EF: ExtensionField<F>,
    {
        let num_vars = eval_point.len();
        debug_assert_eq!(out.len(), 1 << num_vars);

        // Evaluate eq(x, z) for all x ∈ {0,1}^n
        // Note: We iterate in reverse order to match the big-endian bit indexing used by the optimized version
        for (x, o) in out.iter_mut().enumerate().take(1 << num_vars) {
            let mut eq_val = scalar;
            for (i, &z_i) in eval_point.iter().enumerate().rev() {
                let x_i = ((x >> (num_vars - 1 - i)) & 1) as u64;
                if x_i == 1 {
                    eq_val *= z_i;
                } else {
                    eq_val *= EF::ONE - z_i;
                }
            }
            if INITIALIZED {
                *o += eq_val;
            } else {
                *o = eq_val;
            }
        }
    }

    /// Naive implementation for base field points.
    ///
    /// Converts base field points to extension field and delegates to eval_eq.
    fn eval_eq_base<F, EF, const INITIALIZED: bool>(eval_point: &[F], out: &mut [EF], scalar: EF)
    where
        F: Field,
        EF: ExtensionField<F>,
    {
        let eval_point_ext: Vec<_> = eval_point.iter().map(|&x| EF::from(x)).collect();
        eval_eq::<F, EF, INITIALIZED>(&eval_point_ext, out, scalar);
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
        let scalars = vec![F::from_u64(2), F::from_u64(3), F::from_u64(5)]; // γ_0=2, γ_1=3, γ_2=5

        let mut output_batch = F::zero_vec(4); // 2^2 = 4 elements
        eval_eq_batch::<_, _, false>(evals, &mut output_batch, &scalars);

        // Compute expected result by evaluating individual equality polynomials and summing
        let mut expected_output = F::zero_vec(4);
        let points = [
            vec![F::from_u64(1), F::from_u64(0)], // Point 1: (1, 0)
            vec![F::from_u64(0), F::from_u64(1)], // Point 2: (0, 1)
            vec![F::from_u64(1), F::from_u64(1)], // Point 3: (1, 1)
        ];
        for (point, &scalar) in points.iter().zip(scalars.iter()) {
            let mut temp_output = F::zero_vec(4);
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
        // 2 rows (variables) × 2 columns (points)
        let evals = RowMajorMatrixView::new(&evals_data, 2);
        let scalars = vec![EF4::from_u64(2), EF4::from_u64(3)];

        let mut output_batch = EF4::zero_vec(4);
        eval_eq_base_batch::<_, _, false>(evals, &mut output_batch, &scalars);

        // Compare with individual evaluations
        let mut expected_output = EF4::zero_vec(4);
        let points = [
            vec![F::from_u64(1), F::from_u64(0)], // Point 1: (1, 0)
            vec![F::from_u64(0), F::from_u64(1)], // Point 2: (0, 1)
        ];
        for (point, &scalar) in points.iter().zip(scalars.iter()) {
            let mut temp_output = EF4::zero_vec(4);
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

            let mut output_batch = EF4::zero_vec(out_len);
            eval_eq_batch::<F, EF4, false>(evals,  &mut output_batch,&scalars,);

            // Compute using individual evaluations and manual summation
            let mut expected_output = EF4::zero_vec(out_len);
            for (point, &scalar) in eval_points.iter().zip(scalars.iter()) {
                let point_ext: Vec<EF4> = point.iter().map(|&f| EF4::from(f)).collect();
                let mut temp_output = EF4::zero_vec(out_len);
                eval_eq::<F, EF4, false>(&point_ext, &mut temp_output, scalar);
                EF4::add_slices(&mut expected_output, &temp_output);
            }

            prop_assert_eq!(output_batch, expected_output);
        }
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
        eval_eq_batch::<F, EF4, false>(evals, &mut output_batch_parallel, &scalars);

        // Verify correctness by comparing against basic batch evaluation
        let mut output_batch_basic = EF4::zero_vec(1 << num_vars);
        let mut workspace = EF4::zero_vec(2 * num_points * num_vars);
        eval_eq_batch_basic::<F, EF4, EF4, false>(
            evals,
            &scalars,
            &mut output_batch_basic,
            &mut workspace,
        );

        assert_eq!(
            output_batch_parallel, output_batch_basic,
            "Parallel batched path should match basic batched evaluation"
        );
    }

    #[test]
    fn test_eval_eq_base_batch_parallel_path() {
        // Calculate the threshold to guarantee the parallel path is taken.
        let packing_width = <F as Field>::Packing::WIDTH;
        let num_threads = current_num_threads().next_power_of_two();
        let log_num_threads = log2_strict_usize(num_threads);
        let threshold = packing_width.ilog2() as usize + 1 + log_num_threads;

        // Use enough variables to exceed the threshold
        let num_vars = threshold + 1;
        let num_points = 3;

        // Create random base field evaluation points and extension field scalars
        let mut rng = SmallRng::seed_from_u64(9876);
        let eval_points: Vec<Vec<F>> = (0..num_points)
            .map(|_| (0..num_vars).map(|_| rng.random()).collect())
            .collect();

        let scalars: Vec<EF4> = (0..num_points).map(|_| rng.random()).collect();

        // Create matrix layout: rows are variables, columns are points
        let mut evals_data = Vec::with_capacity(num_vars * num_points);
        for var_idx in 0..num_vars {
            for point in &eval_points {
                evals_data.push(point[var_idx]);
            }
        }
        let evals = RowMajorMatrixView::new(&evals_data, num_points);
        let out_len = 1 << num_vars;

        // Run the parallel version
        let mut output_parallel = EF4::zero_vec(out_len);
        eval_eq_base_batch::<F, EF4, false>(evals, &mut output_parallel, &scalars);

        // Run the sequential version for comparison
        let mut output_basic = EF4::zero_vec(out_len);
        let mut workspace = EF4::zero_vec(2 * num_points * num_vars);
        eval_eq_batch_basic::<F, F, EF4, false>(evals, &scalars, &mut output_basic, &mut workspace);

        // Assert that the parallel version matches the sequential version
        assert_eq!(
            output_parallel, output_basic,
            "Parallel base-field batched path should match basic batched evaluation"
        );
    }
}
