use p3_field::{
    Algebra, ExtensionField, Field, PackedFieldExtension, PackedValue, PrimeCharacteristicRing,
};
use p3_maybe_rayon::prelude::*;
use p3_util::{iter_array_chunks_padded, log2_strict_usize};

/// Log of number of threads to spawn.
/// Long term this should be a modifiable parameter and potentially be in an optimization file somewhere.
/// I've chosen 32 here as my machine has 20 logical cores.
#[cfg(feature = "parallel")]
const LOG_NUM_THREADS: usize = 5;
#[cfg(not(feature = "parallel"))]
/// If the parallel feature is not enabled, we default to a single thread.
const LOG_NUM_THREADS: usize = 0;

/// The number of threads to spawn for parallel computations.
const NUM_THREADS: usize = 1 << LOG_NUM_THREADS;

/// Computes the multilinear equality polynomial `eq(x, z)` over all $x ∈ \{0,1\}^n$ using an optimized method.
///
/// Given a constraint point `z ∈ EF^n` and a scalar `α`, this function evaluates the equality polynomial:
///
/// \begin{equation}
/// \mathrm{eq}(x, z) = \prod_{i=0}^{n-1} \left( x_i z_i + (1 - x_i)(1 - z_i) \right)
/// \end{equation}
///
/// for all binary vectors $x ∈ \{0,1\}^n$, and stores the scaled results into the `out` buffer.
///
/// # Output Structure
/// The `out` buffer must have length exactly $2^n$, where $n = \texttt{eval.len()}$.
///
/// Each index `i` in `out` corresponds to the binary vector $x$ given by the **big-endian** bit decomposition of `i`.
/// That is:
/// - `out[0]` corresponds to $x = (0, 0, ..., 0)$
/// - `out[1]` corresponds to $x = (0, 0, ..., 1)$
/// - ...
/// - `out[2^n - 1]` corresponds to $x = (1, 1, ..., 1)$
///
/// # Behavior of `INITIALIZED`
/// If `INITIALIZED = false`, each value in `out` is overwritten with the computed result.
/// If `INITIALIZED = true`, the computed result is added to the existing value in `out`.
///
/// # Arguments
/// - `eval`: Constraint point `z ∈ EF^n`
/// - `out`: Mutable slice of `EF` of size `2^n`
/// - `scalar`: Scalar multiplier α ∈ `EF`
#[inline]
pub fn eval_eq<F, EF, const INITIALIZED: bool>(eval: &[EF], out: &mut [EF], scalar: EF)
where
    F: Field,
    EF: ExtensionField<F>,
{
    // We assume that `F::Packing::WIDTH` is a power of 2.
    let packing_width = F::Packing::WIDTH;
    // debug_assert!(packing_width > 1);

    // Ensure that the output buffer size is correct:
    // It should be of size `2^n`, where `n` is the number of variables.
    debug_assert_eq!(out.len(), 1 << eval.len());

    // If the number of variables is small, there is no need to use
    // parallelization or packings.
    if eval.len() <= packing_width + 1 + LOG_NUM_THREADS {
        // A basic recursive approach.
        eval_eq_basic::<_, _, _, INITIALIZED>(eval, out, scalar);
    } else {
        let log_packing_width = log2_strict_usize(packing_width);
        let eval_len_min_packing = eval.len() - log_packing_width;

        // We split eval into three parts:
        // - eval[..LOG_NUM_THREADS] (the first LOG_NUM_THREADS elements)
        // - eval[LOG_NUM_THREADS..eval_len_min_packing] (the middle elements)
        // - eval[eval_len_min_packing..] (the last log_packing_width elements)

        // The middle elements are the ones which will be computed in parallel.
        // The last log_packing_width elements are the ones which will be packed.

        // We make a buffer of elements of size `NUM_THREADS`.
        let mut parallel_buffer = EF::ExtensionPacking::zero_vec(NUM_THREADS);
        let out_chunk_size = out.len() / NUM_THREADS;

        // Compute the equality polynomial corresponding to the last log_packing_width elements
        // and pack these.
        parallel_buffer[0] = packed_eq_poly(&eval[eval_len_min_packing..], scalar);

        // Update the buffer so it contains the evaluations of the equality polynomial
        // with respect to parts one and three.
        fill_buffer(eval[..LOG_NUM_THREADS].iter().rev(), &mut parallel_buffer);

        // Finally do all computations involving the middle elements.
        out.par_chunks_exact_mut(out_chunk_size)
            .zip(parallel_buffer.par_iter())
            .for_each(|(out_chunk, buffer_val)| {
                eval_eq_packed::<_, _, INITIALIZED>(
                    &eval[LOG_NUM_THREADS..(eval.len() - log_packing_width)],
                    out_chunk,
                    *buffer_val,
                );
            });
    }
}

/// Computes the multilinear equality polynomial `eq(x, z)` over all $x ∈ \{0,1\}^n$
/// using a base field point `z ∈ F^n`.
///
/// This function is a base-field optimized version of [`eval_eq`] and is more efficient
/// when `z` lies in the base field `F` rather than the extension field.
///
/// It evaluates the equality polynomial:
///
/// \begin{equation}
/// \mathrm{eq}(x, z) = \prod_{i=0}^{n-1} \left( x_i z_i + (1 - x_i)(1 - z_i) \right)
/// \end{equation}
///
/// and stores the scaled results into the `out` buffer.
///
/// # Output Structure
/// The `out` buffer must have length exactly $2^n$, where $n = \texttt{eval.len()}$.
///
/// Each index `i` corresponds to $x$ given by the **big-endian** bit decomposition of `i`.
///
/// # Behavior of `INITIALIZED`
/// If `INITIALIZED = false`, each value in `out` is overwritten with the computed result.
/// If `INITIALIZED = true`, the computed result is added to the existing value in `out`.
///
/// # Arguments
/// - `eval`: Constraint point `z ∈ F^n`
/// - `out`: Mutable slice of `EF` of size `2^n`
/// - `scalar`: Scalar multiplier α ∈ `EF`
#[inline]
pub fn eval_eq_base<F, EF, const INITIALIZED: bool>(eval: &[F], out: &mut [EF], scalar: EF)
where
    F: Field,
    EF: ExtensionField<F>,
{
    // we assume that packing_width is a power of 2.
    let packing_width = F::Packing::WIDTH;

    // Ensure that the output buffer size is correct:
    // It should be of size `2^n`, where `n` is the number of variables.
    debug_assert_eq!(out.len(), 1 << eval.len());

    // If the number of variables is small, there is no need to use
    // parallelization or packings.
    if eval.len() <= packing_width + 1 + LOG_NUM_THREADS {
        // A basic recursive approach.
        eval_eq_basic::<_, _, _, INITIALIZED>(eval, out, scalar);
    } else {
        let log_packing_width = log2_strict_usize(packing_width);
        let eval_len_min_packing = eval.len() - log_packing_width;

        // We split eval into three parts:
        // - eval[..LOG_NUM_THREADS] (the first LOG_NUM_THREADS elements)
        // - eval[LOG_NUM_THREADS..eval_len_min_packing] (the middle elements)
        // - eval[eval_len_min_packing..] (the last log_packing_width elements)

        // The middle elements are the ones which will be computed in parallel.
        // The last log_packing_width elements are the ones which will be packed.

        // We make a buffer of PackedField elements of size `NUM_THREADS`.
        // Note that this is a slightly different strategy to `eval_eq` which instead
        // uses PackedExtensionField elements. Whilst this involves slightly more mathematical
        // operations, it seems to be faster in practice due to less data moving around.
        let mut parallel_buffer = F::Packing::zero_vec(NUM_THREADS);
        let out_chunk_size = out.len() / NUM_THREADS;

        // Compute the equality polynomial corresponding to the last log_packing_width elements
        // and pack these.
        parallel_buffer[0] = packed_eq_poly(&eval[eval_len_min_packing..], F::ONE);

        // Update the buffer so it contains the evaluations of the equality polynomial
        // with respect to parts one and three.
        fill_buffer(eval[..LOG_NUM_THREADS].iter().rev(), &mut parallel_buffer);

        // Finally do all computations involving the middle elements.
        out.par_chunks_exact_mut(out_chunk_size)
            .zip(parallel_buffer.par_iter())
            .for_each(|(out_chunk, buffer_val)| {
                base_eval_eq_packed::<_, _, INITIALIZED>(
                    &eval[LOG_NUM_THREADS..(eval.len() - log_packing_width)],
                    out_chunk,
                    *buffer_val,
                    scalar,
                );
            });
    }
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

/// Compute the scaled multilinear equality polynomial over `{0,1}` for a single variable.
///
/// # Arguments
/// - `eval`: Slice containing the evaluation point `[z_0]` (must have length 1).
/// - `scalar`: A field element `α ∈ 𝔽` used to scale the result.
///
/// # Returns
/// An array `[α ⋅ (1 - z_0), α ⋅ z_0]` representing the scaled evaluations
/// of `eq(x, z)` for `x ∈ {0,1}`.
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

/// Compute the scaled multilinear equality polynomial over `{0,1}²`.
///
/// # Arguments
/// - `eval`: Slice containing the evaluation points `[z_0, z_1]` (must have length 2).
/// - `scalar`: A field element `α ∈ 𝔽` used to scale the result.
///
/// # Returns
/// An array `[α ⋅ eq((0,0), z), α ⋅ eq((0,1), z), α ⋅ eq((1,0), z), α ⋅ eq((1,1), z)]`
/// representing the scaled evaluations of `eq(x, z)` for `x ∈ {0,1}²`.
#[inline(always)]
fn eval_eq_2<F, FP>(eval: &[F], scalar: FP) -> [FP; 4]
where
    F: Field,
    FP: Algebra<F>,
{
    assert_eq!(eval.len(), 2);

    // Extract z_0, z_1 from the evaluation point
    let z_0 = eval[0];
    let z_1 = eval[1];

    // Compute s1 = α ⋅ z_0 = α ⋅ eq(1, -) and s0 = α - s1 = α ⋅ (1 - z_0) = α ⋅ eq(0, -)
    let s1 = scalar.clone() * z_0;
    let s0 = scalar - s1.clone();

    // For x_0 = 0:
    // - s01 = s0 ⋅ z_1 = α ⋅ (1 - z_0) ⋅ z_1 = α ⋅ eq((0,1), z)
    // - s00 = s0 - s01 = α ⋅ (1 - z_0)(1 - z_1) = α ⋅ eq((0,0), z)
    let s01 = s0.clone() * z_1;
    let s00 = s0 - s01.clone();

    // For x_0 = 1:
    // - s11 = s1 ⋅ z_1 = α ⋅ z_0 ⋅ z_1 = α ⋅ eq((1,1), z)
    // - s10 = s1 - s11 = α ⋅ z_0(1 - z_1) = α ⋅ eq((1,0), z)
    let s11 = s1.clone() * z_1;
    let s10 = s1 - s11.clone();

    // Return values in lexicographic order of x = (x_0, x_1)
    [s00, s01, s10, s11]
}

/// Compute the scaled multilinear equality polynomial over `{0,1}³`.
///
/// # Arguments
/// - `eval`: Slice containing the evaluation points `[z_0, z_1, z_2]` (must have length 3).
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

    // Extract z_0, z_1, z_2 from the evaluation point
    let z_0 = eval[0];
    let z_1 = eval[1];
    let z_2 = eval[2];

    // First dimension split: scalar * z_0 and scalar * (1 - z_0)
    let s1 = scalar.clone() * z_0; // α ⋅ z_0
    let s0 = scalar - s1.clone(); // α ⋅ (1 - z_0)

    // Second dimension split:
    // Group (0, x1) branch using s0 = α ⋅ (1 - z_0)
    let s01 = s0.clone() * z_1; // α ⋅ (1 - z_0) ⋅ z_1
    let s00 = s0 - s01.clone(); // α ⋅ (1 - z_0) ⋅ (1 - z_1)

    // Group (1, x1) branch using s1 = α ⋅ z_0
    let s11 = s1.clone() * z_1; // α ⋅ z_0 ⋅ z_1
    let s10 = s1 - s11.clone(); // α ⋅ z_0 ⋅ (1 - z_1)

    // Third dimension split:
    // For (0,0,x2) branch
    let s001 = s00.clone() * z_2; // α ⋅ (1 - z_0)(1 - z_1) ⋅ z_2
    let s000 = s00 - s001.clone(); // α ⋅ (1 - z_0)(1 - z_1) ⋅ (1 - z_2)

    // For (0,1,x2) branch
    let s011 = s01.clone() * z_2; // α ⋅ (1 - z_0) ⋅ z_1 ⋅ z_2
    let s010 = s01 - s011.clone(); // α ⋅ (1 - z_0) ⋅ z_1 ⋅ (1 - z_2)

    // For (1,0,x2) branch
    let s101 = s10.clone() * z_2; // α ⋅ z_0 ⋅ (1 - z_1) ⋅ z_2
    let s100 = s10 - s101.clone(); // α ⋅ z_0 ⋅ (1 - z_1) ⋅ (1 - z_2)

    // For (1,1,x2) branch
    let s111 = s11.clone() * z_2; // α ⋅ z_0 ⋅ z_1 ⋅ z_2
    let s110 = s11 - s111.clone(); // α ⋅ z_0 ⋅ z_1 ⋅ (1 - z_2)

    // Return all 8 evaluations in lexicographic order of x ∈ {0,1}³
    [s000, s001, s010, s011, s100, s101, s110, s111]
}

/// Computes the multilinear equality polynomial `eq(x, z)` over all $x ∈ \{0,1\}^n$ via a recursive algorithm.
///
/// # Output Structure
/// The `out` buffer must have length exactly $2^n$, where $n = \texttt{eval.len()}$.
///
/// Each index `i` corresponds to $x$ given by the **big-endian** bit decomposition of `i`.
///
/// # Behavior of `INITIALIZED`
/// If `INITIALIZED = false`, each value in `out` is overwritten with the computed result.
/// If `INITIALIZED = true`, the computed result is added to the existing value in `out`.
///
/// # Arguments
/// - `eval`: Constraint point `z ∈ IF^n`
/// - `out`: Mutable slice of `EF` of size `2^n`
/// - `scalar`: Scalar multiplier α ∈ `EF`
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

            eval_eq_basic::<_, _, _, INITIALIZED>(tail, low, s0);
            eval_eq_basic::<_, _, _, INITIALIZED>(tail, high, s1);
        }
    }
}

/// Computes the equality polynomial evaluations via a simple recursive algorithm.
///
/// Unlike [`eval_eq_basic`], this function makes heavy use of packed values to speed up computations.
/// In particular `scalar` should be passed in as a packed value coming from [`packed_eq_poly`].
///
/// Essentially using packings this functions computes
///
/// ```text
/// eq(X) = scalar[j] * ∏ (1 - X_i + 2X_i z_i)
/// ```
///
/// for a collection of `i` at the same time. Here `scalar[j]` should be thought of as evaluations of an equality
/// polynomial over different variables so `eq(X)` ends up being the evaluation of the equality polynomial over
/// the combined set of variables.
///
/// It then updates the output buffer `out` with the computed values by adding them in.
#[inline]
fn eval_eq_packed<F, EF, const INITIALIZED: bool>(
    eval: &[EF],
    out: &mut [EF],
    scalar: EF::ExtensionPacking,
) where
    F: Field,
    EF: ExtensionField<F>,
{
    // Ensure that the output buffer size is correct:
    // It should be of size `2^n`, where `n` is the number of variables.
    let width = F::Packing::WIDTH;
    debug_assert_eq!(out.len(), width << eval.len());

    match eval.len() {
        0 => {
            let result: Vec<EF> = EF::ExtensionPacking::to_ext_iter([scalar]).collect();
            add_or_set::<_, INITIALIZED>(out, &result);
        }
        1 => {
            // Manually unroll for single variable case
            let eq_evaluations = eval_eq_1(eval, scalar);
            let result: Vec<EF> = EF::ExtensionPacking::to_ext_iter(eq_evaluations).collect();
            add_or_set::<_, INITIALIZED>(out, &result);
        }
        2 => {
            // Manually unroll for two variables case
            let eq_evaluations = eval_eq_2(eval, scalar);
            let result: Vec<EF> = EF::ExtensionPacking::to_ext_iter(eq_evaluations).collect();
            add_or_set::<_, INITIALIZED>(out, &result);
        }
        3 => {
            const EVAL_LEN: usize = 8;

            // Manually unroll for three variable case
            let eq_evaluations = eval_eq_3(eval, scalar);

            // Unpack the evaluations back into EF elements and add to output.
            // We use `iter_array_chunks_padded` to allow us to use `add_slices` without
            // needing a vector allocation. Note that `eq_evaluations: [EF::ExtensionPacking: 8]`
            // so we know that `out.len() = 8 * F::Packing::WIDTH` meaning we can use `chunks_exact_mut`
            // and `iter_array_chunks_padded` will never actually pad anything.
            // This avoids the allocation used to accumulate `result` in the other branches. We could
            // do a similar strategy in those branches but, those branches should only be hit
            // infrequently in small cases which are already sufficiently fast.
            iter_array_chunks_padded::<_, EVAL_LEN>(
                EF::ExtensionPacking::to_ext_iter(eq_evaluations),
                EF::ZERO,
            )
            .zip(out.chunks_exact_mut(EVAL_LEN))
            .for_each(|(res, out_chunk)| {
                if INITIALIZED {
                    EF::add_slices(out_chunk, &res);
                } else {
                    out_chunk.copy_from_slice(&res);
                }
            });
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

            eval_eq_packed::<_, _, INITIALIZED>(tail, low, s0);
            eval_eq_packed::<_, _, INITIALIZED>(tail, high, s1);
        }
    }
}

/// Computes the equality polynomial evaluations via a simple recursive algorithm.
///
/// Unlike [`eval_eq_basic`], this function makes heavy use of packed values to speed up computations.
/// In particular `scalar` should be passed in as a packed value coming from [`packed_eq_poly`].
///
/// Essentially using packings this functions computes
///
/// ```text
/// eq(X) = scalar[j] * ∏ (1 - X_i + 2X_i z_i)
/// ```
///
/// for a collection of `i` at the same time. Here `scalar[j]` should be thought of as evaluations of an equality
/// polynomial over different variables so `eq(X)` ends up being the evaluation of the equality polynomial over
/// the combined set of variables.
///
/// It then updates the output buffer `out` with the computed values by adding them in.
#[inline]
fn base_eval_eq_packed<F, EF, const INITIALIZED: bool>(
    eval_points: &[F],
    out: &mut [EF],
    eq_evals: F::Packing,
    scalar: EF,
) where
    F: Field,
    EF: ExtensionField<F>,
{
    // Ensure that the output buffer size is correct:
    // It should be of size `2^n`, where `n` is the number of variables.
    let width = F::Packing::WIDTH;
    debug_assert_eq!(out.len(), width << eval_points.len());

    match eval_points.len() {
        0 => {
            scale_and_add::<_, _, INITIALIZED>(out, eq_evals.as_slice(), scalar);
        }
        1 => {
            let eq_evaluations = eval_eq_1(eval_points, eq_evals);
            scale_and_add::<_, _, INITIALIZED>(
                out,
                F::Packing::unpack_slice(&eq_evaluations),
                scalar,
            );
        }
        2 => {
            let eq_evaluations = eval_eq_2(eval_points, eq_evals);
            scale_and_add::<_, _, INITIALIZED>(
                out,
                F::Packing::unpack_slice(&eq_evaluations),
                scalar,
            );
        }
        3 => {
            let eq_evaluations = eval_eq_3(eval_points, eq_evals);
            scale_and_add::<_, _, INITIALIZED>(
                out,
                F::Packing::unpack_slice(&eq_evaluations),
                scalar,
            );
        }
        _ => {
            let (&x, tail) = eval_points.split_first().unwrap();

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
            let s1 = eq_evals * x; // Contribution when `X_i = 1`
            let s0 = eq_evals - s1; // Contribution when `X_i = 0`

            base_eval_eq_packed::<_, _, INITIALIZED>(tail, low, s0, scalar);
            base_eval_eq_packed::<_, _, INITIALIZED>(tail, high, s1, scalar);
        }
    }
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
    // TODO: We can probably add a custom method to Plonky3 to handle this more efficiently (and use packings).
    // This approach is faster than collecting `scalar * eq_eval` into a vector and using `add_slices`. Presumably
    // this is because we avoid the allocation.
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

/// Computes equality polynomial evaluations and packs them into a `PackedFieldExtension`.
///
/// Note that when `F = EF` is a PrimeField, `EF::ExtensionPacking = F::Packing` so this can
/// also be used to compute initial packed evaluations of the equality polynomial over base
/// field elements (instead of extension field elements).
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
}
