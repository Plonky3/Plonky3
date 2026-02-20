use alloc::vec::Vec;

use p3_maybe_rayon::prelude::*;
use tracing::instrument;

use crate::field::Field;
use crate::{FieldArray, PackedValue, PrimeCharacteristicRing};

/// Batch multiplicative inverses with Montgomery's trick
/// This is Montgomery's trick. At a high level, we invert the product of the given field
/// elements, then derive the individual inverses from that via multiplication.
///
/// The usual Montgomery trick involves calculating an array of cumulative products,
/// resulting in a long dependency chain. To increase instruction-level parallelism, we
/// compute WIDTH separate cumulative product arrays that only meet at the end.
///
/// # Panics
/// This will panic if any of the inputs is zero.
#[instrument(level = "debug", skip_all)]
#[must_use]
pub fn batch_multiplicative_inverse<F: Field>(x: &[F]) -> Vec<F> {
    // How many elements to invert in one thread.
    const CHUNK_SIZE: usize = 1024;

    let n = x.len();
    let mut result = F::zero_vec(n);

    x.par_chunks(CHUNK_SIZE)
        .zip(result.par_chunks_mut(CHUNK_SIZE))
        .for_each(|(x, result)| {
            batch_multiplicative_inverse_helper(x, result);
        });

    result
}

/// Batch compute `scale / x[i]` using Montgomery's trick with one extra multiplication.
/// Equivalent to `batch_multiplicative_inverse(x).iter().map(|inv| scale * inv).collect()`
/// but saves one multiplication per element by folding the scale into the batch inversion.
///
/// # Panics
/// This will panic if any of the inputs is zero.
#[instrument(level = "debug", skip_all)]
#[must_use]
pub fn batch_multiplicative_inverse_scaled<F: Field>(x: &[F], scale: F) -> Vec<F> {
    const CHUNK_SIZE: usize = 1024;

    let n = x.len();
    let mut result = F::zero_vec(n);

    x.par_chunks(CHUNK_SIZE)
        .zip(result.par_chunks_mut(CHUNK_SIZE))
        .for_each(|(x_chunk, result_chunk)| {
            batch_multiplicative_inverse_scaled_helper(x_chunk, result_chunk, scale);
        });

    result
}

/// Like `batch_multiplicative_inverse`, but writes the result to the given output buffer.
fn batch_multiplicative_inverse_helper<F: Field>(x: &[F], result: &mut [F]) {
    // Higher WIDTH increases instruction-level parallelism, but too high a value will cause us
    // to run out of registers.
    const WIDTH: usize = 4;

    let n = x.len();
    assert_eq!(result.len(), n);
    if !n.is_multiple_of(WIDTH) {
        // There isn't a very clean way to do this with FieldArray; for now just do it in serial.
        // Another simple (though suboptimal) workaround would be to make two separate calls, one
        // for the packed part and one for the remainder.
        return batch_multiplicative_inverse_general(x, result, |x| x.inverse());
    }

    let x_packed = FieldArray::<F, 4>::pack_slice(x);
    let result_packed = FieldArray::<F, 4>::pack_slice_mut(result);

    batch_multiplicative_inverse_general(x_packed, result_packed, |x_packed| x_packed.inverse());
}

fn batch_multiplicative_inverse_scaled_helper<F: Field>(x: &[F], result: &mut [F], scale: F) {
    batch_multiplicative_inverse_general_scaled(x, result, scale, |x| x.inverse());
}

/// A simple single-threaded implementation of Montgomery's trick. Since not all `PrimeCharacteristicRing`s
/// support inversion, this takes a custom inversion function.
pub(crate) fn batch_multiplicative_inverse_general<F, Inv>(x: &[F], result: &mut [F], inv: Inv)
where
    F: PrimeCharacteristicRing + Copy,
    Inv: Fn(F) -> F,
{
    let n = x.len();
    assert_eq!(result.len(), n);
    if n == 0 {
        return;
    }

    result[0] = F::ONE;
    for i in 1..n {
        result[i] = result[i - 1] * x[i - 1];
    }

    let product = result[n - 1] * x[n - 1];
    let mut inv = inv(product);

    for i in (0..n).rev() {
        result[i] *= inv;
        inv *= x[i];
    }
}

/// Like `batch_multiplicative_inverse_general` but outputs `scale / x[i]` instead of `1 / x[i]`,
/// using one extra multiplication total by folding the scale into the product inverse.
pub(crate) fn batch_multiplicative_inverse_general_scaled<F, Inv>(
    x: &[F],
    result: &mut [F],
    scale: F,
    inv: Inv,
) where
    F: PrimeCharacteristicRing + Copy,
    Inv: Fn(F) -> F,
{
    let n = x.len();
    assert_eq!(result.len(), n);
    if n == 0 {
        return;
    }

    result[0] = F::ONE;
    for i in 1..n {
        result[i] = result[i - 1] * x[i - 1];
    }

    let product = result[n - 1] * x[n - 1];
    let mut inv = inv(product) * scale;

    for i in (0..n).rev() {
        result[i] *= inv;
        inv *= x[i];
    }
}
