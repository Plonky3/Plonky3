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

/// Like `batch_multiplicative_inverse`, but writes the result to the given output buffer.
fn batch_multiplicative_inverse_helper<F: Field>(x: &[F], result: &mut [F]) {
    // Higher WIDTH increases instruction-level parallelism, but too high a value will cause us
    // to run out of registers.
    const WIDTH: usize = 4;

    let n = x.len();
    // No need to re-assert this - the caller guarantees it
    debug_assert_eq!(result.len(), n);
    
    // If array length isn't divisible by WIDTH, process as many elements as possible
    // using the packed implementation, then handle the remainder
    if n % WIDTH != 0 {
        let packed_len = n - (n % WIDTH);
        
        if packed_len > 0 {
            let (x_packed, x_remainder) = x.split_at(packed_len);
            let (result_packed, result_remainder) = result.split_at_mut(packed_len);
            
            // Process the packed portion
            let x_packed_arrays = FieldArray::<F, 4>::pack_slice(x_packed);
            let result_packed_arrays = FieldArray::<F, 4>::pack_slice_mut(result_packed);
            batch_multiplicative_inverse_general(x_packed_arrays, result_packed_arrays, |x_packed| x_packed.inverse());
            
            // Process the remainder
            batch_multiplicative_inverse_general(x_remainder, result_remainder, |x| x.inverse());
        } else {
            // Small array, just use the serial implementation
            batch_multiplicative_inverse_general(x, result, |x| x.inverse());
        }
        return;
    }

    // Handle the case where the length is divisible by WIDTH
    let x_packed = FieldArray::<F, 4>::pack_slice(x);
    let result_packed = FieldArray::<F, 4>::pack_slice_mut(result);
    batch_multiplicative_inverse_general(x_packed, result_packed, |x_packed| x_packed.inverse());
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
