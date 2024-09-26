use alloc::vec;
use alloc::vec::Vec;

use p3_maybe_rayon::prelude::*;
use tracing::instrument;

use crate::field::Field;

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
    let mut result = vec![F::zero(); n];

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
    // JN note: WIDTH is 4. The code is specialized to this value and will need
    // modification if it is changed. I tried to make it more generic, but Rust's const
    // generics are not yet good enough.

    let n = x.len();
    assert_eq!(result.len(), n);
    if n < WIDTH {
        return batch_multiplicative_inverse_small(x, result);
    }

    // Buf is reused for a few things to save allocations.
    // Fill buf with cumulative product of x, only taking every 4th value. Concretely, buf will
    // be [
    //   x[0], x[1], x[2], x[3],
    //   x[0] * x[4], x[1] * x[5], x[2] * x[6], x[3] * x[7],
    //   x[0] * x[4] * x[8], x[1] * x[5] * x[9], x[2] * x[6] * x[10], x[3] * x[7] * x[11],
    //   ...
    // ].
    // If n is not a multiple of WIDTH, the result is truncated from the end. For example,
    // for n == 5, we get [x[0], x[1], x[2], x[3], x[0] * x[4]].
    // let mut buf: Vec<F> = Vec::with_capacity(n);
    // cumul_prod holds the last WIDTH elements of buf. This is redundant, but it's how we
    // convince LLVM to keep the values in the registers.
    let mut cumul_prod: [F; WIDTH] = x[..WIDTH].try_into().unwrap();
    result[0..WIDTH].copy_from_slice(&cumul_prod);
    for i in WIDTH..n {
        cumul_prod[i % WIDTH] *= x[i];
        result[i] = cumul_prod[i % WIDTH];
    }

    let mut a_inv = {
        // This is where the four dependency chains meet.
        // Take the last four elements of buf and invert them all.
        let c01 = cumul_prod[0] * cumul_prod[1];
        let c23 = cumul_prod[2] * cumul_prod[3];
        let c0123 = c01 * c23;
        let c0123inv = c0123.inverse();
        let c01inv = c0123inv * c23;
        let c23inv = c0123inv * c01;
        [
            c01inv * cumul_prod[1],
            c01inv * cumul_prod[0],
            c23inv * cumul_prod[3],
            c23inv * cumul_prod[2],
        ]
    };

    for i in (WIDTH..n).rev() {
        // buf[i - WIDTH] has not been written to by this loop, so it equals
        // x[i % WIDTH] * x[i % WIDTH + WIDTH] * ... * x[i - WIDTH].
        result[i] = result[i - WIDTH] * a_inv[i % WIDTH];
        // buf[i] now holds the inverse of x[i].
        a_inv[i % WIDTH] *= x[i];
    }
    for i in (0..WIDTH).rev() {
        result[i] = a_inv[i];
    }

    for (&bi, &xi) in result.iter().zip(x) {
        // Sanity check only.
        debug_assert_eq!(bi * xi, F::one());
    }
}

/// A simple single-threaded implementation of Montgomery's trick, suitable for small n.
fn batch_multiplicative_inverse_small<F: Field>(x: &[F], result: &mut [F]) {
    let n = x.len();
    assert_eq!(result.len(), n);
    if n == 0 {
        return;
    }

    result[0] = F::one();
    for i in 1..n {
        result[i] = result[i - 1] * x[i - 1];
    }

    let product = result[n - 1] * x[n - 1];
    let mut inv = product.inverse();

    for i in (0..n).rev() {
        result[i] *= inv;
        inv *= x[i];
    }
}
