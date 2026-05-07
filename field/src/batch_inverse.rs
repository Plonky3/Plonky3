use alloc::vec::Vec;

use p3_maybe_rayon::prelude::*;
use tracing::instrument;

use crate::field::Field;
use crate::{FieldArray, PackedValue, PrimeCharacteristicRing};

/// Compute the multiplicative inverse of every element in a slice via Montgomery's trick.
///
/// Replaces `n` field inversions with one inversion plus `~3n` multiplications:
/// - forward pass: build prefix products of the inputs,
/// - one inversion of the full product,
/// - reverse pass: derive each individual inverse from the prefix products.
///
/// The forward pass is a long dependency chain. It is parallelised on two axes:
/// - 4-lane packed arrays — four independent chains run side by side,
/// - 1024-element chunks — dispatched across Rayon workers.
///
/// Lengths not a multiple of 4 finish with a scalar pass on the trailing 1..=3 elements.
///
/// # Panics
///
/// Panics if any input is zero.
#[instrument(level = "debug", skip_all)]
#[must_use]
pub fn batch_multiplicative_inverse<F: Field>(x: &[F]) -> Vec<F> {
    // 1024-element chunks per Rayon task.
    //
    // Why 1024:
    //   - amortizes the one field inversion per chunk over many multiplies,
    //   - leaves enough chunks for work-stealing on long slices.
    const CHUNK_SIZE: usize = 1024;

    // 4-lane packing.
    //
    // Why 4:
    //   - smallest packed-field width on every backend,
    //   - wider lanes risk register spills in the per-lane dependency chains.
    const WIDTH: usize = 4;

    // Pre-allocate the output: each Rayon task writes a disjoint sub-slice.
    let mut result = F::zero_vec(x.len());

    x.par_chunks(CHUNK_SIZE)
        .zip(result.par_chunks_mut(CHUNK_SIZE))
        .for_each(|(x_chunk, result_chunk)| {
            // Phase 1 — split the chunk:
            //   - packed: 4-aligned prefix viewed as 4-lane arrays,
            //   - tail:   0..=3 trailing scalars (m = n - n%4).
            //
            //     x_chunk:  [ x_0 .. x_{m-1} | x_m .. x_{n-1} ]
            //               └──── packed ────┘└──── tail ────┘
            let (x_packed, x_tail) = FieldArray::<F, WIDTH>::pack_slice_with_suffix(x_chunk);
            let (result_packed, result_tail) =
                FieldArray::<F, WIDTH>::pack_slice_with_suffix_mut(result_chunk);

            // Phase 2 — packed pass: 4 independent Montgomery chains, one per lane.
            //
            // Final inversion lands on a 4-lane array → one scalar inversion per chunk.
            batch_multiplicative_inverse_general(x_packed, result_packed, |y| y.inverse());

            // Phase 3 — tail pass: 0..=3 leftover scalars.
            //
            // Empty when n % 4 == 0; this call then returns immediately.
            batch_multiplicative_inverse_general(x_tail, result_tail, |y| y.inverse());
        });

    result
}

/// A simple single-threaded implementation of Montgomery's trick. Since not all `PrimeCharacteristicRing`s
/// support inversion, this takes a custom inversion function.
///
/// Unlike [`batch_multiplicative_inverse`], this writes into a caller-provided buffer,
/// avoiding heap allocation. This makes it suitable for small, fixed-size inputs
/// such as packed field lanes.
#[inline]
pub fn batch_multiplicative_inverse_general<F, Inv>(x: &[F], result: &mut [F], inv: Inv)
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
