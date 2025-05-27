use alloc::vec::Vec;
use core::cell::RefCell;
use core::iter;

use p3_field::{Field, PackedFieldPow2, TwoAdicField};
use p3_matrix::Matrix;
use p3_matrix::dense::RowMajorMatrix;
use p3_matrix::util::reverse_matrix_index_bits;
use p3_maybe_rayon::prelude::*;
use p3_util::{log2_strict_usize, reverse_slice_index_bits};

use crate::{Butterfly, DitButterfly, TwiddleFreeButterfly, TwoAdicSubgroupDft};

/// Computes the optimal workload size for `T` to fit in L1 cache (32 KB).
///
/// Ensures efficient memory access by dividing the cache size by `T`'s size.
/// The result represents how many elements of `T` can be processed per thread.
///
/// Helps minimize cache misses and improve performance in parallel workloads.
#[must_use]
const fn workload_size<T: Sized>() -> usize {
    const L1_CACHE_SIZE: usize = 1 << 15; // 32 KB
    L1_CACHE_SIZE / size_of::<T>()
}

/// A FFT algorithm which divides a butterfly network's layers into two halves.
///
/// Unlike other FFT algorithms, this algorithm is optimized for small batch sizes.
/// Hence it does not do any parallelization.
///
/// For the first half, we apply a butterfly network with smaller blocks in earlier layers,
/// i.e. either DIT or Bowers G. Then we bit-reverse, and for the second half, we continue executing
/// the same network but in bit-reversed order. This way we're always working with small blocks,
/// so within each half, we can have a certain amount of parallelism with no cross-thread
/// communication.
#[derive(Default, Clone, Debug)]
pub struct Radix2DitSmallBatch<F: Field> {
    /// Memoized twiddle factors for each length log_n.
    ///
    /// twiddles are stored in reverse order so twiddle[i]
    /// is a vector of length 2^{i + 1} designed to be used in the round
    /// of size 2^{i + 2}. E.g. twiddles[0] = vec![1, i] and will be
    /// used in the round of size 4. Twiddles are not stored
    /// for the final round of size 2, as the only twiddle is 1.
    ///
    /// TODO: The use of RefCell means this can't be shared across
    /// threads; consider using RwLock or finding a better design
    /// instead.
    twiddles: RefCell<Vec<Vec<F>>>,

    /// Memoized inverse twiddle factors for each length log_n.
    inv_twiddles: RefCell<Vec<Vec<F>>>,
}

impl<F: TwoAdicField> Radix2DitSmallBatch<F> {
    pub fn new(n: usize) -> Self {
        let res = Self {
            twiddles: RefCell::default(),
            inv_twiddles: RefCell::default(),
        };
        res.update_twiddles(n);
        res
    }

    /// Given a field element `gen` of order n where `n = 2^lg_n`,
    /// return a vector of vectors `table` where table[i] is the
    /// vector of twiddle factors for an fft of length n/2^i. The
    /// values g_i^k for k >= i/2 are skipped as these are just the
    /// negatives of the other roots (using g_i^{i/2} = -1). The
    /// value gen^0 = 1 is included to aid consistency between the
    /// packed and non-packed variants.
    fn roots_of_unity_table(&self, n: usize) -> Vec<Vec<F>> {
        let lg_n = log2_strict_usize(n);
        let generator = F::two_adic_generator(lg_n);
        let half_n = 1 << (lg_n - 1);
        // nth_roots = [1, g, g^2, g^3, ..., g^{n/2 - 1}]
        let nth_roots: Vec<_> = generator.powers().take(half_n).collect();

        (0..lg_n)
            .map(|i| {
                let mut twiddles: Vec<F> = nth_roots.iter().step_by(1 << i).copied().collect();
                reverse_slice_index_bits(&mut twiddles);
                twiddles
            })
            .collect()
    }

    /// Compute twiddle factors, or take memoized ones if already available.
    fn update_twiddles(&self, fft_len: usize) {
        // TODO: This recomputes the entire table from scratch if we
        // need it to be bigger, which is wasteful.

        // roots_of_unity_table(fft_len) returns a vector of twiddles of length log_2(fft_len).
        let curr_max_fft_len = 1 << self.twiddles.borrow().len();
        if fft_len > curr_max_fft_len {
            let new_twiddles = self.roots_of_unity_table(fft_len);
            // We can obtain the inverse twiddles by reversing and
            // negating the twiddles.
            let new_inv_twiddles = new_twiddles
                .iter()
                .map(|ts| {
                    // The first twiddle is still one, we reverse and negate the rest...
                    iter::once(F::ONE)
                        .chain(ts[1..].iter().rev().map(|&t| -t))
                        .collect()
                })
                .collect();
            self.twiddles.replace(new_twiddles);
            self.inv_twiddles.replace(new_inv_twiddles);
        }
    }
}

impl<F> TwoAdicSubgroupDft<F> for Radix2DitSmallBatch<F>
where
    F: TwoAdicField,
    F::Packing: PackedFieldPow2,
{
    type Evaluations = RowMajorMatrix<F>;

    fn dft_batch(&self, mut mat: RowMajorMatrix<F>) -> RowMajorMatrix<F> {
        let h = mat.height();
        let log_h = log2_strict_usize(h);

        self.update_twiddles(h);
        let root_table = self.twiddles.borrow();

        let half_lg_h = log_h / 2;
        let rough_sqrt_height = 1 << half_lg_h;

        // Choose the number of rows to process in parallel per chunk
        let num_par_rows = (workload_size::<F>() / mat.width())
            .next_power_of_two()
            .max(rough_sqrt_height)
            .min(h);

        // let num_par_rows = rough_sqrt_height;
        let log_num_par_rows = log2_strict_usize(num_par_rows);

        let chunk_size = num_par_rows * mat.width();

        // dit_layer_0_par(&mut mat.values);

        // DIT butterfly
        for layer in 0..(log_h - log_num_par_rows) {
            // Note that the length of `root_table[log_h - layer - 1]` is `1 << layer`.
            dit_layer_par(&mut mat.values, &root_table[log_h - layer - 1]);
        }

        // Once the blocks are small enough, we can split the matrix
        // into chunks of size `chunk_size` and process them in parallel.
        // This avoids passing data between threads, which can be expensive.
        par_remaining_layers(
            &mut mat.values,
            chunk_size,
            &root_table,
            log_h - log_num_par_rows,
            log_h,
        );

        reverse_matrix_index_bits(&mut mat);
        mat
    }
}

/// Applies one layer of the Radix-2 DIT FFT butterfly network.
///
/// Splits the matrix into blocks of rows and performs in-place butterfly operations
/// on each block. Uses a `TwiddleFreeButterfly` for the first pair and `DitButterfly`
/// with precomputed twiddles for the rest.
///
/// # Arguments
/// - `vec`: Mutable vector whose height is a power of two.
/// - `twiddles`: Precomputed twiddle factors for this layer.
#[inline]
fn dit_layer_par<F: Field>(vec: &mut [F], twiddles: &[F]) {
    debug_assert_eq!(
        vec.len() % twiddles.len(),
        0,
        "Vector length must be divisible by the number of twiddles"
    );
    let size = vec.len();
    let num_blocks = twiddles.len();

    let outer_block_size = size / num_blocks;
    let half_outer_block_size = outer_block_size / 2;

    vec.par_chunks_exact_mut(outer_block_size)
        .enumerate()
        .for_each(|(ind, block)| {
            // Split each block vertically into top (hi) and bottom (lo) halves
            let (hi_chunk, lo_chunk) = block.split_at_mut(half_outer_block_size);

            // If num_blocks is small, we probably are not using all available threads.
            // This is an estimate of the optimal block size to make use of all threads.
            // That being said, this is a bit of a magic number which I determined through
            // experimentation. It is likely that different hardware might work better with
            // a slightly different value.
            const NUM_THREADS: usize = 32;
            let inner_block_size = size / (2 * num_blocks).max(NUM_THREADS);

            hi_chunk
                .par_chunks_exact_mut(inner_block_size)
                .zip(lo_chunk.par_chunks_exact_mut(inner_block_size))
                .for_each(|(hi_chunk, lo_chunk)| {
                    if ind == 0 {
                        // The first pair doesn't require a twiddle factor
                        TwiddleFreeButterfly.apply_to_rows(hi_chunk, lo_chunk);
                    } else {
                        // Apply DIT butterfly using the twiddle factor at index `ind - 1`
                        DitButterfly(twiddles[ind]).apply_to_rows(hi_chunk, lo_chunk);
                    }
                });
        });
}

#[inline]
fn par_remaining_layers<F: Field>(
    vec: &mut [F],
    chunk_size: usize,
    root_table: &[Vec<F>],
    log_num_par_rows: usize,
    log_h: usize,
) {
    vec.par_chunks_exact_mut(chunk_size)
        .enumerate()
        .for_each(|(index, chunk)| {
            for layer in log_num_par_rows..log_h {
                let num_twiddles_per_block = 1 << (layer - log_num_par_rows);
                dit_layer(
                    chunk,
                    &(root_table[log_h - layer - 1]
                        [(index * num_twiddles_per_block)..((index + 1) * num_twiddles_per_block)]),
                );
            }
        });
}

/// Applies one layer of the Radix-2 DIT FFT butterfly network.
///
/// Splits the matrix into blocks of rows and performs in-place butterfly operations
/// on each block. Uses a `TwiddleFreeButterfly` for the first pair and `DitButterfly`
/// with precomputed twiddles for the rest.
///
/// # Arguments
/// - `mat`: Mutable matrix view with height as a power of two.
/// - `twiddles`: Precomputed twiddle factors for this layer.
#[inline]
fn dit_layer<F: Field>(vec: &mut [F], twiddles: &[F]) {
    let size = vec.len();
    let num_blocks = twiddles.len();

    let block_size = size / num_blocks;
    let half_block_size = block_size / 2;

    vec.chunks_exact_mut(block_size)
        .zip(twiddles.iter())
        .for_each(|(block, &twiddle)| {
            // Split each block vertically into top (hi) and bottom (lo) halves
            let (hi_chunk, lo_chunk) = block.split_at_mut(half_block_size);

            // Apply DIT butterfly using the twiddle factor at index `ind - 1`
            DitButterfly(twiddle).apply_to_rows(hi_chunk, lo_chunk);
        });
}
