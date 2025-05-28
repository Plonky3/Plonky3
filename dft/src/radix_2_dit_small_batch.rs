use alloc::vec::Vec;
use core::cell::RefCell;

use p3_field::{Field, TwoAdicField};
use p3_matrix::Matrix;
use p3_matrix::bitrev::{BitReversedMatrixView, BitReversibleMatrix};
use p3_matrix::dense::{RowMajorMatrix, RowMajorMatrixViewMut};
use p3_maybe_rayon::prelude::*;
use p3_util::{log2_strict_usize, reverse_slice_index_bits};

use crate::{Butterfly, DitButterfly, TwiddleFreeButterfly, TwoAdicSubgroupDft};

/// Estimates the optimal workload size for `T` to fit in L1 cache.
///
/// Approximates the size of the L1 cache by 32 KB. Used to determine the number of
/// chunks to process in parallel.
#[must_use]
const fn workload_size<T: Sized>() -> usize {
    const L1_CACHE_SIZE: usize = 1 << 15; // 32 KB
    L1_CACHE_SIZE / size_of::<T>()
}

/// A FFT algorithm which divides a butterfly network's layers into two halves.
///
/// Unlike other FFT algorithms, this algorithm is optimized for small batch sizes.
/// It also stores its twiddle factors and only re-computes if it is asked to do a
/// larger FFT.
///
/// Instead of parallelizing across rows, this algorithm parallelizes across groups of rows
/// with the same twiddle factors. This allows it to make use of field packings far more than
/// the standard methods. Additionally, once the chunk size is small enough, it computes
/// the last set of layers fully on a single thread, which avoids the overhead of
/// passing data between threads.
#[derive(Default, Clone, Debug)]
pub struct Radix2DitSmallBatch<F: Field> {
    /// Memoized twiddle factors for each length log_n.
    ///
    /// For each `i`, `twiddles[i]` contains a list of twiddles stored in
    /// bit reversed order. The final set of twiddles `twiddles[-1]` is the
    /// one element vectors `[1]` and more general `twiddles[-i]` has length `2^i`.
    ///
    /// TODO: The use of RefCell means this can't be shared across
    /// threads; consider using RwLock or finding a better design
    /// instead.
    twiddles: RefCell<Vec<Vec<F>>>,
}

impl<F: TwoAdicField> Radix2DitSmallBatch<F> {
    /// Create a new `Radix2DitSmallBatch` instance with precomputed twiddles for the given size.
    ///
    /// The input `n` should be a power of two, representing the maximal FFT size you expect to handle.
    pub fn new(n: usize) -> Self {
        let res = Self {
            twiddles: RefCell::default(),
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
            self.twiddles.replace(new_twiddles);
        }
    }
}

impl<F> TwoAdicSubgroupDft<F> for Radix2DitSmallBatch<F>
where
    F: TwoAdicField,
{
    type Evaluations = BitReversedMatrixView<RowMajorMatrix<F>>;

    fn dft_batch(&self, mut mat: RowMajorMatrix<F>) -> Self::Evaluations {
        let h = mat.height();
        let w = mat.width();
        let log_h = log2_strict_usize(h);

        self.update_twiddles(h);
        let root_table = self.twiddles.borrow();
        let len = root_table.len();
        let root_table = &root_table[len - log_h..];

        // Find the number of rows which can roughly fit in L1 cache.
        // The strategy will be to do a standard round-by-round parallelization
        // until the chunk size is smaller than `num_par_rows * mat.width()` after which we
        // send `num_par_rows` chunks to each thread and do the remainder of the
        // fft without transferring any more data between threads.
        let num_par_rows = (workload_size::<F>() / w).next_power_of_two().min(h); // Ensure we don't exceed the height of the matrix.
        let log_num_par_rows = log2_strict_usize(num_par_rows);
        let chunk_size = num_par_rows * w;

        // For the layers involving blocks larger than `num_par_rows`, we will
        // parallelize across the blocks.
        for twiddles in root_table[log_num_par_rows..].iter().rev() {
            dit_layer_par(&mut mat.as_view_mut(), twiddles);
        }

        // Once the blocks are small enough, we can split the matrix
        // into chunks of size `chunk_size` and process them in parallel.
        // This avoids passing data between threads, which can be expensive.
        par_remaining_layers(&mut mat.values, chunk_size, root_table, log_num_par_rows);

        // Finally we bit-reverse the matrix to ensure the output is in the correct order.
        mat.bit_reverse_rows()
    }
}

/// Applies one layer of the Radix-2 DIT FFT butterfly network making use of parallelization.
///
/// Splits the matrix into blocks of rows and performs in-place butterfly operations
/// on each block. Uses a `TwiddleFreeButterfly` for the first pair and `DitButterfly`
/// with precomputed twiddles for the rest.
///
/// Each block is processed in parallel, if the blocks are large enough they themselves
/// are split into parallel sub-blocks.
///
/// # Arguments
/// - `mat`: Mutable matrix whose height is a power of two.
/// - `twiddles`: Precomputed twiddle factors for this layer.
#[inline]
fn dit_layer_par<F: Field>(mat: &mut RowMajorMatrixViewMut<F>, twiddles: &[F]) {
    debug_assert!(
        mat.height() % twiddles.len() == 0,
        "Matrix height must be divisible by the number of twiddles"
    );
    let size = mat.values.len();
    let num_blocks = twiddles.len();

    let outer_block_size = size / num_blocks;
    let half_outer_block_size = outer_block_size / 2;

    mat.values
        .par_chunks_exact_mut(outer_block_size)
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

/// Splits the matrix into chunks of size `chunk_size` and performs
/// the remaining layers of the FFT in parallel on each chunk.
///
/// This avoids passing data between threads, which can be expensive.
#[inline]
fn par_remaining_layers<F: Field>(
    mat: &mut [F],
    chunk_size: usize,
    root_table: &[Vec<F>],
    log_num_par_rows: usize,
) {
    mat.par_chunks_exact_mut(chunk_size)
        .enumerate()
        .for_each(|(index, chunk)| {
            for (layer, twiddles) in root_table[..log_num_par_rows].iter().rev().enumerate() {
                let num_twiddles_per_block = 1 << layer;
                let start = index * num_twiddles_per_block;
                let twiddle_range = start..(start + num_twiddles_per_block);
                dit_layer(chunk, &(twiddles[twiddle_range]));
            }
        });
}

/// Applies one layer of the Radix-2 DIT FFT butterfly network on a single core.
///
/// Splits the matrix into blocks of rows and performs in-place butterfly operations
/// on each block.
///
/// # Arguments
/// - `vec`: Mutable vector whose height is a power of two.
/// - `twiddles`: Precomputed twiddle factors for this layer.
#[inline]
fn dit_layer<F: Field>(vec: &mut [F], twiddles: &[F]) {
    debug_assert_eq!(
        vec.len() % twiddles.len(),
        0,
        "Vector length must be divisible by the number of twiddles"
    );
    let size = vec.len();
    let num_blocks = twiddles.len();

    let block_size = size / num_blocks;
    let half_block_size = block_size / 2;

    vec.chunks_exact_mut(block_size)
        .zip(twiddles)
        .for_each(|(block, &twiddle)| {
            // Split each block vertically into top (hi) and bottom (lo) halves
            let (hi_chunk, lo_chunk) = block.split_at_mut(half_block_size);

            // Apply DIT butterfly using the twiddle factor at index `ind - 1`
            DitButterfly(twiddle).apply_to_rows(hi_chunk, lo_chunk);
        });
}
