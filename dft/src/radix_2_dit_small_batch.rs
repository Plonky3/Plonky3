//! An FFT implementation optimized for small batch sizes.

use alloc::vec::Vec;
use core::cell::RefCell;
use core::iter;

use p3_field::{Field, PackedField, PackedValue, TwoAdicField, scale_slice_in_place_single_core};
use p3_matrix::Matrix;
use p3_matrix::dense::{RowMajorMatrix, RowMajorMatrixViewMut};
use p3_matrix::util::reverse_matrix_index_bits;
use p3_maybe_rayon::prelude::*;
use p3_util::{flatten_to_base, log2_strict_usize, reverse_slice_index_bits};

use crate::{
    Butterfly, DifButterfly, DifButterflyZeros, DitButterfly, TwiddleFreeButterfly,
    TwoAdicSubgroupDft,
};

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
/// the standard methods even for low width matrices. Once the chunk size is small enough, it
/// computes a large set of layers fully on a single thread, which avoids the overhead of
/// passing data between threads.
#[derive(Default, Clone, Debug)]
pub struct Radix2DFTSmallBatch<F> {
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

    /// Similar to `twiddles`, but stored the inverses used for the inverse fft.
    inv_twiddles: RefCell<Vec<Vec<F>>>,
}

impl<F: TwoAdicField> Radix2DFTSmallBatch<F> {
    /// Create a new `Radix2DitSmallBatch` instance with precomputed twiddles for the given size.
    ///
    /// The input `n` should be a power of two, representing the maximal FFT size you expect to handle.
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
            .map(|i| nth_roots.iter().step_by(1 << i).copied().collect())
            .collect()
    }

    /// Compute twiddle and inv_twiddle factors, or take memoized ones if already available.
    fn update_twiddles(&self, fft_len: usize) {
        // TODO: This recomputes the entire table from scratch if we
        // need it to be bigger, which is wasteful.

        // roots_of_unity_table(fft_len) returns a vector of twiddles of length log_2(fft_len).
        let curr_max_fft_len = 1 << self.twiddles.borrow().len();
        if fft_len > curr_max_fft_len {
            let mut new_twiddles = self.roots_of_unity_table(fft_len);
            let mut new_inv_twiddles: Vec<Vec<F>> = new_twiddles
                .iter()
                .map(|ts| {
                    // The first twiddle is still one, instead of inverting, we can
                    // just reverse and negate.
                    iter::once(F::ONE)
                        .chain(ts[1..].iter().rev().map(|&f| -f))
                        .collect()
                })
                .collect();

            new_twiddles.iter_mut().for_each(|ts| {
                reverse_slice_index_bits(ts);
            });
            new_inv_twiddles.iter_mut().for_each(|ts| {
                reverse_slice_index_bits(ts);
            });

            self.twiddles.replace(new_twiddles);
            self.inv_twiddles.replace(new_inv_twiddles);
        }
    }
}

impl<F> TwoAdicSubgroupDft<F> for Radix2DFTSmallBatch<F>
where
    F: TwoAdicField,
{
    type Evaluations = RowMajorMatrix<F>;

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
        par_remaining_layers(&mut mat.values, chunk_size, &root_table[..log_num_par_rows]);

        // Finally we bit-reverse the matrix to ensure the output is in the correct order.
        reverse_matrix_index_bits(&mut mat);
        mat
    }

    fn idft_batch(&self, mut mat: RowMajorMatrix<F>) -> RowMajorMatrix<F> {
        let h = mat.height();
        let w = mat.width();
        let log_h = log2_strict_usize(h);

        self.update_twiddles(h);
        let root_table = self.inv_twiddles.borrow();
        let len = root_table.len();
        let root_table = &root_table[len - log_h..];

        // Find the number of rows which can roughly fit in L1 cache.
        // The strategy is literally the same as `dft_batch` but in reverse.
        // We start by moving `num_par_rows` rows onto each thread and doing
        // `num_par_rows` layers of the DFT. After this we recombine and do
        // a standard round-by-round parallelization for the remaining layers.
        let num_par_rows = (workload_size::<F>() / w).next_power_of_two().min(h); // Ensure we don't exceed the height of the matrix.
        let log_num_par_rows = log2_strict_usize(num_par_rows);
        let chunk_size = num_par_rows * w;

        // Need to start by bit-reversing the matrix.
        reverse_matrix_index_bits(&mut mat);

        // For the initial blocks, they are small enough that we can split the matrix
        // into chunks of size `chunk_size` and process them in parallel.
        // This avoids passing data between threads, which can be expensive.
        // We also divide by the height of the matrix while the data is nicely partitioned
        // on each core.
        par_initial_layers(
            &mut mat.values,
            chunk_size,
            &root_table[..log_num_par_rows],
            log_h,
        );

        // For the layers involving blocks larger than `num_par_rows`, we will
        // parallelize across the blocks.
        for twiddles in root_table[log_num_par_rows..].iter() {
            dif_layer_par(&mut mat.as_view_mut(), twiddles);
        }

        mat
    }

    fn coset_lde_batch(
        &self,
        mut mat: RowMajorMatrix<F>,
        added_bits: usize,
        shift: F,
    ) -> Self::Evaluations {
        let h = mat.height();
        let w = mat.width();
        let log_h = log2_strict_usize(h);

        self.update_twiddles(h << added_bits);
        let root_table = self.twiddles.borrow();
        let inv_root_table = self.inv_twiddles.borrow();
        let len = root_table.len();

        let root_table = &root_table[len - (log_h + added_bits)..];
        let inv_root_table = &inv_root_table[len - log_h..];
        let output_height = h << added_bits;

        // The matrix which we will use to store the output.
        let output_values = F::zero_vec(output_height * w);
        let mut output = RowMajorMatrix::new(output_values, w);

        // The strategy is reasonably straightforward.
        // The rough idea is we want to squash together the dft and idft code which will
        // cancel out the `reverse_matrix_index_bits`.

        // This lets us do all of the inner layers on a single thread reducing the amount
        // of data we need to transfer.

        // For technical reasons, we need to swap the twiddle factors, using the inverse
        // twiddles for the initial layers and the normal twiddles for the final layers.
        // This lets us interpret the initial transformation as the idft giving us coefficients
        // and the final transformation as the dft giving us evaluations.

        // Find the number of rows which can roughly fit in L1 cache.
        // The strategy will be to do a standard round-by-round parallelization
        // until the chunk size is smaller than `num_par_rows * mat.width()` after which we
        // send `num_par_rows` chunks to each thread and do the remainder of the
        // fft without transferring any more data between threads.
        let num_par_rows = (workload_size::<F>() / w).next_power_of_two().min(h); // Ensure we don't exceed the height of the matrix.
        let log_num_par_rows = log2_strict_usize(num_par_rows);

        // For the layers involving blocks larger than `num_par_rows`, we will
        // parallelize across the blocks.
        for twiddles in inv_root_table[log_num_par_rows..].iter().rev() {
            dit_layer_par(&mut mat.as_view_mut(), twiddles);
        }

        // Now do all the inner layers at once. This does the final `log_num_par_rows` of
        // the initial transformation, then copies the values of mat to output, scales then
        // and does the first `log_num_par_rows + added_bits` layers of the final transformation.
        par_middle_layers(
            &mut mat.as_view_mut(),
            &mut output.as_view_mut(),
            num_par_rows,
            &root_table[..(log_num_par_rows + added_bits)],
            &inv_root_table[..log_num_par_rows],
            added_bits,
            shift,
        );

        // For the layers involving blocks larger than `num_par_rows`, we will
        // parallelize across the blocks.
        for twiddles in root_table[(log_num_par_rows + added_bits)..].iter() {
            dif_layer_par(&mut output.as_view_mut(), twiddles);
        }

        output
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
fn dif_layer_par<F: Field>(mat: &mut RowMajorMatrixViewMut<F>, twiddles: &[F]) {
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
                        DifButterfly(twiddles[ind]).apply_to_rows(hi_chunk, lo_chunk);
                    }
                });
        });
}

/// Splits the matrix into chunks of size `chunk_size` and performs
/// the remaining layers of the FFT in parallel on each chunk.
///
/// This avoids passing data between threads, which can be expensive.
#[inline]
fn par_remaining_layers<F: Field>(mat: &mut [F], chunk_size: usize, root_table: &[Vec<F>]) {
    mat.par_chunks_exact_mut(chunk_size)
        .enumerate()
        .for_each(|(index, chunk)| {
            for (layer, twiddles) in root_table.iter().rev().enumerate() {
                let num_twiddles_per_block = 1 << layer;
                let start = index * num_twiddles_per_block;
                let twiddle_range = start..(start + num_twiddles_per_block);
                dit_layer(chunk, &twiddles[twiddle_range]);
            }
        });
}

/// Splits the matrix into chunks of size `chunk_size` and performs
/// the initial layers of the iFFT in parallel on each chunk.
///
/// This avoids passing data between threads, which can be expensive.
///
/// Basically identical to [par_remaining_layers] but in reverse.
#[inline]
fn par_initial_layers<F: Field>(
    mat: &mut [F],
    chunk_size: usize,
    root_table: &[Vec<F>],
    log_height: usize,
) {
    let num_rounds = root_table.len();
    let height_inv = F::ONE.div_2exp_u64(log_height as u64);
    mat.par_chunks_exact_mut(chunk_size)
        .enumerate()
        .for_each(|(index, chunk)| {
            // Divide all elements by the height of the matrix.
            scale_slice_in_place_single_core(height_inv, chunk);

            for (layer, twiddles) in root_table.iter().enumerate() {
                let num_twiddles_per_block = 1 << (num_rounds - layer - 1);
                let start = index * num_twiddles_per_block;
                let twiddle_range = start..(start + num_twiddles_per_block);
                dif_layer(chunk, &twiddles[twiddle_range]);
            }
        });
}

fn par_middle_layers<F: Field>(
    in_mat: &mut RowMajorMatrixViewMut<F>,
    out_mat: &mut RowMajorMatrixViewMut<F>,
    num_par_rows: usize,
    root_table: &[Vec<F>],
    inv_root_table: &[Vec<F>],
    added_bits: usize,
    shift: F,
) {
    debug_assert_eq!(in_mat.width(), out_mat.width());
    debug_assert_eq!(in_mat.height() << added_bits, out_mat.height());

    let width = in_mat.width();
    let height = in_mat.height();
    let num_rounds = root_table.len();
    let in_chunk_size = num_par_rows * width;
    let out_chunk_size = in_chunk_size << added_bits;

    let log_height = log2_strict_usize(height);
    let inv_height = F::ONE.div_2exp_u64(log_height as u64);

    let scaling_packed: Vec<F::Packing> = F::Packing::packed_shifted_powers(shift, inv_height)
        .take(height.div_ceil(F::Packing::WIDTH))
        .collect::<Vec<_>>();
    let mut scaling: Vec<F> = unsafe { flatten_to_base(scaling_packed) };
    scaling.truncate(height);
    reverse_slice_index_bits(&mut scaling);

    in_mat
        .values
        .par_chunks_exact_mut(in_chunk_size)
        .zip(out_mat.values.par_chunks_exact_mut(out_chunk_size))
        .zip(scaling.par_chunks_exact_mut(num_par_rows))
        .enumerate()
        .for_each(|(index, ((in_chunk, out_chunk), scaling))| {
            for (layer, twiddles) in inv_root_table.iter().rev().enumerate() {
                let num_twiddles_per_block = 1 << layer;
                let start = index * num_twiddles_per_block;
                let twiddle_range = start..(start + num_twiddles_per_block);
                dit_layer(in_chunk, &twiddles[twiddle_range]);
            }

            // Copy the values to the output matrix and scale appropriately.
            in_chunk
                .chunks_exact(width)
                .zip(scaling)
                .zip(out_chunk.chunks_exact_mut(width << added_bits))
                .for_each(|((in_row, scale), out_row)| {
                    out_row
                        .iter_mut()
                        .zip(in_row.iter())
                        .for_each(|(out_val, in_val)| {
                            *out_val = *in_val * *scale;
                        });
                });

            for (layer, twiddles) in root_table.iter().enumerate() {
                let num_twiddles_per_block = 1 << (num_rounds - layer - 1);
                let start = index * num_twiddles_per_block;
                let twiddle_range = start..(start + num_twiddles_per_block);
                // While
                if layer < added_bits {
                    dif_layer_zeros(out_chunk, &twiddles[twiddle_range], added_bits - layer - 1);
                } else {
                    dif_layer(out_chunk, &twiddles[twiddle_range]);
                }
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

            // Apply DIT butterfly
            DitButterfly(twiddle).apply_to_rows(hi_chunk, lo_chunk);
        });
}

/// Applies one layer of the Radix-2 DIF FFT butterfly network on a single core.
///
/// Splits the matrix into blocks of rows and performs in-place butterfly operations
/// on each block.
///
/// # Arguments
/// - `vec`: Mutable vector whose height is a power of two.
/// - `twiddles`: Precomputed twiddle factors for this layer.
#[inline]
fn dif_layer<F: Field>(vec: &mut [F], twiddles: &[F]) {
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

            // Apply DIF butterfly
            DifButterfly(twiddle).apply_to_rows(hi_chunk, lo_chunk);
        });
}

/// Applies one layer of the Radix-2 DIF FFT butterfly network on a single core to
/// a recently zero-padded matrix.
///
/// Splits the matrix into blocks of rows and performs in-place butterfly operations
/// on each block.
///
/// Assume `added_bits = 2`. Then the rows of our matrix look like:
/// ```text
/// [R0, 0, 0, 0, R1, 0, 0, 0, ...]
/// ```
/// Thus the first two butterfly layers can be implemented more simply as they map the matrix to:
/// ```text
/// After Layer 0: [R0, T00 * R0, 0, 0, R1, T01 * R1, 0, 0, ...]
/// After Layer 1: [R0, T00 * R0, T10 * R0, T10 * T00 * R0, R1, T01 * R1, T11 * R1, T11 * T01 * R1, ...].
/// ```
///
/// # Arguments
/// - `vec`: Mutable vector whose height is a power of two.
/// - `twiddles`: Precomputed twiddle factors for this layer.
/// - `skip`: `(1 << skip) - 1` is the number of entirely zero
///   blocks between each non zero block.
#[inline]
fn dif_layer_zeros<F: Field>(vec: &mut [F], twiddles: &[F], skip: usize) {
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
        .step_by(1 << skip) // Skip the zero blocks
        .for_each(|(block, &twiddle)| {
            // Split each block vertically into top (hi) and bottom (lo) halves
            let (hi_chunk, lo_chunk) = block.split_at_mut(half_block_size);

            // Apply DIF butterfly making use of the fact that `lo_chunk` is zero.
            DifButterflyZeros(twiddle).apply_to_rows(hi_chunk, lo_chunk);
        });
}
