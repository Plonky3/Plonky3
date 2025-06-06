//! An FFT implementation optimized for small batch sizes.

use alloc::vec::Vec;
use core::cell::RefCell;
use core::iter;
use core::marker::PhantomData;

use itertools::Itertools;
use p3_field::{Field, PackedField, PackedValue, TwoAdicField, scale_slice_in_place_single_core};
use p3_matrix::Matrix;
use p3_matrix::dense::{RowMajorMatrix, RowMajorMatrixViewMut};
use p3_matrix::util::reverse_matrix_index_bits;
use p3_maybe_rayon::prelude::*;
use p3_util::{as_base_slice, flatten_to_base, log2_strict_usize, reverse_slice_index_bits};

use crate::{
    Butterfly, DifButterfly, DifButterflyZeros, DitButterfly, TwiddleFreeButterfly,
    TwoAdicSubgroupDft,
};

/// The number of layers to compute in each parallelization.
const LAYERS_PER_GROUP: usize = 3;

/// An FFT algorithm which divides a butterfly network's layers into two halves.
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
    /// Create a new `Radix2DFTSmallBatch` instance with precomputed twiddles for the given size.
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
        // need it to be larger, which is wasteful.

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

        // The strategy will be to do a standard round-by-round parallelization
        // until the chunk size is smaller than `num_par_rows * mat.width()` after which we
        // send `num_par_rows` chunks to each thread and do the remainder of the
        // fft without transferring any more data between threads.
        let num_par_rows = estimate_num_rows_in_l1::<F>(h, w);
        let log_num_par_rows = log2_strict_usize(num_par_rows);
        let chunk_size = num_par_rows * w;

        // For the layers involving blocks larger than `num_par_rows`, we will
        // parallelize across the blocks.

        let multi_layer_dit = MultiLayerDitButterfly::new();

        // We do `LAYERS_PER_GROUP` layers of the DFT at once, to minimize how much data we need to transfer
        // between threads.
        for (dit_0, dit_1, dit_2) in root_table[log_num_par_rows..]
            .iter()
            .rev()
            .map(|slice| unsafe { as_base_slice::<DitButterfly<F>, F>(slice) }) // Safe as DitButterfly is #[repr(transparent)]
            .tuples()
        {
            dft_layer_par_triple(&mut mat.as_view_mut(), dit_0, dit_1, dit_2, multi_layer_dit);
        }

        // If the total number of layers is not a multiple of `LAYERS_PER_GROUP`,
        // we need to handle the remaining layers separately.
        let corr = (log_h - log_num_par_rows) % LAYERS_PER_GROUP;
        dft_layer_par_extra_layers(
            &mut mat.as_view_mut(),
            &root_table[log_num_par_rows..log_num_par_rows + corr],
            multi_layer_dit,
        );

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
        let num_par_rows = estimate_num_rows_in_l1::<F>(h, w);
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

        let multi_layer_dif = MultiLayerDifButterfly::new();

        // If the total number of layers is not a multiple of `LAYERS_PER_GROUP`,
        // we need to handle the remaining layers separately.
        let corr = (log_h - log_num_par_rows) % LAYERS_PER_GROUP;
        dft_layer_par_extra_layers(
            &mut mat.as_view_mut(),
            &root_table[log_num_par_rows..log_num_par_rows + corr],
            multi_layer_dif,
        );

        // We do `LAYERS_PER_GROUP` layers of the DFT at once, to minimize how much data we need to transfer
        // between threads.
        for (dif_0, dif_1, dif_2) in root_table[(log_num_par_rows + corr)..]
            .iter()
            .map(|slice| unsafe { as_base_slice::<DifButterfly<F>, F>(slice) }) // Safe as DifButterfly is #[repr(transparent)]
            .tuples()
        {
            dft_layer_par_triple(&mut mat.as_view_mut(), dif_2, dif_1, dif_0, multi_layer_dif);
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
        let mut out = RowMajorMatrix::new(output_values, w);

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
        let num_par_rows = estimate_num_rows_in_l1::<F>(h, w);
        let num_inner_dit_layers = log2_strict_usize(num_par_rows);
        let num_inner_dif_layers = num_inner_dit_layers + added_bits;

        // We will do large DFT/iDFT layers in batches of `LAYERS_PER_GROUP`. We start with
        // the dit layers.
        let multi_layer_dit = MultiLayerDitButterfly::new();
        for (dit_0, dit_1, dit_2) in inv_root_table[num_inner_dit_layers..]
            .iter()
            .rev()
            .map(|slice| unsafe { as_base_slice::<DitButterfly<F>, F>(slice) }) // Safe as DitButterfly is #[repr(transparent)]
            .tuples()
        {
            dft_layer_par_triple(&mut mat.as_view_mut(), dit_0, dit_1, dit_2, multi_layer_dit);
        }

        // If the total number of layers is not a multiple of `LAYERS_PER_GROUP`,
        // we need to handle the remaining layers separately.
        let corr = (log_h - num_inner_dit_layers) % LAYERS_PER_GROUP;
        dft_layer_par_extra_layers(
            &mut mat.as_view_mut(),
            &inv_root_table[num_inner_dit_layers..num_inner_dit_layers + corr],
            multi_layer_dit,
        );

        // Now do all the inner layers at once. This does the final `log_num_par_rows` of
        // the initial transformation, then copies the values of mat to output, scales then
        // and does the first `log_num_par_rows + added_bits` layers of the final transformation.
        par_middle_layers(
            &mut mat.as_view_mut(),
            &mut out.as_view_mut(),
            num_par_rows,
            &root_table[..(num_inner_dif_layers)],
            &inv_root_table[..num_inner_dit_layers],
            added_bits,
            shift,
        );

        // We are left with the final dif layers.
        let multi_layer_dif = MultiLayerDifButterfly::new();

        // If the total number of layers is not a multiple of `LAYERS_PER_GROUP`,
        // we need to handle the remaining layers separately.
        dft_layer_par_extra_layers(
            &mut out.as_view_mut(),
            &root_table[num_inner_dif_layers..num_inner_dif_layers + corr],
            multi_layer_dif,
        );

        // We do `LAYERS_PER_GROUP` layers of the DFT at once, to minimize how much data we need to transfer
        // between threads.
        for (dif_0, dif_1, dif_2) in root_table[(num_inner_dif_layers + corr)..]
            .iter()
            .map(|slice| unsafe { as_base_slice::<DifButterfly<F>, F>(slice) }) // Safe as DifButterfly is #[repr(transparent)]
            .tuples()
        {
            dft_layer_par_triple(&mut out.as_view_mut(), dif_2, dif_1, dif_0, multi_layer_dif);
        }

        out
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
fn dft_layer_par<F: Field, B: Butterfly<F>>(mat: &mut RowMajorMatrixViewMut<F>, twiddles: &[B]) {
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
            let num_threads = current_num_threads();
            let inner_block_size = size / (2 * num_blocks).max(num_threads);

            hi_chunk
                .par_chunks_mut(inner_block_size)
                .zip(lo_chunk.par_chunks_mut(inner_block_size))
                .for_each(|(hi_chunk, lo_chunk)| {
                    if ind == 0 {
                        // The first pair doesn't require a twiddle factor
                        TwiddleFreeButterfly.apply_to_rows(hi_chunk, lo_chunk);
                    } else {
                        // Apply DIT butterfly using the twiddle factor at index `ind - 1`
                        twiddles[ind].apply_to_rows(hi_chunk, lo_chunk);
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
            remaining_layers(chunk, root_table, index);
        });
}

/// Performs a collection of DIT layers on a chunk of the matrix.
fn remaining_layers<F: Field>(chunk: &mut [F], root_table: &[Vec<F>], index: usize) {
    for (layer, twiddles) in root_table.iter().rev().enumerate() {
        let num_twiddles_per_block = 1 << layer;
        let start = index * num_twiddles_per_block;
        let twiddle_range = start..(start + num_twiddles_per_block);
        // Safe as DitButterfly is #[repr(transparent)]
        let dit_twiddles: &[DitButterfly<F>] = unsafe { as_base_slice(&twiddles[twiddle_range]) };
        dft_layer(chunk, dit_twiddles);
    }
}

/// Splits the matrix into chunks of size `chunk_size` and performs
/// the initial layers of the iFFT in parallel on each chunk.
///
/// This avoids passing data between threads, which can be expensive.
///
/// Basically identical to [par_remaining_layers] but in reverse and we
/// also divide by the height.
#[inline]
fn par_initial_layers<F: Field>(
    mat: &mut [F],
    chunk_size: usize,
    root_table: &[Vec<F>],
    log_height: usize,
) {
    let inv_height = F::ONE.div_2exp_u64(log_height as u64);
    mat.par_chunks_exact_mut(chunk_size)
        .enumerate()
        .for_each(|(index, chunk)| {
            // Divide all elements by the height of the matrix.
            scale_slice_in_place_single_core(chunk, inv_height);
            initial_layers(chunk, root_table, index);
        });
}

/// Performs a collection of DIF layers on a chunk of the matrix.
#[inline]
fn initial_layers<F: Field>(chunk: &mut [F], root_table: &[Vec<F>], index: usize) {
    let num_rounds = root_table.len();

    for (layer, twiddles) in root_table.iter().enumerate() {
        let num_twiddles_per_block = 1 << (num_rounds - layer - 1);
        let start = index * num_twiddles_per_block;
        let twiddle_range = start..(start + num_twiddles_per_block);
        // Safe as DifButterfly is #[repr(transparent)]
        let dif_twiddles: &[DifButterfly<F>] = unsafe { as_base_slice(&twiddles[twiddle_range]) };
        dft_layer(chunk, dif_twiddles);
    }
}

/// Splits the matrix into chunks of size `chunk_size` and performs
/// the middle layers of a coset_lde in parallel on each chunk.
///
/// Similar to [par_remaining_layers] followed by [par_initial_layers]
/// with a scaling and copying operation in between.
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
            remaining_layers(in_chunk, inv_root_table, index);

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

            // We can do something cheaper than standard DFT layers for the first `added_bits` layers.
            // as there are a lot of zeroes in the out_chunk.
            for (layer, twiddles) in root_table[..added_bits].iter().enumerate() {
                let num_twiddles_per_block = 1 << (num_rounds - layer - 1);
                let start = index * num_twiddles_per_block;
                let twiddle_range = start..(start + num_twiddles_per_block);

                // Safe as DifButterflyZeros is #[repr(transparent)]
                let dif_twiddles_zeros: &[DifButterflyZeros<F>] =
                    unsafe { as_base_slice(&twiddles[twiddle_range]) };
                dft_layer_zeros(out_chunk, dif_twiddles_zeros, added_bits - layer - 1);
            }

            initial_layers(out_chunk, &root_table[added_bits..], index);
        });
}

/// Applies one layer of the Radix-2 FFT butterfly network on a single core.
///
/// Splits the matrix into blocks of rows and performs in-place butterfly operations
/// on each block.
///
/// # Arguments
/// - `vec`: Mutable vector whose height is a power of two.
/// - `twiddles`: Precomputed twiddle factors for this layer.
#[inline]
fn dft_layer<F: Field, B: Butterfly<F>>(vec: &mut [F], twiddles: &[B]) {
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
            twiddle.apply_to_rows(hi_chunk, lo_chunk);
        });
}

/// Applies two layers of the Radix-2 FFT butterfly network making use of parallelization.
///
/// Splits the matrix into blocks of rows and performs in-place butterfly operations
/// on each block. Advantage of doing two layers at once is it reduces the amount of
/// data transferred between threads.
///
/// # Arguments
/// - `mat`: Mutable matrix whose height is a power of two.
/// - `twiddles_small`: Precomputed twiddle factors for the layer with the smallest block size.
/// - `twiddles_large`: Precomputed twiddle factors for the layer with the largest block size.
/// - `multi_butterfly`: Multi-layer butterfly which applies the two layers in the correct order.
#[inline]
fn dft_layer_par_double<F: Field, B: Butterfly<F>, M: MultiLayerButterfly<F, B>>(
    mat: &mut RowMajorMatrixViewMut<F>,
    twiddles_small: &[B],
    twiddles_large: &[B],
    multi_butterfly: M,
) {
    debug_assert!(
        mat.height() % twiddles_small.len() == 0,
        "Matrix height must be divisible by the number of twiddles"
    );
    let size = mat.values.len();
    let num_blocks = twiddles_small.len();

    let outer_block_size = size / num_blocks;
    let quarter_outer_block_size = outer_block_size / 4;

    // Estimate the optimal size of the inner chunks so that all data fits in L1 cache.
    // Note that 4 inner chunks are processed in each parallel thread so we divide by 4.
    let inner_chunk_size =
        (workload_size::<F>().next_power_of_two() / 4).min(quarter_outer_block_size);

    mat.values
        .par_chunks_exact_mut(outer_block_size)
        .enumerate()
        .for_each(|(ind, block)| {
            // Split each block into four quarters. Each quarter will be further split into
            // sub-chunks processed in parallel.
            let chunk_par_iters_0 = block
                .chunks_exact_mut(quarter_outer_block_size)
                .map(|chunk| chunk.par_chunks_mut(inner_chunk_size))
                .collect::<Vec<_>>();
            let chunk_par_iters_1 = zip_par_iter_vec(chunk_par_iters_0);
            chunk_par_iters_1.into_iter().tuples().for_each(|(hi, lo)| {
                hi.zip(lo).for_each(|chunks| {
                    let mut chunks: DoubleLayerBlockDecomposition<F> = chunks.into();
                    multi_butterfly.apply_2_layers(
                        &mut chunks,
                        ind,
                        twiddles_small,
                        twiddles_large,
                    );
                });
            });
        });
}

/// Applies three layers of a Radix-2 FFT butterfly network making use of parallelization.
///
/// Splits the matrix into blocks of rows and performs in-place butterfly operations
/// on each block. Advantage of doing three layers at once is it reduces the amount of
/// data transferred between threads.
///
/// # Arguments
/// - `mat`: Mutable matrix whose height is a power of two.
/// - `twiddles_small`: Precomputed twiddle factors for the layer with the smallest block size.
/// - `twiddles_med`: Precomputed twiddle factors for the middle layer.
/// - `twiddles_large`: Precomputed twiddle factors for the layer with the largest block size.
/// - `multi_butterfly`: Multi-layer butterfly which applies the three layers in the correct order.
#[inline]
fn dft_layer_par_triple<F: Field, B: Butterfly<F>, M: MultiLayerButterfly<F, B>>(
    mat: &mut RowMajorMatrixViewMut<F>,
    twiddles_small: &[B],
    twiddles_med: &[B],
    twiddles_large: &[B],
    multi_butterfly: M,
) {
    debug_assert!(
        mat.height() % twiddles_small.len() == 0,
        "Matrix height must be divisible by the number of twiddles"
    );
    let size = mat.values.len();
    let num_blocks = twiddles_small.len();

    let outer_block_size = size / num_blocks;
    let eighth_outer_block_size = outer_block_size / 8;

    // Estimate the optimal size of the inner chunks so that all data fits in L1 cache.
    // Note that 8 inner chunks are processed in each parallel thread so we divide by 8.
    let inner_chunk_size =
        (workload_size::<F>().next_power_of_two() / 8).min(eighth_outer_block_size);

    mat.values
        .par_chunks_exact_mut(outer_block_size)
        .enumerate()
        .for_each(|(ind, block)| {
            // Split each block into eight equal parts. Each part will be further split into
            // sub-chunks processed in parallel.
            let chunk_par_iters_0 = block
                .chunks_exact_mut(eighth_outer_block_size)
                .map(|chunk| chunk.par_chunks_mut(inner_chunk_size))
                .collect::<Vec<_>>();
            let chunk_par_iters_1 = zip_par_iter_vec(chunk_par_iters_0);
            let chunk_par_iters_2 = zip_par_iter_vec(chunk_par_iters_1);
            chunk_par_iters_2.into_iter().tuples().for_each(|(hi, lo)| {
                hi.zip(lo).for_each(|chunks| {
                    let mut chunks: TripleLayerBlockDecomposition<F> = chunks.into();
                    multi_butterfly.apply_3_layers(
                        &mut chunks,
                        ind,
                        twiddles_small,
                        twiddles_med,
                        twiddles_large,
                    );
                })
            });
        });
}

/// Applies the remaining layers of the Radix-2 FFT butterfly network in parallel.
///
/// This function is used to correct for the fact that the total number of layers is
/// may not be a multiple of `LAYERS_PER_GROUP`.
fn dft_layer_par_extra_layers<F: Field, B: Butterfly<F>, M: MultiLayerButterfly<F, B>>(
    mat: &mut RowMajorMatrixViewMut<F>,
    root_table: &[Vec<F>],
    multi_layer: M,
) {
    // If the total number of layers is not a multiple of `LAYERS_PER_GROUP`,
    // we need to handle the remaining layers separately.
    match root_table.len() {
        1 => {
            // Safe as DitButterfly is #[repr(transparent)]
            let fft_layer: &[B] = unsafe { as_base_slice(&root_table[0]) };
            dft_layer_par(&mut mat.as_view_mut(), fft_layer);
        }
        2 => {
            let fft_layer_0: &[B] = unsafe { as_base_slice(&root_table[0]) };
            let fft_layer_1: &[B] = unsafe { as_base_slice(&root_table[1]) };
            dft_layer_par_double(
                &mut mat.as_view_mut(),
                fft_layer_1,
                fft_layer_0,
                multi_layer,
            );
        }
        0 => {}
        _ => unreachable!("The number of layers must be 0, 1 or 2"),
    }
}

/// Applies one layer of the Radix-2 FFT butterfly network on a single core to
/// a recently zero-padded matrix.
///
/// Splits the matrix into blocks of rows and performs in-place butterfly operations
/// on each block.
///
/// Assume `added_bits = 2` and we are doing a decimation in frequency approach.
/// Then the rows of our matrix look like:
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
fn dft_layer_zeros<F: Field, B: Butterfly<F>>(vec: &mut [F], twiddles: &[B], skip: usize) {
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
            twiddle.apply_to_rows(hi_chunk, lo_chunk);
        });
}

/// A struct representing a decomposition of an FFT block into four sub-blocks.
/// This is used to apply two layers of the FFT butterfly network to the block.
struct DoubleLayerBlockDecomposition<'a, F: Field> {
    hi_hi: &'a mut [F],
    hi_lo: &'a mut [F],
    lo_hi: &'a mut [F],
    lo_lo: &'a mut [F],
}

impl<F: Field> DoubleLayerBlockDecomposition<'_, F> {
    /// Performs an FFT layer on the sub-blocks using a single twiddle factor.
    #[inline]
    fn fft_layer_single_twiddle<Fly: Butterfly<F>>(&mut self, butterfly: Fly) {
        butterfly.apply_to_rows(self.hi_hi, self.lo_hi);
        butterfly.apply_to_rows(self.hi_lo, self.lo_lo);
    }

    /// Performs an FFT layer on the sub-blocks using a pair of twiddle factors.
    ///
    /// The inputs are differentiated in order to allow the first input to potentially
    /// be a `TwiddleFreeButterfly`, which does not require a twiddle factor.
    #[inline]
    fn fft_layer_double_twiddle<Fly0: Butterfly<F>, Fly1: Butterfly<F>>(
        &mut self,
        fly0: Fly0,
        fly1: Fly1,
    ) {
        fly0.apply_to_rows(self.hi_hi, self.hi_lo);
        fly1.apply_to_rows(self.lo_hi, self.lo_lo);
    }
}

impl<'a, F: Field> From<((&'a mut [F], &'a mut [F]), (&'a mut [F], &'a mut [F]))>
    for DoubleLayerBlockDecomposition<'a, F>
{
    fn from(value: ((&'a mut [F], &'a mut [F]), (&'a mut [F], &'a mut [F]))) -> Self {
        Self {
            hi_hi: value.0.0,
            hi_lo: value.0.1,
            lo_hi: value.1.0,
            lo_lo: value.1.1,
        }
    }
}

/// A struct representing a decomposition of an FFT block into eight sub-blocks.
/// This is used to apply three layers of the FFT butterfly network to the block.
struct TripleLayerBlockDecomposition<'a, F: Field> {
    hi_hi_hi: &'a mut [F],
    hi_hi_lo: &'a mut [F],
    hi_lo_hi: &'a mut [F],
    hi_lo_lo: &'a mut [F],
    lo_hi_hi: &'a mut [F],
    lo_hi_lo: &'a mut [F],
    lo_lo_hi: &'a mut [F],
    lo_lo_lo: &'a mut [F],
}

impl<F: Field> TripleLayerBlockDecomposition<'_, F> {
    /// Performs an FFT layer on the sub-blocks using a single twiddle factor.
    #[inline]
    fn fft_layer_single_twiddle<Fly: Butterfly<F>>(&mut self, butterfly: Fly) {
        butterfly.apply_to_rows(self.hi_hi_hi, self.lo_hi_hi);
        butterfly.apply_to_rows(self.hi_hi_lo, self.lo_hi_lo);
        butterfly.apply_to_rows(self.hi_lo_hi, self.lo_lo_hi);
        butterfly.apply_to_rows(self.hi_lo_lo, self.lo_lo_lo);
    }

    /// Performs an FFT layer on the sub-blocks using a pair of twiddle factors.
    ///
    /// The inputs are differentiated in order to allow the first input to potentially
    /// be a `TwiddleFreeButterfly`, which does not require a twiddle factor.
    #[inline]
    fn fft_layer_double_twiddle<Fly0: Butterfly<F>, Fly1: Butterfly<F>>(
        &mut self,
        fly0: Fly0,
        fly1: Fly1,
    ) {
        fly0.apply_to_rows(self.hi_hi_hi, self.hi_lo_hi);
        fly0.apply_to_rows(self.hi_hi_lo, self.hi_lo_lo);
        fly1.apply_to_rows(self.lo_hi_hi, self.lo_lo_hi);
        fly1.apply_to_rows(self.lo_hi_lo, self.lo_lo_lo);
    }

    /// Performs an FFT layer on the sub-blocks using a four twiddle factors.
    ///
    /// The inputs are differentiated in order to allow the first input to potentially
    /// be a `TwiddleFreeButterfly`, which does not require a twiddle factor.
    #[inline]
    fn fft_layer_quad_twiddle<Fly0: Butterfly<F>, Flies: Butterfly<F>>(
        &mut self,
        fly0: Fly0,
        butterflies: &[Flies],
    ) {
        debug_assert!(butterflies.len() == 3);
        fly0.apply_to_rows(self.hi_hi_hi, self.hi_hi_lo);
        butterflies[0].apply_to_rows(self.hi_lo_hi, self.hi_lo_lo);
        butterflies[1].apply_to_rows(self.lo_hi_hi, self.lo_hi_lo);
        butterflies[2].apply_to_rows(self.lo_lo_hi, self.lo_lo_lo);
    }
}

impl<'a, F: Field>
    From<(
        ((&'a mut [F], &'a mut [F]), (&'a mut [F], &'a mut [F])),
        ((&'a mut [F], &'a mut [F]), (&'a mut [F], &'a mut [F])),
    )> for TripleLayerBlockDecomposition<'a, F>
{
    fn from(
        value: (
            ((&'a mut [F], &'a mut [F]), (&'a mut [F], &'a mut [F])),
            ((&'a mut [F], &'a mut [F]), (&'a mut [F], &'a mut [F])),
        ),
    ) -> Self {
        Self {
            hi_hi_hi: value.0.0.0,
            hi_hi_lo: value.0.0.1,
            hi_lo_hi: value.0.1.0,
            hi_lo_lo: value.0.1.1,
            lo_hi_hi: value.1.0.0,
            lo_hi_lo: value.1.0.1,
            lo_lo_hi: value.1.1.0,
            lo_lo_lo: value.1.1.1,
        }
    }
}

/// Estimates the optimal workload size for `T` to fit in L1 cache.
///
/// Approximates the size of the L1 cache by 32 KB. Used to determine the number of
/// chunks to process in parallel.
#[must_use]
const fn workload_size<T: Sized>() -> usize {
    const L1_CACHE_SIZE: usize = 1 << 15; // 32 KB
    L1_CACHE_SIZE / size_of::<T>()
}

/// Estimates the optimal number of rows of a `RowMajorMatrix<T>` to take in each parallel chunk.
///
/// Designed to ensure that `<T> * estimate_num_rows_par() * width` is roughly the size of the L1 cache.
///
/// Assumes that height is a power of two and always outputs a power of two.
#[must_use]
fn estimate_num_rows_in_l1<T: Sized>(height: usize, width: usize) -> usize {
    (workload_size::<T>() / width)
        .next_power_of_two()
        .min(height) // Ensure we don't exceed the height of the matrix.
}

/// Given a vector of parallel iterators, zip all pairs together.
///
/// This lets us simulate the izip!() macro but for our possibly parallel iterators.
///
/// This function assumes that the input vector has an even number of elements. If
/// it is given an odd number of elements, the last element will be ignored.
#[inline]
fn zip_par_iter_vec<I: IndexedParallelIterator>(
    in_vec: Vec<I>,
) -> Vec<impl IndexedParallelIterator<Item = (I::Item, I::Item)>> {
    in_vec
        .into_iter()
        .tuples()
        .map(|(hi, lo)| hi.zip(lo))
        .collect::<Vec<_>>()
}

trait MultiLayerButterfly<F: Field, B: Butterfly<F>> {
    fn apply_2_layers(
        &self,
        chunk_decomposition: &mut DoubleLayerBlockDecomposition<F>,
        ind: usize,
        twiddles_small: &[B],
        twiddles_large: &[B],
    );

    fn apply_3_layers(
        &self,
        chunk_decomposition: &mut TripleLayerBlockDecomposition<F>,
        ind: usize,
        twiddles_small: &[B],
        twiddles_med: &[B],
        twiddles_large: &[B],
    );
}

#[derive(Debug, Clone, Copy)]
struct MultiLayerDitButterfly<F> {
    _phantom: PhantomData<F>,
}

impl<F: Field> MultiLayerDitButterfly<F> {
    fn new() -> Self {
        Self {
            _phantom: PhantomData,
        }
    }
}

impl<F: Field> MultiLayerButterfly<F, DitButterfly<F>> for MultiLayerDitButterfly<F> {
    #[inline]
    fn apply_2_layers(
        &self,
        chunk_decomposition: &mut DoubleLayerBlockDecomposition<F>,
        ind: usize,
        twiddles_small: &[DitButterfly<F>],
        twiddles_large: &[DitButterfly<F>],
    ) {
        if ind == 0 {
            chunk_decomposition.fft_layer_single_twiddle(TwiddleFreeButterfly);
            chunk_decomposition.fft_layer_double_twiddle(TwiddleFreeButterfly, twiddles_large[1]);
        } else {
            chunk_decomposition.fft_layer_single_twiddle(twiddles_small[ind]);
            chunk_decomposition
                .fft_layer_double_twiddle(twiddles_large[2 * ind], twiddles_large[2 * ind + 1]);
        }
    }

    #[inline]
    fn apply_3_layers(
        &self,
        chunk_decomposition: &mut TripleLayerBlockDecomposition<F>,
        ind: usize,
        twiddles_small: &[DitButterfly<F>],
        twiddles_med: &[DitButterfly<F>],
        twiddles_large: &[DitButterfly<F>],
    ) {
        if ind == 0 {
            chunk_decomposition.fft_layer_single_twiddle(TwiddleFreeButterfly);
            chunk_decomposition.fft_layer_double_twiddle(TwiddleFreeButterfly, twiddles_med[1]);
            chunk_decomposition.fft_layer_quad_twiddle(TwiddleFreeButterfly, &twiddles_large[1..4]);
        } else {
            chunk_decomposition.fft_layer_single_twiddle(twiddles_small[ind]);
            chunk_decomposition
                .fft_layer_double_twiddle(twiddles_med[2 * ind], twiddles_med[2 * ind + 1]);
            chunk_decomposition.fft_layer_quad_twiddle(
                twiddles_large[4 * ind],
                &twiddles_large[4 * ind + 1..4 * (ind + 1)],
            );
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct MultiLayerDifButterfly<F> {
    _phantom: PhantomData<F>,
}

impl<F: Field> MultiLayerDifButterfly<F> {
    fn new() -> Self {
        Self {
            _phantom: PhantomData,
        }
    }
}

impl<F: Field> MultiLayerButterfly<F, DifButterfly<F>> for MultiLayerDifButterfly<F> {
    #[inline]
    fn apply_2_layers(
        &self,
        chunk_decomposition: &mut DoubleLayerBlockDecomposition<F>,
        ind: usize,
        twiddles_small: &[DifButterfly<F>],
        twiddles_large: &[DifButterfly<F>],
    ) {
        if ind == 0 {
            chunk_decomposition.fft_layer_double_twiddle(TwiddleFreeButterfly, twiddles_large[1]);
            chunk_decomposition.fft_layer_single_twiddle(TwiddleFreeButterfly);
        } else {
            chunk_decomposition
                .fft_layer_double_twiddle(twiddles_large[2 * ind], twiddles_large[2 * ind + 1]);
            chunk_decomposition.fft_layer_single_twiddle(twiddles_small[ind]);
        }
    }

    #[inline]
    fn apply_3_layers(
        &self,
        chunk_decomposition: &mut TripleLayerBlockDecomposition<F>,
        ind: usize,
        twiddles_small: &[DifButterfly<F>],
        twiddles_med: &[DifButterfly<F>],
        twiddles_large: &[DifButterfly<F>],
    ) {
        if ind == 0 {
            chunk_decomposition.fft_layer_quad_twiddle(TwiddleFreeButterfly, &twiddles_large[1..4]);
            chunk_decomposition.fft_layer_double_twiddle(TwiddleFreeButterfly, twiddles_med[1]);
            chunk_decomposition.fft_layer_single_twiddle(TwiddleFreeButterfly);
        } else {
            chunk_decomposition.fft_layer_quad_twiddle(
                twiddles_large[4 * ind],
                &twiddles_large[4 * ind + 1..4 * (ind + 1)],
            );
            chunk_decomposition
                .fft_layer_double_twiddle(twiddles_med[2 * ind], twiddles_med[2 * ind + 1]);
            chunk_decomposition.fft_layer_single_twiddle(twiddles_small[ind]);
        }
    }
}
