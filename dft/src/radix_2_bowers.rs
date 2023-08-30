use alloc::vec;
use alloc::vec::Vec;

use p3_field::{Field, TwoAdicField};
use p3_matrix::dense::RowMajorMatrix;
use p3_matrix::Matrix;
use p3_util::log2_strict_usize;

use crate::util::{divide_by_height, reverse_bits, reverse_matrix_index_bits};
use crate::{dif_butterfly, dit_butterfly, TwoAdicSubgroupDft};

/// The Bowers G FFT algorithm.
/// See: "Improved Twiddle Access for Fast Fourier Transforms"
#[derive(Default, Clone)]
pub struct Radix2Bowers;

impl<F: TwoAdicField> TwoAdicSubgroupDft<F> for Radix2Bowers {
    fn dft_batch(&self, mut mat: RowMajorMatrix<F>) -> RowMajorMatrix<F> {
        reverse_matrix_index_bits(&mut mat);
        dft_input_reversed(&mut mat);
        mat
    }

    /// Compute the inverse DFT of each column in `mat`.
    fn idft_batch(&self, mut mat: RowMajorMatrix<F>) -> RowMajorMatrix<F> {
        idft_output_reversed(&mut mat);
        reverse_matrix_index_bits(&mut mat);
        mat
    }

    fn lde_batch(&self, mut mat: RowMajorMatrix<F>, added_bits: usize) -> RowMajorMatrix<F> {
        idft_output_reversed(&mut mat);
        bit_reversed_zero_pad(&mut mat, added_bits);
        dft_input_reversed(&mut mat);
        mat
    }

    fn coset_lde_batch(
        &self,
        mut mat: RowMajorMatrix<F>,
        added_bits: usize,
        shift: F,
    ) -> RowMajorMatrix<F> {
        let h = mat.height();

        idft_output_reversed(&mut mat);

        // Rescale coefficients; see the default impl for an explanation.
        // The difference here is that we stay in bit-reversed order.
        for (row, power) in shift.powers().take(h).enumerate() {
            let row = mat.row_mut(reverse_bits(row, h));
            row.iter_mut().for_each(|coeff| {
                *coeff *= power;
            })
        }

        bit_reversed_zero_pad(&mut mat, added_bits);

        dft_input_reversed(&mut mat);

        mat
    }
}

fn dft_input_reversed<F: TwoAdicField>(mat: &mut RowMajorMatrix<F>) {
    let h = mat.height();
    let log_h = log2_strict_usize(h);

    let root = F::primitive_root_of_unity(log_h);
    let twiddles: Vec<F> = root.powers().take(h / 2).collect();

    bowers_g(mat, &twiddles);
}

fn idft_output_reversed<F: TwoAdicField>(mat: &mut RowMajorMatrix<F>) {
    let h = mat.height();
    let log_h = log2_strict_usize(h);

    let root_inv = F::primitive_root_of_unity(log_h).inverse();
    let twiddles: Vec<F> = root_inv.powers().take(h / 2).collect();

    bowers_g_t(mat, &twiddles);
    divide_by_height(mat);
}

/// Append zeros to the "end" of the given matrix, except that the matrix is in bit-reversed order,
/// so in actuality we're interleaving zero rows.
#[inline]
fn bit_reversed_zero_pad<F: Field>(mat: &mut RowMajorMatrix<F>, added_bits: usize) {
    if added_bits == 0 {
        return;
    }

    // This is equivalent to:
    //     reverse_matrix_index_bits(mat);
    //     mat
    //         .values
    //         .resize(mat.values.len() << added_bits, F::ZERO);
    //     reverse_matrix_index_bits(mat);
    // But rather than implement it with bit reversals, we directly construct the resulting matrix,
    // whose rows are zero except for rows whose low `added_bits` bits are zero.

    let w = mat.width;
    let mut values = vec![F::ZERO; mat.values.len() << added_bits];
    for i in (0..mat.values.len()).step_by(w) {
        values[(i << added_bits)..((i << added_bits) + w)].copy_from_slice(&mat.values[i..i + w]);
    }
    *mat = RowMajorMatrix::new(values, w);
}

/// Bowers G FFT, minus the bit reversal.
fn bowers_g<F: Field>(mat: &mut RowMajorMatrix<F>, twiddles: &[F]) {
    let log_h = log2_strict_usize(mat.height());
    for log_half_block_size in 0..log_h {
        bowers_g_layer(mat, log_half_block_size, twiddles);
    }
}

fn bowers_g_t<F: Field>(mat: &mut RowMajorMatrix<F>, twiddles: &[F]) {
    let log_h = log2_strict_usize(mat.height());
    for log_half_block_size in (0..log_h).rev() {
        bowers_g_t_layer(mat, log_half_block_size, twiddles);
    }
}

/// One layer of a Bowers G network. Equivalent to `bowers_g_t_layer` except for the butterfly.
fn bowers_g_layer<F: Field>(
    mat: &mut RowMajorMatrix<F>,
    log_half_block_size: usize,
    twiddles: &[F],
) {
    let h = mat.height();
    let log_block_size = log_half_block_size + 1;
    let half_block_size = 1 << log_half_block_size;
    let block_size = 1 << log_block_size;
    let num_blocks = h >> log_block_size;

    for block in 0..num_blocks {
        let block_start = block * block_size;
        let twiddle = twiddles[reverse_bits(block, num_blocks) * half_block_size];
        for butterfly_hi in block_start..block_start + half_block_size {
            let butterfly_lo = butterfly_hi + half_block_size;
            dif_butterfly(mat, butterfly_hi, butterfly_lo, twiddle);
        }
    }
}

/// One layer of a Bowers G^T network. Equivalent to `bowers_g_layer` except for the butterfly.
fn bowers_g_t_layer<F: Field>(
    mat: &mut RowMajorMatrix<F>,
    log_half_block_size: usize,
    twiddles: &[F],
) {
    let h = mat.height();
    let log_block_size = log_half_block_size + 1;
    let half_block_size = 1 << log_half_block_size;
    let block_size = 1 << log_block_size;
    let num_blocks = h >> log_block_size;

    for block in 0..num_blocks {
        let block_start = block * block_size;
        let twiddle = twiddles[reverse_bits(block, num_blocks) * half_block_size];
        for butterfly_hi in block_start..block_start + half_block_size {
            let butterfly_lo = butterfly_hi + half_block_size;
            dit_butterfly(mat, butterfly_hi, butterfly_lo, twiddle);
        }
    }
}

#[cfg(test)]
mod tests {
    use p3_baby_bear::BabyBear;
    use p3_field::AbstractField;
    use p3_goldilocks::Goldilocks;
    use p3_matrix::dense::RowMajorMatrix;
    use rand::thread_rng;

    use crate::radix_2_bowers::Radix2Bowers;
    use crate::{NaiveDft, TwoAdicSubgroupDft};

    #[test]
    fn dft_matches_naive() {
        type F = BabyBear;
        let mut rng = thread_rng();
        let mat = RowMajorMatrix::<F>::rand(&mut rng, 64, 3);
        let dft_naive = NaiveDft.dft_batch(mat.clone());
        let dft_bowers = Radix2Bowers.dft_batch(mat);
        assert_eq!(dft_naive, dft_bowers);
    }

    #[test]
    fn idft_matches_naive() {
        type F = BabyBear;
        let mut rng = thread_rng();
        let mat = RowMajorMatrix::<F>::rand(&mut rng, 64, 3);
        let idft_naive = NaiveDft.idft_batch(mat.clone());
        let idft_bowers = Radix2Bowers.idft_batch(mat);
        assert_eq!(idft_naive, idft_bowers);
    }

    #[test]
    fn lde_matches_naive() {
        type F = BabyBear;
        let mut rng = thread_rng();
        let mat = RowMajorMatrix::<F>::rand(&mut rng, 64, 3);
        let lde_naive = NaiveDft.lde_batch(mat.clone(), 1);
        let lde_bowers = Radix2Bowers.lde_batch(mat, 1);
        assert_eq!(lde_naive, lde_bowers);
    }

    #[test]
    fn coset_lde_matches_naive() {
        type F = BabyBear;
        let mut rng = thread_rng();
        let mat = RowMajorMatrix::<F>::rand(&mut rng, 64, 3);
        let shift = F::multiplicative_group_generator();
        let coset_lde_naive = NaiveDft.coset_lde_batch(mat.clone(), 1, shift);
        let coset_lde_bowers = Radix2Bowers.coset_lde_batch(mat, 1, shift);
        assert_eq!(coset_lde_naive, coset_lde_bowers);
    }

    #[test]
    fn dft_idft_consistency() {
        type F = Goldilocks;
        let mut rng = thread_rng();
        let original = RowMajorMatrix::<F>::rand(&mut rng, 64, 3);
        let dft = Radix2Bowers.dft_batch(original.clone());
        let idft = Radix2Bowers.idft_batch(dft);
        assert_eq!(original, idft);
    }
}
