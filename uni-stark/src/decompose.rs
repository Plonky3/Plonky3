use alloc::vec;
use alloc::vec::Vec;

use itertools::izip;
use p3_field::{AbstractExtensionField, TwoAdicField};
use p3_matrix::dense::RowMajorMatrix;
use p3_maybe_rayon::prelude::*;
use p3_util::log2_strict_usize;
use tracing::instrument;

use self::dit_decompose::fft_decompose;

/// Decompose the quotient polynomial into chunks using a generalization of even-odd decomposition.
/// Then, arrange the results in a row-major matrix, so that each chunk of the decomposed polynomial
/// becomes `D` columns of the resulting matrix, where `D` is the field extension degree.
#[instrument(name = "decompose and flatten quotient", skip_all)]
pub fn decompose_and_flatten<Val, Challenge>(
    quotient_poly: Vec<Challenge>,
    shift: Challenge,
    log_chunks: usize,
) -> RowMajorMatrix<Val>
where
    Val: TwoAdicField,
    Challenge: AbstractExtensionField<Val> + TwoAdicField,
{
    let chunks: Vec<Vec<Challenge>> = decompose(quotient_poly, shift, log_chunks);
    let degree = chunks[0].len();
    let quotient_chunks_flattened: Vec<Val> = (0..degree)
        .into_par_iter()
        .flat_map_iter(|row| {
            chunks
                .iter()
                .flat_map(move |chunk| chunk[row].as_base_slice().iter().copied())
        })
        .collect();
    let challenge_ext_degree = <Challenge as AbstractExtensionField<Val>>::D;
    RowMajorMatrix::new(
        quotient_chunks_flattened,
        challenge_ext_degree << log_chunks,
    )
}

/// A generalization of even-odd decomposition.
fn decompose<F: TwoAdicField>(poly: Vec<F>, shift: F, log_chunks: usize) -> Vec<Vec<F>> {
    // For now, we use a naive recursive method.
    // A more optimized method might look similar to a decimation-in-time FFT,
    // but only the first `log_chunks` layers. It should also be parallelized.

    if log_chunks == 0 {
        return vec![poly];
    }

    fft_decompose(poly, shift, log_chunks)
}

mod dit_decompose {
    use alloc::vec::Vec;
    use p3_field::TwoAdicField;
    use p3_util::log2_strict_usize;

    pub fn fft_decompose<F: TwoAdicField>(mut poly: Vec<F>, shift: F, log_chunks: usize) -> Vec<Vec<F>> {
        let n = poly.len();
        let log_n = log2_strict_usize(n);
    
        // Ensure the poly length is a power of 2
        assert!(n.is_power_of_two());
    
        // Apply the shift for coset FFT
        apply_shift(&mut poly, &shift);
    
        // Transform the polynomial using a partial FFT (only up to log_chunks layers)
        partial_fft(&mut poly, log_n, log_chunks);
    
        // Split the transformed polynomial into chunks
        poly.chunks(1 << log_chunks).map(|chunk| chunk.to_vec()).collect()
    }
    
    fn apply_shift<F: TwoAdicField>(poly: &mut [F], shift: &F) {
        let n = poly.len();
        let g = F::two_adic_generator(log2_strict_usize(n));
    
        for (i, item) in poly.iter_mut().enumerate() {
            *item = *item * shift.pow(&g.exp_usize(i));
        }
    }    
    
    fn partial_fft<F: TwoAdicField>(poly: &mut [F], log_n: usize, log_chunks: usize) {
        let n = poly.len();
        if n <= 1 || log_chunks == 0 {
            return;
        }
    
        // Bit-reverse the order of the coefficients
        for i in 0..n {
            let rev = bit_reverse(i, log_n);
            if i < rev {
                poly.swap(i, rev);
            }
        }
    
        // Perform the butterfly operations only up to log_chunks layers
        let mut m = 1;
        for _ in 0..log_chunks {
            let w_m = F::two_adic_generator(m * 2);
            let mut k = 0;
            while k < n {
                let mut w = F::one();
                for j in 0..m {
                    let t = w * poly[k + j + m];
                    let u = poly[k + j];
                    poly[k + j] = u + t;
                    poly[k + j + m] = u - t;
                    w = w * w_m;
                }
                k += m * 2;
            }
            m *= 2;
        }
    }
    
    fn bit_reverse(mut x: usize, log_n: usize) -> usize {
        let mut result = 0;
        for _ in 0..log_n {
            result = (result << 1) | (x & 1);
            x >>= 1;
        }
        result
    }    
}

#[cfg(test)]
mod tests {
    use itertools::Itertools;
    use p3_baby_bear::BabyBear;
    use p3_dft::{Radix2Dit, TwoAdicSubgroupDft};
    use p3_field::AbstractField;
    use p3_util::reverse_slice_index_bits;
    use rand::{thread_rng, Rng};

    use super::*;

    // If we decompose evaluations over a coset s*g^i, we should get
    // evaluations over s^log_chunks * g^(log_chunks*i).
    #[test]
    fn test_decompose_coset() {
        type F = BabyBear;

        let mut rng = thread_rng();
        let dft = Radix2Dit::default();

        let log_n = 5;
        let n = 1 << log_n;
        let log_chunks = 3;
        let chunks = 1 << log_chunks;
        let shift = F::generator();

        let coeffs = (0..n).map(|_| rng.gen::<F>()).collect::<Vec<_>>();

        let coset_evals = dft.coset_dft(coeffs.clone(), shift);
        let mut decomp = decompose(coset_evals, shift, log_chunks);

        reverse_slice_index_bits(&mut decomp);

        for (i, e) in decomp.into_iter().enumerate() {
            let chunk_coeffs = coeffs.iter().cloned().skip(i).step_by(chunks).collect_vec();
            assert_eq!(
                dft.coset_dft(chunk_coeffs, shift.exp_power_of_2(log_chunks)),
                e
            );
        }
    }
}
