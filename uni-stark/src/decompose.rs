use alloc::vec;
use alloc::vec::Vec;

use itertools::izip;
use p3_field::{AbstractExtensionField, TwoAdicField};
use p3_matrix::dense::RowMajorMatrix;
use p3_maybe_rayon::{MaybeIntoParIter, ParallelIterator};
use p3_util::log2_strict_usize;

use crate::StarkConfig;

/// Decompose the quotient polynomial into chunks using a generalization of even-odd decomposition.
/// Then, arrange the results in a row-major matrix, so that each chunk of the decomposed polynomial
/// becomes `D` columns of the resulting matrix, where `D` is the field extension degree.
pub fn decompose_and_flatten<SC: StarkConfig>(
    quotient_poly: Vec<SC::Challenge>,
    shift: SC::Challenge,
    log_chunks: usize,
) -> RowMajorMatrix<SC::Val> {
    let chunks: Vec<Vec<SC::Challenge>> = decompose(quotient_poly, shift, log_chunks);
    let degree = chunks[0].len();
    let quotient_chunks_flattened: Vec<SC::Val> = (0..degree)
        .into_par_iter()
        .flat_map_iter(|row| {
            chunks
                .iter()
                .flat_map(move |chunk| chunk[row].as_base_slice().iter().copied())
        })
        .collect();
    let challenge_ext_degree = <SC::Challenge as AbstractExtensionField<SC::Val>>::D;
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

    let n = poly.len();
    debug_assert!(n > 1);
    let log_n = log2_strict_usize(n);
    let half_n = poly.len() / 2;
    let g_inv = F::two_adic_generator(log_n).inverse();

    let mut even = Vec::with_capacity(half_n);
    let mut odd = Vec::with_capacity(half_n);

    // Note that
    //     p_e(g^(2i)) = (p(g^i) + p(g^(n/2 + i))) / 2
    //     p_o(g^(2i)) = (p(g^i) - p(g^(n/2 + i))) / (2 s g^i)

    //     p_e(g^(2i)) = (a + b) / 2
    //     p_o(g^(2i)) = (a - b) / (2 s g^i)
    let one_half = F::two().inverse();
    let (first, second) = poly.split_at(half_n);
    for (g_inv_power, &a, &b) in izip!(g_inv.shifted_powers(shift.inverse()), first, second) {
        let sum = a + b;
        let diff = a - b;
        even.push(sum * one_half);
        odd.push(diff * one_half * g_inv_power);
    }

    let mut combined = decompose(even, shift.square(), log_chunks - 1);
    combined.extend(decompose(odd, shift.square(), log_chunks - 1));
    combined
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
        let dft = Radix2Dit;

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
