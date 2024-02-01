use alloc::vec;
use alloc::vec::Vec;

use p3_field::{AbstractExtensionField, TwoAdicField};
use p3_matrix::dense::RowMajorMatrix;
use p3_maybe_rayon::prelude::*;
use p3_util::log2_strict_usize;
use tracing::instrument;

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

    let n = poly.len();
    debug_assert!(n > 1);
    let log_n = log2_strict_usize(n);
    let half_n = poly.len() / 2;
    let g_inv = F::two_adic_generator(log_n).inverse();

    let one_half = F::two().inverse();
    let (first, second) = poly.split_at(half_n);

    // Note that
    //     p_e(g^(2i)) = (p(g^i) + p(g^(n/2 + i))) / 2
    //     p_o(g^(2i)) = (p(g^i) - p(g^(n/2 + i))) / (2 s g^i)

    //     p_e(g^(2i)) = (a + b) / 2
    //     p_o(g^(2i)) = (a - b) / (2 s g^i)
    let mut g_powers = g_inv.shifted_powers(shift.inverse());
    let g_powers = (0..first.len())
        .map(|_| g_powers.next().unwrap())
        .collect::<Vec<_>>();
    let (even, odd): (Vec<_>, Vec<_>) = first
        .par_iter()
        .zip(second.par_iter())
        .zip(g_powers.par_iter())
        .map(|((&a, &b), g_inv_power)| {
            let sum = a + b;
            let diff = a - b;
            (sum * one_half, diff * one_half * *g_inv_power)
        })
        .unzip();

    let (even_decomp, odd_decomp) = rayon::join(
        || decompose(even, shift.square(), log_chunks - 1),
        || decompose(odd, shift.square(), log_chunks - 1),
    );

    let mut combined = even_decomp;
    combined.extend(odd_decomp);
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
