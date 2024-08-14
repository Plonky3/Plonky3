use alloc::vec::Vec;

use itertools::{izip, Itertools};
use p3_field::TwoAdicField;
use p3_matrix::dense::RowMajorMatrix;
use p3_matrix::Matrix;
use p3_maybe_rayon::prelude::*;
use p3_util::{log2_strict_usize, reverse_slice_index_bits};
use tracing::instrument;

/// Fold a polynomial
/// ```ignore
/// p(x) = p_even(x^2) + x p_odd(x^2)
/// ```
/// into
/// ```ignore
/// p_even(x) + beta p_odd(x)
/// ```
/// Expects input to be bit-reversed evaluations.
#[instrument(skip_all, level = "debug")]
pub fn fold_even_odd<F: TwoAdicField>(poly: Vec<F>, beta: F) -> Vec<F> {
    // We use the fact that
    //     p_e(x^2) = (p(x) + p(-x)) / 2
    //     p_o(x^2) = (p(x) - p(-x)) / (2 x)
    // that is,
    //     p_e(g^(2i)) = (p(g^i) + p(g^(n/2 + i))) / 2
    //     p_o(g^(2i)) = (p(g^i) - p(g^(n/2 + i))) / (2 g^i)
    // so
    //     result(g^(2i)) = p_e(g^(2i)) + beta p_o(g^(2i))
    //                    = (1/2 + beta/2 g_inv^i) p(g^i)
    //                    + (1/2 - beta/2 g_inv^i) p(g^(n/2 + i))
    let m = RowMajorMatrix::new(poly, 2);
    let g_inv = F::two_adic_generator(log2_strict_usize(m.height()) + 1).inverse();
    let one_half = F::two().inverse();
    let half_beta = beta * one_half;

    // TODO: vectorize this (after we have packed extension fields)

    // beta/2 times successive powers of g_inv
    let mut powers = g_inv
        .shifted_powers(half_beta)
        .take(m.height())
        .collect_vec();
    reverse_slice_index_bits(&mut powers);

    m.par_rows()
        .zip(powers)
        .map(|(mut row, power)| {
            let (r0, r1) = row.next_tuple().unwrap();
            (one_half + power) * r0 + (one_half - power) * r1
        })
        .collect()
}

/// Fold a polynomial by an arity higher than 2.
pub fn fold<F: TwoAdicField>(poly: Vec<F>, beta: F, log_arity: usize) -> Vec<F> {
    // Let h = 2^log_arity. Write a polynomial p(x) as sum_{i=0}^{h-1} x^i p_i(x^h).
    // We seek a vector of evaluations of the polynomial p'(x) = sum_{i=1}^{h-1} beta^i p_i(x), given
    // evaluations of p(x).
    //
    // Let z be an h-th root of unity. We use the formula:
    // p_j(x) = 1/(h x^j) sum_{k=0}^{h-1} z^{-i*j}p(z^i*x), which is basically an inverse Fourier transform.
    // Plugging this in to the expression for p'(x) gives:
    // p'(x) = sum_{i,j=1}^{h-1} beta^i p(z^i * x) * beta^j * z^{-i*j} / (h x^j).

    let m = RowMajorMatrix::new(poly, 1 << log_arity);
    let g_inv = F::two_adic_generator(log2_strict_usize(m.height()) + log_arity).inverse();
    let normalizing_factor = F::from_canonical_u32(1 << log_arity).inverse();

    // TODO: vectorize this (after we have packed extension fields)

    // successive powers of g_inv
    let mut g_powers = g_inv.powers().take(m.height()).collect_vec();
    reverse_slice_index_bits(&mut g_powers);

    let root_of_unity = F::two_adic_generator(log_arity);
    let mut roots_of_unity = root_of_unity
        .inverse()
        .powers()
        .take(1 << log_arity)
        .collect_vec();
    reverse_slice_index_bits(&mut roots_of_unity);

    m.par_rows()
        .zip(g_powers)
        .map(|(row, power)| {
            assert!(row.len() == 1 << log_arity);
            row.zip(roots_of_unity.iter())
                .map(|(r, root)| {
                    r * normalizing_factor
                        * izip!(
                            beta.powers().take(1 << log_arity),
                            root.powers(),
                            power.powers()
                        )
                        .map(|(a, b, c)| a * b * c)
                        .sum()
                })
                .sum()
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use itertools::izip;
    use p3_baby_bear::BabyBear;
    use p3_dft::{Radix2Dit, TwoAdicSubgroupDft};
    use p3_field::AbstractField;
    use rand::{thread_rng, Rng};

    use super::*;

    #[test]
    fn test_fold_even_odd() {
        type F = BabyBear;

        let mut rng = thread_rng();

        let log_n = 10;
        let n = 1 << log_n;
        let coeffs = (0..n).map(|_| rng.gen::<F>()).collect::<Vec<_>>();

        let dft = Radix2Dit::default();
        let evals = dft.dft(coeffs.clone());

        let even_coeffs = coeffs.iter().cloned().step_by(2).collect_vec();
        let even_evals = dft.dft(even_coeffs);

        let odd_coeffs = coeffs.iter().cloned().skip(1).step_by(2).collect_vec();
        let odd_evals = dft.dft(odd_coeffs);

        let beta = rng.gen::<F>();
        let expected = izip!(even_evals, odd_evals)
            .map(|(even, odd)| even + beta * odd)
            .collect::<Vec<_>>();

        // fold_even_odd takes and returns in bitrev order.
        let mut folded = evals;
        reverse_slice_index_bits(&mut folded);
        folded = fold_even_odd(folded, beta);
        reverse_slice_index_bits(&mut folded);

        assert_eq!(expected, folded);
    }

    #[test]
    fn test_higher_arity_fold() {
        type F = BabyBear;

        let mut rng = thread_rng();

        let log_arity = 4;
        let log_n = 10;
        let n = 1 << log_n;
        let coeffs = (0..n).map(|_| rng.gen::<F>()).collect::<Vec<_>>();

        let dft = Radix2Dit::default();
        let evals = dft.dft(coeffs.clone());

        let beta = rng.gen::<F>();
        let mut result = evals.clone();
        let mut new_beta = beta;
        for _ in 0..log_arity {
            result = fold_even_odd(result, new_beta);
            new_beta = new_beta.square();
        }

        let folded = fold(evals, beta, log_arity);

        assert_eq!(result.len(), folded.len());

        assert_eq!(result, folded);
    }
}
