use alloc::vec;
use alloc::vec::Vec;

use itertools::izip;
use p3_field::{PackedField, TwoAdicField};
use p3_util::{ceil_div_usize, log2_strict_usize};
use tracing::instrument;

/// Fold a polynomial
/// ```ignore
/// p(x) = p_even(x^2) + x p_odd(x^2)
/// ```
/// into
/// ```ignore
/// p_even(x) + beta p_odd(x)
/// ```
#[instrument(skip_all, level = "debug")]
pub fn fold_even_odd<F: TwoAdicField>(poly: &[F], beta: F) -> Vec<F> {
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

    let n = poly.len();
    debug_assert!(n > 1);

    let log_n = log2_strict_usize(n);

    let g_inv = F::two_adic_generator(log_n).inverse();
    let one_half = F::TWO.inverse();
    let half_beta = beta * one_half;

    // beta/2 times successive powers of g_inv
    let powers = g_inv.packed_powers::<F::Packing>(half_beta);

    // pack first / second polys, rounding up to the nearest multiple of packing width
    let half_n = n / 2;
    let cutoff = (half_n / F::Packing::WIDTH) * F::Packing::WIDTH;

    let (first, second) = poly.split_at(n / 2);
    let first_leftover = F::Packing::from_fn(|i| {
        if cutoff + i < first.len() {
            first[i]
        } else {
            F::ZERO
        }
    });
    let second_leftover = F::Packing::from_fn(|i| {
        if cutoff + i < second.len() {
            second[cutoff + i]
        } else {
            F::ZERO
        }
    });
    let first = F::Packing::pack_slice(&first[..cutoff])
        .iter()
        .chain(core::iter::once(&first_leftover));
    let second = F::Packing::pack_slice(&second[..cutoff])
        .iter()
        .chain(core::iter::once(&second_leftover));

    // allocate and pack result, rounding up to the nearest multiple of packing width
    let nearest_mutliple_of_packing_width =
        F::Packing::WIDTH * ceil_div_usize(half_n, F::Packing::WIDTH);
    let mut res = vec![F::ZERO; nearest_mutliple_of_packing_width];
    let res_packed = F::Packing::pack_slice_mut(&mut res);

    let one_half = F::Packing::from_fn(|_| one_half);
    for (src, dst) in izip!(powers, first, second)
        .map(|(power, &a, &b)| (one_half + power) * a + (one_half - power) * b)
        .zip(res_packed.iter_mut())
    {
        *dst = src;
    }

    res.truncate(half_n);
    res
}

#[cfg(test)]
mod tests {
    use itertools::Itertools;
    use p3_baby_bear::BabyBear;
    use p3_dft::{Radix2Dit, TwoAdicSubgroupDft};
    use rand::{thread_rng, Rng};

    use super::*;

    #[test]
    fn test_fold_even_odd() {
        type F = BabyBear;

        let mut rng = thread_rng();

        let log_n = 10;
        let n = 1 << log_n;
        let coeffs = (0..n).map(|_| rng.gen::<F>()).collect::<Vec<_>>();

        let dft = Radix2Dit;
        let evals = dft.dft(coeffs.clone());

        let even_coeffs = coeffs.iter().cloned().step_by(2).collect_vec();
        let even_evals = dft.dft(even_coeffs);

        let odd_coeffs = coeffs.iter().cloned().skip(1).step_by(2).collect_vec();
        let odd_evals = dft.dft(odd_coeffs);

        let beta = rng.gen::<F>();
        let expected = izip!(even_evals, odd_evals)
            .map(|(even, odd)| even + beta * odd)
            .collect::<Vec<_>>();
        let got = fold_even_odd(&evals, beta);
        assert_eq!(expected, got);
    }
}
