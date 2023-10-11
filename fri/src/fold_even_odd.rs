use alloc::vec;
use alloc::vec::Vec;

use itertools::izip;
use p3_field::{PackedField, Powers, TwoAdicField};
use p3_util::log2_strict_usize;
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

    if n < F::Packing::WIDTH {
        let mut res = vec![F::ZERO; n / 2];
        let (first, second) = poly.split_at(n / 2);
        fold_even_odd_packed::<F, F>(res.iter_mut(), first.iter(), second.iter(), n, beta);
        res
    } else {
        let half_n = n / 2;
        let nearest_mutliple_of_packing_width = (half_n + F::Packing::WIDTH - 1) / F::Packing::WIDTH;
        let cutoff = (half_n / F::Packing::WIDTH) * F::Packing::WIDTH;

        let mut res = vec![F::ZERO; nearest_mutliple_of_packing_width * F::Packing::WIDTH];
        let res_packed = F::Packing::pack_slice_mut(&mut res);

        let (first, second) = poly.split_at(n / 2);
        let first_leftover =
            F::Packing::from_fn(|i| if cutoff + i < first.len() { first[i] } else { F::ZERO });
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

        fold_even_odd_packed::<F, F::Packing>(res_packed.iter_mut(), first, second, n, beta);
        res.truncate(half_n);
        res
    }
}

fn fold_even_odd_packed<'a, F: TwoAdicField, P: PackedField<Scalar = F>>(
    dst: impl Iterator<Item = &'a mut P>,
    first: impl Iterator<Item = &'a P>,
    second: impl Iterator<Item = &'a P>,
    n: usize,
    beta: F,
) {
    let log_n = log2_strict_usize(n);

    let g_inv = F::two_adic_generator(log_n).inverse();
    let one_half = F::TWO.inverse();
    let half_beta = beta * one_half;

    // beta/2 times successive powers of g_inv
    let powers = Powers {
        base: P::from_fn(|_| g_inv),
        current: P::from_fn(|_| half_beta),
    };

    let one_half = P::from_fn(|_| one_half);
    for (src, dst) in izip!(powers, first, second)
        .map(|(power, &a, &b)| (one_half + power) * a + (one_half - power) * b)
        .zip(dst)
    {
        *dst = src;
    }
}
