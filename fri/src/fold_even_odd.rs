use alloc::vec::Vec;

use itertools::izip;
use p3_field::{Powers, TwoAdicField};
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
pub(crate) fn fold_even_odd<F: TwoAdicField>(poly: &[F], beta: F) -> Vec<F> {
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
    let powers = Powers {
        base: g_inv,
        current: half_beta,
    };

    let (first, second) = poly.split_at(n / 2);
    izip!(powers, first, second)
        .map(|(power, a, b)| (one_half + power) * *a + (one_half - power) * *b)
        .collect()
}
