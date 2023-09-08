use alloc::vec::Vec;

use itertools::Itertools;
use p3_field::Field;
use tracing::instrument;

/// Fold a polynomial
/// ```ignore
/// p(x) = p_even(x^2) + x p_odd(x^2)
/// ```
/// into
/// ```ignore
/// p_even(x) + beta p_odd(x)
/// ```
#[instrument(skip_all)]
pub(crate) fn fold_even_odd<F: Field>(poly: &[F], beta: F) -> Vec<F> {
    // We use the fact that
    //     p_e(x^2) = (p(x) + p(-x)) / 2
    //     p_o(x^2) = (p(x) - p(-x)) / (2 x)
    // that is,
    //     p_e(g^(2i)) = (p(g^i) + p(g^(n/2 + i))) / 2
    //     p_o(g^(2i)) = (p(g^i) - p(g^(n/2 + i))) / (2 g^i)
    // so
    //     result(g^(2i)) = p_e(g^(2i)) + beta p_o(g^(2i))
    //                    = (1/2 + beta/2) p(g^i) + (1/2 - beta/2) p(g^(n/2 + i))

    let n = poly.len();
    debug_assert!(n > 1);
    debug_assert!(n.is_power_of_two());

    let one_half = F::TWO.inverse();
    let half_beta = beta / F::TWO;
    let one_plus_beta_div_2 = one_half + half_beta;
    let one_minus_beta_div_2 = one_half - half_beta;

    let (first, second) = poly.split_at(n / 2);
    first
        .iter()
        .zip_eq(second)
        .map(|(a, b)| one_plus_beta_div_2 * *a + one_minus_beta_div_2 * *b)
        .collect()
}
