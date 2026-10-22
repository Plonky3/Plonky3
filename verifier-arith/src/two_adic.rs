//! Kernels over two-adic multiplicative cosets.

use alloc::vec::Vec;

use p3_field::coset::TwoAdicMultiplicativeCoset;
use p3_field::{Dup, Field, PrimeCharacteristicRing, TwoAdicField};
use p3_util::{log2_strict_usize, reverse_bits_len};

use crate::Arith;

/// Lagrange selectors of a space, evaluated at a point outside it.
///
/// A selector of a subset is nonzero on the subset and zero on the rest of the space.
///
/// These selectors are not normalized.
#[derive(Clone, Debug)]
pub struct LagrangeSelectors<T> {
    /// A Lagrange selector corresponding to the first point in the space.
    pub is_first_row: T,
    /// A Lagrange selector corresponding to the last point in the space.
    pub is_last_row: T,
    /// A Lagrange selector corresponding the subset of all but the last point.
    pub is_transition: T,
    /// The inverse of the vanishing polynomial which is a Lagrange selector corresponding to the empty set
    pub inv_vanishing: T,
}

/// The interpolant at `point` of a row on a two-adic coset, folded by halving.
///
/// The row holds `2^log_arity` values in bit-reversed order on `s * <w>`.
///
/// Here `s = g^rev(index)`, `g` has order `2^(log_height + log_arity)` and `w` has order `2^log_arity`.
///
/// Each round pairs `x` with `-x`, so the only index-dependent factor is a power of `1/g`.
///
/// # Panics
///
/// When the row does not hold `2^log_arity` values.
pub fn fold_coset_row_by_halving<A>(
    arith: &mut A,
    index: &A::Index,
    log_height: usize,
    log_arity: usize,
    beta: A::EF,
    evals: &[A::EF],
) -> A::EF
where
    A: Arith + ?Sized,
    A::Val: TwoAdicField,
{
    assert_eq!(evals.len(), 1 << log_arity, "one evaluation per row entry");

    // Round `r` folds with `beta^(2^r)`, and pairs entries `2k` and `2k + 1`, at `x` and `-x`:
    //
    //     (p(x) + p(-x)) / 2 + beta * (p(x) - p(-x)) / (2x).
    let mut s_inv = arith.reversed_index_power(
        A::Val::two_adic_generator(log_height + log_arity).inverse(),
        index,
        log_height,
    );
    let half = A::Val::ONE.halve();

    let mut row = evals.to_vec();
    let mut beta = beta;
    for round in 0..log_arity {
        let log_pairs = log_arity - round - 1;
        let w_inv = A::Val::two_adic_generator(log_pairs + 1).inverse();
        row = row
            .as_chunks::<2>()
            .0
            .iter()
            .enumerate()
            .map(|(k, [lo, hi])| {
                // Pair `k` sits at `x = s^(2^round) * w_round^rev(k)`.
                let x_inv_half =
                    s_inv.dup() * (w_inv.exp_u64(reverse_bits_len(k, log_pairs) as u64) * half);
                (lo.dup() + hi.dup()) * A::F::from(half)
                    + (lo.dup() - hi.dup()) * beta.dup() * x_inv_half
            })
            .collect();
        s_inv = s_inv.square();
        beta = beta.square();
    }

    row.pop().expect("one value remains after folding")
}

/// `generator^rev(index)`, where `rev` reverses the low `bits` bits of the index.
///
/// Built from the index bits and selections alone, which is how [`Arith::reversed_index_power`] computes it by default.
pub fn reversed_index_power_by_bits<A: Arith + ?Sized>(
    arith: &mut A,
    generator: A::Val,
    index: &A::Index,
    bits: usize,
) -> A::F {
    let index_bits = arith.index_bits(index, bits);

    // Bit `i` of the index is bit `bits - 1 - i` of the exponent.
    //
    //     generator^rev(index) = prod_i (bit_i ? generator^(2^(bits - 1 - i)) : 1)
    let mut power = generator;
    let mut result = A::F::ONE;
    for bit in index_bits.iter().rev() {
        let factor = arith.select_base(bit, A::F::from(power), A::F::ONE);
        result *= factor;
        power = power.square();
    }
    result
}

/// The vanishing polynomial `(x / shift)^|H| - 1` of a coset `shift * H`, at `point`.
pub fn vanishing_at<A>(coset: &TwoAdicMultiplicativeCoset<A::Val>, point: &A::EF) -> A::EF
where
    A: Arith<Val: TwoAdicField>,
{
    (point.dup() * A::F::from(coset.shift_inverse())).exp_power_of_2(coset.log_size()) - A::EF::ONE
}

/// The Lagrange selectors of a coset `shift * H` at `point`.
///
/// With `u = point / shift`, `Z(u) = u^|H| - 1` and `h` the generator of `H`:
///
/// - `is_first_row = Z(u) / (u - 1)`.
/// - `is_last_row = Z(u) / (u - h^-1)`.
/// - `is_transition = u - h^-1`.
/// - `inv_vanishing = 1 / Z(u)`.
///
/// # Errors
///
/// `in_coset()` when `point` lies in the coset.
pub fn selectors_at<A, E>(
    arith: &mut A,
    coset: &TwoAdicMultiplicativeCoset<A::Val>,
    point: &A::EF,
    in_coset: impl FnOnce() -> E,
) -> Result<LagrangeSelectors<A::EF>, E>
where
    A: Arith<Val: TwoAdicField>,
{
    let unshifted = point.dup() * A::F::from(coset.shift_inverse());
    let z_h = unshifted.exp_power_of_2(coset.log_size()) - A::EF::ONE;
    let is_transition = unshifted.dup() - A::F::from(coset.subgroup_generator().inverse());

    // A nonzero `Z(u)` keeps `u` off the subgroup, so every denominator is nonzero.
    let inverses = arith.batch_inverse(
        &[unshifted - A::EF::ONE, is_transition.dup(), z_h.dup()],
        |_| in_coset(),
    )?;
    let [first, last, vanishing] =
        <[A::EF; 3]>::try_from(inverses).unwrap_or_else(|_| unreachable!("one inverse per input"));

    Ok(LagrangeSelectors {
        is_first_row: z_h.dup() * first,
        is_last_row: z_h * last,
        is_transition,
        inv_vanishing: vanishing,
    })
}

/// The polynomial with the given values on the coset `shift * H`, in natural order, at `point`.
///
/// With `x_i = shift * w^i` and `n = |H|`, the barycentric form is
///
/// ```text
///     f(z) = (z^n - shift^n) / (n * shift^n) * sum_i v_i * x_i / (z - x_i)
/// ```
///
/// # Errors
///
/// `in_coset()` when `point` lies in the coset.
pub fn interpolate_coset_at<A, E>(
    arith: &mut A,
    shift: A::Val,
    values: &[A::Val],
    point: &A::EF,
    in_coset: impl FnOnce() -> E,
) -> Result<A::EF, E>
where
    A: Arith<Val: TwoAdicField>,
{
    let log_size = log2_strict_usize(values.len());
    let points: Vec<A::Val> = A::Val::two_adic_generator(log_size)
        .shifted_powers(shift)
        .take(values.len())
        .collect();
    let differences: Vec<A::EF> = points
        .iter()
        .map(|&x| point.dup() - A::F::from(x))
        .collect();
    let inverses = arith.batch_inverse(&differences, |_| in_coset())?;

    let shift_pow = shift.exp_power_of_2(log_size);
    let scale = (A::Val::from_usize(values.len()) * shift_pow).inverse();
    let vanishing = point.dup().exp_power_of_2(log_size) - A::F::from(shift_pow);
    let sum: A::EF = values
        .iter()
        .zip(&points)
        .zip(inverses)
        .map(|((&v, &x), inverse)| inverse * A::F::from(v * x))
        .sum();
    Ok(vanishing * A::F::from(scale) * sum)
}

/// Every periodic column of a trace domain `shift * H` at `point`.
///
/// A column of period `p` repeats every `p` rows, so it is the polynomial on the coset `shift^(n/p) * H_p` at `point^(n/p)`.
///
/// # Errors
///
/// `in_coset()` when a folded point lies in its coset.
pub fn periodic_values_at<A, E>(
    arith: &mut A,
    coset: &TwoAdicMultiplicativeCoset<A::Val>,
    columns: &[Vec<A::Val>],
    point: &A::EF,
    mut in_coset: impl FnMut() -> E,
) -> Result<Vec<A::EF>, E>
where
    A: Arith<Val: TwoAdicField>,
{
    columns
        .iter()
        .map(|column| {
            let folds = coset.log_size() - log2_strict_usize(column.len());
            let shift = coset.shift().exp_power_of_2(folds);
            let folded = point.dup().exp_power_of_2(folds);
            interpolate_coset_at(arith, shift, column, &folded, &mut in_coset)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use p3_baby_bear::BabyBear;
    use p3_field::extension::BinomialExtensionField;
    use p3_field::{BasedVectorSpace, Field};
    use p3_util::reverse_bits_len;
    use proptest::prelude::*;

    use super::*;
    use crate::Native;

    type F = BabyBear;
    type EF = BinomialExtensionField<BabyBear, 4>;
    type N = Native<F, EF>;

    fn ef(coefficients: [u32; 4]) -> EF {
        EF::from_basis_coefficients_fn(|i| F::new(coefficients[i]))
    }

    proptest! {
        #[test]
        fn reversed_index_power_matches_exponentiation(log_size in 0usize..=20, index: u32) {
            // Mask the index to the domain, as a query index always is.
            let index = index as usize & ((1 << log_size) - 1);
            let expected = F::two_adic_generator(log_size)
                .exp_u64(reverse_bits_len(index, log_size) as u64);

            let generator = F::two_adic_generator(log_size);
            prop_assert_eq!(N::new().reversed_index_power(generator, &index, log_size), expected);
            prop_assert_eq!(
                reversed_index_power_by_bits(&mut N::new(), generator, &index, log_size),
                expected
            );
        }

        #[test]
        fn interpolation_on_a_coset_matches_the_interpolant(log_size in 0usize..=4, point: [u32; 4]) {
            // A random polynomial of degree below |H|, checked at its coset values and at `point`.
            let n = 1usize << log_size;
            let coefficients: Vec<F> = (0..n).map(|i| F::new(7 * i as u32 + 3)).collect();
            let eval = |x: EF| coefficients.iter().rev().fold(EF::ZERO, |acc, &c| acc * x + c);
            let shift = F::GENERATOR;
            let values: Vec<F> = F::two_adic_generator(log_size)
                .shifted_powers(shift)
                .take(n)
                .map(|x| eval(EF::from(x)).as_basis_coefficients_slice()[0])
                .collect();
            let point = ef(point);
            let got = interpolate_coset_at(&mut N::new(), shift, &values, &point, || ());
            prop_assume!(got.is_ok());
            prop_assert_eq!(got.unwrap(), eval(point));
        }

        #[test]
        fn selectors_match_their_definition(log_size in 1usize..=12, point: [u32; 4]) {
            let coset = TwoAdicMultiplicativeCoset::new(F::GENERATOR, log_size).unwrap();
            let point = ef(point);
            let unshifted = point * coset.shift_inverse();
            let z_h = unshifted.exp_power_of_2(log_size) - EF::ONE;
            prop_assume!(!z_h.is_zero());

            let sels = selectors_at(&mut N::new(), &coset, &point, || ()).unwrap();
            let h_inv = coset.subgroup_generator().inverse();

            prop_assert_eq!(sels.is_first_row, z_h / (unshifted - EF::ONE));
            prop_assert_eq!(sels.is_last_row, z_h / (unshifted - h_inv));
            prop_assert_eq!(sels.is_transition, unshifted - h_inv);
            prop_assert_eq!(sels.inv_vanishing, z_h.inverse());
            prop_assert_eq!(vanishing_at::<N>(&coset, &point), z_h);
        }
    }

    #[test]
    fn selectors_reject_a_point_of_the_coset() {
        // The shift itself is the first point of the coset.
        let coset = TwoAdicMultiplicativeCoset::new(F::GENERATOR, 3).unwrap();
        let point = EF::from(coset.shift());

        assert_eq!(
            selectors_at(&mut N::new(), &coset, &point, || "in coset").unwrap_err(),
            "in coset"
        );
    }
}
