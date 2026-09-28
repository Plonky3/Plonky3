//! The exponent lift: the bits of an integer select powers of one multiplicative generator.
//!
//! ```text
//!     g        a generator of the challenge field's multiplicative group
//!     G_t      g^(2^t), by repeated squaring
//!
//!     prod_t (1 + x_t * (G_t - 1))  =  prod_{t : x_t = 1} G_t  =  g^(sum_t x_t * 2^t)  =  g^x
//! ```
//!
//! Every leaf factor is one or the power its bit selects, so a product of leaves is a power of `g`.

use alloc::vec::Vec;
use core::iter;

use p3_field::Field;
use p3_multilinear_util::point::Point;

/// Returns `g^(2^t)` for every `t < count`.
pub(super) fn generator_squarings<F: Field>(count: usize) -> Vec<F> {
    // Squaring doubles the exponent, so the walk visits consecutive powers of two.
    iter::successors(Some(F::GENERATOR), |power| Some(power.square()))
        .take(count)
        .collect()
}

/// Returns the factor-tree leaf weights `g^(2^(i + j)) - 1`, with `i` major and `j` minor.
///
/// The bit pair `(i, j)` of the two factors contributes `2^(i + j)` to their product.
pub(super) fn factor_weights<F: Field>(width: usize) -> Vec<F> {
    // The largest bit pair reaches the exponent 2^(2w - 2).
    let powers = generator_squarings::<F>(2 * width - 1);
    let mut weights = Vec::with_capacity(width * width);
    for i in 0..width {
        weights.extend(powers[i..i + width].iter().map(|&power| power - F::ONE));
    }
    weights
}

/// Returns the result-tree leaf weights `g^(2^t) - 1` for every bit `t` of the two limbs.
///
/// The low limb holds bits `0..w` and the high limb bits `w..2w` of the claimed product.
pub(super) fn result_weights<F: Field>(width: usize) -> Vec<F> {
    generator_squarings::<F>(2 * width)
        .into_iter()
        .map(|power| power - F::ONE)
        .collect()
}

/// Evaluates the multilinear extension of a table at a point.
pub(super) fn evaluate<F: Field>(table: &[F], point: &[F]) -> F {
    // The first coordinate addresses the most significant index bit.
    debug_assert_eq!(table.len(), 1 << point.len());
    Point::new(point)
        .equality_weights_msb()
        .into_iter()
        .zip(table)
        .map(|(weight, &value)| weight * value)
        .sum()
}

#[cfg(test)]
mod tests {
    use p3_binary_field::BinaryField128;
    use p3_field::PrimeCharacteristicRing;
    use proptest::prelude::*;

    use super::*;

    type F = BinaryField128;

    // The lift of an integer below 2^128, split into two exponentiations by 64-bit halves.
    fn lift(value: u128) -> F {
        let high = F::GENERATOR
            .exp_power_of_2(64)
            .exp_u64((value >> 64) as u64);
        F::GENERATOR.exp_u64(value as u64) * high
    }

    // The product of the selected factor-tree leaves of one row.
    fn factor_root(a: u64, b: u64, width: usize) -> F {
        let weights = factor_weights::<F>(width);
        (0..width * width)
            .filter(|&cell| (a >> (cell / width)) & (b >> (cell % width)) & 1 == 1)
            .map(|cell| weights[cell] + F::ONE)
            .product()
    }

    // The product of the selected result-tree leaves of one row.
    fn result_root(low: u64, high: u64, width: usize) -> F {
        let weights = result_weights::<F>(width);
        let limbs = u128::from(low) | (u128::from(high) << width);
        (0..2 * width)
            .filter(|&bit| (limbs >> bit) & 1 == 1)
            .map(|bit| weights[bit] + F::ONE)
            .product()
    }

    #[test]
    fn the_weight_tables_have_the_documented_layout() {
        // Fixture state: four-bit words, so sixteen bit pairs and eight result bits.
        let squarings = generator_squarings::<F>(8);
        let factor = factor_weights::<F>(4);
        let result = result_weights::<F>(4);

        // Pair (i, j) sits at i * 4 + j and carries the power 2^(i + j).
        assert_eq!(factor.len(), 16);
        assert_eq!(factor[0], squarings[0] - F::ONE);
        assert_eq!(factor[4 + 3], squarings[4] - F::ONE);
        assert_eq!(factor[15], squarings[6] - F::ONE);

        // Result bit t carries the power 2^t.
        assert_eq!(result.len(), 8);
        assert_eq!(result[7], squarings[7] - F::ONE);
    }

    #[test]
    fn the_wraparound_is_the_one_collision_of_a_128_bit_lift() {
        // The multiplicative group has order 2^128 - 1, so that exponent lifts to one.
        assert_eq!(lift(u128::MAX), F::ONE);

        // A zero factor also lifts to one.
        //
        // Claiming both limbs all-ones for a zero product therefore passes the lift alone.
        assert_eq!(factor_root(0, 17, 64), F::ONE);
        assert_eq!(result_root(u64::MAX, u64::MAX, 64), F::ONE);
    }

    proptest! {
        #[test]
        fn both_trees_lift_their_integer(a: u64, b: u64) {
            // Invariant: each root is g raised to the integer its bits encode.
            let product = u128::from(a) * u128::from(b);
            prop_assert_eq!(factor_root(a, b, 64), lift(product));
            prop_assert_eq!(result_root(product as u64, (product >> 64) as u64, 64), lift(product));
        }

        #[test]
        fn a_32_bit_product_lifts_below_the_group_order(a: u32, b: u32) {
            // Invariant: the 64-bit exponent is far below 2^128 - 1, so no reduction occurs.
            let product = u64::from(a) * u64::from(b);
            let (low, high) = (product & 0xFFFF_FFFF, product >> 32);
            prop_assert_eq!(factor_root(a.into(), b.into(), 32), F::GENERATOR.exp_u64(product));
            prop_assert_eq!(result_root(low, high, 32), F::GENERATOR.exp_u64(product));
        }

        #[test]
        fn a_wrong_limb_moves_the_result_root(a: u64, b: u64, flip in 0usize..128) {
            // Mutation: flip one bit of the honest limbs.
            //
            // The lift is injective below 2^128 - 1, so the roots must now differ.
            let product = u128::from(a) * u128::from(b);
            let wrong = product ^ (1u128 << flip);
            prop_assume!(wrong != u128::MAX);
            let (low, high) = (wrong as u64, (wrong >> 64) as u64);
            prop_assert_ne!(result_root(low, high, 64), factor_root(a, b, 64));
        }
    }
}
