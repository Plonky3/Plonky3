//! Combining a run of values against the successive powers of the indeterminate.

/// The most values one combination covers.
///
/// Value `k` is shifted left by `k`, so the last shift is by one less than this.
///
/// A shift of a 128-bit word is only defined below 128, which is what fixes the ceiling.
///
/// The accumulator is wide enough for that ceiling.
/// The top value reaches degree `127 + 127`, inside the 256 bits the two halves hold.
pub(crate) const MAX_POWERS: usize = 128;

/// Combine values against the successive powers of the indeterminate, reducing once.
///
/// ```text
///     sum_k values_k * x^k
/// ```
///
/// # Algorithm
///
/// In the polynomial basis, multiplying by `x^k` shifts the coefficients up by `k`.
///
/// So the combination is shifts and exclusive ors, with no multiplication at all:
///
/// ```text
///     values_0  ->  |            v0            |
///     values_1  ->   |            v1           |
///     values_2  ->    |            v2          |
///                    ^ each value one place further up
///
///                   |   low half   |  high half |
/// ```
///
/// Reduction modulo the field polynomial is linear over `GF(2)`.
///
/// The accumulated sum therefore pays for it once, instead of every term paying separately.
///
/// # Panics
///
/// Panics on more values than the accumulator holds.
#[inline]
pub(crate) fn poly_dot_powers_128(values: impl ExactSizeIterator<Item = u128>) -> u128 {
    assert!(
        values.len() <= MAX_POWERS,
        "a combination covers at most {MAX_POWERS} values"
    );

    // The unreduced sum, as the low and high halves of one 256-bit polynomial.
    let (mut low, mut high) = (0u128, 0u128);

    for (k, value) in values.enumerate() {
        // What `x^k` leaves below degree 128.
        low ^= value << k;

        // What it carries above degree 128.
        //
        // The direct spelling is a shift right by `128 - k`.
        //
        // That is out of range at `k = 0`.
        //
        // Halving first and shifting one place less selects the same bits and stays in range.
        //
        // Nothing is lost to the halving.
        //
        // The bit it drops matters only at `k = 128`, which the length check rules out.
        high ^= (value >> 1) >> (127 - k);
    }

    super::reduce_128(low, high)
}

#[cfg(test)]
mod tests {
    use alloc::vec::Vec;

    use proptest::prelude::*;

    use super::*;
    use crate::clmul::poly_mul_128;

    /// The indeterminate, whose powers the combination runs through.
    const X: u128 = 2;

    /// The combination computed the slow way, one full product per term.
    fn reference(values: &[u128]) -> u128 {
        // Walk the powers of the indeterminate alongside the values, multiplying each pair.
        let mut power = 1u128;
        let mut sum = 0u128;
        for &value in values {
            sum ^= poly_mul_128(value, power);
            power = poly_mul_128(power, X);
        }
        sum
    }

    #[test]
    fn an_empty_run_combines_to_zero() {
        // An empty sum is zero, and the reduction of a zero accumulator must preserve that.
        assert_eq!(poly_dot_powers_128([].into_iter()), 0);
    }

    #[test]
    fn a_single_value_is_returned_unchanged() {
        // Fixture state: one value, weighted by `x^0 = 1`.
        //
        // Nothing is shifted and nothing overflows, so the reduction is the identity here.
        assert_eq!(poly_dot_powers_128([0x1234u128].into_iter()), 0x1234);
    }

    #[test]
    fn the_shift_path_matches_the_multiplying_reference_at_the_boundaries() {
        // The values that drive the shift and the reduction to their extremes.
        //
        //     0            contributes nothing, so the accumulator must stay clean
        //     1            the bare power of the indeterminate
        //     x^127        the highest degree, so its shift spills furthest
        //     all ones     every coefficient live, so every shifted bit collides
        let extremes = [0u128, 1, 1 << 127, u128::MAX];

        // Each extreme is placed at the widest run, where the last value shifts by 127.
        for value in extremes {
            let run = alloc::vec![value; MAX_POWERS];
            assert_eq!(
                poly_dot_powers_128(run.iter().copied()),
                reference(&run),
                "value={value:#x}"
            );
        }
    }

    #[test]
    #[should_panic(expected = "a combination covers at most")]
    fn a_run_past_the_accumulator_is_refused() {
        // Mutation: one value beyond the ceiling.
        //
        //     value 128 would shift left by 128, which leaves the word entirely.
        let run = alloc::vec![1u128; MAX_POWERS + 1];
        poly_dot_powers_128(run.iter().copied());
    }

    proptest! {
        #[test]
        fn the_shift_path_matches_the_multiplying_reference(
            values in prop::collection::vec(any::<u128>(), 0..=MAX_POWERS),
        ) {
            // Invariant: one reduction at the end agrees with one reduction per term.
            prop_assert_eq!(
                poly_dot_powers_128(values.iter().copied()),
                reference(&values),
            );
        }

        #[test]
        fn the_combination_is_additive_in_the_values(
            left in prop::collection::vec(any::<u128>(), 1..=32),
            seed: u64,
        ) {
            // The combination is a fixed linear map, so it must respect addition.
            //
            //     f(a) + f(b) = f(a + b)
            //
            // A map that mixed the terms differently per run would fail this.
            let right = left
                .iter()
                .enumerate()
                .map(|(index, value)| value ^ seed.wrapping_mul(index as u64 + 1) as u128)
                .collect::<Vec<_>>();
            let sum = left
                .iter()
                .zip(&right)
                .map(|(a, b)| a ^ b)
                .collect::<Vec<_>>();

            prop_assert_eq!(
                poly_dot_powers_128(left.iter().copied())
                    ^ poly_dot_powers_128(right.iter().copied()),
                poly_dot_powers_128(sum.iter().copied()),
            );
        }
    }
}
