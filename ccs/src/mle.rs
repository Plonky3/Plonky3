//! Closed-form multilinear extensions, with points read most significant bit first.

use p3_field::Field;

/// Multilinear extension of the indicator y = x + offset, with no wrap past the last row.
///
/// Runs a ripple-carry adder from bit zero, keeping one weight per carry value.
///
/// # Panics
///
/// - The points differ in length.
pub(crate) fn shift_indicator<EF: Field>(x: &[EF], y: &[EF], offset: usize) -> EF {
    let n = x.len();
    assert_eq!(n, y.len(), "a shift relates two points of the same length");

    // Offset >= 2^n pushes every row past the end.
    if n < usize::BITS as usize && offset >> n != 0 {
        return EF::ZERO;
    }

    // Start with no carry.
    let mut no_carry = EF::ONE;
    let mut carry = EF::ZERO;

    // Bit zero is the last coordinate.
    for (bit, (&xb, &yb)) in x.iter().rev().zip(y.iter().rev()).enumerate() {
        // Equality factors for both values of each side.
        let (x0, x1) = (EF::ONE - xb, xb);
        let (y0, y1) = (EF::ONE - yb, yb);

        // Output bit equal to, above, or below the input bit.
        let same = x0 * y0 + x1 * y1;
        let flip_up = x0 * y1;
        let flip_down = x1 * y0;

        // Offset bit here; zero past the machine word.
        let offset_bit = u32::try_from(bit)
            .ok()
            .and_then(|bit| offset.checked_shr(bit))
            .is_some_and(|rest| rest & 1 == 1);

        // Adder transitions on (input bit, output bit) for each offset bit and carry.
        (no_carry, carry) = if offset_bit {
            (no_carry * flip_up, no_carry * flip_down + carry * same)
        } else {
            (no_carry * same + carry * flip_up, carry * flip_down)
        };
    }

    // A final carry would overflow past the last row.
    no_carry
}

#[cfg(test)]
mod tests {
    use alloc::vec::Vec;

    use p3_baby_bear::BabyBear;
    use p3_field::extension::BinomialExtensionField;
    use p3_field::{BasedVectorSpace, PrimeCharacteristicRing};
    use p3_multilinear_util::point::Point;
    use proptest::prelude::*;

    use super::*;

    /// Base field.
    type F = BabyBear;
    /// Challenge field.
    type EF = BinomialExtensionField<F, 4>;

    /// Boolean point of an index, most significant bit first.
    fn vertex(index: usize, n: usize) -> Vec<EF> {
        (0..n)
            .map(|i| EF::from_bool((index >> (n - 1 - i)) & 1 == 1))
            .collect()
    }

    /// Dense multilinear extension of an index-pair predicate at a point.
    fn dense_mle(x: &[EF], y: &[EF], f: impl Fn(usize, usize) -> bool) -> EF {
        let n = x.len();
        let mut acc = EF::ZERO;
        // Sum over every vertex pair.
        for i in 0..1 << n {
            for j in 0..1 << n {
                if f(i, j) {
                    acc += Point::eval_eq(x, &vertex(i, n)) * Point::eval_eq(y, &vertex(j, n));
                }
            }
        }
        acc
    }

    fn point(n: usize) -> impl Strategy<Value = Vec<EF>> {
        // Random extension coordinates.
        prop::collection::vec(any::<[u32; 4]>(), n).prop_map(|limbs| {
            limbs
                .into_iter()
                .map(|l| EF::from_basis_coefficients_fn(|k| F::from_u32(l[k])))
                .collect()
        })
    }

    #[test]
    fn shift_indicator_is_the_adder_on_every_vertex_pair() {
        // Invariant: on Boolean inputs the closed form is the indicator of y = x + k.
        //
        // Fixture state: 3 variables, every offset 0..8.
        let n = 3;
        for offset in 0..1 << n {
            for i in 0..1 << n {
                for j in 0..1 << n {
                    let got = shift_indicator(&vertex(i, n), &vertex(j, n), offset);
                    assert_eq!(
                        got,
                        EF::from_bool(j == i + offset),
                        "offset {offset}, ({i}, {j})"
                    );
                }
            }
        }
    }

    proptest! {
        #[test]
        fn shift_indicator_matches_the_dense_extension(
            x in point(3),
            y in point(3),
            offset in 0usize..9,
        ) {
            // Invariant: the adder recurrence is the unique multilinear extension.
            //
            // Fixture state: offset 8 hits the early exit.
            let expected = dense_mle(&x, &y, |i, j| j == i + offset);
            prop_assert_eq!(shift_indicator(&x, &y, offset), expected);
        }
    }
}
