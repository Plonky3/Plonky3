//! `GF(4)` as a small subfield of the tower's `GF(2^128)`.
//!
//! Each level above `GF(4)` is two coefficients over the level below, in its low and high halves.
//! Unrolled down to `GF(4)`, the 2-bit chunks of a `GF(2^128)` element are its coordinates over
//! `GF(4)`. The chunk at bits `2i` and `2i + 1` is `b0 + b1·X_0`, with `b0` in the lower bit.
//!
//! `GF(4)` itself is the chunk at bit 0 with every other chunk zero.
//! A `GF(4)` scalar lies in every level, so it multiplies both halves at each split separately.
//! Hence it multiplies each chunk on its own, with nothing carried between chunks.

use core::ops::{Add, Mul, Sub};

use p3_field::op_assign_macros::{impl_add_base_field, impl_sub_base_field};
use p3_field::{Algebra, HasSubfield};

use crate::tower::TowerLevel;
use crate::{BinaryField2, BinaryField128};

/// The coefficient of `1` in every 2-bit chunk.
const LOW_BITS: u128 = 0x5555_5555_5555_5555_5555_5555_5555_5555;

/// The coefficient of `X_0` in every 2-bit chunk.
const HIGH_BITS: u128 = !LOW_BITS;

/// Multiply every 2-bit chunk by the generator `X_0` of `GF(4)`.
///
/// `X_0² = X_0 + 1`, so `X_0·(b0 + b1·X_0) = b1 + (b0 + b1)·X_0`.
#[inline]
const fn mul_x0(bits: u128) -> u128 {
    let b1 = bits & HIGH_BITS;
    (b1 >> 1) ^ ((bits & LOW_BITS) << 1) ^ b1
}

impl From<BinaryField2> for BinaryField128 {
    /// `GF(4)` occupies the low two bits; every other coefficient vanishes.
    #[inline]
    fn from(x: BinaryField2) -> Self {
        Self::from_repr(u128::from(x.to_repr()))
    }
}

impl_add_base_field!(BinaryField128, BinaryField2);
impl_sub_base_field!(BinaryField128, BinaryField2);

impl Mul<BinaryField2> for BinaryField128 {
    type Output = Self;

    /// `x·(c0 + c1·X_0) = c0·x + c1·(X_0·x)`.
    ///
    /// Each coefficient of the scalar selects its term through an all-ones or all-zeros mask,
    /// so there is no product and no branch on the scalar.
    #[inline]
    fn mul(self, rhs: BinaryField2) -> Self {
        let bits = self.to_repr();
        let c = rhs.to_repr();
        let c0 = u128::from(c & 1).wrapping_neg();
        let c1 = u128::from((c >> 1) & 1).wrapping_neg();
        Self::from_repr((c0 & bits) ^ (c1 & mul_x0(bits)))
    }
}

impl Mul<BinaryField128> for BinaryField2 {
    type Output = BinaryField128;

    #[inline]
    fn mul(self, rhs: BinaryField128) -> BinaryField128 {
        rhs * self
    }
}

impl Algebra<BinaryField2> for BinaryField128 {}

impl HasSubfield<BinaryField2> for BinaryField128 {
    /// An element lies in `GF(4)` exactly when no bit above the lowest two is set.
    #[inline]
    fn as_subfield(&self) -> Option<BinaryField2> {
        let bits = self.to_repr();
        (bits & !3 == 0).then(|| BinaryField2::from_repr(bits as u8))
    }

    /// The bits above the lowest two, OR-ed across the slice, vanish exactly when every element
    /// lies in `GF(4)`.
    ///
    /// The fold has no early exit, so the loop has no data-dependent branch.
    #[inline]
    fn all_in_subfield(values: &[Self]) -> bool {
        values.iter().fold(0, |acc, v| acc | v.to_repr()) & !3 == 0
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec::Vec;

    use p3_field::{HasSubfield, PrimeCharacteristicRing};
    use p3_field_testing::test_has_subfield;
    use proptest::prelude::*;

    use crate::tower::TowerLevel;
    use crate::{BinaryField2, BinaryField8, BinaryField128};

    /// Every element of `GF(4)`, in bit-pattern order.
    fn gf4() -> impl Iterator<Item = BinaryField2> {
        (0..4).map(BinaryField2::from_repr)
    }

    #[test]
    fn the_embedding_is_a_ring_homomorphism_on_every_pair() {
        assert_eq!(
            BinaryField128::from(BinaryField2::ZERO),
            BinaryField128::ZERO
        );
        assert_eq!(BinaryField128::from(BinaryField2::ONE), BinaryField128::ONE);

        for a in gf4() {
            for b in gf4() {
                let (x, y) = (BinaryField128::from(a), BinaryField128::from(b));
                assert_eq!(BinaryField128::from(a + b), x + y, "{a} + {b}");
                assert_eq!(BinaryField128::from(a * b), x * y, "{a} * {b}");

                // `reference_mul` recurses to the bottom of the tower without dispatching, so
                // this ties the embedding to the recursive definition, not to a fast path.
                assert_eq!(BinaryField128::from(a * b), x.reference_mul(y), "{a} * {b}");
            }
        }
    }

    /// `GF(4)` has the same bit patterns inside `GF(2^8)`, so embedding it directly agrees with
    /// embedding it through the byte level.
    #[test]
    fn the_embedding_factors_through_the_byte_level() {
        let byte = |a: BinaryField2| BinaryField8::from_repr(a.to_repr());

        for a in gf4() {
            assert_eq!(BinaryField128::from(a), BinaryField128::from(byte(a)));

            for b in gf4() {
                assert_eq!(byte(a * b), byte(a) * byte(b), "{a} * {b}");
            }
        }
    }

    #[test]
    fn every_subfield_element_narrows_back_to_itself() {
        for a in gf4() {
            assert_eq!(BinaryField128::from(a).as_subfield(), Some(a));
        }
    }

    #[test]
    fn the_has_subfield_contract_holds() {
        test_has_subfield::<BinaryField128, BinaryField2>();
    }

    /// A single bit above the lowest two keeps an element out, whichever bit it is.
    ///
    /// Among subfield elements, that element alone must spoil the slice, wherever it sits.
    #[test]
    fn every_bit_above_the_lowest_two_lies_outside_the_subfield() {
        let members: Vec<BinaryField128> = gf4().map(BinaryField128::from).collect();
        for s in 0..4u128 {
            for k in 2..128 {
                let x = BinaryField128::from_repr(s | (1u128 << k));
                assert_eq!(x.as_subfield(), None, "{s} with bit {k} set");

                let mut values = members.clone();
                values.insert(k % (members.len() + 1), x);
                assert!(
                    !BinaryField128::all_in_subfield(&values),
                    "{s} with bit {k} set among the subfield elements"
                );
            }
        }
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(1000))]

        #[test]
        fn the_masked_product_agrees_with_the_tower_product(bits: u128) {
            let x = BinaryField128::from_repr(bits);
            for s in gf4() {
                let image = BinaryField128::from(s);
                prop_assert_eq!(x * s, x * image, "scalar {}", s);
                prop_assert_eq!(s * x, image * x, "scalar {}", s);
            }
        }

        /// The shift reaches every magnitude, including the patterns below four.
        #[test]
        fn narrowing_accepts_exactly_the_low_two_bits(bits: u128, shift in 0u32..128) {
            let bits = bits >> shift;
            let x = BinaryField128::from_repr(bits);
            match x.as_subfield() {
                Some(s) => {
                    prop_assert!(bits < 4);
                    prop_assert_eq!(BinaryField128::from(s), x);
                }
                None => prop_assert!(bits >= 4),
            }
        }

        /// Mostly subfield elements, so that whole slices lie in the subfield often.
        /// The others are shifted to every magnitude, so some sit just above the subfield.
        #[test]
        fn all_in_subfield_agrees_with_narrowing_each_element(
            bits in prop::collection::vec(
                prop_oneof![
                    3 => 0u128..4,
                    1 => (any::<u128>(), 0u32..128).prop_map(|(bits, shift)| bits >> shift),
                ],
                0..12,
            ),
        ) {
            let values: Vec<BinaryField128> =
                bits.into_iter().map(BinaryField128::from_repr).collect();
            prop_assert_eq!(
                BinaryField128::all_in_subfield(&values),
                values.iter().all(|v| v.as_subfield().is_some())
            );
        }
    }
}
