//! The packing of the polynomial-basis `GF(2^128)` on AArch64: two elements side by side.
//!
//! `PMULL` reaches one 128-bit element per register, so every lane runs the scalar kernels.
//! The width is for the code around the arithmetic: a lane group reads two consecutive rows
//! of a column in one load, and one pass of an evaluation serves both rows.

use core::iter::{Product, Sum};
use core::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};

use p3_field::op_assign_macros::{
    impl_add_assign, impl_add_base_field, impl_div_methods, impl_mul_base_field, impl_mul_methods,
    impl_packed_field_div, impl_packed_value, impl_rng, impl_sub_assign, impl_sub_base_field,
    impl_sum_prod_base_field, ring_sum,
};
use p3_field::{
    Algebra, Field, PackedField, PackedFieldPow2, PackedValue, PrimeCharacteristicRing,
};
use rand::distr::{Distribution, StandardUniform};
use rand::{Rng, RngExt};

use crate::gf2::characteristic_two_methods;
use crate::tower::TowerLevel;
use crate::{BinaryField128, Gf2, Ghash128, clmul};

/// Elements per packed value.
const WIDTH: usize = 2;

/// Two polynomial-basis `GF(2^128)` elements, operated on lane by lane.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
// Needed for the packed layout contract.
#[repr(transparent)]
#[must_use]
pub struct PackedGhash128([Ghash128; WIDTH]);

impl PackedGhash128 {
    /// The same element in every lane.
    #[inline]
    const fn broadcast(value: Ghash128) -> Self {
        Self([value; WIDTH])
    }

    /// Each lane of `self` and `rhs` combined by `op`.
    #[inline(always)]
    fn zip_with(self, rhs: Self, op: impl Fn(Ghash128, Ghash128) -> Ghash128) -> Self {
        Self([op(self.0[0], rhs.0[0]), op(self.0[1], rhs.0[1])])
    }
}

impl From<Ghash128> for PackedGhash128 {
    #[inline]
    fn from(value: Ghash128) -> Self {
        Self::broadcast(value)
    }
}

impl Add for PackedGhash128 {
    type Output = Self;

    #[inline]
    fn add(self, rhs: Self) -> Self {
        self.zip_with(rhs, Add::add)
    }
}

impl Sub for PackedGhash128 {
    type Output = Self;

    #[inline]
    #[allow(clippy::suspicious_arithmetic_impl)]
    fn sub(self, rhs: Self) -> Self {
        // Subtraction coincides with addition in characteristic 2.
        self + rhs
    }
}

impl Neg for PackedGhash128 {
    type Output = Self;

    #[inline]
    fn neg(self) -> Self {
        // `-x = x` in characteristic 2.
        self
    }
}

impl Mul for PackedGhash128 {
    type Output = Self;

    #[inline]
    fn mul(self, rhs: Self) -> Self {
        // The two products share no operand, so the core overlaps them.
        self.zip_with(rhs, Mul::mul)
    }
}

impl PrimeCharacteristicRing for PackedGhash128 {
    type PrimeSubfield = Gf2;

    const ZERO: Self = Self::broadcast(Ghash128::ZERO);
    const ONE: Self = Self::broadcast(Ghash128::ONE);
    // The characteristic is 2, so `TWO = ONE + ONE = ZERO`.
    const TWO: Self = Self::broadcast(Ghash128::ZERO);
    // The characteristic is 2, so `NEG_ONE = ONE`.
    const NEG_ONE: Self = Self::broadcast(Ghash128::ONE);

    #[inline]
    fn from_prime_subfield(f: Self::PrimeSubfield) -> Self {
        Self::broadcast(Ghash128::from_prime_subfield(f))
    }

    characteristic_two_methods!();

    #[inline]
    fn square(&self) -> Self {
        Self(self.0.map(|lane| lane.square()))
    }

    /// `x·(x - 1) = x² + x` in characteristic 2, through the squaring shortcut in every lane.
    #[inline]
    fn bool_check(&self) -> Self {
        self.square() + *self
    }

    #[inline]
    fn dot_product<const N: usize>(u: &[Self; N], v: &[Self; N]) -> Self {
        // Each lane sums its own unreduced products and folds the modulus once.
        Self(core::array::from_fn(|lane| {
            Ghash128::from_repr(clmul::poly_dot_128(
                u.iter()
                    .zip(v)
                    .map(|(a, b)| (a.0[lane].to_repr(), b.0[lane].to_repr())),
            ))
        }))
    }
}

impl_add_assign!(PackedGhash128);
impl_sub_assign!(PackedGhash128);
impl_mul_methods!(PackedGhash128);
ring_sum!(PackedGhash128);
impl_rng!(PackedGhash128);

impl_add_base_field!(PackedGhash128, Ghash128);
impl_sub_base_field!(PackedGhash128, Ghash128);
impl_mul_base_field!(PackedGhash128, Ghash128);
impl_div_methods!(PackedGhash128, Ghash128);
impl_packed_field_div!(PackedGhash128);
impl_sum_prod_base_field!(PackedGhash128, Ghash128);

impl Algebra<Ghash128> for PackedGhash128 {}

impl From<Gf2> for PackedGhash128 {
    /// `GF(2)` is the prime subfield, embedded as `{ZERO, ONE}` in every lane.
    #[inline]
    fn from(x: Gf2) -> Self {
        Self::from_prime_subfield(x)
    }
}

impl_add_base_field!(PackedGhash128, Gf2);
impl_sub_base_field!(PackedGhash128, Gf2);
impl_mul_base_field!(PackedGhash128, Gf2);

impl Algebra<Gf2> for PackedGhash128 {}

impl From<BinaryField128> for PackedGhash128 {
    /// The same field element, seen in the polynomial basis, in every lane.
    ///
    /// The change of basis is a field isomorphism, so it makes this packing an algebra over
    /// the tower, exactly as it does for one lane's [`Ghash128`].
    #[inline]
    fn from(x: BinaryField128) -> Self {
        Self::broadcast(Ghash128::from(x))
    }
}

impl_add_base_field!(PackedGhash128, BinaryField128);
impl_sub_base_field!(PackedGhash128, BinaryField128);
impl_mul_base_field!(PackedGhash128, BinaryField128);

impl Algebra<BinaryField128> for PackedGhash128 {}

impl_packed_value!(PackedGhash128, Ghash128, WIDTH);

// SAFETY: the transparent array satisfies the packed layout contract.
// Arithmetic acts independently on each 128-bit field element.
unsafe impl PackedField for PackedGhash128 {
    type Scalar = Ghash128;
}

// SAFETY: the width is two, a power of two.
unsafe impl PackedFieldPow2 for PackedGhash128 {
    /// # Panics
    /// Panics if the block length does not divide the width.
    #[inline]
    fn interleave(&self, other: Self, block_len: usize) -> (Self, Self) {
        match block_len {
            1 => (Self([self.0[0], other.0[0]]), Self([self.0[1], other.0[1]])),
            2 => (*self, other),
            _ => panic!("unsupported block length"),
        }
    }
}

#[cfg(test)]
mod tests {
    use p3_field::{PackedFieldPow2, PackedValue, PrimeCharacteristicRing};
    use p3_field_testing::test_packed_binary_field;
    use proptest::prelude::*;

    use super::WIDTH;
    use crate::tower::TowerLevel;
    use crate::{Ghash128, PackedGhash128};

    /// The bit patterns a random search is unlikely to reach.
    const SPECIAL: [u128; 4] = [0, u128::MAX, 1 << 127, 0x87];

    /// The first extreme bit patterns, one per lane.
    fn specials() -> PackedGhash128 {
        PackedValue::from_fn(|i| Ghash128::from_repr(SPECIAL[i]))
    }

    /// Two distinct lanes from two patterns.
    fn packed(values: [u128; WIDTH]) -> PackedGhash128 {
        PackedValue::from_fn(|i| Ghash128::from_repr(values[i]))
    }

    /// Every lane of every operation against the scalar field on that lane alone.
    fn lanes_agree(
        a: [u128; WIDTH],
        b: [u128; WIDTH],
        c: [u128; WIDTH],
    ) -> Result<(), TestCaseError> {
        let (x, y, z) = (packed(a), packed(b), packed(c));
        for lane in 0..WIDTH {
            let (s, t, u) = (
                Ghash128::from_repr(a[lane]),
                Ghash128::from_repr(b[lane]),
                Ghash128::from_repr(c[lane]),
            );
            prop_assert_eq!((x + y).as_slice()[lane], s + t);
            prop_assert_eq!((x * y).as_slice()[lane], s * t);
            prop_assert_eq!(x.square().as_slice()[lane], s.square());
            prop_assert_eq!(x.bool_check().as_slice()[lane], s * (s - Ghash128::ONE));
            prop_assert_eq!(
                PackedGhash128::dot_product(&[x, y], &[y, z]).as_slice()[lane],
                s * t + t * u
            );
        }
        Ok(())
    }

    #[test]
    fn every_lane_matches_the_scalar_field_at_the_corners() {
        // Lane 0 walks every triple of corners, and lane 1 the next corner along each one, so
        // every pattern meets every other in both lanes and a lane crossing lands on another.
        let next = |i: usize| SPECIAL[(i + 1) % SPECIAL.len()];
        for (i, &x) in SPECIAL.iter().enumerate() {
            for (j, &y) in SPECIAL.iter().enumerate() {
                for (k, &z) in SPECIAL.iter().enumerate() {
                    lanes_agree([x, next(i)], [y, next(j)], [z, next(k)])
                        .unwrap_or_else(|e| panic!("{x:#x}, {y:#x}, {z:#x}: {e}"));
                }
            }
        }
    }

    proptest! {
        /// Every lane of every operation matches the scalar field on that lane alone.
        #[test]
        fn every_lane_matches_the_scalar_field(
            a in prop::array::uniform::<_, WIDTH>(any::<u128>()),
            b in prop::array::uniform::<_, WIDTH>(any::<u128>()),
            c in prop::array::uniform::<_, WIDTH>(any::<u128>()),
        ) {
            lanes_agree(a, b, c)?;
        }
    }

    #[test]
    fn interleave_swaps_whole_elements() {
        let (a, b) = (packed([1, 2]), packed([3, 4]));
        assert_eq!(a.interleave(b, 1), (packed([1, 3]), packed([2, 4])));
        assert_eq!(a.interleave(b, 2), (a, b));
    }

    test_packed_binary_field!(
        crate::PackedGhash128,
        &[crate::PackedGhash128::ZERO],
        &[crate::PackedGhash128::ONE],
        super::specials()
    );
}
