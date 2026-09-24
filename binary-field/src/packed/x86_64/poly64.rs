//! The packing of `GF(2^64)`, one element per quadword of a wide register.
//!
//! Four elements per 256-bit register, with or without `AVX-512`.
//!
//! A product is two carryless multiplies for the whole register, one per quadword parity.
//!
//! The reduction is shifts and one byte shuffle, so it never competes for the multiplier.

use core::iter::{Product, Sum};
use core::mem::transmute;
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

use super::lanes::gf64::{self as lanes, Reg, WIDTH_64};
use crate::clmul::wide::{Lanes64, Wide};
use crate::gf2::characteristic_two_methods;
use crate::{Gf2, Poly64};

/// Several elements of `GF(2^64)`, one per quadword of a register.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
// Needed to make the transmutes below sound.
#[repr(transparent)]
#[must_use]
pub struct PackedPoly64(pub(super) [Poly64; WIDTH_64]);

impl PackedPoly64 {
    /// The register holding these elements.
    #[inline(always)]
    pub(super) fn to_vector(self) -> Reg {
        // SAFETY: an element is `repr(transparent)` over `u64`.
        //
        // The array is then one register's worth of contiguous quadwords, its own layout.
        //
        // This type is `repr(transparent)` over that array.
        unsafe { transmute(self) }
    }

    /// The elements held in a register.
    #[inline(always)]
    pub(super) fn from_vector(vector: Reg) -> Self {
        // SAFETY: the inverse of the transmute above.
        //
        // Every bit pattern is a valid element, so no value can be out of range.
        unsafe { transmute(vector) }
    }

    /// The same element in every lane.
    #[inline]
    const fn broadcast(value: Poly64) -> Self {
        // Constant, so the field's constants can be built from it.
        Self([value; WIDTH_64])
    }
}

impl From<Poly64> for PackedPoly64 {
    #[inline]
    fn from(value: Poly64) -> Self {
        // The same element in every lane.
        Self::broadcast(value)
    }
}

impl Add for PackedPoly64 {
    type Output = Self;

    #[inline]
    #[allow(clippy::suspicious_arithmetic_impl)]
    fn add(self, rhs: Self) -> Self {
        // Addition in characteristic 2 is `XOR`, lane by lane.
        Self::from_vector(self.to_vector().xor(rhs.to_vector()))
    }
}

impl Sub for PackedPoly64 {
    type Output = Self;

    #[inline]
    #[allow(clippy::suspicious_arithmetic_impl)]
    fn sub(self, rhs: Self) -> Self {
        // Subtraction coincides with addition in characteristic 2.
        self + rhs
    }
}

impl Neg for PackedPoly64 {
    type Output = Self;

    #[inline]
    fn neg(self) -> Self {
        // `-x = x` in characteristic 2.
        self
    }
}

impl Mul for PackedPoly64 {
    type Output = Self;

    /// Two carryless multiplies for the register, then a shift fold.
    #[inline]
    fn mul(self, rhs: Self) -> Self {
        // Even and odd quadwords each give one 128-bit product per 128-bit lane.
        //
        //     even  = [ a_0 b_0 | a_2 b_2 | ... ]
        //     odd   = [ a_1 b_1 | a_3 b_3 | ... ]
        Self::from_vector(Wide::mul(self.to_vector(), rhs.to_vector()).reduce())
    }
}

impl PrimeCharacteristicRing for PackedPoly64 {
    type PrimeSubfield = Gf2;

    const ZERO: Self = Self::broadcast(Poly64::ZERO);
    const ONE: Self = Self::broadcast(Poly64::ONE);
    // The characteristic is 2, so `TWO = ONE + ONE = ZERO`.
    const TWO: Self = Self::broadcast(Poly64::ZERO);
    // The characteristic is 2, so `NEG_ONE = ONE`.
    const NEG_ONE: Self = Self::broadcast(Poly64::ONE);

    #[inline]
    fn from_prime_subfield(f: Self::PrimeSubfield) -> Self {
        // Zero or one in every lane.
        Self::broadcast(Poly64::from_prime_subfield(f))
    }

    characteristic_two_methods!();

    #[inline]
    fn square(&self) -> Self {
        // The same two products as a general multiply, with both operands equal.
        let x = self.to_vector();
        Self::from_vector(Wide::mul(x, x).reduce())
    }

    /// `x (x - 1) = x^2 + x` in characteristic 2, and squaring is the cheaper product.
    #[inline]
    fn bool_check(&self) -> Self {
        // Zero exactly on the two roots of x^2 + x, which are zero and one.
        self.square() + *self
    }

    /// Reduction is linear, so the whole sum reduces once.
    #[inline]
    fn dot_product<const N: usize>(u: &[Self; N], v: &[Self; N]) -> Self {
        // Two carryless multiplies and two exclusive ors per term, nothing else.
        let sum = u.iter().zip(v).fold(Wide::zero(), |sum, (a, b)| {
            sum.xor(Wide::mul(a.to_vector(), b.to_vector()))
        });

        // One fold for the whole sum.
        Self::from_vector(sum.reduce())
    }
}

impl_add_assign!(PackedPoly64);
impl_sub_assign!(PackedPoly64);
impl_mul_methods!(PackedPoly64);
ring_sum!(PackedPoly64);
impl_rng!(PackedPoly64);

impl_add_base_field!(PackedPoly64, Poly64);
impl_sub_base_field!(PackedPoly64, Poly64);
impl_mul_base_field!(PackedPoly64, Poly64);
impl_div_methods!(PackedPoly64, Poly64);
impl_packed_field_div!(PackedPoly64);
impl_sum_prod_base_field!(PackedPoly64, Poly64);

impl Algebra<Poly64> for PackedPoly64 {}

impl From<Gf2> for PackedPoly64 {
    /// `GF(2)` is the prime subfield, embedded as `{ZERO, ONE}` in every lane.
    #[inline]
    fn from(x: Gf2) -> Self {
        Self::from_prime_subfield(x)
    }
}

impl_add_base_field!(PackedPoly64, Gf2);
impl_sub_base_field!(PackedPoly64, Gf2);
impl_mul_base_field!(PackedPoly64, Gf2);

impl Algebra<Gf2> for PackedPoly64 {}

impl_packed_value!(PackedPoly64, Poly64, WIDTH_64);

// SAFETY: the transparent array satisfies the packed layout contract.
// Arithmetic acts independently on each quadword.
unsafe impl PackedField for PackedPoly64 {
    type Scalar = Poly64;
}

// SAFETY: the width is four, a power of two.
unsafe impl PackedFieldPow2 for PackedPoly64 {
    /// # Panics
    /// Panics if the block length does not divide the width.
    #[inline]
    fn interleave(&self, other: Self, block_len: usize) -> (Self, Self) {
        // Blocks of one, two or four quadwords swap between the two registers.
        let (a, b) = lanes::interleave_64(self.to_vector(), other.to_vector(), block_len);
        (Self::from_vector(a), Self::from_vector(b))
    }
}

#[cfg(test)]
mod tests {
    use p3_field::{PackedValue, PrimeCharacteristicRing};
    use p3_field_testing::test_packed_binary_field;
    use proptest::prelude::*;

    use super::super::lanes::gf64::WIDTH_64;
    use crate::{PackedPoly64, Poly64};

    /// The bit patterns a random search is unlikely to reach.
    ///
    /// ```text
    ///     0            nothing to reduce
    ///     all ones     every coefficient of the product is live
    ///     x^63         the highest degree, so the fold spills furthest
    ///     0x1b         the modulus tail itself
    ///     0xf << 60    the whole top nibble the byte table folds
    /// ```
    const SPECIAL: [u64; 5] = [0, u64::MAX, 1 << 63, 0x1b, 0xf << 60];

    /// One extreme bit pattern per lane, cycling through the list.
    fn specials() -> PackedPoly64 {
        PackedValue::from_fn(|i| Poly64::new(SPECIAL[i % SPECIAL.len()]))
    }

    /// A packing of the given bit patterns.
    fn packed(bits: [u64; WIDTH_64]) -> PackedPoly64 {
        PackedValue::from_fn(|i| Poly64::new(bits[i]))
    }

    /// Every lane of a packed result against the scalar result for that lane.
    fn lanes_agree(
        packed: PackedPoly64,
        scalar: impl Fn(usize) -> Poly64,
    ) -> Result<(), TestCaseError> {
        // One assertion per lane, so a mismatch names the lane that disagreed.
        for (lane, &value) in packed.as_slice().iter().enumerate() {
            prop_assert_eq!(value, scalar(lane), "lane {}", lane);
        }
        Ok(())
    }

    #[test]
    fn every_lane_matches_the_scalar_field_at_the_corners() {
        // Invariant: every lane multiplies independently of its neighbours.
        //
        // Fixture state: lane 0 carries the pair under test, 5 x 5 = 25 pairs.
        //
        // The other lanes rotate through the corners.
        //
        // A result that crossed a lane then lands on a different extreme.
        for (i, &x) in SPECIAL.iter().enumerate() {
            for (j, &y) in SPECIAL.iter().enumerate() {
                let a: [u64; WIDTH_64] =
                    core::array::from_fn(|l| if l == 0 { x } else { SPECIAL[(i + l) % 5] });
                let b: [u64; WIDTH_64] =
                    core::array::from_fn(|l| if l == 0 { y } else { SPECIAL[(j + l) % 5] });
                let (pa, pb) = (packed(a), packed(b));
                let lane = |bits: [u64; WIDTH_64], l: usize| Poly64::new(bits[l]);

                lanes_agree(pa * pb, |l| lane(a, l) * lane(b, l)).unwrap();
                lanes_agree(pa.square(), |l| lane(a, l).square()).unwrap();
            }
        }
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(2000))]

        #[test]
        fn every_lane_matches_the_scalar_field(
            a in any::<[u64; WIDTH_64]>(),
            b in any::<[u64; WIDTH_64]>(),
            c in any::<[u64; WIDTH_64]>(),
        ) {
            // Invariant: every packed operation is the scalar one, lane by lane.
            let (pa, pb, pc) = (packed(a), packed(b), packed(c));
            let lane = |bits: [u64; WIDTH_64], l: usize| Poly64::new(bits[l]);

            lanes_agree(pa * pb, |l| lane(a, l) * lane(b, l))?;
            lanes_agree(pa.square(), |l| lane(a, l).square())?;
            lanes_agree(pa.bool_check(), |l| lane(a, l).bool_check())?;

            // The deferred reduction of a dot product against one reduction per term.
            //
            //     reduce(ab + bc)  =  reduce(ab) + reduce(bc)
            lanes_agree(PackedPoly64::dot_product(&[pa, pb], &[pb, pc]), |l| {
                lane(a, l) * lane(b, l) + lane(b, l) * lane(c, l)
            })?;
        }
    }

    test_packed_binary_field!(
        crate::PackedPoly64,
        &[crate::PackedPoly64::ZERO],
        &[crate::PackedPoly64::ONE],
        super::specials()
    );
}
