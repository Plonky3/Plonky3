//! The packing of the polynomial-basis `GF(2^128)` over the wide carryless multiply.
//!
//! `VPCLMULQDQ` applies the carryless multiply to every 128-bit lane of a wide register.
//!
//! One field element is exactly one lane, so the scalar kernel becomes the packed one.
//!
//! Every other operation is exclusive or, a lane-local shift, or a permutation of lanes.
//!
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

use super::lanes::{self, WIDTH};
use crate::gf2::characteristic_two_methods;
use crate::packed::split::{HIGH_BY_HIGH, LOW_BY_LOW, fold_shifted};
use crate::{BinaryField128, Gf2, Ghash128};

/// Several elements of the polynomial-basis `GF(2^128)`, one per 128-bit lane of a register.
///
/// Two under `AVX2`, four under `AVX-512`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
// Needed to make the transmutes below sound.
#[repr(transparent)]
#[must_use]
pub struct PackedGhash128([Ghash128; WIDTH]);

impl PackedGhash128 {
    /// The register holding these elements.
    #[inline]
    #[must_use]
    fn to_vector(self) -> lanes::Reg {
        // SAFETY: the scalar is `repr(transparent)` over `u128`.
        //
        // So the array is `WIDTH` contiguous `u128` values, the register's own layout.
        //
        // This type is `repr(transparent)` over that array.
        unsafe { transmute(self) }
    }

    /// The elements held in a register.
    #[inline]
    fn from_vector(vector: lanes::Reg) -> Self {
        // SAFETY: the inverse of the transmute above.
        //
        // Every bit pattern is a valid element, so no value can be out of range.
        unsafe { transmute(vector) }
    }

    /// The same element in every lane.
    #[inline]
    const fn broadcast(value: Ghash128) -> Self {
        Self([value; WIDTH])
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
    #[allow(clippy::suspicious_arithmetic_impl)]
    fn add(self, rhs: Self) -> Self {
        // Addition in characteristic 2 is `XOR`, lane by lane.
        Self::from_vector(lanes::xor(self.to_vector(), rhs.to_vector()))
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
        let (x, y) = (self.to_vector(), rhs.to_vector());

        // The two diagonal half products.
        let low = lanes::clmul::<LOW_BY_LOW>(x, y);
        let high = lanes::clmul::<HIGH_BY_HIGH>(x, y);

        // Karatsuba reaches the middle coefficient with one product instead of two.
        //
        //     middle = (a0 + a1)(b0 + b1) + a0 b0 + a1 b1
        //
        // The scalar kernel takes the schoolbook form instead.
        //
        // A wide carryless multiply has half the throughput of the 128-bit one on Zen 4 and 5.
        //
        // So trading a product for two shuffles and three exclusive ors pays only here.
        //
        // Measured on Zen 5, four lanes: 0.47 ns per element against 0.55 for schoolbook.
        let mixed_x = lanes::xor(x, lanes::swap_halves(x));
        let mixed_y = lanes::xor(y, lanes::swap_halves(y));
        let middle = lanes::xor(
            lanes::xor(low, high),
            lanes::clmul::<LOW_BY_LOW>(mixed_x, mixed_y),
        );

        // Inner fold brings the top limb down, outer fold finishes the reduction.
        Self::from_vector(fold_shifted(low, fold_shifted(middle, high)))
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
        let x = self.to_vector();

        // The cross term of `(p0 + p1 x^64)^2` doubles to zero.
        //
        // So the square has no middle coefficient, and the inner fold has nothing to add to.
        let low = lanes::clmul::<LOW_BY_LOW>(x, x);
        let high = lanes::clmul::<HIGH_BY_HIGH>(x, x);

        Self::from_vector(fold_shifted(low, fold_shifted(lanes::zero(), high)))
    }

    /// `x (x - 1) = x^2 - x = x^2 + x` in characteristic 2.
    ///
    /// A square skips the cross-term carryless multiplies a general product pays for.
    #[inline]
    fn bool_check(&self) -> Self {
        self.square() + *self
    }

    #[inline]
    fn dot_product<const N: usize>(u: &[Self; N], v: &[Self; N]) -> Self {
        let (mut low, mut high, mut middle) = (lanes::zero(), lanes::zero(), lanes::zero());

        for (a, b) in u.iter().zip(v) {
            let (x, y) = (a.to_vector(), b.to_vector());

            // Accumulate the three polynomial coefficients without reducing.
            low = lanes::xor(low, lanes::clmul::<LOW_BY_LOW>(x, y));
            high = lanes::xor(high, lanes::clmul::<HIGH_BY_HIGH>(x, y));

            // Karatsuba's third product, from the sum of each operand's halves.
            let mixed_x = lanes::xor(x, lanes::swap_halves(x));
            let mixed_y = lanes::xor(y, lanes::swap_halves(y));
            middle = lanes::xor(middle, lanes::clmul::<LOW_BY_LOW>(mixed_x, mixed_y));
        }

        // Remove the diagonal contributions from the accumulated third product.
        middle = lanes::xor(middle, lanes::xor(low, high));

        // Reduction is linear, so the whole sum folds the modulus once.
        Self::from_vector(fold_shifted(low, fold_shifted(middle, high)))
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
    /// The change of basis is a field isomorphism.
    ///
    /// So this packing is an algebra over the tower, exactly as one lane is.
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
//
// Arithmetic acts independently on each 128-bit field element.
unsafe impl PackedField for PackedGhash128 {
    type Scalar = Ghash128;
}

// SAFETY: the width is two or four, both powers of two.
unsafe impl PackedFieldPow2 for PackedGhash128 {
    /// # Panics
    /// Panics if the block length does not divide the width.
    #[inline]
    fn interleave(&self, other: Self, block_len: usize) -> (Self, Self) {
        let (a, b) = lanes::interleave(self.to_vector(), other.to_vector(), block_len);
        (Self::from_vector(a), Self::from_vector(b))
    }
}

#[cfg(test)]
mod tests {
    use p3_field::{PackedValue, PrimeCharacteristicRing};
    use p3_field_testing::test_packed_binary_field;
    use proptest::prelude::*;

    use super::lanes::{self, WIDTH};
    use crate::packed::split::model::Model;
    use crate::packed::split::{
        HIGH_BY_HIGH, HIGH_BY_LOW, LOW_BY_HIGH, LOW_BY_LOW, Lanes, SplitScalar, fold_shifted,
    };
    use crate::{Ghash128, PackedGhash128};

    /// The bit patterns a random search is unlikely to reach.
    ///
    /// Each one drives the fold of the modulus to an extreme:
    ///
    /// ```text
    ///     0            nothing to reduce
    ///     all ones     every coefficient of the product is live
    ///     x^127        the highest degree, so the fold spills furthest
    ///     0x87         the modulus tail itself
    /// ```
    const SPECIAL: [u128; 4] = [0, u128::MAX, 1 << 127, 0x87];

    /// One extreme bit pattern per lane.
    fn specials() -> PackedGhash128 {
        PackedValue::from_fn(|i| Ghash128::from_le_bytes(SPECIAL[i].to_le_bytes()))
    }

    /// The elements a register holds, read back one per lane.
    fn lanes_of(register: lanes::Reg) -> [u128; WIDTH] {
        let mut out = [0u128; WIDTH];

        // SAFETY: the destination is exactly one register of contiguous 128-bit integers.
        //
        // The store is the unaligned form, so the array's own alignment is irrelevant.
        unsafe { lanes::store(out.as_mut_ptr(), register) };

        out
    }

    /// A register holding the given elements, one per lane.
    fn register_of(values: [u128; WIDTH]) -> lanes::Reg {
        // SAFETY: the source is exactly one register of contiguous 128-bit integers.
        unsafe { lanes::load(values.as_ptr()) }
    }

    /// The two 64-bit halves of every lane, exchanged.
    fn swapped_halves(values: [u128; WIDTH]) -> [u128; WIDTH] {
        // Rotating a 128-bit value by half its width is exactly the exchange.
        core::array::from_fn(|i| values[i].rotate_left(64))
    }

    /// Each lane-local operation on the register, against the same operation on the model.
    ///
    /// One assertion per operation, so a mismatch names the intrinsic that disagreed.
    ///
    /// The whole-element interleave is the one left out, since it crosses lanes by design.
    fn lanes_conform(
        a: [u128; WIDTH],
        b: [u128; WIDTH],
        scalar: u128,
    ) -> Result<(), TestCaseError> {
        let (x, y) = (register_of(a), register_of(b));
        let (mx, my) = (Model(a), Model(b));

        prop_assert_eq!(lanes_of(lanes::Reg::zero()), Model::<WIDTH>::zero().0);
        prop_assert_eq!(
            lanes_of(lanes::Reg::broadcast(scalar)),
            Model::<WIDTH>::broadcast(scalar).0
        );
        prop_assert_eq!(lanes_of(lanes::Reg::tail()), Model::<WIDTH>::tail().0);
        prop_assert_eq!(lanes_of(x.xor(y)), mx.xor(my).0);
        prop_assert_eq!(lanes_of(x.unpack_low_64(y)), mx.unpack_low_64(my).0);
        prop_assert_eq!(
            lanes_of(x.clmul::<LOW_BY_LOW>(y)),
            mx.clmul::<LOW_BY_LOW>(my).0
        );
        prop_assert_eq!(
            lanes_of(x.clmul::<HIGH_BY_LOW>(y)),
            mx.clmul::<HIGH_BY_LOW>(my).0
        );
        prop_assert_eq!(
            lanes_of(x.clmul::<LOW_BY_HIGH>(y)),
            mx.clmul::<LOW_BY_HIGH>(my).0
        );
        prop_assert_eq!(
            lanes_of(x.clmul::<HIGH_BY_HIGH>(y)),
            mx.clmul::<HIGH_BY_HIGH>(my).0
        );

        // Not a trait method, so the model cannot supply the expectation.
        //
        // It carries the only hand-written shuffle immediate here, which is why it is pinned.
        prop_assert_eq!(lanes_of(lanes::swap_halves(x)), swapped_halves(a));

        // The two composites built from those operations.
        //
        // A lane-crossing slip therefore shows up here as well as in the primitives above.
        prop_assert_eq!(lanes_of(fold_shifted(x, y)), fold_shifted(mx, my).0);
        prop_assert_eq!(
            lanes_of(SplitScalar::new(scalar).apply(x)),
            SplitScalar::new(scalar).apply(mx).0
        );

        Ok(())
    }

    #[test]
    fn the_register_matches_the_scalar_model_at_the_corners() {
        // Invariant: each operation is lane-local, so distinct lanes must stay distinct.
        //
        // Lane 0 carries the pair under test, so all sixteen combinations are reached.
        //
        // That includes the squaring-shaped case where both operands are the same value.
        //
        // The other lanes rotate through the corners.
        //
        // A value that crosses a lane boundary then lands on a different extreme and mismatches.
        for (i, &x) in SPECIAL.iter().enumerate() {
            for (j, &y) in SPECIAL.iter().enumerate() {
                let a = core::array::from_fn(|lane| match lane {
                    0 => x,
                    _ => SPECIAL[(i + lane) % SPECIAL.len()],
                });
                let b = core::array::from_fn(|lane| match lane {
                    0 => y,
                    _ => SPECIAL[(j + lane) % SPECIAL.len()],
                });

                // The multiplier walks the corners too, so the split runs from each extreme.
                for scalar in SPECIAL {
                    lanes_conform(a, b, scalar)
                        .unwrap_or_else(|e| panic!("a {x:#x}, b {y:#x}, scalar {scalar:#x}: {e}"));
                }
            }
        }
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(256))]

        #[test]
        fn the_register_matches_the_scalar_model(
            a in prop::array::uniform(any::<u128>()),
            b in prop::array::uniform(any::<u128>()),
            scalar in any::<u128>(),
        ) {
            // The seam the scalar model alone cannot reach.
            //
            // Each intrinsic must be the one the algebra was written against, lane by lane.
            lanes_conform(a, b, scalar)?;
        }
    }

    #[test]
    fn bool_check_matches_the_general_product_in_every_lane() {
        // Invariant: the squaring shortcut answers what the general product answers.
        //
        // Every booleanity constraint an AIR states takes this shortcut.
        //
        // The corners cover the two roots the check accepts and the extreme reductions.
        for (index, &value) in SPECIAL.iter().enumerate() {
            // Lane 0 carries the value under test.
            //
            // The rest rotate through the corners, so a lane-crossing result lands elsewhere.
            let lanes: [Ghash128; WIDTH] = core::array::from_fn(|lane| {
                let pattern = if lane == 0 {
                    value
                } else {
                    SPECIAL[(index + lane) % SPECIAL.len()]
                };
                Ghash128::from_le_bytes(pattern.to_le_bytes())
            });
            let packed: PackedGhash128 = PackedValue::from_fn(|lane| lanes[lane]);

            for (lane, &scalar) in lanes.iter().enumerate() {
                assert_eq!(
                    packed.bool_check().as_slice()[lane],
                    scalar * (scalar - Ghash128::ONE),
                    "lane {lane} of corner {value:#x}"
                );
                assert_eq!(packed.bool_check().as_slice()[lane], scalar.bool_check());
            }
        }

        // Zero and one are the roots the check accepts, so both must vanish in every lane.
        for root in [Ghash128::ZERO, Ghash128::ONE] {
            let packed: PackedGhash128 = PackedValue::from_fn(|_| root);
            assert!(
                packed
                    .bool_check()
                    .as_slice()
                    .iter()
                    .all(|&value| value == Ghash128::ZERO)
            );
        }
    }

    test_packed_binary_field!(
        crate::PackedGhash128,
        &[crate::PackedGhash128::ZERO],
        &[crate::PackedGhash128::ONE],
        super::specials()
    );
}
